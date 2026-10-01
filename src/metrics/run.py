"""Build outputs/metrics.parquet, outputs/crop_anomalies.parquet and two supporting files
(outputs/metrics_national.parquet, outputs/metrics_diagnostics.csv) for included countries.

Run: python -m src.metrics.run        (USE_DUMMY=1 for dummy inputs)
"""
import numpy as np
import pandas as pd

from ..common import paths
from ..common.contract import YEAR_MAX, YEAR_MIN, load, summary, validate
from . import decompose as dec
from . import faces, response
from .detrend import FRAC, detrend, lowess_trend
from .diversity import hill1
from .events import DROUGHT_THR

YEARS = np.arange(YEAR_MIN, YEAR_MAX + 1)


def country_metrics(g, spei, event_years, frac=FRAC, thr=DROUGHT_THR):
    """All metrics for one country.

    g: that country's rows of crops.parquet. spei: Series year -> spei12_w.
    event_years: drought event years. Returns (metrics dict, national df, anomalies df, diagnostics dict).
    """
    spei = spei.reindex(YEARS)
    g = g[g["value_const"].notna()]

    # National value index.
    nat = detrend(g.groupby("year")["value_const"].sum().reindex(YEARS).to_numpy(), YEARS, frac)
    nat.insert(0, "year", YEARS)
    index = nat.set_index("year")["index"]
    drought_years = spei.index[spei <= thr]
    nat["spei12_w"] = spei.to_numpy()
    nat["is_drought"] = nat["year"].isin(drought_years)
    nat["is_event"] = nat["year"].isin(event_years)

    m = {
        "hill1": hill1(g.groupby("item_code")["area_ha"].sum() / len(YEARS)),
        "stability": faces.stability(nat["y"], nat["resid"]),
        "resistance": faces.resistance(index, event_years),
        "recovery": faces.recovery(index, event_years),
        "vulnerability": faces.vulnerability(index),
        "vuln_drought": faces.vulnerability(index, drought_years),
        "n_events": faces.n_usable_events(index, event_years),
    }

    # Per-crop yield anomalies and drought response, fitted on non-imputed years only.
    share = g.groupby("item_code")["value_const"].sum() / g["value_const"].sum()
    anoms = []
    for code, c in g.groupby("item_code"):
        c = c[~c["is_imputed"] & np.isfinite(c["yield_t_ha"]) & (c["yield_t_ha"] > 0)]
        if share[code] < response.MIN_SHARE or len(c) < response.MIN_YEARS:
            continue
        d = detrend(c["yield_t_ha"].to_numpy(), c["year"].to_numpy(), frac)
        beta = response.resp_beta(d["anom"], spei.reindex(c["year"]).to_numpy())
        if not np.isfinite(beta):
            continue
        anoms.append(pd.DataFrame({
            "year": c["year"].to_numpy(), "item": c["item"].iloc[0], "yield_anom": d["anom"].to_numpy(),
            "value_share": float(share[code]), "resp_beta": beta,
        }).dropna(subset=["yield_anom"]))
    anoms = pd.concat(anoms, ignore_index=True) if anoms else pd.DataFrame(
        columns=["year", "item", "yield_anom", "value_share", "resp_beta"])
    per_crop = anoms.drop_duplicates("item")
    m["resp_div"] = response.resp_div(per_crop["resp_beta"], per_crop["value_share"])
    m["resp_divergence"] = response.resp_divergence(per_crop["resp_beta"])

    # Portfolio decomposition on crops with a value in every year of the window.
    wide = g.pivot_table(index="year", columns="item_code", values="value_const", aggfunc="sum").reindex(YEARS)
    wide = wide.loc[:, wide.notna().all() & (wide.mean() > 0)]
    if wide.shape[1] >= 2:
        resid = np.column_stack([wide[c].to_numpy() - lowess_trend(wide[c].to_numpy(), YEARS, frac) for c in wide])
        d = dec.decompose(wide.to_numpy(), resid)
    else:
        d = {"phi_sync": np.nan, "mean_crop_cv": np.nan, "cv_total": np.nan}
    m["phi_sync"], m["mean_crop_cv"] = d["phi_sync"], d["mean_crop_cv"]

    diag = {
        "n_years_value": int(nat["y"].notna().sum()),
        "n_drought_years": int(len(drought_years)),
        "n_events_catalogue": int(len(event_years)),
        "n_crops_response": int(len(per_crop)),
        "response_value_share": float(per_crop["value_share"].sum()),
        "n_crops_decomp": int(wide.shape[1]),
        "decomp_value_share": float(wide.sum().sum() / g["value_const"].sum()),
        "cv_total_decomp": d["cv_total"],
    }
    return m, nat, anoms, diag


def compute(crops, climate, events, included, frac=FRAC, thr=DROUGHT_THR):
    """Metrics for every country in `included`. Returns dict of DataFrames:
    metrics, crop_anomalies, national, diagnostics."""
    metrics, national, anomalies, diagnostics = [], [], [], []
    spei_by = {k: v.set_index("year")["spei12_w"] for k, v in climate.groupby("iso3")}
    events_by = events.groupby("iso3")["year"].apply(list).to_dict()
    for iso3, g in crops[crops["iso3"].isin(included)].groupby("iso3", sort=True):
        spei = spei_by.get(iso3, pd.Series(dtype=float))
        m, nat, an, diag = country_metrics(g, spei, events_by.get(iso3, []), frac, thr)
        metrics.append({"iso3": iso3, **m})
        diagnostics.append({"iso3": iso3, **diag})
        national.append(nat.assign(iso3=iso3))
        anomalies.append(an.assign(iso3=iso3))
    metrics = pd.DataFrame(metrics)
    metrics["n_events"] = metrics["n_events"].astype(int)
    national = pd.concat(national, ignore_index=True).rename(columns={"y": "value_const"})
    national = national[["iso3", "year", "value_const", "trend", "resid", "anom", "index",
                         "spei12_w", "is_drought", "is_event"]]
    anomalies = pd.concat(anomalies, ignore_index=True)[
        ["iso3", "year", "item", "yield_anom", "value_share", "resp_beta"]]
    anomalies = anomalies.astype({"iso3": str, "year": int, "item": str, "yield_anom": float,
                                  "value_share": float, "resp_beta": float})
    return {"metrics": metrics, "crop_anomalies": anomalies, "national": national,
            "diagnostics": pd.DataFrame(diagnostics)}


def load_inputs():
    """Contract inputs. If climate.parquet / events.csv do not exist yet, the climate-dependent
    metrics come out as NaN and a warning is printed."""
    crops, quality = load("crops"), load("quality")
    if paths.CLIMATE.exists() and paths.EVENTS.exists():
        climate, events = load("climate"), load("events")
    else:
        print("WARNING: climate.parquet / events.csv not found. resistance, recovery, vuln_drought, "
              "resp_div and resp_divergence will be missing.")
        climate = pd.DataFrame({"iso3": pd.Series(dtype=str), "year": pd.Series(dtype=int), "spei12_w": pd.Series(dtype=float)})
        events = pd.DataFrame({"iso3": pd.Series(dtype=str), "year": pd.Series(dtype=int)})
    included = quality.loc[quality["included"], "iso3"].tolist()
    return crops, climate, events, included


def main():
    crops, climate, events, included = load_inputs()
    out = compute(crops, climate, events, included)
    metrics, anomalies, national, diag = out["metrics"], out["crop_anomalies"], out["national"], out["diagnostics"]

    validate(metrics, "metrics")
    validate(anomalies, "crop_anomalies")
    assert set(metrics["iso3"]) <= set(included)
    assert not national.duplicated(["iso3", "year"]).any()
    assert national["year"].between(YEAR_MIN, YEAR_MAX).all()
    assert (diag["decomp_value_share"].dropna() <= 1 + 1e-9).all()

    paths.OUT.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(paths.METRICS, index=False)
    anomalies.to_parquet(paths.CROP_ANOMALIES, index=False)
    national.to_parquet(paths.NATIONAL, index=False)
    diag.to_csv(paths.OUT / "metrics_diagnostics.csv", index=False)

    print(summary(metrics, "metrics"))
    print(summary(anomalies, "crop_anomalies"))
    print(summary(national, "national"))
    no_climate = sorted(set(metrics["iso3"]) - set(climate["iso3"]))
    print(f"\ncountries: {len(metrics)} (included in quality.csv: {len(included)}; without climate data: {len(no_climate)} {no_climate})")
    print(f"events used: {int(metrics['n_events'].sum())} in {int((metrics['n_events'] > 0).sum())} countries "
          f"(catalogue: {int(diag['n_events_catalogue'].sum())})")
    missing = [f"{c}={100 * metrics[c].isna().mean():.0f}" for c in metrics.columns if metrics[c].isna().any()]
    print("% missing per metric: " + (", ".join(missing) or "none"))
    print("medians: " + ", ".join(f"{c}={metrics[c].median():.3g}" for c in metrics.columns[1:]))


if __name__ == "__main__":
    main()
