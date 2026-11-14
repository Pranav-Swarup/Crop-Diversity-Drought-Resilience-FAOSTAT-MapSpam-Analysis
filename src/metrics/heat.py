"""Resilience faces against heat (hot warm seasons) and compound hot-dry years, for included countries.

Heat variable: heat_z_w from data/clean/heat.parquet (src.climate.heat): cropland-weighted Tmax of the 3
warmest months, LOWESS-detrended and standardised per country. Hot year: heat_z_w >= +1.0.
Dry year: spei12_w <= -1.0. Compound year: both.

Outputs:
  outputs/metrics_heat.parquet        iso3, resistance_heat, recovery_heat, vuln_heat, vuln_compound,
                                      n_heat_events, n_hot_years, n_compound_years
  outputs/stats/heat_spearman.csv     Spearman rho (bootstrap 95% CI) of each heat face vs hill1 and resp_div
  outputs/stats/heat_panel.csv        model, term, beta, se, ci_lo, ci_hi, p, n_obs, n_countries, r2_within
  figures/results/10_heat_compound.png

Panel models (country and year fixed effects, SE clustered by country; estimator = src.metrics.panel.fe_fit;
outcome `anom` = relative anomaly of national crop value; z(hill1) across the sample countries):
  heat_only         anom ~ heat + heat z(hill1)
  additive          anom ~ spei + heat + spei z(hill1) + heat z(hill1)
  additive_stn      additive, dropping countries with mean stn_share_w < 0.5 (CRU there is mostly climatology)
  year_types        anom ~ dry_only + hot_only + compound          (0/1 dummies; reference = neither)
  year_types_hill1  year_types + each dummy x z(hill1)
`compound_minus_sum` (model year_types) = compound - (dry_only + hot_only): the extra loss of a compound year
beyond the sum of a dry-only and a hot-only year. Its SE is the delta method on the clustered covariance.

resp_div is the diversity of crop responses to SPEI, not to heat; its rows in heat_spearman.csv are a
cross-check, not a test of heat-response diversity.

Needs outputs/metrics_national.parquet, outputs/metrics.parquet, data/clean/heat.parquet, heat_events.csv.

Run: python -m src.metrics.heat      (real data only: heat has no dummy version)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as st

from ..common import paths
from ..common.contract import load, summary, validate
from . import faces, panel
from .data import PREDICTORS, analysis_table
from .events import DROUGHT_THR
from .figures import DPI, FIGSIZE, INK, MUTED, region_colors
from .stats import spearman_boot

HOT_THR = 1.0
STN_MIN = 0.5
HEAT_FACES = ["resistance_heat", "recovery_heat", "vuln_heat", "vuln_compound"]
YEAR_TYPES = ["dry_only", "hot_only", "compound"]
TYPE_LABELS = {"dry_only": "Dry only", "hot_only": "Hot only", "compound": "Hot and dry"}
COLS = ["model", "term", "beta", "se", "ci_lo", "ci_hi", "p", "n_obs", "n_countries", "r2_within"]
FIG_NAME = "10_heat_compound.png"


def year_flags(df):
    """Add dry_yr, hot_yr and the exclusive 0/1 dummies dry_only, hot_only, compound to a frame with
    spei and heat. Rows with a missing spei or heat get NaN dummies."""
    df = df.copy()
    ok = df["spei"].notna() & df["heat"].notna()
    dry, hot = df["spei"] <= DROUGHT_THR, df["heat"] >= HOT_THR
    df["dry_only"] = (dry & ~hot).astype(float).where(ok)
    df["hot_only"] = (hot & ~dry).astype(float).where(ok)
    df["compound"] = (dry & hot).astype(float).where(ok)
    return df


def compute(national, climate, heat, heat_ev):
    """One row per country in `national`."""
    ev_by = heat_ev.groupby("iso3")["year"].apply(list).to_dict()
    yrs = year_flags(climate[["iso3", "year", "spei12_w"]].rename(columns={"spei12_w": "spei"})
                     .merge(heat[["iso3", "year", "heat_z_w"]].rename(columns={"heat_z_w": "heat"}),
                            on=["iso3", "year"], how="outer"))
    hot_years = yrs[yrs["heat"] >= HOT_THR].groupby("iso3")["year"].apply(list).to_dict()
    comp_years = yrs[yrs["compound"] == 1].groupby("iso3")["year"].apply(list).to_dict()
    rows = []
    for iso3, g in national.groupby("iso3", sort=True):
        index = g.set_index("year")["index"]
        ev, hot, comp = ev_by.get(iso3, []), hot_years.get(iso3, []), comp_years.get(iso3, [])
        rows.append({"iso3": iso3,
                     "resistance_heat": faces.resistance(index, ev),
                     "recovery_heat": faces.recovery(index, ev),
                     "vuln_heat": faces.vulnerability(index, hot),
                     "vuln_compound": faces.vulnerability(index, comp),
                     "n_heat_events": faces.n_usable_events(index, ev),
                     "n_hot_years": int(index.dropna().index.isin(hot).sum()),
                     "n_compound_years": int(index.dropna().index.isin(comp).sum())})
    out = pd.DataFrame(rows)
    return out.astype({"n_heat_events": int, "n_hot_years": int, "n_compound_years": int})


def spearman_heat_table(df):
    rows = [{"face": f, "predictor": p, **spearman_boot(df[p], df[f])} for f in HEAT_FACES for p in PREDICTORS]
    return pd.DataFrame(rows)[["face", "predictor", "rho", "ci_lo", "ci_hi", "p", "n"]]


def panel_frame(heat):
    """Country-year panel of src.metrics.panel plus heat, stn_mean and the year-type dummies."""
    df = panel.load_panel().merge(
        heat[["iso3", "year", "heat_z_w"]].rename(columns={"heat_z_w": "heat"}), on=["iso3", "year"], how="left")
    df = df.merge(heat.groupby("iso3")["stn_share_w"].mean().rename("stn_mean").reset_index(), on="iso3", how="left")
    return year_flags(df)


def linear_combo(fit, weights):
    """Estimate, SE, CI and p of sum(weights[term] * beta[term]) from the clustered covariance."""
    w = np.array([weights.get(t, 0.0) for t in fit["params"].index])
    b = float(w @ fit["params"].to_numpy())
    se = float(np.sqrt(w @ fit["cov"].to_numpy() @ w))
    tcrit = st.t.ppf(0.975, fit["df_t"])
    return {"beta": b, "se": se, "ci_lo": b - tcrit * se, "ci_hi": b + tcrit * se,
            "p": float(2 * st.t.sf(abs(b / se), fit["df_t"]))}


def run_models(df):
    """Returns (coefficient table, dict of fits)."""
    specs = {
        "heat_only": (df, ["heat"], ["hill1"]),
        "additive": (df, ["spei", "heat"], ["hill1"]),
        "additive_stn": (df[df["stn_mean"] >= STN_MIN], ["spei", "heat"], ["hill1"]),
        "year_types": (df, YEAR_TYPES, []),
        "year_types_hill1": (df, YEAR_TYPES, ["hill1"]),
    }
    rows, fits = [], {}
    for name, (sample, slopes, inter) in specs.items():
        res = panel.fit_model(sample, slopes, inter, (), ())
        assert res is not None, f"{name}: too few countries"
        fit = res[0]
        fits[name] = fit
        meta = {"n_obs": fit["nobs"], "n_countries": fit["n_clusters"], "r2_within": fit["r2_within"]}
        rows += [{"model": name, **r, **meta} for r in panel.coef_rows(fit)]
        if name == "year_types":
            combo = linear_combo(fit, {"compound": 1.0, "dry_only": -1.0, "hot_only": -1.0})
            rows.append({"model": name, "term": "compound_minus_sum", **combo, **meta})
    return pd.DataFrame(rows)[COLS], fits


def fig_heat(df, coef, spearman, n_types):
    """Left: FE estimates of the value anomaly by year type. Right: hill1 vs vuln_heat."""
    colors = region_colors(df)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE)

    c = coef[(coef["model"] == "year_types") & coef["term"].isin(YEAR_TYPES)].set_index("term").loc[YEAR_TYPES]
    x = np.arange(len(YEAR_TYPES))
    palette = plt.get_cmap("tab10")
    ax1.bar(x, 100 * c["beta"], color=[palette(1), palette(3), palette(5)], width=0.6)
    ax1.errorbar(x, 100 * c["beta"], yerr=[100 * (c["beta"] - c["ci_lo"]), 100 * (c["ci_hi"] - c["beta"])],
                 fmt="none", ecolor=INK, capsize=3, linewidth=0.8)
    ax1.axhline(0, color=MUTED, linewidth=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{TYPE_LABELS[t]}\n(n = {n_types[t]})" for t in YEAR_TYPES])
    ax1.set_ylabel("National crop value anomaly vs. normal years (% of trend)")
    ax1.set_title("Value anomaly by year type\ncountry and year fixed effects, 95% CI (clustered by country)",
                  fontsize=7.5, color=INK, loc="left")

    d = df.dropna(subset=["hill1", "vuln_heat"])
    for region, g in d.groupby("region"):
        ax2.scatter(g["hill1"], g["vuln_heat"], s=16, color=colors[region], edgecolor="white", linewidth=0.4)
    s = spearman[(spearman["face"] == "vuln_heat") & (spearman["predictor"] == "hill1")].iloc[0]
    ax2.set_title(f"Vulnerability in hot years, by country\nSpearman ρ = {s.rho:+.2f} [{s.ci_lo:+.2f}, {s.ci_hi:+.2f}], n = {s.n}",
                  fontsize=7.5, color=INK, loc="left")
    ax2.set_xlabel("Crop diversity, hill1 (effective number of crops)")
    ax2.set_ylabel("Share of hot years with $I_t$ < 0.9 (vuln_heat)")

    handles = [plt.Line2D([], [], marker="o", ls="", color=col, markersize=4, label=r) for r, col in colors.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 0.035))
    fig.suptitle("Heat, drought and compound hot-dry years", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.11, 1, 0.95))
    fig.text(0.01, 0.01, f"Source: FAOSTAT (QCL, QV), SPEIbase v2.10, CRU TS v4.10 (Tmax), MapSPAM 2020. 1993–2023, "
             f"n = {len(df)} countries. Hot: detrended warm-season Tmax z ≥ 1; dry: SPEI-12 ≤ −1.",
             fontsize=6, color=MUTED, ha="left", va="bottom")
    out = paths.fig_dir("results") / FIG_NAME
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out.relative_to(paths.ROOT)}")


def main():
    assert not paths.USE_DUMMY, "heat has no dummy data: run without USE_DUMMY"
    national = pd.read_parquet(paths.NATIONAL)
    climate, heat, heat_ev = load("climate"), load("heat"), load("heat_events")
    included = load("quality").query("included")["iso3"]
    national = national[national["iso3"].isin(included)]
    assert set(national["iso3"]) <= set(heat["iso3"]), "included country without a heat series"

    mh = compute(national, climate, heat, heat_ev)
    validate(mh, "metrics_heat")
    assert set(mh["iso3"]) <= set(included)
    assert mh["n_heat_events"].sum() <= len(heat_ev)
    assert (mh["n_compound_years"] <= mh["n_hot_years"]).all()

    df = analysis_table().merge(mh, on="iso3", how="inner")
    sp = spearman_heat_table(df)
    assert not sp.duplicated(["face", "predictor"]).any() and len(sp) == len(HEAT_FACES) * len(PREDICTORS)
    assert sp["rho"].dropna().between(-1, 1).all()

    pf = panel_frame(heat)
    coef, fits = run_models(pf)
    assert not coef.duplicated(["model", "term"]).any()
    assert np.isfinite(coef[["beta", "se"]]).all().all() and (coef["se"] > 0).all()
    assert (coef["ci_lo"] <= coef["beta"]).all() and (coef["beta"] <= coef["ci_hi"]).all()
    assert coef["p"].between(0, 1).all() and coef["r2_within"].between(0, 1).all()
    assert (pf[YEAR_TYPES].sum(axis=1).dropna() <= 1).all(), "year-type dummies overlap"

    paths.OUT.mkdir(parents=True, exist_ok=True)
    paths.STATS_DIR.mkdir(parents=True, exist_ok=True)
    mh.to_parquet(paths.METRICS_HEAT, index=False)
    sp.to_csv(paths.STATS_DIR / "heat_spearman.csv", index=False)
    coef.to_csv(paths.STATS_DIR / "heat_panel.csv", index=False)

    print(summary(mh, "metrics_heat"))
    print(f"countries with >=1 usable heat event: {(mh['n_heat_events'] > 0).sum()}; heat events used: {int(mh['n_heat_events'].sum())}; "
          f"hot years: {int(mh['n_hot_years'].sum())}; compound years: {int(mh['n_compound_years'].sum())} "
          f"in {(mh['n_compound_years'] > 0).sum()} countries")
    print("medians: " + ", ".join(f"{c}={mh[c].median():.3f}" for c in HEAT_FACES))
    print("\nSpearman rho [bootstrap 95% CI]:")
    for r in sp.itertuples():
        print(f"  {r.face:<16} vs {r.predictor:<8} rho={r.rho:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}]  p={r.p:.3f}  n={r.n}")
    pd.set_option("display.width", 220)
    print("\npanel coefficients (anom in fractions of trend; spei in SPEI units; heat in SD; hill1 as z-score):")
    print(coef.round(4).to_string(index=False))

    used = pf.dropna(subset=["anom"] + YEAR_TYPES)
    n_types = {t: int(used[t].sum()) for t in YEAR_TYPES}
    fig_heat(df, coef, sp, n_types)


if __name__ == "__main__":
    main()
