"""Out-of-sample test of the crop drought-response slopes -> outputs/stats/ and figures/results/.

Each crop's response slope (OLS of yield anomaly on SPEI-12) is estimated on TRAIN (1993-2012)
and evaluated on TEST (2013-2023).

  oos_crop.csv       one row per iso3 x item: n_train, n_test, beta_train, beta_test (slope fitted on
                     TEST only), value_share, and on TEST years the Pearson r, RMSE of the prediction
                     (intercept_train + beta_train * spei), RMSE of the zero-anomaly baseline,
                     skill = 1 - MSE_pred / MSE_baseline
  oos_national.csv   one row per iso3 x test year: pred = sum_i s_i * beta_train_i * spei_it,
                     actual = sum_i s_i * yield_anom_it (s_i = value_share renormalised over the crops
                     with a training slope and data in that year), covered_share = sum of the
                     (unrenormalised) value_share of those crops, plus the national value anomaly
                     `anom` and `is_drought` from metrics_national.parquet
  oos_summary.csv    quantity / value rows
  figures/results/08_oos_prediction.png

Trends, crop selection and value weights use the training period. Holdout yields and
national values are compared with the last fitted training-period LOWESS level.
Predictions are conditional on observed holdout SPEI and on the reporting-quality sample.

Run: python -m src.metrics.oos        (USE_DUMMY=1 for dummy inputs)
"""
import numpy as np
import pandas as pd

from ..common import paths
from ..common.contract import YEAR_MAX, YEAR_MIN
from . import response
from .detrend import FRAC, MIN_POINTS, detrend
from .run import crop_exposure, YEARS, load_inputs
from .stats import spearman_boot

TRAIN = (1993, 2012)
TEST = (2013, 2023)
MIN_TRAIN_YEARS = 15
MIN_TEST_YEARS = 8
MIN_SHARE = response.MIN_SHARE
SERIES_COLS = ["iso3", "item", "year", "yield_anom", "spei", "value_share"]


def training_anomaly(values, years, frac=FRAC):
    """Fit the trend on training observations and freeze its last level throughout holdout."""
    y = np.asarray(values, dtype=float)
    train = np.asarray(years) <= TRAIN[1]
    result = np.full(len(y), np.nan)
    if np.isfinite(y[train]).sum() >= MIN_POINTS:
        fit = detrend(y[train], np.asarray(years)[train], frac)
        result[train] = fit["anom"]
        levels = fit.loc[fit["trend"].gt(0), "trend"]
        if len(levels):
            result[~train] = y[~train] / levels.iloc[-1] - 1
    return result


def country_series(iso3, g, spei, c_spei_df, frac=FRAC):
    """Per-crop yield anomaly and crop SPEI pairs for one country (long DataFrame, SERIES_COLS),
    with crop shares, eligibility and trends estimated from the training period."""
    spei = spei.reindex(YEARS)
    g = g[g["value_const"].notna()]
    pivot = (c_spei_df.pivot(index="year", columns="mapspam_crop", values="crop_spei12_w")
             if not c_spei_df.empty else pd.DataFrame())
    train_value = g.loc[g.year.between(*TRAIN)].groupby("item_code").value_const.sum()
    share = train_value / train_value.sum()
    out = []
    for code, c in g.groupby("item_code"):
        c = c[~c["is_imputed"] & np.isfinite(c["yield_t_ha"]) & (c["yield_t_ha"] > 0)]
        c = c.sort_values("year")
        if share.get(code, 0) < MIN_SHARE or c.year.between(*TRAIN).sum() < MIN_TRAIN_YEARS:
            continue
        anomaly = training_anomaly(c.yield_t_ha, c.year, frac)
        item = c["item"].iloc[0]
        crop_spei = crop_exposure(code, pivot, c.year)
        out.append(pd.DataFrame({"iso3": iso3, "item": item, "year": c["year"].to_numpy(),
                                 "yield_anom": anomaly, "spei": crop_spei.to_numpy(),
                                 "value_share": float(share[code])}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=SERIES_COLS)


def build_series(crops, climate, crop_climate, included):
    spei_by = {k: v.set_index("year")["spei12_w"] for k, v in climate.groupby("iso3")}
    cc_by = {k: v for k, v in crop_climate.groupby("iso3")}
    empty_cc = pd.DataFrame(columns=["year", "mapspam_crop", "crop_spei12_w"])
    parts = [country_series(iso3, g, spei_by.get(iso3, pd.Series(dtype=float)), cc_by.get(iso3, empty_cc))
             for iso3, g in crops[crops["iso3"].isin(included)].groupby("iso3", sort=True)]
    parts = [p for p in parts if len(p)]
    s = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=SERIES_COLS)
    return s.astype({"iso3": str, "item": str, "year": int, "yield_anom": float, "spei": float, "value_share": float})


def fit_line(spei, anom):
    """OLS of anom on spei over finite pairs. Returns (n, slope, intercept); NaN slope if SPEI is constant."""
    s, a = np.asarray(spei, dtype=float), np.asarray(anom, dtype=float)
    ok = np.isfinite(s) & np.isfinite(a)
    n = int(ok.sum())
    if n < 2 or np.ptp(s[ok]) == 0:
        return n, np.nan, np.nan
    slope, icpt = np.polyfit(s[ok], a[ok], 1)
    return n, float(slope), float(icpt)


def train_slopes(series, min_years=MIN_TRAIN_YEARS):
    """One row per iso3 x item with a slope fitted on TRAIN; crops with fewer than min_years pairs are dropped."""
    rows = []
    tr = series[series["year"].between(*TRAIN)]
    for (iso3, item), c in tr.groupby(["iso3", "item"], sort=True):
        n, b, a = fit_line(c["spei"], c["yield_anom"])
        if n < min_years or not np.isfinite(b):
            continue
        rows.append({"iso3": iso3, "item": item, "n_train": n, "beta_train": b, "intercept_train": a,
                     "value_share": float(c["value_share"].iloc[0])})
    return pd.DataFrame(rows, columns=["iso3", "item", "n_train", "beta_train", "intercept_train", "value_share"])


def _pearson(x, y):
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def crop_table(series, slopes, min_test=MIN_TEST_YEARS):
    """Crop-level TEST evaluation for crops in `slopes` with at least min_test finite TEST pairs."""
    te = series[series["year"].between(*TEST) & series["spei"].notna() & series["yield_anom"].notna()]
    te = te.set_index(["iso3", "item"]).sort_index()
    rows = []
    for r in slopes.itertuples():
        if (r.iso3, r.item) not in te.index:
            continue
        c = te.loc[(r.iso3, r.item)]
        if len(c) < min_test:
            continue
        pred = r.intercept_train + r.beta_train * c["spei"].to_numpy()
        act = c["yield_anom"].to_numpy()
        mse_pred, mse_base = float(np.mean((act - pred) ** 2)), float(np.mean(act ** 2))
        _, beta_test, _ = fit_line(c["spei"], act)
        rows.append({"iso3": r.iso3, "item": r.item, "n_train": r.n_train, "n_test": len(c),
                     "beta_train": r.beta_train, "beta_test": beta_test, "value_share": r.value_share,
                     "r": _pearson(pred, act), "rmse_pred": np.sqrt(mse_pred), "rmse_baseline": np.sqrt(mse_base),
                     "skill": 1 - mse_pred / mse_base if mse_base > 0 else np.nan})
    cols = ["iso3", "item", "n_train", "n_test", "beta_train", "beta_test", "value_share",
            "r", "rmse_pred", "rmse_baseline", "skill"]
    return pd.DataFrame(rows, columns=cols)


def national_table(series, slopes, national):
    """Predicted vs actual national yield-driven anomaly for each iso3 x TEST year."""
    te = series[series["year"].between(*TEST) & series["spei"].notna() & series["yield_anom"].notna()]
    d = te.merge(slopes[["iso3", "item", "beta_train"]], on=["iso3", "item"])
    d["s"] = d["value_share"] / d.groupby(["iso3", "year"])["value_share"].transform("sum")
    d["pred_i"] = d["s"] * d["beta_train"] * d["spei"]
    d["act_i"] = d["s"] * d["yield_anom"]
    out = d.groupby(["iso3", "year"], as_index=False).agg(
        pred=("pred_i", "sum"), actual=("act_i", "sum"), covered_share=("value_share", "sum"), n_crops=("item", "size"))
    ref = national[["iso3", "year", "anom", "is_drought"]]
    return out.merge(ref, on=["iso3", "year"], how="left")


def _boot_rows(name, res):
    return [(f"{name}_{k}", res[k]) for k in ("rho", "ci_lo", "ci_hi", "n")]


def summary_table(crop, nat, slopes, national):
    rows = [("n_crops_tested", len(crop)), ("n_countries_crops", crop["iso3"].nunique()),
            ("n_countries_national", nat["iso3"].nunique()), ("n_national_country_years", len(nat))]
    rows += _boot_rows("spearman_beta_train_vs_beta_test", spearman_boot(crop["beta_train"], crop["beta_test"]))
    ok = crop["beta_test"].notna()
    rows.append(("share_same_sign", float((np.sign(crop.loc[ok, "beta_train"]) == np.sign(crop.loc[ok, "beta_test"])).mean())))
    rows += [("median_crop_skill", float(crop["skill"].median())), ("share_skill_positive", float((crop["skill"] > 0).mean())),
             ("pooled_r_national_all", _pearson(nat["pred"].to_numpy(), nat["actual"].to_numpy())),
             ("n_national_all", len(nat))]
    dr = nat[nat["is_drought"].fillna(False).astype(bool)]
    rows += [("pooled_r_national_drought", _pearson(dr["pred"].to_numpy(), dr["actual"].to_numpy())),
             ("n_national_drought", len(dr))]
    # Training-period resp_div vs SD of the TEST-period national anomaly.
    div = slopes.groupby("iso3").apply(lambda x: response.resp_div(x["beta_train"], x["value_share"]), include_groups=False)
    te = national[national["year"].between(*TEST)]
    sd_value = te.groupby("iso3")["anom"].agg(lambda x: x.std(ddof=1) if x.notna().sum() >= MIN_TEST_YEARS else np.nan)
    sd_yield = nat.groupby("iso3")["actual"].agg(lambda x: x.std(ddof=1) if x.notna().sum() >= MIN_TEST_YEARS else np.nan)
    rows += _boot_rows("spearman_resp_div_train_vs_sd_value_anom_test", spearman_boot(div, sd_value.reindex(div.index)))
    rows += _boot_rows("spearman_resp_div_train_vs_sd_yield_anom_test", spearman_boot(div, sd_yield.reindex(div.index)))
    return pd.DataFrame(rows, columns=["quantity", "value"])


def full_window_check(series, anomalies):
    """Slopes fitted on all years (response.resp_beta) vs resp_beta in crop_anomalies.parquet.
    Returns (n_crops_compared, max_abs_difference)."""
    mine = series.groupby(["iso3", "item"]).apply(
        lambda c: response.resp_beta(c["yield_anom"], c["spei"]), include_groups=False).rename("beta_all")
    ref = anomalies.drop_duplicates(["iso3", "item"]).set_index(["iso3", "item"])["resp_beta"]
    j = pd.concat([mine, ref], axis=1, join="inner")
    return len(j), float((j["beta_all"] - j["resp_beta"]).abs().max())


def fig_oos(crop, nat, n_countries):
    import matplotlib.pyplot as plt
    from .figures import FIGSIZE, INK, MUTED, save

    c = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    ax = axes[0]
    d = crop.dropna(subset=["beta_train", "beta_test"])
    lim = float(np.nanpercentile(np.abs(np.r_[d["beta_train"], d["beta_test"]]), 99)) * 1.1
    ax.scatter(d["beta_train"], d["beta_test"], s=10, color=c[0], alpha=0.6, edgecolor="white", linewidth=0.3)
    ax.plot([-lim, lim], [-lim, lim], color=MUTED, linewidth=0.8, label="1:1")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Slope fitted 1993-2012 (yield anomaly per unit SPEI-12)")
    ax.set_ylabel("Slope fitted 2013-2023 (yield anomaly per unit SPEI-12)")
    ax.set_title(f"Crop slopes, n = {len(d)} crops (axes clipped at 99th pct)", fontsize=8, color=INK, loc="left")
    ax.legend(frameon=False, loc="upper left")
    ax = axes[1]
    dr = nat[nat["is_drought"].fillna(False).astype(bool)]
    ax.scatter(dr["pred"], dr["actual"], s=10, color=c[1], alpha=0.6, edgecolor="white", linewidth=0.3)
    if len(dr):
        lim = float(np.nanpercentile(np.abs(np.r_[dr["pred"], dr["actual"]]), 99)) * 1.1
        ax.plot([-lim, lim], [-lim, lim], color=MUTED, linewidth=0.8)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    ax.set_xlabel("Predicted national yield anomaly (fraction of trend)")
    ax.set_ylabel("Actual national yield anomaly (fraction of trend)")
    ax.set_title(f"Drought years 2013-2023, n = {len(dr)} country-years", fontsize=8, color=INK, loc="left")
    fig.suptitle("Out-of-sample prediction from slopes fitted on 1993-2012", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    save(fig, "08_oos_prediction.png", n_countries)


def main():
    crops, climate, crop_climate, events, included = load_inputs()
    series = build_series(crops, climate, crop_climate, included)
    national = pd.read_parquet(paths.NATIONAL)
    national = national.sort_values(["iso3", "year"])
    for _, group in national.groupby("iso3"):
        national.loc[group.index, "anom"] = training_anomaly(group.value_const, group.year)

    slopes = train_slopes(series)
    crop = crop_table(series, slopes)
    nat = national_table(series, slopes, national)
    summ = summary_table(crop, nat, slopes, national)

    assert not crop.duplicated(["iso3", "item"]).any()
    assert not nat.duplicated(["iso3", "year"]).any()
    assert nat["year"].between(*TEST).all()
    assert YEAR_MIN <= TEST[0] and TEST[1] <= YEAR_MAX
    assert np.isfinite(crop["beta_train"]).all() and np.isfinite(crop["beta_test"]).all()
    assert crop["value_share"].between(0, 1).all() and nat["covered_share"].between(0, 1 + 1e-9).all()

    paths.STATS_DIR.mkdir(parents=True, exist_ok=True)
    crop.to_csv(paths.STATS_DIR / "oos_crop.csv", index=False)
    nat.to_csv(paths.STATS_DIR / "oos_national.csv", index=False)
    summ.to_csv(paths.STATS_DIR / "oos_summary.csv", index=False)
    fig_oos(crop, nat, nat["iso3"].nunique())

    print(f"train {TRAIN}, test {TEST}, crops with training slopes: {len(slopes)}")
    for r in summ.itertuples():
        print(f"  {r.quantity:<52} {r.value:.4g}")


if __name__ == "__main__":
    main()
