"""Trade and price buffers -> outputs/stats/trade_buffer.csv, price_response.csv, figures/data/03 and 04.

trade_buffer.csv (included countries, data/clean/quality.csv):
  per country, OLS slope of the LOWESS relative anomaly of cereal import_t (src.metrics.detrend.detrend,
  x = year) on spei12_w. The FBSH (1993-2009) and FBS (2010-2023) segments are detrended separately
  because of the methodology break; the two anomaly series are pooled for the regression. Also mean import_dep.
  Rows with iso3 = "ALL": Spearman rho and bootstrap 95% CI (src.metrics.stats.spearman_boot) between the
  countries' mean import_dep and each of stability, resistance, vulnerability and hill1 (outputs/metrics.parquet),
  one row per metric in column `outcome`.
price_response.csv (included countries):
  per iso3 x item with >= 15 non-imputed price years, OLS slope of the LOWESS relative anomaly of the
  nominal USD producer price on the crop's own yield anomaly (outputs/crop_anomalies.parquet).
  Prices are nominal USD and no deflator is available, so the anomaly is of the nominal price; price
  changes common to all crops (general inflation, exchange rates) are not removed beyond the LOWESS trend.
  Per country, item = "ALL" gives the mean slope weighted by each item's mean value_share (n = items used).

Run: python -m src.crops.buffers
"""
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from ..common import paths
from ..common.contract import YEAR_MAX, YEAR_MIN, load
from ..metrics.detrend import detrend
from ..metrics.stats import spearman_boot

MIN_PRICE_YEARS = 15
MIN_TRADE_YEARS = 10
BUFFER_FACES = ["stability", "resistance", "vulnerability", "hill1"]
OUT_FIG = paths.fig_dir("data")
FIG_TRADE, FIG_PRICE = "03_import_dependency.png", "04_price_response.png"
SRC_TRADE = "Source: FAOSTAT Food Balances (FBSH 1993-2009, FBS 2010-2023), cereals excluding beer; import_dep = imports / (production + imports - exports)."
SRC_PRICE = "Source: FAOSTAT Producer Prices (nominal USD/t, annual) and QCL yields; LOWESS anomalies (frac 0.5)."


def ols_slope(y, x):
    """Slope, p and n of OLS y ~ x on complete pairs (NaN if fewer than 3 pairs or x constant)."""
    d = pd.DataFrame({"y": np.asarray(y, dtype=float), "x": np.asarray(x, dtype=float)}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 3 or d["x"].std() == 0:
        return np.nan, np.nan, len(d)
    fit = sm.OLS(d["y"], sm.add_constant(d["x"])).fit()
    return float(fit.params["x"]), float(fit.pvalues["x"]), len(d)


def trade_buffer(trade, climate, included, metrics):
    t = trade[trade["iso3"].isin(included)].sort_values(["iso3", "year"])
    rows = []
    for iso3, g in t.groupby("iso3"):
        parts = []
        for _, seg in g.groupby("fbs_source"):
            if seg["import_t"].notna().sum() >= MIN_TRADE_YEARS:
                an = detrend(seg["import_t"].to_numpy(), seg["year"].to_numpy())["anom"].to_numpy()
                parts.append(pd.DataFrame({"year": seg["year"].to_numpy(), "anom": an}))
        slope, p, n = np.nan, np.nan, 0
        if parts:
            a = pd.concat(parts).merge(climate[climate["iso3"] == iso3][["year", "spei12_w"]], on="year")
            slope, p, n = ols_slope(a["anom"], a["spei12_w"])
        rows.append({"iso3": iso3, "outcome": "", "n": n, "slope": slope, "p": p, "mean_import_dep": g["import_dep"].mean(),
                     "rho": np.nan, "ci_lo": np.nan, "ci_hi": np.nan})
    out = pd.DataFrame(rows)
    m = out[["iso3", "mean_import_dep"]].merge(metrics, on="iso3")
    summ = []
    for face in BUFFER_FACES:
        r = spearman_boot(m["mean_import_dep"], m[face])
        summ.append({"iso3": "ALL", "outcome": face, "n": r["n"], "slope": np.nan, "p": r["p"], "mean_import_dep": np.nan,
                     "rho": r["rho"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"]})
    return pd.concat([out, pd.DataFrame(summ)], ignore_index=True)


def price_response(prices, anoms, included):
    p = prices[prices["iso3"].isin(included) & ~prices["is_imputed"]].copy()
    n_years = p.groupby(["iso3", "item"])["year"].transform("size")
    p = p[n_years >= MIN_PRICE_YEARS].sort_values(["iso3", "item", "year"])
    ya = anoms[["iso3", "year", "item", "yield_anom"]]
    rows = []
    for (iso3, item), g in p.groupby(["iso3", "item"]):
        # detrend over the full window grid so that years without a price stay gaps
        grid = pd.DataFrame({"year": np.arange(YEAR_MIN, YEAR_MAX + 1)}).merge(g[["year", "price_usd_t"]], on="year", how="left")
        grid["price_anom"] = detrend(grid["price_usd_t"].to_numpy(), grid["year"].to_numpy())["anom"].to_numpy()
        a = grid.merge(ya[(ya["iso3"] == iso3) & (ya["item"] == item)], on="year")
        slope, pv, n = ols_slope(a["price_anom"], a["yield_anom"])
        if n >= MIN_PRICE_YEARS and np.isfinite(slope):
            rows.append({"iso3": iso3, "item": item, "n": n, "slope": slope, "p": pv})
    items = pd.DataFrame(rows)
    share = anoms.groupby(["iso3", "item"], as_index=False)["value_share"].mean().rename(columns={"value_share": "mean_value_share"})
    items = items.merge(share, on=["iso3", "item"], how="left")
    ok = items[items["mean_value_share"] > 0]
    w = ok.groupby("iso3").apply(lambda g: pd.Series({"n": len(g), "slope": np.average(g["slope"], weights=g["mean_value_share"]),
                                                      "mean_value_share": g["mean_value_share"].sum()}), include_groups=False).reset_index()
    w["item"], w["p"] = "ALL", np.nan
    w["n"] = w["n"].astype(int)
    return pd.concat([items, w[items.columns]], ignore_index=True)[["iso3", "item", "n", "slope", "p", "mean_value_share"]]


def setup():
    plt.rcParams.update({"font.family": "sans-serif", "figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white", "font.size": 10})


def fig_import_dependency(tb):
    geo = gpd.read_file(paths.COUNTRIES_GEOJSON).merge(tb[["iso3", "mean_import_dep"]], on="iso3", how="left")
    fig, ax = plt.subplots(figsize=(8, 5))
    geo.plot(ax=ax, color="#d9d9d9", linewidth=0.2, edgecolor="white")
    geo.dropna(subset=["mean_import_dep"]).plot(ax=ax, column="mean_import_dep", cmap="viridis", vmin=0, vmax=1, linewidth=0.2,
                                                edgecolor="white", legend=True,
                                                legend_kwds={"label": "Mean cereal import dependency, 1993-2023 (ratio; colour scale capped at 1)",
                                                             "orientation": "horizontal", "shrink": 0.6, "pad": 0.22})
    ax.set_xlim(-180, 180)
    ax.set_ylim(-58, 84)
    ax.set_xlabel("Longitude (degrees)")
    ax.set_ylabel("Latitude (degrees)")
    ax.set_title("Cereal import dependency, included countries")
    fig.text(0.01, 0.01, SRC_TRADE + " Grey = not included or no data.", fontsize=5.5, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT_FIG / FIG_TRADE, dpi=300)
    plt.close(fig)


def fig_price_response(items, regions):
    d = items.merge(regions, on="iso3")
    order = d.groupby("region")["slope"].median().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([d.loc[d["region"] == r, "slope"] for r in order], orientation="horizontal", showfliers=False,
               tick_labels=[f"{r} (n={int((d['region'] == r).sum())})" for r in order])
    ax.axvline(0, color="#555555", lw=0.8)
    ax.set_xlabel("Slope of price anomaly on yield anomaly\n(ratio, nominal USD price; outliers not shown)")
    ax.set_ylabel("World Bank region (country-crop pairs)")
    ax.set_title("Producer price response to the crop's own yield anomaly")
    fig.text(0.01, 0.01, SRC_PRICE, fontsize=7, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT_FIG / FIG_PRICE, dpi=300)
    plt.close(fig)


def main():
    setup()
    quality = load("quality")
    included = set(quality.loc[quality["included"], "iso3"])
    trade = pd.read_parquet(paths.TRADE)
    prices = pd.read_parquet(paths.PRICES)
    climate, metrics, anoms = load("climate"), load("metrics"), load("crop_anomalies")
    countries = pd.read_csv(paths.COUNTRIES, dtype=str, keep_default_na=False)

    tb = trade_buffer(trade, climate, included, metrics[["iso3"] + BUFFER_FACES])
    pr = price_response(prices, anoms, included)
    paths.STATS_DIR.mkdir(parents=True, exist_ok=True)
    tb.to_csv(paths.STATS_DIR / "trade_buffer.csv", index=False)
    pr.to_csv(paths.STATS_DIR / "price_response.csv", index=False)

    c = tb[tb["iso3"] != "ALL"]
    assert set(tb.loc[tb["iso3"] == "ALL", "outcome"]) == set(BUFFER_FACES)
    assert not c.duplicated("iso3").any() and set(c["iso3"]) <= included
    assert c["n"].between(0, YEAR_MAX - YEAR_MIN + 1).all() and c["p"].dropna().between(0, 1).all()
    assert np.isfinite(c["slope"].dropna()).all() and (c["mean_import_dep"].dropna() >= 0).all()
    items = pr[pr["item"] != "ALL"]
    wmean = pr[pr["item"] == "ALL"]
    assert not items.duplicated(["iso3", "item"]).any() and (items["n"] >= MIN_PRICE_YEARS).all()
    assert items["p"].between(0, 1).all() and np.isfinite(items["slope"]).all()
    assert set(pr["iso3"]) <= included and not wmean.duplicated("iso3").any() and set(wmean["iso3"]) <= set(items["iso3"])

    fig_import_dependency(c)
    fig_price_response(items, countries[["iso3", "region"]])
    for f in (FIG_TRADE, FIG_PRICE):
        assert (OUT_FIG / f).stat().st_size > 20_000, f

    print(f"trade_buffer: {len(c)} countries, {int(c['slope'].notna().sum())} with a slope, "
          f"{int((c['p'] < 0.05).sum())} with p < 0.05, {100 * c.isna().to_numpy().mean():.1f}% missing cells; "
          f"included countries without trade rows: {len(included - set(trade['iso3']))}")
    print(tb[tb["iso3"] == "ALL"][["outcome", "n", "rho", "ci_lo", "ci_hi", "p"]].round(3).to_string(index=False))
    print(f"price_response: {len(items)} country-item slopes, {items['iso3'].nunique()} countries, {len(wmean)} country means, "
          f"{100 * pr.isna().to_numpy().mean():.1f}% missing cells; item slopes with p < 0.05: {100 * (items['p'] < 0.05).mean():.1f}%")


if __name__ == "__main__":
    main()
