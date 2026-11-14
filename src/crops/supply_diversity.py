"""Supply-source diversity of national cereal supply: own production plus import partners, weighted by
how differently their droughts behave.

A country's cereal supply comes from "sources": its own production and each import partner. Sources hit by
the same droughts are not separate bets. Each source's drought exposure series is that source country's
spei12_w (data/clean/climate.parquet, national cropland-weighted December SPEI-12, 1993-2023).

Per importer c (only `included` countries in data/clean/quality.csv), computed separately for
share_source = "reporter" (data/clean/trade_partners.parquet, importer-reported; >= 10 reporting years) and
share_source = "mirror" (data/clean/trade_partners_mirror.parquet, exporter-reported; >= 10 years with any
flow). The two sources are never blended.
  s_dom      = sum_t prod_t / sum_t (prod_t + import_t) over years where both are present in
               data/clean/trade.parquet (FAOSTAT Food Balances, cereals). This is a level share, so it uses
               both the FBSH (1993-2009) and the FBS (2010-2023) segments.
  w_p        = partner p's share of c's cereal grain import tonnes summed over the window. Partners without a
               complete 1993-2023 spei12_w series are dropped and w renormalised; partner_cover = share of
               import tonnes kept. Countries with no partner left get no row.
  s          = [s_dom] + [(1 - s_dom) w_p];  R = Pearson correlation of the sources' spei12_w series.
  supply_div     = lc_diversity(s)                         (effective number of sources, no similarity)
  supply_div_exp = lc_diversity(s, similarity_from_corr(R)) (similarity = shared drought exposure)
  supply_ratio   = supply_div_exp / supply_div
  import_div, import_div_exp = the same for w, with Z of the partners only
  partner_corr   = sum_p w_p corr(spei_c, spei_p)   (do my suppliers have drought when I do)
  partner_dry_share_t = sum_p w_p 1[spei_p,t <= -1.0];  partner_dry_own_dry / partner_dry_other = its mean over
               c's own drought years (spei12_w <= -1.0) / over its other years.
The index is src.metrics.similarity.lc_diversity (single implementation, Leinster & Cobbold 2012).

Limits (facts about the data, not caveats to be argued away):
  - partner shares use unprocessed grains (trade matrix), while the domestic/import split uses the Food
    Balance cereal aggregate, which also counts processed products in grain equivalent;
  - the recorded partner can be a re-exporter or transit country (e.g. NLD, BLX), not the grower;
  - a partner's exposure is its whole national cropland SPEI, not its cereal area or its export surplus;
  - shares are window totals (time-invariant), so a partner that mattered in some years only is averaged.

Outputs:
  outputs/supply_diversity.parquet   one row per iso3 x share_source (columns below in COLS)
  outputs/supply_exposure.parquet    iso3, share_source, year, partner_spei (= sum_p w_p spei_p,t), partner_dry_share
  outputs/stats/supply_spearman.csv  Spearman rho (bootstrap CI): stab_supply, stab_prod, supply_buffer vs diversity
  outputs/stats/supply_panel.csv     country-year panel models (below)
  outputs/stats/supply_source_compare.csv  agreement of the reporter and mirror versions
  figures/data/05_supply_diversity.png     (share_source = mirror, which covers more countries)

Outcomes: stab_supply = 1 / SD of the relative anomaly of supply_t; stab_prod = the same for prod_t;
supply_buffer = stab_supply / stab_prod. Anomalies are LOWESS (src.metrics.detrend.detrend), computed per
FBS segment (FBSH, FBS) with segments of fewer than 10 years skipped, then pooled (as trade_buffer in
src/crops/buffers.py does).

Panel models (src.metrics.panel.fe_fit and coef_rows: country and year fixed effects, SE clustered by country;
rows with any missing value dropped first). Year fixed effects absorb whatever is common to all countries in a
year, including world-wide drought shocks and price movements; the coefficients are within-year contrasts.
  imports     relative anomaly of import_t ~ own_dry + partner_dry_share + own_dry x partner_dry_share
  supply      relative anomaly of supply_t ~ the same terms
  supply_div  relative anomaly of supply_t ~ own_dry + own_dry x z(supply_div_exp) + own_dry x z(s_dom)
own_dry = 1 if the country's own spei12_w <= -1.0. z() is across the sample countries (one value per country);
the main effects of z() are absorbed by the country fixed effects.

Needs data/clean/trade_partners.parquet and trade_partners_mirror.parquet (python -m src.crops.trade_matrix),
trade.parquet, climate.parquet, quality.csv.

Run: python -m src.crops.supply_diversity
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..common import paths
from ..common.contract import YEAR_MAX, YEAR_MIN, load
from ..metrics import panel
from ..metrics.detrend import detrend
from ..metrics.events import DROUGHT_THR
from ..metrics.figures import DPI, FIGSIZE, INK, MUTED, region_colors
from ..metrics.similarity import lc_diversity, similarity_from_corr
from ..metrics.stats import spearman_boot

MIN_PARTNER_YEARS = 10
MIN_SEGMENT_YEARS = 10
YEARS = np.arange(YEAR_MIN, YEAR_MAX + 1)
SOURCES = {"reporter": paths.CLEAN / "trade_partners.parquet", "mirror": paths.CLEAN / "trade_partners_mirror.parquet"}
OUT_DIV = paths.OUT / "supply_diversity.parquet"
OUT_EXPO = paths.OUT / "supply_exposure.parquet"
FIG_NAME = "05_supply_diversity.png"
COLS = ["iso3", "share_source", "s_dom", "n_partners", "n_years", "partner_cover", "supply_div", "supply_div_exp",
        "supply_ratio", "import_div", "import_div_exp", "partner_corr", "partner_dry_own_dry", "partner_dry_other"]
PREDICTORS = ["supply_div", "supply_div_exp", "supply_ratio", "import_div_exp", "partner_corr", "s_dom"]
FACES = ["stab_supply", "stab_prod", "supply_buffer"]
PANEL_COLS = ["share_source", "model", "term", "beta", "se", "ci_lo", "ci_hi", "p", "n_obs", "n_countries", "r2_within"]
COMPARE_VARS = ["supply_div_exp", "import_div_exp", "partner_corr"]
SRC_LINE = ("Source: FAOSTAT Detailed Trade Matrix and Food Balances (cereals), SPEIbase v2.10, MapSPAM 2020, 1993-2023, "
            "n = {n} countries.\nPartner shares from exporter-reported flows (mirror), which cover more countries than importer reports.")


# ------------------------------------------------------------------ pure functions (tested)

def complete_spei(climate):
    """Wide table year x iso3 of spei12_w, keeping only countries with a complete 1993-2023 series."""
    w = climate.pivot(index="year", columns="iso3", values="spei12_w").reindex(YEARS)
    return w.loc[:, w.notna().all()]


def domestic_share(trade):
    """s_dom = sum prod_t / sum (prod_t + import_t) over the years where both are present (NaN if none)."""
    d = trade.dropna(subset=["prod_t", "import_t"])
    tot = (d["prod_t"] + d["import_t"]).sum()
    return float(d["prod_t"].sum() / tot) if len(d) and tot > 0 else np.nan


def partner_weights(flows, spei):
    """flows: Series partner iso3 -> import tonnes over the window. Returns (w, cover): shares of the partners
    that have a spei series (renormalised to sum 1) and the share of tonnes kept."""
    flows = flows[flows > 0]
    kept = flows[flows.index.isin(spei.columns)]
    cover = float(kept.sum() / flows.sum()) if flows.sum() > 0 else 0.0
    return (kept / kept.sum() if kept.sum() > 0 else kept), cover


def country_supply(iso3, s_dom, flows, spei, thr=DROUGHT_THR):
    """Supply-diversity row and yearly exposure for one importer, or None if it has no own series or no partner.
    flows: Series partner -> window import tonnes; spei: complete_spei() table."""
    if iso3 not in spei.columns:
        return None
    w, cover = partner_weights(flows, spei)
    w = w.drop(labels=[iso3], errors="ignore")
    if len(w) == 0:
        return None
    w = w / w.sum()
    own = spei[iso3]
    x = pd.concat([own, spei[w.index]], axis=1)
    r = np.corrcoef(x.to_numpy(), rowvar=False)
    z = similarity_from_corr(r)
    s = np.concatenate([[s_dom], (1.0 - s_dom) * w.to_numpy()])
    assert np.isclose(s.sum(), 1.0) and (s >= -1e-12).all()
    div, div_exp = lc_diversity(s), lc_diversity(s, z)
    imp_div, imp_div_exp = lc_diversity(w.to_numpy()), lc_diversity(w.to_numpy(), z[1:, 1:])
    expo = pd.DataFrame({"year": spei.index.to_numpy(),
                         "partner_spei": spei[w.index].to_numpy() @ w.to_numpy(),
                         "partner_dry_share": (spei[w.index].to_numpy() <= thr).astype(float) @ w.to_numpy()})
    dry = (own <= thr).to_numpy()
    row = {"iso3": iso3, "s_dom": float(s_dom), "n_partners": int(len(w)), "partner_cover": cover,
           "supply_div": div, "supply_div_exp": div_exp, "supply_ratio": div_exp / div,
           "import_div": imp_div, "import_div_exp": imp_div_exp,
           "partner_corr": float(w.to_numpy() @ r[0, 1:]),
           "partner_dry_own_dry": float(expo.loc[dry, "partner_dry_share"].mean()) if dry.any() else np.nan,
           "partner_dry_other": float(expo.loc[~dry, "partner_dry_share"].mean()) if (~dry).any() else np.nan}
    return row, expo


def segment_anomalies(trade, col):
    """Relative anomalies of `col` (LOWESS, per fbs_source segment, segments of < 10 years skipped), pooled.
    Returns DataFrame iso3, year, anom."""
    out = []
    for (iso3, _), g in trade.sort_values("year").groupby(["iso3", "fbs_source"]):
        if g[col].notna().sum() < MIN_SEGMENT_YEARS:
            continue
        an = detrend(g[col].to_numpy(), g["year"].to_numpy())["anom"].to_numpy()
        out.append(pd.DataFrame({"iso3": iso3, "year": g["year"].to_numpy(), "anom": an}))
    return pd.concat(out, ignore_index=True).replace([np.inf, -np.inf], np.nan).dropna(subset=["anom"])


# ------------------------------------------------------------------ pipeline

def build_source(share_source, flows_df, trade, spei, included):
    """Rows and exposure for one share_source."""
    rows, expos = [], []
    n_years = flows_df.groupby("iso3")["year"].nunique()
    tot = flows_df.groupby(["iso3", "partner_iso3"])["import_t"].sum()
    s_dom = trade.groupby("iso3").apply(domestic_share, include_groups=False)
    for iso3 in sorted(included):
        if n_years.get(iso3, 0) < MIN_PARTNER_YEARS or not np.isfinite(s_dom.get(iso3, np.nan)):
            continue
        res = country_supply(iso3, s_dom[iso3], tot.loc[iso3], spei)
        if res is None:
            continue
        row, expo = res
        row.update(share_source=share_source, n_years=int(n_years[iso3]))
        rows.append(row)
        expos.append(expo.assign(iso3=iso3, share_source=share_source))
    div = pd.DataFrame(rows)[COLS]
    ex = pd.concat(expos, ignore_index=True)[["iso3", "share_source", "year", "partner_spei", "partner_dry_share"]]
    return div, ex.astype({"year": int})


def check_div(div):
    assert not div.duplicated(["iso3", "share_source"]).any(), "duplicate iso3-share_source"
    assert (div["supply_div_exp"] >= 1 - 1e-9).all() and (div["supply_div_exp"] <= div["supply_div"] + 1e-9).all()
    assert (div["s_dom"].between(0, 1)).all() and div["partner_corr"].between(-1 - 1e-9, 1 + 1e-9).all()
    assert (div["import_div_exp"] <= div["import_div"] + 1e-9).all() and (div["partner_cover"].between(0, 1)).all()
    assert div["partner_dry_own_dry"].dropna().between(0, 1).all() and div["partner_dry_other"].dropna().between(0, 1).all()
    assert (div["n_years"] >= MIN_PARTNER_YEARS).all() and (div["n_partners"] >= 1).all()


def outcomes(trade, iso3s):
    t = trade[trade["iso3"].isin(iso3s)]
    sup, prod = segment_anomalies(t, "supply_t"), segment_anomalies(t, "prod_t")
    out = pd.DataFrame({"iso3": sorted(iso3s)}).set_index("iso3")
    for name, a in [("stab_supply", sup), ("stab_prod", prod)]:
        g = a.groupby("iso3")["anom"]
        out[name] = (1.0 / g.std(ddof=1)).where(g.count() >= MIN_SEGMENT_YEARS)
    out["supply_buffer"] = out["stab_supply"] / out["stab_prod"]
    return out.reset_index()


def spearman_table(share_source, div, out):
    d = div.merge(out, on="iso3")
    rows = [{"share_source": share_source, "face": f, "predictor": p, **spearman_boot(d[p], d[f])} for f in FACES for p in PREDICTORS]
    return pd.DataFrame(rows)[["share_source", "face", "predictor", "rho", "ci_lo", "ci_hi", "p", "n"]]


def panel_models(share_source, div, expo, trade, climate):
    sup, imp = segment_anomalies(trade, "supply_t"), segment_anomalies(trade, "import_t")
    base = expo.merge(climate[["iso3", "year", "spei12_w"]], on=["iso3", "year"])
    base["own_dry"] = (base["spei12_w"] <= DROUGHT_THR).astype(float)
    base["own_dry_x_partner_dry_share"] = base["own_dry"] * base["partner_dry_share"]
    z = div.set_index("iso3")[["supply_div_exp", "s_dom"]]
    z = (z - z.mean()) / z.std(ddof=1)
    base["own_dry_x_z_supply_div_exp"] = base["own_dry"] * base["iso3"].map(z["supply_div_exp"])
    base["own_dry_x_z_s_dom"] = base["own_dry"] * base["iso3"].map(z["s_dom"])
    specs = {"imports": (imp, ["own_dry", "partner_dry_share", "own_dry_x_partner_dry_share"]),
             "supply": (sup, ["own_dry", "partner_dry_share", "own_dry_x_partner_dry_share"]),
             "supply_div": (sup, ["own_dry", "own_dry_x_z_supply_div_exp", "own_dry_x_z_s_dom"])}
    rows = []
    for model, (anom, terms) in specs.items():
        d = base.merge(anom, on=["iso3", "year"])[["iso3", "year", "anom"] + terms].replace([np.inf, -np.inf], np.nan).dropna()
        fit = panel.fe_fit(d, "anom", terms)
        for r in panel.coef_rows(fit):
            rows.append({"share_source": share_source, "model": model, **r, "n_obs": int(fit["nobs"]),
                         "n_countries": int(fit["n_clusters"]), "r2_within": fit["r2_within"]})
    return pd.DataFrame(rows)[PANEL_COLS]


def source_compare(div):
    wide = div.pivot(index="iso3", columns="share_source", values=COMPARE_VARS).dropna()
    rows = [{"variable": v, **spearman_boot(wide[(v, "reporter")], wide[(v, "mirror")])} for v in COMPARE_VARS]
    return pd.DataFrame(rows)[["variable", "rho", "ci_lo", "ci_hi", "p", "n"]]


def fig_supply(div, regions):
    d = div[div["share_source"] == "mirror"].merge(regions, on="iso3")
    colors = region_colors(d)
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    ax = axes[0]
    for region, g in d.groupby("region"):
        ax.scatter(g["supply_div"], g["supply_div_exp"], s=16, color=colors[region], edgecolor="white", linewidth=0.4, label=region)
    hi = max(d["supply_div"].max(), 1) * 1.05
    ax.plot([1, hi], [1, hi], color=MUTED, lw=0.8, ls="--")
    ax.set_xlabel("Effective number of supply sources\n(no similarity)")
    ax.set_ylabel("Effective number of supply sources\n(sources weighted by drought correlation)")
    ax.set_title("Supply sources with and without exposure similarity", fontsize=8, color=INK)
    lab = pd.concat([d.nlargest(3, "supply_div"), d.assign(gap=d["supply_div"] - d["supply_div_exp"]).nlargest(3, "gap")]).drop_duplicates("iso3")
    for _, r in lab.iterrows():
        ax.annotate(r["iso3"], (r["supply_div"], r["supply_div_exp"]), fontsize=6, color=INK, xytext=(2, 2), textcoords="offset points")
    ax.legend(fontsize=5.5, frameon=False, loc="upper left")
    ax = axes[1]
    e = d.dropna(subset=["partner_dry_other", "partner_dry_own_dry"])
    for region, g in e.groupby("region"):
        ax.scatter(g["partner_dry_other"], g["partner_dry_own_dry"], s=16, color=colors[region], edgecolor="white", linewidth=0.4)
    m = max(e["partner_dry_other"].max(), e["partner_dry_own_dry"].max()) * 1.05
    ax.plot([0, m], [0, m], color=MUTED, lw=0.8, ls="--")
    n_above = int((e["partner_dry_own_dry"] > e["partner_dry_other"]).sum())
    ax.set_xlabel("Import-weighted share of suppliers in drought\nin the country's other years (share)")
    ax.set_ylabel("Import-weighted share of suppliers in drought\nin the country's own drought years (share)")
    ax.set_title(f"Suppliers in drought: own drought years vs other years ({n_above} of {len(e)} above 1:1)", fontsize=8, color=INK)
    fig.text(0.01, 0.01, SRC_LINE.format(n=len(d)), fontsize=5.5, color="#555555")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(paths.fig_dir("data") / FIG_NAME, dpi=DPI)
    plt.close(fig)


def main():
    quality = load("quality")
    included = set(quality.loc[quality["included"], "iso3"])
    trade = pd.read_parquet(paths.TRADE)
    climate = load("climate")
    spei = complete_spei(climate)
    regions = pd.read_csv(paths.COUNTRIES, dtype=str, keep_default_na=False)[["iso3", "region"]]
    divs, expos, sp, pn = [], [], [], []
    for name, path in SOURCES.items():
        flows = pd.read_parquet(path)
        div, ex = build_source(name, flows, trade, spei, included)
        check_div(div)
        assert (ex.groupby("iso3").size() == len(YEARS)).all() and ex["partner_dry_share"].between(0, 1).all()
        sp.append(spearman_table(name, div, outcomes(trade, set(div["iso3"]))))
        pn.append(panel_models(name, div, ex, trade, climate))
        divs.append(div)
        expos.append(ex)
    div, ex = pd.concat(divs, ignore_index=True), pd.concat(expos, ignore_index=True)
    check_div(div)
    assert not ex.duplicated(["iso3", "share_source", "year"]).any()
    cmp_ = source_compare(div)
    sp, pn = pd.concat(sp, ignore_index=True), pd.concat(pn, ignore_index=True)
    assert pn["p"].between(0, 1).all() and np.isfinite(pn["beta"]).all() and sp["rho"].dropna().between(-1, 1).all()

    paths.OUT.mkdir(parents=True, exist_ok=True)
    paths.STATS_DIR.mkdir(parents=True, exist_ok=True)
    div.to_parquet(OUT_DIV, index=False)
    ex.to_parquet(OUT_EXPO, index=False)
    sp.to_csv(paths.STATS_DIR / "supply_spearman.csv", index=False)
    pn.to_csv(paths.STATS_DIR / "supply_panel.csv", index=False)
    cmp_.to_csv(paths.STATS_DIR / "supply_source_compare.csv", index=False)
    fig_supply(div, regions)
    assert (paths.fig_dir("data") / FIG_NAME).stat().st_size > 20_000

    for name, d in div.groupby("share_source"):
        above = int((d["partner_dry_own_dry"] > d["partner_dry_other"]).sum())
        both = int(d[["partner_dry_own_dry", "partner_dry_other"]].notna().all(axis=1).sum())
        print(f"supply_diversity [{name}]: countries={len(d)}  missing={100 * d.isna().to_numpy().mean():.1f}%  "
              f"median supply_div={d['supply_div'].median():.3f} supply_div_exp={d['supply_div_exp'].median():.3f} "
              f"supply_ratio={d['supply_ratio'].median():.3f} partner_corr={d['partner_corr'].median():.3f} "
              f"s_dom={d['s_dom'].median():.3f} partner_cover={d['partner_cover'].median():.3f}")
        print(f"   partner_dry_own_dry median {d['partner_dry_own_dry'].median():.3f} vs partner_dry_other median "
              f"{d['partner_dry_other'].median():.3f}; above the 1:1 line in {above} of {both} countries")
    print(f"supply_exposure: rows={len(ex)}  countries={ex['iso3'].nunique()}  years={ex['year'].min()}-{ex['year'].max()}")
    print("\nSpearman (stats/supply_spearman.csv):")
    print(sp.round(3).to_string(index=False))
    print("\nPanel (stats/supply_panel.csv):")
    print(pn.round(4).to_string(index=False))
    print("\nReporter vs mirror (stats/supply_source_compare.csv):")
    print(cmp_.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
