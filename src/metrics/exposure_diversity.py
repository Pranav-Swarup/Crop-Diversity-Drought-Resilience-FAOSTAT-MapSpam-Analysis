"""Exposure-weighted crop diversity: Leinster-Cobbold diversity of exposure classes, with similarity = correlation
of the classes' drought exposure. Crops add diversity only to the extent that they face different droughts.

Definitions (per included country, window 1993-2023):
  exposure class   one MapSPAM 2020 class or a merged class (src.metrics.crop_classes)
  p_c              share of class c in the mean harvested area: FAOSTAT area_ha summed over the class's items and
                   years / 31, rows with a value_const only (same averaging and row filter as hill1 in run.py)
  e_c(t)           exposure series of class c = crop_spei12_w of its MapSPAM class (December SPEI-12 averaged over
                   the class's MapSPAM 2020 physical area in the country). Merged class: weighted mean of the member
                   series, weights = MapSPAM physical area of each member in the country (mapspam_area.parquet), or
                   equal weights if that file is missing. Members without a series in the country are left out.
  fallback         a class with FAOSTAT area but no member series in the country takes the national spei12_w;
                   fallback_share = share of p on such classes
  R                Pearson correlation of the e_c over 1993-2023 (pairwise complete, >= 25 common years)
  hill1_ms         lc_diversity(p, Z = I, q = 1): the Hill number at exposure-class resolution
  hill1_exp        lc_diversity(p, Z = similarity_from_corr(R), q = 1)   (negative correlations -> 0)
  exp_ratio        hill1_exp / hill1_ms
  exp_corr         mean_offdiag(p, R): p_i p_j weighted mean of the off-diagonal r_ij (unclipped)
  hill2_ms, hill2_exp   the same at q = 2
  sd_spei          SD (ddof 1) of the national spei12_w over the window
  n_drought_years  number of window years with national spei12_w <= -1.0
  log_area         log of the mean total harvested area (ha) = log(sum of the class mean areas)

Variants (each computed only if its input file exists; else the columns are NaN and a "skipped" line is printed):
  hill1_exp_pre, exp_corr_pre   R from crop_climate_pre.parquet, 1961-1992 (no in-window climate). Fallback
                   classes take a pre-window national series rebuilt as the MapSPAM-area-weighted mean of all class
                   series of the country (needs mapspam_area.parquet; the same rebuild on 1993-2023 is compared with
                   climate.parquet spei12_w and printed).
  hill1_exp_season, exp_corr_season   R from crop_spei6_harv_w (SPEI-6 at the harvest month, crop_climate_season)
                   for member classes that have it, December crop_spei12_w for the others; season_share = share of
                   p carried by member series with a calendar (merged classes split by member weight)
  rf_share         rainfed share of the country's MapSPAM 2020 physical area (sum rf_ha / sum area_ha)
  hill1_exp_main, exp_corr_main, overseas_share   robustness for detached territories: classes with more than
                   half of their in-country MapSPAM area in detached parts of the country geometry are removed
                   (p renormalised). Detached part: a polygon of data/clean/countries.geojson not linked to the
                   largest polygon by a chain of polygons each within LINK_KM of the next (great-circle distance
                   between nearest points). overseas_share = p share of the removed classes.

Outputs:
  outputs/exposure_diversity.parquet   one row per included country (columns above)
  outputs/stats/expdiv_spearman.csv    face, predictor, rho, ci_lo, ci_hi, p, n   (spearman_boot)
  outputs/stats/expdiv_ols.csv         outcome, model, term, beta, se, p, n, r2, adj_r2   (stats.ols_z: z-scored
                                       outcome and terms, HC3; stability enters untransformed, as in stats.py)
  outputs/stats/expdiv_panel.csv       model, term, beta, se, ci_lo, ci_hi, p, n_obs, n_countries, r2_within
                                       (panel.fit_model: anom on spei and spei x z(country var), country and year
                                       fixed effects, SE clustered by country)
  figures/results/11_exposure_diversity.png

Two channels are tested separately. (i) exposure: outcome sd_spei or n_drought_years on exp_corr (+ log_area),
in expdiv_spearman / expdiv_ols. (ii) sensitivity: spei x z(exp_corr) in the panel, given the national spei.

Run: python -m src.metrics.exposure_diversity      (real data only)
"""
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401  (registers the .rio accessor)
import shapely
import xarray as xr
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ..common import paths
from ..common.contract import YEAR_MAX, YEAR_MIN, load
from . import crop_classes, panel
from .data import CONTROLS, analysis_table
from .events import DROUGHT_THR
from .figures import DPI, FIGSIZE, INK, MUTED, region_colors
from .similarity import lc_diversity, mean_offdiag, similarity_from_corr
from .stats import ols_z, spearman_boot

YEARS = np.arange(YEAR_MIN, YEAR_MAX + 1)
PRE_YEARS = np.arange(1961, 1993)
MIN_COMMON = 25
LINK_KM = 1000.0
OVERSEAS_MIN = 0.5
LOWCORR = 0.2
N_INCLUDED = 107

OUT_FILE = paths.OUT / "exposure_diversity.parquet"
MAPSPAM_AREA = paths.CLEAN / "mapspam_area.parquet"
CROP_CLIMATE_PRE = paths.CLEAN / "crop_climate_pre.parquet"
CROP_CLIMATE_SEASON = paths.CLEAN / "crop_climate_season.parquet"
CROPLAND_NPZ = paths.RAW / "derived" / "cropland_05deg.npz"
SPEARMAN_FILE, OLS_FILE, PANEL_FILE = "expdiv_spearman.csv", "expdiv_ols.csv", "expdiv_panel.csv"
FIG_NAME = "11_exposure_diversity.png"

OUTCOMES = ["stability", "resistance", "recovery", "vulnerability", "vuln_drought", "sd_spei", "n_drought_years"]
BASE_PREDICTORS = ["hill1", "hill1_ms", "hill1_exp", "exp_ratio", "exp_corr", "log_area"]
VARIANT_PREDICTORS = ["hill1_exp_pre", "exp_corr_pre", "hill1_exp_season", "exp_corr_season", "rf_share",
                      "hill1_exp_main", "exp_corr_main"]
COLUMNS = ["iso3", "n_classes", "hill1_ms", "hill1_exp", "exp_ratio", "exp_corr", "hill2_ms", "hill2_exp",
           "fallback_share", "rest_share", "sd_spei", "n_drought_years", "log_area",
           "hill1_exp_pre", "exp_corr_pre", "hill1_exp_season", "exp_corr_season", "season_share", "rf_share",
           "hill1_exp_main", "exp_corr_main", "overseas_share"]
PANEL_COLS = ["model", "term", "beta", "se", "ci_lo", "ci_hi", "p", "n_obs", "n_countries", "r2_within"]


# ---------------------------------------------------------------- classes and series

def class_areas(g):
    """Mean harvested area (ha) per exposure class label for one country's crops rows (rows with value_const)."""
    g = g[g["value_const"].notna()]
    lab = g["item_code"].map(lambda c: crop_classes.class_label(crop_classes.ITEMS[c][1]))
    a = g.groupby(lab)["area_ha"].sum() / len(YEARS)
    return a[a > 0].sort_index()


def members_of(label):
    return tuple(label.split("+"))


def class_series(label, wide, national, weights=None):
    """Exposure series of one class. wide: DataFrame year x mapspam_crop for the country; national: Series year ->
    national spei; weights: dict mapspam_crop -> MapSPAM area (None = equal weights).
    Returns (Series, fallback flag, dict member -> weight share used)."""
    have = [m for m in members_of(label) if m in wide.columns]
    if not have:
        return national.rename(label), True, {}
    w = np.array([1.0 if weights is None else float(weights.get(m, 0.0)) for m in have])
    if w.sum() <= 0:                       # members present with no recorded area: fall back to equal weights
        w = np.ones(len(have))
    w = w / w.sum()
    s = (wide[have] * w).sum(axis=1, min_count=len(have))
    return s.rename(label), False, dict(zip(have, w))


def corr_matrix(series):
    """Pearson R of a dict label -> Series (pairwise complete). Asserts >= MIN_COMMON common years and finite R."""
    df = pd.concat(series.values(), axis=1)
    ok = df.notna().astype(int)
    common = ok.T @ ok
    assert (common.to_numpy() >= MIN_COMMON).all(), f"fewer than {MIN_COMMON} common years: {common.min().min()}"
    r = df.corr(min_periods=MIN_COMMON)
    assert np.isfinite(r.to_numpy()).all(), "undefined exposure correlation (constant series?)"
    return r


def diversity(p, r):
    """hill1_exp, exp_corr, hill2_exp for shares p (Series) and correlation r (aligned DataFrame)."""
    r = r.loc[p.index, p.index].to_numpy()
    z = similarity_from_corr(r)
    return lc_diversity(p.to_numpy(), z, 1.0), mean_offdiag(p.to_numpy(), r), lc_diversity(p.to_numpy(), z, 2.0)


def pivot(df, value, iso3):
    d = df[df["iso3"] == iso3]
    return d.pivot(index="year", columns="mapspam_crop", values=value)


# ---------------------------------------------------------------- geometry (detached territories)

def _haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.deg2rad, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def _pair_km(geoms, i, j):
    """Great-circle distance (km) between the nearest points of polygons geoms[i], geoms[j] (lon/lat), taking the
    shorter of the direct and the +-360 degree shifted comparison (polygons split at the antimeridian)."""
    a, b = geoms[i], geoms[j]
    best = np.full(len(i), np.inf)
    for shift in (0.0, 360.0, -360.0):
        bs = shapely.transform(b, lambda c, s=shift: c + np.array([s, 0.0]))
        line = shapely.shortest_line(a, bs)
        c = shapely.get_coordinates(line).reshape(-1, 2, 2)
        best = np.minimum(best, _haversine_km(c[:, 0, 0], c[:, 0, 1], c[:, 1, 0] - shift, c[:, 1, 1]))
    return best


def detached_parts(world, iso3s, link_km=LINK_KM):
    """GeoDataFrame of polygons not chained to the largest polygon of their country (single linkage at link_km).
    Columns iso3, geometry, area_km2, km_to_main (distance to the nearest polygon of the main cluster)."""
    rows = []
    for iso3, g in world[world["iso3"].isin(iso3s)].groupby("iso3"):
        parts = g.explode(index_parts=False).reset_index(drop=True)
        n = len(parts)
        if n < 2:
            continue
        area = (parts.to_crs("EPSG:6933").area / 1e6).to_numpy()
        geoms = parts.geometry.to_numpy()
        i, j = np.triu_indices(n, 1)
        d = _pair_km(geoms, i, j)
        link = d <= link_km
        adj = coo_matrix((np.ones(link.sum()), (i[link], j[link])), shape=(n, n))
        _, comp = connected_components(adj, directed=False)
        main = comp == comp[int(area.argmax())]
        if main.all():
            continue
        dist = np.full((n, n), np.inf)
        dist[i, j] = d
        dist[j, i] = d
        for k in np.flatnonzero(~main):
            rows.append({"iso3": iso3, "geometry": geoms[k], "area_km2": float(area[k]),
                         "km_to_main": float(dist[k, main].min())})
    return gpd.GeoDataFrame(rows, columns=["iso3", "geometry", "area_km2", "km_to_main"], crs=world.crs)


def overseas_crop_share(world, iso3s):
    """Share of each country's MapSPAM physical area per class located in detached parts (same 0.5 degree cell x
    coverage-fraction weights as src.climate). Returns (DataFrame iso3, mapspam_crop, overseas_frac, detached parts)
    or (None, parts) if the cropland raster is missing."""
    parts = detached_parts(world, iso3s)
    if not CROPLAND_NPZ.exists():
        return None, parts
    from ..climate.aggregate import coverage_fractions
    lat = np.arange(89.75, -90, -0.5)
    lon = np.arange(-179.75, 180, 0.5)
    template = xr.DataArray(np.zeros((360, 720)), coords={"lat": lat, "lon": lon}, dims=("lat", "lon")) \
        .rio.write_crs(4326).rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    countries = world[world["iso3"].isin(parts["iso3"].unique())][["iso3", "geometry"]]
    det = parts.dissolve("iso3").reset_index()[["iso3", "geometry"]]
    cov_all = coverage_fractions(countries.reset_index(drop=True), template)
    cov_det = coverage_fractions(det, template)
    npz = np.load(CROPLAND_NPZ)
    rows = []
    for iso3 in det["iso3"]:
        ca, fa = cov_all[iso3]
        cd, fd = cov_det.get(iso3, (np.array([], int), np.array([])))
        for crop in npz.files:
            flat = npz[crop].ravel()
            tot = (flat[ca] * fa).sum()
            if tot > 0:
                rows.append({"iso3": iso3, "mapspam_crop": crop, "overseas_frac": float((flat[cd] * fd).sum() / tot)})
    return pd.DataFrame(rows), parts


# ---------------------------------------------------------------- per-country table

def compute(crops, climate, crop_climate, included, mapspam_area=None, pre=None, season=None, overseas=None):
    """One row per country in `included`. Optional inputs may be None. Also returns per-class detail rows."""
    nat_by = {k: v.set_index("year")["spei12_w"].reindex(YEARS) for k, v in climate.groupby("iso3")}
    area_by = ({k: v.set_index("mapspam_crop") for k, v in mapspam_area.groupby("iso3")}
               if mapspam_area is not None else {})
    ov_by = ({k: v.set_index("mapspam_crop")["overseas_frac"] for k, v in overseas.groupby("iso3")}
             if overseas is not None else {})
    crops = crops[crops["iso3"].isin(included)]
    rows, detail = [], []
    for iso3, g in crops.groupby("iso3", sort=True):
        a = class_areas(g)
        p = a / a.sum()
        national = nat_by[iso3]
        wide = pivot(crop_climate, "crop_spei12_w", iso3).reindex(YEARS)
        weights = area_by[iso3]["area_ha"].to_dict() if iso3 in area_by else None
        ser, fb, wts = {}, {}, {}
        for lab in p.index:
            ser[lab], fb[lab], wts[lab] = class_series(lab, wide, national, weights)
        r = corr_matrix(ser)
        h1e, ec, h2e = diversity(p, r)
        row = {"iso3": iso3, "n_classes": int(len(p)),
               "hill1_ms": lc_diversity(p.to_numpy()), "hill1_exp": h1e, "exp_corr": ec,
               "hill2_ms": lc_diversity(p.to_numpy(), None, 2.0), "hill2_exp": h2e,
               "fallback_share": float(p[[k for k in p.index if fb[k]]].sum()),
               "rest_share": float(p.get("REST", 0.0)),
               "sd_spei": float(national.std(ddof=1)),
               "n_drought_years": int((national <= DROUGHT_THR).sum()),
               "log_area": float(np.log(a.sum()))}
        row["exp_ratio"] = row["hill1_exp"] / row["hill1_ms"]

        # pre-window correlation
        row["hill1_exp_pre"] = row["exp_corr_pre"] = np.nan
        if pre is not None:
            wpre = pivot(pre, "crop_spei12_w", iso3).reindex(PRE_YEARS)
            nat_pre = None
            if weights is not None:
                aw = pd.Series(weights).reindex(wpre.columns).fillna(0.0)
                nat_pre = (wpre * aw).sum(axis=1, min_count=1) / aw.sum()
            if nat_pre is not None or not any(fb.values()):
                sp = {lab: class_series(lab, wpre, nat_pre, weights)[0] for lab in p.index}
                row["hill1_exp_pre"], row["exp_corr_pre"], _ = diversity(p, corr_matrix(sp))

        # seasonal exposure
        row["hill1_exp_season"] = row["exp_corr_season"] = row["season_share"] = np.nan
        if season is not None:
            ws = pivot(season, "crop_spei6_harv_w", iso3).reindex(YEARS)
            mix = wide.copy()
            seasonal = [c for c in ws.columns if c in mix.columns and ws[c].notna().sum() >= MIN_COMMON]
            mix[seasonal] = ws[seasonal]
            ss = {lab: class_series(lab, mix, national, weights)[0] for lab in p.index}
            row["hill1_exp_season"], row["exp_corr_season"], _ = diversity(p, corr_matrix(ss))
            row["season_share"] = float(sum(p[lab] * sum(w for m, w in wts[lab].items() if m in seasonal)
                                            for lab in p.index))

        # rainfed share of MapSPAM physical area
        row["rf_share"] = (float(area_by[iso3]["rf_ha"].sum() / area_by[iso3]["area_ha"].sum())
                           if iso3 in area_by else np.nan)

        # robustness: drop classes located mainly in detached territories
        row["hill1_exp_main"], row["exp_corr_main"], row["overseas_share"] = row["hill1_exp"], row["exp_corr"], 0.0
        if overseas is None:
            row["hill1_exp_main"] = row["exp_corr_main"] = row["overseas_share"] = np.nan
        elif iso3 in ov_by:
            of = ov_by[iso3]
            frac = {lab: sum(w * of.get(m, 0.0) for m, w in wts[lab].items()) for lab in p.index}
            keep = [lab for lab in p.index if frac[lab] <= OVERSEAS_MIN]
            row["overseas_share"] = float(p.drop(keep).sum())
            pk = p[keep] / p[keep].sum()
            row["hill1_exp_main"], row["exp_corr_main"], _ = diversity(pk, r)

        rows.append(row)
        for lab in p.index:
            detail.append({"iso3": iso3, "exposure_class": lab, "p": float(p[lab]), "area_ha": float(a[lab]),
                           "fallback": fb[lab], "r_national": float(ser[lab].corr(national)),
                           "mapspam_ha": (sum(weights.get(m, 0.0) for m in members_of(lab))
                                          if weights is not None else np.nan)})
    out = pd.DataFrame(rows)[COLUMNS]
    return out.astype({"n_classes": int, "n_drought_years": int}), pd.DataFrame(detail)


# ---------------------------------------------------------------- statistics

def spearman_table(df, predictors):
    rows = [{"face": f, "predictor": x, **spearman_boot(df[x], df[f])} for f in OUTCOMES for x in predictors]
    return pd.DataFrame(rows)[["face", "predictor", "rho", "ci_lo", "ci_hi", "p", "n"]]


def ols_models(df):
    """dict model -> terms. a-f as specified; variant versions of c and d for every variant column present."""
    m = {"a": ["hill1_ms"], "b": ["hill1_ms", "log_area"], "c": ["hill1_ms", "exp_corr"],
         "d": ["hill1_ms", "exp_corr", "log_area"], "e": ["hill1_exp"],
         "f": ["hill1_ms", "exp_corr", "log_area"] + CONTROLS}
    for v in ("pre", "season", "main"):
        col = f"exp_corr_{v}"
        if df[col].notna().any():
            m[f"c_{v}"] = ["hill1_ms", col]
            m[f"d_{v}"] = ["hill1_ms", col, "log_area"]
    return m


def ols_table(df):
    rows = []
    for outcome in OUTCOMES:
        for model, terms in ols_models(df).items():
            fit = ols_z(df, outcome, terms)
            n = int(df[[outcome] + terms].replace([np.inf, -np.inf], np.nan).dropna().shape[0])
            for t in terms:
                if fit is None:
                    rows.append({"outcome": outcome, "model": model, "term": t, "beta": np.nan, "se": np.nan,
                                 "p": np.nan, "n": n, "r2": np.nan, "adj_r2": np.nan})
                    continue
                rows.append({"outcome": outcome, "model": model, "term": t, "beta": fit.params[t], "se": fit.bse[t],
                             "p": fit.pvalues[t], "n": int(fit.nobs), "r2": fit.rsquared, "adj_r2": fit.rsquared_adj})
    return pd.DataFrame(rows)


def panel_specs(df):
    s = {"base": ["hill1"], "exp_corr": ["hill1", "exp_corr"], "size": ["hill1", "exp_corr", "log_area"],
         "hill1_exp": ["hill1_exp"]}
    for v in ("pre", "season", "main"):
        col = f"exp_corr_{v}"
        if df[col].notna().any():
            s[f"exp_corr_{v}"] = ["hill1", col]
            s[f"size_{v}"] = ["hill1", col, "log_area"]
    return s


def panel_table(ed):
    pf = panel.load_panel().merge(ed.drop(columns=["n_drought_years"]), on="iso3", how="left")
    rows = []
    for name, inter in panel_specs(ed).items():
        res = panel.fit_model(pf, ["spei"], inter, (), ())
        assert res is not None, f"panel {name}: too few countries"
        fit = res[0]
        meta = {"n_obs": fit["nobs"], "n_countries": fit["n_clusters"], "r2_within": fit["r2_within"]}
        rows += [{"model": name, **r, **meta} for r in panel.coef_rows(fit)]
    return pd.DataFrame(rows)[PANEL_COLS], pf


# ---------------------------------------------------------------- figure

def fig_expdiv(df, sp):
    colors = region_colors(df)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE)
    for region, g in df.groupby("region"):
        ax1.scatter(g["hill1_ms"], g["hill1_exp"], s=16, color=colors[region], edgecolor="white", linewidth=0.4)
        ax2.scatter(g["exp_corr"], g["stability"], s=16, color=colors[region], edgecolor="white", linewidth=0.4)
    lim = (1, df["hill1_ms"].max() * 1.05)
    ax1.plot(lim, lim, color=MUTED, linewidth=0.8, linestyle="--", label="1:1 line")
    lab = pd.concat([df.nlargest(4, "hill1_exp"), df.nlargest(2, "hill1_ms")]).drop_duplicates("iso3") \
        .sort_values("hill1_ms")
    for k, r in enumerate(lab.itertuples()):
        ax1.annotate(r.iso3, (r.hill1_ms, r.hill1_exp), fontsize=6, color=INK, xytext=(0, 8 + 9 * (k % 3)),
                     textcoords="offset points", ha="center",
                     arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.4})
    ax1.set_xlabel("Crop diversity at exposure-class resolution, hill1_ms\n(effective number of classes)")
    ax1.set_ylabel("Exposure-weighted diversity, hill1_exp\n(effective number of distinct exposures)")
    ax1.set_title("Diversity with and without exposure similarity", fontsize=7.5, color=INK, loc="left")
    ax1.legend(frameon=False, fontsize=6.5, loc="upper left")

    s = sp[(sp["face"] == "stability") & (sp["predictor"] == "exp_corr")].iloc[0]
    ax2.set_yscale("log")
    ax2.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax2.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax2.set_yticks([t for t in (2, 5, 10, 20, 50, 100, 200)
                    if df["stability"].min() / 1.5 <= t <= df["stability"].max() * 1.5])
    ax2.set_xlabel("Mean exposure correlation between crop classes,\nexp_corr (Pearson r, unitless)")
    ax2.set_ylabel("Stability: 1 / CV of national value (log scale)")
    ax2.set_title(f"Stability vs exposure correlation\nSpearman ρ = {s.rho:+.2f} [{s.ci_lo:+.2f}, {s.ci_hi:+.2f}], "
                  f"n = {s.n}", fontsize=7.5, color=INK, loc="left")

    handles = [plt.Line2D([], [], marker="o", ls="", color=c, markersize=4, label=r) for r, c in colors.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 0.035))
    fig.suptitle("Exposure-weighted crop diversity (Leinster-Cobbold, similarity = correlation of class SPEI-12)",
                 fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.11, 1, 0.95))
    fig.text(0.01, 0.01, f"Source: FAOSTAT (QCL, QV), SPEIbase v2.10, MapSPAM 2020. 1993–2023, n = {len(df)} countries. "
             "Class exposure = December SPEI-12 over each MapSPAM class's area.",
             fontsize=6, color=MUTED, ha="left", va="bottom")
    out = paths.fig_dir("results") / FIG_NAME
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out.relative_to(paths.ROOT)}")


# ---------------------------------------------------------------- main

def optional(path):
    if path.exists():
        print(f"using {path.relative_to(paths.ROOT)}")
        return pd.read_parquet(path)
    print(f"skipped: {path.relative_to(paths.ROOT)} not found")
    return None


def main():
    assert not paths.USE_DUMMY, "exposure_diversity has no dummy data: run without USE_DUMMY"
    crops, climate, quality = load("crops"), load("climate"), load("quality")
    crop_climate = pd.read_parquet(paths.CROP_CLIMATE)
    included = sorted(quality.loc[quality["included"], "iso3"])
    crop_classes.check(crops, crop_climate["mapspam_crop"].unique())
    mapspam_area, pre, season = optional(MAPSPAM_AREA), optional(CROP_CLIMATE_PRE), optional(CROP_CLIMATE_SEASON)
    if season is not None:
        assert {"iso3", "year", "mapspam_crop", "crop_spei6_harv_w"} <= set(season.columns), season.columns
    if pre is not None:
        assert pre["year"].between(PRE_YEARS[0], PRE_YEARS[-1]).all(), "crop_climate_pre has years outside 1961-1992"
    world = gpd.read_file(paths.COUNTRIES_GEOJSON)
    overseas, parts = overseas_crop_share(world, included)
    if overseas is None:
        print(f"skipped: {CROPLAND_NPZ.relative_to(paths.ROOT)} not found (no overseas robustness columns)")

    ed, detail = compute(crops, climate, crop_climate, included, mapspam_area, pre, season, overseas)

    # assertions on the country table
    assert not ed["iso3"].duplicated().any()
    assert set(ed["iso3"]) == set(included) and len(ed) == N_INCLUDED, f"{len(ed)} countries"
    assert (ed["hill1_exp"] >= 1 - 1e-9).all() and (ed["hill1_exp"] <= ed["hill1_ms"] + 1e-9).all()
    assert (ed["exp_ratio"] > 0).all() and (ed["exp_ratio"] <= 1 + 1e-9).all()
    assert ed["exp_corr"].between(-1, 1).all()
    assert (ed["hill2_exp"] <= ed["hill2_ms"] + 1e-9).all() and (ed["hill2_exp"] >= 1 - 1e-9).all()
    assert ed["fallback_share"].between(0, 1).all() and ed["rest_share"].between(0, 1).all()
    for v in ("pre", "season", "main"):
        h = ed[f"hill1_exp_{v}"].dropna()
        assert (h >= 1 - 1e-9).all() and ed[f"exp_corr_{v}"].dropna().between(-1, 1).all()
    assert ed["overseas_share"].dropna().between(0, 1).all()
    assert ed["rf_share"].dropna().between(0, 1 + 1e-9).all()
    for c in ["hill1_ms", "hill1_exp", "exp_corr", "sd_spei", "log_area"]:
        assert ed[c].notna().all(), f"{c} has missing values"

    metrics = load("metrics")
    m = ed.merge(metrics[["iso3", "hill1"]], on="iso3")
    rho_h = spearman_boot(m["hill1"], m["hill1_ms"])
    assert rho_h["rho"] > 0.8, f"hill1_ms vs contract hill1 Spearman {rho_h['rho']:.3f}"
    assert (m["hill1_ms"] <= m["hill1"] + 1e-9).all(), "merging items into classes cannot raise the Hill number"

    df = analysis_table().merge(ed, on="iso3", how="inner")
    assert len(df) == N_INCLUDED
    predictors = BASE_PREDICTORS + [c for c in VARIANT_PREDICTORS if df[c].notna().any()]
    sp = spearman_table(df, predictors)
    assert not sp.duplicated(["face", "predictor"]).any() and sp["rho"].dropna().between(-1, 1).all()
    ols = ols_table(df)
    assert not ols.duplicated(["outcome", "model", "term"]).any()
    assert ols["r2"].dropna().between(0, 1).all()
    coef, pf = panel_table(ed)
    assert not coef.duplicated(["model", "term"]).any()
    assert np.isfinite(coef[["beta", "se"]]).all().all() and (coef["se"] > 0).all()
    assert (coef["ci_lo"] <= coef["beta"]).all() and (coef["beta"] <= coef["ci_hi"]).all()
    assert coef["p"].between(0, 1).all() and coef["r2_within"].between(0, 1).all()

    paths.STATS_DIR.mkdir(parents=True, exist_ok=True)
    ed.to_parquet(OUT_FILE, index=False)
    sp.to_csv(paths.STATS_DIR / SPEARMAN_FILE, index=False)
    ols.to_csv(paths.STATS_DIR / OLS_FILE, index=False)
    coef.to_csv(paths.STATS_DIR / PANEL_FILE, index=False)
    print(f"wrote {OUT_FILE.relative_to(paths.ROOT)}, outputs/stats/{SPEARMAN_FILE}, {OLS_FILE}, {PANEL_FILE}")

    # ------------------------------------------------ summary
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 400)
    print(f"\nexposure_diversity: {len(ed)} rows, {ed['iso3'].nunique()} countries, window {YEAR_MIN}-{YEAR_MAX}; "
          f"% missing: " + ", ".join(f"{c}={100 * ed[c].isna().mean():.0f}" for c in ed.columns if ed[c].isna().any()))
    print(f"merged-class weights: {'MapSPAM physical area (mapspam_area.parquet)' if mapspam_area is not None else 'EQUAL (mapspam_area.parquet not found)'}")
    print("medians: " + ", ".join(f"{c}={ed[c].median():.3f}" for c in
                                  ["n_classes", "hill1_ms", "hill1_exp", "exp_ratio", "exp_corr", "hill2_ms", "hill2_exp",
                                   "fallback_share", "rest_share", "sd_spei", "n_drought_years"]
                                  + [c for c in VARIANT_PREDICTORS + ["season_share"] if ed[c].notna().any()]))
    print("Spearman between country variables: " + ", ".join(
        f"{a}~{b}={ed[a].corr(ed[b], method='spearman'):+.2f}" for a, b in
        [("exp_corr", "hill1_ms"), ("exp_corr", "log_area"), ("hill1_ms", "log_area"), ("exp_corr", "exp_corr_pre"),
         ("exp_corr", "exp_corr_season")] if ed[b].notna().any()))
    print(f"hill1_ms vs contract hill1: Spearman rho = {rho_h['rho']:.3f} [{rho_h['ci_lo']:.3f}, {rho_h['ci_hi']:.3f}], "
          f"n = {rho_h['n']}; median hill1 = {m['hill1'].median():.3f}, median hill1_ms = {m['hill1_ms'].median():.3f}")
    print(f"countries with fallback_share > 0: {(ed['fallback_share'] > 0).sum()}; max fallback_share "
          f"{ed['fallback_share'].max():.3f}")

    if pre is not None and mapspam_area is not None:
        aw = mapspam_area.set_index(["iso3", "mapspam_crop"])["area_ha"]
        cc = crop_climate.join(aw, on=["iso3", "mapspam_crop"])
        reb = (cc["crop_spei12_w"] * cc["area_ha"]).groupby([cc["iso3"], cc["year"]]).sum() \
            / cc.groupby(["iso3", "year"])["area_ha"].sum()
        cmp = climate.set_index(["iso3", "year"])["spei12_w"].to_frame().join(reb.rename("rebuilt"), how="inner")
        cmp = cmp[cmp.index.get_level_values(0).isin(included)]
        print(f"pre-window national fallback check (1993-2023, included countries): rebuilt vs climate.parquet "
              f"spei12_w max |diff| = {(cmp['spei12_w'] - cmp['rebuilt']).abs().max():.2e}, "
              f"corr = {cmp.corr().iloc[0, 1]:.6f}")

    print("\nshare of mean harvested area on REST and on the national fallback (countries with either > 5%):")
    t = ed[(ed["rest_share"] > 0.05) | (ed["fallback_share"] > 0.05)][["iso3", "rest_share", "fallback_share"]]
    print(t.round(3).to_string(index=False) if len(t) else "  none")
    fbc = detail[detail["fallback"]]
    if len(fbc):
        print("fallback classes (iso3, class, p): " + "; ".join(f"{r.iso3} {r.exposure_class} {r.p:.3f}"
                                                              for r in fbc.itertuples()))

    print("\nlowest exp_corr: ")
    print(ed.nsmallest(10, "exp_corr")[["iso3", "n_classes", "hill1_ms", "hill1_exp", "exp_ratio", "exp_corr",
                                         "exp_corr_main", "overseas_share"]].round(3).to_string(index=False))

    print(f"\ndetached territories (polygons > {LINK_KM:.0f} km chain distance from the main part), included countries:")
    if len(parts):
        pa = parts.groupby("iso3").agg(n_parts=("area_km2", "size"), area_km2=("area_km2", "sum"),
                                       max_km=("km_to_main", "max")).reset_index()
        print(pa.round(0).to_string(index=False))
    if overseas is not None:
        ov = overseas[overseas["overseas_frac"] > OVERSEAS_MIN]
        print(f"MapSPAM classes with > {OVERSEAS_MIN:.0%} of their in-country area in detached parts: "
              + "; ".join(f"{i}: {', '.join(g['mapspam_crop'])}" for i, g in ov.groupby("iso3")))
        t = ed[ed["overseas_share"] > 0][["iso3", "overseas_share", "exp_corr", "exp_corr_main", "hill1_exp",
                                          "hill1_exp_main"]]
        print("countries where the robustness drops classes:")
        print(t.round(3).to_string(index=False) if len(t) else "  none")
    low = detail[detail["r_national"] < LOWCORR]
    print(f"\nclasses whose exposure correlates < {LOWCORR} with the national spei12_w (iso3: class p r):")
    for iso3, g in low.groupby("iso3"):
        print(f"  {iso3}: " + ", ".join(f"{r.exposure_class} {r.p:.3f} {r.r_national:+.2f}" for r in g.itertuples()))
    for iso3 in ("FRA", "NLD", "USA"):
        d = detail[detail["iso3"] == iso3].sort_values("p", ascending=False)
        print(f"  {iso3} all classes (class p r_national mapspam_ha): " + ", ".join(
            f"{r.exposure_class} {r.p:.3f} {r.r_national:+.2f} {r.mapspam_ha:.0f}" for r in d.itertuples()))
    small = detail[detail["mapspam_ha"].notna() & ~detail["fallback"] & (detail["mapspam_ha"] < 0.01 * detail["area_ha"])]
    print(f"\nclasses with a series but MapSPAM area < 1% of FAOSTAT area: {len(small)} country-classes, "
          f"p summed per country max {small.groupby('iso3')['p'].sum().max() if len(small) else 0:.3f}")

    def sprow(face, pred):
        r = sp[(sp["face"] == face) & (sp["predictor"] == pred)].iloc[0]
        return f"rho={r.rho:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] p={r.p:.3f} n={r.n}"

    def olsrow(outcome, model, term):
        r = ols[(ols["outcome"] == outcome) & (ols["model"] == model) & (ols["term"] == term)].iloc[0]
        return f"beta={r.beta:+.3f} se={r.se:.3f} p={r.p:.3f} n={r.n} adj_r2={r.adj_r2:.3f}"

    print("\nSpearman rho [bootstrap 95% CI]:")
    for r in sp.itertuples():
        print(f"  {r.face:<16} vs {r.predictor:<16} rho={r.rho:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}]  p={r.p:.3f}  n={r.n}")
    print("\nOLS standardised betas (HC3):")
    print(ols.round(3).to_string(index=False))
    print("\npanel coefficients (anom in fractions of trend; spei in SPEI units; interacted vars as z-scores):")
    print(coef.round(4).to_string(index=False))

    print("\nCHANNEL (i) exposure: does spread make the national SPEI less variable?")
    for out in ("sd_spei", "n_drought_years"):
        print(f"  {out} vs exp_corr Spearman {sprow(out, 'exp_corr')}; vs log_area {sprow(out, 'log_area')}")
        print(f"  {out} ~ hill1_ms + exp_corr:            exp_corr {olsrow(out, 'c', 'exp_corr')}")
        print(f"  {out} ~ hill1_ms + exp_corr + log_area: exp_corr {olsrow(out, 'd', 'exp_corr')}; "
              f"log_area {olsrow(out, 'd', 'log_area')}")
    print("CHANNEL (ii) sensitivity: does spread lower the response of anom to a given national SPEI?")
    for model in [k for k in panel_specs(ed) if k != "base"]:
        for t in coef[(coef["model"] == model) & (coef["term"] != "spei")].itertuples():
            print(f"  {model:<16} {t.term:<22} beta={t.beta:+.4f} se={t.se:.4f} p={t.p:.3f} n_countries={t.n_countries}")
    print("exp_corr beyond crop count and size (cross-section, model c and d):")
    for f in ["stability", "resistance", "recovery", "vulnerability", "vuln_drought"]:
        print(f"  {f:<14} c: exp_corr {olsrow(f, 'c', 'exp_corr')} | d: exp_corr {olsrow(f, 'd', 'exp_corr')}, "
              f"log_area {olsrow(f, 'd', 'log_area')}")

    fig_expdiv(df, sp)


if __name__ == "__main__":
    main()
