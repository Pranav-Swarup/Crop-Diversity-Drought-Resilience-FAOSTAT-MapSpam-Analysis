"""Harvest-month SPEI-6 exposure per country and MapSPAM crop class -> data/clean/crop_climate_season.parquet.

Columns: iso3, year, mapspam_crop, crop_spei6_harv_w (1993-2023)

Per 0.5 degree cell and crop: the harvest month is the calendar month (non-leap year) that contains the
maturity day of year of the GGCMI Phase 3 crop calendar (Jagermeyr et al. 2021, Zenodo 10.5281/zenodo.5062513).
The rainfed (rf) calendar is used where it is defined, the irrigated (ir) one in cells where only that is defined.
Exposure of a cell in year y = SPEI-6 of the harvest month of year y (the six months ending in it).
Country series = mean over the country's cells weighted by (class area in cropland_05deg.npz) x (fraction of the
cell inside the country), over cells with both a harvest month and a finite SPEI value. A country-class is kept
only if the share of weight with a valid value is >= MIN_COVERAGE in every year.

Calendar crop -> MapSPAM class: mai MAIZ, ri1 RICE, soy SOYB, bar BARL, rye OCER, mil PMIL and MILL, sor SORG,
pea OPUL, bea BEAN, sgb SUGB, sgc SUGC, cas CASS, sun SUNF, rap RAPE, nut GROU, cot COTT, pot POTA, and
wheat WHEA using the winter-wheat (wwh) maturity day where it is defined, spring wheat (swh) elsewhere.

Facts about the calendar files: every GGCMI file (all crops, rf and ir) is defined on the same 67,420 land cells.
So the rainfed calendar is always the one used (the irrigated one never is), and for wheat the winter-wheat
calendar is used in every cell of the main file (spring wheat never). Winter and spring wheat maturity months
differ by more than one month on 40.3% of global wheat area. As a sensitivity check, WHEA is also computed with
the spring-wheat (swh) maturity day in every cell -> data/clean/crop_climate_season_swh.parquet (same columns,
class WHEA only, same weights, coverage rule and assertions). The main file is not affected by it.

Run: python -m src.climate.download_calendar, then python -m src.climate.exposure_season
"""
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr

from ..common import paths
from ..common.contract import YEAR_MAX, YEAR_MIN
from . import cropland
from .aggregate import MIN_COVERAGE, coverage_fractions, load_spei, weighted_mean
from .download_calendar import CROPS, cal_file

OUT = paths.CLEAN / "crop_climate_season.parquet"
OUT_SWH = paths.CLEAN / "crop_climate_season_swh.parquet"
YEARS = np.arange(YEAR_MIN, YEAR_MAX + 1)
CAL_TO_SPAM = {"mai": ["MAIZ"], "ri1": ["RICE"], "soy": ["SOYB"], "bar": ["BARL"], "rye": ["OCER"],
               "mil": ["PMIL", "MILL"], "sor": ["SORG"], "pea": ["OPUL"], "bea": ["BEAN"], "sgb": ["SUGB"],
               "sgc": ["SUGC"], "cas": ["CASS"], "sun": ["SUNF"], "rap": ["RAPE"], "nut": ["GROU"],
               "cot": ["COTT"], "pot": ["POTA"]}
CLASSES = sorted([c for v in CAL_TO_SPAM.values() for c in v] + ["WHEA"])
MONTH_END = np.cumsum([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])     # last day of year of each month, non-leap


def describe(path):
    ds = xr.open_dataset(path)
    print(f"calendar file {path.name}: dims {dict(ds.sizes)}; coords {list(ds.coords)} "
          f"(lat {float(ds.lat[0])} -> {float(ds.lat[-1])}, lon {float(ds.lon[0])} -> {float(ds.lon[-1])})")
    print(f"  attrs {ds.attrs}")
    for v in ds.data_vars:
        print(f"  var {v} {ds[v].dims} {ds[v].dtype} {ds[v].attrs}")


def maturity(crop, regime):
    """Maturity day of year on the SPEI grid as a flat float array (NaN = undefined)."""
    ds = xr.open_dataset(cal_file(crop, regime))
    ds = ds.sortby("lat", ascending=False).sortby("lon")
    assert ds["maturity_day"].dims == ("lat", "lon") and ds["maturity_day"].shape == (360, 720)
    assert np.isclose(float(ds.lat[0]), 89.75) and np.isclose(float(ds.lon[0]), -179.75), "calendar grid is not the SPEI grid"
    assert np.allclose(ds.lat.values, np.arange(89.75, -90, -0.5)) and np.allclose(ds.lon.values, np.arange(-179.75, 180, 0.5))
    m = ds["maturity_day"].values.astype(float).ravel()
    assert np.nanmin(m) >= 1 and np.nanmax(m) <= 365, f"{crop} {regime}: maturity day outside 1..365"
    return m


def harvest_month(mday):
    """Month index 0..11 containing day of year `mday` (NaN stays NaN)."""
    out = np.full(mday.shape, np.nan)
    ok = np.isfinite(mday)
    out[ok] = np.searchsorted(MONTH_END, mday[ok], side="left")
    return out


def rf_then_ir(crop):
    rf, ir = maturity(crop, "rf"), maturity(crop, "ir")
    return np.where(np.isfinite(rf), rf, ir), rf, ir


def series(classes, month, npz, cover, s6):
    """Country-class harvest-month SPEI-6 series. Returns (DataFrame, dropped-by-class dict, n country-classes with area)."""
    rows, dropped, n_total = [], {}, 0
    for iso3, (cells, frac) in sorted(cover.items()):
        if len(cells) == 0:
            continue
        for c in classes:
            w = npz[c].ravel()[cells] * frac
            if w.sum() <= 0:
                continue
            n_total += 1
            mon = month[c][cells]
            idx = np.flatnonzero(np.isfinite(mon))
            vals = np.full((len(YEARS), len(cells)), np.nan)
            vals[:, idx] = s6[:, mon[idx].astype(int), cells[idx]]
            mean, cov = weighted_mean(vals, w)
            if not (np.nanmin(cov) >= MIN_COVERAGE) or not np.isfinite(mean).all():
                dropped[c] = dropped.get(c, 0) + 1
                continue
            rows.append(pd.DataFrame({"iso3": iso3, "year": YEARS, "mapspam_crop": c, "crop_spei6_harv_w": mean}))
    return pd.concat(rows, ignore_index=True).astype({"year": int}), dropped, n_total


def check(out, classes):
    key = ["iso3", "year", "mapspam_crop"]
    assert not out.duplicated(key).any()
    assert set(out["mapspam_crop"]) <= set(classes)
    assert np.isfinite(out["crop_spei6_harv_w"]).all() and out["crop_spei6_harv_w"].abs().max() < 10
    assert (out.groupby(["iso3", "mapspam_crop"])["year"].apply(lambda s: sorted(s) == list(YEARS))).all(), "incomplete years"
    return out.sort_values(key).reset_index(drop=True)


def main():
    if not cropland.OUT_NPZ.exists():
        print("cropland_05deg.npz missing: run python -m src.climate.cropland first")
        sys.exit(1)
    missing = [f"{c}_{r}" for c in CROPS for r in ("rf", "ir") if not cal_file(c, r).exists()]
    if missing:
        print(f"calendar files missing: {missing}; run python -m src.climate.download_calendar")
        sys.exit(1)

    describe(cal_file("wwh", "rf"))

    # harvest month per class
    month = {}
    for crop, classes in CAL_TO_SPAM.items():
        md, _, _ = rf_then_ir(crop)
        for c in classes:
            month[c] = harvest_month(md)
    wwh, wwh_rf, _ = rf_then_ir("wwh")
    swh, _, _ = rf_then_ir("swh")
    month["WHEA"] = harvest_month(np.where(np.isfinite(wwh), wwh, swh))
    assert sorted(month) == CLASSES and len(CLASSES) == 19, CLASSES

    npz = np.load(cropland.OUT_NPZ)
    wheat_area = npz["WHEA"].ravel()
    mw, ms = harvest_month(wwh), harvest_month(swh)
    both = np.isfinite(mw) & np.isfinite(ms)
    d = np.abs(mw - ms)
    d = np.minimum(d, 12 - d)                                # circular distance in months
    differs = both & (d > 1)
    print(f"wheat: cells with winter maturity defined {int(np.isfinite(mw).sum())}, spring {int(np.isfinite(ms).sum())}; "
          f"cells where winter is used {int(np.isfinite(wwh).sum())}, where only spring would be used "
          f"{int((~np.isfinite(wwh) & np.isfinite(swh)).sum())}")
    print(f"wheat: share of global wheat area where winter and spring maturity months differ by more than one month: "
          f"{wheat_area[differs].sum() / wheat_area.sum():.1%} (area with both defined {wheat_area[both].sum() / wheat_area.sum():.1%})")

    s6 = load_spei(6).values.reshape(len(YEARS), 12, -1)
    template = load_spei(12).isel(time=0).rio.write_crs(4326).rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    cover = coverage_fractions(gpd.read_file(paths.COUNTRIES_GEOJSON), template)

    out, dropped, n_total = series(CLASSES, month, npz, cover, s6)

    key = ["iso3", "year", "mapspam_crop"]
    assert len(set(CLASSES)) == 19
    out = check(out, CLASSES)
    out.to_parquet(OUT, index=False)

    n_kept = out.groupby(["iso3", "mapspam_crop"]).ngroups
    print(f"crop_climate_season: {len(out)} rows, {out['iso3'].nunique()} countries, {out['mapspam_crop'].nunique()} classes, "
          f"years {out['year'].min()}-{out['year'].max()}, country-classes kept {n_kept} of {n_total} with area; "
          f"missing {out.isna().mean().mean():.1%}")
    print(f"country-classes dropped for coverage < {MIN_COVERAGE:.0%} in some year: {sum(dropped.values())}; by class: "
          f"{dict(sorted(dropped.items(), key=lambda kv: -kv[1]))}")

    ref = pd.read_parquet(paths.CLEAN_REAL / "crop_climate.parquet")
    m = out.merge(ref, on=key)
    print("Pearson r, crop_spei6_harv_w vs December crop_spei12_w (1993-2023):")
    for crop in ("MAIZ", "WHEA"):
        res = []
        for iso in ("USA", "IND", "FRA", "BRA"):
            g = m[(m["iso3"] == iso) & (m["mapspam_crop"] == crop)]
            res.append(f"{iso} {g['crop_spei6_harv_w'].corr(g['crop_spei12_w']):+.2f}" if len(g) else f"{iso} n/a")
        print(f"  {crop}: " + ", ".join(res))
    print(f"Saved: {OUT}")

    # Sensitivity: WHEA with the spring-wheat maturity day in every cell.
    month_swh = dict(month)
    month_swh["WHEA"] = ms
    sw, sw_dropped, sw_total = series(["WHEA"], month_swh, npz, cover, s6)
    sw = check(sw, ["WHEA"])
    sw.to_parquet(OUT_SWH, index=False)
    main_w = out[out["mapspam_crop"] == "WHEA"]
    print(f"crop_climate_season_swh: {len(sw)} rows, {sw['iso3'].nunique()} countries, class WHEA, years {sw['year'].min()}-{sw['year'].max()}, "
          f"country-classes kept {sw['iso3'].nunique()} of {sw_total}, dropped for coverage {sum(sw_dropped.values())}, missing {sw.isna().mean().mean():.1%}")
    assert set(sw["iso3"]) == set(main_w["iso3"]), "winter-rule and spring-rule WHEA cover different countries"
    mm = main_w.merge(sw, on=["iso3", "year"], suffixes=("_win", "_spr"))
    r = mm.groupby("iso3").apply(lambda g: g["crop_spei6_harv_w_win"].corr(g["crop_spei6_harv_w_spr"]), include_groups=False)
    area = pd.read_parquet(paths.CLEAN / "mapspam_area.parquet").query("mapspam_crop == 'WHEA'").set_index("iso3")["area_ha"]
    top = area.reindex(r.index).nlargest(10).index
    print(f"WHEA winter-rule vs spring-rule harvest-month SPEI-6, Pearson r per country (n = {len(r)}): "
          f"median {r.median():+.3f}, 10th percentile {r.quantile(0.10):+.3f}, min {r.min():+.3f}")
    print("ten wheat-area-largest countries (wheat area Mha, r):")
    for iso in top:
        print(f"  {iso}  {area[iso] / 1e6:6.2f}  {r[iso]:+.3f}")
    print(f"Saved: {OUT_SWH}")


if __name__ == "__main__":
    main()
