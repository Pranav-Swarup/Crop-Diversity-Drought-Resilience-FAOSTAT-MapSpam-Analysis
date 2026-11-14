"""District SPEI-12 for India -> data/clean/india/climate.parquet (dist_code, year, spei12_unw, spei12_w, n_cells).

Same method as src.climate.aggregate, on the matched district polygons (src.india.geo) instead of countries:
  spei12_unw  December SPEI-12, mean over the 0.5 degree cells touching the district, weighted by the cell area inside it
              (coverage fraction x cos latitude). Always computed.
  spei12_w    the same weighted by MapSPAM cropland (the design's drought variable). Only computed if
              data/raw/derived/cropland_05deg.npy exists (needs the manual MapSPAM download, then src.climate.cropland);
              otherwise the column is left empty, NOT replaced by spei12_unw.
spei4kh_unw / spei4kh_w   kharif variant (robustness only): SPEI-4 of September (June-September rainfall and PET), same two weightings;
              only if data/raw/spei/spei04.nc exists. Not used by the core analysis.
Years 1993-2022 (December of year Y covers the monsoon of agricultural year Y-(Y+1)).

A 0.5 degree cell (~55 km) is as large as many districts: neighbouring districts share cells, so district SPEI varies
little within a state-year. `n_cells` and the printed within-state spread document this.

Run: python -m src.india.climate
"""
import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401
import xarray as xr

from ..common import paths
from ..climate import cropland
from ..climate.aggregate import coverage_fractions, weighted_mean
from ..climate.download import SPEI_DIR

OUT = paths.CLEAN_REAL / "india"
YEARS = np.arange(1993, 2023)
KHARIF_FILE = "spei04.nc"


def load_month_spei(fname="spei12.nc", month=12):
    """Values of one calendar month of every year 1993-2022, (years, cells), top-left origin grid like src.climate.aggregate."""
    ds = xr.open_dataset(SPEI_DIR / fname)
    da = ds["spei"].sel(time=slice("1993-01-01", "2022-12-31")).sortby("lat", ascending=False).sortby("lon")
    assert da.shape[1:] == (360, 720), da.shape
    assert np.isclose(float(da.lat[0]), 89.75) and np.isclose(float(da.lon[0]), -179.75)
    assert len(da.time) == 12 * len(YEARS)
    dec = da.isel(time=slice(month - 1, None, 12))
    assert list(dec.time.dt.month.values) == [month] * len(YEARS) and list(dec.time.dt.year.values) == list(YEARS)
    return da, dec.values.reshape(len(YEARS), -1)


def main():
    da, dec = load_month_spei()
    kh = load_month_spei(KHARIF_FILE, 9)[1] if (SPEI_DIR / KHARIF_FILE).exists() else None   # kharif: SPEI-4 of September = June-September
    template = da.isel(time=0).rio.write_crs(4326).rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    gdf = gpd.read_file(OUT / "districts.geojson").rename(columns={"dist_code": "iso3"})   # coverage_fractions keys on iso3
    cover = coverage_fractions(gdf, template)
    coslat = np.repeat(np.cos(np.deg2rad(da.lat.values)), 720)
    crop = np.load(cropland.OUT).ravel() if cropland.OUT.exists() else None
    if crop is None:
        print("NOTE: cropland raster missing: spei12_w NOT computed (MapSPAM is a manual download; see src.climate.download)")

    rows = []
    for code, (cells, frac) in sorted(cover.items()):
        if len(cells) == 0:
            continue
        unw, cov = weighted_mean(dec[:, cells], coslat[cells] * frac)
        w = np.full(len(YEARS), np.nan)
        if crop is not None and (crop[cells] * frac).sum() > 0:
            w, _ = weighted_mean(dec[:, cells], crop[cells] * frac)
        r = pd.DataFrame({"dist_code": code, "year": YEARS, "spei12_unw": unw, "spei12_w": w, "n_cells": len(cells), "valid_share": cov})
        if kh is not None:                                   # same weights as the December SPEI-12 columns
            r["spei4kh_unw"], _ = weighted_mean(kh[:, cells], coslat[cells] * frac)
            r["spei4kh_w"] = np.nan
            if crop is not None and (crop[cells] * frac).sum() > 0:
                r["spei4kh_w"], _ = weighted_mean(kh[:, cells], crop[cells] * frac)
        rows.append(r)
    out = pd.concat(rows, ignore_index=True).astype({"year": int})

    assert not out.duplicated(["dist_code", "year"]).any()
    assert out.spei12_unw.abs().max() < 10 and out.year.between(1993, 2022).all()
    assert set(out.dist_code) <= set(gdf.iso3)
    out.to_parquet(OUT / "climate.parquet", index=False)

    d = pd.read_csv(OUT / "districts.csv", dtype={"dist_code": str}).set_index("dist_code")
    out["state"] = out.dist_code.map(d.state_name)
    within = out.groupby(["state", "year"]).spei12_unw.transform(lambda s: s - s.mean())
    print(f"rows {len(out):,}; districts {out.dist_code.nunique()}; years {out.year.min()}-{out.year.max()}; "
          f"missing spei12_unw {out.spei12_unw.isna().mean():.1%}; spei12_w {'computed' if crop is not None else 'not computed'}")
    print(f"cells per district: median {out.groupby('dist_code').n_cells.first().median():.0f}, "
          f"districts with a single cell {(out.groupby('dist_code').n_cells.first() == 1).mean():.1%}")
    print(f"SD of spei12_unw: total {out.spei12_unw.std():.2f}; within state-year (after removing the state-year mean) {within.std():.2f}")


if __name__ == "__main__":
    main()
