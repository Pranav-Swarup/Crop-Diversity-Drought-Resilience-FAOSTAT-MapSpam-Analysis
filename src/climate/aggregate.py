"""SPEIbase v2.10 -> cropland-weighted country-year drought exposure -> data/clean/climate.parquet.

First-iteration pipeline written from src/metrics' side; Prathmesh owns this module.

  spei12_w    cropland-weighted mean of SPEI-12 at December
  spei12_unw  same, weighted by cell area only (cos latitude)
  spei6min_w  minimum over the 12 months of cropland-weighted SPEI-6

Run: python -m src.climate.aggregate
"""
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from exactextract import exact_extract

from ..common import paths
from ..common.contract import YEAR_MAX, YEAR_MIN, summary, validate
from . import cropland
from .download import SPEI_DIR

MIN_COVERAGE = 0.9


def load_spei(n):
    """SPEI-n for the window as (time, lat north->south, lon west->east) plus its time index."""
    ds = xr.open_dataset(SPEI_DIR / f"spei{n:02d}.nc")
    da = ds["spei"].sel(time=slice(f"{YEAR_MIN}-01-01", f"{YEAR_MAX}-12-31")).sortby("lat", ascending=False).sortby("lon")
    assert da.shape[1:] == (360, 720), da.shape
    assert np.isclose(float(da.lat[0]), 89.75) and np.isclose(float(da.lon[0]), -179.75), "SPEI grid is not the expected 0.5 degree grid"
    assert len(da.time) == 12 * (YEAR_MAX - YEAR_MIN + 1), f"SPEI-{n}: {len(da.time)} months in the window"
    return da


def coverage_fractions(gdf, template):
    """Per country: (flat cell index, fraction of the cell inside the country) on the 0.5 degree grid."""
    res = exact_extract(template, gdf, ["cell_id", "coverage"], include_cols=["iso3"], output="pandas")
    return {r.iso3: (np.asarray(r.cell_id, dtype=int), np.asarray(r.coverage, dtype=float)) for r in res.itertuples()}


def weighted_mean(values, w):
    """values: (time, cells); w: (cells,). Mean over cells with a finite value. Also returns
    the share of total weight that had a finite value."""
    ok = np.isfinite(values)
    wsum = (ok * w).sum(axis=1)
    mean = np.where(wsum > 0, np.nansum(values * w, axis=1) / np.where(wsum > 0, wsum, 1), np.nan)
    return mean, (wsum / w.sum() if w.sum() > 0 else np.full(len(values), np.nan))


import argparse

def compute_per_crop(spei12, years, cover):
    print("Computing per-crop climate weights...")
    dec = spei12.isel(time=slice(11, None, 12)).values.reshape(len(years), -1)
    
    npz = np.load(cropland.OUT_NPZ)
    
    rows = []
    for iso3, (cells, frac) in sorted(cover.items()):
        if len(cells) == 0:
            continue
            
        for crop_name in npz.files:
            crop_flat = npz[crop_name].ravel()
            w_crop = crop_flat[cells] * frac
            if w_crop.sum() > 0:
                w, _ = weighted_mean(dec[:, cells], w_crop)
                if np.isfinite(w).any():
                    df = pd.DataFrame({"iso3": iso3, "year": years, "mapspam_crop": crop_name, "crop_spei12_w": w})
                    df = df[df["crop_spei12_w"].notna()]
                    rows.append(df)
                    
    out = pd.concat(rows, ignore_index=True)
    out = out.astype({"year": int}).reset_index(drop=True)
    
    out_path = paths.CLEAN / "crop_climate.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"Saved per-crop climate data to {out_path} ({len(out)} rows, {out['iso3'].nunique()} countries)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-crop", action="store_true", help="Also compute per-crop spei12_w")
    args = parser.parse_args()

    if not cropland.OUT.exists():
        print("cropland raster missing: run python -m src.climate.cropland first")
        sys.exit(1)
    crop = np.load(cropland.OUT)                      # (360, 720), north -> south
    spei12, spei6 = load_spei(12), load_spei(6)
    years = np.arange(YEAR_MIN, YEAR_MAX + 1)
    dec = spei12.isel(time=slice(11, None, 12))
    assert list(dec.time.dt.month.values) == [12] * len(years)
    dec = dec.values.reshape(len(years), -1)
    s6 = spei6.values.reshape(len(years), 12, -1)

    template = spei12.isel(time=0).rio.write_crs(4326).rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    gdf = gpd.read_file(paths.COUNTRIES_GEOJSON)
    cover = coverage_fractions(gdf, template)
    coslat = np.repeat(np.cos(np.deg2rad(spei12.lat.values)), 720)
    crop_flat = crop.ravel()

    rows, low_cov = [], []
    for iso3, (cells, frac) in sorted(cover.items()):
        if len(cells) == 0:
            continue
        w_crop, w_area = crop_flat[cells] * frac, coslat[cells] * frac
        unw, _ = weighted_mean(dec[:, cells], w_area)
        if w_crop.sum() > 0:
            w, cov = weighted_mean(dec[:, cells], w_crop)
            m6 = np.stack([weighted_mean(s6[:, m, cells], w_crop)[0] for m in range(12)], axis=1).min(axis=1)
            if np.nanmin(cov) < MIN_COVERAGE:
                low_cov.append((iso3, float(np.nanmin(cov))))
        else:
            w, m6 = np.full(len(years), np.nan), np.full(len(years), np.nan)
        rows.append(pd.DataFrame({"iso3": iso3, "year": years, "spei12_w": w, "spei12_unw": unw, "spei6min_w": m6}))
    out = pd.concat(rows, ignore_index=True)
    out = out[out["spei12_w"].notna() | out["spei12_unw"].notna()].astype({"year": int}).reset_index(drop=True)

    validate(out, "climate")
    assert out[["spei12_w", "spei12_unw", "spei6min_w"]].abs().max().max() < 10
    out.to_parquet(paths.CLIMATE, index=False)
    print(summary(out, "climate"))
    print(f"countries covered: {out['iso3'].nunique()}; with cropland weights: {out.loc[out['spei12_w'].notna(), 'iso3'].nunique()}")
    print(f"countries with < {MIN_COVERAGE:.0%} valid cropland coverage in some year: {[(i, round(c, 2)) for i, c in low_cov]}")
    diff = (out["spei12_w"] - out["spei12_unw"]).abs().groupby(out["iso3"]).mean().nlargest(10)
    print("mean |spei12_w - spei12_unw|, top 10: " + ", ".join(f"{i} {v:.2f}" for i, v in diff.items()))

    if args.per_crop:
        if not cropland.OUT_NPZ.exists():
            print("Warning: OUT_NPZ not found, cannot compute per-crop")
        else:
            compute_per_crop(spei12, years, cover)

if __name__ == "__main__":
    main()
