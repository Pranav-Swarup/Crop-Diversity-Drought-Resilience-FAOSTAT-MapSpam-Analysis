"""Per-crop December SPEI-12 exposure for 1961-1992 -> data/clean/crop_climate_pre.parquet.

Same definition as src.climate.aggregate.compute_per_crop (December SPEI-12; weights = crop area x fraction of the
cell inside the country; mean over cells with a finite value), with a local year-range loader because
aggregate.load_spei is fixed to 1993-2023. Before the pre-window is computed, the same code is run for 1993-2023
and must reproduce data/clean/crop_climate.parquet (same keys, max abs difference < 1e-6).

Columns: iso3, year, mapspam_crop, crop_spei12_w

Run: python -m src.climate.exposure_pre
"""
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr

from ..common import paths
from . import cropland
from .aggregate import coverage_fractions, weighted_mean
from .download import SPEI_DIR

OUT = paths.CLEAN / "crop_climate_pre.parquet"
PRE = (1961, 1992)
WINDOW = (1993, 2023)


def load_spei_years(n, y0, y1):
    """SPEI-n for years y0..y1 as (time, lat north->south, lon west->east)."""
    ds = xr.open_dataset(SPEI_DIR / f"spei{n:02d}.nc")
    da = ds["spei"].sel(time=slice(f"{y0}-01-01", f"{y1}-12-31")).sortby("lat", ascending=False).sortby("lon")
    assert da.shape[1:] == (360, 720), da.shape
    assert np.isclose(float(da.lat[0]), 89.75) and np.isclose(float(da.lon[0]), -179.75), "SPEI grid is not the expected 0.5 degree grid"
    assert len(da.time) == 12 * (y1 - y0 + 1), f"SPEI-{n}: {len(da.time)} months in {y0}-{y1}"
    return da


def per_crop(y0, y1, cover, npz):
    spei = load_spei_years(12, y0, y1)
    years = np.arange(y0, y1 + 1)
    dec = spei.isel(time=slice(11, None, 12))
    assert list(dec.time.dt.month.values) == [12] * len(years) and list(dec.time.dt.year.values) == list(years)
    dec = dec.values.reshape(len(years), -1)
    rows = []
    for iso3, (cells, frac) in sorted(cover.items()):
        if len(cells) == 0:
            continue
        for crop in npz.files:
            w_crop = npz[crop].ravel()[cells] * frac
            if w_crop.sum() > 0:
                w, _ = weighted_mean(dec[:, cells], w_crop)
                if np.isfinite(w).any():
                    df = pd.DataFrame({"iso3": iso3, "year": years, "mapspam_crop": crop, "crop_spei12_w": w})
                    rows.append(df[df["crop_spei12_w"].notna()])
    return pd.concat(rows, ignore_index=True).astype({"year": int})


def main():
    if not cropland.OUT_NPZ.exists():
        print("cropland_05deg.npz missing: run python -m src.climate.cropland first")
        sys.exit(1)
    npz = np.load(cropland.OUT_NPZ)
    spei = load_spei_years(12, *WINDOW)
    template = spei.isel(time=0).rio.write_crs(4326).rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    cover = coverage_fractions(gpd.read_file(paths.COUNTRIES_GEOJSON), template)

    # 1. Reproduce the existing 1993-2023 file.
    ref = pd.read_parquet(paths.CLEAN_REAL / "crop_climate.parquet")
    new = per_crop(*WINDOW, cover, npz)
    key = ["iso3", "year", "mapspam_crop"]
    m = ref.merge(new, on=key, how="outer", suffixes=("_ref", "_new"), indicator=True)
    n_only = (m["_merge"] != "both").sum()
    maxdiff = (m["crop_spei12_w_ref"] - m["crop_spei12_w_new"]).abs().max()
    print(f"reproduction 1993-2023: ref {len(ref)} rows, recomputed {len(new)} rows, keys not in both: {n_only}, "
          f"max abs difference {maxdiff:.3e}")
    assert n_only == 0, "key sets differ from crop_climate.parquet"
    assert maxdiff < 1e-6, f"max abs difference {maxdiff} vs crop_climate.parquet"

    # 2. Pre-window.
    out = per_crop(*PRE, cover, npz)
    assert not out.duplicated(key).any()
    assert np.isfinite(out["crop_spei12_w"]).all() and out["crop_spei12_w"].abs().max() < 10
    full = list(range(PRE[0], PRE[1] + 1))
    bad = out.groupby(["iso3", "mapspam_crop"])["year"].apply(lambda s: sorted(s) != full)
    assert not bad.any(), f"{int(bad.sum())} country-crops with incomplete years 1961-1992"
    out = out.sort_values(key).reset_index(drop=True)
    out.to_parquet(OUT, index=False)

    n_pairs = out.groupby(["iso3", "mapspam_crop"]).ngroups
    ref_pairs = ref.groupby(["iso3", "mapspam_crop"]).ngroups
    print(f"crop_climate_pre: {len(out)} rows, {out['iso3'].nunique()} countries, {out['mapspam_crop'].nunique()} crops, "
          f"years {out['year'].min()}-{out['year'].max()}, country-crops {n_pairs} (1993-2023 file: {ref_pairs}), "
          f"missing {out.isna().mean().mean():.1%}")
    print(f"crop_spei12_w: mean {out['crop_spei12_w'].mean():+.3f}, sd {out['crop_spei12_w'].std():.3f}, "
          f"range {out['crop_spei12_w'].min():+.2f} to {out['crop_spei12_w'].max():+.2f}")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
