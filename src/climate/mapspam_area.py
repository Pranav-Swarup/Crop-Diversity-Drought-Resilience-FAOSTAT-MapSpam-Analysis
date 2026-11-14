"""MapSPAM 2020 physical area per country and crop, split into irrigated and rainfed -> data/clean/mapspam_area.parquet.

Columns: iso3, mapspam_crop, area_ha (all technologies), irr_ha (irrigated), rf_ha (rainfed).
Country total = sum over the country's 0.5 degree cells of (crop area in the cell x fraction of the cell inside
the country), the same weights as src.climate.aggregate.compute_per_crop. Only rows with area_ha > 0 are kept.

Also writes the 0.5 degree irrigated and rainfed rasters (same key convention as cropland_05deg.npz):
  data/raw/derived/cropland_05deg_irr.npz, data/raw/derived/cropland_05deg_rf.npz

MapSPAM's A is not forced to equal I + R: the global gap per crop is printed.

Run: python -m src.climate.mapspam_area
"""
import re
import sys
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rioxarray  # noqa: F401  (registers the .rio accessor)

from ..common import paths
from .aggregate import coverage_fractions, load_spei
from .cropland import BLOCK, OUT_NPZ, SHAPE_5MIN
from .download import MAPSPAM_HELP, SPAM_DIR, SPAM_FILE

OUT = paths.CLEAN / "mapspam_area.parquet"
OUT_IRR = paths.RAW / "derived" / "cropland_05deg_irr.npz"
OUT_RF = paths.RAW / "derived" / "cropland_05deg_rf.npz"
TECH = {"I": re.compile(r"_A_([A-Z]{3,4})_I\.tif$"), "R": re.compile(r"_A_([A-Z]{3,4})_R\.tif$")}


def read_half(zf_path, name):
    """One GeoTIFF from the zip, summed from 5 arcmin to 0.5 degree exactly as cropland.py does."""
    with rasterio.open(f"zip://{zf_path}!{name}") as r:
        assert (r.height, r.width) == SHAPE_5MIN, f"{name}: unexpected shape {r.shape}"
        assert abs(r.bounds.left + 180) < 1e-6 and abs(r.bounds.top - 90) < 1e-6, f"{name}: not global ({r.bounds})"
        a = r.read(1, masked=True).filled(0).astype(np.float64)
    a = np.where(np.isfinite(a) & (a > 0), a, 0)
    return a.reshape(SHAPE_5MIN[0] // BLOCK, BLOCK, SHAPE_5MIN[1] // BLOCK, BLOCK).sum(axis=(1, 3))


def build_tech(src, tech):
    with zipfile.ZipFile(src) as zf:
        names = sorted(n for n in zf.namelist() if TECH[tech].search(n))
    assert names, f"no {tech} GeoTIFFs found"
    data = {TECH[tech].search(n).group(1): read_half(src, n) for n in names}
    assert len(data) == 46, f"expected 46 crops for technology {tech}, found {len(data)}"
    return data


def main():
    src = SPAM_DIR / SPAM_FILE
    if not src.exists():
        print(MAPSPAM_HELP)
        sys.exit(1)
    if not OUT_NPZ.exists():
        print("cropland_05deg.npz missing: run python -m src.climate.cropland first")
        sys.exit(1)

    allc = dict(np.load(OUT_NPZ))
    irr, rf = build_tech(src, "I"), build_tech(src, "R")
    OUT_IRR.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_IRR, **irr)
    np.savez_compressed(OUT_RF, **rf)
    crops = sorted(allc)
    assert len(crops) == 46 and sorted(irr) == crops and sorted(rf) == crops, "crop sets differ between technologies"

    spei12 = load_spei(12)
    template = spei12.isel(time=0).rio.write_crs(4326).rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    cover = coverage_fractions(gpd.read_file(paths.COUNTRIES_GEOJSON), template)

    rows = []
    for iso3, (cells, frac) in sorted(cover.items()):
        if len(cells) == 0:
            continue
        for c in crops:
            rows.append((iso3, c, float((allc[c].ravel()[cells] * frac).sum()),
                         float((irr[c].ravel()[cells] * frac).sum()), float((rf[c].ravel()[cells] * frac).sum())))
    out = pd.DataFrame(rows, columns=["iso3", "mapspam_crop", "area_ha", "irr_ha", "rf_ha"])
    out = out[out["area_ha"] > 0].reset_index(drop=True)

    assert out["mapspam_crop"].nunique() == 46
    assert (out[["area_ha", "irr_ha", "rf_ha"]] >= 0).all().all()
    assert not out.duplicated(["iso3", "mapspam_crop"]).any()
    tot_mha = out["area_ha"].sum() / 1e6
    assert 800 < tot_mha < 2000, f"global area {tot_mha:.0f} Mha outside plausible range"

    g = out.groupby("mapspam_crop")[["area_ha", "irr_ha", "rf_ha"]].sum()
    g["gap"] = (g["irr_ha"] + g["rf_ha"]) / g["area_ha"] - 1
    bad = g[g["gap"].abs() > 0.02]

    out.to_parquet(OUT, index=False)
    print(f"mapspam_area: {len(out)} rows, {out['iso3'].nunique()} countries, {out['mapspam_crop'].nunique()} crops, "
          f"missing {out.isna().mean().mean():.1%}")
    print(f"global area (A) {tot_mha:.0f} Mha; irrigated {out['irr_ha'].sum() / 1e6:.0f} Mha; rainfed {out['rf_ha'].sum() / 1e6:.0f} Mha")
    print(f"crops where (I + R) / A - 1 exceeds 2% globally: {len(bad)}")
    for c, r in bad.iterrows():
        print(f"  {c}: A {r.area_ha / 1e6:.3f} Mha, I+R {(r.irr_ha + r.rf_ha) / 1e6:.3f} Mha, gap {r.gap:+.2%}")
    print(f"largest |gap| over all crops: {g['gap'].abs().max():.3%}")
    print(f"Saved: {OUT}, {OUT_IRR}, {OUT_RF}")
    assert bad.empty, f"I + R differs from A by more than 2% for {list(bad.index)}; reported above, not forced"


if __name__ == "__main__":
    main()
