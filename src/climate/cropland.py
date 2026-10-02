"""MapSPAM 2020 physical area, all crops, summed onto the SPEI 0.5 degree grid.

First-iteration pipeline written from src/metrics' side; Prathmesh owns this module.
Writes:
  data/raw/derived/cropland_05deg.npy  — total physical area (ha per cell), used by aggregate.py
  data/raw/derived/cropland_05deg.npz  — per-crop arrays (ha per cell), for crop-specific analyses

Run: python -m src.climate.cropland
"""
import re
import sys
import zipfile

import numpy as np
import rasterio

from ..common import paths
from .download import MAPSPAM_HELP, SPAM_DIR, SPAM_FILE

OUT_NPY = paths.RAW / "derived" / "cropland_05deg.npy"
OUT_NPZ = paths.RAW / "derived" / "cropland_05deg.npz"
# Legacy alias so aggregate.py can still do `from .cropland import OUT`
OUT = OUT_NPY
BLOCK = 6                       # 5 arcmin -> 0.5 degree
SHAPE_5MIN = (2160, 4320)
# "A" = all technologies (MapSPAM readme: *_A = all, _I irrigated, _R rainfed, ...).
ALL_TECH = re.compile(r"_A_([A-Z]{3,4})_A\.tif$")


def main():
    src = SPAM_DIR / SPAM_FILE
    if not src.exists():
        print(MAPSPAM_HELP)
        sys.exit(1)

    crops_data = {}
    crops = []
    total = np.zeros((SHAPE_5MIN[0] // BLOCK, SHAPE_5MIN[1] // BLOCK), dtype=np.float64)

    with zipfile.ZipFile(src) as zf:
        names = sorted(n for n in zf.namelist() if ALL_TECH.search(n))
        assert names, f"no all-technology GeoTIFFs found; file names look like {zf.namelist()[:5]}"
        for n in names:
            with rasterio.open(f"zip://{src}!{n}") as r:
                assert (r.height, r.width) == SHAPE_5MIN, f"{n}: unexpected shape {r.shape}"
                assert abs(r.bounds.left + 180) < 1e-6 and abs(r.bounds.top - 90) < 1e-6, f"{n}: not global ({r.bounds})"
                a = r.read(1, masked=True).filled(0).astype(np.float64)
            a_clean = np.where(np.isfinite(a) & (a > 0), a, 0)

            # downsample 5-arcmin -> 0.5 degree
            half = a_clean.reshape(SHAPE_5MIN[0] // BLOCK, BLOCK, SHAPE_5MIN[1] // BLOCK, BLOCK).sum(axis=(1, 3))

            crop_name = ALL_TECH.search(n).group(1)
            crops_data[crop_name] = half
            crops.append(crop_name)
            total += half

    assert len(crops) == len(set(crops)) == 46, f"expected 46 crops (MapSPAM readme), found {len(crops)}"
    assert total.shape == (360, 720)

    mha = total.sum() / 1e6
    # Global physical crop area is about 1.2-1.3 billion ha.
    assert 800 < mha < 2000, f"global cropland {mha:.0f} Mha is outside the plausible range"

    OUT_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_NPY, total)
    np.savez_compressed(OUT_NPZ, **crops_data)

    print(f"crops summed: {len(crops)} {crops}")
    print(f"global physical crop area: {mha:.0f} Mha; cells with cropland: {(total > 0).sum()} of {total.size}")
    print(f"Saved: {OUT_NPY} (total) and {OUT_NPZ} (per-crop)")


if __name__ == "__main__":
    main()

