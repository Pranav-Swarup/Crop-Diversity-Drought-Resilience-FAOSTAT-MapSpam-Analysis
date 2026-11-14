"""Download CRU TS v4.10 monthly maximum temperature (tmx), 0.5 degree, for the heat-stress extension.

Four decadal NetCDF files covering 1991-2025 (the window 1993-2023 is cut in src.climate.heat).
Each `.nc.gz` is kept as downloaded and unpacked next to it as `.nc`.

Source: Harris et al. 2020, Sci Data 7, 109, https://doi.org/10.1038/s41597-020-0453-3
        https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_4.10/ (no login; Open Database License)

Run: python -m src.climate.download_cru
"""
import gzip
import shutil

from ..common import paths
from ..common.sources import download

CRU_DIR = paths.RAW / "cru"
CRU_VERSION = "4.10"
CRU_DOI = "10.1038/s41597-020-0453-3"
CRU_URL = "https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_4.10/cruts.2604091129.v4.10/tmx/{name}.gz"
PERIODS = ["1991.2000", "2001.2010", "2011.2020", "2021.2025"]


def nc_name(period):
    return f"cru_ts{CRU_VERSION}.{period}.tmx.dat.nc"


def nc_files():
    """Paths of the unpacked NetCDF files, in time order."""
    return [CRU_DIR / nc_name(p) for p in PERIODS]


def main():
    for period in PERIODS:
        name = nc_name(period)
        gz = download(f"CRU TS v{CRU_VERSION} tmx {period}", CRU_URL.format(name=name), CRU_DIR / f"{name}.gz",
                      doi=CRU_DOI, note="Monthly mean daily maximum temperature, 0.5 degree, deg C (UEA CRU)")
        nc = CRU_DIR / name
        if not nc.exists() or nc.stat().st_mtime < gz.stat().st_mtime:
            with gzip.open(gz, "rb") as src, open(nc, "wb") as dst:
                shutil.copyfileobj(src, dst)
            print(f"unpacked: {nc.name} ({nc.stat().st_size / 1e6:.1f} MB)")
    missing = [f.name for f in nc_files() if not f.exists()]
    assert not missing, f"CRU files missing: {missing}"
    print(f"\nCRU TS tmx files: {len(PERIODS)}")


if __name__ == "__main__":
    main()
