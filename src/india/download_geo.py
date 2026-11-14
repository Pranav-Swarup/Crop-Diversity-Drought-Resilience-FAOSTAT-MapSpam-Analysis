"""Download the geographic inputs for district SPEI (phase 4): India district polygons and SPEI-12.

- geoBoundaries gbOpen India ADM2 (districts) and ADM1 (states), boundary year 2021, 736 districts, source lgdirectory.gov.in
  (Pathways Data Pvt. Ltd.), ODbL 1.0. http://www.geoboundaries.org/ . Used because the ICRISAT record ships no polygons; this is
  a different source from the design and needs team agreement. DES codes (2022-23) and these 2021 polygons are matched by name.
- SPEIbase v2.10 SPEI-12 (same file as src.climate.download; skipped if already there). DOI 10.20350/digitalCSIC/16497.
MapSPAM (cropland weights) is NOT downloaded here: it sits behind a Dataverse guestbook form (see src.climate.download).

Run: python -m src.india.download_geo
"""
import requests

from ..common import paths
from ..common.sources import download
from ..climate.download import SPEI_DIR, SPEI_DOI, SPEI_URL

GEO_DIR = paths.RAW / "india_geo"
API = "https://www.geoboundaries.org/api/current/gbOpen/IND/{lvl}/"


def main():
    for lvl in ("ADM2", "ADM1"):
        meta = requests.get(API.format(lvl=lvl), timeout=60).json()
        assert meta["boundaryISO"] == "IND" and meta["boundaryType"] == lvl
        download(f"geoBoundaries gbOpen India {lvl}", meta["gjDownloadURL"], GEO_DIR / f"geoBoundaries-IND-{lvl}.geojson",
                 note=f"Boundary year {meta['boundaryYearRepresented']}, {meta['admUnitCount']} units, source {meta['boundarySource']}, "
                      f"licence {meta['boundaryLicense']}. Used for district SPEI because ICRISAT ships no polygons; not the design source.")
    download("SPEIbase v2.10 SPEI-12", SPEI_URL.format(n=12, nn="12"), SPEI_DIR / "spei12.nc", doi=SPEI_DOI,
             note="Global 0.5 degree monthly netCDF, 1901-2023 (CSIC)")


if __name__ == "__main__":
    main()
