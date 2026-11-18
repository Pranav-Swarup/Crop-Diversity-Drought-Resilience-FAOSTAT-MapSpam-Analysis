"""Download the pinned India district boundaries and both district climate inputs.

geoBoundaries revision 9469f09 supplies ADM2 districts (2021) and ADM1 states
(2011). The district climate variables use SPEIbase v2.10 SPEI-4 and SPEI-12
and the MapSPAM cropland weights prepared by src.climate.cropland.

Run: python -m src.india.download_geo
"""


from ..common import paths
from ..common.sources import download
from ..climate.download import SPEI_DIR, SPEI_DOI, SPEI_URL

GEO_DIR = paths.RAW / "india_geo"
RELEASE = "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/IND"
BOUNDARIES = {
    "ADM1": "2011, 36 units, Creative Commons Attribution 2.5 India",
    "ADM2": "2021, 736 units, Open Data Commons Open Database License 1.0",
}


def main():
    for level, metadata in BOUNDARIES.items():
        name = f"geoBoundaries-IND-{level}.geojson"
        download(f"geoBoundaries gbOpen India {level}", f"{RELEASE}/{level}/{name}",
                 GEO_DIR / name, note=f"geoBoundaries revision 9469f09, {metadata}.")
    for scale in (4, 12):
        fname = f"spei{scale:02d}.nc"
        download(f"SPEIbase v2.10 SPEI-{scale}", SPEI_URL.format(n=scale, nn=f"{scale:02d}"),
                 SPEI_DIR / fname, doi=SPEI_DOI, note="CSIC SPEIbase v2.10 monthly netCDF")


if __name__ == "__main__":
    main()
