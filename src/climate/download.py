"""Download SPEIbase v2.10 (SPEI-6, SPEI-12) and Natural Earth admin-0; check the MapSPAM file.

MapSPAM is behind a Harvard Dataverse guestbook form, so it cannot be scripted. Download
`spam2020V2r2_global_physical_area.geotiff.zip` by hand into data/raw/mapspam/ (see MAPSPAM_HELP).

Run: python -m src.climate.download
"""
import sys

from ..common import paths
from ..common.sources import download, log_source

SPEI_DIR = paths.RAW / "spei"
NE_DIR = paths.RAW / "naturalearth"
SPAM_DIR = paths.RAW / "mapspam"

SPEI_DOI = "10.20350/digitalCSIC/16497"
SPEI_URL = "https://digital.csic.es/bitstream/10261/364137/{n}/spei{nn}.nc"
NE_URL = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip"
SPAM_DOI = "10.7910/DVN/SWPENT"
SPAM_FILE = "spam2020V2r2_global_physical_area.geotiff.zip"
MAPSPAM_HELP = f"""
MapSPAM file missing: data/raw/mapspam/{SPAM_FILE}
Harvard Dataverse asks for a guestbook form before download, so this step is manual:
  1. Open https://doi.org/{SPAM_DOI}
  2. In the file list, folder Global_Geotiff, download {SPAM_FILE} (about 66 MB)
     and Readme_SPAM2020V2r2.txt
  3. Put both in data/raw/mapspam/ (do not unzip) and re-run this script.
"""


def main():
    for n in (6, 12):
        download(f"SPEIbase v2.10 SPEI-{n}", SPEI_URL.format(n=n, nn=f"{n:02d}"), SPEI_DIR / f"spei{n:02d}.nc",
                 doi=SPEI_DOI, note="Global 0.5 degree monthly netCDF, 1901-2023 (CSIC)")
    download("Natural Earth 1:10m admin-0 countries", NE_URL, NE_DIR / "ne_10m_admin_0_countries.zip",
             note="Version 5.1.1")

    spam = SPAM_DIR / SPAM_FILE
    if not spam.exists():
        print(MAPSPAM_HELP)
        sys.exit(1)
    assert spam.stat().st_size > 5e7, f"{SPAM_FILE}: file looks truncated"
    log_source("MapSPAM 2020 v2 (release 2) global physical area", f"https://doi.org/{SPAM_DOI}", spam,
               doi=SPAM_DOI, note="Downloaded manually (Dataverse guestbook). GeoTIFFs, 5 arcmin.")
    print("\nSPEI files: 2, Natural Earth: 1, MapSPAM: 1")


if __name__ == "__main__":
    main()
