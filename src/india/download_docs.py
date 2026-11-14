"""Download the documentation files of the official ICRISAT DLD record (ICRISAT Dataverse) into data/raw/icrisat/docs/.

Source: Swamikannu et al. 2021, "District Level Data (DLD) and Dashboard for Agriculture and Allied-sectors in India",
        ICRISAT Dataverse, V1, doi:10.21421/D2/XFB1BZ
        https://dataverse.icrisat.org/dataset.xhtml?persistentId=doi:10.21421/D2/XFB1BZ
The record holds documentation only (no data files); the data sits on data.icrisat.org/dld.
Terms: ICRISAT Data Sharing Agreement, http://data.icrisat.org/document/ICRISAT-Data-Sharing-Agreement.html

Run: python -m src.india.download_docs
"""
import requests

from ..common import paths
from ..common.sources import download

HOST = "https://dataverse.icrisat.org"
DOI = "10.21421/D2/XFB1BZ"
DIR = paths.RAW / "icrisat" / "docs"


def main():
    meta = requests.get(f"{HOST}/api/datasets/:persistentId/", params={"persistentId": f"doi:{DOI}"}, timeout=60).json()
    files = meta["data"]["latestVersion"]["files"]
    assert files, "no files in the Dataverse record"
    for f in files:
        assert not f.get("restricted"), f"{f['dataFile']['filename']} is restricted; log in and download by hand"
        df = f["dataFile"]
        dest = download(f"ICRISAT DLD documentation: {df['filename']}", f"{HOST}/api/access/datafile/{df['id']}",
                        DIR / df["filename"], doi=DOI,
                        note="Official ICRISAT Dataverse record (documentation only). Terms: ICRISAT Data Sharing Agreement.")
        assert dest.stat().st_size == df["filesize"], f"{df['filename']}: size differs from Dataverse metadata"
    print(f"\nICRISAT DLD documentation: {len(files)} files in {DIR}")


if __name__ == "__main__":
    main()
