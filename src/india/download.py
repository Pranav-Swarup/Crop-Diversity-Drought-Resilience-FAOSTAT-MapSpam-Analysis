"""Download the Mendeley Data mirror of ICRISAT district-level data into data/raw/icrisat/.

THIS IS A THIRD-PARTY MIRROR, NOT THE OFFICIAL ICRISAT DLD (data.icrisat.org/dld was unreachable
on 2026-11-14). Used with the team member's approval as a fallback. Differences from the design
in docs/phase4-design.md: author-merged unapportioned panel, 560 districts, 1990-2015, one file.
Source: Mohapatra, S., Mendeley Data, V1, doi:10.17632/ywp3y5j9vv.1
        https://data.mendeley.com/datasets/ywp3y5j9vv/1
The file's SHA-256 is checked against the hash published by Mendeley.

Run: python -m src.india.download
"""
import hashlib

import requests

from ..common import paths
from ..common.sources import download

API = "https://data.mendeley.com/public-api/datasets/ywp3y5j9vv/files?folder_id=root&version=1"
DIR = paths.RAW / "icrisat"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    files = requests.get(API, timeout=60).json()
    assert len(files) == 1, f"expected 1 file in the Mendeley record, found {[f['filename'] for f in files]}"
    f = files[0]
    cd = f["content_details"]
    dest = download("ICRISAT district-level data (Mendeley mirror, unapportioned)",
                    cd["download_url"], DIR / f["filename"], doi="10.17632/ywp3y5j9vv.1",
                    note="THIRD-PARTY MIRROR of ICRISAT DLD, not the official source. "
                         "Author-merged unapportioned panel, 560 districts, 1990-2015. "
                         "Licence/terms not stated on the Mendeley record; official terms: ICRISAT Data Sharing Agreement.")
    assert dest.stat().st_size == cd["size"], "size differs from Mendeley metadata"
    assert sha256(dest) == cd["sha256_hash"], "SHA-256 differs from Mendeley metadata"
    print(f"\nICRISAT mirror: {dest.name}, {dest.stat().st_size / 1e6:.1f} MB, sha256 verified")


if __name__ == "__main__":
    main()
