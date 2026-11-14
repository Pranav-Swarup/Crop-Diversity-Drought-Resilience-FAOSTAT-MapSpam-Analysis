"""Download the district-level crop Area, Production, Yield (APY) CSV of India (Ministry of Agriculture, DES) into data/raw/india_apy/.

ALTERNATIVE TO THE ICRISAT DLD, used because data.icrisat.org/dld was unreachable on 2026-11-14 (explicit decision of
the team member running this; the team has not yet agreed). It is a different source from docs/phase4-design.md:
Directorate of Economics and Statistics (DES), Ministry of Agriculture & Farmers Welfare, as republished on the
India Data Portal CKAN instance (a "ckandev" host, so not the Ministry's own server).
Package: https://ckandev.indiadataportal.com/dataset/area-production-yield-apy
Licence: Open Data Commons Attribution License (as listed by the portal).
No checksum is published, so the SHA-256 is recorded in SOURCES.md for later comparison.

Run: python -m src.india.download_apy
"""
import hashlib

import requests

from ..common import paths
from ..common.sources import download, log_source

API = "https://ckandev.indiadataportal.com/api/3/action/package_show?id=area-production-yield-apy"
DIR = paths.RAW / "india_apy"
WANT = {"Crop Wise Area Production Yield": "crop_wise_apy_des.csv",
        "Crop-wise Area, Production & Yield (UPAG)": "crop_wise_apy_upag.csv"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    pkg = requests.get(API, timeout=60).json()
    assert pkg["success"], "CKAN package_show failed"
    res = {r["name"]: r for r in pkg["result"]["resources"]}
    for name, fname in WANT.items():
        assert name in res, f"resource missing from the portal: {name}"
        r = res[name]
        dest = download(f"India district APY, {name} (India Data Portal mirror of DES)", r["url"], DIR / fname,
                        note="ALTERNATIVE to ICRISAT DLD, not the design source. DES / Ministry of Agriculture data on the "
                             "India Data Portal CKAN dev instance. Licence: ODC Attribution. "
                             f"Portal metadata_modified {pkg['result']['metadata_modified']}.")
        assert dest.stat().st_size > 1e6, f"{fname} too small"
        digest = sha256(dest)
        log_source(f"India district APY, {name} (India Data Portal mirror of DES)", r["url"], dest,
                   note=f"SHA-256 {digest}. ALTERNATIVE to ICRISAT DLD, not the design source. DES / Ministry of Agriculture "
                        "data on the India Data Portal CKAN dev instance. Licence: ODC Attribution.")
        print(f"  {fname}: {dest.stat().st_size / 1e6:.1f} MB, sha256 {digest[:16]}...")


if __name__ == "__main__":
    main()
