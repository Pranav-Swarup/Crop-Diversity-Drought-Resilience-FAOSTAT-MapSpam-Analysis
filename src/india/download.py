"""Fetch the version-one Mendeley mirror workbook and verify its published SHA-256.

Mohapatra, Mendeley Data, doi:10.17632/ywp3y5j9vv.1 (CC BY 4.0).
Run with python -m src.india.download.
"""
import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile

import requests

from ..common import paths
from ..common.sources import log_source

DIR = paths.RAW / "icrisat"
FILENAME = "main merge (droped _merge==2) (560 dist 1990-2015).xls"
FILE_ID = "e3dfd63e-abf7-4417-9331-27c574e035b9"
SIZE = 23739904
SHA256 = "e972b837b29298b8f06cda6fe7ea982d90a7a97a66e9a07fc72df3b66680318d"
FILE_PATH = f"/public-files/datasets/ywp3y5j9vv/files/{FILE_ID}/file_downloaded"
URLS = [f"https://{host}{FILE_PATH}" for host in ("data.mendeley.com", "elsevier.digitalcommonsdata.com")]


def verified(path):
    if not path.is_file() or path.stat().st_size != SIZE:
        return False
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest() == SHA256


def main():
    DIR.mkdir(parents=True, exist_ok=True)
    dest = DIR / FILENAME
    if not verified(dest):
        for url in URLS:
            with NamedTemporaryFile(dir=DIR, delete=False) as handle:
                staged = Path(handle.name)
            try:
                with requests.get(url, stream=True, timeout=120) as response:
                    response.raise_for_status()
                    with staged.open("wb") as handle:
                        for chunk in response.iter_content(1 << 20):
                            handle.write(chunk)
                if not verified(staged):
                    raise ValueError("Downloaded workbook differs from the version-one publisher checksum")
                staged.replace(dest)
                break
            except (requests.RequestException, ValueError):
                if url == URLS[-1]:
                    raise
            finally:
                staged.unlink(missing_ok=True)
    log_source("ICRISAT district-level data (Mendeley mirror, unapportioned)", URLS[0], dest,
               doi="10.17632/ywp3y5j9vv.1", note=f"CC BY 4.0. Version 1, SHA-256 {SHA256}.")
    print(f"Verified workbook: {dest.name} ({SIZE:,} bytes)")


if __name__ == "__main__":
    main()
