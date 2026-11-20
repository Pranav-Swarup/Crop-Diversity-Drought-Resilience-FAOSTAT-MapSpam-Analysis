"""Source downloads and extension-only output paths. Author: Pmjarvis."""

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile

import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "future_work"
RAW = BASE / "local" / "raw"
CACHE = BASE / "local" / "cache"
RESULT = BASE / "results"
FIG = BASE / "figures"
for directory in (RAW, CACHE, RESULT, FIG):
    directory.mkdir(parents=True, exist_ok=True)


def fetch(url, path, checksum=None):
    """Download atomically. Resume an interrupted transfer only on a matching HTTP range."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def matches(p):
        if not checksum:
            if not (BASE / "source_files.csv.gz").exists():
                return True
            from .provenance import source_digest

            expect = source_digest(path)
            if expect is None:
                return True
            algorithm = "sha256"
        else:
            algorithm, expect = checksum.split(":")
        with p.open("rb") as f:
            return hashlib.file_digest(f, algorithm).hexdigest() == expect

    if path.is_file() and matches(path):
        return path
    tmp = path.with_suffix(path.suffix + ".part")
    # Restore the historical catalogue itself when current inventory pages have moved on.
    # File objects and measurement payloads still use their original sources and fingerprints.
    catalogues = BASE / "source_catalogues.zip"
    if catalogues.is_file() and path.is_relative_to(RAW):
        key = path.relative_to(RAW).as_posix()
        with ZipFile(catalogues) as archive:
            if key in archive.namelist():
                tmp.write_bytes(archive.read(key))
                if not matches(tmp):
                    raise ValueError(f"Stored source catalogue differs from its digest: {key}")
                tmp.replace(path)
                return path
    for attempt in range(6):
        try:
            offset = tmp.stat().st_size if tmp.exists() else 0
            headers = {"Accept-Encoding": "identity"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            with requests.get(url, headers=headers, stream=True, timeout=(15, 90)) as resp:
                if resp.status_code == 416 and offset:
                    tmp.unlink()
                    raise IOError("Server rejected partial file")
                resp.raise_for_status()
                if resp.status_code == 206:
                    cr = resp.headers.get("Content-Range", "")
                    if not cr.startswith(f"bytes {offset}-"):
                        raise IOError(f"Invalid HTTP range {cr}")
                else:
                    offset = 0
                with tmp.open("ab" if offset else "wb") as out:
                    for chunk in resp.iter_content(1024 * 1024):
                        out.write(chunk)
                total = resp.headers.get("Content-Range", "").split("/")[-1]
                expected = (
                    int(total)
                    if total.isdigit()
                    else offset + int(resp.headers.get("Content-Length", 0))
                )
                # Some archive replays serve a gzip entity even when identity was requested.
                # iter_content transparently decodes it; its stored length is then the decoded size.
                encoded = resp.headers.get("Content-Encoding", "identity").lower() != "identity"
                if not encoded and expected > offset and tmp.stat().st_size != expected:
                    raise IOError(f"Incomplete download of {path.name}")
            if not matches(tmp):
                tmp.unlink()
                raise IOError(f"Checksum mismatch for {path.name}")
            tmp.replace(path)
            print(f"Downloaded {path.name} ({path.stat().st_size:,} bytes)", flush=True)
            return path
        except (OSError, requests.RequestException) as e:
            if attempt == 5:
                raise
            print(f"Retry {path.name}: {e}", flush=True)
            time.sleep(min(15, 2**attempt))


def spei_path(scale):
    name = f"spei{scale:02d}.nc"
    frozen = ROOT / "data/raw/spei" / name
    if frozen.exists():
        return frozen
    url = f"https://digital.csic.es/bitstream/10261/364137/{scale}/{name}"
    return fetch(url, RAW / "spei" / name)


def calendars():
    for record, target in ((5062513, "calendar"), (3773827, "wheat")):
        meta = fetch(f"https://zenodo.org/api/records/{record}", RAW / target / "metadata.json")
        import json

        files = json.loads(meta.read_text())["files"]
        if target == "wheat":
            files = [f for f in files if f["key"] == "winter_and_spring_wheat_areas_v1_180627.nc4"]
        else:
            files = [f for f in files if f["key"].endswith(".nc4")]
        if not files:
            raise ValueError(f"No expected source files in Zenodo record {record}")

        def get(f):
            name = f["key"]
            url = f"https://zenodo.org/records/{record}/files/{name}?download=1"
            return fetch(url, RAW / target / name, f["checksum"])

        with ThreadPoolExecutor(4) as pool:
            list(pool.map(get, files))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("source", choices=["calendars", "spei"])
    parser.add_argument("scales", type=int, nargs="*")
    args = parser.parse_args()
    if args.source == "calendars":
        calendars()
    else:
        with ThreadPoolExecutor(3) as pool:
            list(pool.map(spei_path, args.scales))
