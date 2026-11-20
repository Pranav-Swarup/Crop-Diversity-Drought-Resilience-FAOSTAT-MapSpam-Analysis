"""Pin Google Cloud object generations for every hourly ERA5 chunk used in the study."""

import base64
import csv
import gzip
import hashlib
from datetime import date
from functools import lru_cache
from urllib.parse import urlsplit

import requests

from .storage import BASE

LEDGER = BASE / "era5_source_objects.csv.gz"


@lru_cache(maxsize=1)
def pinned():
    if not LEDGER.exists():
        return {}
    with gzip.open(LEDGER, "rt") as f:
        return {(r["archive"], r["key"]): r for r in csv.DictReader(f)}


def checked_bytes(content, record):
    actual = base64.b64encode(hashlib.md5(content).digest()).decode()
    if actual != record["md5_base64"]:
        raise ValueError(f"ERA5 object checksum mismatch: {record['archive']} {record['key']}")
    return content


def create():
    from .era5 import AR, AR_START, FIRST, OVERLAP_DAY, WB, WB_START

    first_native = date(2023, 1, 1)
    blocks = {
        "weatherbench": (
            WB,
            range((FIRST - WB_START).days // 2, (OVERLAP_DAY - WB_START).days // 2 + 1),
        ),
        "arco": (
            AR,
            range((first_native - AR_START).days * 24, (date(2024, 1, 1) - AR_START).days * 24),
        ),
    }
    rows = []
    existing = pinned()
    for archive, (base, ids) in blocks.items():
        bucket, prefix = urlsplit(base).path.strip("/").split("/", 1)
        parent = prefix + "/2m_temperature/"
        endpoint = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o"
        expected = set(ids)
        for idx in ids:
            previous = existing.get((archive, f"{idx}.0.0"))
            if previous:
                rows.append(previous)
                expected.remove(idx)
        for length in sorted({len(str(i)) for i in expected}):
            numbers = [i for i in expected if len(str(i)) == length]
            params = {
                "prefix": parent,
                "startOffset": parent + f"{min(numbers)}.0.0",
                "endOffset": parent + f"{max(numbers)}.0.1",
                "fields": "nextPageToken,items(name,generation,md5Hash,timeCreated,size)",
                "maxResults": 1000,
            }
            while True:
                response = requests.get(endpoint, params=params, timeout=45)
                response.raise_for_status()
                result = response.json()
                for obj in result.get("items", []):
                    key = obj["name"].removeprefix(parent)
                    if not key.endswith(".0.0"):
                        continue
                    n = int(key.split(".")[0])
                    if n not in expected or len(str(n)) != length:
                        continue
                    rows.append(
                        dict(
                            archive=archive,
                            key=key,
                            generation=obj["generation"],
                            md5_base64=obj["md5Hash"],
                            bytes=int(obj["size"]),
                            created_utc=obj["timeCreated"],
                        )
                    )
                    expected.remove(n)
                if "nextPageToken" not in result:
                    break
                params["pageToken"] = result["nextPageToken"]
        if expected:
            raise ValueError(f"Missing {len(expected)} hourly source objects in {archive}")
    tmp = LEDGER.with_suffix(".tmp")
    # Suppress timestamp and filename in the gzip header for reproducibility.
    import io

    with (
        tmp.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as gz,
        io.TextIOWrapper(gz, newline="") as f,
    ):
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["archive"], int(r["key"].split(".")[0]))))
    tmp.replace(LEDGER)
    pinned.cache_clear()
    print(f"Pinned {len(rows)} source objects ({sum(int(r['bytes']) for r in rows) / 1e9:.2f} GB).")


if __name__ == "__main__":
    create()
