"""Reconcile the complete DES land-use CSV against the India Data Portal's paginated datastore. Author: Pmjarvis."""

import csv
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from urllib.parse import urlencode

import requests

from .storage import BASE, RAW

SOURCE = "https://ckandev.indiadataportal.com/api/3/action/datastore_search"
RESOURCE = "0fb99c18-a1f7-46f0-b40d-ddffca021319"
FIELDS = [
    "id",
    "year",
    "state_name",
    "state_code",
    "district_name",
    "district_code",
    "food_category",
    "crop_type",
    "sub_crop_type",
    "crop_name",
    "crop_code",
    "season",
    "area",
]
NUMERIC = {"id", "state_code", "district_code", "crop_code", "area"}
LIMIT = 10000
PAGES = BASE / "district_land_use_pages.csv"
CACHE = RAW / "icrisat_archive" / "land_use_datastore"


def canonical(row):
    values = []
    for field, value in zip(FIELDS, row, strict=True):
        s = str(value)
        if value is None or s in ("", "nan"):
            values.append(None)
        elif field in NUMERIC:
            n = Decimal(s)
            if not n.is_finite():
                raise ValueError(f"Nonfinite DES number in {field}")
            values.append(format(n.normalize(), "f"))
        else:
            values.append(s)
    return (json.dumps(values, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def query(offset):
    return (
        SOURCE
        + "?"
        + urlencode(
            {
                "resource_id": RESOURCE,
                "fields": ",".join(FIELDS),
                "limit": LIMIT,
                "offset": offset,
                "sort": "_id asc",
                "records_format": "lists",
            }
        )
    )


def read_page(offset, total):
    path = CACHE / f"{offset}.json"
    if not path.is_file():
        for attempt in range(4):
            try:
                with requests.get(query(offset), timeout=(15, 70)) as response:
                    response.raise_for_status()
                    data = response.content
                if not json.loads(data).get("success"):
                    raise ValueError(f"Datastore rejected offset {offset}")
                tmp = path.with_suffix(".tmp")
                tmp.write_bytes(data)
                tmp.replace(path)
                break
            except (OSError, ValueError):
                if attempt == 3:
                    raise
                time.sleep(2**attempt)
    raw = path.read_bytes()
    data = json.loads(raw, parse_float=Decimal)["result"]
    rows = data["records"]
    if (
        data["total"] != total
        or len(rows) != min(LIMIT, total - offset)
        or [f["id"] for f in data["fields"]] != FIELDS
    ):
        raise ValueError(f"Datastore schema or page length changed at {offset}")
    digest = hashlib.sha256()
    for i, row in enumerate(rows, start=offset):
        if int(row[0]) != i:
            raise ValueError(f"Datastore row id differs at {i}")
        digest.update(canonical(row))
    return {
        "offset": offset,
        "rows": len(rows),
        "canonical_sha256": digest.hexdigest(),
        "receipt_sha256": hashlib.sha256(raw).hexdigest(),
        "url": query(offset),
    }


def compare_export(path, pages):
    total = sum(int(p["rows"]) for p in pages)
    records = [None] * total
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        if next(reader) != FIELDS:
            raise ValueError("Unexpected DES land-use column names or order")
        for row in reader:
            key = int(row[0])
            if not 0 <= key < total or records[key] is not None:
                raise ValueError(f"Unexpected or repeated DES record ID {key}")
            records[key] = canonical(row)
    count = 0
    for page in pages:
        if int(page["offset"]) != count:
            raise ValueError("Missing or overlapping datastore page")
        checksum = hashlib.sha256()
        for i in range(count, count + int(page["rows"])):
            if records[i] is None:
                raise ValueError(f"Missing DES record {i}")
            checksum.update(records[i])
        if checksum.hexdigest() != page["canonical_sha256"]:
            raise ValueError(f"DES file and datastore fields differ from offset {count}")
        count += int(page["rows"])
    return count


def verify_export(path):
    with PAGES.open(newline="") as f:
        return compare_export(path, list(csv.DictReader(f)))


def audit():
    from .district_archive import ARCHIVE, PINS
    from .storage import fetch

    spec = json.loads(PINS.read_text())["des_irrigation"]
    path = fetch(spec["source_url"], ARCHIVE / spec["file"], "sha256:" + spec["sha256"])
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        total = sum(1 for row in reader)
    CACHE.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(3) as pool:
        pages = list(pool.map(lambda n: read_page(n, total), range(0, total, LIMIT)))
    count = compare_export(path, pages)
    if PAGES.exists():
        with PAGES.open(newline="") as f:
            previous = [p["canonical_sha256"] for p in csv.DictReader(f)]
        if previous != [p["canonical_sha256"] for p in pages]:
            raise ValueError("Verified records differ from the saved edition")
    else:
        tmp = PAGES.with_suffix(".tmp")
        with tmp.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(pages[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(pages)
        tmp.replace(PAGES)
    print(
        f"Verified {count:,} records, {len(FIELDS) * count:,} fields and {len(pages)} datastore pages.",
        flush=True,
    )


if __name__ == "__main__":
    audit()
