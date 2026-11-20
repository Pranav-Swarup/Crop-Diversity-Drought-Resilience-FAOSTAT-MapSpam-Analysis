"""Content-addressed sources and generated artifacts of the post-presentation studies."""

import csv
import gzip
import io
from functools import lru_cache
from urllib.parse import quote

from .freeze import digest
from .storage import BASE, FIG, RAW, RESULT, ROOT
from .vegetation import MENU, series_url
from .vegetation_grid import ARCHIVE

SOURCE_PINS = BASE / "source_files.csv.gz"
OUTPUT_PINS = BASE / "outputs.csv"


@lru_cache(maxsize=1)
def _source_hashes():
    with gzip.open(SOURCE_PINS, "rt") as src:
        return {str(ROOT / row["path"]): row["sha256"] for row in csv.DictReader(src)}


def source_digest(path):
    # A downloader passes the intended destination even when hashing its .part file.
    return _source_hashes().get(str(path.resolve()))


def source_paths():
    for sub, record in [("calendar", 5062513), ("wheat", 3773827)]:
        for path in sorted((RAW / sub).glob("*")):
            if path.name == "metadata.json":
                url = f"https://zenodo.org/api/records/{record}"
            elif path.suffix == ".nc4":
                url = f"https://zenodo.org/records/{record}/files/{quote(path.name)}?download=1"
            else:
                continue
            yield path, url
    for scale in range(1, 14):
        name = f"spei{scale:02d}.nc"
        local = ROOT / "data/raw/spei" / name
        yield (
            local if local.exists() else RAW / "spei" / name,
            f"https://digital.csic.es/bitstream/10261/364137/{scale}/{name}",
        )
    for path in sorted((RAW / "noaa").glob("*.html")):
        yield path, MENU if path.name == "country_crop_menu.html" else series_url(*path.stem.split("_"))
    for path in sorted((RAW / "noaa_grid").glob("*")):
        if path.name == "index.html":
            yield path, ARCHIVE
        elif path.suffix == ".nc":
            yield path, ARCHIVE + path.name


def write_sources():
    if SOURCE_PINS.exists():
        verify_files(SOURCE_PINS)
        return
    tmp = SOURCE_PINS.with_suffix(".tmp")
    with (
        tmp.open("wb") as raw,
        gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz,
        io.TextIOWrapper(gz, newline="") as out,
    ):
        writer = csv.writer(out)
        writer.writerow(["path", "source_url", "bytes", "sha256"])
        for path, url in source_paths():
            writer.writerow([path.relative_to(ROOT), url, path.stat().st_size, digest(path)])
    tmp.replace(SOURCE_PINS)


def write_outputs():
    tmp = OUTPUT_PINS.with_suffix(".tmp")
    with tmp.open("w", newline="") as out:
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(["path", "bytes", "sha256"])
        for path in sorted([*RESULT.glob("*"), *FIG.glob("*")]):
            if path.is_file():
                writer.writerow([path.relative_to(ROOT), path.stat().st_size, digest(path)])
    tmp.replace(OUTPUT_PINS)


def verify_files(ledger):
    opener = gzip.open if ledger.suffix == ".gz" else open
    with opener(ledger, "rt") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        path = ROOT / row["path"]
        if (
            not path.is_file()
            or path.stat().st_size != int(row["bytes"])
            or digest(path) != row["sha256"]
        ):
            raise ValueError(
                f"Listed file is missing or differs from its verified digest: {row['path']}"
            )
    if ledger == SOURCE_PINS:
        listed = {row["path"] for row in rows}
        actual = {str(p.relative_to(ROOT)) for p, _ in source_paths()}
        if listed != actual:
            raise ValueError(f"Source inventory differs: {sorted(listed ^ actual)}")
    if ledger == OUTPUT_PINS:
        actual = {str(p.relative_to(ROOT)) for d in (RESULT, FIG) for p in d.iterdir() if p.is_file()}
        if {row["path"] for row in rows} != actual:
            raise ValueError("Output inventory differs from outputs.csv")
    return len(rows)
