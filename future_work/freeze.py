"""Capture the presented working tree once and verify protected bytes on every run."""

import argparse
import csv
import hashlib
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOME = ROOT / "future_work"
MANIFEST = HOME / "baseline.csv"
SNAPSHOT = HOME / "local" / "baseline"
DECK = "Crop Diversity and Drought Resilience (updated) (1).pdf"
DECK_SHA = "493dadbcb4c0a70055ea6bbf0aac589e230881377fe033c3c9a5a60dc8206c58"
# These four captured drafts belong only to the historical snapshot.
RETIRED_SLIDES = {
    "Crop Diversity and Drought Resilience (updated).pdf",
    "Crop Diversity and Drought Resilience Phases 2–4.pptx",
    "data/clean/Crop Diversity and Drought Resilience (1).pdf",
    "data/clean/Crop Diversity and Drought Resilience.pptx",
}
# Generated baseline guides travel with their assets. The current documentation index covers them.
GENERATED = {"figures/deck/INDEX.md", "outputs/dummy/README.md", "data/dummy/README.md"}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory():
    raw = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    files = {Path(p.decode()) for p in raw.split(b"\0") if p}
    for name in ("data", "outputs", "figures", "explorer", "src", "tests", "references"):
        files.update(p.relative_to(ROOT) for p in (ROOT / name).rglob("*") if p.is_file())
    files.update(p.relative_to(ROOT) for pat in ("*.pptx", "*.pdf") for p in ROOT.glob(pat))
    return sorted(
        p
        for p in files
        if p.parts[0] != "future_work"
        and "__pycache__" not in p.parts
        and ".pytest_cache" not in p.parts
        and p.suffix != ".pyc"
    )


def capture():
    if MANIFEST.exists():
        raise SystemExit("A baseline manifest already exists. Use verify.")
    if digest(ROOT / DECK) != DECK_SHA:
        raise SystemExit("The presented PDF differs from the verified slide deck.")
    rows = []
    for path in inventory():
        source, dest = ROOT / path, SNAPSHOT / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        h = digest(source)
        if digest(dest) != h:
            raise RuntimeError(f"Snapshot copy mismatch: {path}")
        editable = (
            path.suffix == ".md"
            and path.as_posix() not in GENERATED
            and path.parts[:2] != ("data", "raw")
        )
        rows.append(
            (path.as_posix(), source.stat().st_size, h, "documentation" if editable else "frozen")
        )
    with MANIFEST.with_suffix(".tmp").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "bytes", "sha256", "policy"])
        writer.writerows(rows)
    MANIFEST.with_suffix(".tmp").replace(MANIFEST)
    print(f"Snapshotted and verified {len(rows)} files at {SNAPSHOT}.", flush=True)


def verify(snapshot=False):
    with MANIFEST.open() as f:
        rows = list(csv.DictReader(f))
    source = SNAPSHOT if snapshot else ROOT
    errors = []
    checked = 0
    for row in rows:
        if not snapshot and row["policy"] == "documentation":
            continue
        path = source / row["path"]
        if not snapshot and row["path"] in RETIRED_SLIDES:
            if path.exists() or path.is_symlink():
                errors.append(row["path"])
            continue
        if (
            not path.is_file()
            or path.stat().st_size != int(row["bytes"])
            or digest(path) != row["sha256"]
        ):
            errors.append(row["path"])
        checked += 1
    if not snapshot:
        known = {r["path"] for r in rows}
        errors.extend(
            str(p)
            for p in inventory()
            if str(p) not in known
            and p.parts[0] in {"data", "outputs", "figures", "explorer", "src", "tests", "references"}
        )
    if errors:
        raise SystemExit("Baseline mismatch: " + ", ".join(errors))
    print(f"Verified {checked} files under {source}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["capture", "verify", "verify-snapshot"])
    action = parser.parse_args().action
    if action == "capture":
        capture()
    else:
        verify(snapshot=action == "verify-snapshot")
