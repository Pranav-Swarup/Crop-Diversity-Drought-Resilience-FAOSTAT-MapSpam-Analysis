"""Download helper that logs every source to data/raw/SOURCES.md."""
from datetime import date

import requests

from . import paths


def log_source(name, url, file, doi="", note=""):
    """Print a source and record it in data/raw/SOURCES.md (one section per name, replaced on re-run)."""
    paths.RAW.mkdir(parents=True, exist_ok=True)
    rel = file.relative_to(paths.ROOT) if file.is_absolute() else file
    entry = [f"## {name}", f"- URL: {url}"]
    if doi:
        entry.append(f"- DOI: {doi}")
    entry += [f"- File: `{rel}`", f"- Retrieved: {date.today().isoformat()}"]
    if note:
        entry.append(f"- Note: {note}")
    entry = "\n".join(entry)

    text = paths.SOURCES_MD.read_text() if paths.SOURCES_MD.exists() else "# Data sources\n"
    sections = text.split("\n## ")
    kept = [s for s in sections[1:] if s.split("\n", 1)[0].strip() != name]
    out = sections[0].rstrip("\n") + "\n\n" + "\n\n".join(["## " + s.rstrip("\n") for s in kept] + [entry]) + "\n"
    paths.SOURCES_MD.write_text(out)
    print(f"source: {name}\n  url: {url}" + (f"\n  doi: {doi}" if doi else "") + f"\n  file: {rel}")


def download(name, url, dest, doi="", note=""):
    """Stream `url` to `dest` unless a complete copy is already there, then log the source."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        size = int(r.headers.get("content-length", 0))
        if dest.exists() and size and dest.stat().st_size == size:
            print(f"already downloaded: {dest.name} ({size / 1e6:.1f} MB)")
        else:
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
            assert not size or tmp.stat().st_size == size, f"{dest.name}: incomplete download"
            tmp.rename(dest)
            print(f"downloaded: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
    log_source(name, url, dest, doi=doi, note=note)
    return dest
