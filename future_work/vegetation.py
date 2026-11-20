"""Official NOAA crop-specific weekly VCI with country and crop identifiers checked against the product menu."""

import re
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from io import StringIO
from urllib.parse import urlencode

import pandas as pd

from .seasons import footprints
from .storage import RAW, RESULT, fetch

NOAA = "https://www.star.nesdis.noaa.gov/smcd/emb/vci/VH/"
MENU = NOAA + "vh_adminMeanByCrop.php?type=Country_Weekly_MeanPlot"
VERSION = "GC_current"


class Menus(HTMLParser):
    def __init__(self):
        super().__init__()
        self.current = None
        self.menus = {}

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "select":
            self.current = a.get("id")
            self.menus[self.current] = {}
        elif tag == "option" and self.current and "id" in a:
            self.menus[self.current][a["id"]] = a.get("value", "")

    def handle_endtag(self, tag):
        if tag == "select":
            self.current = None


def parse_series(text, iso, crop):
    header = f"Mean data for {iso} ,  from 1992 to 2023, weekly; version='{VERSION}'"
    if (
        header not in text
        or f"area with '{crop}'" not in text
        or "year,week, SMN,SMT,VCI,TCI, VHI" not in text
    ):
        raise ValueError(f"Unexpected NOAA header for {iso} {crop}")
    block = re.search(r"<pre>(.*?)</pre>", text, flags=re.S | re.I)
    if not block or not block.group(1).strip():
        return pd.DataFrame(columns=["year", "week", "vci"])
    df = pd.read_csv(
        StringIO(block.group(1)), header=None, usecols=[0, 1, 4], names=["year", "week", "vci"]
    )
    if (
        df.duplicated(["year", "week"]).any()
        or not df.week.between(1, 52).all()
        or not df.year.between(1992, 2023).all()
    ):
        raise ValueError(f"Invalid NOAA calendar for {iso} {crop}")
    df["vci"] = pd.to_numeric(df.vci, errors="coerce").where(lambda v: v.between(0, 100))
    return df


def series_url(iso, crop):
    args = dict(
        provinceID=0,
        country=iso,
        adminVHversion=VERSION,
        yearlyTag="Weekly",
        type="Mean",
        TagCropland=crop,
        year1=1992,
        year2=2023,
    )
    return NOAA + "get_TS_admin.php?" + urlencode(args)


def weekly():
    directory = RAW / "noaa"
    directory.mkdir(parents=True, exist_ok=True)
    menu = Menus()
    menu.feed(fetch(MENU, directory / "country_crop_menu.html").read_text())
    if VERSION not in menu.menus["AdminVH_Version"]:
        raise ValueError("Requested NOAA product version is absent")
    cells = footprints()
    pairs = cells[["iso3", "mapspam_crop"]].drop_duplicates().sort_values(["iso3", "mapspam_crop"])

    def one(pair):
        iso, spam = pair
        crop = "SMIL" if spam == "MILL" else spam
        # The service can return an apparent series even for an unknown three-letter code.
        # Only published country ids are valid, including when the repository uses a composite territory.
        if iso not in menu.menus["Country"] or crop not in menu.menus["CropList"]:
            return None, (iso, spam, "outside official menu", 0)
        url = series_url(iso, crop)
        f = fetch(url, directory / f"{iso}_{crop}.html")
        df = parse_series(f.read_text(), iso, crop)
        n = int(df.vci.notna().sum())
        return df.assign(iso3=iso, mapspam_crop=spam), (
            iso,
            spam,
            "observed" if n else "no valid VCI",
            n,
        )

    rows, coverage = [], []
    with ThreadPoolExecutor(6) as pool:
        for d, cov in pool.map(one, pairs.itertuples(index=False, name=None)):
            if d is not None and not d.empty:
                rows.append(d)
            coverage.append(cov)
    pd.DataFrame(coverage, columns=["iso3", "mapspam_crop", "status", "observed_weeks"]).to_csv(
        RESULT / "vci_source_coverage.csv", index=False
    )
    pd.concat(rows, ignore_index=True).to_parquet(RESULT / "vci_weekly.parquet", index=False)


if __name__ == "__main__":
    weekly()
