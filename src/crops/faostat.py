"""Readers for the FAOSTAT bulk zips in data/raw/faostat/, cut to the 1993-2023 window."""
import hashlib
import zipfile

import pandas as pd

from ..common.contract import YEAR_MAX, YEAR_MIN
from .download import FAO_DIR

QCL = "Production_Crops_Livestock_E"
QV = "Value_of_Production_E"
RL = "Inputs_LandUse_E"
COLS = ["Area Code", "Area Code (M49)", "Area", "Item Code", "Item", "Element Code", "Element", "Year", "Unit", "Value", "Flag"]
FIRST_AGGREGATE_AREA = 5000  # FAOSTAT area codes from 5000 up are regions and groupings


def read(domain, element_codes):
    """Rows of one domain for the given element codes, countries only, 1993-2023 (cached as parquet)."""
    cache = FAO_DIR / f"{domain}_{'_'.join(map(str, element_codes))}.parquet"
    source = FAO_DIR / f"{domain}_All_Data_(Normalized).zip"
    if cache.exists() and cache.stat().st_mtime >= source.stat().st_mtime:  # a re-downloaded zip invalidates the cache
        return pd.read_parquet(cache)
    with zipfile.ZipFile(source) as zf:
        chunks = []
        for ch in pd.read_csv(zf.open(f"{domain}_All_Data_(Normalized).csv"), usecols=COLS,
                              encoding="utf-8", chunksize=500_000):
            keep = (ch["Element Code"].isin(element_codes) & ch["Year"].between(YEAR_MIN, YEAR_MAX)
                    & (ch["Area Code"] < FIRST_AGGREGATE_AREA))
            chunks.append(ch[keep])
    df = pd.concat(chunks, ignore_index=True)
    df.to_parquet(cache, index=False)
    return df


def flags(domain):
    """Flag -> description from the flags file inside the zip."""
    with zipfile.ZipFile(FAO_DIR / f"{domain}_All_Data_(Normalized).zip") as zf:
        f = pd.read_csv(zf.open(f"{domain}_Flags.csv"), skipinitialspace=True)
    f.columns = f.columns.str.strip()
    return dict(zip(f["Flag"].str.strip(), f["Description"].str.strip()))


# Phase 2 extensions: trade (food balances) and producer prices.
FBS = "FoodBalanceSheets_E"
FBSH = "FoodBalanceSheetsHistoric_E"
PRICES = "Prices_E"
ENCODING = {FBSH: "latin-1"}  # the historic food balance CSV is not valid UTF-8
ANNUAL_MONTHS_CODE = 7021  # "Annual value" in the Prices Months column
SUA = "SUA_Crops_Livestock_E"  # Supply Utilization Accounts, 2010 onwards (calorie factors, src/crops/calories.py)


def read_filtered(domain, element_codes, item_codes=None, usecols=None, extra_filter=None, cache_tag=""):
    """Like read(), for domains with other column sets: optional item filter, extra columns, encoding.

    extra_filter(chunk) -> boolean mask applied in addition to element, year and country filters.
    Cached as parquet; `cache_tag` must name any extra_filter so that different filters do not share a cache.
    """
    # The cache name carries a hash of everything that shapes the result (the extra filter is named by `cache_tag`).
    spec = repr((sorted(element_codes), sorted(item_codes) if item_codes is not None else None, usecols, cache_tag))
    tag = "_".join(map(str, element_codes)) + "_" + hashlib.md5(spec.encode()).hexdigest()[:10]
    cache = FAO_DIR / f"{domain}_{tag}.parquet"
    source = FAO_DIR / f"{domain}_All_Data_(Normalized).zip"
    if cache.exists() and cache.stat().st_mtime >= source.stat().st_mtime:
        return pd.read_parquet(cache)
    cols = usecols or COLS
    with zipfile.ZipFile(source) as zf:
        chunks = []
        for ch in pd.read_csv(zf.open(f"{domain}_All_Data_(Normalized).csv"), usecols=cols,
                              encoding=ENCODING.get(domain, "utf-8"), chunksize=500_000, low_memory=False):
            keep = (ch["Element Code"].isin(element_codes) & ch["Year"].between(YEAR_MIN, YEAR_MAX)
                    & (ch["Area Code"] < FIRST_AGGREGATE_AREA))
            if item_codes is not None:
                keep &= ch["Item Code"].isin(item_codes)
            if extra_filter is not None:
                keep &= extra_filter(ch)
            chunks.append(ch[keep])
    df = pd.concat(chunks, ignore_index=True)
    df.to_parquet(cache, index=False)
    return df
