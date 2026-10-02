"""Readers for the FAOSTAT bulk zips in data/raw/faostat/, cut to the 1993-2023 window."""
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
