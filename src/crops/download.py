"""Download FAOSTAT bulk files (QCL, QV, RL) and World Bank WDI indicators.

Run: python -m src.crops.download
"""
import pandas as pd
import requests

from ..common import paths
from ..common.sources import download, log_source

FAO_DIR = paths.RAW / "faostat"
WDI_DIR = paths.RAW / "wdi"
FAO_BASE = "https://bulks-faostat.fao.org/production/"
FAO_FILES = {
    "FAOSTAT QCL (Crops and livestock products)": "Production_Crops_Livestock_E_All_Data_(Normalized).zip",
    "FAOSTAT QV (Value of Agricultural Production)": "Value_of_Production_E_All_Data_(Normalized).zip",
    "FAOSTAT RL (Land Use)": "Inputs_LandUse_E_All_Data_(Normalized).zip",
}
WDI_INDICATORS = {
    "NY.GDP.PCAP.KD": "GDP per capita (constant 2015 US$)",
    "AG.CON.FERT.ZS": "Fertilizer consumption (kg per hectare of arable land)",
}


def download_wdi(indicator):
    url = f"https://api.worldbank.org/v2/country/all/indicator/{indicator}?format=json&date=1993:2023&per_page=20000"
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    meta, rows = r.json()
    assert meta["pages"] == 1, f"{indicator}: response is paginated ({meta['pages']} pages)"
    df = pd.DataFrame({
        "iso3": [x["countryiso3code"] for x in rows],
        "country": [x["country"]["value"] for x in rows],
        "year": [int(x["date"]) for x in rows],
        "value": [x["value"] for x in rows],
    })
    assert len(df) > 5000, f"{indicator}: only {len(df)} rows returned"
    dest = WDI_DIR / f"{indicator}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    log_source(f"World Bank WDI {indicator}", url, dest,
               note=f"{WDI_INDICATORS[indicator]}; API last updated {meta.get('lastupdated')}")
    return dest


def main():
    for name, fname in FAO_FILES.items():
        download(name, FAO_BASE + fname, FAO_DIR / fname)
    for indicator in WDI_INDICATORS:
        download_wdi(indicator)
    for fname in FAO_FILES.values():
        assert (FAO_DIR / fname).stat().st_size > 1e6, fname
    print(f"\nFAOSTAT files: {len(FAO_FILES)}, WDI indicators: {len(WDI_INDICATORS)}")


if __name__ == "__main__":
    main()
