"""WDI + FAOSTAT Land Use -> data/clean/controls.parquet.

First-iteration pipeline written from src/metrics' side; Gursahib owns this module.
gdp_pc and fert_kg_ha are left missing for the merged entities (BLX, SCG, SDX): WDI has
no series for them and a per-capita figure cannot be summed across members.

Run: python -m src.crops.controls
"""
import numpy as np
import pandas as pd

from ..common import paths
from ..common.contract import YEAR_MAX, YEAR_MIN, load, summary, validate
from . import faostat
from .countries import area_map
from .download import WDI_DIR

AREA_ELEMENT = 5110
ITEM_CROPLAND, ITEM_IRRIGATION = 6620, 6690


def wdi(indicator, name):
    df = pd.read_csv(WDI_DIR / f"{indicator}.csv")
    df = df[df["iso3"].notna() & df["value"].notna()]
    return df[["iso3", "year", "value"]].rename(columns={"value": name})


def irrigation_share():
    rl = faostat.read(faostat.RL, [AREA_ELEMENT])
    rl = rl[rl["Item Code"].isin([ITEM_CROPLAND, ITEM_IRRIGATION])]
    names = dict(rl[["Item Code", "Item"]].drop_duplicates().to_numpy())
    assert names == {ITEM_CROPLAND: "Cropland", ITEM_IRRIGATION: "Land area equipped for irrigation"}, names
    assert (rl["Unit"] == "1000 ha").all()
    rl = rl.merge(area_map().dropna(subset=["iso3"])[["area_code", "iso3"]], left_on="Area Code", right_on="area_code")
    wide = rl.pivot_table(index=["iso3", "Year"], columns="Item Code", values="Value", aggfunc=lambda s: s.sum(min_count=1))
    wide = wide[(wide[ITEM_CROPLAND] > 0) & wide[ITEM_IRRIGATION].notna()]
    out = (wide[ITEM_IRRIGATION] / wide[ITEM_CROPLAND]).rename("irrig_share").reset_index()
    return out.rename(columns={"Year": "year"})


def main():
    countries = load("countries")
    grid = pd.MultiIndex.from_product([countries["iso3"], range(YEAR_MIN, YEAR_MAX + 1)], names=["iso3", "year"]).to_frame(index=False)
    out = grid.merge(wdi("NY.GDP.PCAP.KD", "gdp_pc"), on=["iso3", "year"], how="left") \
              .merge(wdi("AG.CON.FERT.ZS", "fert_kg_ha"), on=["iso3", "year"], how="left") \
              .merge(irrigation_share(), on=["iso3", "year"], how="left")
    out = out.astype({"year": int, "gdp_pc": float, "fert_kg_ha": float, "irrig_share": float})

    validate(out, "controls")
    assert len(out) == len(countries) * (YEAR_MAX - YEAR_MIN + 1)
    paths.CLEAN.mkdir(parents=True, exist_ok=True)
    out.to_parquet(paths.CONTROLS, index=False)
    print(summary(out, "controls"))
    for c in ["gdp_pc", "fert_kg_ha", "irrig_share"]:
        none = sorted(out.groupby("iso3")[c].count().loc[lambda s: s == 0].index)
        print(f"{c}: {100 * out[c].isna().mean():.1f}% missing; {len(none)} countries with no value; "
              f"median {out[c].median():.3g}, max {out[c].max():.3g}")
    print(f"irrig_share > 1: {sorted(out.loc[out['irrig_share'] > 1, 'iso3'].unique())}")


if __name__ == "__main__":
    main()
