"""FAOSTAT Producer Prices (USD/tonne, annual) -> data/clean/prices.parquet.

Element 5532 "Producer Price (USD/tonne)"; only rows with Months Code 7021 ("Annual value") are kept
(the file also has monthly rows, which are dropped). Items = the item codes present in crops.parquet,
iso3 in countries.csv, 1993-2023. Prices are nominal USD per tonne as published.
is_imputed = flag E or I (flag meanings confirmed from Prices_E_Flags.csv inside the zip).
Merged entities (BLX, SCG, SDX): where one member area reports an item-year its price is kept; where
several report, the price is the mean weighted by that item-year's production (QCL, element 5510) of
the reporting members; if a reporting member has no positive production the item-year is left out.
The "China" aggregate (area 351) is dropped.

Run: python -m src.crops.prices
"""
import numpy as np
import pandas as pd

from ..common import paths
from ..common.contract import YEAR_MAX, YEAR_MIN, summary
from . import faostat
from .countries import area_map

PRICE_USD = 5532
PRODUCTION = 5510
IMPUTED_FLAGS = {"E": "Estimated value", "I": "Value imputed by a receiving agency"}
PRICE_COLS = ["Area Code", "Area", "Item Code", "Item", "Element Code", "Year", "Months Code", "Unit", "Value", "Flag"]


def merge_members(g):
    """Series for one iso3 x item x year with several reporting member areas: production-weighted mean price."""
    w = g["prod_t"]
    if len(g) == 1:
        return pd.Series({"price_usd_t": g["Value"].iloc[0], "is_imputed": g["is_imputed"].iloc[0], "left_out": False})
    if w.isna().any() or (w <= 0).any():
        return pd.Series({"price_usd_t": np.nan, "is_imputed": False, "left_out": True})
    return pd.Series({"price_usd_t": np.average(g["Value"], weights=w), "is_imputed": bool(g["is_imputed"].any()), "left_out": False})


def main():
    fl = faostat.flags(faostat.PRICES)
    for k, v in IMPUTED_FLAGS.items():
        assert fl.get(k) == v, f"Prices flag {k} means {fl.get(k)!r}, expected {v!r}"
    print(f"Prices flags: {fl}")

    crops = pd.read_parquet(paths.CROPS)
    countries = pd.read_csv(paths.COUNTRIES, dtype=str, keep_default_na=False)
    items = crops[["item_code", "item"]].drop_duplicates("item_code")
    p = faostat.read_filtered(faostat.PRICES, [PRICE_USD], sorted(items["item_code"]), usecols=PRICE_COLS, cache_tag="annual",
                              extra_filter=lambda c: c["Months Code"] == faostat.ANNUAL_MONTHS_CODE)
    assert (p["Unit"] == "USD").all() and (p["Months Code"] == faostat.ANNUAL_MONTHS_CODE).all()
    p = p[(p["Area Code"] != 351) & p["Value"].notna()]
    p["is_imputed"] = p["Flag"].isin(list(IMPUTED_FLAGS))

    amap = area_map().dropna(subset=["iso3"])
    p = p.merge(amap[["area_code", "iso3"]], left_on="Area Code", right_on="area_code")
    p = p[p["iso3"].isin(countries["iso3"])]

    merged = p["iso3"].isin(countries.loc[countries["member_iso3"] != "", "iso3"])
    q = faostat.read(faostat.QCL, [PRODUCTION])[["Area Code", "Item Code", "Year", "Value"]].rename(columns={"Value": "prod_t"})
    p = p.merge(q, on=["Area Code", "Item Code", "Year"], how="left")
    keys = ["iso3", "Year", "Item Code"]
    single = p[~merged].rename(columns={"Value": "price_usd_t"})[keys + ["price_usd_t", "is_imputed"]]
    mg = p[merged].groupby(keys).apply(merge_members, include_groups=False).reset_index()
    n_left_out = int(mg["left_out"].sum())
    n_multi = int((p[merged].groupby(keys).size() > 1).sum())
    mg = mg[~mg["left_out"]].drop(columns="left_out")
    out = pd.concat([single, mg], ignore_index=True)
    out = out.rename(columns={"Year": "year", "Item Code": "item_code"}).merge(items, on="item_code", how="left")
    out = out[["iso3", "year", "item_code", "item", "price_usd_t", "is_imputed"]]
    out = out.astype({"year": int, "item_code": int, "price_usd_t": float, "is_imputed": bool})
    out = out.sort_values(["iso3", "item_code", "year"]).reset_index(drop=True)

    assert not out.duplicated(["iso3", "year", "item_code"]).any(), "duplicate iso3-year-item"
    assert out["iso3"].str.fullmatch(r"[A-Z]{3}").all() and out["year"].between(YEAR_MIN, YEAR_MAX).all()
    assert out["item"].notna().all() and set(out["item_code"]) <= set(items["item_code"])
    assert (out["price_usd_t"] > 0).all() and np.isfinite(out["price_usd_t"]).all()
    assert not {"BEL", "LUX", "SRB", "MNE", "SDN", "SSD"} & set(out["iso3"])
    paths.CLEAN.mkdir(parents=True, exist_ok=True)
    out.to_parquet(paths.PRICES, index=False)

    print(summary(out, "prices"))
    print(f"items: {out['item_code'].nunique()} of {len(items)} crop items; imputed (E, I): {100 * out['is_imputed'].mean():.1f}%")
    print(f"merged entities: {n_multi} item-years with several reporting members (production-weighted), "
          f"{n_left_out} left out for missing or zero production; entities with prices: {sorted(set(out['iso3']) & set(countries.loc[countries['member_iso3'] != '', 'iso3']))}")


if __name__ == "__main__":
    main()
