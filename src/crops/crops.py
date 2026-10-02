"""FAOSTAT QCL + QV -> data/clean/crops.parquet.

First-iteration pipeline written from src/metrics' side; Gursahib owns this module.

Run: python -m src.crops.crops
"""
import numpy as np
import pandas as pd

from ..common import paths
from ..common.contract import summary, validate
from . import faostat
from .countries import area_map

AREA_HARVESTED, PRODUCTION, VALUE_CONST_ID = 5312, 5510, 152
IMPUTED_FLAGS = {"E": "Estimated value", "I": "Value imputed by a receiving agency"}
# Aggregate items that carry an "Area harvested" element (checked against the names below).
AGGREGATE_ITEMS = {1717, 1720, 1723, 1726, 1729, 1732, 1735, 1738, 1804, 1841, 17530}
PRICE_TOLERANCE = 0.02


def item_prices(qcl_prod, qv):
    """Constant 2014-2016 international price per item (I$/t), recovered from FAOSTAT itself.

    FAOSTAT's constant-I$ gross production value is production times one international
    price per item, the same for every country and year. We recover that price as the
    median of value / production and assert that it really is constant.
    """
    m = qcl_prod.merge(qv, on=["Area Code", "Item Code", "Year"])
    m = m[(m["prod_t"] > 1000) & (m["value_fao"] > 100_000)]  # skip rows dominated by rounding
    m["price"] = m["value_fao"] / m["prod_t"]
    p = m.groupby("Item Code")["price"].agg(["median", "min", "max", "count"])
    spread = (p["max"] - p["min"]) / p["median"]
    assert (spread < PRICE_TOLERANCE).all(), f"value/production is not constant for items:\n{p[spread >= PRICE_TOLERANCE]}"
    print(f"item prices recovered for {len(p)} items; max spread of value/production within an item: {100 * spread.max():.2f}%")
    return p["median"]


def main():
    fl = faostat.flags(faostat.QCL)
    for k, v in IMPUTED_FLAGS.items():
        assert fl.get(k) == v, f"QCL flag {k} means {fl.get(k)!r}, expected {v!r}"
    print(f"QCL flags: {fl}")

    qcl = faostat.read(faostat.QCL, [AREA_HARVESTED, PRODUCTION])
    qv = faostat.read(faostat.QV, [VALUE_CONST_ID])
    assert (qcl.loc[qcl["Element Code"] == AREA_HARVESTED, "Unit"] == "ha").all()
    assert (qcl.loc[qcl["Element Code"] == PRODUCTION, "Unit"] == "t").all()
    assert (qv["Unit"] == "1000 Int$").all()

    # Crops = items with an "Area harvested" element, minus aggregates.
    crop_items = qcl.loc[qcl["Element Code"] == AREA_HARVESTED, ["Item Code", "Item"]].drop_duplicates()
    looks_aggregate = crop_items["Item"].str.contains(r"Total$|Primary$|, primary$|Equivalent$", regex=True)
    assert set(crop_items.loc[looks_aggregate, "Item Code"]) == AGGREGATE_ITEMS, \
        f"aggregate item list changed: {crop_items[looks_aggregate].to_dict('records')}"
    crop_codes = set(crop_items["Item Code"]) - AGGREGATE_ITEMS
    qcl = qcl[qcl["Item Code"].isin(crop_codes)]

    key = ["Area Code", "Item Code", "Year"]
    area = qcl[qcl["Element Code"] == AREA_HARVESTED].set_index(key)[["Item", "Value", "Flag"]]
    prod = qcl[qcl["Element Code"] == PRODUCTION].set_index(key)[["Value", "Flag"]]
    df = area.join(prod, lsuffix="_area", rsuffix="_prod", how="inner").reset_index()
    df = df.rename(columns={"Value_area": "area_ha", "Value_prod": "prod_t"})
    df = df[(df["area_ha"] > 0) & df["prod_t"].notna()]
    df["is_imputed"] = df["Flag_area"].isin(list(IMPUTED_FLAGS)) | df["Flag_prod"].isin(list(IMPUTED_FLAGS))

    # Value: FAOSTAT's figure where published, otherwise production x FAOSTAT's own item price.
    qv = qv.rename(columns={"Value": "value_fao"})[key + ["value_fao"]]
    qv["value_fao"] *= 1000.0  # thousand I$ -> I$
    prices = item_prices(df[key + ["prod_t"]], qv)
    df = df.merge(qv, on=key, how="left")
    df["price"] = df["Item Code"].map(prices)
    df["value_const"] = df["value_fao"].fillna(df["prod_t"] * df["price"])
    df["computed"] = df["value_fao"].isna()

    # Map to iso3 and sum merged entities (a merged country-year is imputed if any part is).
    amap = area_map().dropna(subset=["iso3"])
    df = df.merge(amap[["area_code", "iso3"]], left_on="Area Code", right_on="area_code", how="inner")
    filled_iso = df.loc[df["value_fao"].isna() & df["price"].notna()].groupby("iso3")["Year"].agg(["min", "max", "size"])
    # A crop with no price makes the summed value NaN, not a partial sum.
    df["value_const"] = df["value_const"].astype(float)
    out = df.groupby(["iso3", "Year", "Item Code"], as_index=False).agg(
        item=("Item", "first"), area_ha=("area_ha", "sum"), prod_t=("prod_t", "sum"),
        value_const=("value_const", lambda s: s.sum(min_count=len(s))), is_imputed=("is_imputed", "any"),
        computed=("computed", "any"))
    old_and_new = df[df["iso3"].isin(["BLX", "SCG", "SDX"])].groupby(["iso3", "Year"])["area_code"].agg(set)
    for codes in old_and_new:
        assert not ({15, 186, 206} & codes and codes - {15, 186, 206}), f"merged entity and its successors overlap: {codes}"
    out = out.rename(columns={"Year": "year", "Item Code": "item_code"})
    out["yield_t_ha"] = out["prod_t"] / out["area_ha"]
    out = out.astype({"year": int, "item_code": int, "area_ha": float, "prod_t": float, "value_const": float})
    out = out.sort_values(["iso3", "item_code", "year"]).reset_index(drop=True)
    # Rows of the final table whose value is (partly) production x item price rather than FAOSTAT's figure.
    comp = out[out["computed"] & out["value_const"].notna()]
    out = out[["iso3", "year", "item_code", "item", "area_ha", "prod_t", "yield_t_ha", "value_const", "is_imputed"]]

    validate(out, "crops")
    assert np.allclose(out["yield_t_ha"], out["prod_t"] / out["area_ha"])
    assert (out["area_ha"] > 0).all()
    assert "CHN" in set(out["iso3"]) and "IND" in set(out["iso3"])
    paths.CLEAN.mkdir(parents=True, exist_ok=True)
    out.to_parquet(paths.CROPS, index=False)
    filled_iso.to_csv(paths.RAW / "faostat" / "value_filled_from_price.csv")

    print(summary(out, "crops"))
    print(f"items: {out['item_code'].nunique()} crops ({len(AGGREGATE_ITEMS)} aggregate items dropped)")
    n_none = int(out["value_const"].isna().sum())
    print(f"value_const rows: {len(out) - len(comp) - n_none} published by FAOSTAT, {len(comp)} computed as production x item price "
          f"({100 * len(comp) / len(out):.1f}%), {n_none} without a price (left missing)")
    total, late = out["value_const"].sum(), out["year"] >= 2018
    print(f"  share of total value computed from price: {100 * comp['value_const'].sum() / total:.1f}% "
          f"(1993-2017: {100 * comp.loc[comp['year'] < 2018, 'value_const'].sum() / out.loc[~late, 'value_const'].sum():.1f}%, "
          f"2018-2023: {100 * comp.loc[comp['year'] >= 2018, 'value_const'].sum() / out.loc[late, 'value_const'].sum():.1f}%)")
    by_country = (comp.groupby("iso3")["value_const"].sum() / out.groupby("iso3")["value_const"].sum()).dropna()
    print(f"  countries with any computed value: {len(filled_iso)} ({(by_country > 0.1).sum()} with more than 10% of their value computed); "
          f"full list in data/raw/faostat/value_filled_from_price.csv")
    print(f"  items without a price: {sorted(df.loc[df['price'].isna(), 'Item'].unique())}")
    print(f"rows imputed (flags E, I): {100 * out['is_imputed'].mean():.1f}%; "
          f"value imputed: {100 * out.loc[out['is_imputed'], 'value_const'].sum() / out['value_const'].sum():.1f}%")


if __name__ == "__main__":
    main()
