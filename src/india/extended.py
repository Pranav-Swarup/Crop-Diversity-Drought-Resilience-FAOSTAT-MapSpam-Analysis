"""Extended 5-crop window (1993-2015) and irrigation / fertiliser controls from the ICRISAT mirror (Mendeley, unapportioned).

THIRD-PARTY MIRROR (see src.india.download): 5 crops only (rice, pearl millet, chickpea, groundnut, sugarcane), 1990-2015,
current (unapportioned) districts, no documentation of codes. It is never merged into the DES core table: it gives
  data/clean/india/crops_ext5.parquet     5-crop table keyed by the DES dist_code of the SAME polygon (name match, src.india.geo)
  data/clean/india/districts_ext5.csv     stability / split flags for the 1993-2015 window
  data/clean/india/mirror_controls.parquet  dist_code, year, irrig_share, fert_kg_ha (gross irrigated / gross cropped area)
-1 is the file's missing code (ICRISAT documentation, apportioned documentation p.4) and becomes NaN. Prices are the same
constant FAOSTAT India prices as the core (src.india.clean.load_prices). Mirror rice production is compared with DES rice
production where the two overlap (1997-2015) and printed.

Run: python -m src.india.extended
"""
import glob

import numpy as np
import pandas as pd

from ..common import paths
from .clean import load_prices, split_flags
from .geo import load_polygons, match_to_polygons

OUT = paths.CLEAN_REAL / "india"
YEARS = list(range(1993, 2016))
CROPS = {"RICE": ("Rice", 27), "PEARL MILLET": ("Pearl millet", 79), "CHICKPEA": ("Chickpea", 191),
         "GROUNDNUT": ("Groundnut", 242), "SUGARCANE": ("Sugarcane", 156)}
STATE_FIX = {"Orissa": "Odisha"}


def main():
    f = glob.glob(str(paths.RAW / "icrisat" / "*.xls"))
    assert len(f) == 1, f
    m = pd.read_excel(f[0])
    m = m.rename(columns={"Dist Code": "mcode", "Year": "year", "State Name": "state", "Dist Name": "dname"})
    m["state"] = m.state.replace(STATE_FIX)
    assert not m.duplicated(["mcode", "year"]).any()
    ref = m.groupby("mcode").agg(state=("state", "last"), dname=("dname", "last"), first_year=("year", "min")).reset_index()

    g2 = load_polygons()
    mm = match_to_polygons(ref, g2, "mcode", "state", "dname")
    dm = pd.read_csv(OUT / "district_match.csv", dtype={"dist_code": str})[["dist_code", "poly_id"]].dropna()
    pid = pd.Series(np.nan, index=mm.index, dtype=object)
    ok = mm.poly_idx.notna()
    pid[ok] = g2.loc[mm.poly_idx[ok].astype(int), "shapeID"].values
    ref["poly_id"] = pid.values
    ref = ref.merge(dm, on="poly_id", how="left")                  # DES dist_code of the same polygon
    print(f"mirror districts {len(ref)}; matched to a polygon {ref.poly_id.notna().sum()}; also a DES district {ref.dist_code.notna().sum()}")
    assert ref.dist_code.dropna().is_unique, "two mirror districts map to one DES district"
    code_of = ref.set_index("mcode").dist_code

    prices = load_prices()
    rows = []
    for col, (name, item) in CROPS.items():
        d = m[["mcode", "year", f"{col} AREA (1000 ha)", f"{col} PRODUCTION (1000 tons)"]].copy()
        d.columns = ["mcode", "year", "area_k", "prod_k"]
        d = d[d.area_k > 0]
        d["crop"], d["price_intl_dollars_t"] = name, float(prices[item])
        d["area_ha"], d["prod_t"] = d.area_k * 1000, d.prod_k.where(d.prod_k >= 0) * 1000
        rows.append(d.drop(columns=["area_k", "prod_k"]))
    c = pd.concat(rows)
    c = c[c.year.between(YEARS[0], YEARS[-1])]
    # no gap filling; implausible yields dropped like in the core (10x the crop's median)
    y0 = c.prod_t / c.area_ha
    c.loc[y0 > 10 * y0.groupby(c.crop).transform("median"), "prod_t"] = np.nan
    c["yield_t_ha"] = c.prod_t / c.area_ha
    c["value_const"] = c.prod_t * c.price_intl_dollars_t
    c["is_missing"] = c.prod_t.isna()

    # stability and split flags on the 1993-2015 window (same rules as the core)
    a = c.rename(columns={"year": "y", "mcode": "district_code"})
    n_years = a.groupby("district_code").y.nunique()
    stable = set(n_years[n_years == len(YEARS)].index)
    state_of, first = ref.set_index("mcode").state, ref.set_index("mcode").first_year
    first = first.clip(lower=1990)
    flags = split_flags(a, state_of, first.where(first > 1990, YEARS[0]), YEARS)
    dist = ref.copy()
    dist["stable"] = dist.mcode.isin(stable)
    dist["split_flag"] = dist.mcode.isin(flags)
    dist["core"] = dist.stable & ~dist.split_flag & dist.dist_code.notna()
    dist = dist.dropna(subset=["dist_code"])
    c["dist_code"] = c.mcode.map(code_of)
    c = c.dropna(subset=["dist_code"])[["dist_code", "year", "crop", "area_ha", "prod_t", "yield_t_ha", "price_intl_dollars_t", "value_const", "is_missing"]]
    c = c.sort_values(["dist_code", "year", "crop"]).reset_index(drop=True)
    assert not c.duplicated(["dist_code", "year", "crop"]).any() and c.year.between(1993, 2015).all()

    # controls
    k = m.copy()
    for col in ("GROSS CROPPED AREA (1000 ha)", "GROSS IRRIGATED AREA (1000 ha)", "TOTAL FERTILISER CONSUMPTION (tons)"):
        k.loc[k[col] < 0, col] = np.nan                       # -1 = missing code
    k["irrig_share"] = k["GROSS IRRIGATED AREA (1000 ha)"] / k["GROSS CROPPED AREA (1000 ha)"].where(k["GROSS CROPPED AREA (1000 ha)"] > 0)
    k["fert_kg_ha"] = k["TOTAL FERTILISER CONSUMPTION (tons)"] * 1000 / (k["GROSS CROPPED AREA (1000 ha)"].where(k["GROSS CROPPED AREA (1000 ha)"] > 0) * 1000)
    k["irrig_share"] = k.irrig_share.where(k.irrig_share.between(0, 1))
    k["dist_code"] = k.mcode.map(code_of)
    k = k.dropna(subset=["dist_code"])[["dist_code", "year", "irrig_share", "fert_kg_ha"]]
    k = k[k.year.between(1993, 2015)].reset_index(drop=True)
    assert k.irrig_share.dropna().between(0, 1).all()

    # mirror vs DES where both exist (rice)
    cd = pd.read_parquet(OUT / "crops.parquet")
    r = cd[cd.crop == "Rice"].merge(c[c.crop == "Rice"], on=["dist_code", "year"], suffixes=("_des", "_mir"))
    ra, rp = r.area_ha_des / r.area_ha_mir, r.prod_t_des / r.prod_t_mir
    print(f"rice, DES vs mirror (same polygon, {r.year.min()}-{r.year.max()}, {len(r)} district-years): area ratio median {ra.median():.2f} "
          f"(within 5%: {(abs(ra - 1) < .05).mean():.0%}); production ratio median {rp.median():.2f}")

    dist.to_csv(OUT / "districts_ext5.csv", index=False)
    c.to_parquet(OUT / "crops_ext5.parquet", index=False)
    k.to_parquet(OUT / "mirror_controls.parquet", index=False)
    print(f"ext5: rows {len(c):,}, districts {c.dist_code.nunique()}, stable {int(dist.stable.sum())}, split-flagged among stable "
          f"{int((dist.stable & dist.split_flag).sum())}, core {int(dist.core.sum())}")
    print(f"controls: irrig_share non-missing {k.irrig_share.notna().mean():.1%} (median {k.irrig_share.median():.2f}), "
          f"fert_kg_ha non-missing {k.fert_kg_ha.notna().mean():.1%}")


if __name__ == "__main__":
    main()
