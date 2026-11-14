"""FAOSTAT Food Balances (cereals) -> data/clean/trade.parquet and outputs/stats/trade_fbs_overlap.csv.

Item: "Cereals - Excluding Beer" (item code 2905). Elements: Production 5511, Import quantity 5611,
Export quantity 5911, Domestic supply quantity 5301; FBS units are 1000 t, converted to t.
FBSH (food balances, old methodology, to 2013) supplies 1993-2009 and FBS (new methodology, from 2010)
supplies 2010-2023. The two are not blended or adjusted; `fbs_source` records the origin of each row.
import_dep = import_t / (prod_t + import_t - export_t), NaN where the denominator is <= 0.
Merged entities (BLX, SCG, SDX) are the sum of their member areas, and only where every member of one
of the entity's area sets is present in that year and source (otherwise the row is left out).
FAOSTAT publishes one negative export value (Iraq 2010, FBS); it is kept as published.
The "China" aggregate (area 351) is dropped. FBS holds no Sudan or South Sudan rows for cereals,
so SDX has no rows from 2010.
trade_fbs_overlap.csv compares import_dep from both sources in 2010-2013.

Run: python -m src.crops.trade
"""
import numpy as np
import pandas as pd

from ..common import paths
from ..common.contract import YEAR_MAX, YEAR_MIN, summary
from . import faostat
from .countries import area_map

CEREALS = 2905
PRODUCTION, IMPORT, EXPORT, SUPPLY = 5511, 5611, 5911, 5301
ELEMENT_NAMES = {PRODUCTION: "prod_t", IMPORT: "import_t", EXPORT: "export_t", SUPPLY: "supply_t"}
FBSH_LAST_YEAR = 2009  # FBSH for years <= 2009, FBS from 2010
OVERLAP_YEARS = (2010, 2013)
# Alternative sets of FAOSTAT area codes that make up a merged entity in a given year.
AREA_SETS = {"BLX": [{15}, {255, 256}], "SCG": [{186}, {272, 273}], "SDX": [{206}, {276, 277}]}
# Negative values published by FAOSTAT itself (kept as published): FBS Iraq 2010 cereal export quantity = -62 (1000 t).
KNOWN_NEGATIVE = {("IRQ", 2010, "export_t")}
OVERLAP_CSV = paths.STATS_DIR / "trade_fbs_overlap.csv"


def load_source(domain, source, amap):
    """One source's cereal quantities (t) per iso3 x year, merged entities summed."""
    df = faostat.read_filtered(domain, list(ELEMENT_NAMES), [CEREALS])
    assert (df["Unit"] == "1000 t").all(), f"{domain}: unexpected units {df['Unit'].unique()}"
    df = df[df["Area Code"] != 351]
    df = df.merge(amap.dropna(subset=["iso3"])[["area_code", "iso3"]], left_on="Area Code", right_on="area_code")
    df["quantity"] = df["Value"] * 1000.0
    df["name"] = df["Element Code"].map(ELEMENT_NAMES)
    wide = df.pivot_table(index=["iso3", "Year", "Area Code"], columns="name", values="quantity", aggfunc="sum").reset_index()
    areas = wide.groupby(["iso3", "Year"])["Area Code"].agg(set)
    complete = areas[[a in [s for s in AREA_SETS.get(i, [a])] for (i, _), a in areas.items()]].index
    n_drop = len(areas) - len(complete)
    out = wide.groupby(["iso3", "Year"], as_index=False)[list(ELEMENT_NAMES.values())].sum(min_count=1)
    out = out.set_index(["iso3", "Year"]).loc[complete].reset_index().rename(columns={"Year": "year"})
    out["import_dep"] = import_dep(out)
    out["fbs_source"] = source
    return out, n_drop


def import_dep(df):
    denom = df["prod_t"] + df["import_t"] - df["export_t"]
    return (df["import_t"] / denom).where(denom > 0)


def overlap_table(fbsh, fbs):
    lo, hi = OVERLAP_YEARS
    m = fbsh[fbsh["year"].between(lo, hi)].merge(fbs[fbs["year"].between(lo, hi)], on=["iso3", "year"], suffixes=("_fbsh", "_fbs"))
    m["abs_diff"] = (m["import_dep_fbsh"] - m["import_dep_fbs"]).abs()
    m = m.dropna(subset=["abs_diff"])
    t = m.groupby("iso3").agg(n_years=("year", "size"), mean_abs_diff=("abs_diff", "mean"),
                              mean_import_dep_fbsh=("import_dep_fbsh", "mean"), mean_import_dep_fbs=("import_dep_fbs", "mean"))
    t = t.reset_index()
    t.insert(1, "statistic", "country mean")
    summ = pd.DataFrame([
        {"iso3": "ALL", "statistic": "median over countries of mean_abs_diff", "n_years": len(m),
         "mean_abs_diff": t["mean_abs_diff"].median(), "mean_import_dep_fbsh": np.nan, "mean_import_dep_fbs": np.nan},
        {"iso3": "ALL", "statistic": "median of all country-year abs diffs", "n_years": len(m),
         "mean_abs_diff": m["abs_diff"].median(), "mean_import_dep_fbsh": np.nan, "mean_import_dep_fbs": np.nan}])
    return pd.concat([t, summ], ignore_index=True)


def main():
    countries = pd.read_csv(paths.COUNTRIES, dtype=str, keep_default_na=False)
    amap = area_map()
    fbsh, d1 = load_source(faostat.FBSH, "FBSH", amap)
    fbs, d2 = load_source(faostat.FBS, "FBS", amap)
    print(f"country-years left out because a merged entity was incomplete: FBSH {d1}, FBS {d2}")

    out = pd.concat([fbsh[fbsh["year"] <= FBSH_LAST_YEAR], fbs[fbs["year"] > FBSH_LAST_YEAR]], ignore_index=True)
    out = out[out["iso3"].isin(countries["iso3"]) & out["year"].between(YEAR_MIN, YEAR_MAX)]
    out = out[["iso3", "year", "prod_t", "import_t", "export_t", "supply_t", "import_dep", "fbs_source"]]
    out = out.astype({"year": int}).sort_values(["iso3", "year"]).reset_index(drop=True)
    for c in ["prod_t", "import_t", "export_t", "supply_t", "import_dep"]:
        out[c] = out[c].astype(float)

    assert not out.duplicated(["iso3", "year"]).any(), "duplicate iso3-year"
    assert out["iso3"].str.fullmatch(r"[A-Z]{3}").all() and out["year"].between(YEAR_MIN, YEAR_MAX).all()
    qty = out.set_index(["iso3", "year"])[["prod_t", "import_t", "export_t"]].stack()
    neg = qty[qty < 0]
    assert {(i, y, v) for (i, y, v) in neg.index} <= KNOWN_NEGATIVE, f"unexpected negative quantity: {neg.to_dict()}"
    assert set(out["fbs_source"]) <= {"FBS", "FBSH"}
    assert (out.loc[out["year"] <= FBSH_LAST_YEAR, "fbs_source"] == "FBSH").all()
    assert (out.loc[out["year"] > FBSH_LAST_YEAR, "fbs_source"] == "FBS").all()
    d = out["import_dep"].dropna()
    assert np.isfinite(d).all() and (d >= 0).all(), "import_dep not finite and non-negative"
    den = out["prod_t"] + out["import_t"] - out["export_t"]
    assert out.loc[den <= 0, "import_dep"].isna().all() and out.loc[den > 0, "import_dep"].notna().all()
    assert not {"BEL", "LUX", "SRB", "MNE", "SDN", "SSD"} & set(out["iso3"]) and "CHN" in set(out["iso3"])
    paths.CLEAN.mkdir(parents=True, exist_ok=True)
    out.to_parquet(paths.TRADE, index=False)

    ov = overlap_table(fbsh, fbs)
    paths.STATS_DIR.mkdir(parents=True, exist_ok=True)
    ov.to_csv(OVERLAP_CSV, index=False)

    print(summary(out, "trade"))
    print(f"rows by source: {out['fbs_source'].value_counts().to_dict()}; import_dep undefined (denominator <= 0): {int(den.le(0).sum())}")
    print(f"overlap {OVERLAP_YEARS[0]}-{OVERLAP_YEARS[1]}: {ov['iso3'].ne('ALL').sum()} countries; "
          f"median over countries of mean abs diff in import_dep = {ov.iloc[-2]['mean_abs_diff']:.4f}; "
          f"median country-year abs diff = {ov.iloc[-1]['mean_abs_diff']:.4f}")


if __name__ == "__main__":
    main()
