"""FAOSTAT Detailed Trade Matrix (cereal grains, reporter-side imports) -> data/clean/trade_partners.parquet
and outputs/stats/trade_matrix_vs_fbs.csv.

Bilateral cereal imports per importer (iso3), exporter (partner_iso3) and year, 1993-2023, from the importer's
own report (Import quantity 5610, tonnes; Import value 5622, 1000 USD, converted to US dollars).
Cereal grains are unprocessed grains only: FAOSTAT item codes 15 Wheat, 27 Rice (paddy), 28 Husked rice,
31 Rice milled, 32 Rice broken, 44 Barley, 56 Maize (corn), 71 Rye, 75 Oats, 79 Millet, 83 Sorghum,
89 Buckwheat, 92 Quinoa, 94 Fonio, 97 Triticale, 101 Canary seed, 103 Mixed grain, 108 Cereals n.e.c.
Flours, malt, bran, germ, breakfast cereals, cereal preparations, beer and the derived items 29 (milled husked
rice) and 30 (paddy in milled equivalent) are not included.
Tonnes are summed across these grains without conversion; this follows the FAOSTAT Food Balance "Cereals"
aggregate used in src/crops/trade.py (which, unlike this file, also counts processed products in grain equivalent).
Reporters and partners are FAOSTAT countries (area code < 5000, "China" aggregate 351 dropped) mapped to the
project's iso3 with `area_map()`; merged entities (BLX, SCG, SDX) are summed over their member areas, flows
between two areas of one entity are dropped, and areas without an iso3 are dropped (their tonnes are printed).
No gap filling and no mirror flows: only what the importer reported.
trade_matrix_vs_fbs.csv compares the sum over partners with import_t in data/clean/trade.parquet.

Mirror file: data/clean/trade_partners_mirror.parquet has the same columns and meaning (iso3 = importer,
partner_iso3 = exporter) but is built from the exporter's records (Export quantity 5910 t, Export value 5922
1000 USD; the file's Reporter is the exporter and its Partner the importer). Same items, same cleaning. The
importer-reported and exporter-reported files are never blended; a comparison over the cells present in both is
printed (Spearman of import_t, median mirror/reporter ratio, and per importer with >= 20 reporting years the
Spearman correlation of the two files' partner shares over the window).

Needs: python -m src.crops.download_trade_matrix; data/clean/trade.parquet (python -m src.crops.trade).
Run: python -m src.crops.trade_matrix
"""
import zipfile

import numpy as np
import pandas as pd

from ..common import paths
from ..common.contract import YEAR_MAX, YEAR_MIN
from . import faostat
from .countries import area_map
from .download import FAO_DIR

TM = "Trade_DetailedTradeMatrix_E"
TM_ZIP = FAO_DIR / f"{TM}_All_Data_(Normalized).zip"
IMPORT_QTY, IMPORT_VAL = 5610, 5622
EXPORT_QTY, EXPORT_VAL = 5910, 5922
CEREAL_ITEMS = {15: "Wheat", 27: "Rice", 28: "Husked rice", 31: "Rice; milled", 32: "Rice; broken", 44: "Barley",
                56: "Maize (corn)", 71: "Rye", 75: "Oats", 79: "Millet", 83: "Sorghum", 89: "Buckwheat",
                92: "Quinoa", 94: "Fonio", 97: "Triticale", 101: "Canary seed", 103: "Mixed grain", 108: "Cereals n.e.c."}
CEREAL_PATTERN = "wheat|rice|maize|barley|rye|oats|millet|sorghum|buckwheat|quinoa|fonio|triticale|canary|mixed grain|cereal"
CHINA_AGGREGATE = 351
COLS = ["Reporter Country Code", "Reporter Countries", "Partner Country Code", "Partner Countries", "Item Code", "Item",
        "Element Code", "Element", "Year", "Unit", "Value", "Flag"]
STRUCT_CACHE = FAO_DIR / f"{TM}_structure.parquet"
ROWS_CACHE = FAO_DIR / f"{TM}_cereal_flows.parquet"
OUT_CSV = paths.STATS_DIR / "trade_matrix_vs_fbs.csv"
PARTNERS = paths.CLEAN / "trade_partners.parquet"
PARTNERS_MIRROR = paths.CLEAN / "trade_partners_mirror.parquet"


def scan():
    """One chunked pass over the zip: the file's structure and the cereal import rows (cached as parquet)."""
    if ROWS_CACHE.exists() and STRUCT_CACHE.exists() and min(ROWS_CACHE.stat().st_mtime, STRUCT_CACHE.stat().st_mtime) >= TM_ZIP.stat().st_mtime:
        return pd.read_parquet(ROWS_CACHE), pd.read_parquet(STRUCT_CACHE)
    rows, elems, items, header = [], [], [], None
    with zipfile.ZipFile(TM_ZIP) as zf:
        for ch in pd.read_csv(zf.open(f"{TM}_All_Data_(Normalized).csv"), encoding="utf-8", chunksize=500_000, low_memory=False):
            if header is None:
                header = list(ch.columns)
            elems.append(ch.groupby(["Element Code", "Element", "Unit"]).size().rename("n").reset_index())
            is_cereal = ch["Item"].str.contains(CEREAL_PATTERN, case=False, regex=True)
            items.append(ch[is_cereal].groupby(["Item Code", "Item", "Element Code"]).size().rename("n").reset_index())
            keep = (ch["Element Code"].isin([IMPORT_QTY, IMPORT_VAL, EXPORT_QTY, EXPORT_VAL]) & ch["Item Code"].isin(CEREAL_ITEMS)
                    & ch["Year"].between(YEAR_MIN, YEAR_MAX)
                    & (ch["Reporter Country Code"] < faostat.FIRST_AGGREGATE_AREA)
                    & (ch["Partner Country Code"] < faostat.FIRST_AGGREGATE_AREA))
            rows.append(ch.loc[keep, COLS])
    elems = pd.concat(elems).groupby(["Element Code", "Element", "Unit"], as_index=False)["n"].sum()
    elems["kind"] = "element"
    items = pd.concat(items).groupby(["Item Code", "Item", "Element Code"], as_index=False)["n"].sum()
    items["kind"] = "item"
    cols = pd.DataFrame({"Item": header, "kind": "column"})
    struct = pd.concat([cols, elems, items], ignore_index=True)
    df = pd.concat(rows, ignore_index=True)
    FAO_DIR.mkdir(parents=True, exist_ok=True)
    struct.to_parquet(STRUCT_CACHE, index=False)
    df.to_parquet(ROWS_CACHE, index=False)
    return df, struct


def inspect(df, struct):
    """Print columns, element/unit combinations and cereal-named items; assert the expected structure."""
    print("CSV columns:", struct.loc[struct["kind"] == "column", "Item"].tolist())
    el = struct[struct["kind"] == "element"][["Element Code", "Element", "Unit", "n"]]
    print("\nElement Code / Element / Unit combinations (all rows of the file):")
    print(el.to_string(index=False))
    it = struct[struct["kind"] == "item"].copy()
    it["Element Code"] = it["Element Code"].astype(int)
    print("\nItem Code / Item pairs whose name matches cereal terms (rows per element code):")
    piv = it.pivot_table(index=["Item Code", "Item"], columns="Element Code", values="n", aggfunc="sum").fillna(0).astype(int)
    print(piv.to_string())
    print("\nExpected cereal grain items, as named in the file:")
    names = it.drop_duplicates("Item Code").set_index("Item Code")["Item"]
    for code, label in CEREAL_ITEMS.items():
        print(f"  {code:>3} expected '{label}' -> file: '{names.get(code, 'MISSING')}'")
    unit = dict(zip(el["Element Code"], el["Unit"]))
    assert unit[IMPORT_QTY] == "t" and unit[IMPORT_VAL] == "1000 USD", f"unexpected units {unit}"
    assert unit[EXPORT_QTY] == "t" and unit[EXPORT_VAL] == "1000 USD", f"unexpected units {unit}"
    for code, u in [(IMPORT_QTY, "t"), (IMPORT_VAL, "1000 USD"), (EXPORT_QTY, "t"), (EXPORT_VAL, "1000 USD")]:
        assert set(df["Unit"][df["Element Code"] == code]) == {u}, code
    assert set(CEREAL_ITEMS) <= set(df["Item Code"]) | set(), "expected item codes without any import rows: " \
        f"{sorted(set(CEREAL_ITEMS) - set(df['Item Code']))}"


def to_iso3(df, amap, qty):
    """Map reporter and partner area codes to rep_iso3 / par_iso3; return the mapped frame and the dropped quantity rows."""
    m = amap.dropna(subset=["iso3"]).set_index("area_code")["iso3"]
    df = df.assign(rep_iso3=df["Reporter Country Code"].map(m), par_iso3=df["Partner Country Code"].map(m))
    bad = df["rep_iso3"].isna() | df["par_iso3"].isna()
    return df[~bad], df[bad & (df["Element Code"] == qty)]


def build(df, amap, qty, val, mirror):
    """Bilateral cereal flows keyed iso3 = importer, partner_iso3 = exporter, from one side's records.
    mirror=False: importer-reported (Reporter = importer); mirror=True: exporter-reported (Reporter = exporter)."""
    label = "mirror (exporter-reported)" if mirror else "reporter (importer-reported)"
    df = df[df["Element Code"].isin([qty, val])]
    mapped, dropped = to_iso3(df, amap, qty)
    total_t = df.loc[df["Element Code"] == qty, "Value"].sum()
    drop_t = dropped["Value"].sum()
    print(f"\n[{label}] dropped as unmapped (reporter or partner has no iso3): {drop_t:,.0f} t of {total_t:,.0f} t "
          f"({100 * drop_t / total_t:.3f}%), {len(dropped)} rows")
    known = set(amap.dropna(subset=["iso3"])["area_code"])
    for side, code, name in [("reporter", "Reporter Country Code", "Reporter Countries"), ("partner", "Partner Country Code", "Partner Countries")]:
        u = dropped[~dropped[code].isin(known)].groupby([code, name])["Value"].sum().sort_values(ascending=False).head(10)
        print(f"ten largest unmapped {side} areas by tonnes:")
        print(u.round(0).to_string() if len(u) else "  none")
    if mirror:
        mapped = mapped.rename(columns={"rep_iso3": "partner_iso3", "par_iso3": "iso3"})
    else:
        mapped = mapped.rename(columns={"rep_iso3": "iso3", "par_iso3": "partner_iso3"})
    mapped = mapped[mapped["iso3"] != mapped["partner_iso3"]]
    keys = ["iso3", "partner_iso3", "Year"]
    q = mapped[mapped["Element Code"] == qty].groupby(keys, as_index=False)["Value"].sum().rename(columns={"Value": "import_t"})
    v = mapped[mapped["Element Code"] == val].groupby(keys, as_index=False)["Value"].sum().rename(columns={"Value": "val_k"})
    out = q.merge(v, on=keys, how="left")
    out["import_usd"] = out["val_k"] * 1000.0
    out = out.rename(columns={"Year": "year"})[["iso3", "partner_iso3", "year", "import_t", "import_usd"]]
    out = out[out["import_t"] > 0].astype({"year": int})
    out["import_t"] = out["import_t"].astype(float)
    out["import_usd"] = out["import_usd"].astype(float)
    out = out.sort_values(["iso3", "partner_iso3", "year"]).reset_index(drop=True)

    assert not out.duplicated(["iso3", "partner_iso3", "year"]).any(), "duplicate key"
    assert out["iso3"].str.fullmatch(r"[A-Z]{3}").all() and out["partner_iso3"].str.fullmatch(r"[A-Z]{3}").all()
    assert out["year"].between(YEAR_MIN, YEAR_MAX).all()
    assert (out["import_t"] > 0).all() and (out["iso3"] != out["partner_iso3"]).all()
    assert not {"CHI", "CHX"} & (set(out["iso3"]) | set(out["partner_iso3"]))
    assert not {"BEL", "LUX", "SRB", "MNE", "SDN", "SSD"} & (set(out["iso3"]) | set(out["partner_iso3"]))
    assert "CHN" in set(out["iso3"]) | set(out["partner_iso3"]) and (out["import_usd"].dropna() >= 0).all()
    return out


def summarise(out, name):
    print(f"\n{name:<22} rows={len(out)}  importers={out['iso3'].nunique()}  exporters={out['partner_iso3'].nunique()}  "
          f"years={out['year'].min()}-{out['year'].max()}  import_usd missing={100 * out['import_usd'].isna().mean():.1f}%")
    ty = out.groupby("year")["import_t"].sum()
    for y in (1995, 2005, 2015, 2023):
        print(f"  total cereal grain trade {y}: {ty[y]:,.0f} t")
    print("ten largest exporters by tonnes (whole window):")
    print((out.groupby("partner_iso3")["import_t"].sum().sort_values(ascending=False).head(10)).round(0).to_string())
    for c in ("EGY", "JPN", "NGA", "MEX"):
        g = out[out["iso3"] == c].groupby("partner_iso3")["import_t"].sum().sort_values(ascending=False)
        print(f"{c}: " + ", ".join(f"{p} {100 * v / g.sum():.1f}%" for p, v in g.head(5).items()))


def compare_sources(rep, mir):
    """Cells (importer, exporter, year) present in both files, and partner-share agreement per importer."""
    from scipy.stats import spearmanr
    k = ["iso3", "partner_iso3", "year"]
    both = rep.merge(mir, on=k, suffixes=("_rep", "_mir"))
    rho = spearmanr(both["import_t_rep"], both["import_t_mir"])[0]
    ratio = both["import_t_mir"] / both["import_t_rep"]
    print(f"\nmirror vs reporter, cells in both files: {len(both)} of {len(rep)} reporter cells ({100 * len(both) / len(rep):.1f}%) "
          f"and {len(mir)} mirror cells ({100 * len(both) / len(mir):.1f}%)")
    print(f"  Spearman of import_t: {rho:.4f}; median ratio mirror/reporter: {ratio.median():.4f} "
          f"(5th-95th percentile {ratio.quantile(0.05):.3f} - {ratio.quantile(0.95):.3f})")
    ny = rep.groupby("iso3")["year"].nunique()
    long_ = ny[ny >= 20].index
    yrs = rep.groupby("iso3")["year"].agg(set)
    for label, restrict in [("all years of each file", False), ("years reported by the importer only", True)]:
        m_ = mir
        if restrict:
            m_ = mir[[y in yrs.get(i, set()) for i, y in zip(mir["iso3"], mir["year"])]]
        rs = []
        for c in long_:
            a = rep[rep["iso3"] == c].groupby("partner_iso3")["import_t"].sum()
            b = m_[m_["iso3"] == c].groupby("partner_iso3")["import_t"].sum()
            if a.sum() <= 0 or b.sum() <= 0:
                continue
            j = pd.concat([a / a.sum(), b / b.sum()], axis=1).fillna(0.0)
            if len(j) >= 3 and j.iloc[:, 0].nunique() > 1 and j.iloc[:, 1].nunique() > 1:
                rs.append(spearmanr(j.iloc[:, 0], j.iloc[:, 1])[0])
        rs = pd.Series(rs)
        print(f"  partner-share Spearman per importer with >= 20 reporting years ({label}): n={len(rs)}, "
              f"median {rs.median():.3f}, quartiles {rs.quantile(0.25):.3f} - {rs.quantile(0.75):.3f}")


def main():
    df, struct = scan()
    inspect(df, struct)

    df = df[(df["Reporter Country Code"] != CHINA_AGGREGATE) & (df["Partner Country Code"] != CHINA_AGGREGATE)]
    amap = area_map()
    out = build(df, amap, IMPORT_QTY, IMPORT_VAL, mirror=False)
    mir = build(df, amap, EXPORT_QTY, EXPORT_VAL, mirror=True)
    paths.CLEAN.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PARTNERS, index=False)
    mir.to_parquet(PARTNERS_MIRROR, index=False)

    # Diagnostic against the Food Balance cereal imports (importer-reported file).
    tm = out.groupby(["iso3", "year"], as_index=False)["import_t"].sum().rename(columns={"import_t": "tm_import_t"})
    fbs = pd.read_parquet(paths.TRADE)[["iso3", "year", "import_t", "fbs_source"]].rename(columns={"import_t": "fbs_import_t"})
    cmp_ = tm.merge(fbs, on=["iso3", "year"], how="inner")
    cmp_["ratio"] = cmp_["tm_import_t"] / cmp_["fbs_import_t"].where(cmp_["fbs_import_t"] > 0)
    paths.STATS_DIR.mkdir(parents=True, exist_ok=True)
    cmp_[["iso3", "year", "tm_import_t", "fbs_import_t", "ratio", "fbs_source"]].rename(columns={"fbs_import_t": "import_t"}) \
        .to_csv(OUT_CSV, index=False)
    r = cmp_["ratio"].replace([np.inf, -np.inf], np.nan).dropna()
    print(f"\nTM / FBS import ratio over {len(r)} country-years: median {r.median():.3f}, "
          f"5th-95th percentile {r.quantile(0.05):.3f} - {r.quantile(0.95):.3f}")
    med = cmp_.groupby("iso3")["ratio"].median()
    print(f"iso3 with median ratio < 0.3: {sorted(med[med < 0.3].index)}")
    print(f"iso3 with median ratio > 1.5: {sorted(med[med > 1.5].index)}")

    summarise(out, "trade_partners")
    summarise(mir, "trade_partners_mirror")
    compare_sources(out, mir)


if __name__ == "__main__":
    main()


