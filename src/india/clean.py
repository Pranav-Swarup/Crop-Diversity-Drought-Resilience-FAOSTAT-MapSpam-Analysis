"""Clean the DES district crop table (India) into data/clean/india/{districts.csv, crops.parquet}.

Source: data/raw/india_apy/crop_wise_apy_des.csv (src.india.download_apy). Window: agricultural years 1997-98 .. 2022-23,
labelled by their START year (1998..2022): the monsoon of year Y falls inside the December-Y SPEI-12.

Rules (all proposals, none validated by the team yet):
- Season: if a "Total" row exists for (year, district, crop) use it, else sum the season rows (Total = sum of seasons in 99.9% of cases).
- Fixed crop list: crops reported nationally in all 25 years, minus catch-all "Other ..." crops. Keeps crop coverage constant over time.
- A missing row means "not grown or not reported"; it is NOT filled. Diversity uses crops with reported areas throughout the corresponding window.
- Value: prod_t x one constant price per crop = FAOSTAT India value_const / prod_t (an exact constant per item, 2014-16 intl $/t;
  data/clean/crops.parquet). Crops with no clean FAOSTAT counterpart get no price and are left out of the value index only.
- Stable district: has data in all 25 years. split_flag: the state gained a new district code in year t and the district's mean
  fixed-crop area over t..t+2 is below SPLIT_DROP x its mean over t-3..t-1 (a parent that kept its code after losing territory).

Run: python -m src.india.clean
"""
import numpy as np
import pandas as pd

from ..common import paths

RAW = paths.RAW / "india_apy" / "crop_wise_apy_des.csv"
OUT_DIR = paths.CLEAN_REAL / "india"
# 1997-98 is dropped: DES reports rice for only 267 of 349 core districts that year vs ~335 in every later year (thin first-year coverage);
# filling it is not allowed, and a partial first year distorts the value index. Core window = 1998..2022 (agricultural years 1998-99 .. 2022-23).
YEARS = list(range(1998, 2023))
SPLIT_DROP = 0.85
IMPLAUSIBLE = 10          # yield above this multiple of the crop's national median is treated as a source error
UNIT_ISSUE = "Coconut"    # production in nuts, not tonnes

# DES crop -> FAOSTAT item_code (India). Only clear counterparts; others get no price.
# Cotton(Lint) is left out on purpose: DES reports lint, FAOSTAT prices seed cotton (no conversions in this project).
# Coconut is left out: DES coconut production is counted in nuts in the source reports, unit unverified.
PRICE_ITEM = {
    "Rice": 27, "Wheat": 15, "Jowar": 83, "Bajra": 79, "Ragi": 79, "Small Millets": 79, "Maize": 56, "Barley": 44,
    "Gram": 191, "Arhar/Tur": 197, "Masoor": 201, "Moong(Green Gram)": 176, "Urad": 176, "Moth": 176,
    "Groundnut": 242, "Sesamum": 289, "Rapeseed &Mustard": 270, "Soyabean": 236, "Sunflower": 267, "Safflower": 280,
    "Castor Seed": 265, "Linseed": 333, "Niger Seed": 339, "Sugarcane": 156, "Potato": 116, "Onion": 403,
    "Sweet Potato": 122, "Tapioca": 125, "Cashewnut": 217, "Arecanut": 226, "Jute": 780, "Mesta": 782,
    "Dry Chillies": 689, "Coriander": 711, "Black Pepper": 687, "Ginger": 720, "Banana": 486, "Tobacco": 826,
}


def load_prices():
    c = pd.read_parquet(paths.CROPS)
    i = c[(c.iso3 == "IND") & (c.prod_t > 0)].copy()
    i["p"] = i.value_const / i.prod_t
    g = i.groupby("item_code").p.agg(["min", "max", "median"])
    assert ((g["max"] / g["min"]) < 1.001).all(), "FAOSTAT India value_const / prod_t is not constant per item"
    prices = g["median"].copy()
    # Milled rice tonnes are valued at a paddy-equivalent price. NABARD, Water
    # Productivity Mapping of Major Indian Crops, table 8, uses 0.67 t rice/t paddy.
    prices.loc[27] /= 0.67
    return prices


def season_rule(df):
    """One row per (y, district, crop). Returns frame with area_ha, prod_t, n_rows, rule."""
    key = ["y", "district_code", "crop_name"]
    tot = df[df.season == "Total"]
    rest = df[df.season != "Total"]
    has_tot = df.groupby(key).season.apply(lambda s: (s == "Total").any()).rename("has_total")
    a = tot.groupby(key).agg(area_ha=("area", "sum"), prod_t=("production", lambda s: s.sum(min_count=1)), n_rows=("id", "size"))
    a["rule"] = "total"
    b = rest.groupby(key).agg(area_ha=("area", "sum"), prod_t=("production", lambda s: s.sum(min_count=1)), n_rows=("id", "size"))
    b = b[~b.index.isin(a.index)]
    b["rule"] = "sum"
    return pd.concat([a, b]).reset_index().merge(has_tot.reset_index(), on=key, how="left")


def split_flags(a, state_of, first_year, years=YEARS):
    """Districts that kept their code after losing territory: in a state that gained a new district code in year t, the
    district's mean fixed-crop area over t..t+2 vs t-3..t-1, relative to the median of the same ratio over all districts
    of that state (so shared weather and reporting shocks cancel), is below SPLIT_DROP."""
    tot = a.groupby(["district_code", "y"]).area_ha.sum().unstack().reindex(columns=years)
    flagged = {}
    new = first_year[first_year > years[0]]
    for (state, t) in sorted(set(zip(new.index.map(state_of), new))):
        ds = tot.index[tot.index.map(state_of) == state]
        pre = tot.loc[ds, [y for y in range(t - 3, t) if y in years]].mean(axis=1)
        post = tot.loc[ds, [y for y in range(t, t + 3) if y in years]].mean(axis=1)
        ratio = (post / pre).replace([np.inf, -np.inf], np.nan)
        if ratio.notna().sum() < 3:
            continue
        rel = ratio / ratio.median()
        for d in rel.index[rel < SPLIT_DROP]:
            flagged.setdefault(d, []).append(f"{t}:{rel[d]:.2f}")
    return flagged


def main():
    raw = pd.read_csv(RAW, low_memory=False)
    raw["y"] = raw.year.str[:4].astype(int)
    assert set(YEARS) <= set(raw.y.unique()) and raw.y.max() == YEARS[-1], "unexpected year range"
    raw = raw[raw.y.isin(YEARS)].copy()
    assert (raw.area_unit == "Hectare").all() and (raw.production_unit == "Tonnes").all(), "units changed"

    yrs = raw.groupby("crop_name").y.nunique()
    fixed = sorted(c for c in yrs[yrs == len(YEARS)].index if not c.startswith("Other"))
    d = raw[raw.crop_name.isin(fixed)]
    a = season_rule(d)
    a = a[a.area_ha > 0].copy()
    assert not a.duplicated(["y", "district_code", "crop_name"]).any()

    state_of = raw.sort_values("y").groupby("district_code").state_name.last()
    name_of = raw.sort_values("y").groupby("district_code").district_name.last()
    scode_of = raw.sort_values("y").groupby("district_code").state_code.last()
    first_year = raw.groupby("district_code").y.min()
    n_years = a.groupby("district_code").y.nunique()
    stable = n_years[n_years == len(YEARS)].index
    flags = split_flags(a, state_of, first_year)

    districts = pd.DataFrame({"dist_code": state_of.index.astype(str), "dist_name": name_of.values,
                              "state_code": scode_of.values, "state_name": state_of.values}).set_index("dist_code", drop=False)
    districts["boundary_base"] = "DES 2022-23 codes, unapportioned"
    districts["n_years"] = n_years.reindex(state_of.index).fillna(0).astype(int).values
    districts["first_year"] = first_year.values
    districts["stable"] = districts.index.map(lambda c: int(c) in set(stable))
    districts["split_flag"] = districts.index.map(lambda c: int(c) in flags)
    districts["split_detail"] = districts.index.map(lambda c: ";".join(flags.get(int(c), [])))
    districts["core"] = districts.stable & ~districts.split_flag

    prices = load_prices()
    price = {k: float(prices[v]) for k, v in PRICE_ITEM.items() if v in prices.index}
    assert set(price) == set(PRICE_ITEM), f"FAOSTAT item missing for {set(PRICE_ITEM) - set(price)}"

    crops = a.rename(columns={"y": "year", "crop_name": "crop"})
    crops["dist_code"] = crops.district_code.astype(int).astype(str)
    # Production unit problems in the source: dropped (set NaN), never converted or replaced.
    #  - Coconut production is in nuts, not tonnes (median 8,140 "t/ha").
    #  - Isolated gross errors: yield above IMPLAUSIBLE x the crop's national median yield.
    y0 = crops.prod_t / crops.area_ha
    med = y0.groupby(crops.crop).transform("median")
    coconut = crops.crop == UNIT_ISSUE
    implausible = (y0 > IMPLAUSIBLE * med) & ~coconut
    crops.loc[coconut | implausible, "prod_t"] = np.nan
    print(f"production blanked: coconut {int(coconut.sum()):,} rows (unit = nuts); implausible yield {int(implausible.sum()):,} "
          f"rows ({implausible.mean():.2%}), worst crops {implausible.groupby(crops.crop).sum().nlargest(3).to_dict()}")
    assert implausible.mean() < 0.01, "too many implausible yields: check the source"
    crops["yield_t_ha"] = crops.prod_t / crops.area_ha
    crops["_med"] = med
    crops["price_intl_dollars_t"] = crops.crop.map(price)
    crops["value_const"] = crops.prod_t * crops.price_intl_dollars_t
    crops["is_missing"] = crops.prod_t.isna()
    crops = crops[["dist_code", "year", "crop", "area_ha", "prod_t", "yield_t_ha", "price_intl_dollars_t", "value_const",
                   "is_missing", "n_rows", "rule", "_med"]].sort_values(["dist_code", "year", "crop"]).reset_index(drop=True)

    # ---- assertions
    assert not crops.duplicated(["dist_code", "year", "crop"]).any()
    assert crops.year.between(YEARS[0], YEARS[-1]).all() and crops.area_ha.gt(0).all()
    assert crops.prod_t.dropna().ge(0).all() and np.isfinite(crops.yield_t_ha.dropna()).all()
    assert (crops.yield_t_ha.dropna() <= IMPLAUSIBLE * crops._med[crops.yield_t_ha.notna()] + 1e-9).all()
    crops = crops.drop(columns="_med")
    assert set(crops.dist_code) <= set(districts.dist_code)
    assert crops.crop.nunique() == len(fixed)

    # ---- national cross-check against FAOSTAT India (project's own table)
    fao = pd.read_parquet(paths.CROPS)
    fao = fao[fao.iso3 == "IND"]
    chk = []
    for crop, item in [("Rice", 27), ("Wheat", 15), ("Maize", 56), ("Groundnut", 242), ("Sugarcane", 156), ("Potato", 116), ("Gram", 191)]:
        n = crops[(crops.crop == crop) & crops.dist_code.isin(districts.dist_code[districts.stable])].groupby("year").agg(a=("area_ha", "sum"), p=("prod_t", "sum"))
        f = fao[fao.item_code == item].set_index("year")
        j = n.join(f[["area_ha", "prod_t"]], how="inner")
        chk.append((crop, len(j), float((j.a / j.area_ha).median()), float((j.p / j.prod_t).median())))
    chk = pd.DataFrame(chk, columns=["crop", "n_years", "area_ratio_DES_over_FAO", "prod_ratio_DES_over_FAO"])
    print("\nDES (stable districts only) vs FAOSTAT India, median ratio over years:\n", chk.round(2).to_string(index=False))
    cr = chk.set_index("crop")
    for crop in ("Rice", "Wheat"):
        assert 0.8 < cr.loc[crop, "area_ratio_DES_over_FAO"] < 1.2, f"{crop}: DES area differs from FAOSTAT India by >20%"
    assert 0.8 < cr.loc["Wheat", "prod_ratio_DES_over_FAO"] < 1.2, "Wheat: DES production differs from FAOSTAT India by >20%"
    print(f"NOTE rice: production ratio {cr.loc['Rice', 'prod_ratio_DES_over_FAO']:.2f} with area ratio {cr.loc['Rice', 'area_ratio_DES_over_FAO']:.2f} (milled vs paddy).")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    districts.reset_index(drop=True).to_csv(OUT_DIR / "districts.csv", index=False)
    crops.to_parquet(OUT_DIR / "crops.parquet", index=False)
    chk.to_csv(OUT_DIR / "des_vs_fao_check.csv", index=False)

    priced = crops.groupby("crop").area_ha.sum()
    share_priced = priced[priced.index.isin(price)].sum() / priced.sum()
    print(f"\ncrops: {len(fixed)} fixed ({len(price)} priced; priced share of area {share_priced:.1%}); unpriced: "
          f"{sorted(set(fixed) - set(price))}")
    print(f"rows {len(crops):,}; districts with any row {crops.dist_code.nunique()}; stable (25 yrs) {int(districts.stable.sum())}; "
          f"split-flagged among stable {int((districts.stable & districts.split_flag).sum())}; core {int(districts.core.sum())}")
    print(f"season rule: total {int((crops.rule == 'total').sum()):,} / sum {int((crops.rule == 'sum').sum()):,}; "
          f"production missing {crops.is_missing.mean():.2%}")
    print(districts[districts.stable & districts.split_flag].groupby("state_name").size().to_string())


if __name__ == "__main__":
    main()
