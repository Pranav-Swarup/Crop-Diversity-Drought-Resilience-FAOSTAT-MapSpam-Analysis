"""Crop production in kilocalories -> data/clean/calories.parquet and data/clean/calorie_factors.csv.

Source of the energy factors: FAOSTAT Supply Utilization Accounts (SUA_Crops_Livestock_E, 2010-2023),
elements 261 "Calories/Year" (million Kcal) and 5141 "Food supply quantity (tonnes)" (t). For one
country-year and item, the implied energy density is Calories/Year x 1e6 / Food supply quantity (kcal/t).
Elements 664 (kcal/capita/day) and 665 (g/capita/day) give a second, independent route to the same density
(kcal/cap/d / g/cap/d x 1e6) and are used only as a cross-check. Element 5510 (Production, t) is used only to
check that SUA and QCL record production of an item on the same weight basis.

Decisions:
  1. Item matching. SUA item codes are FAOSTAT item codes. A crop item of crops.parquet is matched to SUA
     by item code, and the script asserts that the SUA item name is identical to the crops.parquet name.
     For every in-scope item, the median over country-years of SUA production / QCL production must be
     within 1% of 1 (same weight basis, e.g. paddy rice for item 27). Crop items whose code is absent
     from SUA are out of scope.
  2. One time-invariant factor per item. kcal_per_t = median over country-years of the implied density.
     A factor that varied by country or year would put factor changes into the calorie series; FAOSTAT
     itself applies essentially one factor per item (the spread columns in calorie_factors.csv show
     this), so a single global median loses almost nothing. The quantity-weighted ratio
     (sum Calories / sum Food supply) is written alongside as `kcal_per_t_ratio` for comparison.
     The factors come from 2010-2023 accounts and are applied to production in every year 1993-2023.
  3. Degenerate ratios. A country-year is used only if Food supply quantity >= MIN_FOOD_T tonnes and
     Calories/Year > 0 (both finite). Small quantities carry rounding noise in the published figures.
     The "China" aggregate (area 351) is dropped so that mainland China is not counted twice.
  4. Scope. An item is in scope if its factor rests on at least MIN_COUNTRY_YEARS usable country-years
     from at least MIN_AREAS FAOSTAT areas, and at least MIN_AGREEMENT of those country-years lie within
     1% of the median. The factor is FAOSTAT's figure for the primary item as recorded in food supply: for
     crops used mostly after processing (oil crops, sugar crops, coffee, tea, cocoa) it is the energy FAOSTAT
     assigns to a tonne of the primary item, not the energy of the processed products. No conversion
     chains to derived products are built. Items without a factor (fibre crops, tobacco, rubber, oil palm
     fruit and others; see calorie_factors.csv `reason`) are left out of the calorie series.
  5. Coverage. value_coverage() gives, per country, the share of window-total crop value (value_const,
     rows with a value) held by in-scope items. src/metrics/calorie_faces.py reports it per country.
  6. kcal = prod_t x kcal_per_t, in kcal. It is the potential food energy of production, not food consumed:
     feed, seed, losses, processing losses, exports and non-food uses are not subtracted.
  7. Plausibility bounds (asserted): every factor in (0, 9.0e6] kcal/t (9 kcal/g is the energy density of
     pure fat, the highest of any food component); cereals in [2.5e6, 4.0e6] kcal/t (dry grains are about
     3,000-3,700 kcal/kg, paddy and hulled grains lower because the hull is counted in the weight);
     dry pulses in [2.8e6, 3.8e6] kcal/t.

calorie_factors.csv: one row per crop item of crops.parquet: item_code, item, kcal_per_t, kcal_per_t_ratio,
n_country_years, n_areas, p05, p25, p75, p95 (percentiles of the implied density, kcal/t), iqr_rel
((p75 - p25) / median), share_within_1pct, in_scope, reason. kcal_per_t is filled only for in-scope items.

Run: python -m src.crops.calories
"""
import numpy as np
import pandas as pd

from ..common import paths
from ..common.contract import load, summary
from . import faostat

CALORIES_YEAR, FOOD_T, KCAL_CAP_DAY, G_CAP_DAY, PRODUCTION = 261, 5141, 664, 665, 5510
SUA_ELEMENTS = [CALORIES_YEAR, FOOD_T, KCAL_CAP_DAY, G_CAP_DAY, PRODUCTION]
UNITS = {CALORIES_YEAR: "million Kcal", FOOD_T: "t", KCAL_CAP_DAY: "kcal/cap/d", G_CAP_DAY: "g/cap/d", PRODUCTION: "t"}
CHINA_AGGREGATE = 351

MIN_FOOD_T = 100.0         # t of food supply for a usable country-year
MIN_G_CAP_DAY = 1.0        # g/cap/d for the per-capita cross-check
MIN_COUNTRY_YEARS = 10
MIN_AREAS = 3
MIN_AGREEMENT = 0.9        # share of country-years within 1% of the median factor
AGREE_TOL = 0.01
BASIS_TOL = 0.01           # SUA vs QCL production, median ratio per item
CROSS_TOL = 0.01           # per-capita route vs factor
MIN_BASIS_ROWS = 10

FACTOR_MAX = 9.0e6
CEREALS = {15, 27, 44, 56, 71, 75, 79, 83, 89, 92, 94, 97, 101, 103, 108}
PULSES = {176, 181, 187, 191, 195, 197, 201, 203, 205, 210, 211}
CEREAL_BOUNDS = (2.5e6, 4.0e6)
PULSE_BOUNDS = (2.8e6, 3.8e6)

FACTORS_CSV = paths.CLEAN_REAL / "calorie_factors.csv"
COLUMNS = ["iso3", "year", "item_code", "item", "kcal", "is_imputed"]


def implied_factors(sua):
    """Per item: median implied kcal/t and its spread, from SUA-like rows.

    sua: DataFrame with area_code, item_code, year, calories_mkcal (million kcal), food_t (t).
    Rows with food_t < MIN_FOOD_T, calories <= 0 or non-finite values are ignored.
    Returns a DataFrame indexed by item_code.
    """
    ok = sua[np.isfinite(sua["food_t"]) & np.isfinite(sua["calories_mkcal"])
             & (sua["food_t"] >= MIN_FOOD_T) & (sua["calories_mkcal"] > 0)].copy()
    ok["kcal_per_t"] = ok["calories_mkcal"] * 1e6 / ok["food_t"]
    g = ok.groupby("item_code")
    f = g["kcal_per_t"].agg(kcal_per_t="median", n_country_years="size",
                            p05=lambda s: s.quantile(0.05), p25=lambda s: s.quantile(0.25),
                            p75=lambda s: s.quantile(0.75), p95=lambda s: s.quantile(0.95))
    f["kcal_per_t_ratio"] = g["calories_mkcal"].sum() * 1e6 / g["food_t"].sum()
    f["n_areas"] = g["area_code"].nunique()
    f["iqr_rel"] = (f["p75"] - f["p25"]) / f["kcal_per_t"]
    med = ok["item_code"].map(f["kcal_per_t"])
    f["share_within_1pct"] = ((ok["kcal_per_t"] - med).abs() <= AGREE_TOL * med).groupby(ok["item_code"]).mean()
    return f


def select_scope(f):
    """Add in_scope (bool) and reason (str) to an implied_factors() table."""
    f = f.copy()
    reason = pd.Series("", index=f.index)
    reason[f["share_within_1pct"] < MIN_AGREEMENT] = f"fewer than {MIN_AGREEMENT:.0%} of country-years within 1% of the median"
    reason[f["n_areas"] < MIN_AREAS] = f"fewer than {MIN_AREAS} areas"
    reason[f["n_country_years"] < MIN_COUNTRY_YEARS] = f"fewer than {MIN_COUNTRY_YEARS} usable country-years"
    f["in_scope"] = reason == ""
    f["reason"] = reason
    return f


def kcal_table(crops, factors):
    """crops.parquet rows of in-scope items with kcal = prod_t x kcal_per_t."""
    k = factors.loc[factors["in_scope"], "kcal_per_t"]
    out = crops[crops["item_code"].isin(k.index)].copy()
    out["kcal"] = out["prod_t"] * out["item_code"].map(k)
    return out[COLUMNS].astype({"kcal": float}).sort_values(["iso3", "item_code", "year"]).reset_index(drop=True)


def value_coverage(crops, in_scope_codes):
    """Per iso3: share of window-total value_const (rows with a value) from in-scope items."""
    c = crops[crops["value_const"].notna()]
    covered = c["value_const"].where(c["item_code"].isin(list(in_scope_codes)), 0.0)
    return (covered.groupby(c["iso3"]).sum() / c.groupby("iso3")["value_const"].sum()).rename("kcal_value_coverage")


def load_sua():
    sua = faostat.read_filtered(faostat.SUA, SUA_ELEMENTS)
    for el, unit in UNITS.items():
        u = set(sua.loc[sua["Element Code"] == el, "Unit"])
        assert u == {unit}, f"SUA element {el}: units {u}, expected {unit!r}"
    sua = sua[sua["Area Code"] != CHINA_AGGREGATE]
    assert not sua.duplicated(["Area Code", "Item Code", "Year", "Element Code"]).any(), "SUA: duplicate rows"
    return sua


def wide(sua, elements, names):
    w = sua[sua["Element Code"].isin(elements)].pivot(index=["Area Code", "Item Code", "Year"],
                                                      columns="Element Code", values="Value")
    w = w.reindex(columns=elements).reset_index()
    w.columns = ["area_code", "item_code", "year"] + names
    return w


def check_names(crop_items, sua):
    """Matched codes must carry identical names. Returns the set of crop item codes present in SUA."""
    sua_items = sua[["Item Code", "Item"]].drop_duplicates()
    assert not sua_items["Item Code"].duplicated().any(), "SUA: an item code with more than one name"
    m = crop_items.merge(sua_items, left_on="item_code", right_on="Item Code", how="inner")
    bad = m[m["item"] != m["Item"]]
    assert bad.empty, f"item codes whose SUA name differs from crops.parquet:\n{bad[['item_code', 'item', 'Item']]}"
    return set(m["item_code"])


def check_basis(sua, codes):
    """Median SUA production / QCL production per item (same area, item, year). Returns a table."""
    qcl = faostat.read(faostat.QCL, [5312, 5510])
    qcl = qcl[qcl["Element Code"] == PRODUCTION][["Area Code", "Item Code", "Year", "Value"]]
    s = sua[(sua["Element Code"] == PRODUCTION) & sua["Item Code"].isin(codes)][["Area Code", "Item Code", "Year", "Value"]]
    m = s.merge(qcl, on=["Area Code", "Item Code", "Year"], suffixes=("_sua", "_qcl"))
    m = m[m["Value_qcl"] >= MIN_FOOD_T]
    m["ratio"] = m["Value_sua"] / m["Value_qcl"]
    g = m.groupby("Item Code")["ratio"]
    return pd.DataFrame({"n": g.size(), "median_ratio": g.median(),
                         "share_within_1pct": g.apply(lambda r: ((r - 1).abs() <= 0.01).mean())})


def check_per_capita(sua, factors):
    """Median of kcal/cap/d / g/cap/d x 1e6 per item, against kcal_per_t. Returns relative differences."""
    w = wide(sua, [KCAL_CAP_DAY, G_CAP_DAY], ["kcal_cap_day", "g_cap_day"])
    w = w[(w["g_cap_day"] >= MIN_G_CAP_DAY) & (w["kcal_cap_day"] > 0)]
    alt = (w["kcal_cap_day"] / w["g_cap_day"] * 1e6).groupby(w["item_code"]).agg(["median", "size"])
    alt = alt[alt["size"] >= MIN_COUNTRY_YEARS]
    k = factors.loc[factors["in_scope"], "kcal_per_t"]
    common = alt.index.intersection(k.index)
    return ((alt.loc[common, "median"] - k[common]) / k[common]).abs()


def main():
    crops = load("crops")
    crop_items = crops.groupby(["item_code", "item"], as_index=False)["value_const"].sum()
    assert not crop_items["item_code"].duplicated().any()

    sua = load_sua()
    in_sua = check_names(crop_items[["item_code", "item"]], sua)
    print(f"crop items: {len(crop_items)}; item code present in SUA with an identical name: {len(in_sua)}")

    f = select_scope(implied_factors(wide(sua, [CALORIES_YEAR, FOOD_T], ["calories_mkcal", "food_t"])))
    f = f[f.index.isin(in_sua)]
    t = crop_items.set_index("item_code").join(f, how="left")
    t["in_scope"] = t["in_scope"].fillna(False).astype(bool)
    t.loc[~t.index.isin(in_sua), "reason"] = "item code not in SUA"
    t.loc[t.index.isin(in_sua) & t["n_country_years"].isna(), "reason"] = \
        f"no country-year with food supply >= {MIN_FOOD_T:g} t and calories > 0"
    t["n_country_years"] = t["n_country_years"].fillna(0).astype(int)
    t["n_areas"] = t["n_areas"].fillna(0).astype(int)
    t["reason"] = t["reason"].fillna("")
    t["kcal_per_t"] = t["kcal_per_t"].where(t["in_scope"])

    basis = check_basis(sua, list(t.index[t["in_scope"]]))
    basis_tested = basis[basis["n"] >= MIN_BASIS_ROWS]
    bad = basis_tested[(basis_tested["median_ratio"] - 1).abs() > BASIS_TOL]
    assert bad.empty, f"SUA and QCL production differ in basis for items:\n{bad}"
    print(f"production basis: SUA / QCL median ratio within {BASIS_TOL:.0%} of 1 for all {len(basis_tested)} in-scope items "
          f"with >= {MIN_BASIS_ROWS} matched rows (range {basis_tested['median_ratio'].min():.5f}-"
          f"{basis_tested['median_ratio'].max():.5f}); median share of country-years within 1%: "
          f"{basis_tested['share_within_1pct'].median():.2f}; in-scope items not tested: "
          f"{sorted(set(t.index[t['in_scope']]) - set(basis_tested.index))}")
    cross = check_per_capita(sua, t)
    assert (cross <= CROSS_TOL).all(), f"per-capita route disagrees with kcal_per_t:\n{cross[cross > CROSS_TOL]}"
    print(f"per-capita cross-check (kcal/cap/d / g/cap/d): {len(cross)} in-scope items, max relative difference {cross.max():.5f}")

    factors = t.reset_index()[["item_code", "item", "kcal_per_t", "kcal_per_t_ratio", "n_country_years", "n_areas",
                               "p05", "p25", "p75", "p95", "iqr_rel", "share_within_1pct", "in_scope", "reason"]]
    factors = factors.sort_values("item_code").reset_index(drop=True)

    # Assertions on the factors.
    k = factors.loc[factors["in_scope"]].set_index("item_code")["kcal_per_t"]
    assert np.isfinite(k).all() and (k > 0).all(), "in-scope item without a positive finite factor"
    assert (k <= FACTOR_MAX).all(), f"factor above {FACTOR_MAX:g} kcal/t: {k[k > FACTOR_MAX].to_dict()}"
    for codes, (lo, hi), name in [(CEREALS, CEREAL_BOUNDS, "cereal"), (PULSES, PULSE_BOUNDS, "pulse")]:
        kk = k[k.index.isin(codes)]
        assert kk.between(lo, hi).all(), f"{name} factor outside [{lo:g}, {hi:g}]: {kk[~kk.between(lo, hi)].to_dict()}"
    assert factors.loc[~factors["in_scope"], "kcal_per_t"].isna().all()
    assert (factors.loc[~factors["in_scope"], "reason"] != "").all() and (factors.loc[factors["in_scope"], "reason"] == "").all()
    assert not factors["item_code"].duplicated().any()

    kcal = kcal_table(crops, factors.set_index("item_code"))
    assert not kcal.duplicated(["iso3", "year", "item_code"]).any(), "calories: duplicate keys"
    assert kcal["iso3"].str.fullmatch(r"[A-Z]{3}").all() and kcal["year"].between(1993, 2023).all()
    assert np.isfinite(kcal["kcal"]).all() and (kcal["kcal"] >= 0).all(), "calories: kcal negative or not finite"
    assert pd.api.types.is_bool_dtype(kcal["is_imputed"]) and pd.api.types.is_integer_dtype(kcal["item_code"])
    assert len(kcal) == crops["item_code"].isin(k.index).sum()

    paths.CLEAN_REAL.mkdir(parents=True, exist_ok=True)
    kcal.to_parquet(paths.CALORIES, index=False)
    factors.to_csv(FACTORS_CSV, index=False)

    cov = value_coverage(crops, k.index)
    quality = load("quality")
    cov_inc = cov[cov.index.isin(quality.loc[quality["included"], "iso3"])]
    print(summary(kcal, "calories"))
    print(f"items in scope: {int(factors['in_scope'].sum())} / {len(factors)}; "
          f"share of global crop value in scope: {crop_items.set_index('item_code').loc[k.index, 'value_const'].sum() / crop_items['value_const'].sum():.3f}")
    print(f"factor spread over in-scope items: max iqr_rel {factors.loc[factors['in_scope'], 'iqr_rel'].max():.2e}, "
          f"min share_within_1pct {factors.loc[factors['in_scope'], 'share_within_1pct'].min():.3f}")
    print(f"value coverage, included countries (n={len(cov_inc)}): median {cov_inc.median():.3f}, "
          f"min {cov_inc.min():.3f} ({cov_inc.idxmin()}), 10th pct {cov_inc.quantile(0.1):.3f}; all countries median {cov.median():.3f}")
    out = factors[~factors["in_scope"]]
    print(f"out of scope ({len(out)}):")
    for r in out.itertuples():
        print(f"  {r.item_code:>5} {r.item[:60]:<60} {r.reason}")
    print(f"wrote {paths.CALORIES.relative_to(paths.ROOT)} and {FACTORS_CSV.relative_to(paths.ROOT)}")


if __name__ == "__main__":
    if paths.USE_DUMMY:
        raise SystemExit("src.crops.calories has no dummy version (it reads the FAOSTAT SUA file); run without USE_DUMMY=1.")
    main()
