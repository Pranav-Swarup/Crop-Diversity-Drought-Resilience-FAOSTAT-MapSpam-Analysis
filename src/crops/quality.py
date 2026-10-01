"""Inclusion rules (CLAUDE.md "Country inclusion") -> data/clean/quality.csv.

First-iteration pipeline written from src/metrics' side; Gursahib owns this module.
A year counts as non-imputed when at most half of that year's value comes from rows
flagged E or I. The flat-series screen from Gursahib's prompt is not implemented here.

Run: python -m src.crops.quality
"""
import pandas as pd

from ..common import paths
from ..common.contract import load, summary, validate

MIN_AREA_HA, MIN_CROPS, MIN_GOOD_YEARS = 100_000, 5, 25
MAX_IMPUTED_VALUE_SHARE = 0.5


def main():
    crops, countries = load("crops"), load("countries")
    rows = []
    for iso3 in countries["iso3"]:
        g = crops[(crops["iso3"] == iso3) & crops["value_const"].notna()]
        if g.empty:
            rows.append((iso3, 0, float("nan"), False, "no crop data"))
            continue
        n_crops = int(g["item_code"].nunique())
        mean_area = g.groupby("year")["area_ha"].sum().mean()
        share_imputed = g.loc[g["is_imputed"], "value_const"].sum() / g["value_const"].sum()
        by_year = g.assign(imp=g["value_const"] * g["is_imputed"]).groupby("year")[["imp", "value_const"]].sum()
        good_years = int((by_year["imp"] / by_year["value_const"] <= MAX_IMPUTED_VALUE_SHARE).sum())
        reasons = []
        if mean_area < MIN_AREA_HA:
            reasons.append("mean harvested area < 100,000 ha")
        if n_crops < MIN_CROPS:
            reasons.append("fewer than 5 crops")
        if good_years < MIN_GOOD_YEARS:
            reasons.append(f"fewer than 25 non-imputed years ({good_years})")
        rows.append((iso3, n_crops, share_imputed, not reasons, "; ".join(reasons)))
    q = pd.DataFrame(rows, columns=["iso3", "n_crops", "share_imputed", "included", "reason"])

    validate(q, "quality")
    assert set(q["iso3"]) == set(countries["iso3"])
    q.to_csv(paths.QUALITY, index=False)
    inc = q[q["included"]]
    print(summary(q, "quality"))
    print(f"included: {len(inc)}, excluded: {len(q) - len(inc)}")
    print("exclusion reasons (a country can have several):")
    for r in ["harvested area", "fewer than 5 crops", "non-imputed years", "no crop data"]:
        print(f"  {r}: {q['reason'].str.contains(r).sum()}")
    print(f"crops per included country (median): {int(inc['n_crops'].median())}")
    total = crops["value_const"].sum()
    print(f"value imputed overall: {100 * crops.loc[crops['is_imputed'], 'value_const'].sum() / total:.1f}%; "
          f"included countries hold {100 * crops.loc[crops['iso3'].isin(inc['iso3']), 'value_const'].sum() / total:.1f}% of global value")
    print("included by region:")
    print(countries[countries["iso3"].isin(inc["iso3"])]["region"].value_counts().to_string())
    big = q[~q["included"]].merge(crops.groupby("iso3")["value_const"].sum().rename("v"), on="iso3").nlargest(15, "v")
    print("largest excluded producers: " + ", ".join(f"{r.iso3} ({r.reason})" for r in big.itertuples()))


if __name__ == "__main__":
    main()
