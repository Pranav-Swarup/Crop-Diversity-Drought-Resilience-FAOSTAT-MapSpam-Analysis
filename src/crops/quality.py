"""Inclusion rules (CLAUDE.md "Country inclusion") -> data/clean/quality.csv.

First-iteration pipeline written from src/metrics' side; Gursahib owns this module.
A year counts as non-imputed when at most half of that year's value comes from rows
flagged E or I.

Flat-series screen (flat series fake stability): a crop series (production, tonnes) is listed in
data/clean/flat_series.csv when its LOWESS-detrended CV is < 1%, or when it repeats the same value
3+ years running AND more than half of its years are imputed. Runs in genuinely reported series are
kept. The list is a record only: crops.parquet and the inclusion rules above are not changed by it.

Run: python -m src.crops.quality
"""
import pandas as pd

from ..common import paths
from ..common.contract import load, summary, validate
from ..metrics.detrend import lowess_trend
from ..metrics.faces import cv_resid

MIN_AREA_HA, MIN_CROPS, MIN_GOOD_YEARS = 100_000, 5, 25
MAX_IMPUTED_VALUE_SHARE = 0.5
FLAT_MIN_YEARS, FLAT_CV, FLAT_RUN, FLAT_IMPUTED_SHARE = 10, 0.01, 3, 0.5
FLAT_SERIES = paths.CLEAN / "flat_series.csv"
BREAK_YEAR = 2018


def max_run(v, years):
    """Longest run of identical values in consecutive years."""
    best = run = 1
    for i in range(1, len(v)):
        run = run + 1 if v[i] == v[i - 1] and years[i] == years[i - 1] + 1 else 1
        best = max(best, run)
    return best


def flat_series(crops, q):
    rows = []
    for (iso3, code), g in crops[crops["prod_t"] > 0].sort_values("year").groupby(["iso3", "item_code"]):
        if len(g) < FLAT_MIN_YEARS:
            continue
        y, years = g["prod_t"].to_numpy(float), g["year"].to_numpy()
        cv = cv_resid(y, y - lowess_trend(y, years))
        run, imp = max_run(y, years), g["is_imputed"].mean()
        reasons = []
        if cv < FLAT_CV:
            reasons.append("detrended CV < 1%")
        if run >= FLAT_RUN and imp > FLAT_IMPUTED_SHARE:
            reasons.append(f"same value {run} years running, {100 * imp:.0f}% of years imputed")
        if reasons:
            rows.append((iso3, code, g["item"].iloc[0], len(g), cv, run, imp, g["value_const"].mean(), "; ".join(reasons)))
    out = pd.DataFrame(rows, columns=["iso3", "item_code", "item", "n_years", "detrended_cv", "max_identical_run",
                                      "share_years_imputed", "mean_value_const", "reason"])
    out = out.merge(q[["iso3", "included"]].rename(columns={"included": "country_included"}), on="iso3")
    assert not out.duplicated(["iso3", "item_code"]).any()
    assert out["detrended_cv"].ge(0).all() and out["share_years_imputed"].between(0, 1).all()
    return out


def included_at(crops, max_share):
    """Number of countries passing the three inclusion rules at a given imputed-value cut-off."""
    g = crops[crops["value_const"].notna()]
    by = g.assign(imp=g["value_const"] * g["is_imputed"]).groupby(["iso3", "year"])[["imp", "value_const", "area_ha"]].sum()
    good = (by["imp"] / by["value_const"] <= max_share).groupby("iso3").sum() >= MIN_GOOD_YEARS
    return int((good & (by["area_ha"].groupby("iso3").mean() >= MIN_AREA_HA)
                & (g.groupby("iso3")["item_code"].nunique() >= MIN_CROPS)).sum())


def vanished_share(crops, year=BREAK_YEAR, span=4):
    """Per country, % of the value of the `span` years before `year` held by crops with no rows in the
    `span` years from `year`. FAOSTAT stops reporting some minor crops for EU countries from 2018."""
    g = crops[crops["value_const"].notna()]
    pre, post = g[g["year"].between(year - span, year - 1)], g[g["year"].between(year, year + span - 1)]
    still = post[["iso3", "item_code"]].drop_duplicates().assign(still=True)
    pre = pre.merge(still, on=["iso3", "item_code"], how="left")
    gone = pre[pre["still"].isna()].groupby("iso3")["value_const"].sum()
    return (100 * gone / pre.groupby("iso3")["value_const"].sum()).fillna(0.0)


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
    flat = flat_series(crops, q)
    flat.to_csv(FLAT_SERIES, index=False)
    total_series = crops.loc[crops["prod_t"] > 0].groupby(["iso3", "item_code"]).ngroups
    print(f"flat series listed in {FLAT_SERIES.name}: {len(flat)} of {total_series} crop series "
          f"({flat['country_included'].sum()} in included countries; "
          f"{100 * flat['mean_value_const'].sum() / crops.groupby(['iso3', 'item_code'])['value_const'].mean().sum():.1f}% of value)")
    print(flat["reason"].str.split(";").str[0].str.replace(r"\d+", "N", regex=True).value_counts().to_string())
    print("exclusion reasons (a country can have several):")
    for r in ["harvested area", "fewer than 5 crops", "non-imputed years", "no crop data"]:
        print(f"  {r}: {q['reason'].str.contains(r).sum()}")
    assert included_at(crops, MAX_IMPUTED_VALUE_SHARE) == len(inc)
    print("included at other imputed-value cut-offs: " + ", ".join(f"{int(100 * s)}% -> {included_at(crops, s)}" for s in (0.25, 0.5, 0.75)))
    gone = vanished_share(crops).reindex(inc["iso3"]).fillna(0.0).sort_values(ascending=False)
    print(f"included countries where crops holding > 1% of {BREAK_YEAR - 4}-{BREAK_YEAR - 1} value have no rows from {BREAK_YEAR}: "
          f"{(gone > 1).sum()} (largest: " + ", ".join(f"{k} {v:.1f}%" for k, v in gone.head(6).items()) + ")")
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
