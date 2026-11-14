"""Unit tests for src/crops/calories.py and src/metrics/calorie_faces.py on synthetic inputs.

Run: python -m pytest src/metrics/tests/test_calories.py -q
"""
import numpy as np
import pandas as pd
import pytest

from src.crops import calories
from src.metrics import calorie_faces
from src.metrics.run import YEARS, country_metrics


def sua_rows(item_code, kcal_per_t, quantities, area0=1):
    q = np.asarray(quantities, dtype=float)
    return pd.DataFrame({"area_code": area0 + np.arange(len(q)), "item_code": item_code, "year": 2015,
                         "food_t": q, "calories_mkcal": q * kcal_per_t / 1e6})


def test_factor_recovers_known_kcal_per_t_and_ignores_degenerate_rows():
    good = sua_rows(15, 3.336e6, np.linspace(1_000, 50_000, 12))
    degenerate = pd.DataFrame({
        "area_code": [101, 102, 103, 104, 105], "item_code": 15, "year": 2015,
        "food_t": [0.0, 5.0, 50.0, 2_000.0, np.nan],          # zero, below MIN_FOOD_T (x2), ok, missing
        "calories_mkcal": [3.0, 900.0, 1.0, 0.0, 10.0],        # wild ratios, or zero calories
    })
    f = calories.implied_factors(pd.concat([good, degenerate], ignore_index=True))
    assert f.loc[15, "kcal_per_t"] == pytest.approx(3.336e6, rel=1e-12)
    assert f.loc[15, "kcal_per_t_ratio"] == pytest.approx(3.336e6, rel=1e-12)
    assert f.loc[15, "n_country_years"] == 12
    assert f.loc[15, "n_areas"] == 12
    assert f.loc[15, "share_within_1pct"] == pytest.approx(1.0)
    assert f.loc[15, "iqr_rel"] == pytest.approx(0.0, abs=1e-12)


def test_factor_is_median_and_scope_rules():
    rows = pd.concat([
        sua_rows(1, 1.0e6, [1_000] * 9 + [1_000], area0=1),      # 10 country-years, 10 areas
        sua_rows(2, 2.0e6, [1_000] * 5),                        # too few country-years
        sua_rows(3, 3.0e6, [1_000] * 10).assign(area_code=[1, 2] * 5),   # only 2 areas
    ], ignore_index=True)
    rows.loc[9, "calories_mkcal"] = 1_000 * 5.0e6 / 1e6          # one outlier in item 1
    f = calories.select_scope(calories.implied_factors(rows))
    assert f.loc[1, "kcal_per_t"] == pytest.approx(1.0e6)       # median is not moved by the outlier
    assert f.loc[1, "kcal_per_t_ratio"] > 1.0e6                 # the quantity-weighted ratio is
    assert f.loc[1, "share_within_1pct"] == pytest.approx(0.9)
    assert f.loc[1, "in_scope"]
    assert not f.loc[2, "in_scope"] and "country-years" in f.loc[2, "reason"]
    assert not f.loc[3, "in_scope"] and "areas" in f.loc[3, "reason"]


def synthetic_crops():
    rng = np.random.default_rng(1)
    rows = []
    for code, item, prod0, price in [(15, "Wheat", 1000.0, 200.0), (27, "Rice", 500.0, 300.0), (826, "Unmanufactured tobacco", 50.0, 3000.0)]:
        prod = prod0 * (1 + 0.01 * (YEARS - YEARS[0])) * (1 + rng.normal(0, 0.08, len(YEARS)))
        rows.append(pd.DataFrame({"iso3": "AAA", "year": YEARS, "item_code": code, "item": item, "area_ha": prod / 3,
                                  "prod_t": prod, "yield_t_ha": 3.0, "value_const": prod * price,
                                  "is_imputed": YEARS % 7 == 0}))
    return pd.concat(rows, ignore_index=True)


FACTORS = pd.DataFrame({"kcal_per_t": [3.336e6, 2.68e6, np.nan], "in_scope": [True, True, False]},
                       index=pd.Index([15, 27, 826], name="item_code"))


def test_national_kcal_equals_hand_computation():
    crops = synthetic_crops()
    k = calories.kcal_table(crops, FACTORS)
    assert set(k["item_code"]) == {15, 27}
    assert list(k.columns) == calories.COLUMNS
    y = 2005
    w = crops[(crops["year"] == y) & (crops["item_code"] == 15)].iloc[0]
    r = crops[(crops["year"] == y) & (crops["item_code"] == 27)].iloc[0]
    expected = w["prod_t"] * 3.336e6 + r["prod_t"] * 2.68e6
    assert k[k["year"] == y]["kcal"].sum() == pytest.approx(expected, rel=1e-12)
    assert (k.set_index(["year", "item_code"])["is_imputed"]
            == crops.set_index(["year", "item_code"]).loc[k.set_index(["year", "item_code"]).index, "is_imputed"]).all()
    cov = calories.value_coverage(crops, [15, 27])
    v = crops.groupby("item_code")["value_const"].sum()
    assert cov["AAA"] == pytest.approx((v[15] + v[27]) / v.sum())


def test_faces_on_kcal_equal_faces_on_same_series_as_value():
    crops = synthetic_crops()
    k = calories.kcal_table(crops, FACTORS)
    rng = np.random.default_rng(2)
    spei = pd.Series(rng.normal(0, 1, len(YEARS)), index=YEARS)
    spei[[2000, 2010, 2015]] = [-1.6, -1.2, -2.1]
    events = [2000, 2010, 2015]

    m_kcal = calorie_faces.national_faces(k.groupby("year")["kcal"].sum(), spei, events)
    # The same national series passed through the value pipeline (kcal in the value_const column).
    as_value = crops[crops["item_code"].isin([15, 27])].merge(k[["year", "item_code", "kcal"]], on=["year", "item_code"])
    as_value["value_const"] = as_value["kcal"]
    empty = pd.DataFrame(columns=["year", "mapspam_crop", "crop_spei12_w"])
    m_value, *_ = country_metrics(as_value, spei, empty, events)
    for f in calorie_faces.FACES + ["n_events"]:
        assert m_kcal[f] == pytest.approx(m_value[f], nan_ok=True), f
    assert m_kcal["n_events"] == 3 and np.isfinite(m_kcal["resistance"])


def test_compute_masks_low_coverage_countries():
    crops = synthetic_crops()
    k = calories.kcal_table(crops, FACTORS)
    k2 = k.assign(iso3="BBB")
    climate = pd.DataFrame({"iso3": "AAA", "year": YEARS, "spei12_w": 0.0})
    events = pd.DataFrame({"iso3": pd.Series(dtype=str), "year": pd.Series(dtype=int)})
    cov = pd.Series({"AAA": 0.9, "BBB": 0.3})
    out = calorie_faces.compute(pd.concat([k, k2]), cov, climate, events, ["AAA", "BBB"])
    assert list(out.columns) == calorie_faces.COLUMNS
    a, b = out.set_index("iso3").loc["AAA"], out.set_index("iso3").loc["BBB"]
    assert np.isfinite(a["stability_kcal"]) and np.isnan(a["resistance_kcal"])
    assert b[calorie_faces.KCAL_FACES].isna().all() and b["n_events_kcal"] == 0
