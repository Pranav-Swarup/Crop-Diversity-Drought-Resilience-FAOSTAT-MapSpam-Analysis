"""Synthetic tests for src.metrics.oos. Run: python -m pytest src/metrics/tests/test_oos.py"""
import numpy as np
import pandas as pd
import pytest

from src.metrics import oos, response

YEARS = np.arange(1993, 2024)
RNG_SPEI = np.random.default_rng(1).normal(0, 1, len(YEARS))


def make_series(item, anom, iso3="AAA", share=0.5, spei=RNG_SPEI, years=YEARS):
    return pd.DataFrame({"iso3": iso3, "item": item, "year": years, "yield_anom": anom,
                         "spei": spei, "value_share": share})


def test_exact_linear_crop_recovers_slope_and_perfect_skill():
    s = make_series("Wheat", 0.07 * RNG_SPEI)
    slopes = oos.train_slopes(s)
    crop = oos.crop_table(s, slopes)
    assert slopes["beta_train"].iloc[0] == pytest.approx(0.07)
    assert crop["beta_train"].iloc[0] == pytest.approx(0.07)
    assert crop["beta_test"].iloc[0] == pytest.approx(0.07)
    assert crop["skill"].iloc[0] == pytest.approx(1.0)
    assert crop["r"].iloc[0] == pytest.approx(1.0)


def test_pure_noise_has_no_skill():
    rng = np.random.default_rng(5)
    skills = []
    for k in range(30):
        s = make_series("Noise", rng.normal(0, 0.1, len(YEARS)), spei=rng.normal(0, 1, len(YEARS)))
        skills.append(oos.crop_table(s, oos.train_slopes(s))["skill"].iloc[0])
    assert np.median(skills) < 0.05


def test_crop_with_too_few_training_years_is_dropped():
    anom = 0.07 * RNG_SPEI
    short = make_series("Short", anom).copy()
    short.loc[short["year"].between(1993, 2012)] = short.loc[short["year"].between(1993, 2012)]
    short.loc[short["year"] <= 2006, "yield_anom"] = np.nan  # leaves 6 training years (2007-2012)
    ok = make_series("Full", anom)
    slopes = oos.train_slopes(pd.concat([short, ok], ignore_index=True))
    assert list(slopes["item"]) == ["Full"]


def test_national_aggregation_matches_hand_computation():
    a = make_series("A", 0.10 * RNG_SPEI, share=0.6)
    b = make_series("B", -0.05 * RNG_SPEI + 0.01, share=0.2)
    s = pd.concat([a, b], ignore_index=True)
    slopes = oos.train_slopes(s)
    national = pd.DataFrame({"iso3": "AAA", "year": YEARS, "anom": 0.0, "is_drought": False})
    nat = oos.national_table(s, slopes, national).set_index("year")
    sA, sB = 0.6 / 0.8, 0.2 / 0.8
    bA, bB = slopes.set_index("item").loc["A", "beta_train"], slopes.set_index("item").loc["B", "beta_train"]
    y, i = 2015, list(YEARS).index(2015)
    assert nat.loc[y, "pred"] == pytest.approx(sA * bA * RNG_SPEI[i] + sB * bB * RNG_SPEI[i])
    assert nat.loc[y, "actual"] == pytest.approx(sA * 0.10 * RNG_SPEI[i] + sB * (-0.05 * RNG_SPEI[i] + 0.01))
    assert nat.loc[y, "covered_share"] == pytest.approx(0.8)
    assert nat.index.min() == oos.TEST[0] and nat.index.max() == oos.TEST[1]


def test_full_window_slope_equals_resp_beta():
    a = make_series("A", 0.1 * RNG_SPEI + np.random.default_rng(2).normal(0, 0.02, len(YEARS)))
    anomalies = pd.DataFrame({"iso3": "AAA", "item": "A", "year": YEARS,
                              "resp_beta": response.resp_beta(a["yield_anom"], a["spei"])})
    n, diff = oos.full_window_check(a, anomalies)
    assert n == 1 and diff < 1e-9


def test_holdout_cannot_change_fitted_trend_or_training_anomalies():
    years = np.arange(1993, 2024)
    values = 30.0 + np.arange(len(years)) * 0.4 + np.sin(years)
    original = oos.training_anomaly(values, years)
    holdout = years > oos.TRAIN[1]
    values[holdout] *= 3
    changed = oos.training_anomaly(values, years)
    np.testing.assert_array_equal(original[~holdout], changed[~holdout])
    np.testing.assert_allclose(changed[holdout] + 1, 3 * (original[holdout] + 1))


def test_merged_crop_exposure_requires_every_positive_area_component():
    from src.metrics.crop_classes import COFFEE, class_label
    from src.metrics.run import add_merged_exposures, crop_exposure

    areas = pd.DataFrame({"iso3": ["A", "A"], "mapspam_crop": COFFEE, "area_ha": [1e6, 1.0]})
    climate = pd.DataFrame({"iso3": ["A", "A", "A"], "mapspam_crop": [*COFFEE, COFFEE[0]],
                            "year": [2000, 2000, 2001], "crop_spei12_w": [0.1, 0.9, 0.3]})
    merged = add_merged_exposures(climate, areas)
    pair = merged.loc[merged.mapspam_crop.eq(class_label(COFFEE))].set_index("year")
    assert pair.loc[2000, "crop_spei12_w"] == pytest.approx((0.1 * 1e6 + 0.9) / (1e6 + 1))
    assert pd.isna(pair.loc[2001, "crop_spei12_w"])
    assert crop_exposure(-1, pd.DataFrame({"REST": [1.2]}, index=[2001]), [2001]).isna().all()
