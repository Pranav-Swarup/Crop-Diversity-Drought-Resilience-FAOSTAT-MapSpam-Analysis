"""Unit tests on synthetic series with known answers. Run: python -m pytest src/metrics"""
import numpy as np
import pandas as pd
import pytest

from src.metrics import decompose, faces, response
from src.metrics.detrend import detrend, lowess_trend
from src.metrics.diversity import hill1
from src.metrics.events import find_events

YEARS = np.arange(1993, 2024)


def test_detrend_linear_trend_plus_noise_gives_zero_mean_anomalies():
    rng = np.random.default_rng(0)
    y = (100 + 3 * (YEARS - 1993)) * (1 + rng.normal(0, 0.05, len(YEARS)))
    d = detrend(y, YEARS)
    assert abs(d["anom"].mean()) < 0.02
    assert 0.03 < d["anom"].std() < 0.07
    assert np.allclose(d["index"], 1 + d["anom"])


def test_detrend_exact_line_has_no_anomaly():
    d = detrend(50 + 2.0 * (YEARS - 1993), YEARS)
    assert np.allclose(d["anom"], 0, atol=1e-9)


def test_detrend_skips_nan_and_short_series():
    y = 50 + 2.0 * (YEARS - 1993)
    y[5] = np.nan
    t = lowess_trend(y, YEARS)
    assert np.isnan(t[5]) and np.isfinite(np.delete(t, 5)).all()
    assert np.isnan(lowess_trend(np.arange(5.0))).all()


def test_hill1_even_and_single_crop():
    assert hill1([10, 10, 10, 10]) == pytest.approx(4.0)
    assert hill1([7, 0, 0]) == pytest.approx(1.0)
    assert 1 < hill1([90, 5, 5]) < 2


def test_resistance_and_recovery_known_drop_and_rebound():
    index = pd.Series(1.0, index=YEARS)
    index[2005] = 0.7                      # 30% drop
    index[[2006, 2007, 2008]] = [0.9, 1.0, 1.1]
    assert faces.resistance(index, [2005]) == pytest.approx(0.7)
    assert faces.recovery(index, [2005]) == pytest.approx(1.0 / 0.7)
    assert faces.n_usable_events(index, [2005]) == 1


def test_resistance_is_median_over_events_and_nan_without_events():
    index = pd.Series(1.0, index=YEARS)
    index[2000], index[2008], index[2016] = 0.5, 0.8, 0.9
    assert faces.resistance(index, [2000, 2008, 2016]) == pytest.approx(0.8)
    assert np.isnan(faces.resistance(index, []))
    assert np.isnan(faces.recovery(index, [2022]))  # no 3 years after inside the series


def test_vulnerability_shares():
    index = pd.Series(1.0, index=YEARS)
    index[[2000, 2001, 2010]] = [0.85, 0.89, 0.95]
    assert faces.vulnerability(index) == pytest.approx(2 / 31)
    assert faces.vulnerability(index, [2000, 2010]) == pytest.approx(0.5)
    assert np.isnan(faces.vulnerability(index, []))


def test_stability_is_inverse_cv():
    y = np.array([90.0, 110.0, 90.0, 110.0])
    resid = y - 100
    assert faces.stability(y, resid) == pytest.approx(100 / np.std(resid, ddof=1))


def test_find_events_rule():
    spei = pd.Series(0.0, index=YEARS)
    spei[1994] = -2.0                  # too close to the window start
    spei[[2002, 2003]] = [-1.2, -1.6]  # run: only the first year is an event
    spei[2005] = -1.1                  # drought within the 3 prior years (2003)
    spei[2012] = -2.3                  # clean event
    spei[2022] = -1.5                  # too close to the window end
    ev = find_events(pd.DataFrame({"iso3": "AAA", "year": YEARS, "spei12_w": spei.to_numpy()}))
    assert ev["year"].tolist() == [2002, 2012]
    assert ev["severity"].tolist() == ["moderate", "extreme"]


def test_resp_beta_recovers_slope():
    rng = np.random.default_rng(1)
    spei = rng.normal(size=31)
    assert response.resp_beta(0.08 * spei, spei) == pytest.approx(0.08)
    assert np.isnan(response.resp_beta(0.08 * spei[:19], spei[:19]))  # fewer than 20 years


def test_resp_div_weighted_sd():
    assert response.resp_div([0.1, 0.1, 0.1], [0.5, 0.3, 0.2]) == pytest.approx(0.0)
    assert response.resp_div([0.0, 0.2], [0.5, 0.5]) == pytest.approx(0.1)
    assert response.resp_div([0.0, 0.2], [1.0, 3.0]) == pytest.approx(np.sqrt(0.25 * 0.15**2 + 0.75 * 0.05**2))
    assert np.isnan(response.resp_div([0.1], [1.0]))


def test_resp_divergence_ross_2023():
    assert response.resp_divergence([0.1, 0.3]) == pytest.approx(0.0)      # same sign
    assert response.resp_divergence([-0.2, 0.2]) == pytest.approx(1.0)     # symmetric around zero
    assert response.resp_divergence([-0.1, 0.0, 0.3]) == pytest.approx((0.4 - 0.2) / 0.4)
    assert response.resp_divergence([0.2, 0.2]) == 0.0
    assert np.isnan(response.resp_divergence([0.2]))


def test_decomposition_identity_and_limits():
    rng = np.random.default_rng(2)
    values = 100 + rng.normal(0, 5, size=(31, 6))
    resid = values - values.mean(axis=0)
    d = decompose.decompose(values, resid)
    assert d["cv_total"] == pytest.approx(np.sqrt(d["phi_sync"]) * d["mean_crop_cv"], abs=1e-12)
    assert 0 < d["phi_sync"] < 1
    # Perfectly synchronous crops: phi = 1.
    common = rng.normal(0, 5, 31)
    v = np.column_stack([100 + common, 200 + 2 * common])
    assert decompose.decompose(v, v - v.mean(axis=0))["phi_sync"] == pytest.approx(1.0)
    # Two crops that cancel exactly: phi = 0.
    v = np.column_stack([100 + common, 100 - common])
    assert decompose.decompose(v, v - v.mean(axis=0))["phi_sync"] == pytest.approx(0.0, abs=1e-12)
