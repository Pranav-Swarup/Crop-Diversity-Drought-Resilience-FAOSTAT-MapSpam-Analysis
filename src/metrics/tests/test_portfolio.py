"""Tests of src.metrics.portfolio on synthetic inputs with known answers.
Run: python -m pytest src/metrics/tests/test_portfolio.py -q"""
import numpy as np
import pandas as pd
import pytest

from src.metrics import portfolio as P

YEARS = np.arange(1993, 2024)


def test_two_uncorrelated_equal_risk_crops_split_half_half():
    mu, cov = np.array([1000.0, 1000.0]), np.diag([0.01, 0.01])
    w = P.min_risk(mu, cov, np.zeros(2), np.ones(2), starts=[np.array([0.2, 0.8])])
    assert w == pytest.approx([0.5, 0.5], abs=1e-6)
    # Same answer inside the default bounds around w0 = (0.45, 0.55).
    lo, hi = P.share_bounds([0.45, 0.55])
    w = P.min_risk(mu, cov, lo, hi, starts=[np.array([0.45, 0.55])])
    assert w == pytest.approx([0.5, 0.5], abs=1e-6)
    assert P.evaluate(w, mu, cov, [np.nan, np.nan])["risk_sd"] == pytest.approx(0.1 / np.sqrt(2))


def test_perfectly_correlated_crops_give_no_diversification_gain():
    mu, sd = np.array([800.0, 1200.0, 500.0]), np.array([0.05, 0.10, 0.20])
    cov = np.outer(sd, sd)                                       # correlation 1
    rng = np.random.default_rng(3)
    for _ in range(20):
        w = rng.dirichlet(np.ones(3))
        s = w * mu / (w @ mu)
        assert P.evaluate(w, mu, cov, [0, 0, 0])["risk_sd"] == pytest.approx(s @ sd)   # weighted mean of SDs
    # Equal SDs: risk is the same for every allocation, so the frontier is flat at that SD.
    cov = np.full((3, 3), 0.08 ** 2)
    w0 = np.array([0.5, 0.3, 0.2])
    fr = P.frontier(mu, cov, w0, *P.share_bounds(w0), n_points=5)
    risks = [P.evaluate(w, mu, cov, [0, 0, 0])["risk_sd"] for w in fr["weights"]]
    assert risks == pytest.approx([0.08] * 5)
    assert P.evaluate(fr["w_opt"], mu, cov, [0, 0, 0])["risk_sd"] == pytest.approx(0.08)


def test_evaluate_at_current_weights_equals_direct_calculation():
    rng = np.random.default_rng(4)
    anom = rng.normal(0, [0.05, 0.12, 0.08, 0.2], size=(31, 4))
    mu, w0 = np.array([900.0, 400.0, 2500.0, 150.0]), np.array([0.4, 0.3, 0.2, 0.1])
    beta = np.array([0.05, 0.10, -0.02, 0.0])
    v = (w0 * mu * (1 + anom)).sum(axis=1)                      # total value per ha each year
    e = w0 @ mu
    out = P.evaluate(w0, mu, np.cov(anom, rowvar=False), beta)
    assert out["exp_value_ha"] == pytest.approx(e, rel=1e-12)
    assert out["risk_sd"] == pytest.approx(np.std((v - e) / e, ddof=1), rel=1e-12)
    assert out["drought_change"] == pytest.approx(-1.5 * (w0 * mu * beta).sum() / e, rel=1e-12)
    assert out["drought_coverage"] == pytest.approx(1.0)
    assert P.identity_gap(w0, mu, anom, beta) < 1e-12


def test_drought_change_needs_beta_for_every_grown_crop():
    mu, cov = np.array([1.0, 1.0]), np.eye(2) * 0.01
    out = P.evaluate([0.5, 0.5], mu, cov, [0.1, np.nan])
    assert np.isnan(out["drought_change"]) and out["drought_coverage"] == pytest.approx(0.5)
    out = P.evaluate([1.0, 0.0], mu, cov, [0.1, np.nan])      # crop without beta has no area
    assert out["drought_change"] == pytest.approx(-0.15)


def test_optimiser_respects_bounds_and_sum_to_one():
    rng = np.random.default_rng(5)
    n = 8
    a = rng.normal(size=(31, n)) @ rng.normal(size=(n, n)) * 0.03
    cov = np.cov(a, rowvar=False)
    mu = rng.uniform(200, 3000, n)
    w0 = rng.dirichlet(np.ones(n))
    lo, hi = P.share_bounds(w0)
    assert (lo <= w0).all() and (w0 <= hi).all() and lo.sum() <= 1 <= hi.sum()
    fr = P.frontier(mu, cov, w0, lo, hi)
    for w in list(fr["weights"]) + [fr["w_opt"]]:
        assert w.sum() == pytest.approx(1, abs=1e-8)
        assert (w >= lo - 1e-8).all() and (w <= hi + 1e-8).all()
    r0 = P.evaluate(w0, mu, cov, np.zeros(n))
    r1 = P.evaluate(fr["w_opt"], mu, cov, np.zeros(n))
    assert r1["risk_sd"] <= r0["risk_sd"] + 1e-12
    assert r1["exp_value_ha"] >= r0["exp_value_ha"] * (1 - 1e-8)


def test_frontier_risk_non_decreasing_and_meets_targets():
    rng = np.random.default_rng(6)
    n = 6
    cov = np.cov(rng.normal(size=(31, n)) * rng.uniform(0.03, 0.2, n), rowvar=False)
    mu = rng.uniform(300, 2000, n)
    w0 = np.full(n, 1 / n)
    fr = P.frontier(mu, cov, w0, *P.share_bounds(w0))
    ev = [P.evaluate(w, mu, cov, np.zeros(n)) for w in fr["weights"]]
    risk = np.array([e["risk_sd"] for e in ev])
    assert (np.diff(risk) >= -1e-12).all()
    assert all(e["exp_value_ha"] >= t * (1 - 1e-8) for e, t in zip(ev, fr["targets"]))
    assert fr["targets"][-1] == pytest.approx(P.max_value_weights(mu, *P.share_bounds(w0)) @ mu)


def test_share_bounds_and_max_value_weights():
    lo, hi = P.share_bounds([0.6, 0.3, 0.1])
    assert lo == pytest.approx([0.5, 0.2, 0.05])       # 10 pp cap binds for the 0.6 and 0.3 crops
    assert hi == pytest.approx([0.7, 0.4, 0.15])
    w = P.max_value_weights(np.array([1.0, 3.0, 2.0]), lo, hi)
    assert w == pytest.approx([0.5, 0.4, 0.1])


def test_shrink_cov_keeps_variances_shrinks_correlations_and_is_psd():
    rng = np.random.default_rng(7)
    a = rng.normal(0, [0.05, 0.1, 0.2, 0.08, 0.15], size=(31, 5))
    a[rng.random(a.shape) < 0.15] = np.nan                    # masked (imputed) years
    sh = P.shrink_cov(a)
    c = sh["cov"]
    sd = np.nanstd(a, axis=0, ddof=1)
    assert np.diag(c) == pytest.approx(sd ** 2)
    assert 0 <= sh["lam"] <= 1 and np.linalg.eigvalsh(c).min() > 0
    corr = c / np.outer(sd, sd)
    i, j = 0, 1
    ok = np.isfinite(a[:, i]) & np.isfinite(a[:, j])
    r_ij = np.corrcoef(a[ok, i], a[ok, j])[0, 1]
    assert corr[i, j] == pytest.approx((1 - sh["lam"]) * r_ij + sh["lam"] * sh["rbar"])
    # Independent noise: most of the sample correlation is noise, so shrinkage is strong.
    assert sh["lam"] > 0.3


def test_shrink_cov_target_keeps_common_correlation():
    rng = np.random.default_rng(9)
    common = rng.normal(size=(31, 1))
    a = 0.05 * (0.8 * common + 0.6 * rng.normal(size=(31, 6)))      # true correlation 0.64
    sh = P.shrink_cov(a)
    corr = sh["cov"] / np.outer(a.std(axis=0, ddof=1), a.std(axis=0, ddof=1))
    off = corr[np.triu_indices(6, 1)]
    assert sh["rbar"] == pytest.approx(off.mean(), abs=1e-12)        # shrinking keeps the mean correlation
    assert 0.4 < sh["rbar"] < 0.85


def test_crop_table_drops_imputed_years_from_risk():
    rng = np.random.default_rng(8)
    rows = []
    for code, item, level, noise in [(1, "A", 1000, 0.1), (2, "B", 500, 0.05), (3, "C", 2000, 0.08)]:
        r = level * (1 + rng.normal(0, noise, len(YEARS)))
        imp = np.zeros(len(YEARS), bool)
        if code == 1:
            imp[-8:-5] = True
            r[-8:-5] = r[-9]                                  # carried-forward imputed values
        area = np.full(len(YEARS), 100.0 * code)
        rows.append(pd.DataFrame({"iso3": "AAA", "year": YEARS, "item_code": code, "item": item, "area_ha": area,
                                  "value_const": r * area, "is_imputed": imp}))
    g = pd.concat(rows, ignore_index=True)
    r = P.country_portfolio("AAA", g, pd.Series({"A": 0.1, "B": 0.05}))
    t = r["crops"].set_index("item")
    assert t["in_portfolio"].all() and t.loc["A", "n_years"] == len(YEARS) - 3
    assert r["w0"].sum() == pytest.approx(1) and r["w0"] == pytest.approx([1 / 6, 2 / 6, 3 / 6])
    assert np.isnan(r["beta"][2]) and not t.loc["C", "has_beta"]
    assert r["identity_gap"] < 1e-12


def test_min_risk_matches_brute_force_grid_for_three_crops():
    mu = np.array([1500.0, 700.0, 300.0])
    sd = np.array([0.10, 0.06, 0.15])
    corr = np.array([[1, 0.5, -0.2], [0.5, 1, 0.1], [-0.2, 0.1, 1]])
    cov = corr * np.outer(sd, sd)
    w0 = np.array([0.5, 0.3, 0.2])
    lo, hi = P.share_bounds(w0)
    g = np.linspace(0, 1, 801)
    w1, w2 = np.meshgrid(g, g)
    W = np.column_stack([w1.ravel(), w2.ravel(), 1 - w1.ravel() - w2.ravel()])
    W = W[((W >= lo) & (W <= hi)).all(axis=1) & (W @ mu >= w0 @ mu)]
    s = W * mu / (W @ mu)[:, None]
    brute = np.sqrt(np.einsum("ki,ij,kj->k", s, cov, s)).min()
    opt = P.frontier(mu, cov, w0, lo, hi)["w_opt"]
    assert P.evaluate(opt, mu, cov, np.zeros(3))["risk_sd"] <= brute + 1e-9
