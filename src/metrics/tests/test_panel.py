"""Tests of the two-way FE estimator in src.metrics.panel on simulated panels with known answers.
Run: python -m pytest src/metrics/tests/test_panel.py"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from src.metrics import panel

YEARS = np.arange(1993, 2024)


def simulate(n_countries=80, b1=0.05, b2=-0.02, rho_e=0.0, rho_x=0.0, sd_e=0.05, seed=0, drop=0.0):
    """Panel with country and year effects, anom = a_i + g_t + b1 spei + b2 spei z_i + e. AR(1) spei and e
    within country with coefficients rho_x, rho_e. Optionally drop a share of rows (unbalanced)."""
    rng = np.random.default_rng(seed)
    t = len(YEARS)
    z = rng.normal(size=n_countries)
    a = rng.normal(0, 0.1, n_countries)
    g = rng.normal(0, 0.05, t)

    def ar1(rho, sd):
        x = np.zeros((n_countries, t))
        x[:, 0] = rng.normal(0, sd, n_countries)
        for j in range(1, t):
            x[:, j] = rho * x[:, j - 1] + rng.normal(0, sd * np.sqrt(1 - rho ** 2), n_countries)
        return x

    spei = ar1(rho_x, 1.0) + 0.3 * g[None, :] / 0.05  # spei partly common across countries in a year
    e = ar1(rho_e, sd_e)
    anom = a[:, None] + g[None, :] + b1 * spei + b2 * spei * z[:, None] + e
    df = pd.DataFrame({
        "iso3": np.repeat([f"C{i:03d}" for i in range(n_countries)], t),
        "year": np.tile(YEARS, n_countries),
        "spei": spei.ravel(), "zh": np.repeat(z, t), "anom": anom.ravel(),
    })
    df["spei_x_zh"] = df["spei"] * df["zh"]
    if drop:
        df = df.sample(frac=1 - drop, random_state=seed).sort_values(["iso3", "year"]).reset_index(drop=True)
    return df


def test_known_interaction_recovered():
    df = simulate(n_countries=120, b1=0.05, b2=-0.02, seed=1)
    fit = panel.fe_fit(df, "anom", ["spei", "spei_x_zh"])
    se = np.sqrt(np.diag(fit["cov"]))
    assert fit["params"]["spei"] == pytest.approx(0.05, abs=4 * se[0])
    assert fit["params"]["spei_x_zh"] == pytest.approx(-0.02, abs=4 * se[1])
    assert abs(fit["params"]["spei_x_zh"] + 0.02) < 0.003
    assert 0 < fit["r2_within"] < 1


def test_fe_matches_explicit_dummies_unbalanced():
    df = simulate(n_countries=30, seed=2, drop=0.15)
    terms = ["spei", "spei_x_zh"]
    fit = panel.fe_fit(df, "anom", terms)
    dummies = pd.concat([pd.get_dummies(df["iso3"], drop_first=True, dtype=float),
                         pd.get_dummies(df["year"], drop_first=True, dtype=float)], axis=1)
    x = sm.add_constant(pd.concat([df[terms], dummies], axis=1))
    ref = sm.OLS(df["anom"], x).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df["iso3"])[0]})
    assert np.allclose(fit["params"][terms].to_numpy(), ref.params[terms].to_numpy(), rtol=1e-8, atol=1e-12)
    # Same sandwich; statsmodels counts the country dummies in K, panel.fe_fit does not (nested in clusters).
    n, k_sm = len(df), x.shape[1]
    k_fe = len(terms) + df["year"].nunique() - 1
    ratio = np.sqrt((n - k_sm) / (n - k_fe))
    se = np.sqrt(np.diag(fit["cov"]))
    assert np.allclose(se, ref.bse[terms].to_numpy() * ratio, rtol=1e-8)
    # Naive covariance equals the dummy regression's homoskedastic covariance.
    naive = sm.OLS(df["anom"], x).fit()
    assert np.allclose(np.sqrt(np.diag(fit["cov_naive"])), naive.bse[terms].to_numpy(), rtol=1e-8)


def test_clustered_se_exceed_naive_under_serial_correlation():
    df = simulate(n_countries=100, rho_e=0.8, rho_x=0.8, seed=3)
    fit = panel.fe_fit(df, "anom", ["spei", "spei_x_zh"])
    se_cl = np.sqrt(np.diag(fit["cov"]))
    se_nv = np.sqrt(np.diag(fit["cov_naive"]))
    assert (se_cl > 1.3 * se_nv).all()


def test_clustered_close_to_naive_without_serial_correlation():
    df = simulate(n_countries=100, rho_e=0.0, rho_x=0.0, seed=4)
    fit = panel.fe_fit(df, "anom", ["spei", "spei_x_zh"])
    ratio = np.sqrt(np.diag(fit["cov"])) / np.sqrt(np.diag(fit["cov_naive"]))
    assert ((ratio > 0.75) & (ratio < 1.3)).all()


def test_marginal_effect_and_delta_se():
    df = simulate(n_countries=60, seed=5)
    fit = panel.fe_fit(df, "anom", ["spei", "spei_x_zh"])
    z = np.array([-1.0, 0.0, 2.0])
    me, se = panel.marginal_effect(fit, "spei", "spei_x_zh", z)
    b, v = fit["params"], fit["cov"].to_numpy()
    assert np.allclose(me, b["spei"] + b["spei_x_zh"] * z)
    g = np.column_stack([np.ones_like(z), z])
    assert np.allclose(se, np.sqrt(np.einsum("ij,jk,ik->i", g, v, g)))
    assert se[1] == pytest.approx(np.sqrt(v[0, 0]))


def test_design_zscores_one_value_per_country():
    df = simulate(n_countries=20, seed=6, drop=0.3)  # unequal rows per country
    df["h"] = df["zh"] * 3 + 10
    d, terms, scale = panel.design(df, ["spei"], ["h"])
    per = df.groupby("iso3")["h"].first()
    assert scale["h"] == pytest.approx((per.mean(), per.std(ddof=1)))
    assert terms == ["spei", "spei_x_h"]
    assert np.allclose(d["spei_x_h"], df["spei"] * (df["h"] - per.mean()) / per.std(ddof=1))


def test_annual_hill1_lag_and_imputed_share():
    crops = pd.DataFrame({
        "iso3": ["AAA"] * 6, "year": [2000, 2000, 2001, 2001, 2003, 2003], "item_code": [1, 2] * 3,
        "area_ha": [10.0, 10.0, 30.0, 10.0, 5.0, 5.0], "value_const": [1.0, 3.0, 2.0, 2.0, 1.0, 1.0],
        "is_imputed": [True, False, True, True, False, False],
    })
    h = panel.annual_hill1(crops, ["AAA"]).set_index("year")
    assert h.loc[2000, "hill1_t"] == pytest.approx(2.0)
    assert h.loc[2001, "hill1_lag"] == pytest.approx(2.0)
    assert np.isnan(h.loc[2000, "hill1_lag"]) and np.isnan(h.loc[2003, "hill1_lag"])  # 2002 missing
    s = panel.imputed_value_share(crops, ["AAA"]).set_index("year")["imputed_share"]
    assert s.loc[2000] == pytest.approx(0.25) and s.loc[2001] == pytest.approx(1.0) and s.loc[2003] == 0.0
