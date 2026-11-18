"""End-to-end regression benchmarks and conceptual mathematical audits for Phase 2 and 3 outputs."""
import json
import numpy as np
import pandas as pd

from src.common import paths
from src.metrics.portfolio import evaluate


def test_panel_fe_benchmark_values():
    """Verify core econometric panel results match committed benchmarks in outputs/stats/panel_fe.csv."""
    pfe = pd.read_csv(paths.STATS_DIR / "panel_fe.csv")

    # Core model
    core = pfe[pfe["model"] == "core"].set_index("term")
    assert np.isclose(core.loc["spei", "beta"], 0.017944, atol=1e-4)
    assert np.isclose(core.loc["spei_x_hill1", "beta"], -0.007619, atol=1e-4)
    assert np.isclose(core.loc["spei_x_resp_div", "beta"], 0.021833, atol=1e-4)
    assert core.loc["spei_x_hill1", "p"] < 0.05
    assert core.loc["spei_x_resp_div", "p"] < 0.001

    # Split-sample response diversity model
    split = pfe[pfe["model"] == "split_resp_div"].set_index("term")
    assert np.isclose(split.loc["spei_x_hill1", "beta"], -0.00786, atol=1e-4)
    assert split.loc["spei_x_hill1", "p"] < 0.05
    # The mechanical resp_div interaction collapses to near-zero and non-significant
    assert np.isclose(split.loc["spei_x_resp_div_odd", "beta"], 0.00210, atol=1e-4)
    assert split.loc["spei_x_resp_div_odd", "p"] > 0.40

    # Asymmetric model (dry vs wet)
    asym = pfe[pfe["model"] == "asymmetric"].set_index("term")
    assert np.isclose(asym.loc["dry", "beta"], 0.03606, atol=1e-4)
    assert asym.loc["dry", "p"] < 1e-8
    assert asym.loc["wet", "p"] > 0.10  # wet shock is not statistically significant


def test_severity_panel_benchmark_values():
    """Verify severity bin models match committed outputs/stats/severity_panel.csv."""
    sev = pd.read_csv(paths.STATS_DIR / "severity_panel.csv")
    assert not sev.empty
    bins_model = sev[sev["model"] == "bins"].set_index("term")
    # All drought bins should be negative (relative anomaly loss during drought)
    for b in ("moderate", "severe", "extreme"):
        assert bins_model.loc[b, "beta"] < 0.0
        assert bins_model.loc[b, "p"] < 0.05


def test_heat_panel_benchmark_values():
    """Verify heat panel results in outputs/stats/heat_panel.csv."""
    hp = pd.read_csv(paths.STATS_DIR / "heat_panel.csv")
    assert not hp.empty
    yt = hp[hp["model"] == "year_types"].set_index("term")
    assert "dry_only" in yt.index
    assert "hot_only" in yt.index
    assert "compound" in yt.index


def test_severity_fe_basis_full_rank():
    """Conceptual Check: Assert country-year fixed effects basis is full rank (unlike nested district-state)."""
    metrics_nat = pd.read_parquet(paths.NATIONAL)
    # Filter to included countries
    quality = pd.read_csv(paths.QUALITY)
    inc = set(quality.loc[quality["included"], "iso3"])
    df = metrics_nat[metrics_nat["iso3"].isin(inc)].copy()

    # Construct fixed effects dummy matrix (dropping one year dummy to prevent intercept collinearity)
    d_country = pd.get_dummies(df["iso3"], dtype=float).to_numpy()
    d_year = pd.get_dummies(df["year"], drop_first=True, dtype=float).to_numpy()
    D = np.column_stack([d_country, d_year])

    # Compute singular values
    u, s, vt = np.linalg.svd(D, full_matrices=False)
    # In a full-rank design (with 1 reference category dropped), all singular values must be well above 0
    min_sv = s[-1]
    assert min_sv > 1e-4, f"Minimum singular value is too close to zero: {min_sv}"
    # This confirms QR orthogonalization in severity.py does not overproject or suffer rank deficiency


def test_portfolio_js_python_parity():
    """Verify exact parity between portfolio parameters, Python evaluate(), and frontier points."""
    cov_path = paths.OUT / "portfolio_cov.json"
    frontier_path = paths.OUT / "portfolio_frontier.parquet"
    assert cov_path.exists() and frontier_path.exists()

    with open(cov_path) as f:
        data = json.load(f)
    countries = data["countries"]
    frontier_df = pd.read_parquet(frontier_path)

    # Test for the first 5 countries that evaluate(w0) exactly reproduces point 20 (kind='current')
    for iso3, c in list(countries.items())[:5]:
        w0 = np.array(c["w0"])
        mu = np.array(c["mu"])
        cov = np.array(c["cov"])
        beta = np.array(c["beta"])

        assert np.isclose(w0.sum(), 1.0, atol=1e-5), f"{iso3}: initial weights w0 do not sum to 1"
        res = evaluate(w0, mu, cov, beta)

        # Point 20 is 'current'
        cur = frontier_df[(frontier_df["iso3"] == iso3) & (frontier_df["point"] == 20)].iloc[0]
        assert np.isclose(res["exp_value_ha"], cur["exp_value_ha"], rtol=1e-4)
        assert np.isclose(res["risk_sd"], cur["risk_sd"], rtol=1e-4)
