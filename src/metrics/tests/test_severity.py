"""Unit tests for src.metrics.severity (synthetic data). Run: python -m pytest src/metrics/tests/test_severity.py"""
import numpy as np
import pandas as pd

from src.metrics import panel
from src.metrics.severity import (DROUGHT_BINS, HEAT_BINS, comove_bruteforce, crop_breadth, derived_rows,
                                  drought_bins, fit_outcome, heat_bins, permute_country, wild_cluster_bootstrap)


def test_drought_bins_exclusive_and_inclusive_thresholds():
    s = pd.Series([0.0, -0.99, -1.0, -1.49, -1.5, -1.99, -2.0, -3.0, np.nan])
    b = drought_bins(s)
    assert b["moderate"].tolist()[:8] == [0, 0, 1, 1, 0, 0, 0, 0]
    assert b["severe"].tolist()[:8] == [0, 0, 0, 0, 1, 1, 0, 0]
    assert b["extreme"].tolist()[:8] == [0, 0, 0, 0, 0, 0, 1, 1]
    assert (b["sev_plus"] == b["severe"] + b["extreme"]).iloc[:8].all()
    assert (b[DROUGHT_BINS].sum(axis=1).iloc[:8] == (s.iloc[:8] <= -1.0)).all()
    assert b.iloc[8].isna().all()


def test_heat_bins_exclusive_and_inclusive_thresholds():
    h = pd.Series([0.99, 1.0, 1.49, 1.5, 1.99, 2.0, 4.0, np.nan])
    b = heat_bins(h)
    assert b["hot_mod"].tolist()[:7] == [0, 1, 1, 0, 0, 0, 0]
    assert b["hot_sev"].tolist()[:7] == [0, 0, 0, 1, 1, 0, 0]
    assert b["hot_ext"].tolist()[:7] == [0, 0, 0, 0, 0, 1, 1]
    assert (b[HEAT_BINS].sum(axis=1).iloc[:7] <= 1).all()
    assert b.iloc[7].isna().all()


def test_comove_matches_bruteforce_pairs():
    rng = np.random.default_rng(1)
    rows = []
    for item, share in zip("abcdefg", [0.3, 0.2, 0.15, 0.1, 0.1, 0.08, 0.07]):
        for year in range(1993, 2024):
            rows.append({"iso3": "AAA", "year": year, "item": item, "value_share": share,
                         "yield_anom": rng.normal(0, 0.1 + share)})
    ca = pd.DataFrame(rows)
    ca = ca.drop(ca[(ca["item"] == "g") & (ca["year"] == 2000)].index)    # 6 crops in 2000
    br = crop_breadth(ca).set_index("year")
    sd = ca.groupby("item")["yield_anom"].std(ddof=1)
    for year in (1995, 2000, 2020):
        g = ca[ca["year"] == year]
        z = g["yield_anom"].to_numpy() / sd[g["item"]].to_numpy()
        w = g["value_share"].to_numpy()
        assert np.isclose(br.loc[year, "comove"], comove_bruteforce(w, z))
        assert np.isclose(br.loc[year, "breadth"], (w / w.sum() * (g["yield_anom"] < 0)).sum())
        assert br.loc[year, "n_crops"] == len(g)


def test_breadth_drops_country_years_with_few_crops():
    ca = pd.DataFrame({"iso3": "AAA", "year": np.repeat([2000, 2001, 2002], 4), "item": list("abcd") * 3,
                       "value_share": 0.25, "yield_anom": np.linspace(-0.2, 0.2, 12)})
    assert crop_breadth(ca, min_crops=5).empty
    assert len(crop_breadth(ca, min_crops=4)) == 3


def synthetic_panel(seed=0, n_c=30, years=range(1993, 2024)):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_c):
        h = rng.uniform(2, 20)
        for y in years:
            s = rng.normal(-0.2, 1.0)
            rows.append({"iso3": f"C{i:02d}", "year": y, "spei": s, "hill1": h,
                         "anom": 0.03 * s + rng.normal(0, 0.05)})
    df = pd.DataFrame(rows)
    return pd.concat([df, drought_bins(df["spei"])], axis=1)


def test_permutation_keeps_country_series_and_is_reproducible():
    df = synthetic_panel()
    a = permute_country(df, "hill1", np.random.default_rng(7))
    b = permute_country(df, "hill1", np.random.default_rng(7))
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(a.drop(columns="hill1"), df.drop(columns="hill1"))
    assert (a.groupby("iso3")["hill1"].nunique() == 1).all()
    assert sorted(a.groupby("iso3")["hill1"].first()) == sorted(df.groupby("iso3")["hill1"].first())
    assert not np.allclose(a["hill1"], df["hill1"])
    c = permute_country(df, "hill1", np.random.default_rng(8))
    assert not np.allclose(a["hill1"], c["hill1"])


def test_derived_rows_arithmetic():
    terms = ["moderate", "severe", "extreme", "moderate_x_hill1", "severe_x_hill1", "extreme_x_hill1"]
    beta = np.array([-0.02, -0.05, -0.10, 0.01, 0.02, 0.05])
    cov = np.diag([1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4])
    cov[2, 5] = cov[5, 2] = 1e-4
    fit = {"params": pd.Series(beta, index=terms), "cov": pd.DataFrame(cov, index=terms, columns=terms), "df_t": 50}
    d = pd.DataFrame({"iso3": [f"C{i}" for i in range(11)], "hill1": np.arange(11.0)})   # p10 = 1, p90 = 9
    mu, sd = 5.0, 2.0
    rows = {r["term"]: r for r in derived_rows(fit, d, {"hill1": (mu, sd)}, DROUGHT_BINS)}
    assert np.isclose(rows["extreme_minus_moderate_x_hill1"]["beta"], 0.04)
    assert np.isclose(rows["extreme_minus_moderate_x_hill1"]["se"], np.sqrt(6e-4 + 4e-4))
    assert np.isclose(rows["severe_minus_moderate_x_hill1"]["beta"], 0.01)
    z10, z90 = (1 - mu) / sd, (9 - mu) / sd
    assert np.isclose(rows["extreme_at_p10"]["beta"], -0.10 + 0.05 * z10)
    assert np.isclose(rows["extreme_at_p90"]["beta"], -0.10 + 0.05 * z90)
    assert np.isclose(rows["extreme_at_p90"]["se"], np.sqrt(3e-4 + z90 ** 2 * 6e-4 + 2 * z90 * 1e-4))
    assert np.isclose(rows["moderate_at_p10"]["beta"], -0.02 + 0.01 * z10)


def test_fit_outcome_matches_panel_fit_model():
    df = synthetic_panel()
    a = fit_outcome(df, "anom", DROUGHT_BINS, ["hill1"])[0]
    b = panel.fit_model(df, DROUGHT_BINS, ["hill1"], (), ())[0]
    assert np.allclose(a["params"], b["params"]) and np.allclose(a["cov"], b["cov"])


def test_wild_bootstrap_reproduces_fe_fit_and_seed():
    df = synthetic_panel(n_c=20)
    _, d, _ = fit_outcome(df, "anom", DROUGHT_BINS, ["hill1"])
    terms = DROUGHT_BINS + [f"{b}_x_hill1" for b in DROUGHT_BINS]
    r1 = wild_cluster_bootstrap(d, "anom", terms, "extreme_x_hill1", n_boot=199, seed=3)   # asserts se = fe_fit
    r2 = wild_cluster_bootstrap(d, "anom", terms, "extreme_x_hill1", n_boot=199, seed=3)
    assert r1 == r2
    assert r1["ci_lo"] < r1["beta"] < r1["ci_hi"] and 0 <= r1["p_wcr"] <= 1
