"""Unit tests for the heat extension. Run: python -m pytest src/metrics/tests/test_heat.py"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("exactextract")  # src.climate.heat imports the gridded stack

from src.climate.heat import heat_events, heat_z, warm_months, warm_season_mean  # noqa: E402
from src.metrics.heat import compute, linear_combo, year_flags  # noqa: E402

YEARS = np.arange(1993, 2024)


def events_for(values):
    z = pd.Series(0.0, index=YEARS)
    for y, v in values.items():
        z[y] = v
    return heat_events(pd.DataFrame({"iso3": "AAA", "year": YEARS, "heat_z_w": z.to_numpy()}))


def test_hot_run_gives_first_year_only():
    ev = events_for({2005: 1.2, 2006: 1.7})
    assert ev["year"].tolist() == [2005]
    assert ev["heat_z_w"].tolist() == [1.2]
    assert ev["severity"].tolist() == ["moderate"]


def test_hot_year_within_three_years_of_previous_is_not_event():
    ev = events_for({2002: 1.2, 2004: 1.1, 2012: 2.3})
    assert ev["year"].tolist() == [2002, 2012]
    assert ev["severity"].tolist() == ["moderate", "extreme"]


def test_edges_cool_years_and_inclusive_threshold():
    assert events_for({1994: 2.0, 2022: 1.5}).empty
    assert events_for({2010: -2.0}).empty
    assert events_for({2010: 1.0})["year"].tolist() == [2010]


def test_warm_months_picks_three_warmest_and_skips_empty_cells():
    clim = np.array([5, 6, 9, 14, 19, 24, 27, 26, 21, 15, 9, 6], dtype=float)       # northern cell
    south = np.roll(clim, 6)                                                        # warmest Dec, Jan, Feb
    monthly = np.stack([np.stack([clim, south, np.full(12, np.nan)], axis=1)] * 4)  # (years, 12, cells)
    mask = warm_months(monthly)
    assert np.flatnonzero(mask[:, 0]).tolist() == [5, 6, 7]
    assert np.flatnonzero(mask[:, 1]).tolist() == [0, 1, 11]
    assert not mask[:, 2].any()
    warm = warm_season_mean(monthly, mask)
    assert np.allclose(warm[:, 0], (24 + 27 + 26) / 3)
    assert np.isnan(warm[:, 2]).all()


def test_heat_z_removes_trend_and_has_unit_sd():
    rng = np.random.default_rng(0)
    tmx = 30 + 0.05 * (YEARS - 1993) + rng.normal(0, 0.5, len(YEARS))
    tmx[10] += 3.0
    trend, z = heat_z(YEARS, tmx)
    assert np.isclose(np.std(z, ddof=1), 1.0)
    assert abs(np.polyfit(YEARS, z, 1)[0]) < 0.02
    assert z.argmax() == 10 and z[10] > 2


def test_year_flags_are_exclusive_and_nan_when_input_missing():
    df = year_flags(pd.DataFrame({"spei": [-1.5, -1.5, 0.0, 0.0, np.nan], "heat": [1.2, 0.0, 1.0, 0.0, 2.0]}))
    assert df["compound"].tolist()[:4] == [1, 0, 0, 0]
    assert df["dry_only"].tolist()[:4] == [0, 1, 0, 0]
    assert df["hot_only"].tolist()[:4] == [0, 0, 1, 0]
    assert df.loc[4, ["dry_only", "hot_only", "compound"]].isna().all()


def test_compute_counts_and_vulnerability():
    index = pd.Series(1.0, index=YEARS)
    index[2005], index[2012] = 0.8, 0.85
    national = pd.DataFrame({"iso3": "AAA", "year": YEARS, "index": index.to_numpy()})
    spei = pd.Series(0.0, index=YEARS)
    heat = pd.Series(0.0, index=YEARS)
    heat[2005], heat[2012], heat[2018] = 1.5, 1.1, 1.3
    spei[2005] = -1.4
    climate = pd.DataFrame({"iso3": "AAA", "year": YEARS, "spei12_w": spei.to_numpy()})
    heat_df = pd.DataFrame({"iso3": "AAA", "year": YEARS, "heat_z_w": heat.to_numpy()})
    out = compute(national, climate, heat_df, heat_events(heat_df)).iloc[0]
    assert out["n_hot_years"] == 3 and out["n_compound_years"] == 1 and out["n_heat_events"] == 3
    assert np.isclose(out["vuln_heat"], 2 / 3) and out["vuln_compound"] == 1.0
    assert np.isclose(out["resistance_heat"], 0.85)  # median of 0.8, 0.85, 1.0


def test_linear_combo_matches_hand_calculation():
    terms = ["dry_only", "hot_only", "compound"]
    fit = {"params": pd.Series([-0.04, -0.01, -0.06], index=terms),
           "cov": pd.DataFrame(np.diag([1e-4, 4e-4, 9e-4]), index=terms, columns=terms), "df_t": 100}
    r = linear_combo(fit, {"compound": 1.0, "dry_only": -1.0, "hot_only": -1.0})
    assert np.isclose(r["beta"], -0.01) and np.isclose(r["se"], np.sqrt(14e-4))
    assert r["ci_lo"] < r["beta"] < r["ci_hi"] and 0 < r["p"] < 1
