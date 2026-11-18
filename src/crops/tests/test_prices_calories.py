"""Unit tests for Phase 2 producer prices, calorie conversions, and buffer mechanisms (src/crops/)."""
import numpy as np
import pandas as pd

from src.common import paths
from src.crops.prices import merge_members
from src.crops.buffers import ols_slope


def test_merge_members_single_area():
    """Verify merge_members returns exact single area price."""
    df = pd.DataFrame({
        "Value": [250.0],
        "is_imputed": [False],
        "prod_t": [1000.0]
    })
    res = merge_members(df)
    assert res["price_usd_t"] == 250.0
    assert not res["is_imputed"]
    assert not res["left_out"]


def test_merge_members_multiple_areas_production_weighted():
    """Verify merge_members computes production-weighted average price across member areas."""
    # Member 1: price 100, prod 1000. Member 2: price 200, prod 3000.
    # Weighted price: (100*1000 + 200*3000) / 4000 = 700000 / 4000 = 175.
    df = pd.DataFrame({
        "Value": [100.0, 200.0],
        "is_imputed": [False, True],
        "prod_t": [1000.0, 3000.0]
    })
    res = merge_members(df)
    assert np.isclose(res["price_usd_t"], 175.0)
    assert res["is_imputed"]  # True because any member is imputed
    assert not res["left_out"]


def test_merge_members_missing_production_is_left_out():
    """Verify merge_members leaves out item-year if any reporting member has non-positive or NaN production."""
    df = pd.DataFrame({
        "Value": [100.0, 200.0],
        "is_imputed": [False, False],
        "prod_t": [1000.0, 0.0]
    })
    res = merge_members(df)
    assert pd.isna(res["price_usd_t"])
    assert res["left_out"]


def test_ols_slope_helper():
    """Verify ols_slope computes correct slope, p-value, and handles edge cases."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = 2.0 * x + 1.0
    slope, p, n = ols_slope(y, x)
    assert np.isclose(slope, 2.0)
    assert np.isclose(p, 0.0, atol=1e-5)
    assert n == 5

    # Constant x gives NaN
    slope, p, n = ols_slope([1, 2, 3], [5, 5, 5])
    assert pd.isna(slope)

    # Fewer than 3 points gives NaN
    slope, p, n = ols_slope([1, 2], [1, 2])
    assert pd.isna(slope)


def test_calorie_factors_file_integrity():
    """Verify calorie factors table has valid ranges and in-scope flags."""
    cf = pd.read_csv(paths.CLEAN / "calorie_factors.csv")
    assert "kcal_per_t" in cf.columns
    assert (cf.loc[cf["in_scope"], "kcal_per_t"] > 0).all()
    # primary cereals (wheat, rice, maize) should have ~3-4 million kcal per tonne
    wheat = cf[cf["item"].str.lower().str.contains("wheat") & cf["in_scope"]]
    assert not wheat.empty
    assert (wheat["kcal_per_t"] > 2.5e6).all() and (wheat["kcal_per_t"] < 4.0e6).all()
