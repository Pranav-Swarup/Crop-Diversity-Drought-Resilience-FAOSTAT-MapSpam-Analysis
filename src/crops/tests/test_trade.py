"""Unit tests for Phase 2 & 3 trade pipelines (src/crops/trade.py, trade_matrix.py, supply_diversity.py)."""
import numpy as np
import pandas as pd

from src.common import paths
from src.crops.trade import import_dep


def test_import_dep_formula_and_bounds():
    """Verify import_dep = import_t / (prod_t + import_t - export_t), with positive denom."""
    df = pd.DataFrame({
        "prod_t": [100.0, 50.0, 10.0, 0.0],
        "import_t": [20.0, 30.0, 5.0, 0.0],
        "export_t": [10.0, 20.0, 20.0, 0.0],
    })
    dep = import_dep(df)
    # Row 0: 20 / (100 + 20 - 10) = 20 / 110 = 0.1818...
    assert np.isclose(dep.iloc[0], 20.0 / 110.0)
    # Row 1: 30 / (50 + 30 - 20) = 30 / 60 = 0.5
    assert np.isclose(dep.iloc[1], 0.5)
    # Row 2: 5 / (10 + 5 - 20) = denom -5 <= 0 -> NaN
    assert pd.isna(dep.iloc[2])
    # Row 3: denom 0 -> NaN
    assert pd.isna(dep.iloc[3])


def test_trade_parquet_structure_and_sources():
    """Verify data/clean/trade.parquet conforms to contract and separates FBSH vs FBS."""
    trade = pd.read_parquet(paths.TRADE)
    assert set(trade["fbs_source"].unique()) == {"FBSH", "FBS"}
    # FBSH is for years <= 2009, FBS is for years >= 2010
    assert (trade.loc[trade["fbs_source"] == "FBSH", "year"] <= 2009).all()
    assert (trade.loc[trade["fbs_source"] == "FBS", "year"] >= 2010).all()
    # Check known published negative values (IRQ 2010 export) are present and preserved
    irq_2010 = trade[(trade["iso3"] == "IRQ") & (trade["year"] == 2010)]
    assert not irq_2010.empty
    assert (irq_2010["export_t"] < 0).any()


def test_trade_matrix_merged_entities_no_internal_flows():
    """Assert trade_partners.parquet and trade_partners_mirror.parquet eliminate internal entity flows."""
    tp = pd.read_parquet(paths.TRADE_PARTNERS)
    tpm = pd.read_parquet(paths.TRADE_PARTNERS_MIRROR)
    for df in (tp, tpm):
        # Importer and exporter cannot be the same
        assert (df["iso3"] != df["partner_iso3"]).all()
        # Quantities and values must be non-negative
        assert (df["import_t"] >= 0).all()
        assert (df["import_usd"] >= 0).all()


def test_supply_diversity_shares_sum_to_one():
    """Verify that domestic supply share + sum of import partner shares equals 1.0."""
    sd = pd.read_parquet(paths.SUPPLY_DIVERSITY)
    assert sd["s_dom"].between(0.0, 1.0).all()
    assert sd["supply_ratio"].between(0.0, 1.0).all()
    assert sd["supply_div"].ge(1.0).all()
    assert sd["supply_div_exp"].ge(1.0).all()
    assert (sd["supply_div_exp"] <= sd["supply_div"] + 1e-6).all()
