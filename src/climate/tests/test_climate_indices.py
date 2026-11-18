"""Unit tests for Phase 2 & 3 climate pipelines (src/climate/heat.py, exposure_season.py, mapspam_area.py)."""
import numpy as np
import pandas as pd

from src.common import paths
from src.climate.exposure_season import MONTH_END


def test_month_end_calendar_logic():
    """Verify cumulative day threshold logic accurately maps maturity days to 1-12 months."""
    assert len(MONTH_END) == 12
    assert MONTH_END[-1] == 365
    # Day 15 is January (month 1)
    day = 15
    month = np.searchsorted(MONTH_END, day) + 1
    assert month == 1

    # Day 32 is February (month 2)
    day = 32
    month = np.searchsorted(MONTH_END, day) + 1
    assert month == 2

    # Day 365 is December (month 12)
    day = 365
    month = np.searchsorted(MONTH_END, day) + 1
    assert month == 12


def test_heat_dataset_properties():
    """Verify heat.parquet has normalized heat_z_w and valid station shares."""
    heat = pd.read_parquet(paths.HEAT)
    assert set(heat.columns) == {"iso3", "year", "tmx_warm_w", "tmx_warm_trend", "heat_z_w", "stn_share_w"}
    assert heat["stn_share_w"].between(0.0, 1.0).all()
    # Check that across all countries, heat_z_w is roughly mean 0 and std ~1
    assert abs(heat["heat_z_w"].mean()) < 0.2
    assert np.isclose(heat["heat_z_w"].std(), 1.0, atol=0.2)


def test_heat_events_threshold():
    """Verify all events in heat_events.csv satisfy heat_z_w >= 1.0 and severity rules."""
    he = pd.read_csv(paths.HEAT_EVENTS)
    assert (he["heat_z_w"] >= 1.0).all()
    assert set(he["severity"].unique()) <= {"moderate", "severe", "extreme"}


def test_mapspam_area_consistency():
    """Verify mapspam_area.parquet contains valid technology partitions and global area conservation."""
    ms = pd.read_parquet(paths.MAPSPAM_AREA)
    assert (ms["area_ha"] > 0).all()
    assert (ms["irr_ha"] >= 0).all()
    assert (ms["rf_ha"] >= 0).all()
    # Global technology gap: (irr + rf) / total area is within 2% as documented in mapspam_area.py
    global_tot = ms["area_ha"].sum()
    global_ir_rf = ms["irr_ha"].sum() + ms["rf_ha"].sum()
    gap = abs(global_ir_rf / global_tot - 1.0)
    assert gap < 0.02, f"Global MapSPAM technology gap exceeds 2%: {gap:.2%}"


def test_crop_climate_season_bounds():
    """Verify crop_climate_season.parquet SPEI values are finite and within valid empirical bounds."""
    ccs = pd.read_parquet(paths.CROP_CLIMATE_SEASON)
    assert ccs["crop_spei6_harv_w"].notna().all()
    assert np.isfinite(ccs["crop_spei6_harv_w"]).all()
    # SPEI is standardized; extreme tails across all country-crop-years fall within [-7, +7]
    assert ccs["crop_spei6_harv_w"].between(-7.0, 7.0).all()
