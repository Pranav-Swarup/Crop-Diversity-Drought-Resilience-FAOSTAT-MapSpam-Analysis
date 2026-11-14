"""Unit tests for wet events (mirror of the drought tests). Run: python -m pytest src/metrics/tests/test_wet.py"""
import numpy as np
import pandas as pd

from src.climate.wet_events import wet_events

YEARS = np.arange(1993, 2024)


def events_for(values):
    spei = pd.Series(0.0, index=YEARS)
    for y, v in values.items():
        spei[y] = v
    return wet_events(pd.DataFrame({"iso3": "AAA", "year": YEARS, "spei12_w": spei.to_numpy()}))


def test_known_wet_run_gives_first_year_only():
    ev = events_for({2005: 1.2, 2006: 1.7})
    assert ev["year"].tolist() == [2005]
    assert ev["spei12_w"].tolist() == [1.2]
    assert ev["severity"].tolist() == ["moderate"]


def test_wet_year_within_three_years_of_previous_wet_is_not_event():
    ev = events_for({2002: 1.2, 2004: 1.1, 2012: 2.3})
    assert ev["year"].tolist() == [2002, 2012]
    assert ev["severity"].tolist() == ["moderate", "extreme"]


def test_events_too_close_to_window_edges_are_dropped():
    ev = events_for({1994: 2.0, 2022: 1.5, 2010: 1.6})
    assert ev["year"].tolist() == [2010]
    assert ev["severity"].tolist() == ["severe"]


def test_dry_years_are_not_wet_events():
    assert events_for({2010: -2.0, 2012: -1.2}).empty


def test_threshold_is_inclusive():
    assert events_for({2010: 1.0})["year"].tolist() == [2010]
