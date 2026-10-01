"""Drought years and drought events (CLAUDE.md "Drought event")."""
import pandas as pd

from ..common.contract import YEAR_MAX, YEAR_MIN

DROUGHT_THR = -1.0
PAD = 3  # years needed before and after an event


def severity(spei):
    return "extreme" if spei <= -2.0 else "severe" if spei <= -1.5 else "moderate"


def find_events(climate, thr=DROUGHT_THR, year_min=YEAR_MIN, year_max=YEAR_MAX):
    """Event year = first year of a run of drought years (spei12_w <= thr), with PAD years
    before and after inside the window and the PAD years before drought-free.

    climate: DataFrame with iso3, year, spei12_w. Returns iso3, year, spei12_w, severity.
    """
    rows = []
    for iso3, g in climate.groupby("iso3", sort=True):
        s = g.set_index("year")["spei12_w"].sort_index()
        drought = s <= thr
        for y in s.index[drought]:
            if y - PAD < year_min or y + PAD > year_max:
                continue
            before = drought.reindex(range(y - PAD, y))
            if before.isna().any() or before.any():
                continue
            rows.append((iso3, int(y), float(s.loc[y]), severity(s.loc[y])))
    return pd.DataFrame(rows, columns=["iso3", "year", "spei12_w", "severity"])
