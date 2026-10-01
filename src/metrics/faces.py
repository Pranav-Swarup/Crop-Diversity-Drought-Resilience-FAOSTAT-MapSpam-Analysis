"""The four faces of resilience (Urruty et al. 2016), on the national value index I_t = y_t / trend_t.

All functions take pandas Series indexed by year.
"""
import numpy as np

from .events import PAD

LOSS_THR = 0.9  # I_t below this = loss of more than 10% against trend


def cv_resid(y, resid):
    """CV of detrended residuals: sd(y - trend) / mean(y)."""
    return float(np.nanstd(resid, ddof=1) / np.nanmean(y))


def stability(y, resid):
    """1 / CV of detrended residuals."""
    return 1.0 / cv_resid(y, resid)


def _event_ratios(index, event_years):
    """Per event: (I_t / mean(I_{t-3..t-1}), mean(I_{t+1..t+3}) / I_t). Events without a
    complete 7-year window in `index` are skipped."""
    out = []
    for t in event_years:
        win = index.reindex(range(t - PAD, t + PAD + 1))
        if win.isna().any():
            continue
        before, at, after = win.iloc[:PAD].mean(), win.iloc[PAD], win.iloc[PAD + 1:].mean()
        out.append((at / before, after / at))
    return out


def resistance(index, event_years):
    """Median over events of I_t / mean(I_{t-3..t-1}) (Lloret et al. 2011). NaN with no usable event."""
    r = _event_ratios(index, event_years)
    return float(np.median([a for a, _ in r])) if r else np.nan


def recovery(index, event_years):
    """Median over events of mean(I_{t+1..t+3}) / I_t. NaN with no usable event."""
    r = _event_ratios(index, event_years)
    return float(np.median([b for _, b in r])) if r else np.nan


def n_usable_events(index, event_years):
    return len(_event_ratios(index, event_years))


def vulnerability(index, years=None):
    """Share of years with I_t < 0.9. With `years`, restricted to those years (e.g. drought
    years for vuln_drought); NaN if none of them are in the series."""
    s = index.dropna()
    if years is not None:
        s = s[s.index.isin(list(years))]
    return float((s < LOSS_THR).mean()) if len(s) else np.nan
