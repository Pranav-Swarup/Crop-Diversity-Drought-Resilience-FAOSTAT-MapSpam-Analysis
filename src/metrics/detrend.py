"""LOWESS detrending (CLAUDE.md: statsmodels, frac=0.5, per series over 1993-2023)."""
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

FRAC = 0.5
MIN_POINTS = 10


def lowess_trend(y, x=None, frac=FRAC):
    """LOWESS trend of y on x, evaluated at every x. NaNs in y are skipped in the fit
    and stay NaN in the output. Returns all-NaN if fewer than MIN_POINTS valid values."""
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y), dtype=float) if x is None else np.asarray(x, dtype=float)
    trend = np.full(len(y), np.nan)
    ok = np.isfinite(y)
    if ok.sum() < MIN_POINTS:
        return trend
    trend[ok] = lowess(y[ok], x[ok], frac=frac, return_sorted=False)
    return trend


def detrend(y, x=None, frac=FRAC):
    """Return a DataFrame with y, trend, resid = y - trend, anom = resid / trend, index = y / trend."""
    y = np.asarray(y, dtype=float)
    trend = lowess_trend(y, x, frac)
    trend = np.where(trend > 0, trend, np.nan)  # a non-positive trend makes the ratios meaningless
    return pd.DataFrame({
        "y": y, "trend": trend, "resid": y - trend,
        "anom": (y - trend) / trend, "index": y / trend,
    })
