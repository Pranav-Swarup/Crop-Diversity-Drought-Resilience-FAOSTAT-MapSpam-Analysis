"""Crop diversity."""
import numpy as np


def hill1(mean_area):
    """Effective number of crops: exp(Shannon entropy) of harvested-area shares.

    mean_area: mean harvested area per crop over the window (any positive scale).
    """
    a = np.asarray(mean_area, dtype=float)
    a = a[np.isfinite(a) & (a > 0)]
    if a.size == 0:
        return np.nan
    p = a / a.sum()
    return float(np.exp(-(p * np.log(p)).sum()))
