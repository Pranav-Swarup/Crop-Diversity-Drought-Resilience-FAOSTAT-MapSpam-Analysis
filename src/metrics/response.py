"""Response diversity: how differently a country's crops react to drought."""
import numpy as np

MIN_YEARS = 20       # non-imputed years needed for a crop's slope
MIN_SHARE = 0.01     # minimum mean value share


def resp_beta(yield_anom, spei):
    """OLS slope of yield anomaly on spei12_w. NaN pairs are dropped; NaN if < MIN_YEARS pairs."""
    a, s = np.asarray(yield_anom, dtype=float), np.asarray(spei, dtype=float)
    ok = np.isfinite(a) & np.isfinite(s)
    if ok.sum() < MIN_YEARS or np.ptp(s[ok]) == 0:
        return np.nan
    return float(np.polyfit(s[ok], a[ok], 1)[0])


def resp_div(betas, weights):
    """Value-weighted SD of resp_beta across crops: sqrt(sum w_i (b_i - bbar)^2), with w
    renormalised to sum to 1 and bbar the weighted mean. NaN with fewer than 2 crops."""
    b, w = np.asarray(betas, dtype=float), np.asarray(weights, dtype=float)
    ok = np.isfinite(b) & np.isfinite(w) & (w > 0)
    if ok.sum() < 2:
        return np.nan
    b, w = b[ok], w[ok] / w[ok].sum()
    return float(np.sqrt((w * (b - (w * b).sum()) ** 2).sum()))


def resp_divergence(betas):
    """Divergence of Ross, Petchey, Sasaki & Armitage (2023), "How to measure response
    diversity", Methods Ecol. Evol. 14:1150-1167, section 3.1:

        Divergence = ( max f'(E) - min f'(E) - | |max f'(E)| - |min f'(E)| | ) / ( max f'(E) - min f'(E) )

    where f'(E) is the first derivative of each performance-environment relationship. Our
    responses are linear, so f'(E) is the slope resp_beta and does not vary with E.
    0 when all slopes have the same sign, 1 when min = -max. Unweighted (as in the paper).
    NaN with fewer than 2 crops; 0 when all slopes are equal.
    """
    b = np.asarray(betas, dtype=float)
    b = b[np.isfinite(b)]
    if b.size < 2:
        return np.nan
    hi, lo = b.max(), b.min()
    if hi == lo:
        return 0.0
    return float((hi - lo - abs(abs(hi) - abs(lo))) / (hi - lo))
