"""Similarity-sensitive diversity (Leinster & Cobbold 2012, Ecology 93: 477-489).

Single implementation for every "effective diversity" in the project (crop exposure classes, supply sources).

For shares p (sum 1) and a similarity matrix Z (Z_ii = 1, 0 <= Z_ij <= 1), the diversity of order q is

    D_q^Z(p) = ( sum_i p_i (Zp)_i^(q-1) )^(1/(1-q)),      q != 1
    D_1^Z(p) = exp( - sum_i p_i log (Zp)_i )

(Zp)_i is the "ordinariness" of type i: the share of the portfolio that is similar to it.
  Z = identity       -> the Hill number of order q (q = 1: exp(Shannon) = hill1; q = 2: inverse Simpson)
  Z = all ones       -> 1 (every type is the same bet)
  q = 2              -> 1 / (p' Z p)
Splitting a type into two parts that are perfectly similar to each other leaves D unchanged.

Here similarity = shared drought exposure: Z_ij = max(0, r_ij), r the Pearson correlation of two exposure
series. Negative correlations are set to 0 (the framework needs 0 <= Z <= 1), so a pair of types whose
exposure is anti-correlated counts as fully distinct, not as more than distinct.
"""
import numpy as np


def similarity_from_corr(r):
    """Correlation matrix -> similarity matrix: negatives clipped to 0, diagonal set to 1.
    NaN off-diagonal entries are an error (decide upstream what an undefined correlation means)."""
    z = np.array(r, dtype=float)
    assert z.ndim == 2 and z.shape[0] == z.shape[1], "correlation matrix must be square"
    np.fill_diagonal(z, 1.0)
    assert np.isfinite(z).all(), "correlation matrix has undefined entries"
    assert np.allclose(z, z.T, atol=1e-9), "correlation matrix is not symmetric"
    return np.clip(z, 0.0, 1.0)


def lc_diversity(p, z=None, q=1.0):
    """Leinster-Cobbold diversity of order q. p: non-negative weights (normalised here; zero-weight types
    are dropped). z: similarity matrix aligned with p, or None for the identity. Returns NaN if p is empty."""
    p = np.asarray(p, dtype=float)
    assert p.ndim == 1 and np.isfinite(p).all() and (p >= 0).all(), "shares must be finite and non-negative"
    keep = p > 0
    if not keep.any():
        return np.nan
    p = p[keep] / p[keep].sum()
    if z is None:
        zp = p
    else:
        z = np.asarray(z, dtype=float)
        assert z.shape == (keep.size, keep.size), "similarity matrix does not match the shares"
        z = z[np.ix_(keep, keep)]
        assert np.allclose(np.diag(z), 1.0), "similarity matrix needs a unit diagonal"
        assert (z >= -1e-12).all() and (z <= 1 + 1e-12).all(), "similarities must lie in [0, 1]"
        zp = z @ p
    if np.isclose(q, 1.0):
        return float(np.exp(-(p * np.log(zp)).sum()))
    return float((p * zp ** (q - 1.0)).sum() ** (1.0 / (1.0 - q)))


def mean_offdiag(p, r):
    """Share-weighted mean of the off-diagonal entries of r: sum_{i != j} p_i p_j r_ij / sum_{i != j} p_i p_j.
    NaN with fewer than two types. Used for the mean exposure correlation (r unclipped)."""
    p = np.asarray(p, dtype=float)
    r = np.asarray(r, dtype=float)
    keep = p > 0
    p, r = p[keep] / p[keep].sum(), r[np.ix_(keep, keep)]
    if p.size < 2:
        return np.nan
    w = np.outer(p, p)
    np.fill_diagonal(w, 0.0)
    return float((w * r).sum() / w.sum())
