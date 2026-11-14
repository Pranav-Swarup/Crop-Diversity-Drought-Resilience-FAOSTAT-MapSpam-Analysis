"""Unit tests for similarity-sensitive diversity. Run: python -m pytest src/metrics/tests/test_similarity.py"""
import numpy as np
import pytest

from src.metrics.diversity import hill1
from src.metrics.similarity import lc_diversity, mean_offdiag, similarity_from_corr

P = np.array([0.5, 0.3, 0.15, 0.05])


def test_identity_similarity_gives_hill_numbers():
    assert np.isclose(lc_diversity(P), hill1(P))
    assert np.isclose(lc_diversity(P, np.eye(4)), hill1(P))
    assert np.isclose(lc_diversity(P, np.eye(4), q=2), 1 / (P ** 2).sum())
    assert np.isclose(lc_diversity(np.full(7, 3.0)), 7.0)          # unnormalised equal weights


def test_all_similar_is_one_bet():
    for q in (1, 2, 0.5):
        assert np.isclose(lc_diversity(P, np.ones((4, 4)), q=q), 1.0)


def test_order_two_is_inverse_quadratic_form():
    z = similarity_from_corr([[1, .8, .2, 0], [.8, 1, .1, 0], [.2, .1, 1, .5], [0, 0, .5, 1]])
    assert np.isclose(lc_diversity(P, z, q=2), 1 / (P @ z @ P))


def test_splitting_a_type_into_identical_parts_changes_nothing():
    z3 = np.array([[1, .3, .1], [.3, 1, .6], [.1, .6, 1]])
    p3 = np.array([0.5, 0.3, 0.2])
    # split type 0 into two perfectly similar halves
    z4 = np.array([[1, 1, .3, .1], [1, 1, .3, .1], [.3, .3, 1, .6], [.1, .1, .6, 1]])
    p4 = np.array([0.3, 0.2, 0.3, 0.2])
    for q in (1, 2):
        assert np.isclose(lc_diversity(p3, z3, q=q), lc_diversity(p4, z4, q=q))


def test_more_similarity_means_less_diversity_and_bounds():
    lo = np.full((4, 4), 0.2); np.fill_diagonal(lo, 1)
    hi = np.full((4, 4), 0.7); np.fill_diagonal(hi, 1)
    d_lo, d_hi = lc_diversity(P, lo), lc_diversity(P, hi)
    assert 1 < d_hi < d_lo < hill1(P)


def test_zero_weights_are_dropped_and_empty_is_nan():
    z = np.array([[1, .5, .9], [.5, 1, .9], [.9, .9, 1]])
    assert np.isclose(lc_diversity([0.6, 0.4, 0.0], z), lc_diversity([0.6, 0.4], z[:2, :2]))
    assert np.isnan(lc_diversity([0.0, 0.0]))


def test_similarity_from_corr_clips_negatives_and_rejects_nan():
    z = similarity_from_corr([[1, -0.4], [-0.4, 1]])
    assert z.tolist() == [[1, 0], [0, 1]]
    with pytest.raises(AssertionError):
        similarity_from_corr([[1, np.nan], [np.nan, 1]])


def test_mean_offdiag():
    r = np.array([[1, .2, .6], [.2, 1, -.4], [.6, -.4, 1]])
    p = np.array([0.5, 0.25, 0.25])
    expect = (2 * .125 * .2 + 2 * .125 * .6 + 2 * .0625 * -.4) / (2 * (.125 + .125 + .0625))
    assert np.isclose(mean_offdiag(p, r), expect)
    assert np.isnan(mean_offdiag([1.0], [[1.0]]))
