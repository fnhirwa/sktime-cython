"""Tests for the Cython DTW distance."""

import importlib.util

import numpy as np
import pytest

from sktime_cython.dists_kernels._dtw import dtw_cost_matrix, dtw_distance

# sktime is only present with the `dev` extra; the cibuildwheel wheel-test env
# installs pytest only. find_spec detects absence without importing.
_HAS_SKTIME = importlib.util.find_spec("sktime") is not None


def _series(seed, d=1, m=40):
    rng = np.random.RandomState(seed)
    return rng.normal(size=(d, m))


def test_known_values_1d():
    """Matches sktime's documented 1d doctest value."""
    x = np.array([1, 2, 3, 4])
    y = np.array([5, 6, 7, 8])
    assert dtw_distance(x, y) == pytest.approx(58.0)


def test_known_values_2d():
    """Matches sktime's documented multivariate (dependent DTW) doctest value."""
    x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    y = np.array([[9, 10, 11, 12], [13, 14, 15, 16]])
    assert dtw_distance(x, y) == pytest.approx(512.0)


def test_identical_series_is_zero():
    """DTW of a series with itself is 0."""
    x = _series(0)
    assert dtw_distance(x, x) == pytest.approx(0.0)


def test_symmetry():
    """DTW is symmetric in its arguments."""
    x, y = _series(1), _series(2)
    assert dtw_distance(x, y) == pytest.approx(dtw_distance(y, x))


def test_distance_matches_cost_matrix_corner():
    """The rolling-buffer distance equals cost_matrix[-1, -1] (shared recurrence)."""
    x, y = _series(3), _series(4)
    cm = dtw_cost_matrix(x, y)
    assert dtw_distance(x, y) == pytest.approx(cm[-1, -1])


@pytest.mark.parametrize("d,m", [(1, 25), (3, 30), (5, 18)])
def test_cost_matrix_shape(d, m):
    """cost_matrix is (m1, m2) float64."""
    x, y = _series(5, d=d, m=m), _series(6, d=d, m=m)
    cm = dtw_cost_matrix(x, y)
    assert cm.shape == (m, m)
    assert cm.dtype == np.float64


def test_window_reduces_reachable_region():
    """A tight Sakoe-Chiba band masks off-diagonal cells to inf in the matrix."""
    x, y = _series(7, m=30), _series(8, m=30)
    cm = dtw_cost_matrix(x, y, window=0.1)
    # corners far from the diagonal are out of band -> inf
    assert not np.isfinite(cm[0, -1])
    assert not np.isfinite(cm[-1, 0])
    # the diagonal endpoint stays finite
    assert np.isfinite(cm[-1, -1])


def test_both_bounds_raises():
    """window and itakura_max_slope are mutually exclusive."""
    x, y = _series(0), _series(1)
    with pytest.raises(ValueError, match="only use one bounding matrix"):
        dtw_distance(x, y, window=0.2, itakura_max_slope=0.5)


def test_non_array_raises():
    """Inputs must be numpy arrays."""
    with pytest.raises(ValueError, match="must be a numpy array"):
        dtw_distance([1, 2, 3], np.array([1, 2, 3]))


@pytest.mark.skipif(not _HAS_SKTIME, reason="sktime not installed (dev extra)")
@pytest.mark.parametrize("d", [1, 3])
@pytest.mark.parametrize(
    "kwargs",
    [{}, {"window": 0.2}, {"window": 0.5}, {"itakura_max_slope": 0.5}],
)
def test_cython_matches_numba(d, kwargs):
    """Cython DTW must match sktime's numba implementation (groundtruth)."""
    from sktime.dists_kernels._numba_distances import dtw_distance as numba_dtw

    x, y = _series(11, d=d, m=50), _series(12, d=d, m=50)

    expected = numba_dtw(x, y, **kwargs)
    got = dtw_distance(x, y, **kwargs)
    np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-9)


def test_unbalanced_lengths():
    """DTW between time series of significantly different lengths."""
    x = _series(0, d=2, m=5)
    y = _series(1, d=2, m=30)
    cm = dtw_cost_matrix(x, y)
    assert cm.shape == (5, 30)
    assert dtw_distance(x, y) == pytest.approx(cm[-1, -1])


def test_minimal_length():
    """1-sample time series."""
    x = np.array([[2.0]])
    y = np.array([[5.0]])
    assert dtw_distance(x, y) == pytest.approx(9.0)


def test_non_c_contiguous_input():
    """Handles non-C-contiguous arrays (Fortran-ordered or sliced)."""
    x = np.asfortranarray(_series(0, m=20))
    y = _series(1, m=40)[:, ::2]  # Strided slice
    # Should complete without error or segfault
    res = dtw_distance(x, y)
    assert np.isfinite(res)


def test_distance_and_cost_matrix_parity_with_window():
    """Verify rolling buffer distance equals cost matrix corner WITH windowing."""
    x, y = _series(10, m=30), _series(11, m=35)
    kw = {"window": 0.15}
    cm = dtw_cost_matrix(x, y, **kw)
    dist = dtw_distance(x, y, **kw)
    assert dist == pytest.approx(cm[-1, -1])


def test_fully_masked_window_returns_inf():
    """If the window masks out the path to (m1, m2), distance should be inf."""
    x, y = _series(0, m=10), _series(1, m=10)
    bm = np.full((10, 10), np.inf)
    # Top-left cell valid, but no path to end
    bm[0, 0] = 0.0
    assert np.isinf(dtw_distance(x, y, bounding_matrix=bm))
