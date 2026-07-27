# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython dynamic time warping (DTW) kernels.

Ahead-of-time compiled port of sktime's numba ``_cost_matrix`` kernel
(``sktime.dists_kernels._numba_distances._dtw_numba._cost_matrix``). Same math,
no numba JIT warmup. Two entry points share one recurrence:

* ``cost_matrix`` fills the full dynamic-programming matrix (the reusable
  kernel; foundation for alignment-path recovery in later phases);
* ``distance`` computes only the final DTW cost via a rolling two-row buffer,
  avoiding the O(m1 * m2) allocation when the cost matrix is not needed.

Both use the "dependent" multivariate squared-Euclidean local cost
(DTW_D, Shokoohi-Yekta et al. 2017) and are numerically equivalent to the
numba implementation (verified against it as groundtruth in tests).

The ``bounding_matrix`` follows sktime's sentinel convention: an in-bound
cell holds a finite value (0.0) and an out-of-bound cell holds ``inf``.
To allow compilation with fast-math flags (e.g., ``-ffast-math`` / ``-O3``),
the check is implemented via direct exponent bit-masking (``safe_isfinite``)
rather than ``libc.math.isfinite``, avoiding compiler collapse of standard C
library math macros.

Note on compiler flags: while ``safe_isfinite`` handles fast-math compilation,
ensure that sentinel comparisons involving ``INFINITY`` are validated in your
build environment under ``-ffinite-math-only``.

References
----------
Port of the numba kernel by chrisholder and TonyBagnall in sktime
(BSD-3-Clause), ``sktime/dists_kernels/_numba_distances/_dtw_numba.py``.
Original DTW algorithm: Sakoe & Chiba, IEEE TASSP 26(1):43-49, 1978.
"""

import numpy as np

cimport numpy as cnp
from libc.math cimport INFINITY, fmin

cnp.import_array()


# Avoids strict-aliasing bugs by inspecting double bits as an integer
cdef union Conv:
    double d
    unsigned long long u


cdef inline double _min3(double a, double b, double c) noexcept nogil:
    return fmin(fmin(a, b), c)


cdef inline bint safe_isfinite(double x) noexcept nogil:
    cdef Conv conv
    conv.d = x
    # Standard IEEE 754 double: Exponent is bits 52..62 (11 bits)
    return (conv.u & 0x7FF0000000000000ULL) != 0x7FF0000000000000ULL


cdef inline double _local_cost(
    double[:, ::1] x, double[:, ::1] y, Py_ssize_t i, Py_ssize_t j, Py_ssize_t d
) noexcept nogil:
    """Dependent multivariate squared-Euclidean cost between x[:, i] and y[:, j]."""
    cdef Py_ssize_t k
    cdef double diff, s = 0.0
    for k in range(d):
        diff = x[k, i] - y[k, j]
        s += diff * diff
    return s


def cost_matrix(
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] x,
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] y,
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] bounding_matrix,
):
    """Full DTW cost matrix.

    Port of ``_cost_matrix``. ``x`` and ``y`` are ``(d, m)`` channels-first
    series (possibly different lengths). Returns the ``(m1, m2)`` cost matrix
    (the numba code's ``cost_matrix[1:, 1:]`` slice); the DTW distance is its
    bottom-right entry.
    """
    cdef Py_ssize_t d = x.shape[0]
    cdef Py_ssize_t m1 = x.shape[1]
    cdef Py_ssize_t m2 = y.shape[1]
    cdef double[:, ::1] xv = x
    cdef double[:, ::1] yv = y
    cdef double[:, ::1] bm = bounding_matrix

    # (m1 + 1, m2 + 1) padded with an inf border; [0, 0] = 0 seeds the path.
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] full = np.full(
        (m1 + 1, m2 + 1), INFINITY, dtype=np.float64
    )
    cdef double[:, ::1] c = full
    c[0, 0] = 0.0

    cdef Py_ssize_t i, j
    with nogil:
        for i in range(m1):
            for j in range(m2):
                if safe_isfinite(bm[i, j]):
                    c[i + 1, j + 1] = _local_cost(xv, yv, i, j, d) + _min3(
                        c[i, j + 1], c[i + 1, j], c[i, j]
                    )

    return full[1:, 1:]


def distance(
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] x,
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] y,
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] bounding_matrix,
):
    """DTW distance only, via a rolling two-row buffer.

    Numerically identical to ``cost_matrix(x, y, bm)[-1, -1]`` but uses
    ``O(m2)`` scratch instead of ``O(m1 * m2)``.
    """
    cdef Py_ssize_t d = x.shape[0]
    cdef Py_ssize_t m1 = x.shape[1]
    cdef Py_ssize_t m2 = y.shape[1]
    cdef double[:, ::1] xv = x
    cdef double[:, ::1] yv = y
    cdef double[:, ::1] bm = bounding_matrix

    # prev = row i (padded), curr = row i + 1 (padded); length m2 + 1.
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] prev_a = np.full(
        m2 + 1, INFINITY, dtype=np.float64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] curr_a = np.empty(
        m2 + 1, dtype=np.float64
    )
    cdef double[::1] prev = prev_a
    cdef double[::1] curr = curr_a
    prev[0] = 0.0

    cdef Py_ssize_t i, j
    cdef double[::1] tmp
    cdef double result
    with nogil:
        for i in range(m1):
            curr[0] = INFINITY
            for j in range(m2):
                if safe_isfinite(bm[i, j]):
                    curr[j + 1] = _local_cost(xv, yv, i, j, d) + _min3(
                        prev[j + 1], curr[j], prev[j]
                    )
                else:
                    curr[j + 1] = INFINITY
            tmp = prev
            prev = curr
            curr = tmp
        result = prev[m2]

    return result
