"""Kernel functions and the kernel <-> energy correspondence (paper Table 1).

Only the pieces Phase A needs: the Matern-1/2 (`abs_exp`) correlation used in
A3, and the Brownian/Wiener kernel used in A1/A2 as a hand-checkable, exact
RKHS (its native penalty is \\int h'^2, i.e. the kernel whose minimal-norm
interpolant and whose penalised-regression limit are both the natural LINEAR
spline -- Table 1's "degree 1" row). Matern-3/2 and 5/2 (needed from Phase B
onward) are included too since they cost nothing extra here.
"""
import numpy as np


def abs_exp(d, theta):
    """Matern-1/2 correlation (SMT's `abs_exp`): k(d) = exp(-theta*|d|)."""
    return np.exp(-theta * np.abs(d))


def matern32(d, kappa):
    """Matern-3/2 correlation (SMT's `matern32`): (1+kappa|d|) exp(-kappa|d|)."""
    ad = kappa * np.abs(d)
    return (1.0 + ad) * np.exp(-ad)


def matern52(d, kappa):
    """Matern-5/2 correlation (SMT's `matern52`)."""
    ad = kappa * np.abs(d)
    return (1.0 + ad + ad**2 / 3.0) * np.exp(-ad)


def brownian(x1, x2):
    """k(x,x') = min(x,x') -- the Wiener-process kernel.

    Its RKHS norm is exactly \\int h'^2 subject to h(0)=0, so the minimal-norm
    interpolant (nugget=0) and the penalised-regression limit under this
    kernel are both the natural piecewise-LINEAR spline, with no unknown
    normalisation constant to calibrate. Used in Phase A1/A2 as ground truth
    that does not depend on the Matern/Table-1 scaling derivation used in A3.
    """
    return np.minimum(x1, x2)


def iwp2_kernel(x1, x2):
    """Reproducing kernel for the m=2 (cubic) derivative-penalty functional
    \\int h''^2, modulo the degree-1 null space {1, x} (trend
    `poly_basis(x, degree=1)`) -- the covariance of twice-integrated Brownian
    motion (Kimeldorf & Wahba, 1970):

        G(x,x') = min(x,x')^2 * (3*max(x,x') - min(x,x')) / 6

    Table 1's cubic-spline row (m=2, degree 3), without the Matern/kappa
    detour Corollary 5 uses: plugged into `saddle_point_solve` with
    `nugget=lambda` directly (no `4*kappa**3` rescaling -- there is no kappa
    here), this reproduces `scipy.interpolate.make_smoothing_spline(lam=
    lambda)`'s mean to ~1e-12, and `saddle_point_variance` with the same `K`
    gives the companion predictive variance -- see
    `notebooks/spline_variance_without_kriging.ipynb`.

    Like `brownian`, this kernel implicitly anchors at x=0 (it is the
    covariance of a process started at 0); shift x close to 0 first for good
    conditioning if the domain is far from the origin.
    """
    lo = np.minimum(x1, x2)
    hi = np.maximum(x1, x2)
    return lo**2 * (3.0 * hi - lo) / 6.0


def pairwise_abs_diff(x1, x2):
    """|x_i - x_j| for 1-D inputs, broadcasting x1 as column, x2 as row."""
    x1 = np.asarray(x1).reshape(-1, 1)
    x2 = np.asarray(x2).reshape(1, -1)
    return np.abs(x1 - x2)
