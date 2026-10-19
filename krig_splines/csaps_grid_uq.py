"""2-D csaps.NdGridCubicSmoothingSpline with Bayesian posterior variance.

Extracted from notebooks/iter_01/05_csaps_variance_derivation.ipynb, where the
derivation and three independent verifications live. The formula is not
re-derived here -- this module is the packaged result:

    Var(F[p,q]) = sigma^2 * S1[p,p] * S2[q,q]

with S1, S2 the 1-D natural-cubic-smoothing-spline smoother matrices
PolyharmonicSmoothingSpline(m=2, d=1) already implements, and sigma^2 either
known or estimated via the Wahba-style RSS/(n-edf) convention
(edf = trace(S1)*trace(S2), exact).

`predict()` matches the interface of `krig_splines.grid_smooth_spline_uq.
NdGridCubicSmoothingSplineUQ` (an independent implementation) so both can share
the same plotting helpers (`krig_splines.uq_plotting`) -- see
notebooks/iter_01/06_csaps_variance_implementation_comparison.ipynb.
"""

import numpy as np
from csaps import NdGridCubicSmoothingSpline
from scipy.stats import norm

from krig_splines.polyharmonic import PolyharmonicSmoothingSpline


class NdGridCubicSmoothingSplineUQ(NdGridCubicSmoothingSpline):
    """2-D csaps.NdGridCubicSmoothingSpline with Bayesian posterior variance.

    Derivation and verification: 05_csaps_variance_derivation.ipynb. Only
    2-D grids are supported (the variance derivation has not been extended to
    N-D here).
    """

    def __init__(self, xdata, ydata, smooth=None, sigma2="wahba"):
        super().__init__(xdata, ydata, smooth=smooth)
        if len(xdata) != 2:
            raise NotImplementedError("variance is only implemented for 2-D grids")

        x1, x2 = (np.asarray(a, float) for a in xdata)
        p1, p2 = self.smooth
        if not (0 < p1 <= 1 and 0 < p2 <= 1):
            raise ValueError("variance requires smooth in (0, 1] on both axes (p=0 is a "
                              "degenerate lam=inf line fit, not handled here)")
        lam1, lam2 = (1 - p1) / p1, (1 - p2) / p2
        self._lam = (lam1, lam2)

        # variance does not depend on the data y -- only on x-locations and lam
        self._axis1 = PolyharmonicSmoothingSpline(x1.reshape(-1, 1), np.zeros_like(x1), m=2, lam=lam1, sigma2=1.0)
        self._axis2 = PolyharmonicSmoothingSpline(x2.reshape(-1, 1), np.zeros_like(x2), m=2, lam=lam2, sigma2=1.0)

        var1_grid = self._axis1.variance(x1.reshape(-1, 1))
        var2_grid = self._axis2.variance(x2.reshape(-1, 1))
        self._edf = float(var1_grid.sum() * var2_grid.sum())

        n1, n2 = len(x1), len(x2)
        Zhat = np.asarray(self(xdata))
        rss = float(np.sum((np.asarray(ydata) - Zhat) ** 2))
        if sigma2 == "wahba":
            self._sigma2 = rss / max(n1 * n2 - self._edf, 1e-8)
        else:
            self._sigma2 = float(sigma2)

    @property
    def sigma2(self):
        return self._sigma2

    @property
    def edf(self):
        return self._edf

    def variance(self, xq):
        """Posterior variance on the cartesian grid xq=[x1_query, x2_query],
        same query convention as __call__."""
        x1q, x2q = (np.asarray(a, float) for a in xq)
        v1 = self._axis1.variance(x1q.reshape(-1, 1))
        v2 = self._axis2.variance(x2q.reshape(-1, 1))
        return self._sigma2 * np.outer(v1, v2)

    def predict(self, x_pred, return_std=False, return_ci=False, alpha=0.05):
        """Predict mean/uncertainty, matching
        krig_splines.grid_smooth_spline_uq.NdGridCubicSmoothingSplineUQ.predict's
        interface so both implementations can share the same plotting code.

        Parameters
        ----------
        x_pred : (N, 2) array-like of scattered points, or a tuple/list
            (x1_query, x2_query) of 1-D arrays queried as a cartesian grid
            (same convention as __call__/variance).
        return_std, return_ci : bool
        alpha : float
            Significance level for the 95%-style intervals (default 0.05).

        Returns
        -------
        np.ndarray (just the mean) if neither flag is set, else a dict with
        'mean', 'var', 'std', and -- if return_ci -- 'ci_lower'/'ci_upper'
        (uncertainty of the mean surface) and 'pi_lower'/'pi_upper'
        (uncertainty of a new noisy observation, i.e. variance + sigma^2).

        Unlike the grid-based `variance()`, scattered (N, 2) queries are
        evaluated exactly (no re-interpolation): the underlying
        PolyharmonicSmoothingSpline axis models are evaluated once on each
        axis's *unique* query coordinates (cheap, cartesian), and the result is
        gathered back to the N requested, possibly-off-grid pairs.
        """
        if isinstance(x_pred, (tuple, list)) and not isinstance(x_pred, np.ndarray):
            x1q, x2q = (np.asarray(a, float) for a in x_pred)
            mean = np.asarray(self([x1q, x2q]))
            var = self.variance([x1q, x2q])
        else:
            x_pred = np.asarray(x_pred, float)
            x1_p, x2_p = x_pred[:, 0], x_pred[:, 1]
            x1u, inv1 = np.unique(x1_p, return_inverse=True)
            x2u, inv2 = np.unique(x2_p, return_inverse=True)
            mean_grid = np.asarray(self([x1u, x2u]))
            var_grid = self.variance([x1u, x2u])
            mean = mean_grid[inv1, inv2]
            var = var_grid[inv1, inv2]

        var = np.clip(var, 0.0, None)
        std = np.sqrt(var)

        if not (return_std or return_ci):
            return mean

        res = {"mean": mean, "var": var, "std": std}
        if return_ci:
            z = norm.ppf(1.0 - alpha / 2.0)
            res["ci_lower"] = mean - z * std
            res["ci_upper"] = mean + z * std
            res["pi_lower"] = mean - z * np.sqrt(var + self.sigma2)
            res["pi_upper"] = mean + z * np.sqrt(var + self.sigma2)
        return res
