"""2-D `csaps.NdGridCubicSmoothingSpline` with Bayesian posterior variance.

Merges what were two independently-developed, independently-validated
implementations in the exploratory (`krig_splines`) codebase --
`csaps_grid_uq.py` and `grid_smooth_spline_uq.py` -- into a single class with
two selectable off-grid evaluation strategies, rather than two classes users
had to choose between by trial. Both strategies compute the *same* exact,
closed-form posterior mean/variance ON the training grid
($\\hat Y=S_1 Z S_2^\\top$, $\\mathrm{Var}(F[p,q])=\\sigma^2 S_1[p,p]S_2[q,q]$,
derived in `05_csaps_variance_derivation.ipynb`); they differ only in how
*off-grid* (scattered) query points in `.predict()` are handled:

- `off_grid_method="exact"` (default): evaluate the per-axis
  `PolyharmonicSmoothingSpline(m=2, d=1)` kernel exactly on each axis's
  *unique* query coordinates, gathered back to the requested points. No
  approximation, at the cost of `O(n_unique)` work per axis per call.
- `off_grid_method="fast"`: build a `scipy.interpolate.RectBivariateSpline`
  re-interpolant of the mean/variance surfaces once, at construction time,
  and evaluate off-grid queries through it (a compiled FITPACK routine).
  Confirmed in this project's own speed benchmark
  (`10_variance_prediction_speed_benchmark.ipynb`) to be `~55x-63x` faster
  than `"exact"` off-grid, at the cost of a small (`~1e-3`), already-documented
  re-interpolation error off-grid (`06_csaps_variance_implementation_comparison.ipynb`).
  Exact ON the training grid either way -- this trade-off only applies to
  scattered queries that don't coincide with the grid.

Only 2-D grids are supported (the variance derivation has not been extended
to N-D here).
"""

import numpy as np
from csaps import NdGridCubicSmoothingSpline
from scipy.interpolate import RectBivariateSpline
from scipy.stats import norm

from .polyharmonic import PolyharmonicSmoothingSpline


class NdGridCubicSmoothingSplineUQ(NdGridCubicSmoothingSpline):
    """2-D `csaps.NdGridCubicSmoothingSpline` with Bayesian posterior variance.

    Parameters
    ----------
    xdata : [x1, x2]
        1-D per-axis grid coordinate arrays.
    ydata : (len(x1), len(x2)) array
        Grid response values.
    smooth : float, (float, float), or None
        `csaps`'s own smoothing parameter(s) $p\\in(0,1]$ per axis.
    sigma2 : "wahba" (default) or a known float
        Noise variance. `"wahba"` uses the GCV-style
        $\\mathrm{RSS}/(n_1n_2-\\mathrm{edf})$ estimator,
        $\\mathrm{edf}=\\mathrm{tr}(S_1)\\cdot\\mathrm{tr}(S_2)$ (exact,
        since the smoother is a Kronecker product).
    off_grid_method : "exact" (default) or "fast"
        See module docstring.
    """

    def __init__(self, xdata, ydata, smooth=None, sigma2="wahba", off_grid_method="exact"):
        super().__init__(xdata, ydata, smooth=smooth)
        if len(xdata) != 2:
            raise NotImplementedError("variance is only implemented for 2-D grids")
        if off_grid_method not in ("exact", "fast"):
            raise ValueError(f"off_grid_method must be 'exact' or 'fast', got {off_grid_method!r}")
        self.off_grid_method = off_grid_method

        x1, x2 = (np.asarray(a, float) for a in xdata)
        p1, p2 = self.smooth
        if not (0 < p1 <= 1 and 0 < p2 <= 1):
            raise ValueError("variance requires smooth in (0, 1] on both axes (p=0 is a "
                              "degenerate lam=inf line fit, not handled here)")
        lam1, lam2 = (1 - p1) / p1, (1 - p2) / p2
        self._lam = (lam1, lam2)

        # Variance does not depend on the data y -- only on x-locations and lam.
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

        if self.off_grid_method == "fast":
            mean_grid = Zhat
            var_grid = self.variance(xdata)
            self._mean_spline = RectBivariateSpline(x1, x2, mean_grid, kx=3, ky=3)
            self._var_spline = RectBivariateSpline(x1, x2, var_grid, kx=3, ky=3)

    @property
    def sigma2(self):
        return self._sigma2

    @property
    def edf(self):
        return self._edf

    def variance(self, xq):
        """Posterior variance on the cartesian grid xq=[x1_query, x2_query],
        same query convention as __call__. Always exact, regardless of
        `off_grid_method` -- that flag only affects scattered queries in
        `.predict()`."""
        x1q, x2q = (np.asarray(a, float) for a in xq)
        v1 = self._axis1.variance(x1q.reshape(-1, 1))
        v2 = self._axis2.variance(x2q.reshape(-1, 1))
        return self._sigma2 * np.outer(v1, v2)

    def predict(self, x_pred, return_std=False, return_ci=False, alpha=0.05):
        """Predict mean/uncertainty.

        Parameters
        ----------
        x_pred : (N, 2) array-like of scattered points, or a tuple/list
            (x1_query, x2_query) of 1-D arrays queried as a cartesian grid
            (same convention as __call__/variance). Grid-tuple queries are
            always exact; `off_grid_method` only affects the (N, 2) scattered
            case.
        return_std, return_ci : bool
        alpha : float
            Significance level for the 95%-style intervals (default 0.05).

        Returns
        -------
        np.ndarray (just the mean) if neither flag is set, else a dict with
        'mean', 'var', 'std', and -- if return_ci -- 'ci_lower'/'ci_upper'
        (uncertainty of the mean surface) and 'pi_lower'/'pi_upper'
        (uncertainty of a new noisy observation, i.e. variance + sigma^2).
        """
        if isinstance(x_pred, (tuple, list)) and not isinstance(x_pred, np.ndarray):
            x1q, x2q = (np.asarray(a, float) for a in x_pred)
            mean = np.asarray(self([x1q, x2q]))
            var = self.variance([x1q, x2q])
        else:
            x_pred = np.asarray(x_pred, float)
            x1_p, x2_p = x_pred[:, 0], x_pred[:, 1]
            if self.off_grid_method == "fast":
                mean = self._mean_spline.ev(x1_p, x2_p)
                var = self._var_spline.ev(x1_p, x2_p)
            else:
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
