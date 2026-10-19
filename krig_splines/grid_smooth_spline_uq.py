import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.stats import norm
import csaps


class NdGridCubicSmoothingSplineUQ:
    """
    Extends 2D Tensor-Product Cubic Smoothing Splines with Uncertainty Quantification (UQ).
    
    Computes:
      - Noise variance (sigma^2) and Effective Degrees of Freedom (edf)
      - Posterior surface variance and standard deviation
      - 95% Confidence Intervals (for the underlying mean surface)
      - 95% Prediction Intervals (for future noisy observations)
    """
    def __init__(self, xdata, ydata, smooth=None):
        """
        Parameters
        ----------
        xdata : tuple of 1D arrays
            Grid coordinates (x1, x2).
        ydata : 2D array
            Grid response values of shape (len(x1), len(x2)).
        smooth : float or tuple of floats, optional
            Smoothing parameter(s) p in (0, 1]. Default is 0.5.
        """
        self.xdata = [np.asarray(x, dtype=float) for x in xdata]
        self.ydata = np.asarray(ydata, dtype=float)
        
        if len(self.xdata) != 2:
            raise NotImplementedError("2D grid spline UQ is currently supported.")
            
        self.N1, self.N2 = len(self.xdata[0]), len(self.xdata[1])
        if self.ydata.shape != (self.N1, self.N2):
            raise ValueError(f"ydata shape {self.ydata.shape} must match grid dimensions ({self.N1}, {self.N2}).")
            
        # Parse smoothing parameters p1, p2
        if smooth is None:
            self.p = (0.5, 0.5)
        elif isinstance(smooth, (int, float)):
            self.p = (float(smooth), float(smooth))
        else:
            self.p = (float(smooth[0]), float(smooth[1]))
            
        self._csaps_obj = csaps.NdGridCubicSmoothingSpline(self.xdata, self.ydata, smooth=self.p)
            
        # 1. Compute 1D Reinsch Smoother Matrices A1 and A2
        self.A1 = self._compute_1d_smoother_matrix(self.xdata[0], self.p[0])
        self.A2 = self._compute_1d_smoother_matrix(self.xdata[1], self.p[1])
        
        # 2. Fitted grid values: \hat{Y} = A1 @ Y @ A2^T
        self.y_hat_grid = self.A1 @ self.ydata @ self.A2.T
        
        # 3. Residuals & Effective Degrees of Freedom (edf = tr(A1) * tr(A2))
        residuals = self.ydata - self.y_hat_grid
        self.rss = np.sum(residuals**2)
        self.edf1 = np.trace(self.A1)
        self.edf2 = np.trace(self.A2)
        self.edf = self.edf1 * self.edf2
        
        # Residual Variance Estimate \hat{\sigma}^2
        denom = max(1.0, self.N1 * self.N2 - self.edf)
        self.sigma2 = self.rss / denom
        
        # 4. Posterior Surface Variance on Grid: Var(\hat{y}_{ij}) = \sigma^2 * A1_{ii} * A2_{jj}
        diag_A1 = np.diag(self.A1)
        diag_A2 = np.diag(self.A2)
        self.grid_var = self.sigma2 * np.outer(diag_A1, diag_A2)
        
        # 5. Build Bivariate Spline Evaluators for Mean and Variance
        self._mean_spline = RectBivariateSpline(self.xdata[0], self.xdata[1], self.y_hat_grid, kx=3, ky=3)
        self._var_spline = RectBivariateSpline(self.xdata[0], self.xdata[1], self.grid_var, kx=3, ky=3)

    @staticmethod
    def _compute_1d_smoother_matrix(x, p):
        """Builds 1D Reinsch cubic smoothing spline hat matrix A(p)."""
        N = len(x)
        if N < 3:
            return np.eye(N)
        h = np.diff(x)
        
        # Tridiagonal R matrix
        R = np.zeros((N - 2, N - 2))
        for i in range(N - 2):
            R[i, i] = (h[i] + h[i+1]) / 3.0
            if i < N - 3:
                R[i, i+1] = h[i+1] / 6.0
                R[i+1, i] = h[i+1] / 6.0
                
        # Q matrix (second difference operator)
        Q = np.zeros((N, N - 2))
        for i in range(N - 2):
            Q[i, i] = 1.0 / h[i]
            Q[i+1, i] = -1.0 / h[i] - 1.0 / h[i+1]
            Q[i+2, i] = 1.0 / h[i+1]
            
        if p >= 1.0:
            return np.eye(N)

        # PATCHED (see notebooks/iter_01/06_csaps_variance_implementation_comparison.ipynb
        # for the full derivation and verification):
        #
        # The lines below had a spurious factor of 6 in both the system matrix and the
        # leading coefficient. R, Q here use the standard Green & Silverman normalization
        # (R diagonal (h_i+h_{i+1})/3, off-diagonal h_{i+1}/6) -- for that normalization,
        # the textbook hat-matrix identity (penalized-regression solution
        # f_hat = (I + lam*Q*R^-1*Q^T)^-1 * y, with lam=(1-p)/p) is, via the Woodbury
        # matrix identity:
        #
        #     A = I - (1-p) * Q @ [p*R + (1-p)*Q^T Q]^-1 @ Q^T
        #
        # i.e. a plain (1-p), not 6*(1-p). The stray 6 was most likely carried over from
        # csaps's own internal Reinsch system, which legitimately has a pp=6*(1-p) scaling
        # constant -- but only because csaps's internal r/qtw matrices use a DIFFERENT
        # normalization than the R, Q built above (csaps's own r = 6 * this R). Mixing the
        # two conventions' scaling constants produced this bug.
        #
        # Checked directly against csaps.CubicSmoothingSpline's own actual fitted output
        # (the ground truth both implementations are extending), across 5 grid-size /
        # smoothing-parameter combinations: the version below (as originally shipped) is
        # off by 1e-3 to 5e-2; the corrected version matches csaps to ~1e-15 (machine
        # precision) in every case. Re-run the check yourself:
        #   python -c "
        #   import numpy as np
        #   from csaps import CubicSmoothingSpline
        #   from krig_splines.grid_smooth_spline_uq import NdGridCubicSmoothingSplineUQ as U
        #   g = np.linspace(0,1,7); p = 0.95
        #   y = np.sin(3*g) + 0.05*np.random.default_rng(0).standard_normal(7)
        #   y_csaps = CubicSmoothingSpline(g, y, smooth=p)(g)
        #   y_patched = U._compute_1d_smoother_matrix(g, p) @ y
        #   print('max|csaps - patched| =', np.max(np.abs(y_csaps - y_patched)))"
        #
        # Original (buggy) lines, kept for reference -- do not re-enable:
        # M = 6.0 * (1.0 - p) * (Q.T @ Q) + p * R
        # A = np.eye(N) - 6.0 * (1.0 - p) * Q @ np.linalg.solve(M, Q.T)

        M = p * R + (1.0 - p) * (Q.T @ Q)
        A = np.eye(N) - (1.0 - p) * Q @ np.linalg.solve(M, Q.T)
        return A

    def predict(self, x_pred, return_std=False, return_ci=False, alpha=0.05):
        """
        Predicts surface mean and uncertainty bounds at arbitrary points.

        Parameters
        ----------
        x_pred : array-like of shape (M, 2) or tuple of 1D arrays (x1_arr, x2_arr)
            Points to evaluate.
        return_std : bool
            Whether to return standard deviation and variance.
        return_ci : bool
            Whether to return 95% Confidence and Prediction Intervals.
        alpha : float
            Significance level (default 0.05 for 95% confidence).

        Returns
        -------
        dict or np.ndarray
            Dictionary with keys 'mean', 'std', 'var', 'ci_lower', 'ci_upper', 
            'pi_lower', 'pi_upper' if flags are True; otherwise mean array.
        """
        if isinstance(x_pred, tuple):
            x1_p, x2_p = np.asarray(x_pred[0]), np.asarray(x_pred[1])
            mean = self._mean_spline(x1_p, x2_p)
            var = np.clip(self._var_spline(x1_p, x2_p), a_min=0.0, a_max=None)
        else:
            x_pred = np.asarray(x_pred)
            x1_p, x2_p = x_pred[:, 0], x_pred[:, 1]
            mean = self._mean_spline.ev(x1_p, x2_p)
            var = np.clip(self._var_spline.ev(x1_p, x2_p), a_min=0.0, a_max=None)
            
        std = np.sqrt(var)
        
        if not (return_std or return_ci):
            return mean
            
        res = {'mean': mean, 'var': var, 'std': std}
        
        if return_ci:
            z_crit = norm.ppf(1.0 - alpha / 2.0)
            res['ci_lower'] = mean - z_crit * std
            res['ci_upper'] = mean + z_crit * std
            # Prediction Interval includes observation noise variance sigma^2
            res['pi_lower'] = mean - z_crit * np.sqrt(var + self.sigma2)
            res['pi_upper'] = mean + z_crit * np.sqrt(var + self.sigma2)
            
        return res