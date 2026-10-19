import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.stats import norm

# ORIGINAL (Gemini's code): `from smt.surrogates import RMTB` -- this module
# does not exist in the installed SMT (2.15.0; confirmed via
# `ModuleNotFoundError: No module named 'smt.surrogates'`). The real path is
# `smt.surrogate_models`, already used throughout this project.
from smt.surrogate_models import RMTB


class RMTBUQ(RMTB):
    """
    Subclass of SMT's RMTB (Regularized Minimal-energy Tensor-product B-spline)
    extended with Uncertainty Quantification (UQ) capabilities.

    Computes:
      - Noise variance (sigma^2) and Effective Degrees of Freedom (edf)
      - Posterior surface prediction variance and standard error
      - 95% Confidence Intervals (for the underlying mean surface)
      - 95% Prediction Intervals (for new noisy observations)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sigma2 = None
        self.edf = None
        self.A_factorized = None
        self.A_sparse = None

    def _train(self):
        # 1. Run standard SMT RMTB training to optimize control weights w
        super()._train()

        # 2. Extract training data dimensions
        # ORIGINAL (Gemini's code): `self.x` / `self.y` -- neither attribute
        # exists on this SMT version. SMT stores training data as
        # `self.training_points[name][kx] = (xt, yt)`, `name=None` for the
        # default training set and `kx=0` for function values (confirmed by
        # direct inspection: matches the `X`, `y` passed to
        # `set_training_values` exactly).
        # X_train = self.x
        # y_train = self.y.ravel()
        X_train, y_train_col = self.training_points[None][0]
        y_train = y_train_col.ravel()
        N = X_train.shape[0]

        # 3. Retrieve basis design matrix F (N x K) and energy matrix H (K x K)
        F_train = self._get_basis_matrix(X_train)
        # ORIGINAL (Gemini's code): `self.H` -- does not exist. The real
        # attribute is `self.full_hess` (confirmed in this project's
        # `07_phaseD_rmtb_variance.ipynb`, Step 2).
        # H_energy = self.H
        H_energy = self.full_hess

        # ORIGINAL (Gemini's code):
        # alpha = self.options["energy_evals"][0]
        # beta = self.options["regularize"]
        # Neither option key exists -- the real names are `energy_weight`
        # and `regularization_weight`. More importantly, re-scaling `H_energy`
        # by them here would be a bug even with the right names:
        # `self.full_hess` already has both baked in by SMT internally
        # (confirmed in notebook 07, Step 2: `max|mtx-(full_hess+Phi^T Phi)|
        # = 1.4e-17` with `full_hess` used as-is, no extra scaling). Applying
        # `alpha`/`beta` again here would double-count both terms.
        K = F_train.shape[1]

        # 4. Construct System Matrix A = F^T F + H (H already carries the
        # energy weight and the regularization ridge -- see above). This is
        # exactly what SMT itself exposes as `self.mtx`; rebuilding it
        # explicitly here (rather than just aliasing `self.mtx`) keeps this
        # class's derivation self-contained and independently checkable --
        # confirmed to match `self.mtx` to float64 precision during this
        # patch's validation (see `07_phaseD_rmtb_variance.ipynb`).
        # ORIGINAL (Gemini's code, double-counts alpha/beta -- see above):
        # I_K = sp.eye(K, format="csr")
        # self.A_sparse = (F_train.T @ F_train + alpha * (H_energy + beta * I_K)).tocsc()
        self.A_sparse = (F_train.T @ F_train + H_energy).tocsc()

        # 5. Precompute Sparse SuperLU Factorization for O(1) Local Variance Solves
        self.A_factorized = spla.factorized(self.A_sparse)

        # 6. Estimate Residual Variance sigma^2 & Effective Degrees of Freedom (edf)
        y_pred_train = self._predict_values(X_train).ravel()
        residuals = y_train - y_pred_train
        rss = np.sum(residuals**2)

        # Compute edf = Tr(A^-1 * (F^T F)), one sparse solve per control
        # point -- the same GCV-style effective-degrees-of-freedom quantity
        # used in `07_phaseD_rmtb_variance.ipynb` Step 5b (there computed via
        # a single dense `np.trace(np.linalg.solve(mtx, Phi.T@Phi))` on a
        # 400x400 system; this column-by-column sparse-solve form is the
        # more scalable choice for RMTB's normal regime of large K).
        FtF = (F_train.T @ F_train).tocsc()
        A_inv_FtF_diag_sum = 0.0
        for k in range(K):
            col = FtF.getcol(k).toarray().ravel()
            if np.any(col != 0):
                sol_k = self.A_factorized(col)
                A_inv_FtF_diag_sum += sol_k[k]

        self.edf = A_inv_FtF_diag_sum
        denom = max(1.0, N - self.edf)
        self.sigma2 = rss / denom

    def _get_basis_matrix(self, x):
        """Basis design matrix Phi at points `x`, via SMT's actual API."""
        # ORIGINAL (Gemini's code, `_get_F_matrix_compat`): none of these
        # method names exist on this SMT version. The real method is
        # `_compute_prediction_mtx(x, 0)` (confirmed in notebook 07 to
        # reproduce `full_jac_dict[0]`'s Phi exactly, including at training
        # points -- `kx=0` for function values, matching the `training_points`
        # convention above).
        # if hasattr(self, "_compute_F_matrix"):
        #     return self._compute_F_matrix(x)
        # elif hasattr(self, "_get_F_matrix"):
        #     return self._get_F_matrix(x)
        # elif hasattr(self, "_eval_B_splines"):
        #     return self._eval_B_splines(x)
        # else:
        #     return self._eval_b_splines(x)
        return self._compute_prediction_mtx(x, 0)

    def predict_variances(self, x):
        """
        Computes pointwise prediction surface variances at evaluation points x.

        Parameters
        ----------
        x : np.ndarray of shape (M, D)
            Evaluation points.

        Returns
        -------
        variances : np.ndarray of shape (M,)
            Pointwise surface variance at each prediction point.
        """
        # CSR, not CSC, for efficient row-by-row access below (CSC row
        # slicing works but silently converts per call, which is wasteful
        # over many query points).
        F_test = self._get_basis_matrix(x).tocsr()
        M = x.shape[0]
        variances = np.zeros(M)

        for i in range(M):
            f_i = F_test.getrow(i).toarray().ravel()
            if np.any(f_i != 0):
                # Solve A u_i = f_i^T
                u_i = self.A_factorized(f_i)
                # Var = sigma2 * (f_i . u_i)
                variances[i] = self.sigma2 * np.dot(f_i, u_i)

        return np.maximum(0.0, variances)

    def predict_ci(self, x, alpha=0.05):
        """
        Computes predictions, surface confidence intervals, and prediction intervals.

        Parameters
        ----------
        x : np.ndarray of shape (M, D)
            Evaluation points.
        alpha : float
            Significance level (default 0.05 for 95% bounds).

        Returns
        -------
        dict containing 'mean', 'std', 'var', 'ci_lower', 'ci_upper',
                        'pi_lower', 'pi_upper'
        """
        mean = self.predict_values(x).ravel()
        var = self.predict_variances(x)
        std = np.sqrt(var)

        z_crit = norm.ppf(1.0 - alpha / 2.0)

        return {
            "mean": mean,
            "std": std,
            "var": var,
            "ci_lower": mean - z_crit * std,
            "ci_upper": mean + z_crit * std,
            "pi_lower": mean - z_crit * np.sqrt(var + self.sigma2),
            "pi_upper": mean + z_crit * np.sqrt(var + self.sigma2),
        }

    def predict(self, x_pred, return_std=False, return_ci=False, alpha=0.05):
        """Added during this project's patch (not part of Gemini's original
        code) so this class matches the `predict(x, return_std=, return_ci=,
        alpha=)` interface already shared by
        `krig_splines.csaps_grid_uq.NdGridCubicSmoothingSplineUQ` and
        `krig_splines.grid_smooth_spline_uq.NdGridCubicSmoothingSplineUQ`,
        letting all of them reuse `krig_splines.uq_plotting`.

        Returns mean array if neither flag is set, else the `predict_ci`
        dict (which already has every key this interface needs)."""
        x_pred = np.asarray(x_pred, float)
        if not (return_std or return_ci):
            return self.predict_values(x_pred).ravel()
        return self.predict_ci(x_pred, alpha=alpha)
