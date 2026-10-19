"""This project's own RMTB uncertainty-quantification wrapper.

Implements the same formula as `krig_splines.rmtb_uq.RMTBUQ` (Gemini's,
patched) -- `Var(y_hat(x*)) = sigma^2 * f(x*)^T mtx^-1 f(x*)`, the
`Cov=sigma^2*A` posterior-variance identity (Research Plan Section 2.7) with
`A = mtx^-1` -- but via a deliberately different code path, for a genuine
independent cross-check, following the same pattern already used for
`krig_splines.csaps_grid_uq` vs `krig_splines.grid_smooth_spline_uq`:

- Composition around a plain, unmodified `smt.surrogate_models.RMTB`, not a
  subclass overriding `_train` -- this class never touches SMT's internal
  training machinery, only its post-training public/semi-public outputs
  (`mtx`, `full_jac_dict`, `_compute_prediction_mtx`), already individually
  confirmed in `07_phaseD_rmtb_variance.ipynb`.
- A single dense Cholesky factorization of `mtx`, solved in one batched call
  against every query point's basis row at once (`scipy.linalg.cho_factor`/
  `cho_solve`) -- a different library and algorithm from Gemini's sparse
  per-point SuperLU factorization (`scipy.sparse.linalg.factorized`), not
  just a different call site for the same routine. `mtx` is confirmed SPD
  (`full_hess`, `Phi^T Phi` both PSD; `cond(mtx)` safe per Step 3's sweep) --
  the right precondition for Cholesky rather than a general LU.
- `sigma2="wahba"` (GCV-style, RSS/(n-edf)) or an explicit known value,
  matching `krig_splines.csaps_grid_uq.NdGridCubicSmoothingSplineUQ`'s
  convention, rather than always estimating internally.

`predict()` matches the interface shared by the other three UQ wrapper
classes in this package, so all of them can reuse `krig_splines.uq_plotting`.
See `notebooks/iter_01/07_phaseD_rmtb_variance.ipynb` for the comparison
against Gemini's implementation.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm
from smt.surrogate_models import RMTB


class RMTBUQ:
    def __init__(self, sigma2="wahba", **rmtb_kwargs):
        rmtb_kwargs.setdefault("print_global", False)
        self._model = RMTB(**rmtb_kwargs)
        self._sigma2_arg = sigma2
        self.sigma2 = None
        self.edf = None
        self._mtx = None

    def set_training_values(self, X, y):
        self._model.set_training_values(X, y)
        self._X_train = np.asarray(X, float)
        self._y_train = np.ravel(y)
        return self

    def train(self):
        self._model.train()
        self._mtx = self._model.mtx.toarray()
        self._mtx_chol = cho_factor(self._mtx, lower=True)
        Phi, _PhiT, _c = self._model.full_jac_dict[0]
        Phi = Phi.toarray()

        n = len(self._y_train)
        # Effective degrees of freedom: tr(Phi @ mtx^-1 @ Phi^T) ==
        # tr(mtx^-1 @ Phi^T @ Phi) by the cyclic property of trace -- same
        # quantity Gemini's class computes column-by-column via sparse
        # solves; computed here via the Cholesky factor above, reused from
        # `predict_variances`'s machinery rather than a fresh dense solve.
        self.edf = float(np.trace(cho_solve(self._mtx_chol, Phi.T @ Phi)))

        if self._sigma2_arg == "wahba":
            resid = self._y_train - self._model.predict_values(self._X_train).ravel()
            rss = float(resid @ resid)
            self.sigma2 = rss / max(n - self.edf, 1e-8)
        else:
            self.sigma2 = float(self._sigma2_arg)
        return self

    def _basis_row(self, x):
        f_mtx = self._model._compute_prediction_mtx(np.asarray(x, float), 0)
        return np.asarray(f_mtx.toarray() if hasattr(f_mtx, "toarray") else f_mtx)

    def predict_values(self, x):
        return self._model.predict_values(np.asarray(x, float)).ravel()

    def predict_variances(self, x):
        # One batched Cholesky solve for every query point at once:
        # V = mtx^-1 @ F^T  (m, M), then Var_i = sigma2 * F_i . V[:, i].
        F = self._basis_row(x)
        V = cho_solve(self._mtx_chol, F.T)
        var = self.sigma2 * np.sum(F * V.T, axis=1)
        return np.clip(var, 0.0, None)

    def predict(self, x_pred, return_std=False, return_ci=False, alpha=0.05):
        """Predict mean/uncertainty, matching the interface shared by
        `krig_splines.rmtb_uq.RMTBUQ`, `krig_splines.csaps_grid_uq.
        NdGridCubicSmoothingSplineUQ`, and `krig_splines.grid_smooth_spline_uq.
        NdGridCubicSmoothingSplineUQ`.

        Parameters
        ----------
        x_pred : array-like of shape (M, 2)
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
        x_pred = np.asarray(x_pred, float)
        mean = self.predict_values(x_pred)
        if not (return_std or return_ci):
            return mean

        var = self.predict_variances(x_pred)
        std = np.sqrt(var)
        res = {"mean": mean, "var": var, "std": std}
        if return_ci:
            z = norm.ppf(1.0 - alpha / 2.0)
            res["ci_lower"] = mean - z * std
            res["ci_upper"] = mean + z * std
            res["pi_lower"] = mean - z * np.sqrt(var + self.sigma2)
            res["pi_upper"] = mean + z * np.sqrt(var + self.sigma2)
        return res
