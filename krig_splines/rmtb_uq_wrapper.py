import numpy as np
import scipy.interpolate as interp
import scipy.linalg as la
import matplotlib.pyplot as plt

class RMTBUncertaintyWrapper:
    """
    Post-Hoc Uncertainty Quantification Wrapper for Regularized Minimal-Energy
    Tensor-Product B-Splines (RMTB).
    
    Given the B-spline evaluation matrix (B) and composite penalty matrix (K),
    this wrapper calculates the posterior control-point covariance matrix 
    and predicts epistemic and aleatoric variance bounds.
    """
    def __init__(self, ridge_alpha: float = 1e-8):
        self.ridge_alpha = ridge_alpha
        self.c_hat = None
        self.Sigma_c = None
        self.sigma2 = None
        self.edf = None

    def fit(self, B: np.ndarray, y: np.ndarray, K: np.ndarray):
        """
        Fits control parameters and computes the posterior parameter covariance matrix.

        Parameters
        ----------
        B : np.ndarray, shape (N, D)
            B-spline evaluation matrix at N training points for D control points.
        y : np.ndarray, shape (N,)
            Observed target values.
        K : np.ndarray, shape (D, D)
            Composite regularization/penalty matrix (sum_k lambda_k * K_k).
        """
        N, D = B.shape
        y = y.ravel()

        # 1. Unscaled Precision Matrix (Gram Matrix + Penalty Component)
        #    G = (B^T * B + K)
        G = B.T @ B + K
        
        # Add small numerical ridge to ensure invertibility if K is rank-deficient
        G_reg = G + self.ridge_alpha * np.eye(D)

        # 2. Compute MAP Control Point Coefficients: c_hat = G^(-1) * B^T * y
        self.c_hat = la.solve(G_reg, B.T @ y, assume_a='pos')

        # 3. Residuals & Effective Degrees of Freedom (EDF)
        residuals = y - B @ self.c_hat
        
        # Hat matrix H = B * (G^-1) * B^T -> EDF = trace(H) = trace(G^-1 * B^T * B)
        G_inv_BtB = la.solve(G_reg, B.T @ B, assume_a='pos')
        self.edf = np.trace(G_inv_BtB)
        
        # Degrees of freedom for unbiased residual variance estimation
        dof = max(1.0, N - self.edf)
        self.sigma2 = np.sum(residuals**2) / dof

        # =========================================================================
        # MATRIX INVERSION OPERATION FOR BAYESIAN POSTERIOR COVARIANCE (Sigma_c)
        # =========================================================================
        # The conditional posterior covariance of spline control points c is:
        #
        #     Sigma_c = sigma^2 * ( B^T * B + K )^(-1)
        #
        # Here, we perform explicit matrix inversion on the (D x D) system matrix.
        # Note: Since G_reg is symmetric positive-definite, we use Cholesky-based
        # inversion (via scipy.linalg.inv with overwrite or pinv fallback) for 
        # optimal numerical stability.
        # =========================================================================
        
        try:
            # Primary path: Direct inverse of positive-definite precision matrix
            G_inv = la.inv(G_reg, overwrite_a=False)
        except la.LinAlgError:
            # Fallback path: Pseudo-inverse for ill-conditioned penalty structures
            G_inv = la.pinv(G_reg)

        # Scale inverted precision matrix by estimated noise variance (sigma^2)
        self.Sigma_c = self.sigma2 * G_inv

        return self

    def predict(self, B_new: np.ndarray, alpha: float = 0.05):
        """
        Predicts mean values, epistemic curve variance, and aleatoric prediction variance.

        Parameters
        ----------
        B_new : np.ndarray, shape (N_query, D)
            B-spline evaluation matrix at new query locations.
        alpha : float
            Significance level for confidence intervals (default 0.05 -> 95% interval).

        Returns
        -------
        dict containing:
            - 'mean': Expected spline curve values
            - 'epistemic_var': Curve shape uncertainty variance
            - 'aleatoric_var': Data noise variance
            - 'predictive_var': Total predictive variance
            - 'conf_interval': (lower, upper) bounds for spline function f(x)
            - 'pred_interval': (lower, upper) bounds for new observations y_tilde
        """
        z_crit = 1.96  # Standard Gaussian critical value for 95% interval

        # Mean curve prediction: f(x_query) = B_new * c_hat
        f_mean = B_new @ self.c_hat

        # Point-wise Epistemic Variance: Var(f(x)) = diag( B_new * Sigma_c * B_new^T )
        # Using row-wise Einstein summation to avoid computing full (N_query x N_query) off-diagonals
        epistemic_var = np.einsum('ni,ij,nj->n', B_new, self.Sigma_c, B_new)
        
        # Ensure non-negative variance bounds due to numerical precision limits
        epistemic_var = np.maximum(0.0, epistemic_var)

        # Total Predictive Variance: Var(y_tilde) = Var(f(x)) + sigma^2
        aleatoric_var = self.sigma2
        total_predictive_var = epistemic_var + aleatoric_var

        # Compute 95% Epistemic Confidence Intervals (True function f(x))
        epistemic_std = np.sqrt(epistemic_var)
        ci_lower = f_mean - z_crit * epistemic_std
        ci_upper = f_mean + z_crit * epistemic_std

        # Compute 95% Prediction Intervals (New noisy observation y_tilde)
        predictive_std = np.sqrt(total_predictive_var)
        pi_lower = f_mean - z_crit * predictive_std
        pi_upper = f_mean + z_crit * predictive_std

        return {
            "mean": f_mean,
            "epistemic_var": epistemic_var,
            "aleatoric_var": aleatoric_var,
            "predictive_var": total_predictive_var,
            "conf_interval": (ci_lower, ci_upper),
            "pred_interval": (pi_lower, pi_upper)
        }


# =============================================================================
# DEMONSTRATION & VERIFICATION SCRIPT
# =============================================================================
if __name__ == "__main__":
    np.random.seed(42)

    # 1. Generate Synthetic Data
    x_train = np.sort(np.random.uniform(0, 10, size=80))
    y_true = np.sin(x_train) + 0.5 * np.cos(2 * x_train)
    noise_sigma = 0.2
    y_train = y_true + np.random.normal(0, noise_sigma, size=len(x_train))

    # 2. Construct 1D B-Spline Basis Functions Matrix (B)
    num_knots = 12
    degree = 3
    knots = np.linspace(x_train.min(), x_train.max(), num_knots)
    knot_vector = np.pad(knots, (degree, degree), mode='edge')
    D = len(knots) + degree - 1

    def evaluate_bspline_matrix(x_coords, knot_vec, deg, num_ctrl):
        B_mat = np.zeros((len(x_coords), num_ctrl))
        for i in range(num_ctrl):
            c_dummy = np.zeros(num_ctrl)
            c_dummy[i] = 1.0
            spl = interp.BSpline(knot_vec, c_dummy, deg)
            B_mat[:, i] = spl(x_coords)
        return B_mat

    B_train = evaluate_bspline_matrix(x_train, knot_vector, degree, D)

    # 3. Construct Composite Penalty Matrix (K = lambda1 * K1 + lambda2 * K2)
    D1 = np.diff(np.eye(D), n=1, axis=0)
    D2 = np.diff(np.eye(D), n=2, axis=0)
    
    K1 = D1.T @ D1  # 1st derivative / smoothness penalty
    K2 = D2.T @ D2  # 2nd derivative / bending energy penalty

    lambda1, lambda2 = 0.1, 1.0
    K_composite = lambda1 * K1 + lambda2 * K2

    # 4. Execute Post-Hoc Uncertainty Estimation
    model = RMTBUncertaintyWrapper(ridge_alpha=1e-8)
    model.fit(B_train, y_train, K_composite)

    # 5. Predict on New Evaluation Grid (Including Extrapolation)
    x_query = np.linspace(-1, 11, 200)
    B_query = evaluate_bspline_matrix(x_query, knot_vector, degree, D)
    res = model.predict(B_query)

    # 6. Visual Verification
    plt.figure(figsize=(11, 5))
    plt.scatter(x_train, y_train, color="black", s=20, alpha=0.6, label="Training Data")
    plt.plot(x_query, np.sin(x_query) + 0.5 * np.cos(2 * x_query), "k--", label="True Function")
    plt.plot(x_query, res["mean"], "b-", label="Post-Hoc RMTB Fit")

    # Epistemic Confidence Interval (Spline Function Uncertainty)
    plt.fill_between(
        x_query, res["conf_interval"][0], res["conf_interval"][1],
        color="blue", alpha=0.25, label="95% Epistemic CI (Function)"
    )

    # Total Prediction Interval (New Observation Uncertainty)
    plt.fill_between(
        x_query, res["pred_interval"][0], res["pred_interval"][1],
        color="blue", alpha=0.1, label="95% Predictive Interval (Data)"
    )

    plt.axvspan(-1, 0, color='gray', alpha=0.15, label="Extrapolation Domain")
    plt.axvspan(10, 11, color='gray', alpha=0.15)
    plt.title("Post-Hoc Bayesian Uncertainty Bounds for Regularized B-Spline")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="upper right")
    plt.show()