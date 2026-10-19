"""Prediction and plotting helpers for single-output SMT surrogates of the RANS CRM wing data.

Inputs follow the SMT example convention: xt[:, 0] = alpha in radians, xt[:, 1] = Mach.
Every function takes one trained model with one output (e.g. a Cd-only or Cl-only RMTB).
"""

import numpy as np
import matplotlib.pyplot as plt

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi


# ---------------------------------------------------------------------------
# Prediction (no matplotlib here)
# ---------------------------------------------------------------------------

def _predict(model, alpha_deg, mach):
    """Evaluate a single-output model at broadcastable alpha (deg) / Mach arrays."""
    alpha_deg, mach = np.broadcast_arrays(np.asarray(alpha_deg, float),
                                          np.asarray(mach, float))
    x = np.column_stack([alpha_deg.ravel() * DEG2RAD, mach.ravel()])
    y = model.predict_values(x)
    if y.ndim == 2 and y.shape[1] != 1:
        raise ValueError(f"Expected a single-output model, got {y.shape[1]} outputs")
    return y.ravel().reshape(alpha_deg.shape)


def predict_alpha_sweeps(model, mach_numbers, alpha_deg):
    """Return {mach: y(alpha)} for each Mach number."""
    return {m: _predict(model, alpha_deg, m) for m in mach_numbers}


def predict_mach_sweeps(model, alphas_deg, mach):
    """Return {alpha_deg: y(mach)} for each angle of attack."""
    return {a: _predict(model, a, mach) for a in alphas_deg}


def predict_grid(model, alpha_deg, mach):
    """Return (ALPHA_deg, MACH, Y) meshgrids, shape (len(alpha), len(mach))."""
    A, M = np.meshgrid(alpha_deg, mach, indexing="ij")
    return A, M, _predict(model, A, M)


# ---------------------------------------------------------------------------
# Plotting (no model calls here)
# ---------------------------------------------------------------------------

def plot_alpha_sweeps(ax, alpha_deg, sweeps, xt=None, yt=None, label="y"):
    for (mach, y), color in zip(sweeps.items(), plt.cm.tab10.colors):
        ax.plot(alpha_deg, y, color=color, label=f"M={mach}")
        if xt is not None:
            mask = np.isclose(xt[:, 1], mach)
            ax.plot(xt[mask, 0] * RAD2DEG, yt[mask], "o", color=color)
    ax.set(xlabel="alpha (deg)", ylabel=label)
    ax.legend()


def plot_mach_sweeps(ax, mach, sweeps, label="y"):
    for (alpha, y), color in zip(sweeps.items(), plt.cm.tab10.colors):
        ax.plot(mach, y, color=color, label=f"alpha={alpha}°")
    ax.set(xlabel="Mach number", ylabel=label)
    ax.legend()


def plot_contour(ax, A, M, Y, xt=None, label="y", levels=20):
    pcm = ax.pcolormesh(M, A, Y, cmap="rainbow", shading="auto")
    ax.contour(M, A, Y, levels, colors="k", linewidths=0.5)
    if xt is not None:
        ax.plot(xt[:, 1], xt[:, 0] * RAD2DEG, "ko", markersize=3)
    ax.figure.colorbar(pcm, ax=ax)
    ax.set(xlabel="Mach number", ylabel="alpha (deg)", title=label)


def plot_model(model, xt=None, yt=None, label="y",
               mach_numbers=(0.45, 0.68, 0.80, 0.86),
               alphas_deg=(2.0, 4.0, 6.0),
               alpha_range=(0.0, 8.0), mach_range=(0.45, 0.86),
               num=500, num_grid=50):
    """Predict with one model and draw alpha sweeps, Mach sweeps and a contour map."""
    yt = None if yt is None else np.asarray(yt).ravel()

    alpha_line = np.linspace(*alpha_range, num)
    mach_line = np.linspace(*mach_range, num)
    alpha_sweeps = predict_alpha_sweeps(model, mach_numbers, alpha_line)
    mach_sweeps = predict_mach_sweeps(model, alphas_deg, mach_line)
    A, M, Y = predict_grid(model,
                           np.linspace(*alpha_range, num_grid),
                           np.linspace(*mach_range, num_grid))

    fig, axs = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    plot_alpha_sweeps(axs[0], alpha_line, alpha_sweeps, xt, yt, label)
    plot_mach_sweeps(axs[1], mach_line, mach_sweeps, label)
    plot_contour(axs[2], A, M, Y, xt, label)
    fig.suptitle(label)
    return fig, axs


if __name__ == "__main__":
    from smt.examples.rans_crm_wing.rans_crm_wing import get_rans_crm_wing
    from smt.surrogate_models import RMTB

    xt, yt, xlimits = get_rans_crm_wing()

    models = {}
    for i, name in enumerate(["CD", "CL"]):
        m = RMTB(num_ctrl_pts=20, xlimits=xlimits, nonlinear_maxiter=100,
                 energy_weight=1e-12, print_global=False)
        m.set_training_values(xt, yt[:, i])
        m.train()
        models[name] = m

    for i, (name, m) in enumerate(models.items()):
        plot_model(m, xt, yt[:, i], label=name)
    plt.show()
