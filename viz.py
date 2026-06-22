"""Visualization utilities for Meow model evaluation."""

import numpy as np
import pandas as pd

# Lazy imports — matplotlib is heavy and may not be available on headless servers.
_mpl_available = None


def _check_matplotlib():
    global _mpl_available
    if _mpl_available is None:
        try:
            import matplotlib
            matplotlib.use("Agg")  # non-interactive backend
            import matplotlib.pyplot as plt
            _mpl_available = True
        except ImportError:
            _mpl_available = False
    return _mpl_available


def prediction_scatter(y_true, y_pred, save_path="plots/pred_scatter.png",
                       title="Predicted vs Actual Returns"):
    """Scatter plot of predicted vs actual values with correlation annotation."""
    if not _check_matplotlib():
        return
    import matplotlib.pyplot as plt

    y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

    if len(y_true) < 10:
        return

    r = float(np.corrcoef(y_true, y_pred)[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hexbin(y_true, y_pred, gridsize=50, cmap="Blues", mincnt=1, bins="log")
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
            "r--", lw=1, label="y = x")
    ax.set_xlabel("Actual Return")
    ax.set_ylabel("Predicted Return")
    ax.set_title(title)
    ax.legend()
    ax.text(0.05, 0.95, f"Pearson r = {r:.4f}\nN = {len(y_true):,}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    from log import log
    log.inf("Scatter plot saved to {}".format(save_path))
    plt.close(fig)


def residual_distribution(y_true, y_pred, save_path="plots/residual_dist.png",
                          title="Residual Distribution"):
    """Histogram of prediction residuals."""
    if not _check_matplotlib():
        return
    import matplotlib.pyplot as plt

    y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    residuals = y_pred[mask] - y_true[mask]

    if len(residuals) < 10:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residuals, bins=100, color="steelblue", edgecolor="white",
            alpha=0.8, density=True)
    ax.axvline(0, color="red", lw=1, linestyle="--")
    ax.set_xlabel("Residual (pred − actual)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.text(0.05, 0.95,
            f"Mean = {residuals.mean():.6f}\nStd = {residuals.std():.6f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    from log import log
    log.inf("Residual plot saved to {}".format(save_path))
    plt.close(fig)


def cumulative_return_by_quantile(y_true, y_pred, save_path="plots/cumret_quantile.png",
                                  n_quantiles=5, title="Cumulative Return by Prediction Quantile"):
    """Group predictions into quantiles and plot cumulative returns of each group."""
    if not _check_matplotlib():
        return
    import matplotlib.pyplot as plt

    y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

    if len(y_true) < n_quantiles * 2:
        return

    quantiles = pd.qcut(y_pred, n_quantiles, labels=False, duplicates="drop")
    n_actual = len(np.unique(quantiles))

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, n_actual))
    for q in range(n_actual):
        idx = quantiles == q
        cumret = np.cumsum(y_true[idx])
        label = f"Q{q + 1} (n={idx.sum():,})"
        ax.plot(cumret, label=label, color=colors[q], lw=0.8, alpha=0.85)

    ax.set_xlabel("Sample Index (within quantile)")
    ax.set_ylabel("Cumulative Return")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    ax.axhline(0, color="black", lw=0.5, linestyle="--")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    from log import log
    log.inf("Cumulative return plot saved to {}".format(save_path))
    plt.close(fig)


def feature_gate_importance(gate_values, feature_names, save_path="plots/gate_importance.png",
                            top_n=20):
    """Bar chart of learned feature gate values (feature importance)."""
    if not _check_matplotlib():
        return
    import matplotlib.pyplot as plt

    gates = np.asarray(gate_values).ravel()
    if len(gates) != len(feature_names):
        feature_names = [f"feat_{i}" for i in range(len(gates))]

    idx = np.argsort(gates)[-top_n:][::-1]
    names = [feature_names[i] for i in idx]
    values = gates[idx]

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = plt.cm.Blues(0.3 + 0.7 * (values - values.min()) /
                          (values.max() - values.min() + 1e-8))
    ax.barh(range(len(names)), values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Gate Value (sigmoid)")
    ax.set_title("Top {} Features by Learned Gate Weight".format(top_n))
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    from log import log
    log.inf("Feature gate plot saved to {}".format(save_path))
    plt.close(fig)


def make_all_plots(y_true, y_pred, gate_values=None, feature_names=None,
                   output_dir="plots"):
    """Generate all evaluation plots."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    prediction_scatter(
        y_true, y_pred,
        save_path=os.path.join(output_dir, "pred_scatter.png"))
    residual_distribution(
        y_true, y_pred,
        save_path=os.path.join(output_dir, "residual_dist.png"))
    cumulative_return_by_quantile(
        y_true, y_pred,
        save_path=os.path.join(output_dir, "cumret_quantile.png"))
    if gate_values is not None and feature_names is not None:
        feature_gate_importance(
            gate_values, feature_names,
            save_path=os.path.join(output_dir, "gate_importance.png"))
