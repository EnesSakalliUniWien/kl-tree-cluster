"""Summary plots for validation metrics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def create_validation_plot(df_results):
    """Create a compact summary plot for validation metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    if df_results.empty:
        for ax in axes:
            ax.axis("off")
        fig.suptitle("Validation Results (empty)")
        return fig

    statuses = df_results["status"].fillna("").astype(str)
    status_text = (
        f"ok={int(statuses.eq('ok').sum())}, "
        f"unsupported={int(statuses.eq('unsupported').sum())}, "
        f"skip={int(statuses.eq('skip').sum())}"
    )
    successful = df_results.loc[statuses.eq("ok")].copy()
    if successful.empty:
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"Cluster Validation Summary\nSupport: {status_text}")
        return fig

    x = np.arange(len(successful))
    x_labels = [str(m) for m in successful["method"].tolist()]
    x_title = "method"

    axes[0].plot(x, successful["true_clusters"], label="True", marker="o")
    axes[0].plot(x, successful["found_clusters"], label="Found", marker="o")
    axes[0].set_title("Clusters: True vs Found")
    axes[0].set_xlabel(x_title)
    axes[0].set_ylabel("Count")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(frameon=False)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8)

    axes[1].plot(x, successful["ari"], marker="o")
    axes[1].set_title("ARI")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8)

    axes[2].plot(x, successful["nmi"], marker="o")
    axes[2].set_title("NMI")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8)

    axes[3].plot(x, successful["purity"], marker="o")
    axes[3].set_title("Purity (Homogeneity)")
    axes[3].set_ylim(-0.05, 1.05)
    axes[3].grid(True, alpha=0.3)
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8)

    fig.suptitle(
        f"Cluster Validation Summary\nSupport: {status_text}",
        fontsize=14,
        weight="bold",
    )
    plt.tight_layout(rect=(0.02, 0.04, 0.98, 0.96))
    return fig
