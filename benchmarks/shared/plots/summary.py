"""Summary plots for validation metrics."""

from __future__ import annotations

import matplotlib.pyplot as plt


def create_validation_plot(df_results):
    """Create a compact summary plot for validation metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    if df_results.empty:
        for ax in axes:
            ax.axis("off")
        fig.suptitle("Validation Results (empty)")
        return fig

    axes[0].plot(df_results["True"], label="True", marker="o")
    axes[0].plot(df_results["Found"], label="Found", marker="o")
    axes[0].set_title("Clusters: True vs Found")
    axes[0].set_xlabel("Test case")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(frameon=False)

    axes[1].plot(df_results["ARI"], marker="o")
    axes[1].set_title("ARI")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(df_results["NMI"], marker="o")
    axes[2].set_title("NMI")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(df_results["Purity"], marker="o")
    axes[3].set_title("Purity (Homogeneity)")
    axes[3].set_ylim(-0.05, 1.05)
    axes[3].grid(True, alpha=0.3)

    fig.suptitle("Cluster Validation Summary", fontsize=14, weight="bold")
    plt.tight_layout()
    return fig
