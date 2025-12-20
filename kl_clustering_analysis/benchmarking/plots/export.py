"""Batch helpers to render and save plots from benchmark results."""

from __future__ import annotations

from pathlib import Path
import logging

import numpy as np
import matplotlib.pyplot as plt

from kl_clustering_analysis.plot.cluster_tree_visualization import (
    plot_tree_with_clusters,
)
from .embedding import (
    create_clustering_comparison_plot,
    create_clustering_comparison_plot_3d,
)
from .manifold import create_manifold_alignment_plot

logger = logging.getLogger(__name__)


def create_umap_plots_from_results(
    test_results: list,
    output_dir: Path,
    timestamp: str,
    verbose: bool = True,
) -> None:
    output_dir.mkdir(exist_ok=True)

    results_by_case = {}
    for result in test_results:
        case_num = result["test_case_num"]
        if case_num not in results_by_case:
            results_by_case[case_num] = []
        results_by_case[case_num].append(result)

    for case_num, case_results in results_by_case.items():
        if not case_results:
            continue

        first_result = case_results[0]
        meta = first_result["meta"]
        if verbose:
            print(f"  Creating UMAP comparison plot for test case {case_num}...")

        labels_to_plot = {"Ground Truth": first_result.get("y_true")}
        for res in case_results:
            labels_to_plot[res["method_name"]] = res["labels"]

        try:
            fig = create_clustering_comparison_plot(
                X_original=first_result["X_original"],
                labels_dict=labels_to_plot,
                test_case_num=case_num,
                meta=meta,
            )
            fig.savefig(
                output_dir / f"umap_test_{case_num}_{timestamp}.png",
                dpi=200,
                bbox_inches="tight",
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Skipping UMAP plot for test %s: %s", case_num, exc)
        finally:
            plt.close("all")


def create_umap_3d_plots_from_results(
    test_results: list,
    output_dir: Path,
    timestamp: str,
    verbose: bool = True,
) -> None:
    output_dir.mkdir(exist_ok=True)
    for result in test_results:
        i = result["test_case_num"]
        meta = result["meta"]
        found_clusters = meta.get("found_clusters", meta["n_clusters"])
        if verbose:
            print(f"  Creating 3D UMAP comparison plot for test case {i}...")
        try:
            fig = create_clustering_comparison_plot_3d(
                result["X_original"],
                result.get("y_true"),
                np.asarray(result["kl_labels"]),
                test_case_num=i,
                meta=meta,
            )
            fig.savefig(
                output_dir
                / f"umap3d_test_{i}_{found_clusters}_clusters_{timestamp}.png",
                dpi=200,
                bbox_inches="tight",
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Skipping 3D UMAP plot for test %s: %s", i, exc)
        finally:
            plt.close("all")


def create_manifold_plots_from_results(
    test_results: list,
    output_dir: Path,
    timestamp: str,
    verbose: bool = True,
) -> None:
    output_dir.mkdir(exist_ok=True)
    for result in test_results:
        i = result["test_case_num"]
        meta = result["meta"]
        found_clusters = meta.get("found_clusters", meta["n_clusters"])
        if verbose:
            print(f"  Creating manifold plot for test case {i}...")
        try:
            fig, mantel_r, mantel_p = create_manifold_alignment_plot(
                result["X_original"],
                np.asarray(result["kl_labels"]),
                test_case_num=i,
                meta=meta,
                y_true=result.get("y_true"),
            )
            fig.savefig(
                output_dir
                / f"manifold_test_{i}_{found_clusters}_clusters_{timestamp}.png",
                dpi=200,
                bbox_inches="tight",
            )
            if verbose:
                print(
                    f"    Saved manifold diagnostics (r={mantel_r:.2f}, p={mantel_p:.3f})"
                )
        except Exception as exc:  # pragma: no cover
            logger.warning("Skipping manifold plot for test %s: %s", i, exc)
        finally:
            plt.close("all")


def create_tree_plots_from_results(
    test_results: list,
    output_dir: Path,
    timestamp: str,
    verbose: bool = True,
) -> None:
    output_dir.mkdir(exist_ok=True)
    for result in test_results:
        tree_t = result["tree"]
        decomp_t = result["decomposition"]
        meta = result["meta"]
        found_clusters = meta.get("found_clusters", meta["n_clusters"])
        i = result["test_case_num"]
        if verbose:
            print(f"  Creating tree plot for test case {i}...")
        fig, _ax = plot_tree_with_clusters(
            tree=tree_t,
            decomposition_results=decomp_t,
            results_df=getattr(tree_t, "stats_df", None),
            use_labels=True,
            width=1000,
            height=700,
            node_size=20,
            font_size=9,
            title=f"Hierarchical Tree with KL Divergence Clusters\nTest Case {i}",
            show=False,
        )
        fig.savefig(
            output_dir / f"tree_test_{i}_{found_clusters}_clusters_{timestamp}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)


__all__ = [
    "create_umap_plots_from_results",
    "create_umap_3d_plots_from_results",
    "create_manifold_plots_from_results",
    "create_tree_plots_from_results",
]
