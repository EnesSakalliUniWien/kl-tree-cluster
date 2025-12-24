"Batch helpers to render and save plots from benchmark results."

from __future__ import annotations

from pathlib import Path
import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from kl_clustering_analysis.plot.cluster_tree_visualization import (
    plot_tree_with_clusters,
)
from .embedding import (
    create_clustering_comparison_plot,
    create_clustering_comparison_plot_3d,
)
from .manifold import create_manifold_alignment_plot

logger = logging.getLogger(__name__)


def _format_params_for_display(params: dict) -> str:
    """Creates a human-readable string from a parameter dictionary."""
    if not params:
        return ""
    return ", ".join(f"{k}={v}" for k, v in sorted(params.items()))


def _format_params_for_filename(params: dict) -> str:
    """Creates a filesystem-safe, consistent string from a parameter dictionary."""
    if not params:
        return ""
    # Sort items for consistency
    items = sorted(params.items())
    # Format as key-value pairs, sanitize, and join
    # e.g., {'metric': 'euclidean', 'res': 1.0} -> "metric-euclidean_res-1p0"
    return "_".join(f"{k}-{str(v).replace('.', 'p')}" for k, v in items)


def create_umap_plots_from_results(
    test_results: list,
    output_dir: Path,
    timestamp: str | None = None,
    verbose: bool = True,
    save: bool = True,
    collect: bool = False,
    collected: list | None = None,
    *,
    pdf: PdfPages | None = None,
) -> list:
    if save:
        output_dir.mkdir(exist_ok=True)
    figs: list = collected if collected is not None else []

    results_by_case = {}
    for result in test_results:
        case_num = result["test_case_num"]
        results_by_case.setdefault(case_num, []).append(result)

    for case_num, case_results in results_by_case.items():
        if not case_results:
            continue

        first_result = case_results[0]
        meta = first_result["meta"]
        if verbose:
            print(f"  Creating UMAP comparison plot for test case {case_num}...")

        labels_to_plot = {"Ground Truth": first_result.get("y_true")}
        for res in case_results:
            method_name = res["method_name"]
            params = res.get("params", {})
            param_str = _format_params_for_display(params)

            unique_key = f"{method_name} ({param_str})" if param_str else method_name
            if res.get("labels") is not None:
                labels_to_plot[unique_key] = res["labels"]

        try:
            fig = create_clustering_comparison_plot(
                X_original=first_result["X_original"],
                labels_dict=labels_to_plot,
                test_case_num=case_num,
                meta=meta,
            )
            if pdf is not None:
                pdf.savefig(fig, bbox_inches="tight", pad_inches=0)
                plt.close(fig)
            elif save:
                filename = (
                    f"umap_comparison_case_{case_num}_{timestamp}.png"
                    if timestamp
                    else f"umap_comparison_case_{case_num}.png"
                )
                fig.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
                plt.close(fig)
            elif collect:
                figs.append({"figure": fig, "test_case_num": case_num})
            else:
                plt.close(fig)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Skipping UMAP plot for test %s: %s", case_num, exc)
        finally:
            if not collect:
                plt.close("all")

    return figs


def create_umap_3d_plots_from_results(
    test_results: list,
    output_dir: Path,
    timestamp: str | None = None,
    verbose: bool = True,
    save: bool = True,
    collect: bool = False,
    collected: list | None = None,
    *,
    pdf: PdfPages | None = None,
) -> list:
    figs: list = collected if collected is not None else []
    if save:
        output_dir.mkdir(exist_ok=True)

    for result in test_results:
        if result.get("labels") is None:
            continue

        i = result["test_case_num"]
        method_name = result["method_name"]
        params = result.get("params", {})
        param_str_display = _format_params_for_display(params)

        method_name_safe = method_name.replace(" ", "_")
        param_str_safe = _format_params_for_filename(params)
        filename = (
            f"umap3d_case_{i}_{method_name_safe}_{param_str_safe}_{timestamp}.png"
            if timestamp
            else f"umap3d_case_{i}_{method_name_safe}_{param_str_safe}.png"
        )

        title = f"3D UMAP - {method_name} ({param_str_display})\nTest Case {i}"

        if verbose:
            print(f"  Creating 3D UMAP plot for {filename}...")
        try:
            fig = create_clustering_comparison_plot_3d(
                result["X_original"],
                result.get("y_true"),
                np.asarray(result["labels"]),
                test_case_num=i,
                meta=result["meta"],
                title=title,
            )
            if pdf is not None:
                pdf.savefig(fig, bbox_inches="tight", pad_inches=0)
                plt.close(fig)
            elif save:
                fig.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
                plt.close(fig)
            elif collect:
                figs.append({"figure": fig, "test_case_num": i})
            else:
                plt.close(fig)
        except Exception as exc:
            logger.warning("Skipping 3D UMAP plot for test %s: %s", i, exc)
        finally:
            if not collect:
                plt.close("all")
    return figs


def create_manifold_plots_from_results(
    test_results: list,
    output_dir: Path,
    timestamp: str | None = None,
    verbose: bool = True,
    save: bool = True,
    collect: bool = False,
    collected: list | None = None,
    *,
    pdf: PdfPages | None = None,
) -> list:
    figs: list = collected if collected is not None else []
    if save:
        output_dir.mkdir(exist_ok=True)
    for result in test_results:
        if result.get("labels") is None:
            continue

        i = result["test_case_num"]
        method_name = result["method_name"]
        params = result.get("params", {})
        param_str_display = _format_params_for_display(params)

        method_name_safe = method_name.replace(" ", "_")
        param_str_safe = _format_params_for_filename(params)
        filename = (
            f"manifold_case_{i}_{method_name_safe}_{param_str_safe}_{timestamp}.png"
            if timestamp
            else f"manifold_case_{i}_{method_name_safe}_{param_str_safe}.png"
        )

        title = (
            f"Manifold Alignment - {method_name} ({param_str_display})\nTest Case {i}"
        )

        if verbose:
            print(f"  Creating manifold plot for {filename}...")

        try:
            fig, mantel_r, mantel_p = create_manifold_alignment_plot(
                result["X_original"],
                np.asarray(result["labels"]),
                test_case_num=i,
                meta=result["meta"],
                y_true=result.get("y_true"),
                title=title,
            )
            if pdf is not None:
                pdf.savefig(fig, bbox_inches="tight", pad_inches=0)
                plt.close(fig)
            elif save:
                fig.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
                if verbose:
                    print(
                        f"    Saved manifold diagnostics (r={mantel_r:.2f}, p={mantel_p:.3f})"
                    )
                plt.close(fig)
            elif collect:
                figs.append({"figure": fig, "test_case_num": i})
            else:
                plt.close(fig)
        except Exception as exc:
            logger.warning("Skipping manifold plot for test %s: %s", i, exc)
        finally:
            if not collect:
                plt.close("all")
    return figs


def create_tree_plots_from_results(
    test_results: list,
    output_dir: Path,
    timestamp: str | None = None,
    verbose: bool = True,
    save: bool = True,
    collect: bool = False,
    collected: list | None = None,
    *,
    pdf: PdfPages | None = None,
) -> list:
    figs: list = collected if collected is not None else []
    if save:
        output_dir.mkdir(exist_ok=True)

    # Group tree-capable runs by test case so we can render a grid per case.
    results_by_case: dict[int, list[dict]] = {}
    for result in test_results:
        if not result.get("tree") or not result.get("decomposition"):
            continue
        case_num = int(result.get("test_case_num", 0))
        results_by_case.setdefault(case_num, []).append(result)

    for case_num, case_results in sorted(results_by_case.items()):
        if not case_results:
            continue

        # Layout: similar to UMAP grids, but default to 2 columns for readability.
        n_items = len(case_results)
        n_cols = 2 if n_items > 1 else 1
        n_rows = (n_items + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12 * n_cols, 7 * n_rows))
        axes = np.atleast_1d(axes).ravel()

        meta = case_results[0].get("meta", {})
        meta_text = (
            f"samples={meta.get('n_samples', '?')}, "
            f"features={meta.get('n_features', '?')}, "
            f"generator={meta.get('generator', 'unknown')}, "
            f"noise={meta.get('noise', 'n/a')}"
        )
        fig.suptitle(
            f"Tree Comparisons – Test Case {case_num}\n{meta_text}",
            fontsize=16,
            weight="bold",
            y=0.98,
        )

        for idx, result in enumerate(case_results):
            ax = axes[idx]
            tree_t = result["tree"]
            decomp_t = result["decomposition"]
            method_name = result.get("method_name", "KL Divergence")
            params = result.get("params", {})
            param_str_display = _format_params_for_display(params)
            found_clusters = result.get("meta", {}).get("found_clusters", "?")

            title = (
                f"{method_name}\n({param_str_display})\nfound={found_clusters}"
                if param_str_display
                else f"{method_name}\nfound={found_clusters}"
            )

            plot_tree_with_clusters(
                tree=tree_t,
                decomposition_results=decomp_t,
                results_df=getattr(tree_t, "stats_df", None),
                use_labels=True,
                node_size=12,
                font_size=9,
                title=title,
                ax=ax,
                show=False,
            )

        for j in range(n_items, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()

        filename = (
            f"tree_case_{case_num}_grid_{timestamp}.png"
            if timestamp
            else f"tree_case_{case_num}_grid.png"
        )
        if pdf is not None:
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
        elif save:
            if verbose:
                print(f"  Creating tree grid plot for {filename}...")
            fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
            plt.close(fig)
        elif collect:
            figs.append({"figure": fig, "test_case_num": case_num})
        else:
            plt.close(fig)

    return figs


__all__ = [
    "create_umap_plots_from_results",
    "create_umap_3d_plots_from_results",
    "create_manifold_plots_from_results",
    "create_tree_plots_from_results",
]
