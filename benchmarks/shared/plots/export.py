"Batch helpers to render and save plots from benchmark results."

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from benchmarks.shared.pdf_utils import PDF_PAGE_SIZE_INCHES, prepare_pdf_figure
from kl_clustering_analysis.plot.cluster_tree_visualization import plot_tree_with_clusters

from .embedding import (
    create_clustering_comparison_plot_3d,
    create_clustering_comparison_plots,
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
            umap_figs = create_clustering_comparison_plots(
                X_original=first_result["X_original"],
                labels_dict=labels_to_plot,
                test_case_num=case_num,
                meta=meta,
            )
            n_pages = len(umap_figs)
            for page_idx, fig in enumerate(umap_figs, start=1):
                page_suffix = f"_p{page_idx}" if n_pages > 1 else ""
                if pdf is not None:
                    prepare_pdf_figure(fig)
                    pdf.savefig(fig)
                    plt.close(fig)
                elif save:
                    filename = (
                        f"umap_comparison_case_{case_num}{page_suffix}_{timestamp}.png"
                        if timestamp
                        else f"umap_comparison_case_{case_num}{page_suffix}.png"
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


def _group_results_by_case(test_results: list) -> dict[int, list[dict]]:
    results_by_case: dict[int, list[dict]] = {}
    for result in test_results:
        case_num = int(result.get("test_case_num", 0))
        results_by_case.setdefault(case_num, []).append(result)
    return results_by_case


def _save_or_collect_figure(
    fig: plt.Figure,
    *,
    pdf: PdfPages | None,
    save: bool,
    collect: bool,
    figs: list,
    output_path: Path | None,
    test_case_num: int,
) -> None:
    if pdf is not None:
        prepare_pdf_figure(fig)
        pdf.savefig(fig)
        plt.close(fig)
        return
    if save and output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return
    if collect:
        figs.append({"figure": fig, "test_case_num": test_case_num})
        return
    plt.close(fig)


def _create_tree_figures_for_case(
    *,
    case_num: int,
    case_results: list[dict],
) -> list[plt.Figure]:
    # Keep only tree-capable runs.
    tree_results = [
        r for r in case_results if r.get("tree") is not None and r.get("decomposition") is not None
    ]
    if not tree_results:
        return []

    meta = tree_results[0].get("meta", {})
    meta_text = (
        f"samples={meta.get('n_samples', '?')}, "
        f"features={meta.get('n_features', '?')}, "
        f"generator={meta.get('generator', 'unknown')}, "
        f"noise={meta.get('noise', 'n/a')}"
    )

    figs: list[plt.Figure] = []
    n_items = len(tree_results)

    for idx, result in enumerate(tree_results, start=1):
        fig, ax = plt.subplots(1, 1, figsize=PDF_PAGE_SIZE_INCHES)
        fig.suptitle(
            f"Tree Comparisons – Test Case {case_num}\n{meta_text}",
            fontsize=16,
            weight="bold",
            y=0.97,
        )

        tree_t = result["tree"]
        decomp_t = result["decomposition"]
        method_name = result.get("method_name", "KL Divergence")
        params = result.get("params", {})
        param_str_display = _format_params_for_display(params)
        found_clusters = result.get("meta", {}).get("found_clusters", "?")

        page_tag = f" ({idx}/{n_items})" if n_items > 1 else ""
        title = (
            f"{method_name}{page_tag}\n({param_str_display})\nfound={found_clusters}"
            if param_str_display
            else f"{method_name}{page_tag}\nfound={found_clusters}"
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
        fig.subplots_adjust(top=0.84, bottom=0.06, left=0.03, right=0.97)
        figs.append(fig)

    return figs


def create_umap_then_tree_plots_from_results(
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
    """For each test case: render UMAP grid, then the matching tree grid (if available)."""
    if save:
        output_dir.mkdir(exist_ok=True)
    figs: list = collected if collected is not None else []

    results_by_case = _group_results_by_case(test_results)

    for case_num, case_results in sorted(results_by_case.items()):
        if not case_results:
            continue

        first_result = case_results[0]
        meta = first_result.get("meta", {})
        if verbose:
            print(f"  Creating UMAP→Tree plots for test case {case_num}...")

        labels_to_plot = {"Ground Truth": first_result.get("y_true")}
        for res in case_results:
            method_name = res.get("method_name", "")
            params = res.get("params", {})
            param_str = _format_params_for_display(params)
            unique_key = f"{method_name} ({param_str})" if param_str else method_name
            if res.get("labels") is not None:
                labels_to_plot[unique_key] = res["labels"]

        try:
            umap_figs = create_clustering_comparison_plots(
                X_original=first_result["X_original"],
                labels_dict=labels_to_plot,
                test_case_num=case_num,
                meta=meta,
            )
            n_pages = len(umap_figs)
            for page_idx, umap_fig in enumerate(umap_figs, start=1):
                page_suffix = f"_p{page_idx}" if n_pages > 1 else ""
                umap_filename = (
                    f"umap_comparison_case_{case_num}{page_suffix}_{timestamp}.png"
                    if timestamp
                    else f"umap_comparison_case_{case_num}{page_suffix}.png"
                )
                _save_or_collect_figure(
                    umap_fig,
                    pdf=pdf,
                    save=save,
                    collect=collect,
                    figs=figs,
                    output_path=(output_dir / umap_filename) if save else None,
                    test_case_num=case_num,
                )
        except Exception as exc:  # pragma: no cover
            logger.warning("Skipping UMAP plot for test %s: %s", case_num, exc)
        finally:
            if not collect:
                plt.close("all")

        # Tree pages for same case (optional).
        try:
            tree_figs = _create_tree_figures_for_case(case_num=case_num, case_results=case_results)
            if not tree_figs:
                continue
            for tree_idx, tree_fig in enumerate(tree_figs, start=1):
                tree_filename = (
                    f"tree_case_{case_num}_{tree_idx}_{timestamp}.png"
                    if timestamp
                    else f"tree_case_{case_num}_{tree_idx}.png"
                )
                _save_or_collect_figure(
                    tree_fig,
                    pdf=pdf,
                    save=save,
                    collect=collect,
                    figs=figs,
                    output_path=(output_dir / tree_filename) if save else None,
                    test_case_num=case_num,
                )
        except Exception as exc:  # pragma: no cover
            logger.warning("Skipping tree plot for test %s: %s", case_num, exc)
        finally:
            if not collect:
                plt.close("all")

    return figs


def create_tree_then_umap_plots_from_results(
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
    """For each test case: render tree grid (if available), then the matching UMAP grid."""
    if save:
        output_dir.mkdir(exist_ok=True)
    figs: list = collected if collected is not None else []

    results_by_case = _group_results_by_case(test_results)

    for case_num, case_results in sorted(results_by_case.items()):
        if not case_results:
            continue

        first_result = case_results[0]
        meta = first_result.get("meta", {})
        if verbose:
            print(f"  Creating Tree→UMAP plots for test case {case_num}...")

        # Tree pages first (optional).
        try:
            tree_figs = _create_tree_figures_for_case(case_num=case_num, case_results=case_results)
            for tree_idx, tree_fig in enumerate(tree_figs, start=1):
                tree_filename = (
                    f"tree_case_{case_num}_{tree_idx}_{timestamp}.png"
                    if timestamp
                    else f"tree_case_{case_num}_{tree_idx}.png"
                )
                _save_or_collect_figure(
                    tree_fig,
                    pdf=pdf,
                    save=save,
                    collect=collect,
                    figs=figs,
                    output_path=(output_dir / tree_filename) if save else None,
                    test_case_num=case_num,
                )
        except Exception as exc:  # pragma: no cover
            logger.warning("Skipping tree plot for test %s: %s", case_num, exc)
        finally:
            if not collect:
                plt.close("all")

        # UMAP grid second (always attempted).
        labels_to_plot = {"Ground Truth": first_result.get("y_true")}
        for res in case_results:
            method_name = res.get("method_name", "")
            params = res.get("params", {})
            param_str = _format_params_for_display(params)
            unique_key = f"{method_name} ({param_str})" if param_str else method_name
            if res.get("labels") is not None:
                labels_to_plot[unique_key] = res["labels"]

        try:
            umap_figs = create_clustering_comparison_plots(
                X_original=first_result["X_original"],
                labels_dict=labels_to_plot,
                test_case_num=case_num,
                meta=meta,
            )
            n_pages = len(umap_figs)
            for page_idx, umap_fig in enumerate(umap_figs, start=1):
                page_suffix = f"_p{page_idx}" if n_pages > 1 else ""
                umap_filename = (
                    f"umap_comparison_case_{case_num}{page_suffix}_{timestamp}.png"
                    if timestamp
                    else f"umap_comparison_case_{case_num}{page_suffix}.png"
                )
                _save_or_collect_figure(
                    umap_fig,
                    pdf=pdf,
                    save=save,
                    collect=collect,
                    figs=figs,
                    output_path=(output_dir / umap_filename) if save else None,
                    test_case_num=case_num,
                )
        except Exception as exc:  # pragma: no cover
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
                prepare_pdf_figure(fig)
                pdf.savefig(fig)
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

        title = f"Manifold Alignment - {method_name} ({param_str_display})\nTest Case {i}"

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
                prepare_pdf_figure(fig)
                pdf.savefig(fig)
                plt.close(fig)
            elif save:
                fig.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
                if verbose:
                    print(f"    Saved manifold diagnostics (r={mantel_r:.2f}, p={mantel_p:.3f})")
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

    # Group tree-capable runs by test case so we can render one tree per page.
    results_by_case: dict[int, list[dict]] = {}
    for result in test_results:
        if not result.get("tree") or not result.get("decomposition"):
            continue
        case_num = int(result.get("test_case_num", 0))
        results_by_case.setdefault(case_num, []).append(result)

    for case_num, case_results in sorted(results_by_case.items()):
        tree_figs = _create_tree_figures_for_case(case_num=case_num, case_results=case_results)
        if not tree_figs:
            continue

        for tree_idx, fig in enumerate(tree_figs, start=1):
            filename = (
                f"tree_case_{case_num}_{tree_idx}_{timestamp}.png"
                if timestamp
                else f"tree_case_{case_num}_{tree_idx}.png"
            )
            if pdf is not None:
                prepare_pdf_figure(fig)
                pdf.savefig(fig)
                plt.close(fig)
            elif save:
                if verbose:
                    print(f"  Creating tree plot for {filename}...")
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
    "create_umap_then_tree_plots_from_results",
    "create_tree_then_umap_plots_from_results",
]
