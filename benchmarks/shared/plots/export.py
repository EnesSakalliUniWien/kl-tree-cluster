"Batch helpers to render and save plots from benchmark results."

from __future__ import annotations

import logging
import textwrap
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from kl_clustering_analysis.plot.cluster_tree_visualization import plot_tree_with_clusters
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages

from benchmarks.shared.result_records import ComputedResultRecord
from benchmarks.shared.util.params import format_params_for_display
from benchmarks.shared.util.pdf.layout import (
    PDF_PAGE_SIZE_INCHES,
    PDF_WIDE_PAGE_SIZE_INCHES,
    prepare_pdf_figure,
    set_pdf_page_size,
)

from .embedding import (
    create_clustering_comparison_plot_3d,
    create_clustering_comparison_plots,
)

logger = logging.getLogger(__name__)


def _format_ari_nmi(result: ComputedResultRecord, *, compact: bool = False) -> str:
    """Format primary benchmark metrics when available on a computed result record."""
    ari = result.ari
    nmi = result.nmi
    parts: list[str] = []
    found_clusters = result.meta.get("found_clusters")
    if isinstance(found_clusters, (int, float, np.integer, np.floating)) and np.isfinite(
        float(found_clusters)
    ):
        parts.append(
            f"K={int(found_clusters)}" if compact else f"Found_K={int(found_clusters)}"
        )
    if isinstance(ari, (int, float, np.floating)) and np.isfinite(float(ari)):
        parts.append(f"A={float(ari):.3f}" if compact else f"ARI={float(ari):.3f}")
    if isinstance(nmi, (int, float, np.floating)) and np.isfinite(float(nmi)):
        parts.append(f"N={float(nmi):.3f}" if compact else f"NMI={float(nmi):.3f}")
    outlier_f1 = result.outlier_f1
    singleton_hit = result.singleton_outlier_isolated
    grouped_recovery = result.grouped_outlier_cluster_recovered
    if isinstance(outlier_f1, (int, float, np.floating)) and np.isfinite(float(outlier_f1)):
        parts.append(f"OF1={float(outlier_f1):.3f}" if compact else f"Outlier_F1={float(outlier_f1):.3f}")
    if isinstance(singleton_hit, (int, float, np.floating)) and np.isfinite(float(singleton_hit)):
        parts.append(f"SOI={float(singleton_hit):.0f}" if compact else f"Singleton_Isolated={float(singleton_hit):.0f}")
    if isinstance(grouped_recovery, (int, float, np.floating)) and np.isfinite(float(grouped_recovery)):
        parts.append(f"GOR={float(grouped_recovery):.0f}" if compact else f"Grouped_Outlier_Recovered={float(grouped_recovery):.0f}")
    return ", ".join(parts)


def _format_params_for_filename(params: dict) -> str:
    """Creates a filesystem-safe, consistent string from a parameter dictionary."""
    if not params:
        return ""
    # Sort items for consistency
    items = sorted(params.items())
    # Format as key-value pairs, sanitize, and join
    # e.g., {'metric': 'euclidean', 'res': 1.0} -> "metric-euclidean_res-1p0"
    return "_".join(f"{k}-{str(v).replace('.', 'p')}" for k, v in items)


def _shorten_title_line(text: str, width: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= width:
        return text
    return textwrap.shorten(text, width=width, placeholder="...")


def _wrap_title_line(text: str, width: int) -> list[str]:
    return textwrap.wrap(
        _shorten_title_line(text, width * 2),
        width=width,
        break_long_words=True,
        break_on_hyphens=False,
    )


def _tree_plot_style(tree, decomposition: dict) -> dict[str, object]:
    """Return size and legend defaults scaled for tree density."""
    node_count = len(list(tree.nodes()))
    num_clusters = int(decomposition.get("num_clusters", 0))
    dense = node_count > 80 or num_clusters > 20
    return {
        "figsize": PDF_WIDE_PAGE_SIZE_INCHES if dense else PDF_PAGE_SIZE_INCHES,
        "node_size": 5 if dense else 12,
        "font_size": 7 if dense else 9,
        "max_cluster_legend_entries": 0 if num_clusters > 20 else 20,
        "subplots_right": 0.82 if num_clusters <= 20 else 0.88,
    }


def create_umap_plots_from_results(
    test_results: list[ComputedResultRecord],
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
        case_num = result.test_case_num
        results_by_case.setdefault(case_num, []).append(result)

    for case_num, case_results in results_by_case.items():
        if not case_results:
            continue

        first_result = case_results[0]
        meta = first_result.meta
        if verbose:
            print(f"  Creating UMAP comparison plot for test case {case_num}...")

        labels_to_plot = {"Ground Truth": first_result.y_true}
        for res in case_results:
            method_name = res.method_name
            params = res.params
            param_str = format_params_for_display(params)

            unique_key = f"{method_name} ({param_str})" if param_str else method_name
            metrics_text = _format_ari_nmi(res, compact=True)
            if metrics_text:
                unique_key = f"{unique_key} [{metrics_text}]"
            if res.labels is not None:
                labels_to_plot[unique_key] = res.labels

        _cache_key = f"{meta['name']}_n{meta['n_samples']}"

        umap_figs = create_clustering_comparison_plots(
            X_original=first_result.x_original,
            labels_dict=labels_to_plot,
            test_case_num=case_num,
            meta=meta,
            cache_key=_cache_key,
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
        if not collect:
            plt.close("all")

    return figs


def _group_results_by_case(
    test_results: list[ComputedResultRecord],
) -> dict[int, list[ComputedResultRecord]]:
    results_by_case: dict[int, list[ComputedResultRecord]] = {}
    for result in test_results:
        case_num = int(result.test_case_num)
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
    case_results: list[ComputedResultRecord],
) -> list[plt.Figure]:
    # Keep only tree-capable runs.
    tree_results = [r for r in case_results if r.tree is not None and r.decomposition is not None]
    if not tree_results:
        return []

    meta = tree_results[0].meta
    meta_text = (
        f"samples={meta['n_samples']}, "
        f"features={meta['n_features']}, "
        f"generator={meta['generator']}, "
        f"noise={meta['noise']}"
    )

    figs: list[plt.Figure] = []
    n_items = len(tree_results)

    for idx, result in enumerate(tree_results, start=1):
        plot_style = _tree_plot_style(result.tree, result.decomposition)
        fig, ax = plt.subplots(1, 1, figsize=plot_style["figsize"])
        set_pdf_page_size(fig, plot_style["figsize"])
        fig.suptitle(
            f"Tree Comparisons – Test Case {case_num}\n{meta_text}",
            fontsize=16,
            weight="bold",
            y=0.97,
        )

        tree_t = result.tree
        decomp_t = result.decomposition
        annotations_df = result.annotations
        method_name = result.method_name
        params = result.params
        param_str_display = format_params_for_display(params)
        metrics_text = _format_ari_nmi(result)

        page_tag = f" ({idx}/{n_items})" if n_items > 1 else ""
        title_lines = _wrap_title_line(f"{method_name}{page_tag}", 72)
        if param_str_display:
            title_lines.extend(_wrap_title_line(f"({param_str_display})", 72))
        if metrics_text:
            title_lines.extend(_wrap_title_line(metrics_text, 72))
        title = "\n".join(title_lines)

        plot_tree_with_clusters(
            tree=tree_t,
            decomposition_results=decomp_t,
            annotations_df=annotations_df,
            use_labels=True,
            node_size=int(plot_style["node_size"]),
            font_size=int(plot_style["font_size"]),
            title=title,
            ax=ax,
            max_cluster_legend_entries=int(plot_style["max_cluster_legend_entries"]),
            legend_outside=True,
        )
        fig.subplots_adjust(
            top=0.82,
            bottom=0.06,
            left=0.03,
            right=float(plot_style["subplots_right"]),
        )
        figs.append(fig)

    return figs


def _create_tree_panel_renderers_for_case(
    *,
    case_results: list[ComputedResultRecord],
) -> list[tuple[str, Callable[[Axes], None]]]:
    """Return compact tree renderers that can be embedded in comparison pages."""
    tree_results = [r for r in case_results if r.tree is not None and r.decomposition is not None]
    panels: list[tuple[str, Callable[[Axes], None]]] = []

    for idx, result in enumerate(tree_results, start=1):
        method_name = result.method_name
        params_display = format_params_for_display(result.params)
        metrics_text = _format_ari_nmi(result, compact=True)
        title_parts = [f"{method_name} tree ({idx}/{len(tree_results)})"]
        if metrics_text:
            title_parts.append(metrics_text)
        if params_display:
            title_parts.append(params_display)
        panel_title = "\n".join(title_parts[:3])

        def _render_tree(ax: Axes, *, result: ComputedResultRecord = result) -> None:
            plot_tree_with_clusters(
                tree=result.tree,
                decomposition_results=result.decomposition,
                annotations_df=result.annotations,
                use_labels=True,
                node_size=8,
                font_size=6,
                title="",
                ax=ax,
                show_legend=False,
            )

        panels.append((panel_title, _render_tree))

    return panels


def create_umap_then_tree_plots_from_results(
    test_results: list[ComputedResultRecord],
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
        meta = first_result.meta
        if verbose:
            print(f"  Creating UMAP→Tree plots for test case {case_num}...")

        labels_to_plot = {"Ground Truth": first_result.y_true}
        for res in case_results:
            method_name = res.method_name
            params = res.params
            param_str = format_params_for_display(params)
            unique_key = f"{method_name} ({param_str})" if param_str else method_name
            metrics_text = _format_ari_nmi(res, compact=True)
            if metrics_text:
                unique_key = f"{unique_key} [{metrics_text}]"
            if res.labels is not None:
                labels_to_plot[unique_key] = res.labels

        _cache_key = f"{meta['name']}_n{meta['n_samples']}"

        umap_figs = create_clustering_comparison_plots(
            X_original=first_result.x_original,
            labels_dict=labels_to_plot,
            test_case_num=case_num,
            meta=meta,
            cache_key=_cache_key,
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
        if not collect:
            plt.close("all")

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
        if not collect:
            plt.close("all")

    return figs


def create_tree_then_umap_plots_from_results(
    test_results: list[ComputedResultRecord],
    output_dir: Path,
    timestamp: str | None = None,
    verbose: bool = True,
    save: bool = True,
    collect: bool = False,
    collected: list | None = None,
    *,
    pdf: PdfPages | None = None,
) -> list:
    """For each test case: render tree pages first, then UMAP comparison pages."""
    if save:
        output_dir.mkdir(exist_ok=True)
    figs: list = collected if collected is not None else []

    results_by_case = _group_results_by_case(test_results)

    for case_num, case_results in sorted(results_by_case.items()):
        if not case_results:
            continue

        first_result = case_results[0]
        meta = first_result.meta
        if verbose:
            print(f"  Creating Tree→UMAP plots for test case {case_num}...")

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

        labels_to_plot = {"Ground Truth": first_result.y_true}
        for res in case_results:
            method_name = res.method_name
            params = res.params
            param_str = format_params_for_display(params)
            unique_key = f"{method_name} ({param_str})" if param_str else method_name
            metrics_text = _format_ari_nmi(res, compact=True)
            if metrics_text:
                unique_key = f"{unique_key} [{metrics_text}]"
            if res.labels is not None:
                labels_to_plot[unique_key] = res.labels

        _cache_key = f"{meta['name']}_n{meta['n_samples']}"

        umap_figs = create_clustering_comparison_plots(
            X_original=first_result.x_original,
            labels_dict=labels_to_plot,
            test_case_num=case_num,
            meta=meta,
            cache_key=_cache_key,
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
        if not collect:
            plt.close("all")

    return figs


def create_umap_3d_plots_from_results(
    test_results: list[ComputedResultRecord],
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
        if result.labels is None:
            continue

        i = result.test_case_num
        method_name = result.method_name
        params = result.params
        param_str_display = format_params_for_display(params)
        metrics_text = _format_ari_nmi(result)

        method_name_safe = method_name.replace(" ", "_")
        param_str_safe = _format_params_for_filename(params)
        filename = (
            f"umap3d_case_{i}_{method_name_safe}_{param_str_safe}_{timestamp}.png"
            if timestamp
            else f"umap3d_case_{i}_{method_name_safe}_{param_str_safe}.png"
        )

        title = f"3D UMAP - {method_name} ({param_str_display})\nTest Case {i}"
        if metrics_text:
            title = f"{title}\n{metrics_text}"

        if verbose:
            print(f"  Creating 3D UMAP plot for {filename}...")
        fig = create_clustering_comparison_plot_3d(
            result.x_original,
            result.y_true,
            np.asarray(result.labels),
            test_case_num=i,
            meta=result.meta,
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
        if not collect:
            plt.close("all")
    return figs


def create_manifold_plots_from_results(
    test_results: list[ComputedResultRecord],
    output_dir: Path,
    timestamp: str | None = None,
    verbose: bool = True,
    save: bool = True,
    collect: bool = False,
    collected: list | None = None,
    *,
    pdf: PdfPages | None = None,
) -> list:
    from .manifold import create_manifold_alignment_plot

    figs: list = collected if collected is not None else []
    if save:
        output_dir.mkdir(exist_ok=True)
    for result in test_results:
        if result.labels is None:
            continue

        i = result.test_case_num
        method_name = result.method_name
        params = result.params
        param_str_display = format_params_for_display(params)
        metrics_text = _format_ari_nmi(result)

        method_name_safe = method_name.replace(" ", "_")
        param_str_safe = _format_params_for_filename(params)
        filename = (
            f"manifold_case_{i}_{method_name_safe}_{param_str_safe}_{timestamp}.png"
            if timestamp
            else f"manifold_case_{i}_{method_name_safe}_{param_str_safe}.png"
        )

        title = f"Manifold Alignment - {method_name} ({param_str_display})\nTest Case {i}"
        if metrics_text:
            title = f"{title}\n{metrics_text}"

        if verbose:
            print(f"  Creating manifold plot for {filename}...")

        fig, mantel_r, mantel_p = create_manifold_alignment_plot(
            result.x_original,
            np.asarray(result.labels),
            test_case_num=i,
            meta=result.meta,
            y_true=result.y_true,
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
        if not collect:
            plt.close("all")
    return figs


def create_tree_plots_from_results(
    test_results: list[ComputedResultRecord],
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
    results_by_case: dict[int, list[ComputedResultRecord]] = {}
    for result in test_results:
        if result.tree is None or result.decomposition is None:
            continue
        case_num = int(result.test_case_num)
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
