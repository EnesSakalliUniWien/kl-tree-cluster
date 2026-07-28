from __future__ import annotations

from pathlib import Path

import pandas as pd
from benchmarks.cloud.aws_tree_strategy_semantic_panel import (
    MANIFEST_NAME,
    PANEL_NAME,
    AwsTreeStrategyPanelConfig,
    parse_alpha_summary_paths,
    run_panel,
)
from benchmarks.diagnostics.spectral.stability.tree_strategy_semantic_panel import (
    build_semantic_panel,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(rows).to_csv(path, index=False)


def _write_fixture(root: Path) -> tuple[Path, Path, Path]:
    blob = root / "benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis"
    diagnostics = root / "benchmarks/results/diagnostics"
    alpha_summary = root / "extra_alpha.csv"

    _write_csv(
        blob / "12_context_quality_tree_comparison_20260610/context_method_quality_summary.csv",
        [
            {
                "method": "full_adaptive_diffusion",
                "n_clusters": 63,
                "gene_singleton_fraction": 0.01,
                "medium_context_clusters_size_5_80": 40,
                "mean_non_singleton_jaccard_gene_weighted": 0.14,
                "enriched_non_singleton_fraction_q05": 0.92,
            },
            {
                "method": "raw_kak_binary_10_15",
                "n_clusters": 101,
                "gene_singleton_fraction": 0.02,
                "medium_context_clusters_size_5_80": 50,
                "mean_non_singleton_jaccard_gene_weighted": 0.11,
                "enriched_non_singleton_fraction_q05": 0.76,
            },
        ],
    )
    _write_csv(
        blob
        / "13_subspace_lens_vs_main_adaptive_contexts_20260610/subspace_lens_vs_main_context_summary.csv",
        [
            {
                "lens_family": "raw_kak",
                "lens": "binary__adaptive_modes_10_15",
                "lens_refinement_score": 0.89,
            }
        ],
    )
    _write_csv(
        blob
        / "21_kak_lens_feature_axis_clustering_debug_20260612/kak_lens_feature_axis_clustering_summary.csv",
        [
            {
                "lens_id": "raw_kak__binary__adaptive_modes_10_15",
                "tree_linkage_method": "average",
                "edge_alpha": 0.001,
                "sibling_alpha": 0.01,
                "n_clusters": 101,
                "singleton_gene_fraction": 0.02,
                "top_axis_connection_feature": "GO:demo",
                "top_axis_connection_score": 0.003,
            }
        ],
    )
    _write_csv(
        diagnostics
        / "recursive_pvalue_geometry_full_20260606/recursive_pvalue_geometry_summary.csv",
        [
            {
                "summary_id": "edge_parent_sibling_pvalue_coupling",
                "scope": "global",
                "value": 0.31,
            },
            {
                "summary_id": "recursive_sibling_pvalue_continuity",
                "scope": "global",
                "value": 0.54,
            },
        ],
    )
    _write_csv(
        diagnostics
        / "path_conditioned_barycentric_action_20260606_guard_utility/kak_radius_angle_action_summary.csv",
        [{"model_id": "radius_angle_action", "median_auc": 0.89}],
    )
    _write_csv(
        diagnostics
        / "path_conditioned_barycentric_action_20260606_guard_utility/kak_internal_geometry_block_summary.csv",
        [
            {
                "run_id": "binary__adaptive_modes_10_15",
                "angle_to_leading_axis_deg_median": 70.0,
                "independent_fraction_median": 0.94,
                "sibling_separation_parent_ratio_median": 1.8,
            }
        ],
    )
    _write_csv(
        alpha_summary,
        [
            {
                "lens_id": "raw_kak__binary__adaptive_modes_10_15",
                "tree_linkage_method": "average",
                "edge_alpha": 0.001,
                "sibling_alpha": 0.01,
                "status": "ok",
                "n_clusters": 101,
                "gene_singleton_fraction": 0.02,
                "alpha_lens_score": 0.88,
            }
        ],
    )
    return blob, diagnostics, alpha_summary


def test_build_semantic_panel_normalizes_requested_columns(tmp_path: Path) -> None:
    blob, diagnostics, alpha_summary = _write_fixture(tmp_path)

    panel = build_semantic_panel(
        blob_results_dir=blob,
        diagnostics_results_dir=diagnostics,
        alpha_summary_paths=[alpha_summary],
    )

    assert {
        "tree_strategy",
        "lens_family",
        "linkage",
        "alpha",
        "n_clusters",
        "singleton_fraction",
        "medium_contexts",
        "GO Jaccard",
        "enrichment_fraction",
        "main_context_refinement",
        "axis_feature_bridge",
        "edge/sibling p-value continuity",
        "radius_angle_action",
    }.issubset(panel.columns)
    assert "main_context_tree" in set(panel["semantic_interpretation"])
    raw = panel[panel["lens_id"].eq("raw_kak__binary__adaptive_modes_10_15")]
    assert raw["axis_feature_bridge"].str.contains("GO:demo").any()
    assert raw["radius_angle_action"].str.contains("auc=0.890000").any()


def test_aws_wrapper_writes_panel_manifest_and_report(tmp_path: Path) -> None:
    blob, diagnostics, alpha_summary = _write_fixture(tmp_path)
    output_dir = tmp_path / "out"

    panel = run_panel(
        AwsTreeStrategyPanelConfig(
            output_dir=output_dir,
            blob_results_dir=blob,
            diagnostics_results_dir=diagnostics,
            alpha_summary_paths=(alpha_summary,),
        )
    )

    assert not panel.empty
    assert (output_dir / PANEL_NAME).exists()
    assert (output_dir / "tree_strategy_semantic_panel_report.md").exists()
    assert (output_dir / MANIFEST_NAME).exists()


def test_parse_alpha_summary_paths() -> None:
    assert parse_alpha_summary_paths("a.csv,b.csv") == (Path("a.csv"), Path("b.csv"))
    assert parse_alpha_summary_paths(None) == ()
