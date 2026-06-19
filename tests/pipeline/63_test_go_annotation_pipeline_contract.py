from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pandas as pd
from scripts.analysis.audit_go_annotation_analysis_levels import audit_paths
from scripts.analysis.inventory_go_annotation_datasets import (
    canonical_matrix_name,
    dataset_slug_from_matrix_name,
    matrix_inventory,
)
from scripts.analysis.run_go_annotation_feature_matrix_pipeline import (
    build_pipeline_plan,
    matrix_slug,
)


def test_audit_distinguishes_go_annotation_analysis_levels(tmp_path: Path) -> None:
    quality = tmp_path / "quality"
    quality.mkdir()
    (quality / "feature_matrix_quality_report.pdf").write_text("pdf", encoding="utf-8")
    pd.DataFrame([{"metric": "n_genes", "value": 3}]).to_csv(
        quality / "matrix_quality_summary.csv",
        index=False,
    )

    method_matrix = tmp_path / "method_matrix"
    method_matrix.mkdir()
    pd.DataFrame(
        [
            {
                "status": "ok",
                "method_version": "current",
                "tree_geometry": "raw_cosine_subspace",
                "assignments_path": "assignments/current.csv",
            }
        ]
    ).to_csv(method_matrix / "method_tree_matrix_summary.csv", index=False)
    (method_matrix / "assignments").mkdir()
    (method_matrix / "assignments/current.csv").write_text("gene,cluster_id\nA,0\n", encoding="utf-8")

    mixed = tmp_path / "mixed_go_ic"
    mixed.mkdir()
    pd.DataFrame(
        [
            {
                "run_id": "current__raw_cosine_subspace__binary__block",
                "family": "current__raw_cosine_subspace",
                "go_bic_active_per_gene": 1.0,
            }
        ]
    ).to_csv(mixed / "allgo_new_quality_aware_go_ic_tree_ranking.csv", index=False)
    method_pdf = mixed / "allgo_new_quality_aware_go_ic_by_method/current"
    method_pdf.mkdir(parents=True)
    (method_pdf / "current_tree_pages.pdf").write_text("pdf", encoding="utf-8")

    canonical = tmp_path / "canonical"
    (canonical / "rankings").mkdir(parents=True)
    (canonical / "subspaces").mkdir()
    (canonical / "connected_results_manifest.json").write_text("{}", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "run_id": "current__adaptive_diffusion_cosine_subspace__binary__block",
                "method_version": "current",
                "tree_geometry": "adaptive_diffusion_cosine_subspace",
                "status": "ok",
            }
        ]
    ).to_csv(
        canonical / "rankings/current_adaptive_diffusion_subspace_tree_ranking.csv",
        index=False,
    )

    audit = audit_paths([quality, method_matrix, mixed, canonical])
    by_path = {Path(row.path).name: row for row in audit.itertuples(index=False)}

    assert by_path["quality"].level_code == "00_matrix_quality_only"
    assert by_path["method_matrix"].level_code == "10_candidate_tree_generation"
    assert by_path["mixed_go_ic"].level_code == "20_mixed_method_go_ic_reader_report"
    assert by_path["canonical"].level_code == "30_canonical_current_subspace_pipeline"
    assert bool(by_path["canonical"].is_canonical) is True


def test_pipeline_plan_uses_canonical_current_stage_and_optional_audit_matrix(tmp_path: Path) -> None:
    input_path = tmp_path / "feature_matrix_toy_go.tsv"
    input_path.write_text("gene\tGO:0000001\nA\t1\nB\t0\n", encoding="utf-8")
    args = Namespace(
        input=input_path,
        output_dir=tmp_path / "out",
        dataset_label=None,
        python="python",
        skip_quality=False,
        skip_current_subspace=False,
        include_method_matrix=True,
        edge_alpha=None,
        sibling_alpha=None,
        max_rank=12,
        min_segment_length=4,
        max_segments=3,
        diffusion_k_neighbors=5,
        diffusion_time=2,
        diffusion_components=8,
        adaptive_bandwidth_type="-1/(d+2)",
        adaptive_epsilon="median",
        adaptive_metric="euclidean",
        weightings=["binary"],
        block_names=["adaptive_common_mode_01"],
        method_versions=["current"],
        tree_geometries=["adaptive_diffusion_cosine_subspace"],
    )

    stages = build_pipeline_plan(args, args.output_dir)
    assert [stage.stage_id for stage in stages] == [
        "00_dataset_inventory",
        "05_matrix_quality",
        "10_current_adaptive_diffusion_subspace_tree",
        "20_method_tree_matrix_audit",
    ]
    assert stages[0].analysis_level == "dataset_inventory_and_naming_preflight"
    assert stages[2].analysis_level == "30_canonical_current_subspace_pipeline"
    assert "--dataset-label" in stages[2].command
    assert "toy_go" in stages[2].command
    assert stages[3].optional is True


def test_matrix_slug_removes_feature_matrix_prefix() -> None:
    assert matrix_slug(Path("feature_matrix_julia_allGO_new.tsv")) == "julia_allgo_new"


def test_dataset_inventory_accepts_feature_matrix_prefix_and_flags_download_duplicate(
    tmp_path: Path,
) -> None:
    feature_root = tmp_path / "data" / "feature_matrices"
    feature_root.mkdir(parents=True)
    canonical = feature_root / "feature_matrix_allGO_new_interactome.tsv"
    duplicate = tmp_path / "feature_matrix_allGO_new_interactome (1).tsv"
    content = "gene\tGO:0000001\tGO:0000002\nA\t1\t0\nB\t0\t1\n"
    canonical.write_text(content, encoding="utf-8")
    duplicate.write_text(content, encoding="utf-8")

    assert dataset_slug_from_matrix_name(canonical) == "allgo_new_interactome"
    assert canonical_matrix_name(canonical) == "feature_matrix_allGO_new_interactome.tsv"

    inventory = matrix_inventory(feature_root, [duplicate])
    by_name = {Path(row.path).name: row for row in inventory.itertuples(index=False)}

    assert by_name["feature_matrix_allGO_new_interactome.tsv"].naming_status == "canonical"
    assert by_name["feature_matrix_allGO_new_interactome (1).tsv"].naming_status == (
        "outside_feature_root"
    )
    assert bool(by_name["feature_matrix_allGO_new_interactome.tsv"].has_duplicate_copy) is True
