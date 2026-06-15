from __future__ import annotations

import pandas as pd
from benchmarks.diagnostics.calibration.overlap_branch_incidence_junction_panel import (
    OverlapBranchIncidenceJunctionPanelConfig,
    classify_branch_incidence_geometry,
    classify_metric_family_compatibility,
    fisher_weighted_vector,
    run_overlap_branch_incidence_junction_panel,
)


def test_branch_incidence_geometry_separates_alignment_from_mismatch() -> None:
    assert (
        classify_branch_incidence_geometry(
            incoming_family_outgoing_jaccard_topk=0.40,
            incoming_family_outgoing_abs_cosine=0.10,
            incoming_edge_outgoing_jaccard_topk=0.30,
            incoming_edge_outgoing_abs_cosine=0.10,
            outgoing_sibling_contrast_norm=1.0,
        )
        == "income_outcome_branch_aligned"
    )
    assert (
        classify_branch_incidence_geometry(
            incoming_family_outgoing_jaccard_topk=0.30,
            incoming_family_outgoing_abs_cosine=0.10,
            incoming_edge_outgoing_jaccard_topk=0.05,
            incoming_edge_outgoing_abs_cosine=0.05,
            outgoing_sibling_contrast_norm=1.0,
        )
        == "selected_family_outcome_aligned"
    )
    assert (
        classify_branch_incidence_geometry(
            incoming_family_outgoing_jaccard_topk=0.12,
            incoming_family_outgoing_abs_cosine=0.10,
            incoming_edge_outgoing_jaccard_topk=0.05,
            incoming_edge_outgoing_abs_cosine=0.05,
            outgoing_sibling_contrast_norm=1.0,
        )
        == "weak_income_outcome_branch_alignment"
    )
    assert (
        classify_branch_incidence_geometry(
            incoming_family_outgoing_jaccard_topk=0.00,
            incoming_family_outgoing_abs_cosine=0.00,
            incoming_edge_outgoing_jaccard_topk=0.00,
            incoming_edge_outgoing_abs_cosine=0.00,
            outgoing_sibling_contrast_norm=1.0,
        )
        == "coordinate_income_outcome_branch_mismatch"
    )


def test_metric_family_helpers_capture_noncoordinate_weighting() -> None:
    weighted = fisher_weighted_vector(
        [0.1, 0.1],
        [0.5, 0.99],
        variance_floor=1e-4,
    )

    assert weighted[1] > weighted[0]
    assert (
        classify_metric_family_compatibility(metric_family_alignment_score=0.60)
        == "metric_family_branch_compatible"
    )
    assert (
        classify_metric_family_compatibility(metric_family_alignment_score=0.30)
        == "metric_family_branch_weakly_compatible"
    )
    assert (
        classify_metric_family_compatibility(metric_family_alignment_score=0.10)
        == "metric_family_branch_mismatch"
    )


def test_run_branch_incidence_panel_writes_outputs(tmp_path) -> None:
    outputs = run_overlap_branch_incidence_junction_panel(
        OverlapBranchIncidenceJunctionPanelConfig(
            output_dir=tmp_path,
            suite="binary",
            case_names=("binary_2clusters",),
            data_roles=("null",),
            sibling_alpha=0.01,
            edge_alpha=0.001,
            replicates=1,
            base_seed=20260613,
            top_k=4,
        )
    )

    for path in outputs.values():
        assert path.exists()

    rows = pd.read_csv(outputs["rows"])
    assert set(rows.columns) >= {
        "incoming_parent_id",
        "incoming_sibling_id",
        "outgoing_left_child_id",
        "outgoing_right_child_id",
        "incoming_family_outgoing_jaccard_topk",
        "incoming_family_outgoing_fisher_abs_cosine",
        "metric_family_alignment_score",
        "metric_family_compatibility_status",
        "branch_incidence_geometry_status",
    }
