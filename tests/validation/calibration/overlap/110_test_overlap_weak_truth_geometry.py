from __future__ import annotations

import pandas as pd
from benchmarks.diagnostics.calibration.overlap.overlap_weak_truth_geometry import (
    OverlapWeakTruthGeometryConfig,
    build_weak_truth_geometry_families,
    build_weak_truth_geometry_rows,
    classify_truth_geometry,
    run_overlap_weak_truth_geometry,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "case_a",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "aligned",
                "depth": 2,
                "decision_class": "accepted_internal_split",
                "structural_sibling_status": "weak_homogeneity_gain",
                "n_parent": 100,
                "n_left": 48,
                "n_right": 52,
                "barycentric_balance": 0.48,
                "truth_split_ari": 0.70,
                "parent_truth_purity": 0.45,
                "left_truth_purity": 0.90,
                "right_truth_purity": 0.80,
                "homogeneity_gain_min": 0.01,
                "subspace_consensus_jaccard_topk": 0.40,
            },
            {
                "case_id": "case_b",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "fragment",
                "depth": 1,
                "decision_class": "accepted_internal_split",
                "structural_sibling_status": "weak_homogeneity_gain",
                "n_parent": 100,
                "n_left": 20,
                "n_right": 80,
                "barycentric_balance": 0.20,
                "truth_split_ari": 0.20,
                "parent_truth_purity": 0.30,
                "left_truth_purity": 0.95,
                "right_truth_purity": 0.30,
                "homogeneity_gain_min": 0.02,
                "subspace_consensus_jaccard_topk": 0.30,
            },
            {
                "case_id": "case_c",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "wrong_granularity",
                "depth": 1,
                "decision_class": "accepted_internal_split",
                "structural_sibling_status": "weak_homogeneity_gain",
                "n_parent": 100,
                "n_left": 50,
                "n_right": 50,
                "barycentric_balance": 0.50,
                "truth_split_ari": 0.30,
                "parent_truth_purity": 0.40,
                "left_truth_purity": 0.70,
                "right_truth_purity": 0.72,
                "homogeneity_gain_min": 0.03,
                "subspace_consensus_jaccard_topk": 0.20,
            },
            {
                "case_id": "case_d",
                "data_role": "selected_null",
                "replicate": 0,
                "node_id": "null",
                "depth": 1,
                "decision_class": "accepted_internal_split",
                "structural_sibling_status": "weak_homogeneity_gain",
                "n_parent": 100,
                "n_left": 50,
                "n_right": 50,
                "barycentric_balance": 0.50,
                "truth_split_ari": pd.NA,
                "parent_truth_purity": 1.0,
                "left_truth_purity": 1.0,
                "right_truth_purity": 1.0,
                "homogeneity_gain_min": 0.0,
                "subspace_consensus_jaccard_topk": 0.0,
            },
        ]
    )


def test_classify_truth_geometry_modes() -> None:
    assert (
        classify_truth_geometry(
            truth_split_ari=0.7,
            min_child_truth_purity=0.8,
            max_child_truth_purity=0.9,
            child_truth_purity_gap=0.1,
        )
        == "balanced_truth_recovery"
    )
    assert (
        classify_truth_geometry(
            truth_split_ari=0.2,
            min_child_truth_purity=0.3,
            max_child_truth_purity=0.95,
            child_truth_purity_gap=0.65,
        )
        == "one_sided_pure_fragment"
    )
    assert (
        classify_truth_geometry(
            truth_split_ari=0.3,
            min_child_truth_purity=0.7,
            max_child_truth_purity=0.72,
            child_truth_purity_gap=0.02,
        )
        == "balanced_but_wrong_granularity"
    )


def test_truth_geometry_rows_and_families_use_signal_weak_rows_only() -> None:
    row_geometry = build_weak_truth_geometry_rows(_rows())
    family_geometry = build_weak_truth_geometry_families(row_geometry)

    assert set(row_geometry["node_id"]) == {
        "aligned",
        "fragment",
        "wrong_granularity",
    }
    assert set(row_geometry["truth_geometry_mode"]) == {
        "balanced_truth_recovery",
        "one_sided_pure_fragment",
        "balanced_but_wrong_granularity",
    }
    assert set(family_geometry["family_truth_geometry_mode"]) == {
        "family_contains_truth_recovery",
        "family_one_sided_pure_fragment",
        "family_balanced_wrong_granularity",
    }


def test_run_truth_geometry_writes_outputs(tmp_path) -> None:
    rows_path = tmp_path / "rows.csv"
    _rows().to_csv(rows_path, index=False)

    outputs = run_overlap_weak_truth_geometry(
        OverlapWeakTruthGeometryConfig(
            rows_path=rows_path,
            output_dir=tmp_path / "out",
        )
    )

    for path in outputs.values():
        assert path.exists()
