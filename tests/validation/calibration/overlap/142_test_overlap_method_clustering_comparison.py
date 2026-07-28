from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from benchmarks.diagnostics.calibration.overlap.overlap_method_clustering_comparison import (
    DEFAULT_LEFT_METHOD,
    DEFAULT_RIGHT_METHOD,
    load_assignment_record,
    run_analysis,
)


def _write_checkpoint(
    root: Path,
    *,
    case_id: str,
    role: str,
    data_role: str,
    method_id: str,
    replicate: int,
    labels: list[int],
    true_clusters: int,
    ari: float,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    stem = f"case={case_id}__role={role}__method={method_id}__replicate={replicate:04d}"
    genes = pd.DataFrame(
        {
            "case_id": case_id,
            "data_role": data_role,
            "method_id": method_id,
            "replicate": replicate,
            "sample_id": [f"S{i}" for i in range(len(labels))],
            "stable_region_id": "flat_cluster_0",
            "zone_region_id": "zone_root",
            "final_fragment_cluster_id": labels,
            "boundary_root_node": "root",
            "path_decision_classes": "leaf_fragment",
            "path_node_ids": [f"L{i}" for i in range(len(labels))],
            "study_role": "test",
        }
    )
    genes.to_csv(root / f"{stem}.genes.csv", index=False)
    pd.DataFrame(
        {
            "schema_version": ["test/v1"],
            "study_role": ["test"],
            "case_id": [case_id],
            "data_role": [data_role],
            "method_id": [method_id],
            "replicate": [replicate],
            "true_clusters": [true_clusters],
            "found_clusters": [len(set(labels))],
            "ari": [ari],
            "exact_cluster_count": [len(set(labels)) == true_clusters],
            "false_split": [len(set(labels)) > true_clusters],
        }
    ).to_csv(root / f"{stem}.row.csv", index=False)


def test_load_assignment_record_computes_null_truth_metrics(tmp_path: Path) -> None:
    root = tmp_path / "checkpoint_rows"
    _write_checkpoint(
        root,
        case_id="overlap_null",
        role="null",
        data_role="selected_null",
        method_id=DEFAULT_LEFT_METHOD,
        replicate=0,
        labels=[0, 0, 1, 2],
        true_clusters=1,
        ari=0.0,
    )

    record = load_assignment_record(
        root
        / (f"case=overlap_null__role=null__method={DEFAULT_LEFT_METHOD}__replicate=0000.genes.csv")
    )

    assert record.row["n_samples"] == 4
    assert record.row["n_clusters"] == 3
    assert record.row["n_singleton_clusters"] == 2
    assert record.row["external_metrics_available"] is True
    assert record.row["truth_label_source"] == "single_true_cluster_from_row_metadata"
    assert np.isfinite(record.row["ari"])


def test_run_analysis_pairs_methods_and_summarizes(tmp_path: Path) -> None:
    root = tmp_path / "checkpoint_rows"
    _write_checkpoint(
        root,
        case_id="overlap_null",
        role="null",
        data_role="selected_null",
        method_id=DEFAULT_LEFT_METHOD,
        replicate=0,
        labels=[0, 0, 1, 2],
        true_clusters=1,
        ari=0.0,
    )
    _write_checkpoint(
        root,
        case_id="overlap_null",
        role="null",
        data_role="selected_null",
        method_id=DEFAULT_RIGHT_METHOD,
        replicate=0,
        labels=[0, 0, 0, 0],
        true_clusters=1,
        ari=1.0,
    )
    _write_checkpoint(
        root,
        case_id="overlap_signal",
        role="signal",
        data_role="signal",
        method_id=DEFAULT_LEFT_METHOD,
        replicate=0,
        labels=[0, 0, 1, 1],
        true_clusters=2,
        ari=0.5,
    )
    _write_checkpoint(
        root,
        case_id="overlap_signal",
        role="signal",
        data_role="signal",
        method_id=DEFAULT_RIGHT_METHOD,
        replicate=0,
        labels=[0, 0, 1, 1],
        true_clusters=2,
        ari=0.5,
    )

    rows, pairwise, summary = run_analysis(
        checkpoint_root=root,
        output_dir=tmp_path / "out",
    )

    assert len(rows) == 4
    assert len(pairwise) == 2
    null_pair = pairwise[pairwise["case_id"] == "overlap_null"].iloc[0]
    assert null_pair["fragmentation_pattern"] == "left_more_fragmented"
    assert null_pair["delta_n_clusters_left_minus_right"] == 2
    assert null_pair["delta_run_row_ari_left_minus_right"] == -1.0

    signal_rows = rows[rows["case_id"] == "overlap_signal"]
    assert not signal_rows["external_metrics_available"].any()
    signal_pair = pairwise[pairwise["case_id"] == "overlap_signal"].iloc[0]
    assert signal_pair["partition_ari_between_methods"] == 1.0
    assert not summary.empty
    assert (tmp_path / "out" / "overlap_method_clustering_rows.csv").exists()
    assert (tmp_path / "out" / "overlap_method_clustering_pairwise.csv").exists()
    assert (tmp_path / "out" / "overlap_method_clustering_summary.csv").exists()
    assert (tmp_path / "out" / "manifest.json").exists()
