from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from benchmarks.shared.tree_consensus import (
    TREE_CONSENSUS_STRATEGIES,
    build_tree_consensus_tables,
    tree_inference_from_run_id,
)


def _run_id(tree: str) -> str:
    suffix = (
        "tree_builder_neighbor_joining"
        if tree == "neighbor_joining"
        else f"tree_linkage_method_{tree}"
    )
    return (
        f"tbs_diffusion_graphtools_adaptive_nnls::graphtools_adaptive_k_tree_strategy__{suffix}__r0"
    )


def _row(
    *,
    case_id: str,
    tree: str,
    found_clusters: int,
    silhouette: float,
    calinski: float,
    davies: float,
    largest: float,
    ari: float,
) -> dict[str, object]:
    return {
        "test_case": 1,
        "case_id": case_id,
        "case_category": "toy",
        "source_family": "toy_source",
        "feature_representation": "toy_representation",
        "method": "tbs_diffusion_graphtools_adaptive_nnls",
        "run_id": _run_id(tree),
        "benchmark_class": "optional_gpl",
        "benchmark_grid": "graphtools_adaptive_k_tree_strategy",
        "benchmark_repeat": 0,
        "status": "ok",
        "skip_reason": "",
        "true_clusters": 3,
        "found_clusters": found_clusters,
        "labels_length": 12,
        "ari": ari,
        "nmi": ari,
        "macro_f1": ari,
        "purity": ari,
        "silhouette_score": silhouette,
        "calinski_harabasz_index": calinski,
        "davies_bouldin_index": davies,
        "largest_cluster_fraction": largest,
    }


def _labels(case_id: str, labels_by_tree: dict[str, list[int]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for tree, labels in labels_by_tree.items():
        for sample_index, label in enumerate(labels):
            rows.append(
                {
                    "test_case": 1,
                    "case_id": case_id,
                    "method": "tbs_diffusion_graphtools_adaptive_nnls",
                    "run_id": _run_id(tree),
                    "sample_id": f"S{sample_index}",
                    "cluster_label": int(label),
                }
            )
    return pd.DataFrame(rows)


def _baseline_rows(case_id: str) -> list[dict[str, object]]:
    return [
        _row(
            case_id=case_id,
            tree=tree,
            found_clusters=3,
            silhouette=0.2,
            calinski=20.0,
            davies=2.0,
            largest=0.3,
            ari=0.5,
        )
        for tree in TREE_CONSENSUS_STRATEGIES
    ]


def test_tree_inference_from_run_id_recovers_grid_cells() -> None:
    assert tree_inference_from_run_id(_run_id("average")) == "average"
    assert tree_inference_from_run_id(_run_id("ward")) == "ward"
    assert tree_inference_from_run_id(_run_id("neighbor_joining")) == "neighbor_joining"
    with pytest.raises(ValueError, match="Cannot infer tree strategy"):
        tree_inference_from_run_id("tbs::default")


def test_selector_penalizes_fragmentation_against_better_internal_fit() -> None:
    rows = _baseline_rows("fragmented")
    for row in rows:
        if tree_inference_from_run_id(str(row["run_id"])) == "average":
            row.update(
                found_clusters=8,
                silhouette_score=0.05,
                calinski_harabasz_index=8.0,
                davies_bouldin_index=4.0,
                largest_cluster_fraction=0.16,
            )
        if tree_inference_from_run_id(str(row["run_id"])) == "neighbor_joining":
            row.update(
                found_clusters=3,
                silhouette_score=0.30,
                calinski_harabasz_index=80.0,
                davies_bouldin_index=1.1,
                largest_cluster_fraction=0.34,
            )
    labels = _labels(
        "fragmented",
        {tree: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2] for tree in TREE_CONSENSUS_STRATEGIES},
    )

    tables = build_tree_consensus_tables(pd.DataFrame(rows), labels)

    selected = tables.selection.iloc[0]
    assert selected["selector_status"] == "selected"
    assert selected["selected_tree_inference"] == "neighbor_joining"


def test_selector_applies_dominant_cluster_guard() -> None:
    rows = _baseline_rows("dominant")
    for row in rows:
        tree = tree_inference_from_run_id(str(row["run_id"]))
        if tree == "single":
            row.update(
                found_clusters=3,
                silhouette_score=0.50,
                calinski_harabasz_index=100.0,
                davies_bouldin_index=0.8,
                largest_cluster_fraction=0.55,
            )
        if tree == "complete":
            row.update(
                found_clusters=3,
                silhouette_score=0.35,
                calinski_harabasz_index=70.0,
                davies_bouldin_index=1.0,
                largest_cluster_fraction=0.34,
            )
    labels = _labels(
        "dominant",
        {tree: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2] for tree in TREE_CONSENSUS_STRATEGIES},
    )

    tables = build_tree_consensus_tables(pd.DataFrame(rows), labels)

    selected = tables.selection.iloc[0]
    assert selected["selected_tree_inference"] == "complete"


def test_selector_prefers_weighted_for_equivalent_topology_ties() -> None:
    rows = _baseline_rows("tie")
    labels = _labels(
        "tie",
        {tree: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2] for tree in TREE_CONSENSUS_STRATEGIES},
    )

    tables = build_tree_consensus_tables(pd.DataFrame(rows), labels)

    selected = tables.selection.iloc[0]
    assert selected["selected_tree_inference"] == "weighted"


def test_selector_is_invariant_to_external_metrics() -> None:
    rows = _baseline_rows("external")
    labels = _labels(
        "external",
        {tree: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2] for tree in TREE_CONSENSUS_STRATEGIES},
    )
    baseline = build_tree_consensus_tables(pd.DataFrame(rows), labels).selection
    perturbed = pd.DataFrame(rows)
    for column in ["ari", "nmi", "macro_f1", "purity", "true_clusters"]:
        perturbed[column] = np.nan

    selected = build_tree_consensus_tables(perturbed, labels).selection

    assert selected["selected_run_id"].tolist() == baseline["selected_run_id"].tolist()


def test_cases_with_no_valid_topology_are_reported_as_skips() -> None:
    rows = _baseline_rows("invalid")
    for row in rows:
        row.update(status="skip", labels_length=0, found_clusters=0)
    labels = pd.DataFrame(
        columns=["test_case", "case_id", "method", "run_id", "sample_id", "cluster_label"]
    )

    tables = build_tree_consensus_tables(pd.DataFrame(rows), labels)

    selected = tables.selection.iloc[0]
    assert selected["selector_status"] == "skip_no_valid_topology"
    assert pd.isna(selected["selected_run_id"])
