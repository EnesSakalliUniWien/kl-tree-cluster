#!/usr/bin/env python3
"""Run the selected-family traversal diagnostic on one feature matrix."""

from __future__ import annotations

import argparse
import json
import signal
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from benchmarks.diagnostics.calibration.selected.family.multiscale_umap import (
    load_overlay_data,
    render_multiscale_umap_overlay,
)
from benchmarks.diagnostics.calibration.selected.family.selected_family_traversal_panel import (
    SCHEMA_VERSION,
    STUDY_ROLE,
    _build_guard_rows,
    _build_node_decisions,
    _build_regions_and_gene_assignments,
)
from benchmarks.shared.runners.tbs_runner import run_tbs_on_distance
from benchmarks.shared.util.time import format_timestamp_utc


class MatrixRunTimeout(TimeoutError):
    """Raised when the selected-family matrix run exceeds its time budget."""


@contextmanager
def time_limit(seconds: float | None):
    if seconds is None or float(seconds) <= 0.0:
        yield
        return

    def raise_timeout(_signum, _frame):
        raise MatrixRunTimeout(f"selected-family matrix run exceeded {float(seconds):g} seconds")

    previous_handler = signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-matrix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--method-id",
        default="fixed_coordinate_global_passthrough_refined_v1",
    )
    parser.add_argument(
        "--sibling-gate-profile",
        default="fixed_coordinate_global_passthrough_refined_v1",
    )
    parser.add_argument("--edge-alpha", type=float, default=0.001)
    parser.add_argument("--sibling-alpha", type=float, default=0.01)
    parser.add_argument("--tree-distance-metric", default="hamming")
    parser.add_argument("--tree-linkage-method", default="average")
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--umap-coordinates", type=Path)
    parser.add_argument("--top-regions", type=int, default=20)
    return parser.parse_args()


def load_binary_matrix(path: Path) -> pd.DataFrame:
    first_line = path.open(encoding="utf-8").readline()
    separator = "\t" if "\t" in first_line else ","
    data = pd.read_csv(path, sep=separator, index_col=0)
    numeric = data.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Feature matrix contains non-finite values.")
    unique_values = set(np.unique(numeric.to_numpy()))
    if not unique_values <= {0, 1}:
        raise ValueError(
            "Selected-family matrix runner currently expects a binary matrix; "
            f"found values outside {{0,1}}: {sorted(unique_values - {0, 1})[:10]!r}."
        )
    return numeric.astype(int)


def assignments_from_result(result, sample_index: pd.Index) -> pd.DataFrame:
    labels = np.asarray(result.labels, dtype=int)
    sizes = pd.Series(labels).value_counts()
    assignments = pd.DataFrame(
        {
            "sample_id": sample_index.astype(str),
            "cluster_id": labels,
            "cluster_size": [int(sizes[label]) for label in labels],
        }
    )
    return assignments


def run_matrix(args: argparse.Namespace) -> dict[str, object]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_binary_matrix(args.feature_matrix)
    distance = pdist(data.to_numpy(dtype=float), metric=str(args.tree_distance_metric))
    with time_limit(args.timeout_seconds):
        result = run_tbs_on_distance(
            data,
            distance,
            sibling_significance_level=float(args.sibling_alpha),
            tree_linkage_method=str(args.tree_linkage_method),
            edge_alpha=float(args.edge_alpha),
            sibling_gate_profile=str(args.sibling_gate_profile),
            trace_level="full",
        )

    node_decisions = _build_node_decisions(
        case_id=args.feature_matrix.stem,
        data_role="observed",
        method_id=str(args.method_id),
        replicate=0,
        data_seed=0,
        result=result,
    )
    guard_rows = _build_guard_rows(node_decisions, result)
    regions, gene_assignments = _build_regions_and_gene_assignments(
        node_decisions=node_decisions,
        result=result,
    )
    assignments = assignments_from_result(result, data.index)

    assignments_path = args.output_dir / "cluster_assignments.csv"
    node_path = args.output_dir / "multiscale_node_decisions.csv"
    guard_path = args.output_dir / "selected_family_guard_rows.csv"
    regions_path = args.output_dir / "multiscale_regions.csv"
    genes_path = args.output_dir / "multiscale_gene_assignments.csv"
    manifest_path = args.output_dir / "manifest.json"

    assignments.to_csv(assignments_path, index=False)
    node_decisions.to_csv(node_path, index=False)
    guard_rows.to_csv(guard_path, index=False)
    regions.to_csv(regions_path, index=False)
    gene_assignments.to_csv(genes_path, index=False)

    overlay_outputs: dict[str, str] = {}
    if args.umap_coordinates is not None:
        overlay_dir = args.output_dir / "umap_overlay"
        overlay_data = load_overlay_data(
            gene_assignments_path=genes_path,
            umap_coordinates_path=args.umap_coordinates,
            method_id=str(args.method_id),
            data_role="observed",
            replicate=0,
        )
        overlay_dir.mkdir(parents=True, exist_ok=True)
        overlay_table = overlay_dir / "multiscale_umap_overlay_data.csv"
        overlay_plot = overlay_dir / "multiscale_umap_overlay.png"
        overlay_data.to_csv(overlay_table, index=False)
        render_multiscale_umap_overlay(
            overlay_data,
            overlay_plot,
            top_regions=int(args.top_regions),
        )
        overlay_outputs = {
            "overlay_data": str(overlay_table),
            "overlay_plot": str(overlay_plot),
        }

    manifest = {
        "created_at_utc": format_timestamp_utc(),
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "input": {
            "feature_matrix": str(args.feature_matrix),
            "n_samples": int(data.shape[0]),
            "n_features": int(data.shape[1]),
        },
        "config": {
            "method_id": str(args.method_id),
            "sibling_gate_profile": str(args.sibling_gate_profile),
            "edge_alpha": float(args.edge_alpha),
            "sibling_alpha": float(args.sibling_alpha),
            "tree_distance_metric": str(args.tree_distance_metric),
            "tree_linkage_method": str(args.tree_linkage_method),
            "timeout_seconds": args.timeout_seconds,
        },
        "summary": {
            "found_clusters": int(result.found_clusters),
            "stable_regions": int(regions["region_id"].nunique()) if not regions.empty else 0,
            "guard_rows": int(guard_rows.shape[0]),
            "selected_family_guard_blocks": int(guard_rows["guard_blocked"].astype(bool).sum())
            if not guard_rows.empty
            else 0,
        },
        "outputs": {
            "cluster_assignments": str(assignments_path),
            "multiscale_node_decisions": str(node_path),
            "selected_family_guard_rows": str(guard_path),
            "multiscale_regions": str(regions_path),
            "multiscale_gene_assignments": str(genes_path),
            **overlay_outputs,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    try:
        manifest = run_matrix(args)
    except MatrixRunTimeout as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "created_at_utc": format_timestamp_utc(),
            "schema_version": SCHEMA_VERSION,
            "study_role": STUDY_ROLE,
            "input": {"feature_matrix": str(args.feature_matrix)},
            "config": {
                "method_id": str(args.method_id),
                "sibling_gate_profile": str(args.sibling_gate_profile),
                "timeout_seconds": args.timeout_seconds,
            },
            "status": "timeout",
            "reason": str(exc),
        }
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
