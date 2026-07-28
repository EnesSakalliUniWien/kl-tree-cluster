"""Run adaptive cosine/KAK spectral-block trees on a feature matrix.

This is the full-data counterpart to
``adaptive_cosine_kak_benchmark_probe.py``. It writes one row per
weighting/block plus per-block assignments, but does not assume benchmark
ground-truth labels.
"""

from __future__ import annotations

import argparse
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tree_break_selection.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_EDGE_ALPHA,
    DEFAULT_SIBLING_ALPHA,
)
from tree_break_selection.space_separation import (
    adaptive_spectral_blocks,
    cosine_eigendecomposition,
    weight_feature_matrix,
)

from benchmarks.diagnostics.spectral.adaptive_cosine_kak_benchmark_probe import (
    SCHEMA_VERSION,
    run_block_tree,
    sibling_method_counts,
)

MATRIX_SCHEMA_VERSION = f"{SCHEMA_VERSION}/matrix"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run adaptive cosine/KAK spectral-block trees on a feature matrix."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/feature_matrices/feature_matrix_julia_GOCC_GOBP_GOMF_combined.tsv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults under benchmarks/results/diagnostics.",
    )
    parser.add_argument("--edge-alpha", type=float, default=DEFAULT_EDGE_ALPHA)
    parser.add_argument("--sibling-alpha", type=float, default=DEFAULT_SIBLING_ALPHA)
    parser.add_argument(
        "--enforce-internal-support-thresholds",
        action="store_true",
        help="Fail closed when internal empirical-null support is below thresholds.",
    )
    parser.add_argument("--max-rank", type=int, default=80)
    parser.add_argument("--min-segment-length", type=int, default=4)
    parser.add_argument("--max-segments", type=int, default=8)
    parser.add_argument(
        "--weightings",
        nargs="+",
        default=["binary", "tfidf"],
        choices=["binary", "tfidf"],
        help="'binary' means raw matrix values, matching the historical script name.",
    )
    return parser.parse_args()


def default_output_dir(input_path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    return (
        Path("benchmarks/results/diagnostics")
        / f"adaptive_cosine_kak_matrix_probe_{input_path.stem}_{stamp}"
    )


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def load_matrix(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, sep="\t", index_col=0)
    zero_columns = data.columns[(data.sum(axis=0) == 0).to_numpy()]
    if len(zero_columns):
        data = data.drop(columns=zero_columns)
    zero_rows = data.index[(data.sum(axis=1) == 0).to_numpy()]
    if len(zero_rows):
        raise ValueError(
            "Rows with zero feature mass cannot enter cosine/KAK analysis: "
            f"{list(zero_rows[:10])!r}"
        )
    return data.astype(float)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or default_output_dir(args.input)
    assignments_dir = output_dir / "assignments"
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments_dir.mkdir(parents=True, exist_ok=True)

    data = load_matrix(args.input)
    base = {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "input_path": str(args.input),
        "n_samples": int(data.shape[0]),
        "n_features": int(data.shape[1]),
        "edge_alpha": float(args.edge_alpha),
        "sibling_alpha": float(args.sibling_alpha),
        "enforce_internal_support_thresholds": bool(args.enforce_internal_support_thresholds),
    }

    rows: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []
    spectrum_rows: list[dict[str, object]] = []
    cluster_size_rows: list[dict[str, object]] = []

    for weighting in args.weightings:
        print(f"[{weighting}] eigendecomposition", flush=True)
        try:
            values = weight_feature_matrix(data, weighting)
            eigvals, eigvecs = cosine_eigendecomposition(values, args.max_rank)
            total_energy = float(np.sum(eigvals))
            blocks, diagnostics = adaptive_spectral_blocks(
                eigvals,
                min_segment_length=args.min_segment_length,
                max_segments=args.max_segments,
            )
        except Exception as exc:  # noqa: BLE001 - diagnostic records failures.
            rows.append(
                {
                    **base,
                    "weighting": weighting,
                    "block_name": pd.NA,
                    "status": "failed_spectrum",
                    "error": repr(exc),
                }
            )
            continue

        for component_index, eigenvalue in enumerate(eigvals, start=1):
            spectrum_rows.append(
                {
                    **base,
                    "weighting": weighting,
                    "component": int(component_index),
                    "eigenvalue": float(eigenvalue),
                    "fraction_of_kept_operator_energy": (
                        float(eigenvalue / total_energy) if total_energy > 0 else math.nan
                    ),
                }
            )

        for block in blocks:
            print(f"[{weighting}] {block.block_name}", flush=True)
            block_energy = (
                float(np.sum(eigvals[block.block_start - 1 : block.block_end]) / total_energy)
                if total_energy > 0
                else math.nan
            )
            block_record = {
                **base,
                "weighting": weighting,
                "block_id": int(block.block_id),
                "block_name": block.block_name,
                "block_type": block.block_type,
                "block_start": int(block.block_start),
                "block_end": int(block.block_end),
                "subspace_dimensions": int(block.block_end - block.block_start + 1),
                "block_energy_fraction": block_energy,
                "segmentation_bic": diagnostics.get("bic", math.nan),
                "common_mode_isolated": diagnostics.get("common_mode_isolated", False),
            }
            block_rows.append(block_record)
            start_sec = time.perf_counter()
            try:
                assignments, decomposition, annotations_df = run_block_tree(
                    data=data,
                    feature_space=None,
                    eigvals=eigvals,
                    eigvecs=eigvecs,
                    block=block,
                    edge_alpha=args.edge_alpha,
                    sibling_alpha=args.sibling_alpha,
                    enforce_internal_support_thresholds=(args.enforce_internal_support_thresholds),
                )
                elapsed_sec = time.perf_counter() - start_sec
                assignment_path = (
                    assignments_dir
                    / f"{safe_name(weighting)}__{safe_name(block.block_name)}__assignments.csv"
                )
                assignments.to_csv(assignment_path)
                cluster_sizes = assignments["cluster_id"].value_counts().sort_index()
                for cluster_id, size in cluster_sizes.items():
                    cluster_size_rows.append(
                        {
                            **block_record,
                            "cluster_id": int(cluster_id),
                            "cluster_size": int(size),
                        }
                    )
                rows.append(
                    {
                        **block_record,
                        "status": "ok",
                        "n_clusters": int(len(cluster_sizes)),
                        "largest_cluster_fraction": float(cluster_sizes.max() / len(data)),
                        "singleton_fraction": float(
                            (cluster_sizes == 1).sum() / max(len(cluster_sizes), 1)
                        ),
                        "runtime_sec": float(elapsed_sec),
                        "decomposition_num_clusters": int(
                            decomposition.get("num_clusters", len(cluster_sizes))
                        ),
                        "sibling_test_method_counts": sibling_method_counts(annotations_df),
                        "assignments_path": str(assignment_path),
                        "error": "",
                    }
                )
            except Exception as exc:  # noqa: BLE001 - diagnostic records failures.
                rows.append(
                    {
                        **block_record,
                        "status": "failed_gate",
                        "n_clusters": pd.NA,
                        "largest_cluster_fraction": math.nan,
                        "singleton_fraction": math.nan,
                        "runtime_sec": float(time.perf_counter() - start_sec),
                        "decomposition_num_clusters": pd.NA,
                        "sibling_test_method_counts": "",
                        "assignments_path": "",
                        "error": repr(exc),
                    }
                )
            pd.DataFrame(rows).to_csv(output_dir / "matrix_kak_probe_summary.csv", index=False)

    summary = pd.DataFrame(rows)
    blocks_df = pd.DataFrame(block_rows)
    spectrum_df = pd.DataFrame(spectrum_rows)
    cluster_sizes_df = pd.DataFrame(cluster_size_rows)
    summary.to_csv(output_dir / "matrix_kak_probe_summary.csv", index=False)
    blocks_df.to_csv(output_dir / "matrix_kak_probe_blocks.csv", index=False)
    spectrum_df.to_csv(output_dir / "matrix_kak_probe_spectra.csv", index=False)
    cluster_sizes_df.to_csv(output_dir / "matrix_kak_probe_cluster_sizes.csv", index=False)

    status_counts = (
        summary["status"].value_counts(dropna=False).to_dict() if not summary.empty else {}
    )
    ok = summary[summary["status"].eq("ok")] if "status" in summary else pd.DataFrame()
    readme = [
        "# Adaptive Cosine/KAK Matrix Probe",
        "",
        f"Schema: `{MATRIX_SCHEMA_VERSION}`",
        f"Input: `{args.input}`",
        f"Rows x columns: `{data.shape[0]} x {data.shape[1]}`",
        f"Edge alpha: `{args.edge_alpha}`",
        f"Sibling alpha: `{args.sibling_alpha}`",
        (f"Internal support thresholds enforced: `{args.enforce_internal_support_thresholds}`"),
        f"Max rank: `{args.max_rank}`",
        f"Weightings: `{', '.join(args.weightings)}`",
        "",
        "## Status Counts",
        "",
        repr(status_counts),
        "",
        "## Compact Ok Rows",
        "",
        (
            ok[
                [
                    "weighting",
                    "block_name",
                    "block_type",
                    "block_start",
                    "block_end",
                    "block_energy_fraction",
                    "n_clusters",
                    "largest_cluster_fraction",
                    "singleton_fraction",
                ]
            ].to_string(index=False)
            if not ok.empty
            else "No ok rows."
        ),
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(f"Wrote adaptive cosine/KAK matrix probe: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
