"""Probe adaptive cosine/KAK spectral-block trees on benchmark cases.

This diagnostic mirrors the historical adaptive cosine spectral-block script:

1. build a sample-sample cosine operator from the benchmark matrix,
2. eigendecompose it,
3. segment the log eigenspectrum into adaptive blocks,
4. build one average-linkage tree per spectral block,
5. run the normal TreeDecomposition gates on the original benchmark matrix.

It intentionally writes one row per case/weighting/block so the selected-tree
geometry layer is visible instead of being collapsed into one benchmark score.
"""

from __future__ import annotations

import argparse
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from tree_break_selection.hierarchy_analysis.cluster_assignments import (
    build_sample_cluster_assignments,
)
from tree_break_selection.hierarchy_analysis.decomposition.gates.orchestrator import (
    run_gate_annotation_pipeline,
)
from tree_break_selection.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_EDGE_ALPHA,
    DEFAULT_SIBLING_ALPHA,
)
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.inflation_correction.types.inflation_model import (
    DEFAULT_INTERNAL_SUPPORT_THRESHOLDS,
    CalibrationSupportThresholds,
)
from tree_break_selection.space_separation import (
    SpectralBlock,
    coordinates_for_block,
    separate_adaptive_cosine_space,
)
from tree_break_selection.tree.construction import tree_from_linkage
from tree_break_selection.tree.feature_space import FeatureSpace

from benchmarks.shared.cases import get_test_cases_by_suite
from benchmarks.shared.util.case_inputs import prepare_case_inputs

SCHEMA_VERSION = "adaptive_cosine_kak_benchmark_probe/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run adaptive cosine/KAK spectral-block trees on benchmark cases."
    )
    parser.add_argument(
        "--suite",
        default="method_proof",
        choices=[
            "full",
            "binary",
            "categorical",
            "continuous",
            "discretized_gaussian",
            "graph",
            "method_proof",
        ],
    )
    parser.add_argument("--case-limit", type=int, default=0, help="0 means all cases.")
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
    parser.add_argument("--max-rank", type=int, default=40)
    parser.add_argument("--min-segment-length", type=int, default=4)
    parser.add_argument("--max-segments", type=int, default=8)
    parser.add_argument(
        "--weightings",
        nargs="+",
        default=["binary"],
        choices=["binary", "tfidf"],
        help="'binary' means raw matrix values, matching the historical script name.",
    )
    return parser.parse_args()


def default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    return Path("benchmarks/results/diagnostics") / f"adaptive_cosine_kak_probe_{stamp}"


def labels_from_assignments(assignments: pd.DataFrame, sample_index: pd.Index) -> np.ndarray:
    aligned = assignments.loc[sample_index]
    return aligned["cluster_id"].astype(int).to_numpy()


def run_block_tree(
    *,
    data: pd.DataFrame,
    feature_space: FeatureSpace | None,
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    block: SpectralBlock,
    edge_alpha: float,
    sibling_alpha: float,
    enforce_internal_support_thresholds: bool = False,
    internal_support_thresholds: CalibrationSupportThresholds = (
        DEFAULT_INTERNAL_SUPPORT_THRESHOLDS
    ),
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    coords = coordinates_for_block(eigvals, eigvecs, block)
    if coords.shape[1] == 0 or not np.isfinite(coords).all():
        raise ValueError("invalid spectral block coordinates")
    distances = pdist(coords, metric="euclidean")
    if not np.isfinite(distances).all() or np.allclose(distances, 0.0):
        raise ValueError("degenerate spectral block distances")

    linkage_matrix = linkage(distances, method="average")
    tree = tree_from_linkage(linkage_matrix, leaf_names=data.index.tolist())
    tree.populate_node_divergences(data, feature_space=feature_space)
    gate_bundle = run_gate_annotation_pipeline(
        tree,
        tree.annotations_df.copy(),
        edge_alpha=edge_alpha,
        sibling_alpha=sibling_alpha,
        leaf_data=data,
        feature_space=feature_space,
        enforce_internal_support_thresholds=enforce_internal_support_thresholds,
        internal_support_thresholds=internal_support_thresholds,
    )
    decomposition = tree.decompose(
        gate_annotation_bundle=gate_bundle,
        leaf_data=data,
        edge_alpha=edge_alpha,
        sibling_alpha=sibling_alpha,
        feature_space=feature_space,
        enforce_internal_support_thresholds=enforce_internal_support_thresholds,
        internal_support_thresholds=internal_support_thresholds,
    )
    assignments = build_sample_cluster_assignments(decomposition).loc[data.index]
    return assignments, decomposition, tree.annotations_df


def sibling_method_counts(annotations_df: pd.DataFrame) -> str:
    if "Sibling_Test_Method" not in annotations_df.columns:
        return ""
    counts = (
        annotations_df["Sibling_Test_Method"]
        .dropna()
        .astype(str)
        .loc[lambda values: values.str.len() > 0]
        .value_counts()
        .sort_index()
    )
    return ";".join(f"{method}:{int(count)}" for method, count in counts.items())


def run_case(
    *,
    case_num: int,
    case: dict[str, object],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    inputs = prepare_case_inputs(case, selected_methods=[])
    data = inputs.data
    true_labels = inputs.labels
    metadata = inputs.metadata
    feature_space = metadata.get("feature_space")
    if feature_space is not None and not isinstance(feature_space, FeatureSpace):
        feature_space = None

    base = {
        "schema_version": SCHEMA_VERSION,
        "case_num": int(case_num),
        "case_name": str(case["name"]),
        "category": str(case.get("category", "")),
        "generator": str(case["generator"]),
        "n_samples": int(data.shape[0]),
        "n_features": int(data.shape[1]),
        "n_true_clusters": int(len(np.unique(true_labels))),
        "edge_alpha": float(args.edge_alpha),
        "sibling_alpha": float(args.sibling_alpha),
        "enforce_internal_support_thresholds": bool(args.enforce_internal_support_thresholds),
    }

    rows: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []
    spectrum_rows: list[dict[str, object]] = []

    for weighting in args.weightings:
        try:
            space = separate_adaptive_cosine_space(
                data,
                weighting=weighting,
                max_rank=args.max_rank,
                min_segment_length=args.min_segment_length,
                max_segments=args.max_segments,
            )
            eigvals = space.eigenvalues
            eigvecs = space.eigenvectors
            total_energy = float(np.sum(eigvals))
            blocks = space.blocks
            diagnostics = space.diagnostics
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

            start = time.perf_counter()
            try:
                assignments, decomposition, annotations_df = run_block_tree(
                    data=data,
                    feature_space=feature_space,
                    eigvals=eigvals,
                    eigvecs=eigvecs,
                    block=block,
                    edge_alpha=args.edge_alpha,
                    sibling_alpha=args.sibling_alpha,
                    enforce_internal_support_thresholds=(args.enforce_internal_support_thresholds),
                )
                elapsed = time.perf_counter() - start
                labels = labels_from_assignments(assignments, data.index)
                cluster_sizes = pd.Series(labels).value_counts()
                n_clusters = int(len(cluster_sizes))
                rows.append(
                    {
                        **block_record,
                        "status": "ok",
                        "n_clusters": n_clusters,
                        "largest_cluster_fraction": float(cluster_sizes.max() / len(labels)),
                        "singleton_fraction": float(
                            (cluster_sizes == 1).sum() / max(n_clusters, 1)
                        ),
                        "ari": float(adjusted_rand_score(true_labels, labels)),
                        "nmi": float(normalized_mutual_info_score(true_labels, labels)),
                        "runtime_sec": float(elapsed),
                        "decomposition_num_clusters": int(
                            decomposition.get("num_clusters", n_clusters)
                        ),
                        "sibling_test_method_counts": sibling_method_counts(annotations_df),
                        "error": "",
                    }
                )
            except Exception as exc:  # noqa: BLE001 - diagnostic records failures.
                elapsed = time.perf_counter() - start
                rows.append(
                    {
                        **block_record,
                        "status": "failed_gate",
                        "n_clusters": pd.NA,
                        "largest_cluster_fraction": math.nan,
                        "singleton_fraction": math.nan,
                        "ari": math.nan,
                        "nmi": math.nan,
                        "runtime_sec": float(elapsed),
                        "decomposition_num_clusters": pd.NA,
                        "sibling_test_method_counts": "",
                        "error": repr(exc),
                    }
                )

    return rows, block_rows, spectrum_rows


def write_summaries(rows: pd.DataFrame, output_dir: Path) -> None:
    ok = rows[rows["status"].eq("ok")].copy()
    if ok.empty:
        pd.DataFrame().to_csv(output_dir / "kak_benchmark_probe_summary_by_case.csv", index=False)
        pd.DataFrame().to_csv(output_dir / "kak_benchmark_probe_summary_by_block.csv", index=False)
        return

    by_case = (
        ok.sort_values(["case_name", "ari", "nmi"], ascending=[True, False, False])
        .groupby(["case_num", "case_name", "category", "generator"], as_index=False)
        .head(1)
    )
    by_case.to_csv(output_dir / "kak_benchmark_probe_summary_by_case.csv", index=False)

    by_block = (
        ok.groupby(["weighting", "block_type"], as_index=False)
        .agg(
            n_rows=("status", "size"),
            mean_ari=("ari", "mean"),
            median_ari=("ari", "median"),
            mean_nmi=("nmi", "mean"),
            mean_n_clusters=("n_clusters", "mean"),
            mean_largest_cluster_fraction=("largest_cluster_fraction", "mean"),
        )
        .sort_values(["mean_ari", "mean_nmi"], ascending=False)
    )
    by_block.to_csv(output_dir / "kak_benchmark_probe_summary_by_block.csv", index=False)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = get_test_cases_by_suite(args.suite)
    if args.case_limit > 0:
        cases = cases[: args.case_limit]

    all_rows: list[dict[str, object]] = []
    all_block_rows: list[dict[str, object]] = []
    all_spectrum_rows: list[dict[str, object]] = []
    for case_num, case in enumerate(cases, start=1):
        print(f"[{case_num}/{len(cases)}] {case['name']} ({case['generator']})", flush=True)
        rows, block_rows, spectrum_rows = run_case(case_num=case_num, case=case, args=args)
        all_rows.extend(rows)
        all_block_rows.extend(block_rows)
        all_spectrum_rows.extend(spectrum_rows)
        pd.DataFrame(all_rows).to_csv(output_dir / "kak_benchmark_probe_rows.csv", index=False)

    rows_df = pd.DataFrame(all_rows)
    blocks_df = pd.DataFrame(all_block_rows)
    spectrum_df = pd.DataFrame(all_spectrum_rows)
    rows_df.to_csv(output_dir / "kak_benchmark_probe_rows.csv", index=False)
    blocks_df.to_csv(output_dir / "kak_benchmark_probe_blocks.csv", index=False)
    spectrum_df.to_csv(output_dir / "kak_benchmark_probe_spectra.csv", index=False)
    write_summaries(rows_df, output_dir)

    status_counts = (
        rows_df["status"].value_counts(dropna=False).to_dict() if not rows_df.empty else {}
    )
    ok = rows_df[rows_df["status"].eq("ok")] if "status" in rows_df else pd.DataFrame()
    readme = [
        "# Adaptive Cosine/KAK Benchmark Probe",
        "",
        f"Schema: `{SCHEMA_VERSION}`",
        f"Suite: `{args.suite}`",
        f"Cases: `{len(cases)}`",
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
        "## Best Rows",
        "",
        (
            ok.sort_values(["ari", "nmi"], ascending=False).head(20).to_string(index=False)
            if not ok.empty
            else "No ok rows."
        ),
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(f"Wrote adaptive cosine/KAK benchmark probe: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
