"""Run diffusion trees inside adaptive cosine/KAK separated spaces.

For each adaptive cosine/KAK block this diagnostic:

1. computes the existing block coordinates,
2. builds a k-NN diffusion operator in that separated coordinate space,
3. builds an average-linkage tree from diffusion distance, and
4. runs the normal Tree-Break Selection gate pipeline on the original feature matrix.

The result is diagnostic-only. It changes tree geometry, not the production
edge/sibling test surface.
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
    adaptive_spectral_blocks,
    block_adaptive_diffusion_distance,
    block_diffusion_distance,
    coordinates_for_block,
    cosine_eigendecomposition,
    weight_feature_matrix,
)
from tree_break_selection.tree.poset_tree import PosetTree

from benchmarks.diagnostics.spectral.adaptive_cosine_kak_benchmark_probe import (
    SCHEMA_VERSION,
    sibling_method_counts,
)
from benchmarks.diagnostics.spectral.adaptive_cosine_kak_matrix_probe import (
    load_matrix,
    safe_name,
)

DIFFUSION_SCHEMA_VERSION = f"{SCHEMA_VERSION}/matrix_block_diffusion"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Tree-Break Selection diffusion trees inside adaptive cosine/KAK blocks."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/feature_matrices/feature_matrix_julia_GOCC_GOBP_GOMF_combined.tsv"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--reference-labels", type=Path, default=None)
    parser.add_argument("--edge-alpha", type=float, default=DEFAULT_EDGE_ALPHA)
    parser.add_argument("--sibling-alpha", type=float, default=DEFAULT_SIBLING_ALPHA)
    parser.add_argument("--max-rank", type=int, default=80)
    parser.add_argument("--min-segment-length", type=int, default=4)
    parser.add_argument("--max-segments", type=int, default=8)
    parser.add_argument("--diffusion-k-neighbors", type=int, default=15)
    parser.add_argument("--diffusion-time", type=int, default=3)
    parser.add_argument("--diffusion-components", type=int, default=30)
    parser.add_argument(
        "--diffusion-mode",
        choices=("fixed", "adaptive"),
        default="fixed",
        help="Use the local fixed k-NN Gaussian kernel or the pydiffmap adaptive kernel.",
    )
    parser.add_argument("--adaptive-bandwidth-type", default="-1/(d+2)")
    parser.add_argument("--adaptive-epsilon", default="median")
    parser.add_argument("--adaptive-metric", choices=("euclidean",), default="euclidean")
    parser.add_argument(
        "--weightings",
        nargs="+",
        default=["binary", "tfidf"],
        choices=["binary", "tfidf"],
    )
    parser.add_argument(
        "--enforce-internal-support-thresholds",
        action="store_true",
        help="Fail closed when internal empirical-null support is below thresholds.",
    )
    return parser.parse_args()


def default_output_dir(input_path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    return (
        Path("benchmarks/results/diagnostics")
        / f"adaptive_cosine_kak_diffusion_matrix_probe_{input_path.stem}_{stamp}"
    )


def run_block_diffusion_tree(
    *,
    data: pd.DataFrame,
    coords: np.ndarray,
    edge_alpha: float,
    sibling_alpha: float,
    diffusion_k_neighbors: int,
    diffusion_time: int,
    diffusion_components: int,
    diffusion_mode: str,
    adaptive_bandwidth_type: str | float | None,
    adaptive_epsilon: str | float,
    adaptive_metric: str,
    enforce_internal_support_thresholds: bool,
    internal_support_thresholds: CalibrationSupportThresholds = (
        DEFAULT_INTERNAL_SUPPORT_THRESHOLDS
    ),
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame, dict[str, object]]:
    if diffusion_mode == "fixed":
        diffusion_distances, diffusion_metadata = block_diffusion_distance(
            coords,
            k_neighbors=diffusion_k_neighbors,
            diffusion_time=diffusion_time,
            n_components=diffusion_components,
        )
    elif diffusion_mode == "adaptive":
        diffusion_distances, diffusion_metadata = block_adaptive_diffusion_distance(
            coords,
            k_neighbors=diffusion_k_neighbors,
            diffusion_time=diffusion_time,
            n_components=diffusion_components,
            metric=adaptive_metric,
            bandwidth_type=adaptive_bandwidth_type,
            epsilon=adaptive_epsilon,
        )
    else:
        raise ValueError(f"Unknown diffusion mode: {diffusion_mode!r}")
    diffusion_metadata["diffusion_mode"] = diffusion_mode
    linkage_matrix = linkage(diffusion_distances, method="average")
    tree = PosetTree.from_linkage(linkage_matrix, leaf_names=data.index.tolist())
    tree.populate_node_divergences(data)
    gate_bundle = run_gate_annotation_pipeline(
        tree,
        tree.annotations_df.copy(),
        edge_alpha=edge_alpha,
        sibling_alpha=sibling_alpha,
        leaf_data=data,
        enforce_internal_support_thresholds=enforce_internal_support_thresholds,
        internal_support_thresholds=internal_support_thresholds,
    )
    decomposition = tree.decompose(
        gate_annotation_bundle=gate_bundle,
        leaf_data=data,
        edge_alpha=edge_alpha,
        sibling_alpha=sibling_alpha,
        enforce_internal_support_thresholds=enforce_internal_support_thresholds,
        internal_support_thresholds=internal_support_thresholds,
    )
    assignments = build_sample_cluster_assignments(decomposition).loc[data.index]
    return assignments, decomposition, tree.annotations_df, diffusion_metadata


def load_reference_labels(path: Path | None, index: pd.Index) -> pd.DataFrame | None:
    if path is None:
        return None
    labels = pd.read_csv(path)
    if "gene" not in labels.columns:
        raise ValueError("Reference labels CSV must contain a 'gene' column.")
    labels = labels.set_index("gene").reindex(index)
    labels["matched_reference_label"] = labels["reference_cluster_id"].notna()
    return labels


def score_reference(
    assignments: pd.DataFrame, reference_labels: pd.DataFrame | None
) -> dict[str, object]:
    if reference_labels is None:
        return {
            "reference_matched_genes": pd.NA,
            "reference_unmatched_genes": pd.NA,
            "reference_cluster_count_observed": pd.NA,
            "reference_ari": math.nan,
            "reference_nmi": math.nan,
        }

    aligned = reference_labels.reindex(assignments.index)
    matched = aligned["matched_reference_label"].fillna(False).astype(bool)
    if not bool(matched.any()):
        return {
            "reference_matched_genes": 0,
            "reference_unmatched_genes": int(len(aligned)),
            "reference_cluster_count_observed": 0,
            "reference_ari": math.nan,
            "reference_nmi": math.nan,
        }

    y_true = aligned.loc[matched, "reference_cluster_id"].astype(int).to_numpy()
    y_pred = assignments.loc[matched, "cluster_id"].astype(int).to_numpy()
    return {
        "reference_matched_genes": int(matched.sum()),
        "reference_unmatched_genes": int((~matched).sum()),
        "reference_cluster_count_observed": int(pd.Series(y_true).nunique()),
        "reference_ari": float(adjusted_rand_score(y_true, y_pred)),
        "reference_nmi": float(normalized_mutual_info_score(y_true, y_pred)),
    }


def write_reference_ranking(summary: pd.DataFrame, output_dir: Path) -> None:
    cols = [
        "display_order_rank",
        "run_id",
        "weighting",
        "block_name",
        "n_clusters",
        "singleton_fraction",
        "reference_matched_genes",
        "reference_cluster_count_observed",
        "reference_ari",
        "reference_nmi",
    ]
    ok = summary[summary["status"].eq("ok")].copy()
    if ok.empty or "reference_nmi" not in ok.columns:
        return
    ok = ok.sort_values(["reference_nmi", "reference_ari"], ascending=False)
    ok.insert(0, "display_order_rank", np.arange(1, len(ok) + 1))
    ok[cols].to_csv(output_dir / "kak_diffusion_reference_nmi_metrics.csv", index=False)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    frame = ok.dropna(subset=["reference_nmi"]).sort_values("reference_nmi", ascending=True)
    if frame.empty:
        return
    fig_height = max(4.8, 0.42 * len(frame) + 1.8)
    fig, ax = plt.subplots(figsize=(11.2, fig_height))
    labels = frame["run_id"].astype(str).to_list()
    bars = ax.barh(labels, frame["reference_nmi"], color="#4c78a8", alpha=0.86)
    ax.set_xlabel("NMI against Julia reference endotypes")
    ax.set_ylabel("Separated-space diffusion tree")
    ax.set_title("Separated KAK/cosine spaces with diffusion trees")
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlim(0.0, max(1.0, float(frame["reference_nmi"].max()) * 1.08))
    for bar, nmi, clusters in zip(bars, frame["reference_nmi"], frame["n_clusters"], strict=False):
        ax.text(
            float(nmi) + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{float(nmi):.3f} / {int(clusters)} clusters",
            va="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(output_dir / "kak_diffusion_reference_nmi_ranking.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or default_output_dir(args.input)
    assignments_dir = output_dir / "assignments"
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments_dir.mkdir(parents=True, exist_ok=True)

    data = load_matrix(args.input)
    reference_labels = load_reference_labels(args.reference_labels, data.index)
    base = {
        "schema_version": DIFFUSION_SCHEMA_VERSION,
        "input_path": str(args.input),
        "n_samples": int(data.shape[0]),
        "n_features": int(data.shape[1]),
        "edge_alpha": float(args.edge_alpha),
        "sibling_alpha": float(args.sibling_alpha),
        "enforce_internal_support_thresholds": bool(args.enforce_internal_support_thresholds),
        "diffusion_k_neighbors": int(args.diffusion_k_neighbors),
        "diffusion_time": int(args.diffusion_time),
        "diffusion_components": int(args.diffusion_components),
        "diffusion_mode": args.diffusion_mode,
        "adaptive_bandwidth_type": args.adaptive_bandwidth_type,
        "adaptive_epsilon": args.adaptive_epsilon,
        "adaptive_metric": args.adaptive_metric,
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
        except Exception as exc:  # noqa: BLE001 - diagnostic rows preserve failures.
            rows.append(
                {**base, "weighting": weighting, "status": "failed_spectrum", "error": repr(exc)}
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
            print(f"[{weighting}] diffusion {block.block_name}", flush=True)
            block_energy = (
                float(np.sum(eigvals[block.block_start - 1 : block.block_end]) / total_energy)
                if total_energy > 0
                else math.nan
            )
            run_id = f"{weighting}__{block.block_name}"
            block_record = {
                **base,
                "run_id": run_id,
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
                coords = coordinates_for_block(eigvals, eigvecs, block)
                assignments, decomposition, annotations_df, diffusion_metadata = (
                    run_block_diffusion_tree(
                        data=data,
                        coords=coords,
                        edge_alpha=args.edge_alpha,
                        sibling_alpha=args.sibling_alpha,
                        diffusion_k_neighbors=args.diffusion_k_neighbors,
                        diffusion_time=args.diffusion_time,
                        diffusion_components=args.diffusion_components,
                        diffusion_mode=args.diffusion_mode,
                        adaptive_bandwidth_type=args.adaptive_bandwidth_type,
                        adaptive_epsilon=args.adaptive_epsilon,
                        adaptive_metric=args.adaptive_metric,
                        enforce_internal_support_thresholds=(
                            args.enforce_internal_support_thresholds
                        ),
                    )
                )
                assignment_path = (
                    assignments_dir
                    / f"{safe_name(weighting)}__{safe_name(block.block_name)}__diffusion_assignments.csv"
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
                        **score_reference(assignments, reference_labels),
                        "status": "ok",
                        "n_clusters": int(len(cluster_sizes)),
                        "largest_cluster_fraction": float(cluster_sizes.max() / len(data)),
                        "singleton_fraction": float(
                            (cluster_sizes == 1).sum() / max(len(cluster_sizes), 1)
                        ),
                        "runtime_sec": float(time.perf_counter() - start_sec),
                        "decomposition_num_clusters": int(
                            decomposition.get("num_clusters", len(cluster_sizes))
                        ),
                        "sibling_test_method_counts": sibling_method_counts(annotations_df),
                        "diffusion_metadata": repr(diffusion_metadata),
                        "assignments_path": str(assignment_path),
                        "error": "",
                    }
                )
            except Exception as exc:  # noqa: BLE001 - diagnostic rows preserve failures.
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
                        "diffusion_metadata": "",
                        "assignments_path": "",
                        "error": repr(exc),
                    }
                )
            pd.DataFrame(rows).to_csv(
                output_dir / "matrix_kak_diffusion_probe_summary.csv",
                index=False,
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "matrix_kak_diffusion_probe_summary.csv", index=False)
    pd.DataFrame(block_rows).to_csv(
        output_dir / "matrix_kak_diffusion_probe_blocks.csv",
        index=False,
    )
    pd.DataFrame(spectrum_rows).to_csv(
        output_dir / "matrix_kak_diffusion_probe_spectra.csv",
        index=False,
    )
    pd.DataFrame(cluster_size_rows).to_csv(
        output_dir / "matrix_kak_diffusion_probe_cluster_sizes.csv",
        index=False,
    )
    write_reference_ranking(summary, output_dir)

    status_counts = (
        summary["status"].value_counts(dropna=False).to_dict() if not summary.empty else {}
    )
    ok = summary[summary["status"].eq("ok")] if "status" in summary else pd.DataFrame()
    compact_cols = [
        "weighting",
        "block_name",
        "n_clusters",
        "largest_cluster_fraction",
        "singleton_fraction",
        "reference_ari",
        "reference_nmi",
    ]
    readme = [
        "# Adaptive Cosine/KAK Separated-Space Diffusion Probe",
        "",
        f"Schema: `{DIFFUSION_SCHEMA_VERSION}`",
        f"Input: `{args.input}`",
        f"Rows x columns: `{data.shape[0]} x {data.shape[1]}`",
        f"Reference labels: `{args.reference_labels}`",
        f"Edge alpha: `{args.edge_alpha}`",
        f"Sibling alpha: `{args.sibling_alpha}`",
        f"Diffusion k: `{args.diffusion_k_neighbors}`",
        f"Diffusion time: `{args.diffusion_time}`",
        f"Diffusion components: `{args.diffusion_components}`",
        f"Diffusion mode: `{args.diffusion_mode}`",
        f"Adaptive bandwidth type: `{args.adaptive_bandwidth_type}`",
        f"Adaptive epsilon: `{args.adaptive_epsilon}`",
        f"Weightings: `{', '.join(args.weightings)}`",
        "",
        "## Status Counts",
        "",
        repr(status_counts),
        "",
        "## Compact Ok Rows",
        "",
        ok[compact_cols].to_string(index=False) if not ok.empty else "No ok rows.",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(f"Wrote separated-space diffusion probe: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
