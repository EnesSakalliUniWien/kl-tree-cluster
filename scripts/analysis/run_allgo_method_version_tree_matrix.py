#!/usr/bin/env python3
"""Run allGO tree-geometry examples with current and legacy gates.

The rows separate two axes that were easy to conflate:

* ``method_version``: current TBS gates versus the c2ef9a69 legacy gates.
* ``tree_geometry``: whole-matrix adaptive diffusion, raw cosine subspace, or
  adaptive diffusion inside a cosine subspace.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from benchmarks.diagnostics.spectral.adaptive_cosine_kak_benchmark_probe import (
    adaptive_spectral_blocks,
    coords_for_block,
    cosine_eigendecomposition,
    weighted_matrix,
)
from benchmarks.diagnostics.spectral.adaptive_cosine_kak_diffusion_matrix_probe import (
    block_adaptive_diffusion_distance,
)
from benchmarks.diagnostics.spectral.adaptive_cosine_kak_matrix_probe import load_matrix
from benchmarks.shared.runners.legacy_commit_runner import (
    _run_legacy_c2ef9a69_tbs_method,
)
from benchmarks.shared.runners.tbs_diffusion_runner import _build_adaptive_diffusion_distance
from benchmarks.shared.runners.tbs_runner import _run_tbs_on_distance
from benchmarks.shared.types.method_run_result import MethodRunResult
from scipy.spatial.distance import pdist
from tree_break_selection.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_EDGE_ALPHA,
    DEFAULT_SIBLING_ALPHA,
)

METHOD_VERSIONS = ("legacy_c2ef", "current")
TREE_GEOMETRIES = (
    "whole_adaptive_diffusion",
    "raw_cosine_subspace",
    "adaptive_diffusion_cosine_subspace",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--edge-alpha", type=float, default=DEFAULT_EDGE_ALPHA)
    parser.add_argument("--sibling-alpha", type=float, default=DEFAULT_SIBLING_ALPHA)
    parser.add_argument("--max-rank", type=int, default=80)
    parser.add_argument("--min-segment-length", type=int, default=4)
    parser.add_argument("--max-segments", type=int, default=8)
    parser.add_argument(
        "--weightings",
        nargs="+",
        default=["binary", "tfidf"],
        choices=["binary", "tfidf"],
    )
    parser.add_argument("--diffusion-k-neighbors", type=int, default=15)
    parser.add_argument("--diffusion-time", type=int, default=3)
    parser.add_argument("--diffusion-components", type=int, default=30)
    parser.add_argument("--adaptive-bandwidth-type", default="-1/(d+2)")
    parser.add_argument("--adaptive-epsilon", default="median")
    parser.add_argument("--adaptive-metric", default="euclidean")
    parser.add_argument(
        "--block-names",
        nargs="*",
        default=None,
        help="Optional block-name allowlist for cosine-subspace geometries.",
    )
    parser.add_argument(
        "--method-versions",
        nargs="+",
        default=list(METHOD_VERSIONS),
        choices=METHOD_VERSIONS,
    )
    parser.add_argument(
        "--tree-geometries",
        nargs="+",
        default=list(TREE_GEOMETRIES),
        choices=TREE_GEOMETRIES,
    )
    return parser.parse_args()


def safe_name(value: object) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))


def run_method(
    *,
    method_version: str,
    data: pd.DataFrame,
    distances: np.ndarray,
    edge_alpha: float,
    sibling_alpha: float,
) -> MethodRunResult:
    if method_version == "legacy_c2ef":
        return _run_legacy_c2ef9a69_tbs_method(
            data,
            distances,
            sibling_alpha,
            tree_linkage_method="average",
            edge_alpha=edge_alpha,
        )
    if method_version == "current":
        return _run_tbs_on_distance(
            data,
            distances,
            sibling_alpha,
            tree_linkage_method="average",
            edge_alpha=edge_alpha,
        )
    raise ValueError(f"Unknown method_version: {method_version!r}")


def write_assignments(
    *,
    labels: np.ndarray,
    index: pd.Index,
    output_path: Path,
) -> pd.Series:
    cluster_ids = pd.Series(labels.astype(int), index=index.astype(str), name="cluster_id")
    sizes = cluster_ids.value_counts().sort_index()
    assignments = pd.DataFrame({"gene": cluster_ids.index, "cluster_id": cluster_ids.to_numpy()})
    assignments["cluster_size"] = assignments["cluster_id"].map(sizes).astype(int)
    assignments.to_csv(output_path, index=False)
    return sizes


def row_from_result(
    *,
    base: dict[str, object],
    result: MethodRunResult,
    assignments_path: Path,
    elapsed_sec: float,
    cluster_sizes: pd.Series,
) -> dict[str, object]:
    return {
        **base,
        "status": result.status,
        "skip_reason": result.skip_reason or "",
        "n_clusters": int(len(cluster_sizes)),
        "largest_cluster_fraction": float(cluster_sizes.max() / cluster_sizes.sum()),
        "singleton_fraction": float((cluster_sizes == 1).sum() / max(len(cluster_sizes), 1)),
        "singleton_gene_fraction": float((cluster_sizes == 1).sum() / cluster_sizes.sum()),
        "runtime_sec": float(elapsed_sec),
        "assignments_path": str(assignments_path),
        "error": "",
    }


def append_failure(
    rows: list[dict[str, object]],
    *,
    base: dict[str, object],
    start_sec: float,
    exc: Exception,
) -> None:
    rows.append(
        {
            **base,
            "status": "failed",
            "skip_reason": "",
            "n_clusters": pd.NA,
            "largest_cluster_fraction": math.nan,
            "singleton_fraction": math.nan,
            "singleton_gene_fraction": math.nan,
            "runtime_sec": float(time.perf_counter() - start_sec),
            "assignments_path": "",
            "error": repr(exc),
        }
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assignments_dir = args.output_dir / "assignments"
    assignments_dir.mkdir(parents=True, exist_ok=True)

    data = load_matrix(args.input)
    base_common = {
        "input_path": str(args.input),
        "n_samples": int(data.shape[0]),
        "n_features": int(data.shape[1]),
        "edge_alpha": float(args.edge_alpha),
        "sibling_alpha": float(args.sibling_alpha),
        "diffusion_k_neighbors": int(args.diffusion_k_neighbors),
        "diffusion_time": int(args.diffusion_time),
        "diffusion_components": int(args.diffusion_components),
        "adaptive_bandwidth_type": args.adaptive_bandwidth_type,
        "adaptive_epsilon": args.adaptive_epsilon,
        "adaptive_metric": args.adaptive_metric,
    }
    rows: list[dict[str, object]] = []
    cluster_size_rows: list[dict[str, object]] = []

    if "whole_adaptive_diffusion" in args.tree_geometries:
        print("[whole_adaptive_diffusion] distance", flush=True)
        whole_distances, whole_metadata = _build_adaptive_diffusion_distance(
            data,
            k_neighbors=args.diffusion_k_neighbors,
            diffusion_time=args.diffusion_time,
            n_components=args.diffusion_components,
            metric="hamming",
            bandwidth_type=args.adaptive_bandwidth_type,
            epsilon=args.adaptive_epsilon,
            return_metadata=True,
        )
        for method_version in args.method_versions:
            run_id = f"{method_version}__whole_adaptive_diffusion"
            family = f"{method_version}__whole_adaptive_diffusion"
            base = {
                **base_common,
                "method_version": method_version,
                "tree_geometry": "whole_adaptive_diffusion",
                "method_family": family,
                "run_id": run_id,
                "weighting": "whole",
                "block_name": "whole_matrix",
                "block_type": "whole_matrix",
                "block_start": pd.NA,
                "block_end": pd.NA,
                "subspace_dimensions": pd.NA,
                "block_energy_fraction": math.nan,
                "segmentation_bic": math.nan,
                "diffusion_mode": "adaptive",
                "diffusion_metadata": repr(whole_metadata),
            }
            print(f"[{run_id}] gates", flush=True)
            start_sec = time.perf_counter()
            try:
                result = run_method(
                    method_version=method_version,
                    data=data,
                    distances=whole_distances,
                    edge_alpha=args.edge_alpha,
                    sibling_alpha=args.sibling_alpha,
                )
                if result.labels is None:
                    raise RuntimeError("Method returned no labels.")
                path = assignments_dir / f"{safe_name(run_id)}__assignments.csv"
                cluster_sizes = write_assignments(
                    labels=result.labels,
                    index=data.index,
                    output_path=path,
                )
                rows.append(
                    row_from_result(
                        base=base,
                        result=result,
                        assignments_path=path,
                        elapsed_sec=time.perf_counter() - start_sec,
                        cluster_sizes=cluster_sizes,
                    )
                )
                for cluster_id, size in cluster_sizes.items():
                    cluster_size_rows.append({**base, "cluster_id": int(cluster_id), "cluster_size": int(size)})
            except Exception as exc:  # noqa: BLE001 - diagnostic matrix records failures.
                append_failure(rows, base=base, start_sec=start_sec, exc=exc)

    for weighting in args.weightings:
        if not set(args.tree_geometries).intersection(
            {"raw_cosine_subspace", "adaptive_diffusion_cosine_subspace"}
        ):
            continue
        print(f"[{weighting}] cosine eigendecomposition", flush=True)
        values = weighted_matrix(data, weighting)
        eigvals, eigvecs = cosine_eigendecomposition(values, args.max_rank)
        total_energy = float(np.sum(eigvals))
        blocks, diagnostics = adaptive_spectral_blocks(
            eigvals,
            min_segment_length=args.min_segment_length,
            max_segments=args.max_segments,
        )

        for block in blocks:
            if args.block_names is not None and block.block_name not in set(args.block_names):
                continue
            coords = coords_for_block(eigvals, eigvecs, block)
            block_energy = (
                float(np.sum(eigvals[block.block_start - 1 : block.block_end]) / total_energy)
                if total_energy > 0
                else math.nan
            )
            geometry_distances: dict[str, tuple[np.ndarray, dict[str, object]]] = {}
            if "raw_cosine_subspace" in args.tree_geometries:
                geometry_distances["raw_cosine_subspace"] = (
                    pdist(coords, metric="euclidean"),
                    {"kernel": "none", "metric": "euclidean"},
                )
            if "adaptive_diffusion_cosine_subspace" in args.tree_geometries:
                distances, metadata = block_adaptive_diffusion_distance(
                    coords,
                    k_neighbors=args.diffusion_k_neighbors,
                    diffusion_time=args.diffusion_time,
                    n_components=args.diffusion_components,
                    metric=args.adaptive_metric,
                    bandwidth_type=args.adaptive_bandwidth_type,
                    epsilon=args.adaptive_epsilon,
                )
                geometry_distances["adaptive_diffusion_cosine_subspace"] = (
                    distances,
                    {"kernel": "pydiffmap_adaptive", **metadata, "diffusion_mode": "adaptive"},
                )

            for tree_geometry, (distances, metadata) in geometry_distances.items():
                for method_version in args.method_versions:
                    run_id = f"{method_version}__{tree_geometry}__{weighting}__{block.block_name}"
                    family = f"{method_version}__{tree_geometry}"
                    base = {
                        **base_common,
                        "method_version": method_version,
                        "tree_geometry": tree_geometry,
                        "method_family": family,
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
                        "diffusion_mode": metadata.get("diffusion_mode", "none"),
                        "diffusion_metadata": repr(metadata),
                    }
                    print(f"[{run_id}] gates", flush=True)
                    start_sec = time.perf_counter()
                    try:
                        result = run_method(
                            method_version=method_version,
                            data=data,
                            distances=distances,
                            edge_alpha=args.edge_alpha,
                            sibling_alpha=args.sibling_alpha,
                        )
                        if result.labels is None:
                            raise RuntimeError("Method returned no labels.")
                        path = assignments_dir / f"{safe_name(run_id)}__assignments.csv"
                        cluster_sizes = write_assignments(
                            labels=result.labels,
                            index=data.index,
                            output_path=path,
                        )
                        rows.append(
                            row_from_result(
                                base=base,
                                result=result,
                                assignments_path=path,
                                elapsed_sec=time.perf_counter() - start_sec,
                                cluster_sizes=cluster_sizes,
                            )
                        )
                        for cluster_id, size in cluster_sizes.items():
                            cluster_size_rows.append(
                                {
                                    **base,
                                    "cluster_id": int(cluster_id),
                                    "cluster_size": int(size),
                                }
                            )
                    except Exception as exc:  # noqa: BLE001 - diagnostic matrix records failures.
                        append_failure(rows, base=base, start_sec=start_sec, exc=exc)
                    pd.DataFrame(rows).to_csv(
                        args.output_dir / "method_tree_matrix_summary.csv",
                        index=False,
                    )

    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "method_tree_matrix_summary.csv", index=False)
    pd.DataFrame(cluster_size_rows).to_csv(
        args.output_dir / "method_tree_matrix_cluster_sizes.csv",
        index=False,
    )
    ok = summary[summary["status"].eq("ok")] if "status" in summary else pd.DataFrame()
    status_counts = summary["status"].value_counts(dropna=False).to_dict() if not summary.empty else {}
    readme = [
        "# allGO Method-Version x Tree-Geometry Matrix",
        "",
        f"Input: `{args.input}`",
        f"Rows x columns: `{data.shape[0]} x {data.shape[1]}`",
        "",
        "Axes:",
        "- `method_version`: `legacy_c2ef` or `current` gate/decomposition layer.",
        "- `tree_geometry`: `whole_adaptive_diffusion`, `raw_cosine_subspace`, or `adaptive_diffusion_cosine_subspace`.",
        "",
        "`raw_cosine_subspace` is the reader-facing name for the internal KAK/cosine eigenspace diagnostic.",
        "",
        "Status counts:",
        repr(status_counts),
        "",
        "Compact ok rows:",
        (
            ok[
                [
                    "method_version",
                    "tree_geometry",
                    "weighting",
                    "block_name",
                    "n_clusters",
                    "largest_cluster_fraction",
                    "singleton_gene_fraction",
                ]
            ].to_string(index=False)
            if not ok.empty
            else "No ok rows."
        ),
        "",
    ]
    (args.output_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(f"Wrote allGO method/tree matrix: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
