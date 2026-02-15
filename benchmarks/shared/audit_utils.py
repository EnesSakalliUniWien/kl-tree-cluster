"""Utilities for exporting diagnostic audit logs from hierarchical clustering runs.

This module handles the extraction, cleaning, and CSV export of statistical metadata
from tree decompositions, ensuring reports are human-readable and auditable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def export_decomposition_audit(
    computed_results: List[Dict[str, Any]],
    output_root: Path,
    verbose: bool = False,
) -> None:
    """Export detailed statistical and merge audit logs to CSV files.

    Parameters
    ----------
    computed_results
        List of result dictionaries from benchmark_cluster_algorithm.
        Expected to contain 'test_case_num', 'method_name', 'tree', 'stats',
        and optionally 'posthoc_merge_audit'.
    output_root
        Base directory for benchmark results (e.g. benchmarks/results).
        Logs will be saved in [output_root]/audit/.
    verbose
        If True, print status messages for each saved file.
    """
    audit_dir = output_root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    for res in computed_results:
        case_id = res.get("test_case_num", "unknown")
        method = res.get("method_name", "unknown").replace(" ", "_").lower()
        tree = res.get("tree")

        # 1. Export Post-hoc Merge Audit
        # Search in root or 'meta' for possible audit trail
        audit_trail = res.get("posthoc_merge_audit") or res.get("meta", {}).get(
            "posthoc_merge_audit"
        )

        if audit_trail:
            audit_df = pd.DataFrame(audit_trail)

            # Enrich merge audit with human-readable labels if tree is available
            if tree is not None:
                node_labels = nx.get_node_attributes(tree, "label")
                for col in ["left_cluster", "right_cluster", "lca"]:
                    if col in audit_df.columns:
                        # Fallback to ID if label is missing
                        audit_df[f"{col}_label"] = audit_df[col].apply(
                            lambda x: node_labels.get(x, x) if pd.notna(x) else ""
                        )

            audit_file = audit_dir / f"case_{case_id}_{method}_merges.csv"
            audit_df.to_csv(audit_file, index=False)
            if verbose:
                print(f"  Audit log saved: {audit_file.name}")

        # 2. Export Primary Node Stats
        stats_df = res.get("stats")
        if stats_df is not None:
            # ...existing code...
            cols_to_drop = [
                "distribution",
                "kl_divergence_per_column_global",
                "kl_divergence_per_column_local",
            ]
            export_df = stats_df.drop(
                columns=[c for c in cols_to_drop if c in stats_df.columns]
            ).copy()

            if tree is not None:
                node_labels = nx.get_node_attributes(tree, "label")
                node_parents = {c: p for p, c in tree.edges()}

                # Extract branch lengths from edges
                branch_lengths = {}
                for p, c, data in tree.edges(data=True):
                    branch_lengths[c] = data.get("branch_length")

                # Enrich with Label, Parent context, and Branch Length
                export_df.insert(
                    0,
                    "node_label",
                    [node_labels.get(idx, idx) for idx in export_df.index],
                )
                export_df["parent_node"] = export_df.index.map(node_parents)
                export_df["parent_label"] = export_df["parent_node"].apply(
                    lambda x: node_labels.get(x, x) if pd.notna(x) else ""
                )
                export_df["branch_length"] = export_df.index.map(branch_lengths)

            # Polishing: Round numeric values for human scanning, but PRESERVE p-values
            p_val_cols = [
                c
                for c in export_df.columns
                if "p_value" in c.lower() or "_bh" in c.lower() or "corrected" in c.lower()
            ]
            numeric_cols = export_df.select_dtypes(include=[np.number]).columns
            round_cols = [c for c in numeric_cols if c not in p_val_cols]

            export_df[round_cols] = export_df[round_cols].round(6)

            stats_file = audit_dir / f"case_{case_id}_{method}_stats.csv"
            export_df.to_csv(stats_file, index=True)
            if verbose:
                print(f"  Node stats saved: {stats_file.name}")


def _coerce_2d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    return arr.reshape(arr.shape[0], -1)


def _downsample_2d(arr: np.ndarray, max_size: int) -> np.ndarray:
    if max_size <= 0:
        return arr
    rows, cols = arr.shape
    target_rows = min(rows, max_size)
    target_cols = min(cols, max_size)
    bin_rows = int(np.ceil(rows / target_rows))
    bin_cols = int(np.ceil(cols / target_cols))
    pad_rows = bin_rows * target_rows - rows
    pad_cols = bin_cols * target_cols - cols
    if pad_rows or pad_cols:
        # Cast to float to support NaN padding if input is integer
        if not np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float32)

        arr = np.pad(
            arr,
            ((0, pad_rows), (0, pad_cols)),
            mode="constant",
            constant_values=np.nan,
        )
    reshaped = arr.reshape(target_rows, bin_rows, target_cols, bin_cols)
    with np.errstate(all="ignore"):
        return np.nanmean(reshaped, axis=(1, 3))


def _normalize_to_unit(arr: np.ndarray) -> np.ndarray:
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr, dtype=np.float32)
    min_val = float(np.nanmin(arr))
    max_val = float(np.nanmax(arr))
    if max_val - min_val <= 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    scaled = (arr - min_val) / (max_val - min_val)
    scaled[~finite] = 0.0
    return scaled.astype(np.float32)


def _sample_values(values: np.ndarray, max_items: int, rng: np.random.Generator) -> np.ndarray:
    flat = values.reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return flat.astype(np.float32)
    if max_items is not None and flat.size > max_items:
        idx = rng.choice(flat.size, size=max_items, replace=False)
        flat = flat[idx]
    return flat.astype(np.float32)


def _matrix_stats(arr: np.ndarray) -> Dict[str, float]:
    finite = np.isfinite(arr)
    if np.any(finite):
        data = arr[finite]
        min_val = float(np.min(data))
        max_val = float(np.max(data))
        mean_val = float(np.mean(data))
        std_val = float(np.std(data))
    else:
        min_val = max_val = mean_val = std_val = float("nan")
    nnz = int(np.count_nonzero(arr))
    total = int(arr.size)
    nnz_ratio = float(nnz / total) if total else 0.0
    return {
        "min": min_val,
        "max": max_val,
        "mean": mean_val,
        "std": std_val,
        "nnz": float(nnz),
        "nnz_ratio": nnz_ratio,
        "rows": float(arr.shape[0]) if arr.ndim >= 1 else 1.0,
        "cols": float(arr.shape[1]) if arr.ndim >= 2 else 1.0,
    }


def export_matrix_audit(
    matrices: Dict[str, Any],
    output_root: Path,
    tag_prefix: str,
    *,
    step: int = 0,
    include_products: bool = True,
    max_image_size: int = 512,
    max_hist_values: int = 200_000,
    product_max_dim: int = 512,
    product_max_entries: int = 5_000_000,
    verbose: bool = False,
) -> None:
    """Export matrix audits to TensorBoard summaries.

    Requires TensorFlow to be installed; summaries are written under
    [output_root]/audit/tensorboard.
    """
    try:
        import tensorflow as tf
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow is required for matrix audits. Install it (e.g., "
            "`pip install tensorflow`) or disable matrix_audit."
        ) from exc

    log_dir = output_root / "audit" / "tensorboard"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = tf.summary.create_file_writer(str(log_dir))
    rng = np.random.default_rng(0)

    with writer.as_default():
        for name, raw in matrices.items():
            if raw is None:
                continue
            try:
                arr = np.asarray(raw)
                if arr.size == 0:
                    continue

                arr2 = _coerce_2d(arr)
                stats = _matrix_stats(arr2)

                def safe_int(v):
                    try:
                        return int(v)
                    except:
                        return 0

                summary_lines = [
                    f"name: {name}",
                    f"shape: {arr.shape}",
                    f"dtype: {arr.dtype}",
                    f"min: {stats['min']}",
                    f"max: {stats['max']}",
                    f"mean: {stats['mean']}",
                    f"std: {stats['std']}",
                    f"nnz: {safe_int(stats['nnz'])}",
                    f"nnz_ratio: {stats['nnz_ratio']:.6f}",
                ]
                tf.summary.text(f"{tag_prefix}/{name}/summary", "\n".join(summary_lines), step=step)

                for key, val in stats.items():
                    if np.isfinite(val):
                        tf.summary.scalar(f"{tag_prefix}/{name}/{key}", val, step=step)

                hist_vals = _sample_values(arr2, max_hist_values, rng)
                if hist_vals.size:
                    tf.summary.histogram(f"{tag_prefix}/{name}/histogram", hist_vals, step=step)

                # Suppress "Mean of empty slice" warnings for all-NaN blocks
                with np.errstate(all="ignore"):
                    down = _downsample_2d(arr2, max_image_size)

                heat = _normalize_to_unit(down)
                tf.summary.image(
                    f"{tag_prefix}/{name}/heatmap",
                    heat[np.newaxis, ..., np.newaxis],
                    step=step,
                )

                with np.errstate(all="ignore"):
                    sparsity = _downsample_2d((arr2 != 0).astype(np.float32), max_image_size)
                tf.summary.image(
                    f"{tag_prefix}/{name}/sparsity",
                    sparsity[np.newaxis, ..., np.newaxis],
                    step=step,
                )

                if include_products and arr2.ndim == 2:
                    rows, cols = arr2.shape
                    if rows <= product_max_dim and rows * cols <= product_max_entries:
                        gram = arr2 @ arr2.T
                        with np.errstate(all="ignore"):
                            down_g = _downsample_2d(gram, max_image_size)
                        heat_g = _normalize_to_unit(down_g)
                        tf.summary.image(
                            f"{tag_prefix}/{name}/product_rows",
                            heat_g[np.newaxis, ..., np.newaxis],
                            step=step,
                        )
                    if (
                        cols <= product_max_dim
                        and rows * cols <= product_max_entries
                        and cols != rows
                    ):
                        gram = arr2.T @ arr2
                        with np.errstate(all="ignore"):
                            down_g = _downsample_2d(gram, max_image_size)
                        heat_g = _normalize_to_unit(down_g)
                        tf.summary.image(
                            f"{tag_prefix}/{name}/product_cols",
                            heat_g[np.newaxis, ..., np.newaxis],
                            step=step,
                        )
            except Exception as e:
                # Log error but don't crash the benchmark
                err_text = f"Error exporting matrix '{name}': {e}"
                import traceback

                traceback.print_exc()
                tf.summary.text(f"{tag_prefix}/{name}/error", err_text, step=step)
                if verbose:
                    print(f"  FAILED audit for {name}: {e}")

    writer.flush()
    if verbose:
        print(f"  Matrix audit written to: {log_dir}")
