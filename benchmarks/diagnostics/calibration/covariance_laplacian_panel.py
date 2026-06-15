"""Covariance Laplacian diagnostic.

This module treats a covariance matrix as a weighted feature graph after
normalizing to absolute correlation and removing the diagonal. It reports graph
connectivity, Laplacian eigenvalues, and diagonal/off-diagonal mass. It is
diagnostic-only and does not install a covariance model.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "diagnostic_covariance_laplacian_panel_not_calibration"
SCHEMA_VERSION = "covariance_laplacian_panel/v1"


def _as_covariance_matrix(covariance: np.ndarray) -> np.ndarray:
    matrix = np.asarray(covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"covariance must be square; got shape {matrix.shape}.")
    if matrix.shape[0] == 0:
        raise ValueError("covariance must be non-empty.")
    if not np.isfinite(matrix).all():
        raise ValueError("covariance must contain only finite values.")
    if not np.allclose(matrix, matrix.T, atol=1e-10, rtol=1e-8):
        raise ValueError("covariance must be symmetric.")
    return 0.5 * (matrix + matrix.T)


def _absolute_correlation_adjacency(
    covariance: np.ndarray,
    *,
    correlation_threshold: float,
) -> np.ndarray:
    diagonal = np.diag(covariance)
    positive = diagonal > 0.0
    scale = np.zeros_like(diagonal, dtype=float)
    scale[positive] = np.sqrt(diagonal[positive])
    denominator = np.outer(scale, scale)
    correlation = np.zeros_like(covariance, dtype=float)
    valid = denominator > 0.0
    correlation[valid] = covariance[valid] / denominator[valid]
    adjacency = np.abs(np.clip(correlation, -1.0, 1.0))
    np.fill_diagonal(adjacency, 0.0)
    adjacency[adjacency < float(correlation_threshold)] = 0.0
    return adjacency


def _component_sizes(adjacency: np.ndarray) -> list[int]:
    n_nodes = int(adjacency.shape[0])
    seen = np.zeros(n_nodes, dtype=bool)
    sizes: list[int] = []
    neighbors = [np.flatnonzero(adjacency[index] > 0.0) for index in range(n_nodes)]
    for start in range(n_nodes):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for neighbor in neighbors[node]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(int(neighbor))
        sizes.append(size)
    return sizes


def _normalized_laplacian(adjacency: np.ndarray) -> np.ndarray:
    degrees = adjacency.sum(axis=1)
    inv_sqrt = np.zeros_like(degrees)
    positive = degrees > 0.0
    inv_sqrt[positive] = 1.0 / np.sqrt(degrees[positive])
    normalized_adjacency = adjacency * np.outer(inv_sqrt, inv_sqrt)
    laplacian = np.eye(adjacency.shape[0], dtype=float) - normalized_adjacency
    laplacian[~positive, ~positive] = 0.0
    return 0.5 * (laplacian + laplacian.T)


def _laplacian_status(
    *,
    n_components: int,
    off_diagonal_abs_mass_fraction: float,
    algebraic_connectivity: float,
    min_algebraic_connectivity: float,
) -> str:
    if off_diagonal_abs_mass_fraction <= 1e-12:
        return "laplacian_diagonal_covariance"
    if n_components > 1:
        return "laplacian_disconnected_covariance"
    if (
        math.isfinite(algebraic_connectivity)
        and algebraic_connectivity < min_algebraic_connectivity
    ):
        return "laplacian_near_disconnected_covariance"
    return "laplacian_connected_covariance"


def analyze_covariance_laplacian(
    covariance: np.ndarray,
    *,
    matrix_id: str,
    matrix_context_role: str = "unspecified",
    feature_family: str = "unknown",
    correlation_threshold: float = 1e-10,
    min_algebraic_connectivity: float = 1e-3,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return one covariance-Laplacian diagnostic row."""
    if float(correlation_threshold) < 0.0:
        raise ValueError(
            "correlation_threshold must be non-negative; "
            f"got {correlation_threshold!r}."
        )
    if float(min_algebraic_connectivity) < 0.0:
        raise ValueError(
            "min_algebraic_connectivity must be non-negative; "
            f"got {min_algebraic_connectivity!r}."
        )
    matrix = _as_covariance_matrix(covariance)
    n_features = int(matrix.shape[0])
    eigenvalues = np.linalg.eigvalsh(matrix)
    min_eigenvalue = float(np.min(eigenvalues))
    trace = float(np.trace(matrix))
    abs_mass = float(np.sum(np.abs(matrix)))
    diagonal_abs_mass = float(np.sum(np.abs(np.diag(matrix))))
    off_diagonal_abs_mass = max(abs_mass - diagonal_abs_mass, 0.0)
    off_diagonal_abs_mass_fraction = (
        float(off_diagonal_abs_mass / abs_mass) if abs_mass > 0.0 else 0.0
    )
    adjacency = _absolute_correlation_adjacency(
        matrix,
        correlation_threshold=correlation_threshold,
    )
    component_sizes = _component_sizes(adjacency)
    degrees = adjacency.sum(axis=1)
    edge_mask = adjacency > 0.0
    n_possible_edges = n_features * (n_features - 1)
    graph_density = (
        float(np.count_nonzero(edge_mask) / n_possible_edges)
        if n_possible_edges
        else 0.0
    )
    laplacian = _normalized_laplacian(adjacency)
    laplacian_eigenvalues = np.linalg.eigvalsh(laplacian)
    laplacian_eigenvalues[np.abs(laplacian_eigenvalues) < 1e-12] = 0.0
    positive_laplacian = laplacian_eigenvalues[laplacian_eigenvalues > 1e-12]
    algebraic_connectivity = (
        float(positive_laplacian[0])
        if len(component_sizes) == 1 and positive_laplacian.size
        else 0.0
    )
    row: dict[str, object] = {
        "matrix_id": str(matrix_id),
        "matrix_context_role": str(matrix_context_role),
        "feature_family": str(feature_family),
        "n_features": n_features,
        "covariance_trace": trace,
        "covariance_min_eigenvalue": min_eigenvalue,
        "covariance_max_eigenvalue": float(np.max(eigenvalues)),
        "covariance_condition_ratio": (
            float(np.max(eigenvalues) / max(min_eigenvalue, 1e-300))
            if min_eigenvalue > 0.0
            else math.inf
        ),
        "covariance_effective_rank": (
            float(trace**2 / np.sum(eigenvalues**2))
            if trace > 0.0 and np.sum(eigenvalues**2) > 0.0
            else math.nan
        ),
        "diagonal_abs_mass_fraction": (
            float(diagonal_abs_mass / abs_mass) if abs_mass > 0.0 else math.nan
        ),
        "off_diagonal_abs_mass_fraction": off_diagonal_abs_mass_fraction,
        "mean_abs_correlation_degree": float(np.mean(degrees)),
        "max_abs_correlation_degree": float(np.max(degrees)) if degrees.size else 0.0,
        "graph_density": graph_density,
        "n_connected_components": int(len(component_sizes)),
        "largest_component_fraction": float(max(component_sizes) / n_features),
        "normalized_laplacian_lambda2": algebraic_connectivity,
        "normalized_laplacian_lambda_max": float(np.max(laplacian_eigenvalues)),
        "laplacian_zero_eigenvalue_count": int(np.sum(laplacian_eigenvalues == 0.0)),
        "laplacian_status": _laplacian_status(
            n_components=len(component_sizes),
            off_diagonal_abs_mass_fraction=off_diagonal_abs_mass_fraction,
            algebraic_connectivity=algebraic_connectivity,
            min_algebraic_connectivity=float(min_algebraic_connectivity),
        ),
        "correlation_threshold": float(correlation_threshold),
        "min_algebraic_connectivity": float(min_algebraic_connectivity),
        "study_role": STUDY_ROLE,
    }
    if metadata:
        for key, value in metadata.items():
            row[str(key)] = value
    return row


def analyze_covariance_laplacian_panel(
    matrices: Iterable[tuple[str, np.ndarray, Mapping[str, object] | None]],
    *,
    correlation_threshold: float = 1e-10,
    min_algebraic_connectivity: float = 1e-3,
) -> pd.DataFrame:
    """Analyze an iterable of covariance matrices."""
    rows = []
    for matrix_id, covariance, metadata in matrices:
        metadata_dict = dict(metadata or {})
        rows.append(
            analyze_covariance_laplacian(
                covariance,
                matrix_id=matrix_id,
                matrix_context_role=str(
                    metadata_dict.pop("matrix_context_role", "unspecified")
                ),
                feature_family=str(metadata_dict.pop("feature_family", "unknown")),
                correlation_threshold=correlation_threshold,
                min_algebraic_connectivity=min_algebraic_connectivity,
                metadata=metadata_dict,
            )
        )
    return pd.DataFrame.from_records(rows)


def summarize_covariance_laplacian_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize covariance Laplacian rows by family, role, and status."""
    if rows.empty:
        return pd.DataFrame()
    summaries: list[dict[str, object]] = []
    for key, group in rows.groupby(
        ["feature_family", "matrix_context_role", "laplacian_status"],
        sort=True,
    ):
        family, role, status = key
        summaries.append(
            {
                "feature_family": family,
                "matrix_context_role": role,
                "laplacian_status": status,
                "n_matrices": int(group.shape[0]),
                "n_features_median": float(group["n_features"].median()),
                "off_diagonal_abs_mass_fraction_median": float(
                    group["off_diagonal_abs_mass_fraction"].median()
                ),
                "largest_component_fraction_median": float(
                    group["largest_component_fraction"].median()
                ),
                "normalized_laplacian_lambda2_median": float(
                    group["normalized_laplacian_lambda2"].median()
                ),
                "covariance_effective_rank_median": float(
                    group["covariance_effective_rank"].median()
                ),
                "study_role": STUDY_ROLE,
            }
        )
    return pd.DataFrame.from_records(summaries)


def run_covariance_laplacian_panel(
    *,
    matrix_dir: Path,
    output_dir: Path,
    pattern: str = "*.csv",
    correlation_threshold: float = 1e-10,
    min_algebraic_connectivity: float = 1e-3,
) -> dict[str, Path]:
    """Run the covariance Laplacian panel on CSV matrix files in a directory."""
    matrices: list[tuple[str, np.ndarray, Mapping[str, object] | None]] = []
    for path in sorted(matrix_dir.glob(pattern)):
        matrix = pd.read_csv(path, header=None).to_numpy(dtype=float)
        matrices.append((path.stem, matrix, {"matrix_context_role": "csv_matrix"}))
    rows = analyze_covariance_laplacian_panel(
        matrices,
        correlation_threshold=correlation_threshold,
        min_algebraic_connectivity=min_algebraic_connectivity,
    )
    summary = summarize_covariance_laplacian_rows(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "covariance_laplacian_rows.csv"
    summary_path = output_dir / "covariance_laplacian_summary.csv"
    manifest_path = output_dir / "manifest.json"
    rows.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)
    manifest = {
        "created_at_utc": format_timestamp_utc(),
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "matrix_dir": str(matrix_dir),
        "pattern": pattern,
        "outputs": {
            "rows": str(rows_path),
            "summary": str(summary_path),
        },
        "interpretation": (
            "Diagnostic covariance Laplacian panel over absolute-correlation "
            "graphs. Connectedness and algebraic connectivity are descriptive "
            "covariance-geometry evidence, not production calibration."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"rows": rows_path, "summary": summary_path, "manifest": manifest_path}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*.csv")
    parser.add_argument("--correlation-threshold", type=float, default=1e-10)
    parser.add_argument("--min-algebraic-connectivity", type=float, default=1e-3)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    outputs = run_covariance_laplacian_panel(
        matrix_dir=args.matrix_dir,
        output_dir=args.output_dir,
        pattern=args.pattern,
        correlation_threshold=args.correlation_threshold,
        min_algebraic_connectivity=args.min_algebraic_connectivity,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "STUDY_ROLE",
    "analyze_covariance_laplacian",
    "analyze_covariance_laplacian_panel",
    "run_covariance_laplacian_panel",
    "summarize_covariance_laplacian_rows",
]
