"""Smoke test robust center estimators as high-dimensional root anchors.

This diagnostic asks a narrow question: if a latent root is the center of a
high-dimensional branched point cloud, do robust center estimators help recover
that root anchor compared with the centroid?

It is intentionally not a production root-selection rule. In Tree-Break
Selection, the root is a selected topology event, so a center estimate can only
serve as a diagnostic anchor or prior. The selected-root validity and tail
support diagnostics remain separate.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

SCHEMA_VERSION = "robust_root_center_smoke/v1"
STUDY_ROLE = "diagnostic_robust_center_root_anchor_not_calibration"
GENERATED_BY = "benchmarks.diagnostics.calibration.robust_root_center_smoke"
DEFAULT_OUTPUT_DIR = Path(
    "raw/assets/benchmark-results/robust_root_center_smoke_20260625"
)
ROWS_OUTPUT = "robust_root_center_rows.csv"
SUMMARY_OUTPUT = "robust_root_center_summary.csv"
MANIFEST_OUTPUT = "manifest.json"


@dataclass(frozen=True)
class RobustRootCenterSmokeConfig:
    """Runtime contract for the robust root-center smoke test."""

    output_dir: Path = DEFAULT_OUTPUT_DIR
    replicates: int = 40
    dimensions: tuple[int, ...] = (5, 20, 100)
    n_per_cluster: int = 80
    branch_count: int = 4
    branch_sep: float = 2.5
    noise: float = 1.0
    outlier_fraction: float = 0.10
    outlier_distance: float = 12.0
    random_depth_directions: int = 1024
    candidate_pool_size: int = 400
    base_seed: int = 20260625

    @property
    def rows_path(self) -> Path:
        return self.output_dir / ROWS_OUTPUT

    @property
    def summary_path(self) -> Path:
        return self.output_dir / SUMMARY_OUTPUT

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / MANIFEST_OUTPUT


def geometric_median(
    matrix: np.ndarray,
    *,
    tolerance: float = 1e-7,
    max_iter: int = 500,
) -> np.ndarray:
    """Compute the Euclidean geometric median by Weiszfeld iteration."""
    center = np.median(matrix, axis=0)
    for _ in range(max_iter):
        distances = np.linalg.norm(matrix - center, axis=1)
        weights = 1.0 / np.maximum(distances, tolerance)
        updated = (matrix * weights[:, None]).sum(axis=0) / weights.sum()
        if np.linalg.norm(updated - center) < tolerance:
            return updated
        center = updated
    return center


def euclidean_medoid(matrix: np.ndarray) -> np.ndarray:
    """Return the observed point with smallest total Euclidean distance."""
    distances = cdist(matrix, matrix)
    return matrix[int(np.argmin(distances.sum(axis=1)))]


def _candidate_pool(
    matrix: np.ndarray,
    *,
    rng: np.random.Generator,
    candidate_pool_size: int,
) -> np.ndarray:
    """Build continuous center candidates for approximate halfspace depth."""
    candidates = [
        matrix.mean(axis=0),
        np.median(matrix, axis=0),
        geometric_median(matrix),
    ]
    per_block = max(1, candidate_pool_size // 4)
    for sample_size in (5, 10, 25, 50):
        for _ in range(per_block):
            indices = rng.choice(
                matrix.shape[0],
                size=min(sample_size, matrix.shape[0]),
                replace=False,
            )
            candidates.append(matrix[indices].mean(axis=0))
    return np.vstack(candidates)


def _halfspace_depth_scores(
    matrix: np.ndarray,
    candidates: np.ndarray,
    *,
    n_directions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Approximate Tukey/halfspace depth with random projected directions."""
    n_observations, dimension = matrix.shape
    directions = rng.normal(size=(n_directions, dimension))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    projected_matrix = matrix @ directions.T
    projected_candidates = candidates @ directions.T

    depths = np.empty((candidates.shape[0], n_directions), dtype=float)
    for direction_index in range(n_directions):
        sorted_projection = np.sort(projected_matrix[:, direction_index])
        candidate_projection = projected_candidates[:, direction_index]
        left_mass = np.searchsorted(
            sorted_projection,
            candidate_projection,
            side="right",
        )
        right_mass = n_observations - np.searchsorted(
            sorted_projection,
            candidate_projection,
            side="left",
        )
        depths[:, direction_index] = (
            np.minimum(left_mass, right_mass) / n_observations
        )
    return depths.min(axis=1)


def approximate_tukey_candidate_center(
    matrix: np.ndarray,
    *,
    rng: np.random.Generator,
    n_directions: int,
    candidate_pool_size: int,
) -> tuple[np.ndarray, float]:
    """Return the highest random-direction halfspace-depth candidate."""
    candidates = _candidate_pool(
        matrix,
        rng=rng,
        candidate_pool_size=candidate_pool_size,
    )
    scores = _halfspace_depth_scores(
        matrix,
        candidates,
        n_directions=n_directions,
        rng=rng,
    )
    best_index = int(np.argmax(scores))
    return candidates[best_index], float(scores[best_index])


def approximate_tukey_observed_medoid(
    matrix: np.ndarray,
    *,
    rng: np.random.Generator,
    n_directions: int,
) -> tuple[np.ndarray, float]:
    """Return the observed point with highest approximate halfspace depth."""
    scores = _halfspace_depth_scores(
        matrix,
        matrix,
        n_directions=n_directions,
        rng=rng,
    )
    best_index = int(np.argmax(scores))
    return matrix[best_index], float(scores[best_index])


def simulate_branched_cloud(
    *,
    dimension: int,
    n_per_cluster: int,
    branch_count: int,
    branch_sep: float,
    noise: float,
    outlier_fraction: float,
    outlier_distance: float,
    imbalanced: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a branched point cloud with latent root at the origin."""
    directions = rng.normal(size=(branch_count, dimension))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    directions -= directions.mean(axis=0, keepdims=True)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    if imbalanced:
        sizes = np.maximum(
            8,
            np.round(
                n_per_cluster * np.linspace(0.25, 1.75, branch_count)
            ),
        ).astype(int)
    else:
        sizes = np.full(branch_count, n_per_cluster, dtype=int)

    blocks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for branch_index, size in enumerate(sizes):
        blocks.append(
            branch_sep * directions[branch_index]
            + rng.normal(scale=noise, size=(int(size), dimension))
        )
        labels.append(np.full(int(size), branch_index))

    matrix = np.vstack(blocks)
    label_vector = np.concatenate(labels)
    outlier_count = int(round(outlier_fraction * matrix.shape[0]))
    if outlier_count:
        outlier_direction = rng.normal(size=dimension)
        outlier_direction /= np.linalg.norm(outlier_direction)
        outliers = outlier_distance * outlier_direction + rng.normal(
            scale=noise,
            size=(outlier_count, dimension),
        )
        matrix = np.vstack([matrix, outliers])
        label_vector = np.concatenate(
            [label_vector, np.full(outlier_count, -1)]
        )
    return matrix, label_vector


def branch_distance_cv(
    center: np.ndarray,
    matrix: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Summarize how symmetrically the center sees non-outlier branches."""
    non_outlier_mask = labels >= 0
    clean_matrix = matrix[non_outlier_mask]
    clean_labels = labels[non_outlier_mask]
    branch_means = np.vstack(
        [
            clean_matrix[clean_labels == label].mean(axis=0)
            for label in sorted(set(clean_labels))
        ]
    )
    distances = np.linalg.norm(branch_means - center, axis=1)
    return float(distances.std() / (distances.mean() + 1e-12))


def _scenario_grid(config: RobustRootCenterSmokeConfig) -> tuple[dict[str, object], ...]:
    return (
        {
            "scenario": "balanced_clean",
            "outlier_fraction": 0.0,
            "imbalanced": False,
        },
        {
            "scenario": "balanced_10pct_outliers",
            "outlier_fraction": config.outlier_fraction,
            "imbalanced": False,
        },
        {
            "scenario": "imbalanced_clean",
            "outlier_fraction": 0.0,
            "imbalanced": True,
        },
        {
            "scenario": "imbalanced_10pct_outliers",
            "outlier_fraction": config.outlier_fraction,
            "imbalanced": True,
        },
    )


def run_robust_root_center_smoke(
    config: RobustRootCenterSmokeConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the robust center smoke and return row-level and summary outputs."""
    rows: list[dict[str, object]] = []
    for dimension in config.dimensions:
        for scenario_index, scenario in enumerate(_scenario_grid(config)):
            for replicate in range(config.replicates):
                seed = (
                    config.base_seed
                    + 100_000 * int(dimension)
                    + 1_000 * scenario_index
                    + replicate
                )
                rng = np.random.default_rng(seed)
                matrix, labels = simulate_branched_cloud(
                    dimension=int(dimension),
                    n_per_cluster=config.n_per_cluster,
                    branch_count=config.branch_count,
                    branch_sep=config.branch_sep,
                    noise=config.noise,
                    outlier_fraction=float(scenario["outlier_fraction"]),
                    outlier_distance=config.outlier_distance,
                    imbalanced=bool(scenario["imbalanced"]),
                    rng=rng,
                )

                observed_tukey, observed_tukey_depth = (
                    approximate_tukey_observed_medoid(
                        matrix,
                        rng=np.random.default_rng(seed + 11),
                        n_directions=config.random_depth_directions,
                    )
                )
                candidate_tukey, candidate_tukey_depth = (
                    approximate_tukey_candidate_center(
                        matrix,
                        rng=np.random.default_rng(seed + 17),
                        n_directions=config.random_depth_directions,
                        candidate_pool_size=config.candidate_pool_size,
                    )
                )

                method_centers = {
                    "mean_centroid": (matrix.mean(axis=0), math.nan),
                    "coord_median": (np.median(matrix, axis=0), math.nan),
                    "geometric_median": (geometric_median(matrix), math.nan),
                    "euclidean_medoid": (euclidean_medoid(matrix), math.nan),
                    "approx_tukey_observed_medoid": (
                        observed_tukey,
                        observed_tukey_depth,
                    ),
                    "approx_tukey_candidate_center": (
                        candidate_tukey,
                        candidate_tukey_depth,
                    ),
                }
                for method, (center, depth) in method_centers.items():
                    rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "study_role": STUDY_ROLE,
                            "dimension": int(dimension),
                            "scenario": str(scenario["scenario"]),
                            "replicate": int(replicate),
                            "method": method,
                            "root_error": float(np.linalg.norm(center)),
                            "branch_distance_cv": branch_distance_cv(
                                center,
                                matrix,
                                labels,
                            ),
                            "approx_halfspace_depth": depth,
                        }
                    )

    rows_df = pd.DataFrame.from_records(rows)
    summary = (
        rows_df.groupby(["dimension", "scenario", "method"], as_index=False)
        .agg(
            replicate_count=("replicate", "nunique"),
            root_error_mean=("root_error", "mean"),
            root_error_median=("root_error", "median"),
            branch_distance_cv_mean=("branch_distance_cv", "mean"),
            approx_halfspace_depth_median=("approx_halfspace_depth", "median"),
        )
        .sort_values(["dimension", "scenario", "root_error_mean", "method"])
    )
    summary["root_error_rank"] = summary.groupby(["dimension", "scenario"])[
        "root_error_mean"
    ].rank(method="dense")
    summary.insert(0, "schema_version", SCHEMA_VERSION)
    summary.insert(1, "study_role", STUDY_ROLE)
    return rows_df, summary


def write_outputs(config: RobustRootCenterSmokeConfig) -> dict[str, Path]:
    rows, summary = run_robust_root_center_smoke(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(config.rows_path, index=False)
    summary.to_csv(config.summary_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at": datetime.now(UTC).isoformat(),
        "config": asdict(config),
        "outputs": {
            "rows": str(config.rows_path),
            "summary": str(config.summary_path),
        },
        "interpretation": (
            "Robust centers can improve a latent root anchor under outliers, "
            "but center estimation is not selected-root topology validation."
        ),
    }
    config.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return {
        "rows": config.rows_path,
        "summary": config.summary_path,
        "manifest": config.manifest_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--replicates", type=int, default=40)
    parser.add_argument("--dimensions", type=int, nargs="+", default=[5, 20, 100])
    parser.add_argument("--random-depth-directions", type=int, default=1024)
    parser.add_argument("--candidate-pool-size", type=int, default=400)
    parser.add_argument("--base-seed", type=int, default=20260625)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = write_outputs(
        RobustRootCenterSmokeConfig(
            output_dir=args.output_dir,
            replicates=args.replicates,
            dimensions=tuple(args.dimensions),
            random_depth_directions=args.random_depth_directions,
            candidate_pool_size=args.candidate_pool_size,
            base_seed=args.base_seed,
        )
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
