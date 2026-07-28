"""Test centroid, geometric-median, and Tukey-depth rooted trees.

The robust center smoke tests whether a center estimate is close to a latent
root. This follow-up adds a tree step: build one unrooted average-linkage tree,
then choose the root edge closest to each candidate center. The comparison
therefore isolates rooting, not tree construction.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from benchmarks.diagnostics.calibration.root.center.robust_root_center_smoke import (
    STUDY_ROLE as CENTER_STUDY_ROLE,
)
from benchmarks.diagnostics.calibration.root.center.robust_root_center_smoke import (
    approximate_tukey_candidate_center,
    branch_distance_cv,
    geometric_median,
    simulate_branched_cloud,
)

SCHEMA_VERSION = "robust_center_rooted_tree_smoke/v1"
STUDY_ROLE = "diagnostic_robust_center_rooted_tree_not_calibration"
GENERATED_BY = "benchmarks.diagnostics.calibration.root.center.robust_center_rooted_tree_smoke"
DEFAULT_OUTPUT_DIR = Path("raw/assets/benchmark-results/robust_center_rooted_tree_smoke_20260625")
ROWS_OUTPUT = "robust_center_rooted_tree_rows.csv"
SUMMARY_OUTPUT = "robust_center_rooted_tree_summary.csv"
MANIFEST_OUTPUT = "manifest.json"


@dataclass(frozen=True)
class RobustCenterRootedTreeSmokeConfig:
    """Runtime contract for the center-rooted tree smoke test."""

    output_dir: Path = DEFAULT_OUTPUT_DIR
    replicates: int = 30
    dimensions: tuple[int, ...] = (5, 20, 100)
    n_per_cluster: int = 80
    branch_count: int = 4
    branch_sep: float = 2.5
    noise: float = 1.0
    outlier_fraction: float = 0.10
    outlier_distance: float = 12.0
    random_depth_directions: int = 1024
    candidate_pool_size: int = 400
    min_root_side_fraction: float = 0.05
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


@dataclass(frozen=True)
class LinkageTree:
    """Unrooted tree induced by a SciPy linkage matrix."""

    adjacency: dict[int, tuple[int, ...]]
    coordinates: dict[int, np.ndarray]
    leaf_sets: dict[int, frozenset[int]]
    n_leaves: int


def _scenario_grid(config: RobustCenterRootedTreeSmokeConfig) -> tuple[dict[str, object], ...]:
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


def _segment_distance(point: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    direction = right - left
    norm_sq = float(direction @ direction)
    if norm_sq <= 0.0:
        return float(np.linalg.norm(point - left))
    fraction = float(((point - left) @ direction) / norm_sq)
    fraction = min(1.0, max(0.0, fraction))
    projection = left + fraction * direction
    return float(np.linalg.norm(point - projection))


def build_unrooted_linkage_tree(matrix: np.ndarray) -> LinkageTree:
    """Build an undirected binary tree from average linkage."""
    n_leaves = matrix.shape[0]
    if n_leaves < 3:
        raise ValueError("At least three observations are required.")
    linkage_matrix = linkage(pdist(matrix), method="average")
    adjacency: dict[int, list[int]] = {index: [] for index in range(n_leaves)}
    leaf_sets: dict[int, frozenset[int]] = {index: frozenset({index}) for index in range(n_leaves)}
    coordinates: dict[int, np.ndarray] = {index: matrix[index].copy() for index in range(n_leaves)}

    for merge_index, row in enumerate(linkage_matrix):
        left = int(row[0])
        right = int(row[1])
        node = n_leaves + merge_index
        adjacency[node] = [left, right]
        adjacency[left].append(node)
        adjacency[right].append(node)
        leaves = leaf_sets[left] | leaf_sets[right]
        leaf_sets[node] = leaves
        coordinates[node] = matrix[list(leaves)].mean(axis=0)

    return LinkageTree(
        adjacency={key: tuple(value) for key, value in adjacency.items()},
        coordinates=coordinates,
        leaf_sets=leaf_sets,
        n_leaves=n_leaves,
    )


def _component_leaves(tree: LinkageTree, start: int, blocked: int) -> frozenset[int]:
    seen = {blocked}
    queue: deque[int] = deque([start])
    leaves: set[int] = set()
    while queue:
        node = queue.popleft()
        if node in seen:
            continue
        seen.add(node)
        if node < tree.n_leaves:
            leaves.add(node)
        for neighbor in tree.adjacency[node]:
            if neighbor not in seen:
                queue.append(neighbor)
    return frozenset(leaves)


def select_root_edge_nearest_center(
    tree: LinkageTree,
    center: np.ndarray,
    *,
    min_side_fraction: float,
) -> tuple[int, int, float, float, float]:
    """Select a nontrivial tree edge closest to the supplied center."""
    min_side_count = max(2, int(math.ceil(tree.n_leaves * min_side_fraction)))
    true_root = np.zeros_like(center)
    best: tuple[int, int, float, float, float] | None = None
    for left, neighbors in tree.adjacency.items():
        for right in neighbors:
            if left > right:
                continue
            leaves = _component_leaves(tree, left, right)
            side_count = len(leaves)
            other_side_count = tree.n_leaves - side_count
            if side_count < min_side_count or other_side_count < min_side_count:
                continue
            left_coord = tree.coordinates[left]
            right_coord = tree.coordinates[right]
            center_distance = _segment_distance(center, left_coord, right_coord)
            true_root_distance = _segment_distance(
                true_root,
                left_coord,
                right_coord,
            )
            side_fraction = min(side_count, other_side_count) / tree.n_leaves
            candidate = (
                left,
                right,
                center_distance,
                true_root_distance,
                side_fraction,
            )
            if best is None or candidate[2] < best[2]:
                best = candidate
    if best is None:
        raise ValueError("No nontrivial root edge satisfied the side-size guard.")
    return best


def _partition_labels(tree: LinkageTree, edge: tuple[int, int]) -> np.ndarray:
    left_leaves = _component_leaves(tree, edge[0], edge[1])
    labels = np.ones(tree.n_leaves, dtype=int)
    labels[list(left_leaves)] = 0
    return labels


def _branch_integrity(partition: np.ndarray, branch_labels: np.ndarray) -> tuple[float, int]:
    clean_mask = branch_labels >= 0
    clean_labels = branch_labels[clean_mask]
    clean_partition = partition[clean_mask]
    total = clean_labels.size
    if total == 0:
        return math.nan, 0
    preserved = 0.0
    split_branch_count = 0
    for label in sorted(set(clean_labels)):
        branch_partition = clean_partition[clean_labels == label]
        counts = np.bincount(branch_partition, minlength=2)
        branch_size = int(counts.sum())
        preserved += int(counts.max())
        if counts.min() / branch_size >= 0.10:
            split_branch_count += 1
    return float(preserved / total), split_branch_count


def _smaller_side_outlier_fraction(
    partition: np.ndarray,
    branch_labels: np.ndarray,
) -> float:
    counts = np.bincount(partition, minlength=2)
    smaller_side = int(np.argmin(counts))
    side_mask = partition == smaller_side
    if not np.any(side_mask):
        return math.nan
    return float(np.mean(branch_labels[side_mask] < 0))


def _center_methods(
    matrix: np.ndarray,
    *,
    rng: np.random.Generator,
    n_directions: int,
    candidate_pool_size: int,
) -> dict[str, tuple[np.ndarray, float]]:
    tukey_center, tukey_depth = approximate_tukey_candidate_center(
        matrix,
        rng=rng,
        n_directions=n_directions,
        candidate_pool_size=candidate_pool_size,
    )
    return {
        "centroid_rooted": (matrix.mean(axis=0), math.nan),
        "geometric_median_rooted": (geometric_median(matrix), math.nan),
        "tukey_depth_rooted": (tukey_center, tukey_depth),
    }


def run_robust_center_rooted_tree_smoke(
    config: RobustCenterRootedTreeSmokeConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the center-rooted tree comparison."""
    rows: list[dict[str, object]] = []
    for dimension in config.dimensions:
        for scenario_index, scenario in enumerate(_scenario_grid(config)):
            for replicate in range(config.replicates):
                seed = (
                    config.base_seed + 200_000 * int(dimension) + 1_000 * scenario_index + replicate
                )
                matrix, labels = simulate_branched_cloud(
                    dimension=int(dimension),
                    n_per_cluster=config.n_per_cluster,
                    branch_count=config.branch_count,
                    branch_sep=config.branch_sep,
                    noise=config.noise,
                    outlier_fraction=float(scenario["outlier_fraction"]),
                    outlier_distance=config.outlier_distance,
                    imbalanced=bool(scenario["imbalanced"]),
                    rng=np.random.default_rng(seed),
                )
                tree = build_unrooted_linkage_tree(matrix)
                centers = _center_methods(
                    matrix,
                    rng=np.random.default_rng(seed + 17),
                    n_directions=config.random_depth_directions,
                    candidate_pool_size=config.candidate_pool_size,
                )
                for method, (center, depth) in centers.items():
                    edge = select_root_edge_nearest_center(
                        tree,
                        center,
                        min_side_fraction=config.min_root_side_fraction,
                    )
                    partition = _partition_labels(tree, (edge[0], edge[1]))
                    integrity, split_branch_count = _branch_integrity(
                        partition,
                        labels,
                    )
                    rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "study_role": STUDY_ROLE,
                            "center_source_study_role": CENTER_STUDY_ROLE,
                            "dimension": int(dimension),
                            "scenario": str(scenario["scenario"]),
                            "replicate": int(replicate),
                            "method": method,
                            "center_error": float(np.linalg.norm(center)),
                            "center_branch_distance_cv": branch_distance_cv(
                                center,
                                matrix,
                                labels,
                            ),
                            "approx_halfspace_depth": depth,
                            "root_edge_left": int(edge[0]),
                            "root_edge_right": int(edge[1]),
                            "root_edge_distance_to_center": float(edge[2]),
                            "root_edge_distance_to_true_root": float(edge[3]),
                            "root_edge_min_side_fraction": float(edge[4]),
                            "branch_integrity": integrity,
                            "split_branch_count": int(split_branch_count),
                            "smaller_side_outlier_fraction": (
                                _smaller_side_outlier_fraction(partition, labels)
                            ),
                        }
                    )
    rows_df = pd.DataFrame.from_records(rows)
    summary = (
        rows_df.groupby(["dimension", "scenario", "method"], as_index=False)
        .agg(
            replicate_count=("replicate", "nunique"),
            center_error_mean=("center_error", "mean"),
            root_edge_true_distance_mean=(
                "root_edge_distance_to_true_root",
                "mean",
            ),
            root_edge_true_distance_median=(
                "root_edge_distance_to_true_root",
                "median",
            ),
            branch_integrity_mean=("branch_integrity", "mean"),
            split_branch_count_mean=("split_branch_count", "mean"),
            min_side_fraction_mean=("root_edge_min_side_fraction", "mean"),
            smaller_side_outlier_fraction_mean=(
                "smaller_side_outlier_fraction",
                "mean",
            ),
            approx_halfspace_depth_median=("approx_halfspace_depth", "median"),
        )
        .sort_values(["dimension", "scenario", "root_edge_true_distance_mean", "method"])
    )
    summary["root_edge_true_distance_rank"] = summary.groupby(["dimension", "scenario"])[
        "root_edge_true_distance_mean"
    ].rank(method="dense")
    summary["branch_integrity_rank"] = summary.groupby(["dimension", "scenario"])[
        "branch_integrity_mean"
    ].rank(method="dense", ascending=False)
    summary.insert(0, "schema_version", SCHEMA_VERSION)
    summary.insert(1, "study_role", STUDY_ROLE)
    return rows_df, summary


def write_outputs(config: RobustCenterRootedTreeSmokeConfig) -> dict[str, Path]:
    rows, summary = run_robust_center_rooted_tree_smoke(config)
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
            "Center-rooting can improve the selected root edge when outliers "
            "pull the centroid, but it remains a rooting diagnostic rather "
            "than a selected-root validity or calibration proof."
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
    parser.add_argument("--replicates", type=int, default=30)
    parser.add_argument("--dimensions", type=int, nargs="+", default=[5, 20, 100])
    parser.add_argument("--random-depth-directions", type=int, default=1024)
    parser.add_argument("--candidate-pool-size", type=int, default=400)
    parser.add_argument("--min-root-side-fraction", type=float, default=0.05)
    parser.add_argument("--base-seed", type=int, default=20260625)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = write_outputs(
        RobustCenterRootedTreeSmokeConfig(
            output_dir=args.output_dir,
            replicates=args.replicates,
            dimensions=tuple(args.dimensions),
            random_depth_directions=args.random_depth_directions,
            candidate_pool_size=args.candidate_pool_size,
            min_root_side_fraction=args.min_root_side_fraction,
            base_seed=args.base_seed,
        )
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
