#!/usr/bin/env python3
"""Lab exp27: neighborhood-informed Gate 3 padding diagnostics.

Research question
-----------------
When Gate 3 needs more projection rows than the parent PCA head provides,
can nearby non-focal branches (uncle / cousins) supply useful local
structure before the test falls back to random padding?

This experiment keeps production code unchanged and compares four
runtime-only variants on a compact battery:

1. ``random_tail``              — current production behavior
2. ``uncle_then_random``        — try uncle PCA first, then random fallback
3. ``cousins_then_random``      — try cousin PCA first, then random fallback
4. ``neighborhood_then_random`` — uncle, then cousins, then random fallback

For every sibling test that requests padding, the script logs:

- ``local_fill_ratio``: fraction of requested padding filled locally
- ``radius_needed``: smallest non-focal neighborhood radius that fully
  filled the local request (``NaN`` when random fallback was still needed)
- ``residual_energy``: local parent-orthogonal neighborhood energy

Outputs
-------
- ``per_case.csv``: case-level clustering metrics per variant
- ``padding_audit.csv``: per-test padding diagnostics
- ``collateral_span_nodes.csv``: per-collateral-node residual subspace diagnostics
- ``summary.json``: aggregate metrics
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

try:
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover - fallback when scipy is unavailable
    spearmanr = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LAB_ROOT = REPO_ROOT / "debug_scripts" / "enhancement_lab"
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

os.environ.setdefault("KL_TE_N_JOBS", "1")

from kl_clustering_analysis.hierarchy_analysis.statistics.multiple_testing.tree_bh_correction import (  # noqa: E402
    tree_bh_correction,
)

from debug_scripts.enhancement_lab.lab_helpers import (  # noqa: E402
    build_tree_and_data,
    compute_ari,
    enhancement_lab_results_relative,
    temporary_attr,
    temporary_config,
)
from kl_clustering_analysis import config  # noqa: E402
from kl_clustering_analysis.hierarchy_analysis import (  # noqa: E402
    tree_decomposition as tree_decomposition_module,
)
from kl_clustering_analysis.hierarchy_analysis.decomposition.backends.random_projection_backend import (  # noqa: E402
    generate_projection_matrix_backend,
)
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates import (  # noqa: E402
    orchestrator as gate_orchestrator_module,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence import (  # noqa: E402
    annotate_child_parent_divergence,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.multiple_testing.base import (  # noqa: E402
    benjamini_hochberg_correction,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection import (  # noqa: E402
    projected_wald as projected_wald_module,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence import (  # noqa: E402
    adjusted_wald_annotation as adjusted_wald_module,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing import (  # noqa: E402
    sibling_pair_collection as sibling_pair_collection_module,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing import (  # noqa: E402
    wald_statistic as wald_statistic_module,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_config import (  # noqa: E402
    derive_sibling_pca_projections,
    derive_sibling_spectral_dims,
)

BATTERY_CASES = [
    "binary_balanced_low_noise",
    "binary_balanced_low_noise__2",
    "binary_perfect_8c",
    "gauss_extreme_noise_3c",
    "gauss_extreme_noise_highd",
    "overlap_heavy_4c_small_feat",
    "gauss_overlap_4c_med",
]

SOURCE_RADII: dict[str, int] = {
    "uncle": 1,
    "cousins": 2,
}


@dataclass(frozen=True)
class PaddingVariant:
    """Runtime-only neighborhood padding configuration."""

    name: str
    source_order: tuple[str, ...]
    search_strategy: str = "fixed_sources"
    use_random_fallback: bool = True


VARIANTS: dict[str, PaddingVariant] = {
    "random_tail": PaddingVariant("random_tail", ()),
    "uncle_then_random": PaddingVariant("uncle_then_random", ("uncle",)),
    "cousins_then_random": PaddingVariant("cousins_then_random", ("cousins",)),
    "neighborhood_then_random": PaddingVariant("neighborhood_then_random", ("uncle", "cousins")),
    "expanding_collateral_then_random": PaddingVariant(
        "expanding_collateral_then_random",
        (),
        search_strategy="expanding_collateral",
    ),
}


@dataclass
class RuntimeContext:
    """Mutable context shared across monkeypatched sibling tests."""

    case_name: str
    variant: PaddingVariant
    source_maps: dict[str, dict[str, list[np.ndarray]]]
    collateral_shells: dict[str, list[dict[str, Any]]]
    audit_rows: list[dict[str, Any]]
    support_energy_share_threshold: float
    current_test_id: str | None = None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=BATTERY_CASES,
        help="Case names to run. Defaults to the 7-case discriminative battery.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(VARIANTS.keys()),
        choices=sorted(VARIANTS.keys()),
        help="Padding variants to compare.",
    )
    parser.add_argument(
        "--output-dir",
        default=enhancement_lab_results_relative("exp27_neighborhood_padding"),
        help="Directory for CSV/JSON artifacts.",
    )
    parser.add_argument(
        "--sibling-method",
        default=None,
        help="Temporary override for config.SIBLING_TEST_METHOD.",
    )
    parser.add_argument(
        "--sibling-whitening",
        default=None,
        help="Temporary override for config.SIBLING_WHITENING.",
    )
    parser.add_argument(
        "--padding-regime",
        default="all",
        choices=["all", "short_parent_only", "no_parent_only"],
        help="Which padding regime to summarize in aggregate metrics.",
    )
    parser.add_argument(
        "--low-overlap-jaccard-threshold",
        type=float,
        default=0.5,
        help=(
            "Threshold for labeling a non-split mixed node as low-overlap. "
            "Nodes with true_jaccard strictly below this value are flagged."
        ),
    )
    parser.add_argument(
        "--support-energy-share-threshold",
        type=float,
        default=0.2,
        help=(
            "Minimum share of total collateral residual energy required for a "
            "support node to count as energy-significant."
        ),
    )
    parser.add_argument(
        "--mixed-span-similarity-threshold",
        type=float,
        default=0.1,
        help=(
            "Maximum mean subspace similarity for labeling a supported node "
            "as mixed-support rather than single-support."
        ),
    )
    parser.add_argument(
        "--mixed-span-angle-threshold",
        type=float,
        default=75.0,
        help=(
            "Minimum mean principal angle in degrees for labeling a supported "
            "node as mixed-support."
        ),
    )
    parser.add_argument(
        "--mixed-span-rank-ratio-threshold",
        type=float,
        default=0.9,
        help=(
            "Minimum stacked-rank ratio required for labeling a supported node " "as mixed-support."
        ),
    )
    parser.add_argument(
        "--dominant-support-energy-share-threshold",
        type=float,
        default=0.5,
        help=(
            "Minimum max collateral residual-energy share for labeling a "
            "non-mixed supported node as dominant-single-support."
        ),
    )
    parser.add_argument(
        "--gate2-fdr-method",
        default="tree_bh",
        choices=["tree_bh"],
        help="Gate 2 multiple-testing correction. Tree-BH is the only supported option.",
    )
    return parser.parse_args()


def _resolve_pca_head(
    pca_projection: np.ndarray | None,
    pca_eigenvalues: np.ndarray | None,
    target_dim: int,
    n_features: int,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    """Return the PCA head and number of requested padding rows."""
    if pca_projection is None:
        empty_basis = np.empty((0, n_features), dtype=np.float64)
        return empty_basis, None, target_dim

    pca_basis = np.asarray(pca_projection, dtype=np.float64)
    n_used = min(int(pca_basis.shape[0]), int(target_dim))
    head = pca_basis[:n_used]
    eigenvalues = (
        np.asarray(pca_eigenvalues[:n_used], dtype=np.float64)
        if pca_eigenvalues is not None
        else None
    )
    return head, eigenvalues, int(target_dim - n_used)


def _get_uncle(tree, parent: str) -> str | None:
    """Return the parent's uncle node when it exists."""
    predecessors = list(tree.predecessors(parent))
    if not predecessors:
        return None
    grandparent = predecessors[0]
    grandparent_children = list(tree.successors(grandparent))
    if len(grandparent_children) != 2:
        return None
    return grandparent_children[0] if grandparent_children[1] == parent else grandparent_children[1]


def _derive_nonfocal_pca_sources(
    tree,
    raw_pca_projections: dict[str, np.ndarray],
    sibling_dims: dict[str, int],
) -> dict[str, dict[str, list[np.ndarray]]]:
    """Collect non-focal PCA sources keyed by sibling parent node."""
    source_maps: dict[str, dict[str, list[np.ndarray]]] = {
        "uncle": {},
        "cousins": {},
        "neighborhood": {},
    }

    for parent in sibling_dims:
        uncle = _get_uncle(tree, parent)
        if uncle is None:
            continue

        uncle_projections: list[np.ndarray] = []
        cousin_projections: list[np.ndarray] = []

        uncle_projection = raw_pca_projections.get(uncle)
        if uncle_projection is not None:
            uncle_projections.append(np.asarray(uncle_projection, dtype=np.float64))

        for cousin in tree.successors(uncle):
            cousin_projection = raw_pca_projections.get(cousin)
            if cousin_projection is not None:
                cousin_projections.append(np.asarray(cousin_projection, dtype=np.float64))

        if uncle_projections:
            source_maps["uncle"][parent] = uncle_projections
        if cousin_projections:
            source_maps["cousins"][parent] = cousin_projections
        if uncle_projections or cousin_projections:
            source_maps["neighborhood"][parent] = uncle_projections + cousin_projections

    return source_maps


def _derive_collateral_shells(
    tree,
    raw_pca_projections: dict[str, np.ndarray],
    sibling_dims: dict[str, int],
) -> dict[str, list[dict[str, Any]]]:
    """Collect non-focal PCA sources in expanding graph-distance shells."""
    undirected_tree = tree.to_undirected(as_view=True)
    shell_map: dict[str, list[dict[str, Any]]] = {}

    for parent in sibling_dims:
        ancestors = nx.ancestors(tree, parent)
        descendants = nx.descendants(tree, parent)
        distances = nx.single_source_shortest_path_length(undirected_tree, parent)
        shells_by_radius: dict[int, dict[str, Any]] = {}

        for node, distance in distances.items():
            if node == parent or node in ancestors or node in descendants:
                continue

            shell = shells_by_radius.setdefault(
                int(distance),
                {
                    "radius": int(distance),
                    "node_ids": [],
                    "pca_node_ids": [],
                    "pca_entries": [],
                    "projections": [],
                    "leaf_mass": 0,
                    "leaf_mass_with_pca": 0,
                },
            )

            shell["node_ids"].append(node)
            leaf_mass = int(tree.nodes[node].get("leaf_count", 0))
            shell["leaf_mass"] += leaf_mass

            projection = raw_pca_projections.get(node)
            if projection is None:
                continue

            projection_matrix = np.asarray(projection, dtype=np.float64)
            if projection_matrix.ndim != 2 or projection_matrix.shape[0] == 0:
                continue

            shell["pca_node_ids"].append(node)
            shell["pca_entries"].append(
                {
                    "node_id": str(node),
                    "projection": projection_matrix,
                    "leaf_mass": int(leaf_mass),
                }
            )
            shell["projections"].append(projection_matrix)
            shell["leaf_mass_with_pca"] += leaf_mass

        if shells_by_radius:
            shell_map[parent] = [shells_by_radius[radius] for radius in sorted(shells_by_radius)]

    return shell_map


def _stack_valid_rows(
    projections: list[np.ndarray] | None,
    n_features: int,
) -> np.ndarray:
    """Stack projection matrices, keeping only rows with the right width."""
    if not projections:
        return np.empty((0, n_features), dtype=np.float64)

    rows: list[np.ndarray] = []
    for projection in projections:
        matrix = np.asarray(projection, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != n_features or matrix.shape[0] == 0:
            continue
        rows.append(matrix)

    if not rows:
        return np.empty((0, n_features), dtype=np.float64)

    return np.vstack(rows)


def _project_out_basis(candidate_rows: np.ndarray, basis_rows: np.ndarray) -> np.ndarray:
    """Project candidate rows onto the orthogonal complement of basis_rows."""
    if candidate_rows.size == 0 or basis_rows.size == 0:
        return candidate_rows
    return candidate_rows - candidate_rows @ basis_rows.T @ basis_rows


def _extract_local_stage(
    projections: list[np.ndarray] | None,
    *,
    existing_basis: np.ndarray,
    n_features: int,
    max_rows: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build local padding rows from one neighborhood source stage."""
    candidate_rows = _stack_valid_rows(projections, n_features)
    residual_basis, subspace_info = _compute_residual_subspace(
        candidate_rows,
        existing_basis=existing_basis,
    )
    if candidate_rows.shape[0] == 0 or max_rows <= 0:
        return np.empty((0, n_features), dtype=np.float64), {
            "candidate_rows": 0,
            "residual_rank": 0,
            "residual_energy": 0.0,
            "used_residual_energy": 0.0,
            "taken_rows": 0,
        }

    residual_rank = int(subspace_info["residual_rank"])
    singular_values = np.asarray(
        subspace_info.get("residual_singular_values", []),
        dtype=np.float64,
    )
    taken_rows = min(int(max_rows), residual_rank)
    local_rows = residual_basis[:taken_rows]

    return local_rows, {
        "candidate_rows": int(candidate_rows.shape[0]),
        "residual_rank": residual_rank,
        "residual_energy": float(np.sum(singular_values[:residual_rank] ** 2)),
        "used_residual_energy": float(np.sum(singular_values[:taken_rows] ** 2)),
        "taken_rows": int(taken_rows),
    }


def _compute_residual_subspace(
    candidate_rows: np.ndarray,
    *,
    existing_basis: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the parent-orthogonal residual subspace spanned by ``candidate_rows``."""
    n_features = int(existing_basis.shape[1]) if existing_basis.ndim == 2 else 0
    empty_basis = np.empty((0, n_features), dtype=np.float64)
    if candidate_rows.size == 0:
        return empty_basis, {
            "candidate_rows": 0,
            "residual_rank": 0,
            "residual_energy": 0.0,
            "residual_singular_values": [],
        }

    residual = _project_out_basis(candidate_rows, existing_basis)
    _, singular_values, right_vectors = np.linalg.svd(residual, full_matrices=False)
    if len(singular_values) == 0:
        return empty_basis, {
            "candidate_rows": int(candidate_rows.shape[0]),
            "residual_rank": 0,
            "residual_energy": 0.0,
            "residual_singular_values": [],
        }

    tolerance = max(float(singular_values[0]) * 1e-10, 1e-15)
    residual_rank = int(np.sum(singular_values > tolerance))
    residual_basis = (
        np.asarray(right_vectors[:residual_rank], dtype=np.float64)
        if residual_rank > 0
        else empty_basis
    )
    residual_singular_values = np.asarray(
        singular_values[:residual_rank],
        dtype=np.float64,
    )
    return residual_basis, {
        "candidate_rows": int(candidate_rows.shape[0]),
        "residual_rank": int(residual_rank),
        "residual_energy": float(np.sum(residual_singular_values**2)),
        "residual_singular_values": residual_singular_values.tolist(),
    }


def _basis_rank(basis_rows: np.ndarray) -> int:
    """Return the numerical rank of a row-basis matrix."""
    if basis_rows.size == 0 or basis_rows.shape[0] == 0:
        return 0
    _, singular_values, _ = np.linalg.svd(basis_rows, full_matrices=False)
    if len(singular_values) == 0:
        return 0
    tolerance = max(float(singular_values[0]) * 1e-10, 1e-15)
    return int(np.sum(singular_values > tolerance))


def _pairwise_subspace_metrics(
    left_basis: np.ndarray,
    right_basis: np.ndarray,
) -> tuple[float, float, float]:
    """Return similarity and angle diagnostics for two residual subspaces."""
    if left_basis.shape[0] == 0 or right_basis.shape[0] == 0:
        return float("nan"), float("nan"), float("nan")

    cosines = np.linalg.svd(left_basis @ right_basis.T, compute_uv=False)
    if len(cosines) == 0:
        return float("nan"), float("nan"), float("nan")

    cosines = np.clip(np.asarray(cosines, dtype=np.float64), -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(cosines))
    denominator = max(min(int(left_basis.shape[0]), int(right_basis.shape[0])), 1)
    similarity = float(np.sum(cosines**2) / denominator)
    return (
        similarity,
        float(np.mean(angles_deg)),
        float(np.max(angles_deg)),
    )


def _analyze_collateral_span_structure(
    visited_shells: list[dict[str, Any]],
    *,
    reference_basis: np.ndarray,
    n_features: int,
    support_energy_share_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Diagnose whether visited collateral PCA nodes share or mix residual spans."""
    node_rows: list[dict[str, Any]] = []
    support_indices: list[int] = []
    support_bases: list[np.ndarray] = []

    for shell in visited_shells:
        radius = int(shell["radius"])
        for entry in shell.get("pca_entries", []):
            candidate_rows = _stack_valid_rows(
                [np.asarray(entry["projection"], dtype=np.float64)],
                int(n_features),
            )
            residual_basis, info = _compute_residual_subspace(
                candidate_rows,
                existing_basis=reference_basis,
            )
            row = {
                "collateral_node": str(entry["node_id"]),
                "collateral_radius": radius,
                "collateral_leaf_mass": int(entry["leaf_mass"]),
                "candidate_rows": int(info["candidate_rows"]),
                "residual_rank": int(info["residual_rank"]),
                "residual_energy": float(info["residual_energy"]),
                "residual_singular_values": json.dumps(
                    [float(value) for value in info["residual_singular_values"]],
                    separators=(",", ":"),
                ),
                "mean_subspace_similarity_to_others": float("nan"),
                "mean_principal_angle_deg_to_others": float("nan"),
                "max_principal_angle_deg_to_others": float("nan"),
                "unique_rank_contribution": 0,
                "unique_rank_share": float("nan"),
                "span_outlier_score": float("nan"),
            }
            node_rows.append(row)
            if int(info["residual_rank"]) <= 0:
                continue
            support_indices.append(len(node_rows) - 1)
            support_bases.append(residual_basis)

    if not node_rows:
        return {
            "collateral_pca_nodes_visited": 0,
            "collateral_residual_support_nodes": 0,
            "collateral_residual_rank_sum": 0,
            "collateral_stacked_residual_rank": 0,
            "collateral_stacked_rank_ratio": float("nan"),
            "collateral_total_node_residual_energy": 0.0,
            "collateral_max_node_residual_energy_share": float("nan"),
            "collateral_energy_significant_support_nodes": 0,
            "collateral_pairwise_mean_similarity": float("nan"),
            "collateral_pairwise_mean_principal_angle_deg": float("nan"),
            "collateral_pairwise_max_principal_angle_deg": float("nan"),
            "collateral_span_mixture_index": float("nan"),
        }, []

    support_rank_sum = int(sum(int(node_rows[index]["residual_rank"]) for index in support_indices))
    stacked_residual_rank = _basis_rank(np.vstack(support_bases)) if support_bases else 0
    rank_ratio = (
        float(stacked_residual_rank / support_rank_sum) if support_rank_sum > 0 else float("nan")
    )
    total_node_residual_energy = float(
        sum(float(node_rows[index]["residual_energy"]) for index in support_indices)
    )
    max_node_residual_energy_share = float("nan")
    energy_significant_support_nodes = 0

    pairwise_similarities: list[float] = []
    pairwise_mean_angles: list[float] = []
    pairwise_max_angles: list[float] = []
    per_node_similarities: dict[int, list[float]] = {index: [] for index in support_indices}
    per_node_mean_angles: dict[int, list[float]] = {index: [] for index in support_indices}
    per_node_max_angles: dict[int, list[float]] = {index: [] for index in support_indices}

    for left_offset, left_index in enumerate(support_indices):
        for right_offset in range(left_offset + 1, len(support_indices)):
            right_index = support_indices[right_offset]
            similarity, mean_angle_deg, max_angle_deg = _pairwise_subspace_metrics(
                support_bases[left_offset],
                support_bases[right_offset],
            )
            if not np.isfinite(similarity):
                continue
            pairwise_similarities.append(float(similarity))
            pairwise_mean_angles.append(float(mean_angle_deg))
            pairwise_max_angles.append(float(max_angle_deg))
            per_node_similarities[left_index].append(float(similarity))
            per_node_similarities[right_index].append(float(similarity))
            per_node_mean_angles[left_index].append(float(mean_angle_deg))
            per_node_mean_angles[right_index].append(float(mean_angle_deg))
            per_node_max_angles[left_index].append(float(max_angle_deg))
            per_node_max_angles[right_index].append(float(max_angle_deg))

    for support_offset, node_index in enumerate(support_indices):
        row = node_rows[node_index]
        similarities = per_node_similarities[node_index]
        mean_angles = per_node_mean_angles[node_index]
        max_angles = per_node_max_angles[node_index]
        row["mean_subspace_similarity_to_others"] = _mean_or_nan(similarities)
        row["mean_principal_angle_deg_to_others"] = _mean_or_nan(mean_angles)
        row["max_principal_angle_deg_to_others"] = _mean_or_nan(max_angles)
        energy_share = (
            float(float(row["residual_energy"]) / total_node_residual_energy)
            if total_node_residual_energy > 0.0
            else float("nan")
        )
        row["residual_energy_share"] = energy_share

        if len(support_bases) == 1:
            unique_rank_contribution = int(stacked_residual_rank)
        else:
            remaining = [basis for idx, basis in enumerate(support_bases) if idx != support_offset]
            rank_without = _basis_rank(np.vstack(remaining)) if remaining else 0
            unique_rank_contribution = int(stacked_residual_rank - rank_without)

        row["unique_rank_contribution"] = int(unique_rank_contribution)
        row["unique_rank_share"] = (
            float(unique_rank_contribution / stacked_residual_rank)
            if stacked_residual_rank > 0
            else float("nan")
        )

        similarity_penalty = (
            1.0 - float(row["mean_subspace_similarity_to_others"])
            if np.isfinite(float(row["mean_subspace_similarity_to_others"]))
            else 1.0
        )
        row["span_outlier_score"] = float(similarity_penalty * max(unique_rank_contribution, 0))
        if np.isfinite(energy_share):
            max_node_residual_energy_share = (
                energy_share
                if not np.isfinite(max_node_residual_energy_share)
                else max(max_node_residual_energy_share, energy_share)
            )
            if energy_share >= support_energy_share_threshold:
                energy_significant_support_nodes += 1

    pairwise_mean_similarity = _mean_or_nan(pairwise_similarities)
    span_mixture_index = (
        float(rank_ratio * (1.0 - pairwise_mean_similarity))
        if np.isfinite(rank_ratio) and np.isfinite(pairwise_mean_similarity)
        else float("nan")
    )

    return {
        "collateral_pca_nodes_visited": int(len(node_rows)),
        "collateral_residual_support_nodes": int(len(support_indices)),
        "collateral_residual_rank_sum": int(support_rank_sum),
        "collateral_stacked_residual_rank": int(stacked_residual_rank),
        "collateral_stacked_rank_ratio": float(rank_ratio),
        "collateral_total_node_residual_energy": float(total_node_residual_energy),
        "collateral_max_node_residual_energy_share": float(max_node_residual_energy_share),
        "collateral_energy_significant_support_nodes": int(energy_significant_support_nodes),
        "collateral_pairwise_mean_similarity": float(pairwise_mean_similarity),
        "collateral_pairwise_mean_principal_angle_deg": float(_mean_or_nan(pairwise_mean_angles)),
        "collateral_pairwise_max_principal_angle_deg": float(_mean_or_nan(pairwise_max_angles)),
        "collateral_span_mixture_index": float(span_mixture_index),
    }, node_rows


def _generate_random_basis(
    n_features: int,
    k: int,
    *,
    random_state: int | None,
) -> np.ndarray:
    """Generate an orthonormal random fallback basis."""
    return generate_projection_matrix_backend(
        int(n_features),
        int(k),
        random_state=random_state,
        use_cache=False,
    )


def _run_expanding_collateral_search(
    shells: list[dict[str, Any]] | None,
    *,
    existing_basis: np.ndarray,
    n_features: int,
    requested_padding_rows: int,
    support_energy_share_threshold: float,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Expand shell-by-shell until the requested local padding rank is filled."""
    if not shells or requested_padding_rows <= 0:
        empty_span_summary = {
            "collateral_pca_nodes_visited": 0,
            "collateral_residual_support_nodes": 0,
            "collateral_residual_rank_sum": 0,
            "collateral_stacked_residual_rank": 0,
            "collateral_stacked_rank_ratio": float("nan"),
            "collateral_total_node_residual_energy": 0.0,
            "collateral_max_node_residual_energy_share": float("nan"),
            "collateral_energy_significant_support_nodes": 0,
            "collateral_pairwise_mean_similarity": float("nan"),
            "collateral_pairwise_mean_principal_angle_deg": float("nan"),
            "collateral_pairwise_max_principal_angle_deg": float("nan"),
            "collateral_span_mixture_index": float("nan"),
        }
        return [], {
            "shell_diagnostics": [],
            "nodes_visited": 0,
            "nodes_with_pca_visited": 0,
            "leaf_mass_visited": 0,
            "leaf_mass_with_pca_visited": 0,
            "local_rows_filled": 0,
            "residual_energy": 0.0,
            "used_residual_energy": 0.0,
            "radius_needed": math.nan,
            "max_local_radius_used": math.nan,
            "local_fill_ratio": 0.0 if requested_padding_rows > 0 else 1.0,
            "used_radii": [],
            "collateral_span_summary": empty_span_summary,
            "collateral_span_node_diagnostics": [],
        }

    local_parts: list[np.ndarray] = []
    shell_diagnostics: list[dict[str, Any]] = []
    nodes_visited = 0
    nodes_with_pca_visited = 0
    leaf_mass_visited = 0
    leaf_mass_with_pca_visited = 0
    local_rows_filled = 0
    residual_energy = 0.0
    used_residual_energy = 0.0
    radius_needed: float | None = None
    max_local_radius_used: float | None = None
    used_radii: list[int] = []
    visited_shells: list[dict[str, Any]] = []
    remaining_padding = int(requested_padding_rows)
    current_basis = existing_basis.copy()

    for shell in shells:
        visited_shells.append(shell)
        radius = int(shell["radius"])
        nodes_visited += int(len(shell["node_ids"]))
        nodes_with_pca_visited += int(len(shell["pca_node_ids"]))
        leaf_mass_visited += int(shell["leaf_mass"])
        leaf_mass_with_pca_visited += int(shell["leaf_mass_with_pca"])

        local_rows, stage_info = _extract_local_stage(
            shell["projections"],
            existing_basis=current_basis,
            n_features=int(n_features),
            max_rows=remaining_padding,
        )

        residual_energy += float(stage_info["residual_energy"])
        used_residual_energy += float(stage_info["used_residual_energy"])

        shell_diag = {
            "radius": radius,
            "nodes": int(len(shell["node_ids"])),
            "nodes_with_pca": int(len(shell["pca_node_ids"])),
            "leaf_mass": int(shell["leaf_mass"]),
            "leaf_mass_with_pca": int(shell["leaf_mass_with_pca"]),
            "rank_gain": int(local_rows.shape[0]),
            "candidate_rows": int(stage_info["candidate_rows"]),
            "residual_rank": int(stage_info["residual_rank"]),
            "residual_energy": float(stage_info["residual_energy"]),
            "used_residual_energy": float(stage_info["used_residual_energy"]),
        }
        shell_diagnostics.append(shell_diag)

        if local_rows.shape[0] == 0:
            continue

        local_parts.append(local_rows)
        current_basis = (
            local_rows if current_basis.shape[0] == 0 else np.vstack([current_basis, local_rows])
        )
        local_rows_filled += int(local_rows.shape[0])
        remaining_padding -= int(local_rows.shape[0])
        used_radii.append(radius)
        max_local_radius_used = float(radius)
        if remaining_padding <= 0:
            radius_needed = float(radius)
            break

    span_summary, span_node_diagnostics = _analyze_collateral_span_structure(
        visited_shells,
        reference_basis=existing_basis,
        n_features=int(n_features),
        support_energy_share_threshold=float(support_energy_share_threshold),
    )
    return local_parts, {
        "shell_diagnostics": shell_diagnostics,
        "nodes_visited": int(nodes_visited),
        "nodes_with_pca_visited": int(nodes_with_pca_visited),
        "leaf_mass_visited": int(leaf_mass_visited),
        "leaf_mass_with_pca_visited": int(leaf_mass_with_pca_visited),
        "local_rows_filled": int(local_rows_filled),
        "residual_energy": float(residual_energy),
        "used_residual_energy": float(used_residual_energy),
        "radius_needed": float(radius_needed) if radius_needed is not None else math.nan,
        "max_local_radius_used": (
            float(max_local_radius_used) if max_local_radius_used is not None else math.nan
        ),
        "local_fill_ratio": (
            float(local_rows_filled / requested_padding_rows) if requested_padding_rows > 0 else 1.0
        ),
        "used_radii": used_radii,
        "collateral_span_summary": span_summary,
        "collateral_span_node_diagnostics": span_node_diagnostics,
    }


def _finalize_audit_row(audit_row: dict[str, Any]) -> dict[str, Any]:
    """Derive a compact local-span recovery score for one padding audit row.

    The score is intentionally heuristic and diagnostic-only:

    - higher ``local_fill_ratio`` improves the score
    - larger search radius penalizes the score
    - visiting more collateral nodes penalizes the score
    - visiting more collateral leaf mass penalizes the score mildly

    ``requested_padding_rows == 0`` yields ``NaN`` because no recovery was needed.
    """
    requested_padding_rows = int(audit_row.get("requested_padding_rows", 0) or 0)
    if requested_padding_rows <= 0:
        audit_row["local_span_search_effort"] = 0.0
        audit_row["local_span_recovery_score"] = math.nan
        return audit_row

    local_fill_ratio = float(audit_row.get("local_fill_ratio", 0.0) or 0.0)
    nodes_visited = float(audit_row.get("nodes_visited", 0.0) or 0.0)
    leaf_mass_visited = float(audit_row.get("leaf_mass_visited", 0.0) or 0.0)

    radius_needed_raw = audit_row.get("radius_needed", math.nan)
    max_local_radius_raw = audit_row.get("max_local_radius_used", math.nan)

    radius_needed = (
        float(radius_needed_raw)
        if radius_needed_raw not in (None, "") and np.isfinite(float(radius_needed_raw))
        else math.nan
    )
    max_local_radius_used = (
        float(max_local_radius_raw)
        if max_local_radius_raw not in (None, "") and np.isfinite(float(max_local_radius_raw))
        else math.nan
    )

    if (
        not np.isfinite(radius_needed)
        and np.isfinite(max_local_radius_used)
        and local_fill_ratio > 0.0
    ):
        radius_proxy = max_local_radius_used + 1.0
    else:
        radius_proxy = radius_needed

    if local_fill_ratio <= 0.0:
        search_effort = math.inf
        recovery_score = 0.0
    else:
        radius_component = max(radius_proxy - 1.0, 0.0) if np.isfinite(radius_proxy) else 0.0
        node_component = math.log1p(max(nodes_visited, 0.0))
        mass_component = 0.25 * math.log1p(max(leaf_mass_visited, 0.0))
        search_effort = radius_component + node_component + mass_component
        recovery_score = float(local_fill_ratio / (1.0 + search_effort))

    audit_row["local_span_search_effort"] = (
        float(search_effort) if np.isfinite(search_effort) else math.inf
    )
    audit_row["local_span_recovery_score"] = float(recovery_score)
    return audit_row


def _make_custom_builder(
    context: RuntimeContext,
    original_builder,
):
    """Create a projection-basis builder for one neighborhood-padding variant."""

    def custom_builder(
        n_features: int,
        k: int,
        *,
        pca_projection: np.ndarray | None = None,
        pca_eigenvalues: np.ndarray | None = None,
        child_pca_projections: list[np.ndarray] | None = None,
        random_state: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        del child_pca_projections

        test_id = context.current_test_id
        if test_id is None or not test_id.startswith("sibling:"):
            return original_builder(
                n_features=n_features,
                k=k,
                pca_projection=pca_projection,
                pca_eigenvalues=pca_eigenvalues,
                child_pca_projections=None,
                random_state=random_state,
            )

        parent = test_id.split(":", 1)[1]
        target_dim = int(k)
        pca_head, whitening_eigenvalues, requested_padding_rows = _resolve_pca_head(
            pca_projection,
            pca_eigenvalues,
            target_dim,
            int(n_features),
        )

        audit_row: dict[str, Any] = {
            "case": context.case_name,
            "variant": context.variant.name,
            "parent": parent,
            "target_dim": int(target_dim),
            "parent_pca_rows": int(pca_head.shape[0]),
            "requested_padding_rows": int(requested_padding_rows),
            "no_parent_pca": bool(pca_projection is None),
            "source_sequence": ">".join(context.variant.source_order) or "random_only",
        }

        if requested_padding_rows <= 0:
            audit_row.update(
                {
                    "local_rows_filled": 0,
                    "local_fill_ratio": 1.0,
                    "used_random_fallback": False,
                    "radius_needed": 0.0,
                    "max_local_radius_used": 0.0,
                    "used_sources": "",
                    "residual_energy": 0.0,
                    "used_residual_energy": 0.0,
                    "returned_basis_rows": int(pca_head.shape[0]),
                }
            )
            context.audit_rows.append(_finalize_audit_row(audit_row))
            return pca_head, whitening_eigenvalues

        if context.variant.search_strategy == "expanding_collateral":
            local_parts, search_info = _run_expanding_collateral_search(
                context.collateral_shells.get(parent),
                existing_basis=pca_head,
                n_features=int(n_features),
                requested_padding_rows=int(requested_padding_rows),
                support_energy_share_threshold=context.support_energy_share_threshold,
            )

            basis_parts: list[np.ndarray] = []
            if pca_head.shape[0] > 0:
                basis_parts.append(pca_head)
            if local_parts:
                basis_parts.extend(local_parts)

            remaining_padding = int(requested_padding_rows - search_info["local_rows_filled"])
            used_random_fallback = False
            if remaining_padding > 0 and context.variant.use_random_fallback:
                used_random_fallback = True
                basis_parts.append(
                    _generate_random_basis(
                        int(n_features),
                        int(remaining_padding),
                        random_state=random_state,
                    )
                )

            projection_matrix = basis_parts[0] if len(basis_parts) == 1 else np.vstack(basis_parts)
            audit_row.update(
                {
                    "local_rows_filled": int(search_info["local_rows_filled"]),
                    "local_fill_ratio": float(search_info["local_fill_ratio"]),
                    "used_random_fallback": bool(used_random_fallback),
                    "radius_needed": float(search_info["radius_needed"]),
                    "max_local_radius_used": float(search_info["max_local_radius_used"]),
                    "used_sources": "expanding_collateral",
                    "used_radii": json.dumps(search_info["used_radii"]),
                    "shell_diagnostics": json.dumps(
                        search_info["shell_diagnostics"], separators=(",", ":")
                    ),
                    "nodes_visited": int(search_info["nodes_visited"]),
                    "nodes_with_pca_visited": int(search_info["nodes_with_pca_visited"]),
                    "leaf_mass_visited": int(search_info["leaf_mass_visited"]),
                    "leaf_mass_with_pca_visited": int(search_info["leaf_mass_with_pca_visited"]),
                    "residual_energy": float(search_info["residual_energy"]),
                    "used_residual_energy": float(search_info["used_residual_energy"]),
                    "collateral_pca_nodes_visited": int(
                        search_info["collateral_span_summary"]["collateral_pca_nodes_visited"]
                    ),
                    "collateral_residual_support_nodes": int(
                        search_info["collateral_span_summary"]["collateral_residual_support_nodes"]
                    ),
                    "collateral_residual_rank_sum": int(
                        search_info["collateral_span_summary"]["collateral_residual_rank_sum"]
                    ),
                    "collateral_stacked_residual_rank": int(
                        search_info["collateral_span_summary"]["collateral_stacked_residual_rank"]
                    ),
                    "collateral_stacked_rank_ratio": float(
                        search_info["collateral_span_summary"]["collateral_stacked_rank_ratio"]
                    ),
                    "collateral_total_node_residual_energy": float(
                        search_info["collateral_span_summary"][
                            "collateral_total_node_residual_energy"
                        ]
                    ),
                    "collateral_max_node_residual_energy_share": float(
                        search_info["collateral_span_summary"][
                            "collateral_max_node_residual_energy_share"
                        ]
                    ),
                    "collateral_energy_significant_support_nodes": int(
                        search_info["collateral_span_summary"][
                            "collateral_energy_significant_support_nodes"
                        ]
                    ),
                    "collateral_pairwise_mean_similarity": float(
                        search_info["collateral_span_summary"][
                            "collateral_pairwise_mean_similarity"
                        ]
                    ),
                    "collateral_pairwise_mean_principal_angle_deg": float(
                        search_info["collateral_span_summary"][
                            "collateral_pairwise_mean_principal_angle_deg"
                        ]
                    ),
                    "collateral_pairwise_max_principal_angle_deg": float(
                        search_info["collateral_span_summary"][
                            "collateral_pairwise_max_principal_angle_deg"
                        ]
                    ),
                    "collateral_span_mixture_index": float(
                        search_info["collateral_span_summary"]["collateral_span_mixture_index"]
                    ),
                    "collateral_span_node_diagnostics": json.dumps(
                        search_info["collateral_span_node_diagnostics"],
                        separators=(",", ":"),
                    ),
                    "returned_basis_rows": int(projection_matrix.shape[0]),
                }
            )
            context.audit_rows.append(_finalize_audit_row(audit_row))
            return projection_matrix, whitening_eigenvalues

        if not context.variant.source_order:
            projection_matrix, whitening_values = original_builder(
                n_features=n_features,
                k=k,
                pca_projection=pca_projection,
                pca_eigenvalues=pca_eigenvalues,
                child_pca_projections=None,
                random_state=random_state,
            )
            audit_row.update(
                {
                    "local_rows_filled": 0,
                    "local_fill_ratio": 0.0,
                    "used_random_fallback": True,
                    "radius_needed": math.nan,
                    "max_local_radius_used": math.nan,
                    "used_sources": "",
                    "residual_energy": 0.0,
                    "used_residual_energy": 0.0,
                    "returned_basis_rows": int(projection_matrix.shape[0]),
                }
            )
            context.audit_rows.append(_finalize_audit_row(audit_row))
            return projection_matrix, whitening_values

        basis_parts: list[np.ndarray] = []
        existing_basis = pca_head.copy()
        if pca_head.shape[0] > 0:
            basis_parts.append(pca_head)

        remaining_padding = int(requested_padding_rows)
        used_sources: list[str] = []
        max_local_radius_used: float | None = None
        radius_needed: float | None = None
        residual_energy = 0.0
        used_residual_energy = 0.0

        for source_name in context.variant.source_order:
            projections = context.source_maps.get(source_name, {}).get(parent)
            local_rows, stage_info = _extract_local_stage(
                projections,
                existing_basis=existing_basis,
                n_features=int(n_features),
                max_rows=remaining_padding,
            )
            audit_row[f"{source_name}_candidate_rows"] = stage_info["candidate_rows"]
            audit_row[f"{source_name}_residual_rank"] = stage_info["residual_rank"]
            audit_row[f"{source_name}_residual_energy"] = stage_info["residual_energy"]

            residual_energy += float(stage_info["residual_energy"])
            used_residual_energy += float(stage_info["used_residual_energy"])

            if local_rows.shape[0] == 0:
                continue

            used_sources.append(source_name)
            basis_parts.append(local_rows)
            existing_basis = (
                local_rows
                if existing_basis.shape[0] == 0
                else np.vstack([existing_basis, local_rows])
            )
            remaining_padding -= int(local_rows.shape[0])
            max_local_radius_used = float(SOURCE_RADII[source_name])
            if remaining_padding <= 0:
                radius_needed = float(SOURCE_RADII[source_name])
                break

        local_rows_filled = int(requested_padding_rows - remaining_padding)
        used_random_fallback = False
        if remaining_padding > 0 and context.variant.use_random_fallback:
            used_random_fallback = True
            basis_parts.append(
                _generate_random_basis(
                    int(n_features),
                    int(remaining_padding),
                    random_state=random_state,
                )
            )

        projection_matrix = basis_parts[0] if len(basis_parts) == 1 else np.vstack(basis_parts)

        audit_row.update(
            {
                "local_rows_filled": int(local_rows_filled),
                "local_fill_ratio": (
                    float(local_rows_filled / requested_padding_rows)
                    if requested_padding_rows > 0
                    else 1.0
                ),
                "used_random_fallback": bool(used_random_fallback),
                "radius_needed": (float(radius_needed) if radius_needed is not None else math.nan),
                "max_local_radius_used": (
                    float(max_local_radius_used) if max_local_radius_used is not None else math.nan
                ),
                "used_sources": "+".join(used_sources),
                "residual_energy": float(residual_energy),
                "used_residual_energy": float(used_residual_energy),
                "returned_basis_rows": int(projection_matrix.shape[0]),
            }
        )
        context.audit_rows.append(_finalize_audit_row(audit_row))
        return projection_matrix, whitening_eigenvalues

    return custom_builder


def _make_test_wrapper(
    context: RuntimeContext,
    original_function,
):
    """Wrap sibling tests so the custom basis builder knows the active parent."""

    def wrapped(*args, **kwargs):
        context.current_test_id = kwargs.get("test_id")
        try:
            return original_function(*args, **kwargs)
        finally:
            context.current_test_id = None

    return wrapped


def build_case_sources(
    case_name: str,
    *,
    gate2_fdr_method: str,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, list[np.ndarray]]],
    dict[str, list[dict[str, Any]]],
]:
    """Compute sibling-dim and non-focal PCA availability for one case."""
    tree, data_df, y_true, test_case = build_tree_and_data(case_name)

    edge_annotated_df = annotate_child_parent_divergence(
        tree,
        tree.annotations_df.copy(),
        significance_level_alpha=config.SIBLING_ALPHA,
        fdr_method=gate2_fdr_method,
        leaf_data=data_df,
        minimum_projection_dimension=getattr(config, "PROJECTION_MINIMUM_DIMENSION", None),
    )

    sibling_dims = derive_sibling_spectral_dims(tree, edge_annotated_df) or {}
    parent_pca, _ = derive_sibling_pca_projections(edge_annotated_df, sibling_dims)
    raw_pca = edge_annotated_df.attrs.get("_pca_projections") or {}
    source_maps = _derive_nonfocal_pca_sources(tree, raw_pca, sibling_dims)
    collateral_shells = _derive_collateral_shells(tree, raw_pca, sibling_dims)

    case_context = {
        "tree": tree,
        "data_df": data_df,
        "y_true": y_true,
        "test_case": test_case,
        "edge_annotated_df": edge_annotated_df,
        "sibling_dims": sibling_dims,
        "parent_pca": parent_pca or {},
    }
    return case_context, source_maps, collateral_shells


def _make_gate_annotation_pipeline_override(
    gate2_fdr_method: str,
):
    """Override Gate 2 FDR wiring during ``tree.decompose()`` for lab runs."""

    def wrapped(tree, annotations_df, **kwargs):
        kwargs = dict(kwargs)
        kwargs["fdr_method"] = gate2_fdr_method
        return gate_orchestrator_module.run_gate_annotation_pipeline(
            tree,
            annotations_df,
            **kwargs,
        )

    return wrapped


def run_variant_case(
    case_name: str,
    variant: PaddingVariant,
    *,
    gate2_fdr_method: str,
    low_overlap_jaccard_threshold: float,
    support_energy_share_threshold: float,
    mixed_span_similarity_threshold: float,
    mixed_span_angle_threshold: float,
    mixed_span_rank_ratio_threshold: float,
    dominant_support_energy_share_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run one case under one padding variant."""
    case_context, source_maps, collateral_shells = build_case_sources(
        case_name,
        gate2_fdr_method=gate2_fdr_method,
    )
    tree = case_context["tree"]
    data_df = case_context["data_df"]
    y_true = case_context["y_true"]
    test_case = case_context["test_case"]

    audit_rows: list[dict[str, Any]] = []
    runtime_context = RuntimeContext(
        case_name=case_name,
        variant=variant,
        source_maps=source_maps,
        collateral_shells=collateral_shells,
        audit_rows=audit_rows,
        support_energy_share_threshold=float(support_energy_share_threshold),
    )

    custom_builder = _make_custom_builder(
        runtime_context,
        projected_wald_module.build_projection_basis_with_padding,
    )

    wrapped_test = _make_test_wrapper(
        runtime_context,
        wald_statistic_module.sibling_divergence_test,
    )

    started = time.time()
    with ExitStack() as stack:
        stack.enter_context(
            temporary_attr(
                tree_decomposition_module,
                "run_gate_annotation_pipeline",
                _make_gate_annotation_pipeline_override(gate2_fdr_method),
            )
        )
        stack.enter_context(
            temporary_attr(
                projected_wald_module,
                "build_projection_basis_with_padding",
                custom_builder,
            )
        )
        stack.enter_context(
            temporary_attr(
                wald_statistic_module,
                "sibling_divergence_test",
                wrapped_test,
            )
        )
        stack.enter_context(
            temporary_attr(
                adjusted_wald_module,
                "sibling_divergence_test",
                wrapped_test,
            )
        )
        stack.enter_context(
            temporary_attr(
                sibling_pair_collection_module,
                "sibling_divergence_test",
                wrapped_test,
            )
        )
        decomp = tree.decompose(
            leaf_data=data_df,
            alpha_local=config.SIBLING_ALPHA,
            sibling_alpha=config.SIBLING_ALPHA,
        )

    ari = float("nan")
    ari_error: str | None = None
    if y_true is not None:
        try:
            ari = float(compute_ari(decomp, data_df, y_true))
        except ValueError as err:
            ari_error = str(err)

    row = {
        "case": case_name,
        "variant": variant.name,
        "gate2_fdr_method": gate2_fdr_method,
        "true_k": test_case.get("n_clusters"),
        "found_k": int(decomp["num_clusters"]),
        "ari": ari,
        "elapsed_seconds": time.time() - started,
    }
    if row["true_k"] is not None:
        row["delta_found_k"] = int(row["found_k"]) - int(row["true_k"])
    if ari_error is not None:
        row["ari_error"] = ari_error

    node_rows = build_node_diagnostics(
        variant_name=variant.name,
        tree=tree,
        data_df=data_df,
        y_true=y_true,
        audit_rows=audit_rows,
        gate2_fdr_method=gate2_fdr_method,
        low_overlap_jaccard_threshold=low_overlap_jaccard_threshold,
        support_energy_share_threshold=support_energy_share_threshold,
        mixed_span_similarity_threshold=mixed_span_similarity_threshold,
        mixed_span_angle_threshold=mixed_span_angle_threshold,
        mixed_span_rank_ratio_threshold=mixed_span_rank_ratio_threshold,
        dominant_support_energy_share_threshold=dominant_support_energy_share_threshold,
    )

    return row, audit_rows, node_rows


def _mean_or_nan(values: list[float]) -> float:
    """Return mean or NaN when the list is empty."""
    return float(np.mean(values)) if values else float("nan")


def _median_or_nan(values: list[float]) -> float:
    """Return median or NaN when the list is empty."""
    return float(np.median(values)) if values else float("nan")


def _select_padding_rows(
    audit_rows: list[dict[str, Any]],
    *,
    padding_regime: str,
) -> list[dict[str, Any]]:
    """Filter audit rows down to the padding regime being summarized."""
    padding_rows = [row for row in audit_rows if int(row.get("requested_padding_rows", 0)) > 0]
    if padding_regime == "short_parent_only":
        padding_rows = [row for row in padding_rows if not bool(row.get("no_parent_pca", False))]
    elif padding_regime == "no_parent_only":
        padding_rows = [row for row in padding_rows if bool(row.get("no_parent_pca", False))]
    return padding_rows


def summarize_variant(
    case_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    *,
    padding_regime: str,
) -> dict[str, Any]:
    """Aggregate case-level and padding-level diagnostics for one variant."""
    ari_values = [float(row["ari"]) for row in case_rows if not math.isnan(float(row["ari"]))]
    exact_k = sum(
        1
        for row in case_rows
        if row["true_k"] is not None and int(row["found_k"]) == int(row["true_k"])
    )

    padding_rows = _select_padding_rows(audit_rows, padding_regime=padding_regime)

    local_fill_values = [float(row["local_fill_ratio"]) for row in padding_rows]
    residual_energy_values = [float(row["residual_energy"]) for row in padding_rows]
    used_residual_energy_values = [float(row["used_residual_energy"]) for row in padding_rows]
    nodes_visited_values = [
        float(row["nodes_visited"])
        for row in padding_rows
        if "nodes_visited" in row and row["nodes_visited"] != ""
    ]
    nodes_with_pca_visited_values = [
        float(row["nodes_with_pca_visited"])
        for row in padding_rows
        if "nodes_with_pca_visited" in row and row["nodes_with_pca_visited"] != ""
    ]
    leaf_mass_visited_values = [
        float(row["leaf_mass_visited"])
        for row in padding_rows
        if "leaf_mass_visited" in row and row["leaf_mass_visited"] != ""
    ]
    score_values = [
        float(row["local_span_recovery_score"])
        for row in padding_rows
        if "local_span_recovery_score" in row
        and row["local_span_recovery_score"] != ""
        and np.isfinite(float(row["local_span_recovery_score"]))
    ]
    effort_values = [
        float(row["local_span_search_effort"])
        for row in padding_rows
        if "local_span_search_effort" in row
        and row["local_span_search_effort"] != ""
        and np.isfinite(float(row["local_span_search_effort"]))
    ]
    full_local_fill = [
        row for row in padding_rows if not bool(row.get("used_random_fallback", False))
    ]
    radius_values = [
        float(row["radius_needed"])
        for row in full_local_fill
        if np.isfinite(float(row["radius_needed"]))
    ]

    return {
        "cases": len(case_rows),
        "mean_ari": _mean_or_nan(ari_values),
        "median_ari": _median_or_nan(ari_values),
        "exact_k": int(exact_k),
        "k_eq_1": int(sum(1 for row in case_rows if int(row["found_k"]) == 1)),
        "padding_tests": int(len(padding_rows)),
        "mean_local_fill_ratio": _mean_or_nan(local_fill_values),
        "median_local_fill_ratio": _median_or_nan(local_fill_values),
        "full_local_fill_rate": (
            float(len(full_local_fill) / len(padding_rows)) if padding_rows else float("nan")
        ),
        "mean_residual_energy": _mean_or_nan(residual_energy_values),
        "mean_used_residual_energy": _mean_or_nan(used_residual_energy_values),
        "mean_radius_needed": _mean_or_nan(radius_values),
        "mean_nodes_visited": _mean_or_nan(nodes_visited_values),
        "mean_nodes_with_pca_visited": _mean_or_nan(nodes_with_pca_visited_values),
        "mean_leaf_mass_visited": _mean_or_nan(leaf_mass_visited_values),
        "mean_local_span_search_effort": _mean_or_nan(effort_values),
        "mean_local_span_recovery_score": _mean_or_nan(score_values),
        "median_local_span_recovery_score": _median_or_nan(score_values),
        "random_fallback_rate": (
            float(
                sum(bool(row.get("used_random_fallback", False)) for row in padding_rows)
                / len(padding_rows)
            )
            if padding_rows
            else float("nan")
        ),
        "no_parent_pca_padding_tests": int(
            sum(bool(row.get("no_parent_pca", False)) for row in padding_rows)
        ),
        "short_parent_pca_padding_tests": int(
            sum(not bool(row.get("no_parent_pca", False)) for row in padding_rows)
        ),
        "padding_regime": padding_regime,
    }


def _safe_bool_from_case_row(row: dict[str, Any], key: str) -> bool | None:
    """Parse boolean outcome labels from a case row when the inputs are valid."""
    if key == "exact_k":
        if row.get("true_k") in (None, ""):
            return None
        return int(row["found_k"]) == int(row["true_k"])
    if key == "k_eq_1":
        return int(row["found_k"]) == 1
    if key == "under_split":
        if row.get("true_k") in (None, ""):
            return None
        return int(row["found_k"]) < int(row["true_k"])
    raise KeyError(key)


def aggregate_case_diagnostics(
    case_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    *,
    padding_regime: str,
) -> list[dict[str, Any]]:
    """Aggregate padding diagnostics up to one row per case."""
    selected_padding_rows = _select_padding_rows(audit_rows, padding_regime=padding_regime)
    audit_by_case: dict[str, list[dict[str, Any]]] = {}
    for row in selected_padding_rows:
        audit_by_case.setdefault(str(row["case"]), []).append(row)

    aggregated_rows: list[dict[str, Any]] = []
    for case_row in case_rows:
        case_name = str(case_row["case"])
        case_padding_rows = audit_by_case.get(case_name, [])

        local_fill_values = [float(row["local_fill_ratio"]) for row in case_padding_rows]
        score_values = [
            float(row["local_span_recovery_score"])
            for row in case_padding_rows
            if row.get("local_span_recovery_score", "") not in ("", "nan")
            and np.isfinite(float(row["local_span_recovery_score"]))
        ]
        effort_values = [
            float(row["local_span_search_effort"])
            for row in case_padding_rows
            if row.get("local_span_search_effort", "") not in ("", "nan")
            and np.isfinite(float(row["local_span_search_effort"]))
        ]
        nodes_visited_values = [
            float(row["nodes_visited"])
            for row in case_padding_rows
            if row.get("nodes_visited", "") not in ("", "nan")
        ]
        leaf_mass_visited_values = [
            float(row["leaf_mass_visited"])
            for row in case_padding_rows
            if row.get("leaf_mass_visited", "") not in ("", "nan")
        ]
        radius_values = [
            float(row["radius_needed"])
            for row in case_padding_rows
            if row.get("radius_needed", "") not in ("", "nan")
            and np.isfinite(float(row["radius_needed"]))
        ]

        aggregated_row = dict(case_row)
        aggregated_row.update(
            {
                "padding_regime": padding_regime,
                "padding_tests": int(len(case_padding_rows)),
                "mean_local_fill_ratio": _mean_or_nan(local_fill_values),
                "median_local_fill_ratio": _median_or_nan(local_fill_values),
                "full_local_fill_rate": (
                    float(
                        sum(
                            not bool(row.get("used_random_fallback", False))
                            for row in case_padding_rows
                        )
                        / len(case_padding_rows)
                    )
                    if case_padding_rows
                    else float("nan")
                ),
                "random_fallback_rate": (
                    float(
                        sum(
                            bool(row.get("used_random_fallback", False))
                            for row in case_padding_rows
                        )
                        / len(case_padding_rows)
                    )
                    if case_padding_rows
                    else float("nan")
                ),
                "mean_radius_needed": _mean_or_nan(radius_values),
                "mean_nodes_visited": _mean_or_nan(nodes_visited_values),
                "mean_leaf_mass_visited": _mean_or_nan(leaf_mass_visited_values),
                "mean_local_span_search_effort": _mean_or_nan(effort_values),
                "mean_local_span_recovery_score": _mean_or_nan(score_values),
                "median_local_span_recovery_score": _median_or_nan(score_values),
                "min_local_span_recovery_score": (
                    float(np.min(score_values)) if score_values else float("nan")
                ),
                "max_local_span_recovery_score": (
                    float(np.max(score_values)) if score_values else float("nan")
                ),
                "exact_k_flag": _safe_bool_from_case_row(case_row, "exact_k"),
                "k_eq_1_flag": _safe_bool_from_case_row(case_row, "k_eq_1"),
                "under_split_flag": _safe_bool_from_case_row(case_row, "under_split"),
            }
        )
        aggregated_rows.append(aggregated_row)

    return aggregated_rows


def build_gate2_treebh_edge_diagnostics(
    tree,
    annotations_df,
    *,
    alpha: float,
    fdr_method: str,
) -> dict[str, dict[str, Any]]:
    """Audit whether a child edge was tested or blocked by TreeBH ancestry."""
    if str(fdr_method) != "tree_bh":
        return {}

    raw_test_data = annotations_df.attrs.get("_edge_raw_test_data") or {}
    child_ids = [str(child_id) for child_id in raw_test_data.get("child_ids", [])]
    raw_p_values = np.asarray(raw_test_data.get("p_values", []), dtype=np.float64)

    if not child_ids or raw_p_values.ndim != 1 or raw_p_values.shape[0] != len(child_ids):
        return {}

    p_values_for_correction = np.where(np.isfinite(raw_p_values), raw_p_values, 1.0)
    tree_bh_result = tree_bh_correction(
        tree,
        p_values_for_correction,
        child_ids,
        alpha=float(alpha),
    )
    child_index = {child_id: index for index, child_id in enumerate(child_ids)}

    family_indices_by_parent: dict[str, list[int]] = {}
    for index, child_id in enumerate(child_ids):
        predecessors = list(tree.predecessors(child_id))
        if not predecessors:
            continue
        family_indices_by_parent.setdefault(str(predecessors[0]), []).append(index)

    direct_family_diagnostics: dict[str, dict[str, Any]] = {}
    for parent_id, family_indices in family_indices_by_parent.items():
        family_p_values = p_values_for_correction[family_indices]
        family_reject, family_adjusted, _ = benjamini_hochberg_correction(
            family_p_values,
            alpha=float(alpha),
        )
        family_children = [child_ids[index] for index in family_indices]
        direct_family_diagnostics[parent_id] = {
            "children": family_children,
            "reject": family_reject.astype(bool),
            "adjusted": family_adjusted.astype(float),
        }

    def _find_blocking_ancestor(target_child: str) -> dict[str, Any]:
        current = str(target_child)
        generations_above_edge = 0
        while True:
            predecessors = list(tree.predecessors(current))
            if not predecessors:
                return {}

            parent_id = str(predecessors[0])
            family_result = tree_bh_result.family_results.get(parent_id)
            if family_result is not None:
                family_children = [str(child_id) for child_id in family_result.get("child_ids", [])]
                try:
                    within_family_index = family_children.index(current)
                except ValueError:
                    within_family_index = -1

                if within_family_index >= 0 and not bool(
                    family_result.get("rejected", [])[within_family_index]
                ):
                    blocking_child = current
                    blocking_index = child_index.get(blocking_child)
                    blocking_adjusted_p = (
                        float(tree_bh_result.adjusted_p[blocking_index])
                        if blocking_index is not None
                        else float("nan")
                    )
                    return {
                        "blocking_parent": parent_id,
                        "blocking_child": blocking_child,
                        "blocking_child_raw_p": float(
                            family_result.get("p_values", [])[within_family_index]
                        ),
                        "blocking_child_tree_bh_adjusted_p": blocking_adjusted_p,
                        "blocking_parent_adjusted_alpha": float(
                            family_result.get("adjusted_alpha", float("nan"))
                        ),
                        "blocking_generations_above_edge": int(generations_above_edge),
                    }

            current = parent_id
            generations_above_edge += 1

    diagnostics: dict[str, dict[str, Any]] = {}
    for child_id, index in child_index.items():
        predecessors = list(tree.predecessors(child_id))
        parent_id = str(predecessors[0]) if predecessors else None
        family_diag = direct_family_diagnostics.get(parent_id or "", {})
        family_children = family_diag.get("children", [])
        try:
            within_family_index = family_children.index(child_id)
        except ValueError:
            within_family_index = -1

        direct_family_flat_adjusted_p = (
            float(family_diag["adjusted"][within_family_index])
            if within_family_index >= 0
            else float("nan")
        )
        direct_family_flat_reject = (
            bool(family_diag["reject"][within_family_index]) if within_family_index >= 0 else False
        )
        parent_family_tested = (
            bool(parent_id in tree_bh_result.family_results) if parent_id else False
        )
        blocking_info = _find_blocking_ancestor(child_id)
        ancestor_blocked = bool(
            parent_id is not None and not parent_family_tested and bool(blocking_info)
        )

        diagnostics[child_id] = {
            "child": child_id,
            "parent": parent_id,
            "raw_p_value": float(raw_p_values[index]),
            "tree_bh_adjusted_p": float(tree_bh_result.adjusted_p[index]),
            "tree_bh_reject": bool(
                tree_bh_result.child_parent_edge_null_rejected_by_tree_bh[index]
            ),
            "direct_family_flat_bh_adjusted_p": direct_family_flat_adjusted_p,
            "direct_family_flat_bh_reject": bool(direct_family_flat_reject),
            "parent_family_tested": bool(parent_family_tested),
            "ancestor_blocked": bool(ancestor_blocked),
            **blocking_info,
        }

    return diagnostics


def build_node_diagnostics(
    *,
    variant_name: str,
    tree,
    data_df,
    y_true: np.ndarray | None,
    audit_rows: list[dict[str, Any]],
    gate2_fdr_method: str,
    low_overlap_jaccard_threshold: float,
    support_energy_share_threshold: float,
    mixed_span_similarity_threshold: float,
    mixed_span_angle_threshold: float,
    mixed_span_rank_ratio_threshold: float,
    dominant_support_energy_share_threshold: float,
) -> list[dict[str, Any]]:
    """Build one diagnostic row per padded internal node."""
    annotations_df = tree.annotations_df
    descendant_sets = tree.compute_descendant_sets(use_labels=True)
    gate2_edge_diagnostics = build_gate2_treebh_edge_diagnostics(
        tree,
        annotations_df,
        alpha=config.SIBLING_ALPHA,
        fdr_method=gate2_fdr_method,
    )

    def _is_valid_truth_label(value: Any) -> bool:
        if value is None:
            return False
        try:
            return not bool(np.isnan(value))
        except Exception:
            return True

    true_label_by_leaf = (
        {
            str(leaf_label): true_label
            for leaf_label, true_label in zip(data_df.index.tolist(), y_true, strict=False)
            if _is_valid_truth_label(true_label)
        }
        if y_true is not None
        else {}
    )

    node_rows: list[dict[str, Any]] = []
    for audit_row in audit_rows:
        if int(audit_row.get("requested_padding_rows", 0) or 0) <= 0:
            continue

        parent = str(audit_row["parent"])
        children = list(tree.successors(parent))
        if len(children) != 2:
            continue
        left, right = children

        left_leaves = descendant_sets.get(left, set())
        right_leaves = descendant_sets.get(right, set())
        parent_leaves = descendant_sets.get(parent, set())
        left_edge_diag = gate2_edge_diagnostics.get(str(left), {})
        right_edge_diag = gate2_edge_diagnostics.get(str(right), {})

        left_true_labels = sorted(
            {true_label_by_leaf[leaf] for leaf in left_leaves if leaf in true_label_by_leaf}
        )
        right_true_labels = sorted(
            {true_label_by_leaf[leaf] for leaf in right_leaves if leaf in true_label_by_leaf}
        )
        parent_true_labels = sorted(
            {true_label_by_leaf[leaf] for leaf in parent_leaves if leaf in true_label_by_leaf}
        )

        overlap_true_labels = sorted(set(left_true_labels) & set(right_true_labels))
        union_size = len(set(left_true_labels) | set(right_true_labels))
        true_jaccard = (
            float(len(overlap_true_labels) / union_size) if union_size > 0 else float("nan")
        )

        sibling_different = bool(annotations_df.loc[parent, "Sibling_BH_Different"])
        sibling_skipped = bool(annotations_df.loc[parent, "Sibling_Divergence_Skipped"])
        if sibling_different:
            gate3_decision = "split"
        elif sibling_skipped:
            gate3_decision = "skip"
        else:
            gate3_decision = "merge"

        edge_gate_pass = bool(
            annotations_df.loc[left, "Child_Parent_Divergence_Significant"]
        ) or bool(annotations_df.loc[right, "Child_Parent_Divergence_Significant"])
        left_edge_ancestor_blocked = bool(left_edge_diag.get("ancestor_blocked", False))
        right_edge_ancestor_blocked = bool(right_edge_diag.get("ancestor_blocked", False))
        left_edge_direct_family_flat_reject = bool(
            left_edge_diag.get("direct_family_flat_bh_reject", False)
        )
        right_edge_direct_family_flat_reject = bool(
            right_edge_diag.get("direct_family_flat_bh_reject", False)
        )
        gate2_any_child_ancestor_blocked = bool(
            left_edge_ancestor_blocked or right_edge_ancestor_blocked
        )
        gate2_both_children_ancestor_blocked = bool(
            left_edge_ancestor_blocked and right_edge_ancestor_blocked
        )
        gate2_any_child_direct_family_flat_reject = bool(
            left_edge_direct_family_flat_reject or right_edge_direct_family_flat_reject
        )
        gate2_both_children_direct_family_flat_reject = bool(
            left_edge_direct_family_flat_reject and right_edge_direct_family_flat_reject
        )
        true_disjoint_children = (
            len(left_true_labels) > 0
            and len(right_true_labels) > 0
            and len(overlap_true_labels) == 0
        )
        false_merge_flag = bool(true_disjoint_children and gate3_decision != "split")
        recovered_split_flag = bool(true_disjoint_children and gate3_decision == "split")
        mixed_parent_truth = len(parent_true_labels) > 1
        mixed_merge_flag = bool(mixed_parent_truth and gate3_decision != "split")
        low_overlap_truth = bool(
            np.isfinite(true_jaccard) and true_jaccard < low_overlap_jaccard_threshold
        )
        low_overlap_mixed_parent_truth = bool(mixed_parent_truth and low_overlap_truth)
        low_overlap_mixed_merge_flag = bool(
            low_overlap_mixed_parent_truth and gate3_decision != "split"
        )
        gate2_skip_explained_by_ancestor_block = bool(
            sibling_skipped
            and gate2_both_children_ancestor_blocked
            and gate2_any_child_direct_family_flat_reject
        )
        support_nodes = int(audit_row.get("collateral_residual_support_nodes", 0) or 0)
        total_support_energy = float(
            audit_row.get("collateral_total_node_residual_energy", 0.0) or 0.0
        )
        max_node_energy_share = audit_row.get(
            "collateral_max_node_residual_energy_share",
            math.nan,
        )
        significant_support_nodes = int(
            audit_row.get("collateral_energy_significant_support_nodes", 0) or 0
        )
        pairwise_similarity = audit_row.get("collateral_pairwise_mean_similarity", math.nan)
        pairwise_angle_deg = audit_row.get(
            "collateral_pairwise_mean_principal_angle_deg",
            math.nan,
        )
        stacked_rank_ratio = audit_row.get("collateral_stacked_rank_ratio", math.nan)
        pairwise_similarity = (
            float(pairwise_similarity)
            if pairwise_similarity not in (None, "") and np.isfinite(float(pairwise_similarity))
            else math.nan
        )
        pairwise_angle_deg = (
            float(pairwise_angle_deg)
            if pairwise_angle_deg not in (None, "") and np.isfinite(float(pairwise_angle_deg))
            else math.nan
        )
        stacked_rank_ratio = (
            float(stacked_rank_ratio)
            if stacked_rank_ratio not in (None, "") and np.isfinite(float(stacked_rank_ratio))
            else math.nan
        )
        max_node_energy_share = (
            float(max_node_energy_share)
            if max_node_energy_share not in (None, "") and np.isfinite(float(max_node_energy_share))
            else math.nan
        )

        if support_nodes <= 0 or total_support_energy <= 0.0:
            collateral_support_category = "no_support"
            collateral_support_supercategory = "no_support"
        elif (
            np.isfinite(pairwise_similarity)
            and np.isfinite(pairwise_angle_deg)
            and np.isfinite(stacked_rank_ratio)
            and pairwise_similarity <= mixed_span_similarity_threshold
            and pairwise_angle_deg >= mixed_span_angle_threshold
            and stacked_rank_ratio >= mixed_span_rank_ratio_threshold
        ):
            collateral_support_category = "mixed_support_span"
            collateral_support_supercategory = "mixed_support_span"
        elif support_nodes == 1 or (
            np.isfinite(max_node_energy_share)
            and max_node_energy_share >= dominant_support_energy_share_threshold
        ):
            collateral_support_category = "dominant_single_support"
            collateral_support_supercategory = "single_support_span"
        else:
            collateral_support_category = "weak_diffuse_support"
            collateral_support_supercategory = "single_support_span"

        node_row = dict(audit_row)
        node_row.update(
            {
                "variant": variant_name,
                "gate2_fdr_method": gate2_fdr_method,
                "left_child": str(left),
                "right_child": str(right),
                "parent_leaf_count": int(tree.nodes[parent].get("leaf_count", 0)),
                "left_leaf_count": int(tree.nodes[left].get("leaf_count", 0)),
                "right_leaf_count": int(tree.nodes[right].get("leaf_count", 0)),
                "edge_gate_pass": edge_gate_pass,
                "gate3_decision": gate3_decision,
                "sibling_different_flag": sibling_different,
                "sibling_skipped_flag": sibling_skipped,
                "left_child_edge_raw_p_value": left_edge_diag.get("raw_p_value", math.nan),
                "right_child_edge_raw_p_value": right_edge_diag.get("raw_p_value", math.nan),
                "left_child_edge_tree_bh_adjusted_p": left_edge_diag.get(
                    "tree_bh_adjusted_p",
                    math.nan,
                ),
                "right_child_edge_tree_bh_adjusted_p": right_edge_diag.get(
                    "tree_bh_adjusted_p",
                    math.nan,
                ),
                "left_child_edge_tree_bh_reject": bool(left_edge_diag.get("tree_bh_reject", False)),
                "right_child_edge_tree_bh_reject": bool(
                    right_edge_diag.get("tree_bh_reject", False)
                ),
                "left_child_edge_parent_family_tested": bool(
                    left_edge_diag.get("parent_family_tested", False)
                ),
                "right_child_edge_parent_family_tested": bool(
                    right_edge_diag.get("parent_family_tested", False)
                ),
                "left_child_edge_ancestor_blocked": bool(left_edge_ancestor_blocked),
                "right_child_edge_ancestor_blocked": bool(right_edge_ancestor_blocked),
                "left_child_edge_direct_family_flat_bh_adjusted_p": left_edge_diag.get(
                    "direct_family_flat_bh_adjusted_p",
                    math.nan,
                ),
                "right_child_edge_direct_family_flat_bh_adjusted_p": right_edge_diag.get(
                    "direct_family_flat_bh_adjusted_p",
                    math.nan,
                ),
                "left_child_edge_direct_family_flat_bh_reject": bool(
                    left_edge_direct_family_flat_reject
                ),
                "right_child_edge_direct_family_flat_bh_reject": bool(
                    right_edge_direct_family_flat_reject
                ),
                "left_child_edge_blocking_parent": left_edge_diag.get("blocking_parent", ""),
                "right_child_edge_blocking_parent": right_edge_diag.get("blocking_parent", ""),
                "left_child_edge_blocking_child": left_edge_diag.get("blocking_child", ""),
                "right_child_edge_blocking_child": right_edge_diag.get("blocking_child", ""),
                "left_child_edge_blocking_child_raw_p": left_edge_diag.get(
                    "blocking_child_raw_p",
                    math.nan,
                ),
                "right_child_edge_blocking_child_raw_p": right_edge_diag.get(
                    "blocking_child_raw_p",
                    math.nan,
                ),
                "left_child_edge_blocking_child_tree_bh_adjusted_p": left_edge_diag.get(
                    "blocking_child_tree_bh_adjusted_p",
                    math.nan,
                ),
                "right_child_edge_blocking_child_tree_bh_adjusted_p": right_edge_diag.get(
                    "blocking_child_tree_bh_adjusted_p",
                    math.nan,
                ),
                "left_child_edge_blocking_parent_adjusted_alpha": left_edge_diag.get(
                    "blocking_parent_adjusted_alpha",
                    math.nan,
                ),
                "right_child_edge_blocking_parent_adjusted_alpha": right_edge_diag.get(
                    "blocking_parent_adjusted_alpha",
                    math.nan,
                ),
                "left_child_edge_blocking_generations_above_edge": left_edge_diag.get(
                    "blocking_generations_above_edge",
                    math.nan,
                ),
                "right_child_edge_blocking_generations_above_edge": right_edge_diag.get(
                    "blocking_generations_above_edge",
                    math.nan,
                ),
                "gate2_any_child_ancestor_blocked": bool(gate2_any_child_ancestor_blocked),
                "gate2_both_children_ancestor_blocked": bool(gate2_both_children_ancestor_blocked),
                "gate2_any_child_direct_family_flat_reject": bool(
                    gate2_any_child_direct_family_flat_reject
                ),
                "gate2_both_children_direct_family_flat_reject": bool(
                    gate2_both_children_direct_family_flat_reject
                ),
                "gate2_skip_explained_by_ancestor_block": bool(
                    gate2_skip_explained_by_ancestor_block
                ),
                "parent_true_label_count": int(len(parent_true_labels)),
                "left_true_label_count": int(len(left_true_labels)),
                "right_true_label_count": int(len(right_true_labels)),
                "true_overlap_label_count": int(len(overlap_true_labels)),
                "true_jaccard": float(true_jaccard),
                "true_disjoint_children": bool(true_disjoint_children),
                "mixed_parent_truth": bool(mixed_parent_truth),
                "mixed_merge_flag": bool(mixed_merge_flag),
                "low_overlap_truth": bool(low_overlap_truth),
                "low_overlap_jaccard_threshold": float(low_overlap_jaccard_threshold),
                "low_overlap_mixed_parent_truth": bool(low_overlap_mixed_parent_truth),
                "low_overlap_mixed_merge_flag": bool(low_overlap_mixed_merge_flag),
                "support_energy_share_threshold": float(support_energy_share_threshold),
                "mixed_span_similarity_threshold": float(mixed_span_similarity_threshold),
                "mixed_span_angle_threshold": float(mixed_span_angle_threshold),
                "mixed_span_rank_ratio_threshold": float(mixed_span_rank_ratio_threshold),
                "dominant_support_energy_share_threshold": float(
                    dominant_support_energy_share_threshold
                ),
                "collateral_support_category": collateral_support_category,
                "collateral_support_supercategory": collateral_support_supercategory,
                "no_support_flag": bool(collateral_support_category == "no_support"),
                "dominant_single_support_flag": bool(
                    collateral_support_category == "dominant_single_support"
                ),
                "weak_diffuse_support_flag": bool(
                    collateral_support_category == "weak_diffuse_support"
                ),
                "single_support_span_flag": bool(
                    collateral_support_supercategory == "single_support_span"
                ),
                "mixed_support_span_flag": bool(
                    collateral_support_category == "mixed_support_span"
                ),
                "false_merge_flag": bool(false_merge_flag),
                "recovered_split_flag": bool(recovered_split_flag),
            }
        )
        node_rows.append(node_row)

    return node_rows


def build_collateral_span_rows(
    audit_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten per-parent collateral span diagnostics into one row per collateral node."""
    span_rows: list[dict[str, Any]] = []
    for audit_row in audit_rows:
        raw_node_json = audit_row.get("collateral_span_node_diagnostics")
        if not raw_node_json:
            continue
        try:
            node_entries = json.loads(raw_node_json)
        except Exception:
            continue

        parent_fields = {
            "case": audit_row.get("case"),
            "variant": audit_row.get("variant"),
            "parent": audit_row.get("parent"),
            "requested_padding_rows": audit_row.get("requested_padding_rows"),
            "no_parent_pca": audit_row.get("no_parent_pca"),
            "local_fill_ratio": audit_row.get("local_fill_ratio"),
            "used_random_fallback": audit_row.get("used_random_fallback"),
            "radius_needed": audit_row.get("radius_needed"),
            "max_local_radius_used": audit_row.get("max_local_radius_used"),
            "nodes_visited": audit_row.get("nodes_visited"),
            "nodes_with_pca_visited": audit_row.get("nodes_with_pca_visited"),
            "leaf_mass_visited": audit_row.get("leaf_mass_visited"),
            "collateral_pca_nodes_visited": audit_row.get("collateral_pca_nodes_visited"),
            "collateral_residual_support_nodes": audit_row.get("collateral_residual_support_nodes"),
            "collateral_residual_rank_sum": audit_row.get("collateral_residual_rank_sum"),
            "collateral_stacked_residual_rank": audit_row.get("collateral_stacked_residual_rank"),
            "collateral_stacked_rank_ratio": audit_row.get("collateral_stacked_rank_ratio"),
            "collateral_total_node_residual_energy": audit_row.get(
                "collateral_total_node_residual_energy"
            ),
            "collateral_max_node_residual_energy_share": audit_row.get(
                "collateral_max_node_residual_energy_share"
            ),
            "collateral_energy_significant_support_nodes": audit_row.get(
                "collateral_energy_significant_support_nodes"
            ),
            "collateral_pairwise_mean_similarity": audit_row.get(
                "collateral_pairwise_mean_similarity"
            ),
            "collateral_pairwise_mean_principal_angle_deg": audit_row.get(
                "collateral_pairwise_mean_principal_angle_deg"
            ),
            "collateral_pairwise_max_principal_angle_deg": audit_row.get(
                "collateral_pairwise_max_principal_angle_deg"
            ),
            "collateral_span_mixture_index": audit_row.get("collateral_span_mixture_index"),
        }
        for entry in node_entries:
            row = dict(parent_fields)
            row.update(entry)
            span_rows.append(row)
    return span_rows


def _mean_for_flag(
    rows: list[dict[str, Any]],
    *,
    metric_key: str,
    flag_key: str,
    flag_value: bool,
) -> float:
    """Mean of ``metric_key`` among rows matching ``flag_key == flag_value``."""
    values = [
        float(row[metric_key])
        for row in rows
        if row.get(flag_key) == flag_value
        and row.get(metric_key, "") not in ("", "nan")
        and np.isfinite(float(row[metric_key]))
    ]
    return _mean_or_nan(values)


def _spearman_summary(
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
) -> dict[str, Any]:
    """Compute a compact Spearman summary when enough case data exists."""
    paired_values = [
        (float(row[x_key]), float(row[y_key]))
        for row in rows
        if row.get(x_key, "") not in ("", "nan")
        and row.get(y_key, "") not in ("", "nan")
        and np.isfinite(float(row[x_key]))
        and np.isfinite(float(row[y_key]))
    ]
    if len(paired_values) < 3 or spearmanr is None:
        return {"n": len(paired_values), "rho": float("nan"), "p_value": float("nan")}

    x_values = [pair[0] for pair in paired_values]
    y_values = [pair[1] for pair in paired_values]
    result = spearmanr(x_values, y_values)
    return {
        "n": len(paired_values),
        "rho": float(result.correlation) if result is not None else float("nan"),
        "p_value": float(result.pvalue) if result is not None else float("nan"),
    }


def summarize_case_associations(
    case_diagnostic_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize whether case-level recovery diagnostics track outcomes."""
    informative_rows = [
        row
        for row in case_diagnostic_rows
        if int(row.get("padding_tests", 0)) > 0
        and row.get("mean_local_span_recovery_score", "") not in ("", "nan")
        and np.isfinite(float(row["mean_local_span_recovery_score"]))
    ]
    return {
        "cases_with_padding": int(
            sum(int(row.get("padding_tests", 0)) > 0 for row in case_diagnostic_rows)
        ),
        "cases_with_score": int(len(informative_rows)),
        "score_vs_ari_spearman": _spearman_summary(
            informative_rows,
            x_key="mean_local_span_recovery_score",
            y_key="ari",
        ),
        "effort_vs_ari_spearman": _spearman_summary(
            informative_rows,
            x_key="mean_local_span_search_effort",
            y_key="ari",
        ),
        "mean_score_k_eq_1": _mean_for_flag(
            informative_rows,
            metric_key="mean_local_span_recovery_score",
            flag_key="k_eq_1_flag",
            flag_value=True,
        ),
        "mean_score_not_k_eq_1": _mean_for_flag(
            informative_rows,
            metric_key="mean_local_span_recovery_score",
            flag_key="k_eq_1_flag",
            flag_value=False,
        ),
        "mean_score_exact_k": _mean_for_flag(
            informative_rows,
            metric_key="mean_local_span_recovery_score",
            flag_key="exact_k_flag",
            flag_value=True,
        ),
        "mean_score_not_exact_k": _mean_for_flag(
            informative_rows,
            metric_key="mean_local_span_recovery_score",
            flag_key="exact_k_flag",
            flag_value=False,
        ),
        "mean_score_under_split": _mean_for_flag(
            informative_rows,
            metric_key="mean_local_span_recovery_score",
            flag_key="under_split_flag",
            flag_value=True,
        ),
        "mean_score_not_under_split": _mean_for_flag(
            informative_rows,
            metric_key="mean_local_span_recovery_score",
            flag_key="under_split_flag",
            flag_value=False,
        ),
        "mean_fallback_k_eq_1": _mean_for_flag(
            informative_rows,
            metric_key="random_fallback_rate",
            flag_key="k_eq_1_flag",
            flag_value=True,
        ),
        "mean_fallback_not_k_eq_1": _mean_for_flag(
            informative_rows,
            metric_key="random_fallback_rate",
            flag_key="k_eq_1_flag",
            flag_value=False,
        ),
    }


def summarize_node_associations(
    node_rows: list[dict[str, Any]],
    *,
    padding_regime: str,
) -> dict[str, Any]:
    """Summarize whether node-level recovery diagnostics track false merges."""
    selected_rows = _select_padding_rows(node_rows, padding_regime=padding_regime)
    informative_rows = [
        row
        for row in selected_rows
        if row.get("local_span_recovery_score", "") not in ("", "nan")
        and np.isfinite(float(row["local_span_recovery_score"]))
    ]

    score_values = [float(row["local_span_recovery_score"]) for row in informative_rows]
    median_score = _median_or_nan(score_values)
    low_score_rows = [
        row for row in informative_rows if float(row["local_span_recovery_score"]) <= median_score
    ]
    high_score_rows = [
        row for row in informative_rows if float(row["local_span_recovery_score"]) > median_score
    ]

    def _flag_rate(rows: list[dict[str, Any]], flag_key: str) -> float:
        return float(sum(bool(row[flag_key]) for row in rows) / len(rows)) if rows else float("nan")

    def _spearman_bool(x_key: str, y_key: str) -> dict[str, Any]:
        paired = [
            (float(row[x_key]), float(bool(row[y_key])))
            for row in informative_rows
            if row.get(x_key, "") not in ("", "nan") and np.isfinite(float(row[x_key]))
        ]
        if len(paired) < 3 or spearmanr is None:
            return {"n": len(paired), "rho": float("nan"), "p_value": float("nan")}
        result = spearmanr([x for x, _ in paired], [y for _, y in paired])
        return {
            "n": len(paired),
            "rho": float(result.correlation) if result is not None else float("nan"),
            "p_value": float(result.pvalue) if result is not None else float("nan"),
        }

    def _mean_for_category(
        metric_key: str,
        category: str,
        *,
        category_key: str = "collateral_support_category",
    ) -> float:
        values = [
            float(row[metric_key])
            for row in informative_rows
            if row.get(category_key) == category
            and row.get(metric_key, "") not in ("", "nan")
            and np.isfinite(float(row[metric_key]))
        ]
        return _mean_or_nan(values)

    def _flag_rate_for_category(
        flag_key: str,
        category: str,
        *,
        category_key: str = "collateral_support_category",
    ) -> float:
        category_rows = [row for row in informative_rows if row.get(category_key) == category]
        return _flag_rate(category_rows, flag_key)

    return {
        "padding_nodes": int(len(selected_rows)),
        "informative_nodes": int(len(informative_rows)),
        "true_disjoint_nodes": int(
            sum(bool(row["true_disjoint_children"]) for row in selected_rows)
        ),
        "false_merge_nodes": int(sum(bool(row["false_merge_flag"]) for row in selected_rows)),
        "recovered_split_nodes": int(
            sum(bool(row["recovered_split_flag"]) for row in selected_rows)
        ),
        "mixed_parent_nodes": int(sum(bool(row["mixed_parent_truth"]) for row in selected_rows)),
        "mixed_merge_nodes": int(sum(bool(row["mixed_merge_flag"]) for row in selected_rows)),
        "low_overlap_mixed_parent_nodes": int(
            sum(bool(row["low_overlap_mixed_parent_truth"]) for row in selected_rows)
        ),
        "low_overlap_mixed_merge_nodes": int(
            sum(bool(row["low_overlap_mixed_merge_flag"]) for row in selected_rows)
        ),
        "no_support_nodes": int(
            sum(
                row.get("collateral_support_supercategory") == "no_support" for row in selected_rows
            )
        ),
        "single_support_span_nodes": int(
            sum(
                row.get("collateral_support_supercategory") == "single_support_span"
                for row in selected_rows
            )
        ),
        "mixed_support_span_nodes": int(
            sum(
                row.get("collateral_support_supercategory") == "mixed_support_span"
                for row in selected_rows
            )
        ),
        "any_child_ancestor_blocked_nodes": int(
            sum(bool(row.get("gate2_any_child_ancestor_blocked", False)) for row in selected_rows)
        ),
        "both_children_ancestor_blocked_nodes": int(
            sum(
                bool(row.get("gate2_both_children_ancestor_blocked", False))
                for row in selected_rows
            )
        ),
        "skip_explained_by_ancestor_block_nodes": int(
            sum(
                bool(row.get("gate2_skip_explained_by_ancestor_block", False))
                for row in selected_rows
            )
        ),
        "dominant_single_support_nodes": int(
            sum(
                row.get("collateral_support_category") == "dominant_single_support"
                for row in selected_rows
            )
        ),
        "weak_diffuse_support_nodes": int(
            sum(
                row.get("collateral_support_category") == "weak_diffuse_support"
                for row in selected_rows
            )
        ),
        "mean_score_false_merge": _mean_for_flag(
            informative_rows,
            metric_key="local_span_recovery_score",
            flag_key="false_merge_flag",
            flag_value=True,
        ),
        "mean_score_not_false_merge": _mean_for_flag(
            informative_rows,
            metric_key="local_span_recovery_score",
            flag_key="false_merge_flag",
            flag_value=False,
        ),
        "mean_fallback_false_merge": _mean_for_flag(
            informative_rows,
            metric_key="used_random_fallback",
            flag_key="false_merge_flag",
            flag_value=True,
        ),
        "mean_fallback_not_false_merge": _mean_for_flag(
            informative_rows,
            metric_key="used_random_fallback",
            flag_key="false_merge_flag",
            flag_value=False,
        ),
        "mean_score_mixed_merge": _mean_for_flag(
            informative_rows,
            metric_key="local_span_recovery_score",
            flag_key="mixed_merge_flag",
            flag_value=True,
        ),
        "mean_score_not_mixed_merge": _mean_for_flag(
            informative_rows,
            metric_key="local_span_recovery_score",
            flag_key="mixed_merge_flag",
            flag_value=False,
        ),
        "mean_fallback_mixed_merge": _mean_for_flag(
            informative_rows,
            metric_key="used_random_fallback",
            flag_key="mixed_merge_flag",
            flag_value=True,
        ),
        "mean_fallback_not_mixed_merge": _mean_for_flag(
            informative_rows,
            metric_key="used_random_fallback",
            flag_key="mixed_merge_flag",
            flag_value=False,
        ),
        "mean_score_low_overlap_mixed_merge": _mean_for_flag(
            informative_rows,
            metric_key="local_span_recovery_score",
            flag_key="low_overlap_mixed_merge_flag",
            flag_value=True,
        ),
        "mean_score_not_low_overlap_mixed_merge": _mean_for_flag(
            informative_rows,
            metric_key="local_span_recovery_score",
            flag_key="low_overlap_mixed_merge_flag",
            flag_value=False,
        ),
        "mean_fallback_low_overlap_mixed_merge": _mean_for_flag(
            informative_rows,
            metric_key="used_random_fallback",
            flag_key="low_overlap_mixed_merge_flag",
            flag_value=True,
        ),
        "mean_fallback_not_low_overlap_mixed_merge": _mean_for_flag(
            informative_rows,
            metric_key="used_random_fallback",
            flag_key="low_overlap_mixed_merge_flag",
            flag_value=False,
        ),
        "mean_score_no_support": _mean_for_category(
            "local_span_recovery_score",
            "no_support",
            category_key="collateral_support_supercategory",
        ),
        "mean_score_single_support_span": _mean_for_category(
            "local_span_recovery_score",
            "single_support_span",
            category_key="collateral_support_supercategory",
        ),
        "mean_score_mixed_support_span": _mean_for_category(
            "local_span_recovery_score",
            "mixed_support_span",
            category_key="collateral_support_supercategory",
        ),
        "mean_score_dominant_single_support": _mean_for_category(
            "local_span_recovery_score",
            "dominant_single_support",
        ),
        "mean_score_weak_diffuse_support": _mean_for_category(
            "local_span_recovery_score",
            "weak_diffuse_support",
        ),
        "mean_fallback_no_support": _mean_for_category(
            "used_random_fallback",
            "no_support",
            category_key="collateral_support_supercategory",
        ),
        "mean_fallback_single_support_span": _mean_for_category(
            "used_random_fallback",
            "single_support_span",
            category_key="collateral_support_supercategory",
        ),
        "mean_fallback_mixed_support_span": _mean_for_category(
            "used_random_fallback",
            "mixed_support_span",
            category_key="collateral_support_supercategory",
        ),
        "mean_fallback_dominant_single_support": _mean_for_category(
            "used_random_fallback",
            "dominant_single_support",
        ),
        "mean_fallback_weak_diffuse_support": _mean_for_category(
            "used_random_fallback",
            "weak_diffuse_support",
        ),
        "score_vs_false_merge_spearman": _spearman_bool(
            "local_span_recovery_score",
            "false_merge_flag",
        ),
        "effort_vs_false_merge_spearman": _spearman_bool(
            "local_span_search_effort",
            "false_merge_flag",
        ),
        "score_vs_mixed_merge_spearman": _spearman_bool(
            "local_span_recovery_score",
            "mixed_merge_flag",
        ),
        "effort_vs_mixed_merge_spearman": _spearman_bool(
            "local_span_search_effort",
            "mixed_merge_flag",
        ),
        "score_vs_low_overlap_mixed_merge_spearman": _spearman_bool(
            "local_span_recovery_score",
            "low_overlap_mixed_merge_flag",
        ),
        "effort_vs_low_overlap_mixed_merge_spearman": _spearman_bool(
            "local_span_search_effort",
            "low_overlap_mixed_merge_flag",
        ),
        "false_merge_rate_low_score_half": _flag_rate(low_score_rows, "false_merge_flag"),
        "false_merge_rate_high_score_half": _flag_rate(high_score_rows, "false_merge_flag"),
        "mixed_merge_rate_low_score_half": _flag_rate(low_score_rows, "mixed_merge_flag"),
        "mixed_merge_rate_high_score_half": _flag_rate(high_score_rows, "mixed_merge_flag"),
        "low_overlap_mixed_merge_rate_low_score_half": _flag_rate(
            low_score_rows,
            "low_overlap_mixed_merge_flag",
        ),
        "low_overlap_mixed_merge_rate_high_score_half": _flag_rate(
            high_score_rows,
            "low_overlap_mixed_merge_flag",
        ),
        "low_overlap_mixed_merge_rate_no_support": _flag_rate_for_category(
            "low_overlap_mixed_merge_flag",
            "no_support",
            category_key="collateral_support_supercategory",
        ),
        "low_overlap_mixed_merge_rate_single_support_span": _flag_rate_for_category(
            "low_overlap_mixed_merge_flag",
            "single_support_span",
            category_key="collateral_support_supercategory",
        ),
        "low_overlap_mixed_merge_rate_mixed_support_span": _flag_rate_for_category(
            "low_overlap_mixed_merge_flag",
            "mixed_support_span",
            category_key="collateral_support_supercategory",
        ),
        "low_overlap_mixed_merge_rate_dominant_single_support": _flag_rate_for_category(
            "low_overlap_mixed_merge_flag",
            "dominant_single_support",
        ),
        "low_overlap_mixed_merge_rate_weak_diffuse_support": _flag_rate_for_category(
            "low_overlap_mixed_merge_flag",
            "weak_diffuse_support",
        ),
        "low_overlap_mixed_merge_rate_both_children_ancestor_blocked": _flag_rate(
            [
                row
                for row in selected_rows
                if bool(row.get("gate2_both_children_ancestor_blocked", False))
            ],
            "low_overlap_mixed_merge_flag",
        ),
        "low_overlap_mixed_merge_rate_skip_explained_by_ancestor_block": _flag_rate(
            [
                row
                for row in selected_rows
                if bool(row.get("gate2_skip_explained_by_ancestor_block", False))
            ],
            "low_overlap_mixed_merge_flag",
        ),
        "padding_regime": padding_regime,
        "low_overlap_jaccard_threshold": (
            float(selected_rows[0]["low_overlap_jaccard_threshold"])
            if selected_rows
            else float("nan")
        ),
        "support_energy_share_threshold": (
            float(selected_rows[0]["support_energy_share_threshold"])
            if selected_rows
            else float("nan")
        ),
        "mixed_span_similarity_threshold": (
            float(selected_rows[0]["mixed_span_similarity_threshold"])
            if selected_rows
            else float("nan")
        ),
        "mixed_span_angle_threshold": (
            float(selected_rows[0]["mixed_span_angle_threshold"]) if selected_rows else float("nan")
        ),
        "mixed_span_rank_ratio_threshold": (
            float(selected_rows[0]["mixed_span_rank_ratio_threshold"])
            if selected_rows
            else float("nan")
        ),
        "dominant_support_energy_share_threshold": (
            float(selected_rows[0]["dominant_support_energy_share_threshold"])
            if selected_rows
            else float("nan")
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of dict rows to CSV."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run the neighborhood-padding experiment and write artifacts."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_case_rows: list[dict[str, Any]] = []
    case_diagnostic_rows: list[dict[str, Any]] = []
    node_diagnostic_rows: list[dict[str, Any]] = []
    collateral_span_rows: list[dict[str, Any]] = []
    padding_audit_rows: list[dict[str, Any]] = []
    variant_summaries: dict[str, Any] = {}
    case_associations_by_variant: dict[str, Any] = {}
    node_associations_by_variant: dict[str, Any] = {}

    started = time.time()
    with temporary_config(
        SIBLING_TEST_METHOD=args.sibling_method or config.SIBLING_TEST_METHOD,
        SIBLING_WHITENING=args.sibling_whitening or config.SIBLING_WHITENING,
    ):
        for variant_name in args.variants:
            variant = VARIANTS[variant_name]
            variant_case_rows: list[dict[str, Any]] = []
            variant_audit_rows: list[dict[str, Any]] = []
            variant_node_rows: list[dict[str, Any]] = []

            for index, case_name in enumerate(args.cases, start=1):
                print(
                    f"[{index}/{len(args.cases)}] {variant.name:<24s} {case_name}",
                    flush=True,
                )
                case_row, audit_rows, node_rows = run_variant_case(
                    case_name,
                    variant,
                    low_overlap_jaccard_threshold=args.low_overlap_jaccard_threshold,
                    support_energy_share_threshold=args.support_energy_share_threshold,
                    mixed_span_similarity_threshold=args.mixed_span_similarity_threshold,
                    mixed_span_angle_threshold=args.mixed_span_angle_threshold,
                    mixed_span_rank_ratio_threshold=args.mixed_span_rank_ratio_threshold,
                    dominant_support_energy_share_threshold=args.dominant_support_energy_share_threshold,
                    gate2_fdr_method=args.gate2_fdr_method,
                )
                variant_case_rows.append(case_row)
                variant_audit_rows.extend(audit_rows)
                variant_node_rows.extend(node_rows)

            per_case_rows.extend(variant_case_rows)
            padding_audit_rows.extend(variant_audit_rows)
            node_diagnostic_rows.extend(variant_node_rows)
            collateral_span_rows.extend(build_collateral_span_rows(variant_audit_rows))
            variant_case_diagnostics = aggregate_case_diagnostics(
                variant_case_rows,
                variant_audit_rows,
                padding_regime=args.padding_regime,
            )
            case_diagnostic_rows.extend(variant_case_diagnostics)
            variant_summaries[variant.name] = summarize_variant(
                variant_case_rows,
                variant_audit_rows,
                padding_regime=args.padding_regime,
            )
            case_associations_by_variant[variant.name] = summarize_case_associations(
                variant_case_diagnostics
            )
            node_associations_by_variant[variant.name] = summarize_node_associations(
                variant_node_rows,
                padding_regime=args.padding_regime,
            )

    per_case_csv = output_dir / "per_case.csv"
    case_diag_csv = output_dir / "case_diagnostics.csv"
    node_diag_csv = output_dir / "node_diagnostics.csv"
    collateral_span_csv = output_dir / "collateral_span_nodes.csv"
    padding_csv = output_dir / "padding_audit.csv"
    summary_json = output_dir / "summary.json"

    write_csv(per_case_csv, per_case_rows)
    write_csv(case_diag_csv, case_diagnostic_rows)
    write_csv(node_diag_csv, node_diagnostic_rows)
    write_csv(collateral_span_csv, collateral_span_rows)
    write_csv(padding_csv, padding_audit_rows)

    payload = {
        "cases": list(args.cases),
        "variants": list(args.variants),
        "elapsed_seconds": time.time() - started,
        "config": {
            "SIBLING_TEST_METHOD": args.sibling_method or config.SIBLING_TEST_METHOD,
            "SIBLING_WHITENING": args.sibling_whitening or config.SIBLING_WHITENING,
            "PADDING_REGIME": args.padding_regime,
            "LOW_OVERLAP_JACCARD_THRESHOLD": args.low_overlap_jaccard_threshold,
            "SUPPORT_ENERGY_SHARE_THRESHOLD": args.support_energy_share_threshold,
            "MIXED_SPAN_SIMILARITY_THRESHOLD": args.mixed_span_similarity_threshold,
            "MIXED_SPAN_ANGLE_THRESHOLD": args.mixed_span_angle_threshold,
            "MIXED_SPAN_RANK_RATIO_THRESHOLD": args.mixed_span_rank_ratio_threshold,
            "DOMINANT_SUPPORT_ENERGY_SHARE_THRESHOLD": args.dominant_support_energy_share_threshold,
            "GATE2_FDR_METHOD": args.gate2_fdr_method,
        },
        "summary_by_variant": variant_summaries,
        "case_associations_by_variant": case_associations_by_variant,
        "node_associations_by_variant": node_associations_by_variant,
        "artifacts": {
            "per_case_csv": str(per_case_csv),
            "case_diagnostics_csv": str(case_diag_csv),
            "node_diagnostics_csv": str(node_diag_csv),
            "collateral_span_csv": str(collateral_span_csv),
            "padding_audit_csv": str(padding_csv),
        },
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== SUMMARY ===")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
