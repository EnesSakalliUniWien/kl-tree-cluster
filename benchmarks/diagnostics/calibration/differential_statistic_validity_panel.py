"""Differential statistic-validity diagnostics for selected-edge sibling rows.

This diagnostic asks whether the projected-Wald statistic is locally credible
before treating selected-edge sibling evidence as calibratable. It is
diagnostic-only and does not install a production calibration rule.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from tree_break_selection.hierarchy_analysis.statistics.contrast_covariance import (
    build_contrast_covariance,
)
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.pair_testing.collection.record_collection import (
    collect_sibling_pair_records,
)
from tree_break_selection.tree.feature_space import FeatureSpace
from tree_break_selection.tree.poset_tree import PosetTree

from benchmarks.diagnostics.calibration.production_admissibility_contract import (
    evaluate_production_admissibility_components,
    summarize_production_admissibility_contracts,
)
from benchmarks.validation.selected_edge_type1_geometry import (
    _annotate_edges,
    _build_tree_from_data,
    _case_contract,
    _prepare_tree_for_mode,
    _projection_inputs,
    _select_cases,
    parse_alpha_grid,
    parse_names,
    validate_modes,
)

STUDY_ROLE = "diagnostic_differential_statistic_validity_not_calibration"
SCHEMA_VERSION = "differential_statistic_validity_panel/v1"
GENERATED_BY = "benchmarks.diagnostics.calibration.differential_statistic_validity_panel"
DEFAULT_MODES = ("fixed_tree", "selected_tree")
DEFAULT_EDGE_ALPHA_GRID = (0.001,)
DF_BINS = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, math.inf)
DF_BIN_LABELS = (
    "df_0_1",
    "df_1_2",
    "df_2_4",
    "df_4_8",
    "df_8_16",
    "df_16_32",
    "df_ge32",
)
FAIL_CLOSED_STATUS_PRECEDENCE = (
    "wald_metric_boundary_unstable",
    "whitening_unstable",
    "projection_unstable",
    "selection_coupled",
    "nonsmooth_selection_geometry",
    "tail_misaligned_after_differential_checks",
)


@dataclass(frozen=True)
class DifferentialStatisticValidityConfig:
    """Runtime contract for differential statistic-validity diagnostics."""

    output_dir: Path
    suite: str
    case_names: tuple[str, ...]
    modes: tuple[str, ...]
    edge_alphas: tuple[float, ...]
    sibling_alpha: float
    replicates: int
    base_seed: int
    finite_diff_directions: int = 16
    epsilon_scale: float = 1e-4

    @property
    def rows_path(self) -> Path:
        return self.output_dir / "differential_statistic_validity_rows.csv"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "differential_statistic_validity_summary.csv"

    @property
    def production_components_path(self) -> Path:
        return self.output_dir / "production_admissibility_components.csv"

    @property
    def production_summary_path(self) -> Path:
        return self.output_dir / "production_admissibility_summary.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


def _df_bin(value: float) -> str:
    for lower, upper, label in zip(DF_BINS, DF_BINS[1:], DF_BIN_LABELS):
        if lower < float(value) <= upper:
            return label
    raise ValueError(f"degrees of freedom did not match a bin: {value!r}.")


def _variance_status(variance_floor: float) -> str:
    if not math.isfinite(variance_floor) or variance_floor <= 1e-8:
        return "boundary_unstable"
    if variance_floor <= 1e-4:
        return "near_boundary"
    return "interior"


def compute_fisher_geometry_summary(
    distribution: np.ndarray,
    feature_space: FeatureSpace,
    *,
    ridge: float = 1e-12,
) -> dict[str, object]:
    """Return Fisher/covariance floor diagnostics for one node distribution."""
    values = np.asarray(distribution, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"distribution must be one-dimensional; got {values.shape}.")
    eigenvalues: list[float] = []
    variance_floors: list[float] = []
    for block in feature_space.blocks:
        block_values = values[list(block.column_indices)]
        if block.family == "bernoulli":
            variances = block_values * (1.0 - block_values)
            variance_floors.extend(float(value) for value in variances)
            eigenvalues.extend(float(value) for value in variances)
        elif block.family == "categorical":
            p = np.asarray(block_values[:-1], dtype=float)
            covariance = np.diag(p) - np.outer(p, p)
            block_eigenvalues = np.linalg.eigvalsh(0.5 * (covariance + covariance.T))
            positive = block_eigenvalues[block_eigenvalues > 0.0]
            variance_floors.append(float(np.min(positive)) if positive.size else 0.0)
            eigenvalues.extend(float(value) for value in positive)
        else:
            raise ValueError(
                "Differential statistic validity V1 supports Bernoulli and "
                f"categorical feature spaces only; got block family {block.family!r}."
            )
    finite_eigenvalues = np.asarray(eigenvalues, dtype=float)
    positive = finite_eigenvalues[finite_eigenvalues > 0.0]
    variance_floor = (
        float(min(variance_floors)) if variance_floors else float("nan")
    )
    condition = (
        math.inf
        if not math.isfinite(variance_floor) or variance_floor <= 0.0
        else (
            float(np.max(positive + ridge) / np.min(positive + ridge))
            if positive.size
            else math.inf
        )
    )
    return {
        "fisher_variance_floor": variance_floor,
        "fisher_condition_number": condition,
        "fisher_boundary_status": _variance_status(variance_floor),
    }


def projected_quadratic_statistic(z: np.ndarray, projection: np.ndarray) -> float:
    """Return ``||P z||^2`` for a row-orthonormal projection matrix."""
    vector = np.asarray(z, dtype=float)
    matrix = np.asarray(projection, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"z must be one-dimensional; got {vector.shape}.")
    if matrix.ndim != 2 or matrix.shape[1] != vector.shape[0]:
        raise ValueError(
            "projection must be a 2-D matrix with width matching z; "
            f"got projection={matrix.shape}, z={vector.shape}."
        )
    return float(np.sum((matrix @ vector) ** 2))


def projected_quadratic_directional_derivative(
    z: np.ndarray,
    projection: np.ndarray,
    direction: np.ndarray,
) -> float:
    """Return the analytic directional derivative of ``||Pz||^2``."""
    vector = np.asarray(z, dtype=float)
    matrix = np.asarray(projection, dtype=float)
    tangent = np.asarray(direction, dtype=float)
    if tangent.shape != vector.shape:
        raise ValueError(f"direction shape {tangent.shape} does not match z {vector.shape}.")
    return float(2.0 * np.dot(matrix @ vector, matrix @ tangent))


def finite_difference_projected_quadratic_derivative(
    z: np.ndarray,
    projection: np.ndarray,
    direction: np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> float:
    """Return the central finite-difference derivative of ``||Pz||^2``."""
    if float(epsilon) <= 0.0:
        raise ValueError(f"epsilon must be positive; got {epsilon!r}.")
    vector = np.asarray(z, dtype=float)
    tangent = np.asarray(direction, dtype=float)
    return float(
        (
            projected_quadratic_statistic(vector + float(epsilon) * tangent, projection)
            - projected_quadratic_statistic(vector - float(epsilon) * tangent, projection)
        )
        / (2.0 * float(epsilon))
    )


def _rademacher_directions(
    *,
    rng: np.random.Generator,
    n_directions: int,
    dimension: int,
) -> np.ndarray:
    directions = rng.choice((-1.0, 1.0), size=(int(n_directions), int(dimension)))
    norms = np.linalg.norm(directions, axis=1)
    directions = directions / norms[:, None]
    return directions


def _fixed_projection_delta_norm(
    z: np.ndarray,
    projection: np.ndarray,
    *,
    rng: np.random.Generator,
    n_directions: int,
    epsilon_scale: float,
) -> float:
    if projection.shape[0] == 0:
        return 0.0
    directions = _rademacher_directions(
        rng=rng,
        n_directions=n_directions,
        dimension=z.shape[0],
    )
    epsilon = max(float(epsilon_scale), 1e-8)
    derivatives = [
        finite_difference_projected_quadratic_derivative(
            z,
            projection,
            direction,
            epsilon=epsilon,
        )
        for direction in directions
    ]
    statistic = projected_quadratic_statistic(z, projection)
    return float(np.sqrt(np.mean(np.square(derivatives))) / max(abs(statistic), 1.0))


def _eigengap_at_k(eigenvalues: np.ndarray, k: int) -> float:
    values = np.asarray(eigenvalues, dtype=float)
    if int(k) <= 0 or values.size < int(k):
        return float("nan")
    if values.size == int(k):
        return math.inf
    return float(values[int(k) - 1] - values[int(k)])


def _projection_instability_score(eigenvalues: np.ndarray, k: int) -> float:
    values = np.asarray(eigenvalues, dtype=float)
    if int(k) <= 0 or values.size < int(k):
        return float("nan")
    gap = _eigengap_at_k(values, int(k))
    if math.isinf(gap):
        return 0.0
    leading = max(float(values[int(k) - 1]), 0.0)
    return float(leading / max(abs(gap), 1e-12))


def _recomputed_projection_delta_norm(
    z: np.ndarray,
    projection: np.ndarray,
    eigenvalues: np.ndarray,
    *,
    rng: np.random.Generator,
    n_directions: int,
    epsilon_scale: float,
) -> float:
    k = int(projection.shape[0])
    if k == 0:
        return 0.0
    full_projection = np.asarray(projection, dtype=float)
    values = np.asarray(eigenvalues, dtype=float)
    if values.shape[0] < full_projection.shape[0]:
        return float("nan")
    covariance = (full_projection.T * values[: full_projection.shape[0]]) @ full_projection
    base_statistic = projected_quadratic_statistic(z, full_projection[:k])
    scale = max(float(np.linalg.norm(covariance, ord=2)), 1.0)
    deltas: list[float] = []
    for _ in range(int(n_directions)):
        noise = rng.choice((-1.0, 1.0), size=covariance.shape)
        noise = 0.5 * (noise + noise.T)
        noise = noise / max(float(np.linalg.norm(noise, ord=2)), 1.0)
        perturbed = covariance + float(epsilon_scale) * scale * noise
        eigvals, eigvecs = np.linalg.eigh(0.5 * (perturbed + perturbed.T))
        order = np.argsort(eigvals)[::-1]
        recomputed_projection = eigvecs[:, order[:k]].T
        new_statistic = projected_quadratic_statistic(z, recomputed_projection)
        deltas.append(abs(new_statistic - base_statistic) / max(abs(base_statistic), 1.0))
    return float(np.sqrt(np.mean(np.square(deltas)))) if deltas else float("nan")


def _projection_status(
    *,
    eigengap: float,
    instability_score: float,
    recomputed_delta_norm: float,
) -> str:
    if math.isfinite(eigengap) and eigengap <= 1e-8:
        return "projection_unstable_small_gap"
    if math.isfinite(instability_score) and instability_score >= 1e6:
        return "projection_unstable_small_gap"
    if math.isfinite(recomputed_delta_norm) and recomputed_delta_norm >= 0.10:
        return "projection_recomputed_tail_shift"
    return "projection_stable"


def _linkage_margin_gap(data: pd.DataFrame) -> float:
    if data.shape[0] < 3:
        return float("nan")
    distances = pdist(data.to_numpy(dtype=float), metric="hamming")
    if distances.size == 0:
        return float("nan")
    linkage_matrix = linkage(distances, method="average")
    heights = np.sort(linkage_matrix[:, 2].astype(float))
    if heights.size < 2:
        return math.inf
    return float(np.min(np.abs(np.diff(heights))))


def _tree_signature(tree: PosetTree) -> frozenset[frozenset[str]]:
    descendants = tree.compute_descendant_sets(use_labels=True)
    total_leaves = frozenset(tree.get_leaves(return_labels=True))
    signature = []
    for node, leaves in descendants.items():
        leaf_set = frozenset(str(leaf) for leaf in leaves)
        if tree.out_degree(node) > 0 and 1 < len(leaf_set) < len(total_leaves):
            signature.append(leaf_set)
    return frozenset(signature)


def _perturb_hamming_data(
    data: pd.DataFrame,
    feature_space: FeatureSpace,
    rng: np.random.Generator,
) -> pd.DataFrame:
    perturbed = data.copy()
    row_index = int(rng.integers(0, perturbed.shape[0]))
    block = feature_space.blocks[int(rng.integers(0, len(feature_space.blocks)))]
    if block.family == "bernoulli":
        column = perturbed.columns[block.column_indices[0]]
        perturbed.iat[row_index, block.column_indices[0]] = 1 - int(
            round(float(perturbed.at[perturbed.index[row_index], column]))
        )
    elif block.family == "categorical":
        current = perturbed.iloc[row_index, list(block.column_indices)].to_numpy(dtype=float)
        active = int(np.argmax(current))
        replacement = (active + 1 + int(rng.integers(0, len(block.column_indices) - 1))) % len(
            block.column_indices
        )
        for offset, column_index in enumerate(block.column_indices):
            perturbed.iat[row_index, column_index] = 1 if offset == replacement else 0
    else:
        raise ValueError(f"Unsupported Hamming perturbation block family {block.family!r}.")
    return perturbed


def _topology_stable_fraction(
    *,
    tree: PosetTree,
    data: pd.DataFrame,
    feature_space: FeatureSpace,
    mode: str,
    rng: np.random.Generator,
    n_directions: int,
) -> tuple[float, str]:
    if mode == "fixed_tree":
        return 1.0, "fixed_tree_no_selection_derivative"
    original = _tree_signature(tree)
    matches = 0
    for _ in range(int(n_directions)):
        perturbed_data = _perturb_hamming_data(data, feature_space, rng)
        perturbed_tree = _build_tree_from_data(perturbed_data)
        matches += int(_tree_signature(perturbed_tree) == original)
    return float(matches / max(int(n_directions), 1)), "nonsmooth_hamming_selection"


def _status_from_components(
    *,
    fisher_boundary_status: str,
    fixed_projection_delta_norm: float,
    projection_status: str,
    mode: str,
    selection_derivative_status: str,
    topology_stable_fraction: float,
) -> str:
    if fisher_boundary_status != "interior":
        return "wald_metric_boundary_unstable"
    if math.isfinite(fixed_projection_delta_norm) and fixed_projection_delta_norm >= 10.0:
        return "whitening_unstable"
    if projection_status != "projection_stable":
        return "projection_unstable"
    if mode != "fixed_tree" and topology_stable_fraction < 0.8:
        return "selection_coupled"
    if selection_derivative_status == "nonsmooth_hamming_selection":
        return "nonsmooth_selection_geometry"
    return "fixed_subspace_candidate"


def _sibling_validity_rows_for_replicate(
    *,
    case_id: str,
    source_family: str,
    feature_representation: str,
    n_samples: int,
    n_features: int,
    n_categories: int | None,
    replicate: int,
    data_seed: int,
    tree_seed: int,
    mode: str,
    edge_alpha: float,
    sibling_alpha: float,
    finite_diff_directions: int,
    epsilon_scale: float,
    run_id: str,
) -> list[dict[str, object]]:
    tree, data, feature_space, selected_tree, fixed_tree = _prepare_tree_for_mode(
        mode=mode,
        case_id=case_id,
        source_family=source_family,
        feature_representation=feature_representation,
        n_samples=n_samples,
        n_features=n_features,
        n_categories=n_categories,
        tree_seed=tree_seed,
        data_seed=data_seed,
    )
    edge_annotations, spectral_context = _annotate_edges(
        tree,
        data,
        edge_alpha=edge_alpha,
        feature_space=feature_space,
    )
    projection_dimensions, parent_projections, parent_eigenvalues = _projection_inputs(
        tree,
        spectral_context,
    )
    raw_records, _non_binary_nodes = collect_sibling_pair_records(
        tree,
        edge_annotations,
        sibling_projection_dimensions_from_edge_comparisons=projection_dimensions,
        parent_principal_component_projections=parent_projections,
        parent_principal_component_eigenvalues=parent_eigenvalues,
        feature_space=feature_space,
    )
    topology_rng = np.random.default_rng(int(data_seed) + 1_000_003)
    topology_stability, selection_derivative_status = _topology_stable_fraction(
        tree=tree,
        data=data,
        feature_space=feature_space,
        mode=mode,
        rng=topology_rng,
        n_directions=finite_diff_directions,
    )
    margin_gap = _linkage_margin_gap(data)
    rows: list[dict[str, object]] = []
    for record_index, record in enumerate(raw_records):
        k = int(record.sibling_projection_dimension)
        parent_key = record.parent
        parent_projection = np.asarray(parent_projections[parent_key], dtype=float)
        eigenvalues = np.asarray(parent_eigenvalues[parent_key], dtype=float)
        projection = parent_projection[:k]
        left_distribution = np.asarray(tree.nodes[record.left]["distribution"], dtype=float)
        right_distribution = np.asarray(tree.nodes[record.right]["distribution"], dtype=float)
        left_n = float(tree.nodes[record.left]["leaf_count"])
        right_n = float(tree.nodes[record.right]["leaf_count"])
        contrast = build_contrast_covariance(
            left_distribution,
            right_distribution,
            left_n,
            right_n,
            comparison="sibling",
            feature_space=feature_space,
        )
        z = contrast.whitened_vector()
        rng = np.random.default_rng(int(data_seed) + 17_171 + record_index)
        fixed_delta = _fixed_projection_delta_norm(
            z,
            projection,
            rng=rng,
            n_directions=finite_diff_directions,
            epsilon_scale=epsilon_scale,
        )
        recomputed_delta = _recomputed_projection_delta_norm(
            z,
            parent_projection,
            eigenvalues,
            rng=rng,
            n_directions=finite_diff_directions,
            epsilon_scale=epsilon_scale,
        )
        eigengap = _eigengap_at_k(eigenvalues, k)
        instability = _projection_instability_score(eigenvalues, k)
        projection_status = _projection_status(
            eigengap=eigengap,
            instability_score=instability,
            recomputed_delta_norm=recomputed_delta,
        )
        pooled_distribution = (left_n * left_distribution + right_n * right_distribution) / (
            left_n + right_n
        )
        fisher = compute_fisher_geometry_summary(pooled_distribution, feature_space)
        validity_status = _status_from_components(
            fisher_boundary_status=str(fisher["fisher_boundary_status"]),
            fixed_projection_delta_norm=fixed_delta,
            projection_status=projection_status,
            mode=mode,
            selection_derivative_status=selection_derivative_status,
            topology_stable_fraction=topology_stability,
        )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "run_id": run_id,
                "case_id": case_id,
                "mode": mode,
                "replicate": int(replicate),
                "parent_id": str(record.parent),
                "left_child_id": str(record.left),
                "right_child_id": str(record.right),
                "source_family": source_family,
                "feature_representation": feature_representation,
                "edge_alpha": float(edge_alpha),
                "sibling_alpha": float(sibling_alpha),
                "selected_tree": bool(selected_tree),
                "fixed_tree": bool(fixed_tree),
                "data_seed": int(data_seed),
                "tree_seed": int(tree_seed),
                "sibling_raw_stat": float(record.stat),
                "sibling_df": float(record.degrees_of_freedom),
                "sibling_raw_p": float(record.p_value),
                "sibling_projection_dimension": float(record.sibling_projection_dimension),
                "fixed_projection_delta_norm": fixed_delta,
                "recomputed_projection_delta_norm": recomputed_delta,
                "projection_instability_score": instability,
                "eigengap_at_sibling_k": eigengap,
                "projection_status": projection_status,
                "selection_margin_gap": margin_gap,
                "topology_stable_fraction": topology_stability,
                "selection_derivative_status": selection_derivative_status,
                "statistic_validity_status": validity_status,
                "df_bin": _df_bin(float(record.degrees_of_freedom)),
                **fisher,
            }
        )
    return rows


def summarize_differential_statistic_validity(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize differential validity rows by case, mode, family, and df bin."""
    if rows.empty:
        return pd.DataFrame()
    summaries: list[dict[str, object]] = []
    group_columns = ("case_id", "mode", "source_family", "df_bin")
    for key, group in rows.groupby(list(group_columns), sort=True):
        status_counts = group["statistic_validity_status"].value_counts().to_dict()
        summary_status = "fixed_subspace_candidate"
        for status in FAIL_CLOSED_STATUS_PRECEDENCE:
            if int(status_counts.get(status, 0)) > 0:
                summary_status = status
                break
        row = dict(zip(group_columns, key))
        row.update(
            {
                "n_rows": int(group.shape[0]),
                "fixed_projection_delta_norm_q50": float(
                    group["fixed_projection_delta_norm"].quantile(0.50)
                ),
                "recomputed_projection_delta_norm_q50": float(
                    group["recomputed_projection_delta_norm"].quantile(0.50)
                ),
                "projection_instability_score_q50": float(
                    group["projection_instability_score"].quantile(0.50)
                ),
                "eigengap_at_sibling_k_q50": float(
                    group["eigengap_at_sibling_k"].replace(math.inf, np.nan).quantile(0.50)
                ),
                "fisher_variance_floor_min": float(group["fisher_variance_floor"].min()),
                "fisher_condition_number_q50": float(
                    group["fisher_condition_number"].replace(math.inf, np.nan).quantile(0.50)
                ),
                "selection_margin_gap_min": float(group["selection_margin_gap"].min()),
                "topology_stable_fraction_q50": float(
                    group["topology_stable_fraction"].quantile(0.50)
                ),
                "fixed_subspace_candidate_fraction": float(
                    group["statistic_validity_status"].eq("fixed_subspace_candidate").mean()
                ),
                "statistic_validity_status": summary_status,
                "status_counts": json.dumps(status_counts, sort_keys=True),
                "study_role": STUDY_ROLE,
            }
        )
        summaries.append(row)
    return pd.DataFrame.from_records(summaries)


def build_differential_validity_production_components(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build production-admissibility components from differential summaries."""
    records: list[dict[str, object]] = []
    for _, row in summary.iterrows():
        context = (
            f"case_id={row['case_id']}|mode={row['mode']}|"
            f"source_family={row['source_family']}|df_bin={row['df_bin']}"
        )
        records.append(
            {
                "contract_id": "differential_statistic_validity",
                "component_id": f"differential_validity:{context}",
                "component_type": "differential_statistic_validity_panel",
                "component_status": str(row["statistic_validity_status"]),
                "required_for_production": True,
                "notes": "Differential diagnostic status for projected-Wald statistic validity.",
            }
        )
    return pd.DataFrame.from_records(records)


def run_differential_statistic_validity_panel(
    config: DifferentialStatisticValidityConfig,
) -> dict[str, Path]:
    """Run differential statistic-validity diagnostics and write outputs."""
    if int(config.replicates) <= 0:
        raise ValueError("replicates must be positive.")
    if int(config.finite_diff_directions) <= 0:
        raise ValueError("finite_diff_directions must be positive.")
    if float(config.epsilon_scale) <= 0.0:
        raise ValueError("epsilon_scale must be positive.")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = (
        "differential_statistic_validity__"
        f"{config.suite}__{'-'.join(config.modes)}__"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    rows: list[dict[str, object]] = []
    for case in _select_cases(suite=config.suite, case_names=config.case_names):
        (
            case_id,
            source_family,
            feature_representation,
            n_samples,
            n_features,
            n_categories,
        ) = _case_contract(case)
        for replicate in range(int(config.replicates)):
            for edge_alpha in config.edge_alphas:
                for mode in config.modes:
                    data_seed = (
                        int(config.base_seed)
                        + replicate * 1009
                        + int(float(edge_alpha) * 1_000_000)
                    )
                    tree_seed = int(config.base_seed) + replicate * 917 + 17
                    rows.extend(
                        _sibling_validity_rows_for_replicate(
                            case_id=case_id,
                            source_family=source_family,
                            feature_representation=feature_representation,
                            n_samples=n_samples,
                            n_features=n_features,
                            n_categories=n_categories,
                            replicate=replicate,
                            data_seed=data_seed,
                            tree_seed=tree_seed,
                            mode=mode,
                            edge_alpha=float(edge_alpha),
                            sibling_alpha=float(config.sibling_alpha),
                            finite_diff_directions=int(config.finite_diff_directions),
                            epsilon_scale=float(config.epsilon_scale),
                            run_id=run_id,
                        )
                    )
    row_table = pd.DataFrame.from_records(rows)
    summary = summarize_differential_statistic_validity(row_table)
    components = build_differential_validity_production_components(summary)
    production_rows = evaluate_production_admissibility_components(components)
    production_summary = summarize_production_admissibility_contracts(production_rows)

    row_table.to_csv(config.rows_path, index=False)
    summary.to_csv(config.summary_path, index=False)
    production_rows.to_csv(config.production_components_path, index=False)
    production_summary.to_csv(config.production_summary_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "suite": config.suite,
        "case_names": list(config.case_names),
        "modes": list(config.modes),
        "edge_alphas": list(config.edge_alphas),
        "sibling_alpha": float(config.sibling_alpha),
        "replicates": int(config.replicates),
        "base_seed": int(config.base_seed),
        "finite_diff_directions": int(config.finite_diff_directions),
        "epsilon_scale": float(config.epsilon_scale),
        "n_rows": int(row_table.shape[0]),
        "outputs": {
            "rows": str(config.rows_path),
            "summary": str(config.summary_path),
            "production_components": str(config.production_components_path),
            "production_summary": str(config.production_summary_path),
        },
        "interpretation": (
            "Diagnostic differential statistic-validity panel. It does not "
            "promote a production calibration rule."
        ),
    }
    config.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "rows": config.rows_path,
        "summary": config.summary_path,
        "production_components": config.production_components_path,
        "production_summary": config.production_summary_path,
        "manifest": config.manifest_path,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", default="binary")
    parser.add_argument("--case-names")
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument(
        "--edge-alphas",
        default=",".join(str(value) for value in DEFAULT_EDGE_ALPHA_GRID),
    )
    parser.add_argument("--sibling-alpha", type=float, default=0.01)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=20260613)
    parser.add_argument("--finite-diff-directions", type=int, default=16)
    parser.add_argument("--epsilon-scale", type=float, default=1e-4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    outputs = run_differential_statistic_validity_panel(
        DifferentialStatisticValidityConfig(
            output_dir=args.output_dir,
            suite=str(args.suite),
            case_names=parse_names(args.case_names),
            modes=validate_modes(parse_names(args.modes)),
            edge_alphas=parse_alpha_grid(str(args.edge_alphas)),
            sibling_alpha=float(args.sibling_alpha),
            replicates=int(args.replicates),
            base_seed=int(args.base_seed),
            finite_diff_directions=int(args.finite_diff_directions),
            epsilon_scale=float(args.epsilon_scale),
        )
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "DifferentialStatisticValidityConfig",
    "build_differential_validity_production_components",
    "compute_fisher_geometry_summary",
    "finite_difference_projected_quadratic_derivative",
    "projected_quadratic_directional_derivative",
    "projected_quadratic_statistic",
    "run_differential_statistic_validity_panel",
    "summarize_differential_statistic_validity",
]
