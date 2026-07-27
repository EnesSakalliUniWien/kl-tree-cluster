"""Regularized projected-Wald statistic diagnostics.

This module compares plug-in and smoothed sibling Wald statistics under
selected-edge null runs. It is diagnostic-only and does not install a
production statistic.
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
from scipy.stats import chi2
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.pair_testing.collection.pair_observations import (
    identify_binary_sibling_children,
)
from tree_break_selection.tree.feature_space import FeatureSpace

from benchmarks.diagnostics.calibration.production_admissibility_contract import (
    evaluate_production_admissibility_components,
    summarize_production_admissibility_contracts,
)
from benchmarks.validation.selected_edge_type1_geometry import (
    _annotate_edges,
    _case_contract,
    _prepare_tree_for_mode,
    _projection_inputs,
    _select_cases,
    parse_alpha_grid,
    parse_names,
    validate_modes,
)

STUDY_ROLE = "diagnostic_regularized_wald_statistic_not_calibration"
SCHEMA_VERSION = "regularized_wald_statistic_panel/v1"
GENERATED_BY = "benchmarks.diagnostics.calibration.regularized_wald_statistic_panel"
DEFAULT_MODES = ("fixed_tree", "selected_tree")
DEFAULT_EDGE_ALPHA_GRID = (0.001,)
SMOOTHING_RULES = ("plugin", "jeffreys", "dirichlet_1", "root_shrink_0.1")


@dataclass(frozen=True)
class RegularizedWaldStatisticConfig:
    """Runtime contract for regularized Wald statistic diagnostics."""

    output_dir: Path
    suite: str
    case_names: tuple[str, ...]
    modes: tuple[str, ...]
    edge_alphas: tuple[float, ...]
    sibling_alpha: float
    replicates: int
    base_seed: int

    @property
    def rows_path(self) -> Path:
        return self.output_dir / "regularized_wald_statistic_rows.csv"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "regularized_wald_statistic_summary.csv"

    @property
    def production_components_path(self) -> Path:
        return self.output_dir / "production_admissibility_components.csv"

    @property
    def production_summary_path(self) -> Path:
        return self.output_dir / "production_admissibility_summary.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


def regularize_feature_distribution(
    distribution: np.ndarray,
    *,
    sample_size: float,
    feature_space: FeatureSpace,
    smoothing_rule: str,
    root_distribution: np.ndarray | None = None,
) -> np.ndarray:
    """Return a smoothed copy of a feature distribution."""
    values = np.asarray(distribution, dtype=float).copy()
    n = float(sample_size)
    if n <= 0.0 or not np.isfinite(n):
        raise ValueError(f"sample_size must be finite and positive; got {sample_size!r}.")
    if values.ndim != 1 or values.shape[0] != feature_space.raw_dimension:
        raise ValueError(
            "distribution must be one-dimensional with feature-space raw dimension; "
            f"got shape={values.shape}, raw_dimension={feature_space.raw_dimension}."
        )
    if smoothing_rule == "plugin":
        return values
    if smoothing_rule == "root_shrink_0.1":
        if root_distribution is None:
            raise ValueError("root_shrink_0.1 requires root_distribution.")
        root = np.asarray(root_distribution, dtype=float)
        if root.shape != values.shape:
            raise ValueError(
                f"root_distribution shape {root.shape} must match distribution {values.shape}."
            )
        return 0.9 * values + 0.1 * root

    result = values.copy()
    for block in feature_space.blocks:
        indices = list(block.column_indices)
        block_values = values[indices]
        if block.family == "bernoulli":
            alpha = _smoothing_alpha(smoothing_rule)
            result[indices] = (n * block_values + alpha) / (n + 2.0 * alpha)
        elif block.family == "categorical":
            alpha = _smoothing_alpha(smoothing_rule)
            k = len(indices)
            result[indices] = (n * block_values + alpha) / (n + k * alpha)
        else:
            raise ValueError(
                "Regularized Wald V1 supports Bernoulli and categorical blocks; "
                f"got {block.family!r}."
            )
    return result


def _smoothing_alpha(smoothing_rule: str) -> float:
    if smoothing_rule == "jeffreys":
        return 0.5
    if smoothing_rule == "dirichlet_1":
        return 1.0
    raise ValueError(f"Unknown smoothing_rule: {smoothing_rule!r}.")


def _boundary_status_from_distribution(
    distribution: np.ndarray,
    feature_space: FeatureSpace,
) -> tuple[float, str]:
    values = np.asarray(distribution, dtype=float)
    floors: list[float] = []
    for block in feature_space.blocks:
        indices = list(block.column_indices)
        block_values = values[indices]
        if block.family == "bernoulli":
            floors.extend(float(value * (1.0 - value)) for value in block_values)
        elif block.family == "categorical":
            positive = block_values[block_values > 0.0]
            floors.append(float(np.min(positive)) if positive.size else 0.0)
        else:
            raise ValueError(f"Unsupported block family: {block.family!r}.")
    floor = float(min(floors)) if floors else math.nan
    if not math.isfinite(floor) or floor <= 1e-8:
        return floor, "boundary_unstable"
    if floor <= 1e-4:
        return floor, "near_boundary"
    return floor, "interior"


def _sibling_wald_z_scores(
    left: np.ndarray,
    right: np.ndarray,
    *,
    left_sample_size: float,
    right_sample_size: float,
    feature_space: FeatureSpace,
    ridge: float = 1e-12,
) -> np.ndarray:
    left_n = float(left_sample_size)
    right_n = float(right_sample_size)
    pooled = (left_n * left + right_n * right) / (left_n + right_n)
    pieces: list[np.ndarray] = []
    variance_scale = (1.0 / left_n) + (1.0 / right_n)
    for block in feature_space.blocks:
        indices = list(block.column_indices)
        left_block = left[indices]
        right_block = right[indices]
        pooled_block = pooled[indices]
        if block.family == "bernoulli":
            variance = pooled_block * (1.0 - pooled_block) * variance_scale + ridge
            pieces.append((left_block - right_block) / np.sqrt(variance))
        elif block.family == "categorical":
            p = pooled_block[:-1]
            covariance = (np.diag(p) - np.outer(p, p)) * variance_scale
            covariance = covariance + ridge * np.eye(covariance.shape[0])
            pieces.append(
                np.linalg.solve(
                    np.linalg.cholesky(covariance),
                    left_block[:-1] - right_block[:-1],
                )
            )
        else:
            raise ValueError(f"Unsupported block family: {block.family!r}.")
    return np.concatenate(pieces)


def compute_regularized_sibling_wald_variant(
    *,
    left_distribution: np.ndarray,
    right_distribution: np.ndarray,
    left_sample_size: float,
    right_sample_size: float,
    parent_projection: np.ndarray,
    parent_eigenvalues: np.ndarray,
    projection_dimension: int,
    feature_space: FeatureSpace,
    smoothing_rule: str,
    root_distribution: np.ndarray,
) -> dict[str, object]:
    """Compute one regularized sibling projected-Wald statistic variant."""
    _ = np.asarray(parent_eigenvalues, dtype=float)
    left = regularize_feature_distribution(
        left_distribution,
        sample_size=left_sample_size,
        feature_space=feature_space,
        smoothing_rule=smoothing_rule,
        root_distribution=root_distribution,
    )
    right = regularize_feature_distribution(
        right_distribution,
        sample_size=right_sample_size,
        feature_space=feature_space,
        smoothing_rule=smoothing_rule,
        root_distribution=root_distribution,
    )
    left_n = float(left_sample_size)
    right_n = float(right_sample_size)
    pooled = (left_n * left + right_n * right) / (left_n + right_n)
    floor, boundary_status = _boundary_status_from_distribution(pooled, feature_space)
    z = _sibling_wald_z_scores(
        left,
        right,
        left_sample_size=left_sample_size,
        right_sample_size=right_sample_size,
        feature_space=feature_space,
    )
    k = int(projection_dimension)
    projection = np.asarray(parent_projection, dtype=float)[:k]
    if k < 0:
        raise ValueError(f"projection_dimension must be non-negative; got {k!r}.")
    if projection.shape[1] != z.shape[0]:
        raise ValueError(
            f"projection width {projection.shape[1]} must match z dimension {z.shape[0]}."
        )
    statistic = float(np.sum((projection @ z) ** 2)) if k > 0 else 0.0
    p_value = float(chi2.sf(statistic, df=float(k))) if k > 0 else 1.0
    return {
        "smoothing_rule": smoothing_rule,
        "test_statistic": statistic,
        "degrees_of_freedom": float(k),
        "p_value": p_value,
        "boundary_variance_floor": floor,
        "boundary_status": boundary_status,
        "z_norm": float(np.linalg.norm(z)),
    }


def _df_bin(value: float) -> str:
    if value <= 1.0:
        return "df_0_1"
    if value <= 2.0:
        return "df_1_2"
    if value <= 4.0:
        return "df_2_4"
    if value <= 8.0:
        return "df_4_8"
    return "df_ge8"


def _regularized_status(group: pd.DataFrame, *, alpha: float, min_rows: int) -> str:
    if group.shape[0] < int(min_rows):
        return "regularized_insufficient_rows"
    if group["boundary_status"].astype(str).ne("interior").any():
        return "regularized_boundary_unstable"
    tail_rate = float(group["tail_reject_at_0.05"].mean())
    if abs(tail_rate - float(alpha)) > 0.05:
        return "regularized_tail_misaligned"
    if group["mode"].astype(str).ne("fixed_tree").any():
        return "regularized_selected_tree_not_yet_calibrated"
    return "regularized_fixed_tree_candidate"


def summarize_regularized_wald_rows(
    rows: pd.DataFrame,
    *,
    alpha: float = 0.05,
    min_rows: int = 30,
) -> pd.DataFrame:
    """Summarize regularized Wald variants by context and df bin."""
    if rows.empty:
        return pd.DataFrame()
    summaries: list[dict[str, object]] = []
    group_columns = ("case_id", "mode", "source_family", "smoothing_rule", "df_bin")
    for key, group in rows.groupby(list(group_columns), sort=True):
        p_values = pd.to_numeric(group["p_value"], errors="coerce")
        row = dict(zip(group_columns, key))
        row.update(
            {
                "n_rows": int(group.shape[0]),
                "boundary_unstable_fraction": float(
                    group["boundary_status"].astype(str).ne("interior").mean()
                ),
                "test_statistic_q50": float(group["test_statistic"].quantile(0.50)),
                "test_statistic_q90": float(group["test_statistic"].quantile(0.90)),
                "p_value_q50": float(p_values.quantile(0.50)),
                "p_value_q05": float(p_values.quantile(0.05)),
                "tail_rate_at_0.05": float(group["tail_reject_at_0.05"].mean()),
                "regularized_wald_status": _regularized_status(
                    group,
                    alpha=alpha,
                    min_rows=min_rows,
                ),
                "study_role": STUDY_ROLE,
            }
        )
        summaries.append(row)
    return pd.DataFrame.from_records(summaries)


def build_regularized_wald_production_components(summary: pd.DataFrame) -> pd.DataFrame:
    """Build production-admissibility components from variant summaries."""
    records: list[dict[str, object]] = []
    for _, row in summary.iterrows():
        context = (
            f"case_id={row['case_id']}|mode={row['mode']}|"
            f"source_family={row['source_family']}|smoothing_rule={row['smoothing_rule']}|"
            f"df_bin={row['df_bin']}"
        )
        records.append(
            {
                "contract_id": "regularized_wald_statistic",
                "component_id": f"regularized_wald:{context}",
                "component_type": "regularized_wald_statistic_panel",
                "component_status": str(row["regularized_wald_status"]),
                "required_for_production": True,
                "notes": "Diagnostic status for regularized projected-Wald statistic variant.",
            }
        )
    return pd.DataFrame.from_records(records)


def _regularized_rows_for_replicate(
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
    run_id: str,
) -> list[dict[str, object]]:
    tree, _data, feature_space, selected_tree, fixed_tree = _prepare_tree_for_mode(
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
    _edge_annotations, spectral_context = _annotate_edges(
        tree,
        _data,
        edge_alpha=edge_alpha,
        feature_space=feature_space,
    )
    projection_dimensions, parent_projections, parent_eigenvalues = _projection_inputs(
        tree,
        spectral_context,
    )
    root_distribution = np.asarray(tree.nodes[tree.root()]["distribution"], dtype=float)
    rows: list[dict[str, object]] = []
    for parent in tree.nodes:
        children = identify_binary_sibling_children(tree, parent)
        if children is None or parent not in projection_dimensions:
            continue
        left, right = children
        k = int(projection_dimensions[parent])
        if k <= 0:
            continue
        for smoothing_rule in SMOOTHING_RULES:
            variant = compute_regularized_sibling_wald_variant(
                left_distribution=np.asarray(tree.nodes[left]["distribution"], dtype=float),
                right_distribution=np.asarray(tree.nodes[right]["distribution"], dtype=float),
                left_sample_size=float(tree.nodes[left]["leaf_count"]),
                right_sample_size=float(tree.nodes[right]["leaf_count"]),
                parent_projection=parent_projections[parent],
                parent_eigenvalues=parent_eigenvalues[parent],
                projection_dimension=k,
                feature_space=feature_space,
                smoothing_rule=smoothing_rule,
                root_distribution=root_distribution,
            )
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_role": STUDY_ROLE,
                    "run_id": run_id,
                    "case_id": case_id,
                    "mode": mode,
                    "replicate": int(replicate),
                    "parent_id": str(parent),
                    "source_family": source_family,
                    "feature_representation": feature_representation,
                    "selected_tree": bool(selected_tree),
                    "fixed_tree": bool(fixed_tree),
                    "edge_alpha": float(edge_alpha),
                    "sibling_alpha": float(sibling_alpha),
                    "data_seed": int(data_seed),
                    "tree_seed": int(tree_seed),
                    "sibling_df": float(variant["degrees_of_freedom"]),
                    "df_bin": _df_bin(float(variant["degrees_of_freedom"])),
                    "tail_reject_at_0.05": bool(float(variant["p_value"]) <= 0.05),
                    **variant,
                }
            )
    return rows


def run_regularized_wald_statistic_panel(
    config: RegularizedWaldStatisticConfig,
) -> dict[str, Path]:
    """Run regularized Wald statistic diagnostics and write outputs."""
    if int(config.replicates) <= 0:
        raise ValueError("replicates must be positive.")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = (
        "regularized_wald_statistic__"
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
                        _regularized_rows_for_replicate(
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
                            run_id=run_id,
                        )
                    )
    row_table = pd.DataFrame.from_records(rows)
    summary = summarize_regularized_wald_rows(row_table)
    components = build_regularized_wald_production_components(summary)
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
        "n_rows": int(row_table.shape[0]),
        "outputs": {
            "rows": str(config.rows_path),
            "summary": str(config.summary_path),
            "production_components": str(config.production_components_path),
            "production_summary": str(config.production_summary_path),
        },
        "interpretation": (
            "Diagnostic regularized projected-Wald statistic panel. It does not "
            "change production statistic defaults."
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    outputs = run_regularized_wald_statistic_panel(
        RegularizedWaldStatisticConfig(
            output_dir=args.output_dir,
            suite=str(args.suite),
            case_names=parse_names(args.case_names),
            modes=validate_modes(parse_names(args.modes)),
            edge_alphas=parse_alpha_grid(str(args.edge_alphas)),
            sibling_alpha=float(args.sibling_alpha),
            replicates=int(args.replicates),
            base_seed=int(args.base_seed),
        )
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "RegularizedWaldStatisticConfig",
    "build_regularized_wald_production_components",
    "compute_regularized_sibling_wald_variant",
    "regularize_feature_distribution",
    "run_regularized_wald_statistic_panel",
    "summarize_regularized_wald_rows",
]
