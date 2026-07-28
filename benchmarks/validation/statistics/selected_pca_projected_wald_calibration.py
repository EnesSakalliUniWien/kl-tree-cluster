#!/usr/bin/env python3
"""Calibration scaffold for the selected-PCA projected-Wald reference.

This module validates one narrow mathematical question without changing the
production method:

* does the chi-square projected-Wald reference remain calibrated when the PCA
  projection rows and MP projection dimension are selected from the same local
  Gaussian null data used by the sibling contrast?

The simulation is a fixed-membership local sibling-null check. It does not
validate hierarchy construction, tree-selected sibling pairs, sibling FDR,
traversal, empirical-null inflation, or real-data misspecification.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from tree_break_selection.hierarchy_analysis.decomposition.backends.eigen.decomposition import (
    eigendecompose_covariance,
)
from tree_break_selection.hierarchy_analysis.decomposition.backends.eigen.projection import (
    build_pca_projection,
)
from tree_break_selection.hierarchy_analysis.statistics.contrast_covariance import (
    build_null_whitened_tangent_matrix,
    compute_whitened_wald_contrast,
)
from tree_break_selection.hierarchy_analysis.statistics.projection.projected_wald.projected_wald_kernel import (
    run_projected_wald_kernel,
)
from tree_break_selection.hierarchy_analysis.statistics.projection.projection_dimension_estimation.projection_dimension_estimators import (
    estimate_marchenko_pastur_dimension,
)
from tree_break_selection.tree.feature_space import (
    FeatureSpace,
    continuous_feature_space_from_columns,
)

from benchmarks.validation.contracts.report_contract import (
    completed_target_entry as _completed_target_entry,
)
from benchmarks.validation.contracts.report_contract import (
    read_git_commit,
    read_git_worktree_status,
    utc_now,
)
from benchmarks.validation.contracts.report_contract import (
    validate_common_run_inputs as _validate_run_inputs,
)
from benchmarks.validation.contracts.report_contract import (
    validate_complete_report_context as _validate_complete_report_context,
)
from benchmarks.validation.statistics.calibration_support import (
    continuous_covariance,
    summarize_p_value_calibration,
)

SCHEMA_VERSION = "selected_pca_projected_wald_calibration/v1"
GENERATED_BY = "benchmarks.validation.statistics.selected_pca_projected_wald_calibration"
VALIDATION_DESIGN = "fixed_membership_gaussian_sibling_null_with_data_selected_pca"
TARGET_ID = "selected_pca_projected_wald_reference"
PRIMARY_ENDPOINT = "null_rejection_rate_at_alpha"
RANDOM_SEED_POLICY = (
    "A single numpy Generator is initialized from base_seed; all simulation "
    "settings draw independent consecutive replicates from that generator."
)
TARGET_STATUSES = ("missing", "partial", "complete")
COMMON_REQUIRED_OUTPUT_FIELDS = (
    "validation_design",
    "code_commit",
    "git_worktree_status",
    "run_command",
    "random_seed_policy",
    "base_seed",
    "n_replicates",
    "alpha",
    "simulation_grid",
    "primary_endpoint",
    "effect_estimate",
    "confidence_interval",
    "p_value_uniformity_summary",
    "projection_dimension_summary",
    "raw_mp_signal_count_summary",
    "limitations",
    "dimension_grid",
    "sample_size_grid",
    "covariance_profile_grid",
    "minimum_projection_dimension_grid",
    "include_child_mean_rows_grid",
    "ridge",
)


@dataclass(frozen=True)
class ValidationTarget:
    """One selected-PCA validation target."""

    target_id: str
    display_name: str
    validation_question: str
    required_output_fields: tuple[str, ...]
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class SelectedPcaCalibrationSetting:
    """One fixed-membership Gaussian sibling-null setting."""

    dimension: int
    n_left: int
    n_right: int
    covariance_profile: Literal["identity", "ar1_0.6", "ill_conditioned"]
    minimum_projection_dimension: int = 2
    include_child_mean_rows: bool = False

    @property
    def setting_id(self) -> str:
        internal = "childmeans" if self.include_child_mean_rows else "leaves"
        return (
            f"selected_pca_{self.dimension}dim_{self.n_left}x{self.n_right}_"
            f"{self.covariance_profile}_kmin{self.minimum_projection_dimension}_{internal}"
        )


TARGET = ValidationTarget(
    target_id=TARGET_ID,
    display_name="Selected-PCA projected-Wald reference",
    validation_question=(
        "Does the fixed-subspace chi-square projected-Wald reference remain "
        "calibrated when PCA rows and MP dimension are selected from the same "
        "local Gaussian null data used by the sibling contrast?"
    ),
    required_output_fields=COMMON_REQUIRED_OUTPUT_FIELDS,
    limitations=(
        "Fixed-membership local Gaussian sibling null only.",
        "Uses production null-whitened tangent coordinates, MP dimension "
        "selection, and projected-Wald kernel.",
        "Does not validate hierarchy construction, tree-selected sibling pairs, "
        "sibling FDR, traversal, empirical-null inflation, categorical blocks, "
        "or continuous real-data model misspecification.",
    ),
)

DEFAULT_SETTINGS = (
    SelectedPcaCalibrationSetting(
        dimension=8,
        n_left=80,
        n_right=80,
        covariance_profile="identity",
        minimum_projection_dimension=2,
        include_child_mean_rows=False,
    ),
    SelectedPcaCalibrationSetting(
        dimension=8,
        n_left=80,
        n_right=80,
        covariance_profile="identity",
        minimum_projection_dimension=2,
        include_child_mean_rows=True,
    ),
    SelectedPcaCalibrationSetting(
        dimension=16,
        n_left=60,
        n_right=60,
        covariance_profile="ar1_0.6",
        minimum_projection_dimension=2,
        include_child_mean_rows=False,
    ),
    SelectedPcaCalibrationSetting(
        dimension=16,
        n_left=60,
        n_right=60,
        covariance_profile="ar1_0.6",
        minimum_projection_dimension=2,
        include_child_mean_rows=True,
    ),
    SelectedPcaCalibrationSetting(
        dimension=32,
        n_left=50,
        n_right=50,
        covariance_profile="ill_conditioned",
        minimum_projection_dimension=2,
        include_child_mean_rows=False,
    ),
    SelectedPcaCalibrationSetting(
        dimension=32,
        n_left=50,
        n_right=50,
        covariance_profile="ill_conditioned",
        minimum_projection_dimension=2,
        include_child_mean_rows=True,
    ),
)


def create_missing_manifest(*, created_utc: str | None = None) -> dict[str, Any]:
    """Return a missing-evidence manifest for selected-PCA calibration."""

    return {
        "manifest_schema_version": SCHEMA_VERSION,
        "created_utc": created_utc or utc_now(),
        "generated_by": GENERATED_BY,
        "validation_design": VALIDATION_DESIGN,
        "targets": [
            {
                "target_id": TARGET.target_id,
                "display_name": TARGET.display_name,
                "validation_question": TARGET.validation_question,
                "required_output_fields": list(TARGET.required_output_fields),
                "evidence_status": "missing",
                "evidence": {
                    "status": "missing",
                    "source_path": None,
                    "metrics": {},
                    "missing_required_fields": list(TARGET.required_output_fields),
                    "notes": "No selected-PCA projected-Wald evidence has been attached.",
                },
            }
        ],
    }


def run_selected_pca_projected_wald_calibration(
    *,
    n_replicates: int,
    alpha: float,
    base_seed: int,
    code_commit: str,
    git_worktree_status: Sequence[str],
    run_command: str,
    settings: Sequence[SelectedPcaCalibrationSetting] = DEFAULT_SETTINGS,
    ridge: float = 1e-12,
    created_utc: str | None = None,
) -> dict[str, Any]:
    """Run selected-PCA local sibling-null calibration simulations."""

    _validate_run_inputs(
        n_replicates=n_replicates,
        alpha=alpha,
        base_seed=base_seed,
        code_commit=code_commit,
        git_worktree_status=git_worktree_status,
        run_command=run_command,
        ridge=ridge,
    )
    if not settings:
        raise ValueError("settings must contain at least one simulation setting.")

    rng = np.random.default_rng(base_seed)
    results = [
        _simulate_setting(
            rng,
            setting,
            n_replicates=n_replicates,
            alpha=alpha,
            ridge=ridge,
        )
        for setting in settings
    ]

    report = {
        "manifest_schema_version": SCHEMA_VERSION,
        "created_utc": created_utc or utc_now(),
        "generated_by": GENERATED_BY,
        "validation_design": VALIDATION_DESIGN,
        "code_commit": code_commit,
        "git_worktree_status": list(git_worktree_status),
        "run_command": run_command,
        "random_seed_policy": RANDOM_SEED_POLICY,
        "base_seed": base_seed,
        "n_replicates": n_replicates,
        "alpha": alpha,
        "primary_endpoint": PRIMARY_ENDPOINT,
        "targets": [
            _completed_target_entry(
                TARGET,
                results,
                extra_grids={
                    "dimension_grid": sorted({setting.dimension for setting in settings}),
                    "sample_size_grid": sorted(
                        {(setting.n_left, setting.n_right) for setting in settings}
                    ),
                    "covariance_profile_grid": sorted(
                        {setting.covariance_profile for setting in settings}
                    ),
                    "minimum_projection_dimension_grid": sorted(
                        {setting.minimum_projection_dimension for setting in settings}
                    ),
                    "include_child_mean_rows_grid": sorted(
                        {setting.include_child_mean_rows for setting in settings}
                    ),
                    "ridge": ridge,
                },
                primary_endpoint=PRIMARY_ENDPOINT,
            )
        ],
    }

    errors = validate_selected_pca_projected_wald_report(report)
    if errors:
        raise ValueError("invalid selected-PCA report:\n" + "\n".join(errors))
    return report


def validate_selected_pca_projected_wald_report(report: Mapping[str, Any]) -> list[str]:
    """Return structural validation errors for a report or missing manifest."""

    errors: list[str] = []
    if not isinstance(report, Mapping):
        return ["report must be an object"]
    if report.get("manifest_schema_version") != SCHEMA_VERSION:
        errors.append(
            "manifest_schema_version must be "
            f"{SCHEMA_VERSION!r}, got {report.get('manifest_schema_version')!r}"
        )
    if report.get("generated_by") != GENERATED_BY:
        errors.append(f"generated_by must be {GENERATED_BY!r}")
    if report.get("validation_design") != VALIDATION_DESIGN:
        errors.append(f"validation_design must be {VALIDATION_DESIGN!r}")
    targets = report.get("targets")
    if not isinstance(targets, list):
        errors.append("targets must be a list")
        return errors
    if len(targets) != 1:
        errors.append("targets must contain exactly one selected-PCA target")
        return errors
    target_entry = targets[0]
    if not isinstance(target_entry, Mapping):
        errors.append("targets[0] must be an object")
        return errors
    if target_entry.get("target_id") != TARGET_ID:
        errors.append(f"targets[0].target_id must be {TARGET_ID!r}")
    if target_entry.get("required_output_fields") != list(TARGET.required_output_fields):
        errors.append("targets[0].required_output_fields does not match target")
    status = target_entry.get("evidence_status")
    if status not in TARGET_STATUSES:
        errors.append(f"targets[0].evidence_status has unsupported value: {status!r}")
    evidence = target_entry.get("evidence")
    if not isinstance(evidence, Mapping):
        errors.append("targets[0].evidence must be an object")
        return errors
    if evidence.get("status") != status:
        errors.append("targets[0].evidence.status must match evidence_status")
    if status == "missing":
        if evidence.get("source_path") is not None:
            errors.append("targets[0].evidence.source_path must be null when missing")
        if evidence.get("metrics") != {}:
            errors.append("targets[0].evidence.metrics must be empty when missing")
        if evidence.get("missing_required_fields") != list(TARGET.required_output_fields):
            errors.append("targets[0].evidence.missing_required_fields must list all fields")
        return errors

    _validate_complete_report_context(
        report,
        errors,
        primary_endpoint=PRIMARY_ENDPOINT,
    )
    _validate_complete_evidence(evidence, errors)
    return errors


def write_selected_pca_projected_wald_report(
    report: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """Validate and write a selected-PCA report JSON file."""

    errors = validate_selected_pca_projected_wald_report(report)
    if errors:
        raise ValueError("invalid selected-PCA report:\n" + "\n".join(errors))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def write_selected_pca_projected_wald_summary_csv(
    report: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """Write one summary row per simulated selected-PCA setting."""

    errors = validate_selected_pca_projected_wald_report(report)
    if errors:
        raise ValueError("invalid selected-PCA report:\n" + "\n".join(errors))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target_id",
        "setting_id",
        "n_replicates",
        "alpha",
        "rejection_count",
        "rejection_rate",
        "ci_low",
        "ci_high",
        "effect_estimate",
        "ks_statistic",
        "ks_p_value",
        "mean_projection_dimension",
        "mean_raw_mp_signal_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_summary_rows(report))
    return path


def _simulate_setting(
    rng: np.random.Generator,
    setting: SelectedPcaCalibrationSetting,
    *,
    n_replicates: int,
    alpha: float,
    ridge: float,
) -> dict[str, Any]:
    _validate_setting(setting)
    columns = tuple(f"X{index}" for index in range(setting.dimension))
    feature_space = continuous_feature_space_from_columns(columns)
    covariance = continuous_covariance(
        dimension=setting.dimension,
        profile=setting.covariance_profile,
    )
    mean = np.zeros(setting.dimension, dtype=np.float64)

    p_values = np.empty(n_replicates, dtype=np.float64)
    projection_dimensions = np.empty(n_replicates, dtype=np.float64)
    raw_mp_signal_counts = np.empty(n_replicates, dtype=np.float64)

    for replicate_index in range(n_replicates):
        left_samples = rng.multivariate_normal(
            mean,
            covariance,
            size=setting.n_left,
            check_valid="raise",
        )
        right_samples = rng.multivariate_normal(
            mean,
            covariance,
            size=setting.n_right,
            check_valid="raise",
        )
        result = _simulate_replicate(
            left_samples,
            right_samples,
            feature_space=feature_space,
            minimum_projection_dimension=setting.minimum_projection_dimension,
            include_child_mean_rows=setting.include_child_mean_rows,
            ridge=ridge,
        )
        p_values[replicate_index] = result["p_value"]
        projection_dimensions[replicate_index] = result["projection_dimension"]
        raw_mp_signal_counts[replicate_index] = result["raw_mp_signal_count"]

    return _simulation_summary(
        setting=setting,
        n_replicates=n_replicates,
        alpha=alpha,
        p_values=p_values,
        projection_dimensions=projection_dimensions,
        raw_mp_signal_counts=raw_mp_signal_counts,
        contrast_dimension=feature_space.contrast_dimension,
    )


def _simulate_replicate(
    left_samples: np.ndarray,
    right_samples: np.ndarray,
    *,
    feature_space: FeatureSpace,
    minimum_projection_dimension: int,
    include_child_mean_rows: bool,
    ridge: float,
) -> dict[str, float]:
    pooled_samples = np.vstack([left_samples, right_samples])
    parent_mean = np.mean(pooled_samples, axis=0)
    left_mean = np.mean(left_samples, axis=0)
    right_mean = np.mean(right_samples, axis=0)
    pooled_covariance = np.cov(pooled_samples, rowvar=False, ddof=1)
    continuous_covariance_by_block = {"continuous": np.asarray(pooled_covariance)}

    z_scores = compute_whitened_wald_contrast(
        left_mean,
        right_mean,
        float(left_samples.shape[0]),
        float(right_samples.shape[0]),
        comparison="sibling",
        feature_space=feature_space,
        continuous_covariance_by_block=continuous_covariance_by_block,
        ridge=ridge,
    )
    spectral_rows = pooled_samples
    if include_child_mean_rows:
        spectral_rows = np.vstack([pooled_samples, left_mean, right_mean])

    tangent_rows = build_null_whitened_tangent_matrix(
        spectral_rows,
        parent_mean,
        feature_space=feature_space,
        continuous_covariance_by_block=continuous_covariance_by_block,
        ridge=ridge,
    )
    eigendecomposition = eigendecompose_covariance(
        tangent_rows,
        compute_eigenvectors=True,
    )
    if eigendecomposition is None:
        raise ValueError("Selected-PCA simulation produced zero-rank tangent data.")

    dimension_estimate = estimate_marchenko_pastur_dimension(
        eigendecomposition.eigenvalues,
        n_samples=tangent_rows.shape[0],
        n_features=eigendecomposition.active_feature_count,
        effective_independent_rows=pooled_samples.shape[0],
        mp_threshold_rows=tangent_rows.shape[0],
        minimum_projection_dimension=minimum_projection_dimension,
    )
    projection_matrix, pca_eigenvalues = build_pca_projection(
        eigendecomposition,
        projection_dimension=dimension_estimate.test_projection_dimension,
        n_features_total=feature_space.contrast_dimension,
    )
    projected = run_projected_wald_kernel(
        z_scores,
        spectral_k=projection_matrix.shape[0],
        pca_projection=projection_matrix,
        pca_eigenvalues=pca_eigenvalues,
    )
    return {
        "p_value": float(projected.p_value),
        "projection_dimension": float(projected.projection_dimension),
        "raw_mp_signal_count": float(dimension_estimate.raw_mp_signal_count),
    }


def _simulation_summary(
    *,
    setting: SelectedPcaCalibrationSetting,
    n_replicates: int,
    alpha: float,
    p_values: np.ndarray,
    projection_dimensions: np.ndarray,
    raw_mp_signal_counts: np.ndarray,
    contrast_dimension: int,
) -> dict[str, Any]:
    _validate_simulated_vectors(
        n_replicates=n_replicates,
        p_values=p_values,
        projection_dimensions=projection_dimensions,
        raw_mp_signal_counts=raw_mp_signal_counts,
    )
    return {
        "target_id": TARGET_ID,
        "setting_id": setting.setting_id,
        "setting": {
            "dimension": setting.dimension,
            "n_left": setting.n_left,
            "n_right": setting.n_right,
            "covariance_profile": setting.covariance_profile,
            "minimum_projection_dimension": setting.minimum_projection_dimension,
            "include_child_mean_rows": setting.include_child_mean_rows,
            "contrast_dimension": contrast_dimension,
        },
        **summarize_p_value_calibration(
            p_values=p_values,
            n_replicates=n_replicates,
            alpha=alpha,
        ),
        "projection_dimension_summary": _numeric_summary(projection_dimensions),
        "raw_mp_signal_count_summary": _numeric_summary(raw_mp_signal_counts),
    }


def _validate_complete_evidence(
    evidence: Mapping[str, Any],
    errors: list[str],
) -> None:
    if evidence.get("missing_required_fields") != []:
        errors.append("targets[0].evidence.missing_required_fields must be empty")
    metrics = evidence.get("metrics")
    if not isinstance(metrics, Mapping):
        errors.append("targets[0].evidence.metrics must be an object")
        return
    required_metric_keys = {
        "simulation_grid",
        "primary_endpoint",
        "results",
        "limitations",
        "dimension_grid",
        "sample_size_grid",
        "covariance_profile_grid",
        "minimum_projection_dimension_grid",
        "include_child_mean_rows_grid",
        "ridge",
    }
    missing_metric_keys = sorted(required_metric_keys.difference(metrics))
    if missing_metric_keys:
        errors.append(f"targets[0].evidence.metrics missing keys: {missing_metric_keys!r}")
    if metrics.get("primary_endpoint") != PRIMARY_ENDPOINT:
        errors.append("targets[0].evidence.metrics.primary_endpoint is invalid")
    results = metrics.get("results")
    if not isinstance(results, list) or not results:
        errors.append("targets[0].evidence.metrics.results must be a non-empty list")
        return
    for result_index, result in enumerate(results):
        if not isinstance(result, Mapping):
            errors.append(f"targets[0].evidence.metrics.results[{result_index}] must be an object")
            continue
        _validate_result_entry(
            result,
            context=f"targets[0].evidence.metrics.results[{result_index}]",
            errors=errors,
        )


def _validate_result_entry(
    result: Mapping[str, Any],
    *,
    context: str,
    errors: list[str],
) -> None:
    for key in (
        "target_id",
        "setting_id",
        "setting",
        "n_replicates",
        "alpha",
        "rejection_count",
        "rejection_rate",
        "confidence_interval",
        "effect_estimate",
        "p_value_uniformity_summary",
        "projection_dimension_summary",
        "raw_mp_signal_count_summary",
    ):
        if key not in result:
            errors.append(f"{context}.{key} is required")
    if result.get("target_id") != TARGET_ID:
        errors.append(f"{context}.target_id must be {TARGET_ID!r}")
    rejection_rate = result.get("rejection_rate")
    if isinstance(rejection_rate, int | float) and not 0.0 <= rejection_rate <= 1.0:
        errors.append(f"{context}.rejection_rate must be in [0, 1]")
    alpha = result.get("alpha")
    if isinstance(alpha, int | float) and not 0.0 < alpha < 1.0:
        errors.append(f"{context}.alpha must be in (0, 1)")


def _summary_rows(report: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    for result in report["targets"][0]["evidence"]["metrics"]["results"]:
        interval = result["confidence_interval"]
        effect = result["effect_estimate"]
        uniformity = result["p_value_uniformity_summary"]
        projection_summary = result["projection_dimension_summary"]
        mp_summary = result["raw_mp_signal_count_summary"]
        yield {
            "target_id": result["target_id"],
            "setting_id": result["setting_id"],
            "n_replicates": result["n_replicates"],
            "alpha": result["alpha"],
            "rejection_count": result["rejection_count"],
            "rejection_rate": result["rejection_rate"],
            "ci_low": interval["low"],
            "ci_high": interval["high"],
            "effect_estimate": effect["value"],
            "ks_statistic": uniformity["ks_statistic"],
            "ks_p_value": uniformity["ks_p_value"],
            "mean_projection_dimension": projection_summary["mean"],
            "mean_raw_mp_signal_count": mp_summary["mean"],
        }


def _validate_setting(setting: SelectedPcaCalibrationSetting) -> None:
    if setting.dimension <= 0:
        raise ValueError("dimension must be positive.")
    if setting.n_left <= 1 or setting.n_right <= 1:
        raise ValueError("sample sizes must be greater than 1.")
    if setting.minimum_projection_dimension < 0:
        raise ValueError("minimum_projection_dimension must be non-negative.")


def _validate_simulated_vectors(
    *,
    n_replicates: int,
    p_values: np.ndarray,
    projection_dimensions: np.ndarray,
    raw_mp_signal_counts: np.ndarray,
) -> None:
    expected_shape = (n_replicates,)
    if p_values.shape != expected_shape:
        raise ValueError(f"p_values has shape {p_values.shape}; expected {expected_shape}.")
    if projection_dimensions.shape != expected_shape:
        raise ValueError(
            "projection_dimensions has shape "
            f"{projection_dimensions.shape}; expected {expected_shape}."
        )
    if raw_mp_signal_counts.shape != expected_shape:
        raise ValueError(
            "raw_mp_signal_counts has shape "
            f"{raw_mp_signal_counts.shape}; expected {expected_shape}."
        )
    if not np.isfinite(p_values).all() or np.any(p_values < 0.0) or np.any(p_values > 1.0):
        raise ValueError("Simulated p-values must be finite values in [0, 1].")
    if not np.isfinite(projection_dimensions).all() or np.any(projection_dimensions < 0.0):
        raise ValueError("Projection dimensions must be finite and non-negative.")
    if not np.isfinite(raw_mp_signal_counts).all() or np.any(raw_mp_signal_counts < 0.0):
        raise ValueError("Raw MP signal counts must be finite and non-negative.")


def _numeric_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or validate selected-PCA projected-Wald calibration evidence."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser(
        "manifest",
        help="Create a missing-evidence manifest for selected-PCA calibration.",
    )
    manifest_parser.add_argument(
        "-o",
        "--output",
        default="benchmarks/validation/manifests/selected_pca_projected_wald_validation_manifest.json",
        help="Manifest JSON path to write.",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Run fixed-membership selected-PCA projected-Wald calibration simulations.",
    )
    run_parser.add_argument(
        "-o",
        "--output",
        default="benchmarks/results/validation/selected_pca_projected_wald_calibration.json",
        help="Report JSON path to write.",
    )
    run_parser.add_argument(
        "--csv-output",
        default=None,
        help="Optional summary CSV path.",
    )
    run_parser.add_argument("--replicates", type=int, default=500)
    run_parser.add_argument("--seed", type=int, default=20260601)
    run_parser.add_argument("--alpha", type=float, default=0.05)
    run_parser.add_argument("--ridge", type=float, default=1e-12)

    validate_parser = subparsers.add_parser("validate", help="Validate a JSON report.")
    validate_parser.add_argument("report", help="Report or manifest JSON path.")

    subparsers.add_parser("targets", help="Print validation targets.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "manifest":
        manifest = create_missing_manifest()
        path = write_selected_pca_projected_wald_report(manifest, args.output)
        print(f"Wrote selected-PCA projected-Wald manifest: {path}")
        return 0

    if args.command == "run":
        run_command = " ".join(["python", "-m", GENERATED_BY, *sys.argv[1:]])
        report = run_selected_pca_projected_wald_calibration(
            n_replicates=args.replicates,
            alpha=args.alpha,
            base_seed=args.seed,
            code_commit=read_git_commit(),
            git_worktree_status=read_git_worktree_status(),
            run_command=run_command,
            ridge=args.ridge,
        )
        report_path = write_selected_pca_projected_wald_report(report, args.output)
        print(f"Wrote selected-PCA projected-Wald calibration report: {report_path}")
        if args.csv_output is not None:
            csv_path = write_selected_pca_projected_wald_summary_csv(
                report,
                args.csv_output,
            )
            print(f"Wrote selected-PCA projected-Wald summary: {csv_path}")
        return 0

    if args.command == "validate":
        try:
            report = json.loads(Path(args.report).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"ERROR: unable to read report: {exc}", file=sys.stderr)
            return 1
        errors = validate_selected_pca_projected_wald_report(report)
        if errors:
            print("Selected-PCA projected-Wald report is invalid:", file=sys.stderr)
            for error in errors:
                print(f"- {error}", file=sys.stderr)
            return 1
        print("Selected-PCA projected-Wald report is valid.")
        return 0

    if args.command == "targets":
        print(f"{TARGET.target_id}: {TARGET.display_name}")
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
