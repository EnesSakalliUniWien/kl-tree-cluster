#!/usr/bin/env python3
"""Simulation scaffold for feature-covariance calibration evidence.

This module validates two covariance-model questions without changing the
production method:

* high-cardinality categorical drop-last multinomial calibration;
* finite-sample continuous empirical-Gaussian covariance calibration.

The simulations are local sibling-null checks in the full tangent space. They
are evidence for the covariance model under a fixed local contrast, not evidence
for full pipeline selection effects, MP dimension selection, sibling FDR, or
empirical-null inflation.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.stats import chi2, kstest
from tree_break_selection.hierarchy_analysis.statistics.contrast_covariance import (
    compute_whitened_wald_contrast,
)
from tree_break_selection.tree.feature_space import (
    FeatureSpace,
    continuous_feature_space_from_columns,
    infer_feature_space_from_columns,
)

from benchmarks.validation.report_contract import (
    completed_target_entry as _completed_target_entry,
)
from benchmarks.validation.report_contract import (
    validate_common_run_inputs as _validate_run_inputs,
)
from benchmarks.validation.report_contract import (
    validate_complete_report_context as _validate_complete_report_context,
)

SCHEMA_VERSION = "feature_covariance_calibration/v1"
GENERATED_BY = "benchmarks.validation.feature_covariance_calibration"
RANDOM_SEED_POLICY = (
    "A single numpy Generator is initialized from base_seed; all simulation "
    "settings draw independent consecutive replicates from that generator."
)
PRIMARY_ENDPOINT = "null_rejection_rate_at_alpha"
VALIDATION_DESIGN = "fixed_full_tangent_sibling_wald_null"
TARGET_STATUSES = ("missing", "partial", "complete")
TARGET_IDS = (
    "categorical_multinomial_drop_last_covariance",
    "continuous_empirical_gaussian_covariance",
)


@dataclass(frozen=True)
class ValidationTarget:
    """One covariance-validation target."""

    target_id: str
    display_name: str
    validation_question: str
    required_output_fields: tuple[str, ...]
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class CategoricalCalibrationSetting:
    """One high-cardinality categorical sibling-null setting."""

    n_features: int
    n_categories: int
    n_left: int
    n_right: int
    probability_profile: Literal["uniform", "rare_tail"]

    @property
    def setting_id(self) -> str:
        return (
            f"categorical_{self.n_features}features_{self.n_categories}categories_"
            f"{self.n_left}x{self.n_right}_{self.probability_profile}"
        )


@dataclass(frozen=True)
class ContinuousCalibrationSetting:
    """One continuous empirical-Gaussian sibling-null setting."""

    dimension: int
    n_left: int
    n_right: int
    covariance_profile: Literal["identity", "ar1_0.6", "ill_conditioned"]

    @property
    def setting_id(self) -> str:
        return (
            f"continuous_{self.dimension}dim_{self.n_left}x{self.n_right}_{self.covariance_profile}"
        )


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
    "limitations",
)

TARGETS: tuple[ValidationTarget, ...] = (
    ValidationTarget(
        target_id="categorical_multinomial_drop_last_covariance",
        display_name="Categorical multinomial drop-last covariance",
        validation_question=(
            "Does the drop-last multinomial covariance produce calibrated "
            "sibling-null Wald p-values in high-cardinality categorical blocks?"
        ),
        required_output_fields=COMMON_REQUIRED_OUTPUT_FIELDS
        + (
            "category_count_grid",
            "feature_count_grid",
            "probability_profile_grid",
        ),
        limitations=(
            "Fixed full-tangent local sibling null only.",
            "Does not validate PCA-selected projections, MP dimension selection, "
            "sibling FDR, traversal, tree construction, or empirical-null inflation.",
        ),
    ),
    ValidationTarget(
        target_id="continuous_empirical_gaussian_covariance",
        display_name="Continuous empirical-Gaussian covariance",
        validation_question=(
            "Does the per-node empirical-Gaussian covariance estimator produce "
            "calibrated sibling-null Wald p-values in finite samples?"
        ),
        required_output_fields=COMMON_REQUIRED_OUTPUT_FIELDS
        + (
            "dimension_grid",
            "sample_size_grid",
            "covariance_profile_grid",
            "ridge",
        ),
        limitations=(
            "Fixed full-tangent local sibling null only.",
            "Uses the production empirical pooled-node covariance estimate.",
            "Does not validate selected projections, full hierarchy construction, "
            "or continuous real-data model misspecification.",
        ),
    ),
)

DEFAULT_CATEGORICAL_SETTINGS = (
    CategoricalCalibrationSetting(
        n_features=4,
        n_categories=10,
        n_left=80,
        n_right=80,
        probability_profile="uniform",
    ),
    CategoricalCalibrationSetting(
        n_features=4,
        n_categories=20,
        n_left=120,
        n_right=120,
        probability_profile="rare_tail",
    ),
)
DEFAULT_CONTINUOUS_SETTINGS = (
    ContinuousCalibrationSetting(
        dimension=8,
        n_left=80,
        n_right=80,
        covariance_profile="identity",
    ),
    ContinuousCalibrationSetting(
        dimension=24,
        n_left=60,
        n_right=60,
        covariance_profile="ar1_0.6",
    ),
    ContinuousCalibrationSetting(
        dimension=32,
        n_left=40,
        n_right=40,
        covariance_profile="ill_conditioned",
    ),
)


def create_missing_manifest(*, created_utc: str | None = None) -> dict[str, Any]:
    """Return a missing-evidence manifest for covariance-validation targets."""

    return {
        "manifest_schema_version": SCHEMA_VERSION,
        "created_utc": created_utc or _utc_now(),
        "generated_by": GENERATED_BY,
        "validation_design": VALIDATION_DESIGN,
        "targets": [
            {
                "target_id": target.target_id,
                "display_name": target.display_name,
                "validation_question": target.validation_question,
                "required_output_fields": list(target.required_output_fields),
                "evidence_status": "missing",
                "evidence": {
                    "status": "missing",
                    "source_path": None,
                    "metrics": {},
                    "missing_required_fields": list(target.required_output_fields),
                    "notes": "No feature-covariance validation evidence has been attached.",
                },
            }
            for target in TARGETS
        ],
    }


def run_feature_covariance_calibration(
    *,
    n_replicates: int,
    alpha: float,
    base_seed: int,
    code_commit: str,
    git_worktree_status: Sequence[str],
    run_command: str,
    categorical_settings: Sequence[CategoricalCalibrationSetting] = DEFAULT_CATEGORICAL_SETTINGS,
    continuous_settings: Sequence[ContinuousCalibrationSetting] = DEFAULT_CONTINUOUS_SETTINGS,
    ridge: float = 1e-12,
    created_utc: str | None = None,
) -> dict[str, Any]:
    """Run local sibling-null covariance calibration simulations."""

    _validate_run_inputs(
        n_replicates=n_replicates,
        alpha=alpha,
        base_seed=base_seed,
        code_commit=code_commit,
        git_worktree_status=git_worktree_status,
        run_command=run_command,
        ridge=ridge,
    )
    rng = np.random.default_rng(base_seed)

    categorical_results = [
        _simulate_categorical_setting(
            rng,
            setting,
            n_replicates=n_replicates,
            alpha=alpha,
            ridge=ridge,
        )
        for setting in categorical_settings
    ]
    continuous_results = [
        _simulate_continuous_setting(
            rng,
            setting,
            n_replicates=n_replicates,
            alpha=alpha,
            ridge=ridge,
        )
        for setting in continuous_settings
    ]

    report = {
        "manifest_schema_version": SCHEMA_VERSION,
        "created_utc": created_utc or _utc_now(),
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
                TARGETS[0],
                categorical_results,
                extra_grids={
                    "category_count_grid": sorted(
                        {setting.n_categories for setting in categorical_settings}
                    ),
                    "feature_count_grid": sorted(
                        {setting.n_features for setting in categorical_settings}
                    ),
                    "probability_profile_grid": sorted(
                        {setting.probability_profile for setting in categorical_settings}
                    ),
                },
                primary_endpoint=PRIMARY_ENDPOINT,
            ),
            _completed_target_entry(
                TARGETS[1],
                continuous_results,
                extra_grids={
                    "dimension_grid": sorted(
                        {setting.dimension for setting in continuous_settings}
                    ),
                    "sample_size_grid": sorted(
                        {(setting.n_left, setting.n_right) for setting in continuous_settings}
                    ),
                    "covariance_profile_grid": sorted(
                        {setting.covariance_profile for setting in continuous_settings}
                    ),
                    "ridge": ridge,
                },
                primary_endpoint=PRIMARY_ENDPOINT,
            ),
        ],
    }

    errors = validate_feature_covariance_report(report)
    if errors:
        raise ValueError("invalid feature-covariance report:\n" + "\n".join(errors))
    return report


def validate_feature_covariance_report(report: Mapping[str, Any]) -> list[str]:
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
    has_complete_evidence = any(
        isinstance(entry, Mapping) and entry.get("evidence_status") == "complete"
        for entry in targets
    )
    if has_complete_evidence:
        _validate_complete_report_context(
            report,
            errors,
            primary_endpoint=PRIMARY_ENDPOINT,
        )
    target_ids = [entry.get("target_id") for entry in targets if isinstance(entry, Mapping)]
    if target_ids != list(TARGET_IDS):
        errors.append("target_id set and order must match feature-covariance targets")

    target_by_id = {target.target_id: target for target in TARGETS}
    for index, entry in enumerate(targets):
        context = f"targets[{index}]"
        if not isinstance(entry, Mapping):
            errors.append(f"{context} must be an object")
            continue
        target_id = entry.get("target_id")
        if not isinstance(target_id, str):
            errors.append(f"{context}.target_id must be a string")
            continue
        target = target_by_id.get(target_id)
        if target is None:
            errors.append(f"{context}.target_id is unknown: {target_id!r}")
            continue
        if entry.get("required_output_fields") != list(target.required_output_fields):
            errors.append(f"{context}.required_output_fields does not match target")
        status = entry.get("evidence_status")
        if status not in TARGET_STATUSES:
            errors.append(f"{context}.evidence_status has unsupported value: {status!r}")
        evidence = entry.get("evidence")
        if not isinstance(evidence, Mapping):
            errors.append(f"{context}.evidence must be an object")
            continue
        if evidence.get("status") != status:
            errors.append(f"{context}.evidence.status must match evidence_status")
        if status == "missing":
            if evidence.get("source_path") is not None:
                errors.append(f"{context}.evidence.source_path must be null when missing")
            if evidence.get("metrics") != {}:
                errors.append(f"{context}.evidence.metrics must be empty when missing")
            if evidence.get("missing_required_fields") != list(target.required_output_fields):
                errors.append(f"{context}.evidence.missing_required_fields must list all fields")
            continue
        _validate_complete_evidence(evidence, target, context, errors)
    return errors


def write_feature_covariance_report(report: Mapping[str, Any], output_path: str | Path) -> Path:
    """Validate and write a feature-covariance report JSON file."""

    errors = validate_feature_covariance_report(report)
    if errors:
        raise ValueError("invalid feature-covariance report:\n" + "\n".join(errors))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def write_feature_covariance_summary_csv(
    report: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """Write one summary row per simulated setting."""

    errors = validate_feature_covariance_report(report)
    if errors:
        raise ValueError("invalid feature-covariance report:\n" + "\n".join(errors))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(_summary_rows(report))
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
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _simulate_categorical_setting(
    rng: np.random.Generator,
    setting: CategoricalCalibrationSetting,
    *,
    n_replicates: int,
    alpha: float,
    ridge: float,
) -> dict[str, Any]:
    _validate_categorical_setting(setting)
    feature_space = _categorical_feature_space(
        n_features=setting.n_features,
        n_categories=setting.n_categories,
    )
    probabilities = _categorical_probabilities(setting)
    p_values = np.empty(n_replicates, dtype=np.float64)

    for replicate_index in range(n_replicates):
        left = _sample_categorical_distribution(
            rng,
            probabilities,
            sample_size=setting.n_left,
        )
        right = _sample_categorical_distribution(
            rng,
            probabilities,
            sample_size=setting.n_right,
        )
        p_values[replicate_index] = _sibling_full_tangent_p_value(
            left,
            right,
            float(setting.n_left),
            float(setting.n_right),
            feature_space=feature_space,
            ridge=ridge,
        )

    return _simulation_summary(
        target_id="categorical_multinomial_drop_last_covariance",
        setting_id=setting.setting_id,
        n_replicates=n_replicates,
        alpha=alpha,
        p_values=p_values,
        setting={
            "n_features": setting.n_features,
            "n_categories": setting.n_categories,
            "n_left": setting.n_left,
            "n_right": setting.n_right,
            "probability_profile": setting.probability_profile,
            "contrast_dimension": feature_space.contrast_dimension,
        },
    )


def _simulate_continuous_setting(
    rng: np.random.Generator,
    setting: ContinuousCalibrationSetting,
    *,
    n_replicates: int,
    alpha: float,
    ridge: float,
) -> dict[str, Any]:
    _validate_continuous_setting(setting)
    columns = tuple(f"X{index}" for index in range(setting.dimension))
    feature_space = continuous_feature_space_from_columns(columns)
    covariance = _continuous_covariance(setting)
    mean = np.zeros(setting.dimension, dtype=np.float64)
    p_values = np.empty(n_replicates, dtype=np.float64)

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
        left = np.mean(left_samples, axis=0)
        right = np.mean(right_samples, axis=0)
        pooled_samples = np.vstack([left_samples, right_samples])
        pooled_covariance = np.cov(pooled_samples, rowvar=False, ddof=1)
        p_values[replicate_index] = _sibling_full_tangent_p_value(
            left,
            right,
            float(setting.n_left),
            float(setting.n_right),
            feature_space=feature_space,
            continuous_covariance_by_block={"continuous": pooled_covariance},
            ridge=ridge,
        )

    return _simulation_summary(
        target_id="continuous_empirical_gaussian_covariance",
        setting_id=setting.setting_id,
        n_replicates=n_replicates,
        alpha=alpha,
        p_values=p_values,
        setting={
            "dimension": setting.dimension,
            "n_left": setting.n_left,
            "n_right": setting.n_right,
            "covariance_profile": setting.covariance_profile,
            "contrast_dimension": feature_space.contrast_dimension,
        },
    )


def _sibling_full_tangent_p_value(
    left: np.ndarray,
    right: np.ndarray,
    n_left: float,
    n_right: float,
    *,
    feature_space: FeatureSpace,
    ridge: float,
    continuous_covariance_by_block: Mapping[str, np.ndarray] | None = None,
) -> float:
    z_scores = compute_whitened_wald_contrast(
        left,
        right,
        n_left,
        n_right,
        comparison="sibling",
        feature_space=feature_space,
        continuous_covariance_by_block=continuous_covariance_by_block,
        ridge=ridge,
    )
    statistic = float(np.sum(z_scores**2))
    return float(chi2.sf(statistic, df=float(z_scores.shape[0])))


def _categorical_feature_space(*, n_features: int, n_categories: int) -> FeatureSpace:
    columns = tuple(
        f"F{feature_index}_c{category_index}"
        for feature_index in range(n_features)
        for category_index in range(n_categories)
    )
    return infer_feature_space_from_columns(columns)


def _categorical_probabilities(
    setting: CategoricalCalibrationSetting,
) -> np.ndarray:
    if setting.probability_profile == "uniform":
        probabilities = np.full(setting.n_categories, 1.0 / setting.n_categories)
    elif setting.probability_profile == "rare_tail":
        weights = 1.0 / np.arange(1, setting.n_categories + 1, dtype=np.float64)
        probabilities = weights / float(np.sum(weights))
    else:
        raise ValueError(f"Unsupported probability_profile: {setting.probability_profile!r}.")
    return np.tile(probabilities, (setting.n_features, 1))


def _sample_categorical_distribution(
    rng: np.random.Generator,
    probabilities: np.ndarray,
    *,
    sample_size: int,
) -> np.ndarray:
    blocks = [
        rng.multinomial(sample_size, probability_row).astype(np.float64) / sample_size
        for probability_row in probabilities
    ]
    return np.concatenate(blocks, axis=0)


def _continuous_covariance(setting: ContinuousCalibrationSetting) -> np.ndarray:
    if setting.covariance_profile == "identity":
        return np.eye(setting.dimension, dtype=np.float64)
    if setting.covariance_profile == "ar1_0.6":
        indices = np.arange(setting.dimension)
        return 0.6 ** np.abs(indices[:, None] - indices[None, :])
    if setting.covariance_profile == "ill_conditioned":
        eigenvalues = np.geomspace(1.0, 1e-3, num=setting.dimension)
        return np.diag(eigenvalues)
    raise ValueError(f"Unsupported covariance_profile: {setting.covariance_profile!r}.")


def _simulation_summary(
    *,
    target_id: str,
    setting_id: str,
    n_replicates: int,
    alpha: float,
    p_values: np.ndarray,
    setting: Mapping[str, Any],
) -> dict[str, Any]:
    if p_values.shape != (n_replicates,):
        raise ValueError(f"p_values has shape {p_values.shape}; expected {(n_replicates,)}.")
    if not np.isfinite(p_values).all() or np.any(p_values < 0.0) or np.any(p_values > 1.0):
        raise ValueError("Simulated p-values must be finite values in [0, 1].")
    rejection_count = int(np.sum(p_values < alpha))
    rejection_rate = float(rejection_count / n_replicates)
    ci_low, ci_high = _wilson_interval(rejection_count, n_replicates)
    ks_result = kstest(p_values, "uniform")
    return {
        "target_id": target_id,
        "setting_id": setting_id,
        "setting": dict(setting),
        "n_replicates": n_replicates,
        "alpha": alpha,
        "rejection_count": rejection_count,
        "rejection_rate": rejection_rate,
        "confidence_interval": {
            "method": "wilson_95",
            "low": ci_low,
            "high": ci_high,
        },
        "effect_estimate": {
            "name": "rejection_rate_minus_alpha",
            "value": float(rejection_rate - alpha),
        },
        "p_value_uniformity_summary": {
            "ks_statistic": float(ks_result.statistic),
            "ks_p_value": float(ks_result.pvalue),
            "mean": float(np.mean(p_values)),
            "median": float(np.median(p_values)),
            "q05": float(np.quantile(p_values, 0.05)),
            "q95": float(np.quantile(p_values, 0.95)),
        },
    }


def _validate_complete_evidence(
    evidence: Mapping[str, Any],
    target: ValidationTarget,
    context: str,
    errors: list[str],
) -> None:
    if evidence.get("missing_required_fields") != []:
        errors.append(f"{context}.evidence.missing_required_fields must be empty")
    metrics = evidence.get("metrics")
    if not isinstance(metrics, Mapping):
        errors.append(f"{context}.evidence.metrics must be an object")
        return
    required_metric_keys = {
        "simulation_grid",
        "primary_endpoint",
        "results",
        "limitations",
    }
    missing_metric_keys = sorted(required_metric_keys.difference(metrics))
    if missing_metric_keys:
        errors.append(f"{context}.evidence.metrics missing keys: {missing_metric_keys!r}")
    target_specific_metric_keys = sorted(
        set(target.required_output_fields).difference(COMMON_REQUIRED_OUTPUT_FIELDS)
    )
    missing_target_keys = [key for key in target_specific_metric_keys if key not in metrics]
    if missing_target_keys:
        errors.append(f"{context}.evidence.metrics missing target keys: {missing_target_keys!r}")
    if metrics.get("primary_endpoint") != PRIMARY_ENDPOINT:
        errors.append(f"{context}.evidence.metrics.primary_endpoint is invalid")
    results = metrics.get("results")
    if not isinstance(results, list) or not results:
        errors.append(f"{context}.evidence.metrics.results must be a non-empty list")
        return
    for result_index, result in enumerate(results):
        if not isinstance(result, Mapping):
            errors.append(f"{context}.evidence.metrics.results[{result_index}] must be an object")
            continue
        _validate_result_entry(
            result,
            target_id=target.target_id,
            context=f"{context}.evidence.metrics.results[{result_index}]",
            errors=errors,
        )


def _validate_result_entry(
    result: Mapping[str, Any],
    *,
    target_id: str,
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
    ):
        if key not in result:
            errors.append(f"{context}.{key} is required")
    if result.get("target_id") != target_id:
        errors.append(f"{context}.target_id must be {target_id!r}")
    rejection_rate = result.get("rejection_rate")
    if isinstance(rejection_rate, int | float) and not 0.0 <= rejection_rate <= 1.0:
        errors.append(f"{context}.rejection_rate must be in [0, 1]")
    alpha = result.get("alpha")
    if isinstance(alpha, int | float) and not 0.0 < alpha < 1.0:
        errors.append(f"{context}.alpha must be in (0, 1)")


def _summary_rows(report: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    for target in report["targets"]:
        metrics = target["evidence"]["metrics"]
        for result in metrics["results"]:
            interval = result["confidence_interval"]
            effect = result["effect_estimate"]
            uniformity = result["p_value_uniformity_summary"]
            yield {
                "target_id": target["target_id"],
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
            }


def _validate_categorical_setting(setting: CategoricalCalibrationSetting) -> None:
    if setting.n_features <= 0:
        raise ValueError("categorical n_features must be positive.")
    if setting.n_categories < 2:
        raise ValueError("categorical n_categories must be at least 2.")
    if setting.n_left <= 0 or setting.n_right <= 0:
        raise ValueError("categorical sample sizes must be positive.")


def _validate_continuous_setting(setting: ContinuousCalibrationSetting) -> None:
    if setting.dimension <= 0:
        raise ValueError("continuous dimension must be positive.")
    if setting.n_left <= 1 or setting.n_right <= 1:
        raise ValueError("continuous sample sizes must be greater than 1.")


def _wilson_interval(
    successes: int, total: int, *, z_value: float = 1.959963984540054
) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("total must be positive.")
    proportion = successes / total
    denominator = 1.0 + z_value**2 / total
    center = (proportion + z_value**2 / (2.0 * total)) / denominator
    half_width = (
        z_value
        * np.sqrt((proportion * (1.0 - proportion) + z_value**2 / (4.0 * total)) / total)
        / denominator
    )
    return float(max(0.0, center - half_width)), float(min(1.0, center + half_width))


def _read_git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if not commit:
        raise RuntimeError("git rev-parse HEAD returned an empty commit.")
    return commit


def _read_git_worktree_status() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or validate feature-covariance calibration evidence."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser(
        "manifest",
        help="Create a missing-evidence manifest for feature covariance targets.",
    )
    manifest_parser.add_argument(
        "-o",
        "--output",
        default="benchmarks/validation/manifests/feature_covariance_validation_manifest.json",
        help="Manifest JSON path to write.",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Run local sibling-null covariance calibration simulations.",
    )
    run_parser.add_argument(
        "-o",
        "--output",
        default="benchmarks/results/validation/feature_covariance_calibration.json",
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
        path = write_feature_covariance_report(manifest, args.output)
        print(f"Wrote feature-covariance manifest: {path}")
        return 0

    if args.command == "run":
        run_command = " ".join(["python", "-m", GENERATED_BY, *sys.argv[1:]])
        report = run_feature_covariance_calibration(
            n_replicates=args.replicates,
            alpha=args.alpha,
            base_seed=args.seed,
            code_commit=_read_git_commit(),
            git_worktree_status=_read_git_worktree_status(),
            run_command=run_command,
            ridge=args.ridge,
        )
        report_path = write_feature_covariance_report(report, args.output)
        print(f"Wrote feature-covariance calibration report: {report_path}")
        if args.csv_output is not None:
            csv_path = write_feature_covariance_summary_csv(report, args.csv_output)
            print(f"Wrote feature-covariance calibration summary: {csv_path}")
        return 0

    if args.command == "validate":
        try:
            report = json.loads(Path(args.report).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"ERROR: unable to read report: {exc}", file=sys.stderr)
            return 1
        errors = validate_feature_covariance_report(report)
        if errors:
            print("Feature-covariance report is invalid:", file=sys.stderr)
            for error in errors:
                print(f"- {error}", file=sys.stderr)
            return 1
        print("Feature-covariance report is valid.")
        return 0

    if args.command == "targets":
        for target in TARGETS:
            print(f"{target.target_id}: {target.display_name}")
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
