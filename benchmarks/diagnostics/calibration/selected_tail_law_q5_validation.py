"""Q5 selected-tail law validation diagnostic.

This diagnostic fits predeclared selected-tail laws for the selected ratio
``R = W / E_ref`` using edge severity, parent size, feature aspect ratio,
projection dimension, feature family, barycentric leverage, and spectral
geometry. It evaluates the laws with held-out replicates, cases, feature
families, and parent-size bins.

The output is diagnostic evidence only. It does not install an external
production calibrator or a fallback for unsupported runtime contexts.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "diagnostic_q5_selected_tail_law_not_calibration"
RESPONSE_COLUMN = "log_selected_hierarchy_ratio"
SIMULATION_ID_COLUMN = "selected_hierarchy_simulation_id"

REQUIRED_COLUMNS = {
    "case_id",
    "feature_family",
    "parent_size_bin",
    "replicate_index",
    SIMULATION_ID_COLUMN,
    RESPONSE_COLUMN,
    "negative_log10_min_child_edge_bh_p_value",
    "feature_dimension",
    "parent_sample_size",
    "left_child_sample_size",
    "right_child_sample_size",
    "sibling_projection_dimension",
    "selected_eigenvalue_over_mp_upper_bound",
    "selected_eigenvalue_mass_fraction",
    "eigenvalue_effective_rank",
}


@dataclass(frozen=True)
class ModelSpec:
    """Predeclared selected-tail law model."""

    model_id: str
    description: str
    predictors: tuple[str, ...]


def _positive_log(values: pd.Series, *, value_name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise").astype(float)
    if bool((numeric <= 0.0).any()):
        bad_index = numeric[numeric <= 0.0].index[0]
        raise ValueError(
            f"{value_name} must be positive to enter log scale; "
            f"row={int(bad_index)}, value={float(numeric.loc[bad_index])!r}."
        )
    return np.log(numeric)


def _positive_numeric(values: pd.Series, *, value_name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise").astype(float)
    if bool(((numeric <= 0.0) | ~np.isfinite(numeric)).any()):
        bad_index = numeric[((numeric <= 0.0) | ~np.isfinite(numeric))].index[0]
        raise ValueError(
            f"{value_name} must contain finite positive values; "
            f"row={int(bad_index)}, value={float(numeric.loc[bad_index])!r}."
        )
    return numeric


def _dummy_columns(table: pd.DataFrame, column: str) -> tuple[str, ...]:
    values = tuple(sorted(str(value) for value in table[column].dropna().unique()))
    if len(values) <= 1:
        return ()
    created: list[str] = []
    baseline = values[0]
    for value in values[1:]:
        column_name = f"{column}__{value}"
        table[column_name] = table[column].astype(str).eq(value).astype(float)
        created.append(column_name)
    table[f"{column}__baseline"] = baseline
    return tuple(created)


def prepare_q5_selected_tail_records(records: pd.DataFrame) -> pd.DataFrame:
    """Return a modeling table with predeclared Q5 predictors."""
    missing = REQUIRED_COLUMNS - set(records.columns)
    if missing:
        raise ValueError(f"Selected-tail Q5 records are missing columns: {sorted(missing)!r}.")

    table = records.copy()
    table["edge_action"] = pd.to_numeric(
        table["negative_log10_min_child_edge_bh_p_value"],
        errors="raise",
    ).astype(float)
    parent_sample_size = _positive_numeric(
        table["parent_sample_size"],
        value_name="parent_sample_size",
    )
    left_child_sample_size = _positive_numeric(
        table["left_child_sample_size"],
        value_name="left_child_sample_size",
    )
    right_child_sample_size = _positive_numeric(
        table["right_child_sample_size"],
        value_name="right_child_sample_size",
    )
    feature_dimension = _positive_numeric(
        table["feature_dimension"],
        value_name="feature_dimension",
    )
    beta_left = left_child_sample_size / parent_sample_size
    beta_right = right_child_sample_size / parent_sample_size
    beta_sum_error = np.abs(beta_left + beta_right - 1.0)
    if bool((beta_sum_error > 1e-8).any()):
        bad_index = beta_sum_error[beta_sum_error > 1e-8].index[0]
        raise ValueError(
            "left_child_sample_size + right_child_sample_size must equal "
            "parent_sample_size for barycentric Q5 predictors; "
            f"row={int(bad_index)}."
        )
    barycentric_balance = np.minimum(beta_left, beta_right)
    if bool((barycentric_balance <= 0.0).any()):
        bad_index = barycentric_balance[barycentric_balance <= 0.0].index[0]
        raise ValueError(
            "Barycentric balance must be positive for selected-tail modeling; "
            f"row={int(bad_index)}."
        )
    table["log_parent_sample_size"] = np.log(parent_sample_size)
    table["log_feature_parent_aspect_ratio"] = np.log(feature_dimension / parent_sample_size)
    table["left_barycentric_weight"] = beta_left
    table["barycentric_balance"] = barycentric_balance
    table["log_barycentric_leverage"] = np.log(
        np.maximum(beta_left, beta_right) / barycentric_balance
    )
    table["log_sampling_variance_scale"] = np.log(
        (1.0 / left_child_sample_size) + (1.0 / right_child_sample_size)
    )
    table["sibling_projection_dimension"] = pd.to_numeric(
        table["sibling_projection_dimension"],
        errors="raise",
    ).astype(float)
    table["log_selected_eigenvalue_over_mp_upper_bound"] = _positive_log(
        table["selected_eigenvalue_over_mp_upper_bound"],
        value_name="selected_eigenvalue_over_mp_upper_bound",
    )
    table["selected_eigenvalue_mass_fraction"] = pd.to_numeric(
        table["selected_eigenvalue_mass_fraction"],
        errors="raise",
    ).astype(float)
    table["eigenvalue_effective_rank"] = pd.to_numeric(
        table["eigenvalue_effective_rank"],
        errors="raise",
    ).astype(float)
    table[RESPONSE_COLUMN] = pd.to_numeric(table[RESPONSE_COLUMN], errors="raise").astype(
        float
    )
    table["replicate_index"] = pd.to_numeric(table["replicate_index"], errors="raise").astype(
        int
    )

    _dummy_columns(table, "feature_family")
    _dummy_columns(table, "parent_size_bin")
    modeling_columns = [
        "case_id",
        "feature_family",
        "parent_size_bin",
        "replicate_index",
        SIMULATION_ID_COLUMN,
        RESPONSE_COLUMN,
        "edge_action",
        "log_parent_sample_size",
        "log_feature_parent_aspect_ratio",
        "sibling_projection_dimension",
        "left_barycentric_weight",
        "barycentric_balance",
        "log_barycentric_leverage",
        "log_sampling_variance_scale",
        "log_selected_eigenvalue_over_mp_upper_bound",
        "selected_eigenvalue_mass_fraction",
        "eigenvalue_effective_rank",
        *[
            column
            for column in table.columns
            if column.startswith("feature_family__")
            and not column.endswith("__baseline")
        ],
        *[
            column
            for column in table.columns
            if column.startswith("parent_size_bin__")
            and not column.endswith("__baseline")
        ],
    ]
    model_table = table[modeling_columns].replace([np.inf, -np.inf], np.nan).dropna()
    if model_table.empty:
        raise ValueError("Selected-tail Q5 model table has no finite rows.")
    return model_table


def q5_model_specs(model_table: pd.DataFrame) -> tuple[ModelSpec, ...]:
    """Return nested predeclared models for Q5 validation."""
    feature_family_columns = tuple(
        column
        for column in model_table.columns
        if column.startswith("feature_family__")
        and not column.endswith("__baseline")
    )
    parent_size_columns = tuple(
        column
        for column in model_table.columns
        if column.startswith("parent_size_bin__")
        and not column.endswith("__baseline")
    )
    context_predictors = (
        "edge_action",
        "log_parent_sample_size",
        "log_feature_parent_aspect_ratio",
        "sibling_projection_dimension",
        *feature_family_columns,
        *parent_size_columns,
    )
    barycentric_predictors = (
        "barycentric_balance",
        "log_barycentric_leverage",
        "log_sampling_variance_scale",
    )
    spectral_predictors = (
        "log_selected_eigenvalue_over_mp_upper_bound",
        "selected_eigenvalue_mass_fraction",
        "eigenvalue_effective_rank",
    )
    return (
        ModelSpec(
            model_id="q5_without_spectral_geometry",
            description=(
                "log_R ~ edge_action + log(parent_n) + log(p/parent_n) + "
                "projection_dim + feature_family + parent_size_bin"
            ),
            predictors=context_predictors,
        ),
        ModelSpec(
            model_id="q5_barycentric_context",
            description=(
                "log_R ~ edge_action + log(parent_n) + log(p/parent_n) + "
                "projection_dim + feature_family + parent_size_bin + "
                "barycentric_balance + log_barycentric_leverage + log_sampling_scale"
            ),
            predictors=(*context_predictors, *barycentric_predictors),
        ),
        ModelSpec(
            model_id="q5_edge_spectral_only",
            description=(
                "log_R ~ edge_action + log(lambda_k/lambda_MP) + spectral_mass + "
                "effective_rank"
            ),
            predictors=("edge_action", *spectral_predictors),
        ),
        ModelSpec(
            model_id="q5_barycentric_edge_spectral",
            description=(
                "log_R ~ edge_action + barycentric_balance + "
                "log_barycentric_leverage + log_sampling_scale + spectral_geometry"
            ),
            predictors=("edge_action", *barycentric_predictors, *spectral_predictors),
        ),
        ModelSpec(
            model_id="q5_full_selected_tail_law",
            description=(
                "log_R ~ edge_action + log(parent_n) + log(p/parent_n) + projection_dim + "
                "feature_family + parent_size_bin + spectral_geometry"
            ),
            predictors=(*context_predictors, *spectral_predictors),
        ),
        ModelSpec(
            model_id="q5_barycentric_full_selected_tail_law",
            description=(
                "log_R ~ edge_action + log(parent_n) + log(p/parent_n) + "
                "projection_dim + feature_family + parent_size_bin + "
                "barycentric_balance + log_barycentric_leverage + "
                "log_sampling_scale + spectral_geometry"
            ),
            predictors=(*context_predictors, *barycentric_predictors, *spectral_predictors),
        ),
    )


def _binary_auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    if scores.shape[0] != labels.shape[0]:
        raise ValueError("scores and labels must have the same length.")
    labels = labels.astype(bool)
    n_positive = int(np.sum(labels))
    n_negative = int(labels.shape[0] - n_positive)
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.shape[0], dtype=float)
    sorted_scores = scores[order]
    start = 0
    while start < scores.shape[0]:
        end = start + 1
        while end < scores.shape[0] and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = average_rank
        start = end
    rank_sum_positive = float(np.sum(ranks[labels]))
    return float(
        (rank_sum_positive - n_positive * (n_positive + 1) / 2.0)
        / (n_positive * n_negative)
    )


def _fold_ids(table: pd.DataFrame, split_strategy: str, *, n_replicate_folds: int) -> pd.Series:
    if split_strategy == "replicate_modulo":
        return pd.Series(
            table["replicate_index"].astype(int) % int(n_replicate_folds),
            index=table.index,
            dtype=object,
        )
    if split_strategy == "leave_one_case_out":
        return table["case_id"].astype(str)
    if split_strategy == "leave_one_feature_family_out":
        return table["feature_family"].astype(str)
    if split_strategy == "leave_one_parent_size_bin_out":
        return table["parent_size_bin"].astype(str)
    raise ValueError(f"Unknown split strategy {split_strategy!r}.")


def _fit_tail_model(
    train: pd.DataFrame,
    *,
    predictors: Sequence[str],
    response: str,
    alpha: float,
    min_train_rows_per_predictor: int,
) -> dict[str, object]:
    finite_train = train[list(predictors) + [response]].replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()
    active_predictors = tuple(
        predictor
        for predictor in predictors
        if finite_train[predictor].nunique(dropna=True) > 1
    )
    minimum_rows = max(
        len(active_predictors) + 2,
        int(min_train_rows_per_predictor) * max(len(active_predictors), 1),
    )
    if finite_train.shape[0] < minimum_rows:
        return {
            "status": "insufficient_train_rows",
            "active_predictors": active_predictors,
            "n_train_rows": int(finite_train.shape[0]),
        }
    if not active_predictors:
        return {
            "status": "no_nonconstant_predictors",
            "active_predictors": active_predictors,
            "n_train_rows": int(finite_train.shape[0]),
        }

    x_raw = finite_train[list(active_predictors)].to_numpy(dtype=float)
    means = x_raw.mean(axis=0)
    stds = x_raw.std(axis=0, ddof=0)
    if np.any(stds <= 0.0):
        return {
            "status": "zero_standard_deviation_after_filtering",
            "active_predictors": active_predictors,
            "n_train_rows": int(finite_train.shape[0]),
        }
    x = (x_raw - means) / stds
    design = np.column_stack([np.ones(x.shape[0]), x])
    rank = int(np.linalg.matrix_rank(design))
    if rank < design.shape[1]:
        return {
            "status": "rank_deficient_design",
            "active_predictors": active_predictors,
            "n_train_rows": int(finite_train.shape[0]),
            "matrix_rank": rank,
        }

    y = finite_train[response].to_numpy(dtype=float)
    coefficients, _residuals, _rank, singular_values = np.linalg.lstsq(
        design,
        y,
        rcond=None,
    )
    fitted = design @ coefficients
    residuals = y - fitted
    residual_tail_quantile = float(np.quantile(residuals, 1.0 - float(alpha)))
    return {
        "status": "ok",
        "active_predictors": active_predictors,
        "n_train_rows": int(finite_train.shape[0]),
        "coefficients": coefficients,
        "means": means,
        "standard_deviations": stds,
        "train_response_mean": float(np.mean(y)),
        "train_tail_threshold": float(np.quantile(y, 0.9)),
        "residual_tail_quantile": residual_tail_quantile,
        "matrix_rank": rank,
        "condition_number": float(singular_values[0] / singular_values[-1]),
    }


def _predict_tail_model(model: Mapping[str, object], test: pd.DataFrame) -> np.ndarray:
    active_predictors = tuple(str(value) for value in model["active_predictors"])
    x_raw = test[list(active_predictors)].to_numpy(dtype=float)
    means = np.asarray(model["means"], dtype=float)
    stds = np.asarray(model["standard_deviations"], dtype=float)
    x = (x_raw - means) / stds
    design = np.column_stack([np.ones(x.shape[0]), x])
    coefficients = np.asarray(model["coefficients"], dtype=float)
    return design @ coefficients


def _defined_r2(observed: np.ndarray, predicted: np.ndarray, baseline: np.ndarray) -> float:
    baseline_sum_sq = float(np.sum((observed - baseline) ** 2))
    if baseline_sum_sq <= 0.0:
        return float("nan")
    residual_sum_sq = float(np.sum((observed - predicted) ** 2))
    return float(1.0 - residual_sum_sq / baseline_sum_sq)


def _evaluate_one_model_split(
    table: pd.DataFrame,
    *,
    model_spec: ModelSpec,
    split_strategy: str,
    fold_ids: pd.Series,
    alpha: float,
    tail_quantile: float,
    min_train_rows_per_predictor: int,
    min_test_rows: int,
) -> dict[str, object]:
    predictions: list[np.ndarray] = []
    observed_values: list[np.ndarray] = []
    baseline_values: list[np.ndarray] = []
    tail_labels: list[np.ndarray] = []
    residual_threshold_exceedances: list[np.ndarray] = []
    residual_tail_quantiles: list[float] = []
    active_predictor_counts: list[int] = []
    matrix_ranks: list[int] = []
    condition_numbers: list[float] = []
    failures: list[str] = []
    n_train_rows = 0
    n_test_rows = 0

    required_columns = list(model_spec.predictors) + [RESPONSE_COLUMN]
    model_table = table[required_columns].replace([np.inf, -np.inf], np.nan).dropna()
    model_fold_ids = fold_ids.loc[model_table.index]
    for fold in tuple(sorted(model_fold_ids.unique())):
        train = model_table.loc[model_fold_ids.ne(fold)]
        test = model_table.loc[model_fold_ids.eq(fold)]
        n_train_rows += int(train.shape[0])
        n_test_rows += int(test.shape[0])
        if test.shape[0] < min_test_rows:
            failures.append(f"fold_{fold}:insufficient_test_rows")
            continue
        model = _fit_tail_model(
            train,
            predictors=model_spec.predictors,
            response=RESPONSE_COLUMN,
            alpha=alpha,
            min_train_rows_per_predictor=min_train_rows_per_predictor,
        )
        if model["status"] != "ok":
            failures.append(f"fold_{fold}:{model['status']}")
            continue
        y_test = test[RESPONSE_COLUMN].to_numpy(dtype=float)
        y_pred = _predict_tail_model(model, test)
        baseline = np.full(y_test.shape[0], float(model["train_response_mean"]))
        train_tail_threshold = float(np.quantile(train[RESPONSE_COLUMN].to_numpy(), tail_quantile))
        residual_threshold = y_pred + float(model["residual_tail_quantile"])

        predictions.append(y_pred)
        observed_values.append(y_test)
        baseline_values.append(baseline)
        tail_labels.append(y_test >= train_tail_threshold)
        residual_threshold_exceedances.append(y_test > residual_threshold)
        residual_tail_quantiles.append(float(model["residual_tail_quantile"]))
        active_predictor_counts.append(len(tuple(model["active_predictors"])))
        matrix_ranks.append(int(model["matrix_rank"]))
        condition_numbers.append(float(model["condition_number"]))

    status = (
        f"diagnostic_holdout_{split_strategy}"
        if predictions
        else f"no_valid_holdout_folds_{split_strategy}"
    )
    if not predictions:
        return {
            "model_id": model_spec.model_id,
            "model_description": model_spec.description,
            "predictors": ",".join(model_spec.predictors),
            "split_strategy": split_strategy,
            "alpha": float(alpha),
            "tail_quantile": float(tail_quantile),
            "n_folds": int(fold_ids.nunique()),
            "n_train_rows": int(n_train_rows),
            "n_test_rows": int(n_test_rows),
            "holdout_log_ratio_r_squared": np.nan,
            "holdout_mean_absolute_log_error": np.nan,
            "holdout_median_absolute_log_error": np.nan,
            "holdout_tail_auc_from_linear_score": np.nan,
            "holdout_tail_event_rate": np.nan,
            "residual_tail_exceedance_rate": np.nan,
            "residual_tail_exceedance_absolute_error": np.nan,
            "residual_tail_exceedance_standard_error": np.nan,
            "residual_tail_quantile_mean": np.nan,
            "active_predictor_count_mean": np.nan,
            "matrix_rank_min": np.nan,
            "condition_number_max": np.nan,
            "model_status": status,
            "failure_reasons": ";".join(failures),
            "study_role": STUDY_ROLE,
        }

    predicted = np.concatenate(predictions)
    observed = np.concatenate(observed_values)
    baseline = np.concatenate(baseline_values)
    labels = np.concatenate(tail_labels)
    exceedances = np.concatenate(residual_threshold_exceedances)
    absolute_error = np.abs(observed - predicted)
    exceedance_rate = float(np.mean(exceedances))
    exceedance_se = float(
        np.sqrt(exceedance_rate * (1.0 - exceedance_rate) / exceedances.shape[0])
    )
    return {
        "model_id": model_spec.model_id,
        "model_description": model_spec.description,
        "predictors": ",".join(model_spec.predictors),
        "split_strategy": split_strategy,
        "alpha": float(alpha),
        "tail_quantile": float(tail_quantile),
        "n_folds": int(fold_ids.nunique()),
        "n_train_rows": int(n_train_rows),
        "n_test_rows": int(n_test_rows),
        "holdout_log_ratio_r_squared": _defined_r2(observed, predicted, baseline),
        "holdout_mean_absolute_log_error": float(np.mean(absolute_error)),
        "holdout_median_absolute_log_error": float(np.median(absolute_error)),
        "holdout_tail_auc_from_linear_score": _binary_auc_score(predicted, labels),
        "holdout_tail_event_rate": float(np.mean(labels)),
        "residual_tail_exceedance_rate": exceedance_rate,
        "residual_tail_exceedance_absolute_error": float(abs(exceedance_rate - alpha)),
        "residual_tail_exceedance_standard_error": exceedance_se,
        "residual_tail_quantile_mean": float(np.mean(residual_tail_quantiles)),
        "active_predictor_count_mean": float(np.mean(active_predictor_counts)),
        "matrix_rank_min": int(np.min(matrix_ranks)),
        "condition_number_max": float(np.max(condition_numbers)),
        "model_status": status,
        "failure_reasons": ";".join(failures),
        "study_role": STUDY_ROLE,
    }


def evaluate_q5_selected_tail_law(
    records: pd.DataFrame,
    *,
    alpha: float = 0.01,
    tail_quantile: float = 0.9,
    n_replicate_folds: int = 5,
    min_train_rows_per_predictor: int = 20,
    min_test_rows: int = 10,
) -> pd.DataFrame:
    """Fit and validate predeclared Q5 selected-tail law candidates."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1); got {alpha!r}.")
    if not 0.0 < tail_quantile < 1.0:
        raise ValueError(f"tail_quantile must lie in (0, 1); got {tail_quantile!r}.")
    if n_replicate_folds < 2:
        raise ValueError("n_replicate_folds must be at least 2.")

    table = prepare_q5_selected_tail_records(records)
    model_specs = q5_model_specs(table)
    split_strategies = (
        "replicate_modulo",
        "leave_one_case_out",
        "leave_one_feature_family_out",
        "leave_one_parent_size_bin_out",
    )
    rows: list[dict[str, object]] = []
    for split_strategy in split_strategies:
        fold_ids = _fold_ids(
            table,
            split_strategy,
            n_replicate_folds=n_replicate_folds,
        )
        for model_spec in model_specs:
            rows.append(
                _evaluate_one_model_split(
                    table,
                    model_spec=model_spec,
                    split_strategy=split_strategy,
                    fold_ids=fold_ids,
                    alpha=alpha,
                    tail_quantile=tail_quantile,
                    min_train_rows_per_predictor=min_train_rows_per_predictor,
                    min_test_rows=min_test_rows,
                )
            )
    return pd.DataFrame.from_records(rows)


def summarize_q5_selected_tail_law(validation: pd.DataFrame) -> pd.DataFrame:
    """Return one compact summary row per model."""
    if validation.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for model_id, group in validation.groupby("model_id", sort=True):
        valid = group[group["model_status"].astype(str).str.startswith("diagnostic_holdout")]
        rows.append(
            {
                "model_id": model_id,
                "n_valid_splits": int(valid.shape[0]),
                "best_tail_auc": float(valid["holdout_tail_auc_from_linear_score"].max())
                if not valid.empty
                else np.nan,
                "median_tail_auc": float(valid["holdout_tail_auc_from_linear_score"].median())
                if not valid.empty
                else np.nan,
                "median_holdout_r_squared": float(valid["holdout_log_ratio_r_squared"].median())
                if not valid.empty
                else np.nan,
                "median_residual_tail_exceedance_absolute_error": float(
                    valid["residual_tail_exceedance_absolute_error"].median()
                )
                if not valid.empty
                else np.nan,
                "study_role": STUDY_ROLE,
            }
        )
    return pd.DataFrame.from_records(rows)


def run_q5_selected_tail_law_validation(
    *,
    records_path: Path,
    output_dir: Path,
    alpha: float = 0.01,
    tail_quantile: float = 0.9,
    n_replicate_folds: int = 5,
    min_train_rows_per_predictor: int = 20,
    min_test_rows: int = 10,
) -> dict[str, Path]:
    """Run Q5 selected-tail law validation from a row-level record CSV."""
    records = pd.read_csv(records_path)
    validation = evaluate_q5_selected_tail_law(
        records,
        alpha=alpha,
        tail_quantile=tail_quantile,
        n_replicate_folds=n_replicate_folds,
        min_train_rows_per_predictor=min_train_rows_per_predictor,
        min_test_rows=min_test_rows,
    )
    summary = summarize_q5_selected_tail_law(validation)

    output_dir.mkdir(parents=True, exist_ok=True)
    validation_path = output_dir / "q5_selected_tail_law_validation.csv"
    summary_path = output_dir / "q5_selected_tail_law_summary.csv"
    manifest_path = output_dir / "manifest.json"
    validation.to_csv(validation_path, index=False)
    summary.to_csv(summary_path, index=False)
    manifest = {
        "created_at_utc": format_timestamp_utc(),
        "study_role": STUDY_ROLE,
        "records_path": str(records_path),
        "alpha": float(alpha),
        "tail_quantile": float(tail_quantile),
        "n_replicate_folds": int(n_replicate_folds),
        "min_train_rows_per_predictor": int(min_train_rows_per_predictor),
        "min_test_rows": int(min_test_rows),
        "outputs": {
            "validation": str(validation_path),
            "summary": str(summary_path),
        },
        "interpretation": (
            "Diagnostic validation of a predeclared Q5 selected-tail law. "
            "Rows do not create a production external calibration path."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "validation": validation_path,
        "summary": summary_path,
        "manifest": manifest_path,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--tail-quantile", type=float, default=0.9)
    parser.add_argument("--n-replicate-folds", type=int, default=5)
    parser.add_argument("--min-train-rows-per-predictor", type=int, default=20)
    parser.add_argument("--min-test-rows", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    outputs = run_q5_selected_tail_law_validation(
        records_path=args.records,
        output_dir=args.output_dir,
        alpha=args.alpha,
        tail_quantile=args.tail_quantile,
        n_replicate_folds=args.n_replicate_folds,
        min_train_rows_per_predictor=args.min_train_rows_per_predictor,
        min_test_rows=args.min_test_rows,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "STUDY_ROLE",
    "evaluate_q5_selected_tail_law",
    "prepare_q5_selected_tail_records",
    "q5_model_specs",
    "run_q5_selected_tail_law_validation",
    "summarize_q5_selected_tail_law",
]
