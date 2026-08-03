"""Selected quadratic-form law diagnostics for sibling null calibration.

This module is diagnostic-only. It checks whether selected sibling statistics
behave like plain chi-square, scalar-inflated chi-square, Satterthwaite-style
scaled chi-square, or spectrum-conditioned scaled chi-square. It does not add a
production calibration path.

The current sibling-record export stores spectral summaries, not the complete
per-parent eigenvalue vectors. Therefore this diagnostic can evaluate
Satterthwaite-style approximations and scale/df predictors, but it cannot yet
evaluate the exact weighted generalized chi-square law for each parent.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from benchmarks.diagnostics.calibration.reporting import diagnostic_json_default
from benchmarks.shared.util.time import format_timestamp_utc

SCHEMA_VERSION: Final = "selected_quadratic_law_audit/v1"
GENERATED_BY: Final = (
    "benchmarks.diagnostics.calibration.selected.hierarchy.selected_quadratic_law_audit"
)
DEFAULT_RECORDS_CSV: Final = Path(
    "reports/audits/generated/nnls_null_calibration_with_spectral_rank_gap_det/"
    "nnls_null_calibration_records.csv"
)
DEFAULT_PARENT_EIGENVALUES_CSV: Final = Path(
    "reports/audits/generated/nnls_null_calibration_with_spectral_rank_gap_det/"
    "nnls_null_calibration_parent_eigenvalues.csv"
)
DEFAULT_CELLS_CSV: Final = Path(
    "reports/audits/generated/nnls_null_calibration_with_spectral_rank_gap_det/"
    "nnls_null_calibration_cells.csv"
)
DEFAULT_OUTPUT_DIR: Final = Path("reports/selected_quadratic_law_audit")
GROUP_COLUMNS: Final = ("source_case_id", "branch_source", "spectral_context")
REQUIRED_COLUMNS: Final = (
    "source_case_id",
    "branch_source",
    "spectral_context",
    "parent",
    "stat",
    "degrees_of_freedom",
    "reference_scale",
    "branch_length_sum",
    "n_parent",
    "sibling_null_weight",
    "parent_spectral_log_pseudodeterminant",
    "parent_spectral_geometric_mean",
    "parent_effective_rank",
    "parent_eigenvalue_sum",
    "parent_top_spectral_gap",
)
MIN_GROUP_RECORDS: Final = 5
EIGENVALUE_COLUMNS: Final = (
    "source_case_id",
    "branch_source",
    "spectral_context",
    "parent",
    "eigenvalue_index",
    "eigenvalue",
)
FRAME_DIMENSION_COLUMNS: Final = (
    "source_case_id",
    "branch_source",
    "spectral_context",
    "n_samples",
    "n_features",
)
OPTIONAL_FRAME_DIMENSION_COLUMNS: Final = (
    "tree_n_leaves",
    "tree_n_internal_nodes",
    "tree_n_nodes",
    "spectral_total_descendant_leaf_rows",
    "spectral_total_internal_distribution_rows",
    "spectral_total_matrix_rows",
    "spectral_max_internal_distribution_rows",
    "root_descendant_leaf_rows",
    "root_internal_distribution_rows",
    "root_spectral_matrix_rows",
)
OPTIONAL_PARENT_SPECTRAL_FRAME_COLUMNS: Final = (
    "parent_descendant_leaf_rows",
    "parent_internal_distribution_rows",
    "parent_spectral_matrix_rows",
    "parent_active_feature_count",
    "parent_mp_threshold_rows",
)
FACTOR_SKEW_COLUMNS: Final = (
    "sqrt_log_pseudodeterminant",
    "log1p_log_pseudodeterminant",
    "parent_spectral_log_pseudodeterminant",
    "log_geometric_mean",
    "parent_effective_rank",
    "log_eigenvalue_sum",
    "parent_top_spectral_gap",
    "log_branch_length_sum",
    "log_n_parent",
    "sibling_null_weight",
    "log_n_samples",
    "log_n_features",
    "feature_sample_ratio",
    "sample_feature_ratio",
    "log_feature_sample_ratio",
    "parent_sample_fraction",
    "spectrum_positive_rank",
    "spectrum_satterthwaite_df",
    "spectrum_satterthwaite_scale",
    "log_tree_n_leaves",
    "log_tree_n_internal_nodes",
    "log_tree_n_nodes",
    "spectral_internal_row_fraction",
    "root_internal_row_fraction",
    "log_parent_descendant_leaf_rows",
    "parent_internal_row_fraction",
    "parent_spectral_matrix_to_leaf_ratio",
    "log_parent_active_feature_count",
)


@dataclass(frozen=True)
class ScaledChiSquareFit:
    """Parameters and probability-integral-transform checks for scale*chi2(df)."""

    n: int
    df: float
    scale: float
    negative_log_likelihood: float
    uniformity_ks_statistic: float
    uniformity_ks_p_value: float
    rejection_rate_at_0_05: float


def _require_columns(records: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_COLUMNS).difference(records.columns))
    if missing:
        raise ValueError(f"Selected quadratic law audit missing columns: {missing!r}.")


def _numeric(records: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(records[column], errors="coerce")


def prepare_selected_quadratic_records(records: pd.DataFrame) -> pd.DataFrame:
    """Return finite positive sibling records with reusable spectral transforms."""
    _require_columns(records)
    frame = records.copy()
    numeric_columns = [
        "stat",
        "degrees_of_freedom",
        "reference_scale",
        "branch_length_sum",
        "n_parent",
        "sibling_null_weight",
        "parent_spectral_log_pseudodeterminant",
        "parent_spectral_geometric_mean",
        "parent_effective_rank",
        "parent_eigenvalue_sum",
        "parent_top_spectral_gap",
    ]
    for column in numeric_columns:
        frame[column] = _numeric(frame, column)

    finite = np.ones(len(frame), dtype=bool)
    for column in numeric_columns:
        finite &= np.isfinite(frame[column].to_numpy(dtype=float))
    finite &= frame["stat"].to_numpy(dtype=float) > 0.0
    finite &= frame["degrees_of_freedom"].to_numpy(dtype=float) > 0.0
    finite &= frame["reference_scale"].to_numpy(dtype=float) > 0.0
    frame = frame.loc[finite].copy()

    frame["stat_over_nominal_df"] = frame["stat"] / frame["degrees_of_freedom"]
    frame["log_stat_over_nominal_df"] = np.log(frame["stat_over_nominal_df"])
    frame["naive_chi2_p_value"] = stats.chi2.sf(
        frame["stat"] / frame["reference_scale"],
        df=frame["degrees_of_freedom"],
    )
    frame["negative_log10_naive_chi2_p_value"] = -np.log10(
        np.clip(frame["naive_chi2_p_value"], 1e-300, 1.0)
    )
    frame["sqrt_log_pseudodeterminant"] = np.sqrt(
        np.clip(frame["parent_spectral_log_pseudodeterminant"], 0.0, None)
    )
    frame["log1p_log_pseudodeterminant"] = np.log1p(
        np.clip(frame["parent_spectral_log_pseudodeterminant"], 0.0, None)
    )
    frame["log_geometric_mean"] = np.log(
        np.clip(frame["parent_spectral_geometric_mean"], 1e-12, None)
    )
    frame["log_eigenvalue_sum"] = np.log(np.clip(frame["parent_eigenvalue_sum"], 1e-12, None))
    frame["log_branch_length_sum"] = np.log(np.clip(frame["branch_length_sum"], 1e-12, None))
    frame["log_n_parent"] = np.log(np.clip(frame["n_parent"], 1.0, None))
    return frame


def prepare_frame_dimensions(cells: pd.DataFrame) -> pd.DataFrame:
    """Return one global frame-dimension row per selected-law group."""
    missing = sorted(set(FRAME_DIMENSION_COLUMNS).difference(cells.columns))
    if missing:
        raise ValueError(f"Selected quadratic law frame dimensions missing columns: {missing!r}.")
    available_optional = [
        column for column in OPTIONAL_FRAME_DIMENSION_COLUMNS if column in cells.columns
    ]
    frame = cells[
        list(dict.fromkeys([*FRAME_DIMENSION_COLUMNS, *available_optional, "true_clusters"]))
    ].copy()
    for column in ("n_samples", "n_features", "true_clusters", *available_optional):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.drop_duplicates(subset=list(GROUP_COLUMNS))
    frame = frame[np.isfinite(frame["n_samples"]) & np.isfinite(frame["n_features"])].copy()
    frame = frame[(frame["n_samples"] > 0.0) & (frame["n_features"] > 0.0)].copy()
    frame["log_n_samples"] = np.log(frame["n_samples"])
    frame["log_n_features"] = np.log(frame["n_features"])
    frame["feature_sample_ratio"] = frame["n_features"] / frame["n_samples"]
    frame["sample_feature_ratio"] = frame["n_samples"] / frame["n_features"]
    frame["log_feature_sample_ratio"] = np.log(frame["feature_sample_ratio"])
    frame["log_sample_feature_ratio"] = np.log(frame["sample_feature_ratio"])
    if "true_clusters" in frame.columns:
        frame["log_true_clusters"] = np.log(np.clip(frame["true_clusters"], 1.0, None))
    for column in ("tree_n_leaves", "tree_n_internal_nodes", "tree_n_nodes"):
        if column in frame.columns:
            frame[f"log_{column}"] = np.log(np.clip(frame[column], 1.0, None))
    if {
        "spectral_total_internal_distribution_rows",
        "spectral_total_matrix_rows",
    }.issubset(frame.columns):
        frame["spectral_internal_row_fraction"] = (
            frame["spectral_total_internal_distribution_rows"]
            / np.clip(frame["spectral_total_matrix_rows"], 1.0, None)
        )
    if {"root_internal_distribution_rows", "root_spectral_matrix_rows"}.issubset(
        frame.columns
    ):
        frame["root_internal_row_fraction"] = frame["root_internal_distribution_rows"] / np.clip(
            frame["root_spectral_matrix_rows"],
            1.0,
            None,
        )
    return frame


def prepare_parent_eigenvalue_moments(parent_eigenvalues: pd.DataFrame) -> pd.DataFrame:
    """Return per-parent weighted-chi-square moments from exported spectra."""
    missing = sorted(set(EIGENVALUE_COLUMNS).difference(parent_eigenvalues.columns))
    if missing:
        raise ValueError(f"Parent eigenvalue export missing columns: {missing!r}.")
    frame = parent_eigenvalues.copy()
    frame["eigenvalue"] = pd.to_numeric(frame["eigenvalue"], errors="coerce")
    available_frame_columns = [
        column for column in OPTIONAL_PARENT_SPECTRAL_FRAME_COLUMNS if column in frame.columns
    ]
    for column in available_frame_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[np.isfinite(frame["eigenvalue"]) & (frame["eigenvalue"] > 0.0)].copy()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                *GROUP_COLUMNS,
                "parent",
                "spectrum_positive_rank",
                "spectrum_eigenvalue_sum",
                "spectrum_eigenvalue_square_sum",
                "spectrum_satterthwaite_df",
                "spectrum_satterthwaite_scale",
                *available_frame_columns,
            ]
        )
    rows: list[dict[str, object]] = []
    for group_key, group in frame.groupby([*GROUP_COLUMNS, "parent"], dropna=False):
        case_id, branch_source, spectral_context, parent = (str(value) for value in group_key)
        values = np.sort(group["eigenvalue"].to_numpy(dtype=float))[::-1]
        eigen_sum = float(values.sum())
        eigen_square_sum = float(np.sum(np.square(values)))
        if eigen_sum > 0.0 and eigen_square_sum > 0.0:
            spectrum_df = eigen_sum * eigen_sum / eigen_square_sum
            spectrum_scale = eigen_square_sum / eigen_sum
        else:
            spectrum_df = math.nan
            spectrum_scale = math.nan
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "source_case_id": case_id,
                "branch_source": branch_source,
                "spectral_context": spectral_context,
                "parent": parent,
                "spectrum_positive_rank": int(values.size),
                "spectrum_eigenvalue_sum": eigen_sum,
                "spectrum_eigenvalue_square_sum": eigen_square_sum,
                "spectrum_satterthwaite_df": float(spectrum_df),
                "spectrum_satterthwaite_scale": float(spectrum_scale),
                **{
                    column: float(group[column].dropna().iloc[0])
                    if column in group.columns and not group[column].dropna().empty
                    else math.nan
                    for column in available_frame_columns
                },
            }
        )
    moments = pd.DataFrame.from_records(rows)
    if {
        "parent_internal_distribution_rows",
        "parent_spectral_matrix_rows",
    }.issubset(moments.columns):
        moments["parent_internal_row_fraction"] = (
            moments["parent_internal_distribution_rows"]
            / np.clip(moments["parent_spectral_matrix_rows"], 1.0, None)
        )
    if {
        "parent_spectral_matrix_rows",
        "parent_descendant_leaf_rows",
    }.issubset(moments.columns):
        moments["parent_spectral_matrix_to_leaf_ratio"] = (
            moments["parent_spectral_matrix_rows"]
            / np.clip(moments["parent_descendant_leaf_rows"], 1.0, None)
        )
    if "parent_descendant_leaf_rows" in moments.columns:
        moments["log_parent_descendant_leaf_rows"] = np.log(
            np.clip(moments["parent_descendant_leaf_rows"], 1.0, None)
        )
    if "parent_active_feature_count" in moments.columns:
        moments["log_parent_active_feature_count"] = np.log(
            np.clip(moments["parent_active_feature_count"], 1.0, None)
        )
    return moments


def _uniformity_from_p_values(p_values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(p_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return math.nan, math.nan, math.nan
    result = stats.kstest(values, "uniform")
    return (
        float(result.statistic),
        float(result.pvalue),
        float(np.mean(values <= 0.05)),
    )


def _safe_skew(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 3 or np.allclose(finite, finite[0]):
        return math.nan
    return float(stats.skew(finite, bias=False))


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.allclose(x, x[0]):
        return math.nan
    _intercept, slope = _ols_fit(x, y)
    return slope


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return math.nan
    result = stats.spearmanr(x, y)
    return float(result.statistic)


def _factor_skew_scope_rows(
    frame: pd.DataFrame,
    *,
    scope: str,
    scope_case_id: str,
    scope_branch_source: str,
    scope_spectral_context: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary_rows: list[dict[str, object]] = []
    bin_rows: list[dict[str, object]] = []
    targets = {
        "log_stat_over_nominal_df": "log_stat_over_nominal_df",
        "negative_log10_naive_chi2_p_value": "negative_log10_naive_chi2_p_value",
    }
    for factor in FACTOR_SKEW_COLUMNS:
        if factor not in frame.columns:
            continue
        factor_frame = frame[
            [
                factor,
                "log_stat_over_nominal_df",
                "negative_log10_naive_chi2_p_value",
                "naive_chi2_p_value",
            ]
        ].replace([np.inf, -np.inf], np.nan).dropna()
        if factor_frame.shape[0] < MIN_GROUP_RECORDS or factor_frame[factor].nunique() < 2:
            continue
        x = factor_frame[factor].to_numpy(dtype=float)
        q10 = float(np.quantile(x, 0.1))
        q90 = float(np.quantile(x, 0.9))
        low = factor_frame[factor_frame[factor] <= q10]
        high = factor_frame[factor_frame[factor] >= q90]
        for target_label, target_column in targets.items():
            y = factor_frame[target_column].to_numpy(dtype=float)
            summary_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "scope": scope,
                    "source_case_id": scope_case_id,
                    "branch_source": scope_branch_source,
                    "spectral_context": scope_spectral_context,
                    "factor": factor,
                    "target": target_label,
                    "n_records": int(factor_frame.shape[0]),
                    "factor_min": float(np.min(x)),
                    "factor_q10": q10,
                    "factor_median": float(np.median(x)),
                    "factor_q90": q90,
                    "factor_max": float(np.max(x)),
                    "factor_skewness": _safe_skew(x),
                    "target_skewness": _safe_skew(y),
                    "spearman_correlation": _spearman(x, y),
                    "linear_slope": _linear_slope(x, y),
                    "target_mean_low_decile": float(low[target_column].mean()),
                    "target_mean_high_decile": float(high[target_column].mean()),
                    "target_high_minus_low_decile": float(
                        high[target_column].mean() - low[target_column].mean()
                    ),
                    "naive_rejection_rate_low_decile": float(
                        np.mean(low["naive_chi2_p_value"] <= 0.05)
                    ),
                    "naive_rejection_rate_high_decile": float(
                        np.mean(high["naive_chi2_p_value"] <= 0.05)
                    ),
                }
            )

        try:
            binned = factor_frame.copy()
            binned["factor_bin"] = pd.qcut(
                binned[factor],
                q=min(4, int(binned[factor].nunique())),
                duplicates="drop",
            )
        except ValueError:
            continue
        for bin_index, (_interval, bin_frame) in enumerate(
            binned.groupby("factor_bin", observed=False)
        ):
            if bin_frame.empty:
                continue
            bin_x = bin_frame[factor].to_numpy(dtype=float)
            bin_p = bin_frame["naive_chi2_p_value"].to_numpy(dtype=float)
            bin_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "scope": scope,
                    "source_case_id": scope_case_id,
                    "branch_source": scope_branch_source,
                    "spectral_context": scope_spectral_context,
                    "factor": factor,
                    "factor_bin_index": int(bin_index),
                    "n_records": int(bin_frame.shape[0]),
                    "factor_min": float(np.min(bin_x)),
                    "factor_max": float(np.max(bin_x)),
                    "factor_mean": float(np.mean(bin_x)),
                    "mean_log_stat_over_nominal_df": float(
                        bin_frame["log_stat_over_nominal_df"].mean()
                    ),
                    "median_log_stat_over_nominal_df": float(
                        bin_frame["log_stat_over_nominal_df"].median()
                    ),
                    "mean_negative_log10_naive_chi2_p_value": float(
                        bin_frame["negative_log10_naive_chi2_p_value"].mean()
                    ),
                    "median_negative_log10_naive_chi2_p_value": float(
                        bin_frame["negative_log10_naive_chi2_p_value"].median()
                    ),
                    "mean_naive_chi2_p_value": float(np.mean(bin_p)),
                    "naive_rejection_rate_at_0_05": float(np.mean(bin_p <= 0.05)),
                }
            )
    return summary_rows, bin_rows


def build_factor_skew_tables(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build factor-conditioned skew summaries for selected sibling tests."""
    summary_rows: list[dict[str, object]] = []
    bin_rows: list[dict[str, object]] = []
    rows, bins = _factor_skew_scope_rows(
        records,
        scope="all_records",
        scope_case_id="",
        scope_branch_source="",
        scope_spectral_context="",
    )
    summary_rows.extend(rows)
    bin_rows.extend(bins)
    for group_key, group in records.groupby(list(GROUP_COLUMNS), dropna=False):
        case_id, branch_source, spectral_context = (str(value) for value in group_key)
        rows, bins = _factor_skew_scope_rows(
            group,
            scope="case_branch_spectral_context",
            scope_case_id=case_id,
            scope_branch_source=branch_source,
            scope_spectral_context=spectral_context,
        )
        summary_rows.extend(rows)
        bin_rows.extend(bins)
    return pd.DataFrame.from_records(summary_rows), pd.DataFrame.from_records(bin_rows)


def _tail_fit_from_parameters(
    statistics: np.ndarray,
    *,
    df: float | np.ndarray,
    scale: float,
) -> ScaledChiSquareFit:
    y = np.asarray(statistics, dtype=float)
    df_values = np.asarray(df, dtype=float)
    if df_values.ndim == 0:
        df_values = np.full_like(y, float(df_values), dtype=float)
    p_values = stats.chi2.sf(y / scale, df=df_values)
    ks_stat, ks_p, rejection_rate = _uniformity_from_p_values(p_values)
    nll = -float(np.sum(stats.chi2.logpdf(y / scale, df=df_values) - math.log(scale)))
    return ScaledChiSquareFit(
        n=int(y.size),
        df=float(df_values[0]) if np.allclose(df_values, df_values[0]) else math.nan,
        scale=float(scale),
        negative_log_likelihood=nll,
        uniformity_ks_statistic=ks_stat,
        uniformity_ks_p_value=ks_p,
        rejection_rate_at_0_05=rejection_rate,
    )


def _free_scaled_chi_square_mle(statistics: np.ndarray) -> ScaledChiSquareFit:
    y = np.asarray(statistics, dtype=float)
    y = y[np.isfinite(y) & (y > 0.0)]
    if y.size < MIN_GROUP_RECORDS:
        return ScaledChiSquareFit(
            n=int(y.size),
            df=math.nan,
            scale=math.nan,
            negative_log_likelihood=math.nan,
            uniformity_ks_statistic=math.nan,
            uniformity_ks_p_value=math.nan,
            rejection_rate_at_0_05=math.nan,
        )

    def objective(theta: np.ndarray) -> float:
        log_df, log_scale = theta
        df = math.exp(float(log_df))
        scale = math.exp(float(log_scale))
        return -float(np.sum(stats.chi2.logpdf(y / scale, df=df) - math.log(scale)))

    mean = float(np.mean(y))
    variance = float(np.var(y, ddof=1)) if y.size > 1 else 0.0
    moment_df = max(2.0 * mean * mean / variance, 0.05) if variance > 0.0 else 1.0
    moment_scale = max(variance / (2.0 * mean), 1e-9) if variance > 0.0 else max(mean, 1e-9)
    starts = [
        np.array([math.log(1.0), math.log(max(mean, 1e-9))]),
        np.array([math.log(2.0), math.log(max(mean / 2.0, 1e-9))]),
        np.array([math.log(moment_df), math.log(moment_scale)]),
    ]
    best = None
    for start in starts:
        result = minimize(
            objective,
            x0=start,
            method="L-BFGS-B",
            bounds=((math.log(0.025), math.log(200.0)), (math.log(1e-9), math.log(1e9))),
        )
        if best is None or result.fun < best.fun:
            best = result
    if best is None or not best.success:
        return _moment_scaled_chi_square_fit(y)
    df = math.exp(float(best.x[0]))
    scale = math.exp(float(best.x[1]))
    return _tail_fit_from_parameters(y, df=df, scale=scale)


def _moment_scaled_chi_square_fit(statistics: np.ndarray) -> ScaledChiSquareFit:
    """Satterthwaite-style moment match for y ~= scale * chi2(df)."""
    y = np.asarray(statistics, dtype=float)
    y = y[np.isfinite(y) & (y > 0.0)]
    if y.size < MIN_GROUP_RECORDS:
        return ScaledChiSquareFit(
            n=int(y.size),
            df=math.nan,
            scale=math.nan,
            negative_log_likelihood=math.nan,
            uniformity_ks_statistic=math.nan,
            uniformity_ks_p_value=math.nan,
            rejection_rate_at_0_05=math.nan,
        )
    mean = float(np.mean(y))
    variance = float(np.var(y, ddof=1))
    if mean <= 0.0 or variance <= 0.0:
        return ScaledChiSquareFit(
            n=int(y.size),
            df=math.nan,
            scale=math.nan,
            negative_log_likelihood=math.nan,
            uniformity_ks_statistic=math.nan,
            uniformity_ks_p_value=math.nan,
            rejection_rate_at_0_05=math.nan,
        )
    df = 2.0 * mean * mean / variance
    scale = variance / (2.0 * mean)
    return _tail_fit_from_parameters(y, df=df, scale=scale)


def _ols_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([np.ones(len(x)), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(intercept), float(slope)


def _predict_linear(x: np.ndarray, *, intercept: float, slope: float) -> np.ndarray:
    return intercept + slope * np.asarray(x, dtype=float)


def _loo_rmse(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) <= 2:
        return math.nan
    errors: list[float] = []
    for index in range(len(x)):
        mask = np.ones(len(x), dtype=bool)
        mask[index] = False
        intercept, slope = _ols_fit(x[mask], y[mask])
        prediction = intercept + slope * x[index]
        errors.append(float(prediction - y[index]))
    return float(np.sqrt(np.mean(np.square(errors))))


def _single_predictor_rows(
    group_fits: pd.DataFrame,
    *,
    target_column: str,
    target_label: str,
) -> list[dict[str, object]]:
    candidates = (
        "median_sqrt_log_pseudodeterminant",
        "median_log1p_log_pseudodeterminant",
        "median_parent_spectral_log_pseudodeterminant",
        "median_log_geometric_mean",
        "median_parent_effective_rank",
        "median_log_eigenvalue_sum",
        "median_parent_top_spectral_gap",
        "median_log_branch_length_sum",
        "median_log_n_parent",
        "mean_sibling_null_weight",
        "median_log_n_samples",
        "median_log_n_features",
        "median_feature_sample_ratio",
        "median_sample_feature_ratio",
        "median_log_feature_sample_ratio",
        "median_parent_sample_fraction",
        "mean_spectrum_positive_rank",
        "fraction_spectrum_rank_1",
        "fraction_spectrum_rank_2",
        "fraction_spectrum_rank_ge_3",
    )
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        if candidate not in group_fits.columns:
            continue
        frame = group_fits[[candidate, target_column]].replace([np.inf, -np.inf], np.nan).dropna()
        if frame.shape[0] < 3:
            continue
        x = frame[candidate].to_numpy(dtype=float)
        y = frame[target_column].to_numpy(dtype=float)
        if np.allclose(x, x[0]):
            continue
        intercept, slope = _ols_fit(x, y)
        prediction = _predict_linear(x, intercept=intercept, slope=slope)
        ss_res = float(np.sum(np.square(y - prediction)))
        ss_tot = float(np.sum(np.square(y - np.mean(y))))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else math.nan
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "target": target_label,
                "predictor": candidate,
                "n_groups": int(frame.shape[0]),
                "intercept": intercept,
                "slope": slope,
                "r2_in_sample": r2,
                "rmse_in_sample": float(np.sqrt(np.mean(np.square(y - prediction)))),
                "loo_rmse": _loo_rmse(x, y),
            }
        )
    return rows


def _merge_parent_spectrum_moments(
    records: pd.DataFrame,
    parent_eigenvalues: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if parent_eigenvalues is None:
        return records, pd.DataFrame()
    moments = prepare_parent_eigenvalue_moments(parent_eigenvalues)
    if moments.empty:
        return records, moments
    merged = records.merge(
        moments.drop(columns=["schema_version"], errors="ignore"),
        on=[*GROUP_COLUMNS, "parent"],
        how="left",
    )
    return merged, moments


def _merge_frame_dimensions(
    records: pd.DataFrame,
    cells: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cells is None:
        return records, pd.DataFrame()
    dimensions = prepare_frame_dimensions(cells)
    if dimensions.empty:
        return records, dimensions
    merged = records.merge(
        dimensions,
        on=list(GROUP_COLUMNS),
        how="left",
    )
    merged["parent_sample_fraction"] = merged["n_parent"] / merged["n_samples"]
    return merged, dimensions


def build_selected_quadratic_law_audit_tables(
    records: pd.DataFrame,
    *,
    parent_eigenvalues: pd.DataFrame | None = None,
    cells: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Build selected quadratic-law diagnostic tables from sibling records."""
    prepared = prepare_selected_quadratic_records(records)
    prepared, parent_spectrum_moments = _merge_parent_spectrum_moments(
        prepared,
        parent_eigenvalues,
    )
    prepared, frame_dimensions = _merge_frame_dimensions(prepared, cells)
    group_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    eigen_satterthwaite_rows: list[dict[str, object]] = []

    for group_key, group in prepared.groupby(list(GROUP_COLUMNS), dropna=False):
        case_id, branch_source, spectral_context = (str(value) for value in group_key)
        statistics = group["stat"].to_numpy(dtype=float)
        nominal_df = group["degrees_of_freedom"].to_numpy(dtype=float)
        reference_scale = group["reference_scale"].to_numpy(dtype=float)
        naive_p_values = stats.chi2.sf(statistics / reference_scale, df=nominal_df)
        naive_ks, naive_ks_p, naive_rejection = _uniformity_from_p_values(naive_p_values)

        scalar_scale = float(np.sum(statistics) / np.sum(reference_scale * nominal_df))
        scalar_fit = _tail_fit_from_parameters(
            statistics / reference_scale,
            df=nominal_df,
            scale=scalar_scale,
        )
        moment_fit = _moment_scaled_chi_square_fit(statistics)
        mle_fit = _free_scaled_chi_square_mle(statistics)
        spectrum_rank_values = (
            pd.to_numeric(group["spectrum_positive_rank"], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .to_numpy(dtype=float)
            if "spectrum_positive_rank" in group.columns
            else np.asarray([], dtype=float)
        )
        eigen_common_scale = math.nan
        eigen_ks = math.nan
        eigen_ks_p = math.nan
        eigen_rejection = math.nan
        eigen_nll = math.nan
        if {
            "spectrum_satterthwaite_df",
            "spectrum_satterthwaite_scale",
            "spectrum_eigenvalue_sum",
        }.issubset(group.columns):
            spectrum_df = group["spectrum_satterthwaite_df"].to_numpy(dtype=float)
            spectrum_scale = group["spectrum_satterthwaite_scale"].to_numpy(dtype=float)
            spectrum_mean = group["spectrum_eigenvalue_sum"].to_numpy(dtype=float)
            spectrum_mask = (
                np.isfinite(spectrum_df)
                & np.isfinite(spectrum_scale)
                & np.isfinite(spectrum_mean)
                & (spectrum_df > 0.0)
                & (spectrum_scale > 0.0)
                & (spectrum_mean > 0.0)
            )
            if int(spectrum_mask.sum()) >= MIN_GROUP_RECORDS:
                eigen_common_scale = float(
                    np.sum(statistics[spectrum_mask]) / np.sum(spectrum_mean[spectrum_mask])
                )
                p_values = stats.chi2.sf(
                    statistics[spectrum_mask]
                    / (eigen_common_scale * spectrum_scale[spectrum_mask]),
                    df=spectrum_df[spectrum_mask],
                )
                eigen_ks, eigen_ks_p, eigen_rejection = _uniformity_from_p_values(p_values)
                eigen_nll = -float(
                    np.sum(
                        stats.chi2.logpdf(
                            statistics[spectrum_mask]
                            / (eigen_common_scale * spectrum_scale[spectrum_mask]),
                            df=spectrum_df[spectrum_mask],
                        )
                        - np.log(eigen_common_scale * spectrum_scale[spectrum_mask])
                    )
                )
                eigen_satterthwaite_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "source_case_id": case_id,
                        "branch_source": branch_source,
                        "spectral_context": spectral_context,
                        "model": "parent_eigenvector_satterthwaite_common_scale",
                        "n_records": int(spectrum_mask.sum()),
                        "common_scale": eigen_common_scale,
                        "median_spectrum_df": float(np.median(spectrum_df[spectrum_mask])),
                        "median_spectrum_scale": float(np.median(spectrum_scale[spectrum_mask])),
                        "uniformity_ks_statistic": eigen_ks,
                        "uniformity_ks_p_value": eigen_ks_p,
                        "rejection_rate_at_0_05": eigen_rejection,
                        "negative_log_likelihood": eigen_nll,
                    }
                )
        row = {
            "schema_version": SCHEMA_VERSION,
            "source_case_id": case_id,
            "branch_source": branch_source,
            "spectral_context": spectral_context,
            "n_records": int(len(group)),
            "mean_stat": float(np.mean(statistics)),
            "median_stat": float(np.median(statistics)),
            "mean_nominal_df": float(np.mean(nominal_df)),
            "median_nominal_df": float(np.median(nominal_df)),
            "naive_chi2_ks_statistic": naive_ks,
            "naive_chi2_ks_p_value": naive_ks_p,
            "naive_chi2_rejection_rate_at_0_05": naive_rejection,
            "scalar_nominal_df_scale": scalar_scale,
            "scalar_nominal_df_ks_statistic": scalar_fit.uniformity_ks_statistic,
            "scalar_nominal_df_ks_p_value": scalar_fit.uniformity_ks_p_value,
            "scalar_nominal_df_rejection_rate_at_0_05": scalar_fit.rejection_rate_at_0_05,
            "satterthwaite_df": moment_fit.df,
            "satterthwaite_scale": moment_fit.scale,
            "satterthwaite_ks_statistic": moment_fit.uniformity_ks_statistic,
            "satterthwaite_ks_p_value": moment_fit.uniformity_ks_p_value,
            "satterthwaite_rejection_rate_at_0_05": moment_fit.rejection_rate_at_0_05,
            "mle_df": mle_fit.df,
            "mle_scale": mle_fit.scale,
            "mle_ks_statistic": mle_fit.uniformity_ks_statistic,
            "mle_ks_p_value": mle_fit.uniformity_ks_p_value,
            "mle_rejection_rate_at_0_05": mle_fit.rejection_rate_at_0_05,
            "parent_eigenvector_satterthwaite_common_scale": eigen_common_scale,
            "parent_eigenvector_satterthwaite_ks_statistic": eigen_ks,
            "parent_eigenvector_satterthwaite_ks_p_value": eigen_ks_p,
            "parent_eigenvector_satterthwaite_rejection_rate_at_0_05": eigen_rejection,
            "log_mle_df": math.log(mle_fit.df) if np.isfinite(mle_fit.df) and mle_fit.df > 0 else math.nan,
            "log_mle_scale": math.log(mle_fit.scale)
            if np.isfinite(mle_fit.scale) and mle_fit.scale > 0
            else math.nan,
            "median_sqrt_log_pseudodeterminant": float(
                np.median(group["sqrt_log_pseudodeterminant"])
            ),
            "median_log1p_log_pseudodeterminant": float(
                np.median(group["log1p_log_pseudodeterminant"])
            ),
            "median_parent_spectral_log_pseudodeterminant": float(
                np.median(group["parent_spectral_log_pseudodeterminant"])
            ),
            "median_log_geometric_mean": float(np.median(group["log_geometric_mean"])),
            "median_parent_effective_rank": float(np.median(group["parent_effective_rank"])),
            "median_log_eigenvalue_sum": float(np.median(group["log_eigenvalue_sum"])),
            "median_parent_top_spectral_gap": float(
                np.median(group["parent_top_spectral_gap"])
            ),
            "median_log_branch_length_sum": float(np.median(group["log_branch_length_sum"])),
            "median_log_n_parent": float(np.median(group["log_n_parent"])),
            "mean_sibling_null_weight": float(np.mean(group["sibling_null_weight"])),
        }
        if spectrum_rank_values.size:
            row.update(
                {
                    "min_spectrum_positive_rank": float(np.min(spectrum_rank_values)),
                    "q25_spectrum_positive_rank": float(
                        np.quantile(spectrum_rank_values, 0.25)
                    ),
                    "median_spectrum_positive_rank": float(np.median(spectrum_rank_values)),
                    "q75_spectrum_positive_rank": float(
                        np.quantile(spectrum_rank_values, 0.75)
                    ),
                    "max_spectrum_positive_rank": float(np.max(spectrum_rank_values)),
                    "mean_spectrum_positive_rank": float(np.mean(spectrum_rank_values)),
                    "fraction_spectrum_rank_1": float(np.mean(spectrum_rank_values == 1.0)),
                    "fraction_spectrum_rank_2": float(np.mean(spectrum_rank_values == 2.0)),
                    "fraction_spectrum_rank_ge_3": float(np.mean(spectrum_rank_values >= 3.0)),
                }
            )
        if "n_samples" in group.columns:
            row.update(
                {
                    "median_n_samples": float(np.median(group["n_samples"])),
                    "median_n_features": float(np.median(group["n_features"])),
                    "median_log_n_samples": float(np.median(group["log_n_samples"])),
                    "median_log_n_features": float(np.median(group["log_n_features"])),
                    "median_feature_sample_ratio": float(
                        np.median(group["feature_sample_ratio"])
                    ),
                    "median_sample_feature_ratio": float(
                        np.median(group["sample_feature_ratio"])
                    ),
                    "median_log_feature_sample_ratio": float(
                        np.median(group["log_feature_sample_ratio"])
                    ),
                    "median_parent_sample_fraction": float(
                        np.median(group["parent_sample_fraction"])
                    ),
                }
            )
        for column in (
            "tree_n_leaves",
            "tree_n_internal_nodes",
            "tree_n_nodes",
            "spectral_total_descendant_leaf_rows",
            "spectral_total_internal_distribution_rows",
            "spectral_total_matrix_rows",
            "spectral_max_internal_distribution_rows",
            "root_descendant_leaf_rows",
            "root_internal_distribution_rows",
            "root_spectral_matrix_rows",
            "spectral_internal_row_fraction",
            "root_internal_row_fraction",
        ):
            if column in group.columns:
                values = pd.to_numeric(group[column], errors="coerce").dropna()
                if not values.empty:
                    row[f"median_{column}"] = float(np.median(values.to_numpy(dtype=float)))
        for column in (
            "parent_descendant_leaf_rows",
            "parent_internal_distribution_rows",
            "parent_spectral_matrix_rows",
            "parent_active_feature_count",
            "parent_mp_threshold_rows",
            "parent_internal_row_fraction",
            "parent_spectral_matrix_to_leaf_ratio",
        ):
            if column in group.columns:
                values = pd.to_numeric(group[column], errors="coerce").dropna()
                if not values.empty:
                    row[f"median_{column}"] = float(np.median(values.to_numpy(dtype=float)))
        group_rows.append(row)
        for model_name, fit in (
            (
                "naive_nominal_chi2",
                ScaledChiSquareFit(
                    n=int(len(group)),
                    df=math.nan,
                    scale=1.0,
                    negative_log_likelihood=math.nan,
                    uniformity_ks_statistic=naive_ks,
                    uniformity_ks_p_value=naive_ks_p,
                    rejection_rate_at_0_05=naive_rejection,
                ),
            ),
            ("scalar_nominal_df", scalar_fit),
            ("satterthwaite_moment_scaled_chi2", moment_fit),
            ("mle_free_df_scaled_chi2", mle_fit),
        ):
            calibration_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "source_case_id": case_id,
                    "branch_source": branch_source,
                    "spectral_context": spectral_context,
                    "model": model_name,
                    "n_records": int(len(group)),
                    "df": fit.df,
                    "scale": fit.scale,
                    "uniformity_ks_statistic": fit.uniformity_ks_statistic,
                    "uniformity_ks_p_value": fit.uniformity_ks_p_value,
                    "rejection_rate_at_0_05": fit.rejection_rate_at_0_05,
                    "negative_log_likelihood": fit.negative_log_likelihood,
                }
            )

    group_fits = pd.DataFrame.from_records(group_rows)
    factor_skew_summary, factor_skew_bins = build_factor_skew_tables(prepared)
    predictor_rows = [
        *_single_predictor_rows(group_fits, target_column="mle_df", target_label="mle_df"),
        *_single_predictor_rows(
            group_fits, target_column="log_mle_scale", target_label="log_mle_scale"
        ),
    ]
    predictor_fits = pd.DataFrame.from_records(predictor_rows)
    predicted = _build_predicted_spectrum_model_rows(prepared, group_fits, predictor_fits)
    generalized_status = (
        {
            "schema_version": SCHEMA_VERSION,
            "status": "available_parent_eigenvalue_vectors_exported",
            "available_evidence": (
                "long-form positive parent eigenvalues are available and were reduced "
                "to per-record Satterthwaite moments"
            ),
            "required_evidence": (
                "exact quadratic-form weights in the selected projected-Wald coordinate "
                "system before promoting a weighted generalized chi-square law"
            ),
            "diagnostic_consequence": (
                "parent-spectrum Satterthwaite can now be compared; exact weighted "
                "generalized chi-square remains a later stronger diagnostic"
            ),
        }
        if parent_eigenvalues is not None and not parent_spectrum_moments.empty
        else {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked_missing_full_parent_eigenvalue_vectors",
            "available_evidence": (
                "sibling records contain spectral summaries such as rank, "
                "effective rank, gap, pseudodeterminant, and geometric mean"
            ),
            "required_evidence": (
                "per-record positive parent eigenvalue vectors or equivalent "
                "quadratic-form weights"
            ),
            "diagnostic_consequence": (
                "exact or weighted generalized chi-square calibration cannot be "
                "evaluated from the current records CSV"
            ),
        }
    )
    return {
        "prepared_records": prepared,
        "frame_dimensions": frame_dimensions,
        "parent_spectrum_moments": parent_spectrum_moments,
        "group_fits": group_fits,
        "factor_skew_summary": factor_skew_summary,
        "factor_skew_bins": factor_skew_bins,
        "predictor_fits": predictor_fits,
        "calibration_comparison": pd.DataFrame.from_records(calibration_rows),
        "eigen_satterthwaite_comparison": pd.DataFrame.from_records(eigen_satterthwaite_rows),
        "predicted_spectrum_model": predicted,
        "generalized_chi_square_status": pd.DataFrame.from_records([generalized_status]),
    }


def _best_predictor(
    predictor_fits: pd.DataFrame,
    *,
    target: str,
) -> pd.Series | None:
    if predictor_fits.empty or "target" not in predictor_fits.columns:
        return None
    subset = predictor_fits[predictor_fits["target"].eq(target)].copy()
    subset = subset.replace([np.inf, -np.inf], np.nan).dropna(subset=["loo_rmse"])
    if subset.empty:
        return None
    return subset.sort_values(["loo_rmse", "rmse_in_sample", "predictor"]).iloc[0]


def _build_predicted_spectrum_model_rows(
    records: pd.DataFrame,
    group_fits: pd.DataFrame,
    predictor_fits: pd.DataFrame,
) -> pd.DataFrame:
    best_df = _best_predictor(predictor_fits, target="mle_df")
    best_scale = _best_predictor(predictor_fits, target="log_mle_scale")
    if best_df is None or best_scale is None:
        return pd.DataFrame()

    df_predictor = str(best_df["predictor"])
    scale_predictor = str(best_scale["predictor"])
    parameter_columns = list(dict.fromkeys([*GROUP_COLUMNS, df_predictor, scale_predictor]))
    group_parameters = group_fits[parameter_columns].copy()
    group_parameters["predicted_df"] = _predict_linear(
        group_parameters[df_predictor].to_numpy(dtype=float),
        intercept=float(best_df["intercept"]),
        slope=float(best_df["slope"]),
    )
    group_parameters["predicted_scale"] = np.exp(
        _predict_linear(
            group_parameters[scale_predictor].to_numpy(dtype=float),
            intercept=float(best_scale["intercept"]),
            slope=float(best_scale["slope"]),
        )
    )

    rows: list[dict[str, object]] = []
    for _, parameter_row in group_parameters.iterrows():
        mask = np.ones(len(records), dtype=bool)
        for column in GROUP_COLUMNS:
            mask &= records[column].astype(str).to_numpy() == str(parameter_row[column])
        group = records.loc[mask]
        df = max(float(parameter_row["predicted_df"]), 0.025)
        scale = max(float(parameter_row["predicted_scale"]), 1e-9)
        fit = _tail_fit_from_parameters(group["stat"].to_numpy(dtype=float), df=df, scale=scale)
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "source_case_id": str(parameter_row["source_case_id"]),
                "branch_source": str(parameter_row["branch_source"]),
                "spectral_context": str(parameter_row["spectral_context"]),
                "model": "predicted_spectrum_df_and_scale",
                "df_predictor": df_predictor,
                "scale_predictor": scale_predictor,
                "predicted_df": df,
                "predicted_scale": scale,
                "n_records": int(len(group)),
                "uniformity_ks_statistic": fit.uniformity_ks_statistic,
                "uniformity_ks_p_value": fit.uniformity_ks_p_value,
                "rejection_rate_at_0_05": fit.rejection_rate_at_0_05,
                "negative_log_likelihood": fit.negative_log_likelihood,
            }
        )
    return pd.DataFrame.from_records(rows)


def _write_plots(
    *,
    output_dir: Path,
    group_fits: pd.DataFrame,
    factor_skew_summary: pd.DataFrame,
    predictor_fits: pd.DataFrame,
    calibration_comparison: pd.DataFrame,
    eigen_satterthwaite_comparison: pd.DataFrame,
    predicted_spectrum_model: pd.DataFrame,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: dict[str, Path] = {}

    comparison = calibration_comparison[
        [
            "source_case_id",
            "branch_source",
            "spectral_context",
            "model",
            "uniformity_ks_statistic",
        ]
    ].copy()
    if not predicted_spectrum_model.empty:
        comparison = pd.concat(
            [
                comparison,
                predicted_spectrum_model[
                    [
                        "source_case_id",
                        "branch_source",
                        "spectral_context",
                        "model",
                        "uniformity_ks_statistic",
                    ]
                ],
            ],
            ignore_index=True,
        )
    if not eigen_satterthwaite_comparison.empty:
        comparison = pd.concat(
            [
                comparison,
                eigen_satterthwaite_comparison[
                    [
                        "source_case_id",
                        "branch_source",
                        "spectral_context",
                        "model",
                        "uniformity_ks_statistic",
                    ]
                ],
            ],
            ignore_index=True,
        )
    if not comparison.empty:
        pivot = comparison.pivot_table(
            index=["source_case_id", "branch_source", "spectral_context"],
            columns="model",
            values="uniformity_ks_statistic",
            aggfunc="first",
        )
        ax = pivot.plot(kind="barh", figsize=(12, max(5.0, 0.4 * len(pivot))))
        ax.set_xlabel("KS statistic against uniform p-values; lower is better")
        ax.set_title("Selected quadratic-law calibration comparison")
        ax.legend(fontsize=7, loc="best")
        plt.tight_layout()
        path = output_dir / "selected_quadratic_law_ks_comparison.png"
        plt.savefig(path, dpi=180)
        plot_paths["ks_comparison_png"] = path
        path_pdf = output_dir / "selected_quadratic_law_ks_comparison.pdf"
        plt.savefig(path_pdf)
        plot_paths["ks_comparison_pdf"] = path_pdf
        plt.close()

    if not factor_skew_summary.empty:
        overview = factor_skew_summary[
            factor_skew_summary["scope"].eq("all_records")
            & factor_skew_summary["target"].eq("log_stat_over_nominal_df")
        ].copy()
        overview = overview.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["spearman_correlation"]
        )
        if not overview.empty:
            overview["abs_spearman"] = overview["spearman_correlation"].abs()
            overview = overview.sort_values("abs_spearman", ascending=True).tail(14)
            plt.figure(figsize=(10, max(5.0, 0.35 * len(overview))))
            colors = [
                "#b2182b" if value < 0.0 else "#2166ac"
                for value in overview["spearman_correlation"]
            ]
            plt.barh(overview["factor"], overview["spearman_correlation"], color=colors)
            plt.axvline(0.0, color="black", linewidth=0.8)
            plt.xlabel("Spearman correlation with log(stat / nominal df)")
            plt.title("Factor-conditioned skew of the selected sibling test")
            plt.tight_layout()
            path = output_dir / "factor_skew_spearman_overview.png"
            plt.savefig(path, dpi=180)
            plot_paths["factor_skew_spearman_overview_png"] = path
            path_pdf = output_dir / "factor_skew_spearman_overview.pdf"
            plt.savefig(path_pdf)
            plot_paths["factor_skew_spearman_overview_pdf"] = path_pdf
            plt.close()

    for target, y_column, filename, ylabel in (
        ("mle_df", "mle_df", "mle_df_best_spectral_predictor", "free scaled-χ² df MLE"),
        (
            "log_mle_scale",
            "log_mle_scale",
            "log_scale_best_spectral_predictor",
            "log free scaled-χ² scale MLE",
        ),
    ):
        best = _best_predictor(predictor_fits, target=target)
        if best is None:
            continue
        predictor = str(best["predictor"])
        frame = group_fits[[predictor, y_column, "source_case_id", "branch_source"]].dropna()
        if frame.empty:
            continue
        x = frame[predictor].to_numpy(dtype=float)
        order = np.argsort(x)
        xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        ys = _predict_linear(
            xs,
            intercept=float(best["intercept"]),
            slope=float(best["slope"]),
        )
        plt.figure(figsize=(9, 5.5))
        for branch_source, branch_frame in frame.groupby("branch_source"):
            plt.scatter(
                branch_frame[predictor],
                branch_frame[y_column],
                label=str(branch_source),
                s=70,
            )
        plt.plot(xs, ys, color="black", linewidth=2)
        for _, row in frame.iloc[order].iterrows():
            plt.annotate(
                str(row["source_case_id"]).replace("binary_", "").replace("_4c", ""),
                (float(row[predictor]), float(row[y_column])),
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points",
            )
        plt.xlabel(predictor)
        plt.ylabel(ylabel)
        plt.title(
            f"{ylabel} vs {predictor}\n"
            f"R²={float(best['r2_in_sample']):.3f}, LOO RMSE={float(best['loo_rmse']):.3f}"
        )
        plt.legend(fontsize=7)
        plt.tight_layout()
        path = output_dir / f"{filename}.png"
        plt.savefig(path, dpi=180)
        plot_paths[f"{filename}_png"] = path
        path_pdf = output_dir / f"{filename}.pdf"
        plt.savefig(path_pdf)
        plot_paths[f"{filename}_pdf"] = path_pdf
        plt.close()

    return plot_paths


def run_selected_quadratic_law_audit(
    *,
    records_csv: Path = DEFAULT_RECORDS_CSV,
    parent_eigenvalues_csv: Path | None = None,
    cells_csv: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Run the selected quadratic-law diagnostic and write CSV/plot outputs."""
    records = pd.read_csv(records_csv)
    parent_eigenvalues = (
        pd.read_csv(parent_eigenvalues_csv)
        if parent_eigenvalues_csv is not None and parent_eigenvalues_csv.exists()
        else None
    )
    cells = pd.read_csv(cells_csv) if cells_csv is not None and cells_csv.exists() else None
    tables = build_selected_quadratic_law_audit_tables(
        records,
        parent_eigenvalues=parent_eigenvalues,
        cells=cells,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "prepared_records": output_dir / "selected_quadratic_prepared_records.csv",
        "frame_dimensions": output_dir / "selected_quadratic_frame_dimensions.csv",
        "parent_spectrum_moments": output_dir / "selected_quadratic_parent_spectrum_moments.csv",
        "group_fits": output_dir / "selected_quadratic_group_fits.csv",
        "factor_skew_summary": output_dir / "selected_quadratic_factor_skew_summary.csv",
        "factor_skew_bins": output_dir / "selected_quadratic_factor_skew_bins.csv",
        "predictor_fits": output_dir / "selected_quadratic_predictor_fits.csv",
        "calibration_comparison": output_dir / "selected_quadratic_calibration_comparison.csv",
        "eigen_satterthwaite_comparison": output_dir
        / "selected_quadratic_eigen_satterthwaite_comparison.csv",
        "predicted_spectrum_model": output_dir / "selected_quadratic_predicted_spectrum_model.csv",
        "generalized_chi_square_status": output_dir
        / "selected_quadratic_generalized_chi_square_status.csv",
    }
    for key, path in paths.items():
        tables[key].to_csv(path, index=False)

    plot_paths = _write_plots(
        output_dir=output_dir / "plots",
        group_fits=tables["group_fits"],
        factor_skew_summary=tables["factor_skew_summary"],
        predictor_fits=tables["predictor_fits"],
        calibration_comparison=tables["calibration_comparison"],
        eigen_satterthwaite_comparison=tables["eigen_satterthwaite_comparison"],
        predicted_spectrum_model=tables["predicted_spectrum_model"],
    )
    paths.update(plot_paths)

    best_df = _best_predictor(tables["predictor_fits"], target="mle_df")
    best_scale = _best_predictor(tables["predictor_fits"], target="log_mle_scale")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "records_csv": records_csv,
        "parent_eigenvalues_csv": parent_eigenvalues_csv,
        "cells_csv": cells_csv,
        "row_counts": {key: int(value.shape[0]) for key, value in tables.items()},
        "outputs": paths,
        "best_df_predictor": None if best_df is None else best_df.to_dict(),
        "best_log_scale_predictor": None if best_scale is None else best_scale.to_dict(),
        "generalized_chi_square_status": str(
            tables["generalized_chi_square_status"].loc[0, "status"]
        ),
        "diagnostic_only": True,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=diagnostic_json_default) + "\n",
        encoding="utf-8",
    )
    paths["manifest"] = manifest_path
    return paths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records-csv",
        type=Path,
        default=DEFAULT_RECORDS_CSV,
        help="Sibling-record CSV containing selected quadratic statistics.",
    )
    parser.add_argument(
        "--parent-eigenvalues-csv",
        type=Path,
        default=None,
        help="Optional long-form parent eigenvalue CSV exported by the NNLS calibration sweep.",
    )
    parser.add_argument(
        "--cells-csv",
        type=Path,
        default=None,
        help="Optional sweep cell CSV containing global frame dimensions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / format_timestamp_utc(),
        help="Directory for audit outputs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = run_selected_quadratic_law_audit(
        records_csv=args.records_csv,
        parent_eigenvalues_csv=args.parent_eigenvalues_csv,
        cells_csv=args.cells_csv,
        output_dir=args.output_dir,
    )
    print(f"Wrote selected quadratic-law audit to {paths['manifest'].parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
