"""Statistic distribution-shape diagnostic.

This module separates skew that is expected from chi-square degrees of freedom
from residual statistic-shape and tail misalignment. It is diagnostic-only and
does not install a calibration rule.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, kstest

from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "diagnostic_statistic_distribution_shape_panel_not_calibration"
SCHEMA_VERSION = "statistic_distribution_shape_panel/v1"
REQUIRED_COLUMNS = {
    "test_statistic",
    "degrees_of_freedom",
    "statistic_context_role",
}
DF_BINS = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, math.inf)
DF_BIN_LABELS = (
    "df_0_1",
    "df_1_2",
    "df_2_4",
    "df_4_8",
    "df_8_16",
    "df_16_32",
    "df_32_64",
    "df_ge64",
)
DEFAULT_GROUP_COLUMNS = ("test_family", "statistic_context_role", "df_bin")


def infer_satterthwaite_reference_from_eigenvalues(
    eigenvalues: Sequence[float],
) -> tuple[float, float]:
    """Infer ``scale, df`` for ``sum(lambda_i * chi2_1)`` by moment matching."""
    values = np.asarray(tuple(eigenvalues), dtype=float)
    if values.ndim != 1:
        raise ValueError(f"eigenvalues must be one-dimensional; got {values.shape}.")
    if values.size == 0:
        raise ValueError("eigenvalues must contain at least one value.")
    invalid = ~np.isfinite(values) | (values <= 0.0)
    if bool(invalid.any()):
        bad_index = int(np.flatnonzero(invalid)[0])
        raise ValueError(
            "eigenvalues must contain finite positive values; "
            f"index={bad_index}, value={float(values[bad_index])!r}."
        )
    first_moment = float(np.sum(values))
    second_moment = float(np.sum(values**2))
    scale = second_moment / first_moment
    degrees_of_freedom = first_moment**2 / second_moment
    return float(scale), float(degrees_of_freedom)


def _finite_nonnegative(values: pd.Series, *, column_name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise").astype(float)
    invalid = ~np.isfinite(numeric) | (numeric < 0.0)
    if bool(invalid.any()):
        bad_index = invalid[invalid].index[0]
        raise ValueError(
            f"{column_name} must contain finite non-negative values; "
            f"row={int(bad_index)}, value={float(numeric.loc[bad_index])!r}."
        )
    return numeric


def _finite_positive(values: pd.Series, *, column_name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise").astype(float)
    invalid = ~np.isfinite(numeric) | (numeric <= 0.0)
    if bool(invalid.any()):
        bad_index = invalid[invalid].index[0]
        raise ValueError(
            f"{column_name} must contain finite positive values; "
            f"row={int(bad_index)}, value={float(numeric.loc[bad_index])!r}."
        )
    return numeric


def _record_id(table: pd.DataFrame) -> pd.Series:
    if "record_id" in table.columns:
        return table["record_id"].astype(str)
    if {"parent_id", "child_id"} <= set(table.columns):
        return table["parent_id"].astype(str) + "->" + table["child_id"].astype(str)
    if "node_id" in table.columns:
        return table["node_id"].astype(str)
    return pd.Series([f"statistic_record_{index}" for index in table.index], index=table.index)


def _df_bin(value: float) -> str:
    for lower, upper, label in zip(DF_BINS, DF_BINS[1:], DF_BIN_LABELS):
        if lower < value <= upper:
            return label
    raise ValueError(f"degrees_of_freedom did not match a bin: {value!r}.")


def _sample_skew(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.shape[0] < 3:
        return math.nan
    return float(numeric.skew())


def _expected_chi_square_mixture_skew(degrees_of_freedom: pd.Series) -> float:
    """Return raw-statistic skew expected from the observed df mixture."""
    df_values = degrees_of_freedom.to_numpy(dtype=float)
    if df_values.size == 0:
        return math.nan
    mean = float(np.mean(df_values))
    variance_terms = 2.0 * df_values + np.square(df_values - mean)
    variance = float(np.mean(variance_terms))
    if variance <= 0.0:
        return math.nan
    third_terms = (
        8.0 * df_values
        + 3.0 * (df_values - mean) * (2.0 * df_values)
        + np.power(df_values - mean, 3.0)
    )
    third_central = float(np.mean(third_terms))
    return float(third_central / variance**1.5)


def normalize_statistic_distribution_records(records: pd.DataFrame) -> pd.DataFrame:
    """Add chi-square reference variables to statistic records."""
    missing = REQUIRED_COLUMNS - set(records.columns)
    if missing:
        raise ValueError(
            f"Statistic distribution panel is missing columns: {sorted(missing)!r}."
        )
    table = records.copy()
    statistic = _finite_nonnegative(table["test_statistic"], column_name="test_statistic")
    degrees = _finite_positive(table["degrees_of_freedom"], column_name="degrees_of_freedom")
    alternate_degrees = (
        _finite_positive(
            table["alternate_degrees_of_freedom"],
            column_name="alternate_degrees_of_freedom",
        )
        if "alternate_degrees_of_freedom" in table.columns
        else pd.Series(np.nan, index=table.index, dtype=float)
    )
    alternate_scale = (
        _finite_positive(
            table["alternate_reference_scale"],
            column_name="alternate_reference_scale",
        )
        if "alternate_reference_scale" in table.columns
        else pd.Series(1.0, index=table.index, dtype=float)
    )
    family = (
        table["test_family"].astype(str)
        if "test_family" in table.columns
        else pd.Series("unknown", index=table.index)
    )
    has_alternate = alternate_degrees.notna()
    alternate_scaled_statistic = statistic / alternate_scale
    alternate_cdf = pd.Series(np.nan, index=table.index, dtype=float)
    alternate_p = pd.Series(np.nan, index=table.index, dtype=float)
    alternate_cdf.loc[has_alternate] = chi2.cdf(
        alternate_scaled_statistic.loc[has_alternate],
        df=alternate_degrees.loc[has_alternate],
    )
    alternate_p.loc[has_alternate] = chi2.sf(
        alternate_scaled_statistic.loc[has_alternate],
        df=alternate_degrees.loc[has_alternate],
    )
    rows = pd.DataFrame(
        {
            "record_id": _record_id(table),
            "test_family": family,
            "statistic_context_role": table["statistic_context_role"].astype(str),
            "test_statistic": statistic,
            "degrees_of_freedom": degrees,
            "statistic_per_df": statistic / degrees,
            "chi_square_cdf_value": chi2.cdf(statistic, df=degrees),
            "chi_square_p_value": chi2.sf(statistic, df=degrees),
            "chi_square_centered_residual": (statistic - degrees)
            / np.sqrt(2.0 * degrees),
            "expected_chi_square_skew_at_df": np.sqrt(8.0 / degrees),
            "alternate_degrees_of_freedom": alternate_degrees,
            "alternate_reference_scale": alternate_scale,
            "alternate_scaled_statistic": alternate_scaled_statistic,
            "alternate_chi_square_cdf_value": alternate_cdf,
            "alternate_chi_square_p_value": alternate_p,
            "alternate_chi_square_centered_residual": (
                alternate_scaled_statistic - alternate_degrees
            )
            / np.sqrt(2.0 * alternate_degrees),
            "df_difference_alternate_minus_current": alternate_degrees - degrees,
            "df_ratio_alternate_to_current": alternate_degrees / degrees,
            "df_bin": [_df_bin(float(value)) for value in degrees],
            "study_role": STUDY_ROLE,
        }
    )
    passthrough_columns = [
        column
        for column in (
            "case_id",
            "replicate_id",
            "parent_id",
            "child_id",
            "node_id",
            "edge_path_open",
            "barycentric_balance",
            "edge_action",
            "projection_dimension",
            "sibling_projection_dimension",
            "parent_sample_size",
            "child_sample_size",
        )
        if column in table.columns
    ]
    for column in passthrough_columns:
        rows[column] = table[column].to_numpy()
    return rows


def _shape_status(
    *,
    n_rows: int,
    alpha_tail_rate: float,
    alpha: float,
    tail_tolerance: float,
    empirical_skew: float,
    expected_skew: float,
    skew_tolerance: float,
    ks_p_value: float,
    min_rows: int,
) -> str:
    if n_rows < min_rows:
        return "insufficient_rows"
    tail_aligned = (
        math.isfinite(alpha_tail_rate)
        and abs(alpha_tail_rate - alpha) <= tail_tolerance
    )
    skew_aligned = (
        math.isfinite(empirical_skew)
        and math.isfinite(expected_skew)
        and abs(empirical_skew - expected_skew) <= skew_tolerance
    )
    ks_aligned = math.isfinite(ks_p_value) and ks_p_value >= 0.01
    if tail_aligned and skew_aligned and ks_aligned:
        return "chi_square_shape_candidate"
    if skew_aligned and not tail_aligned:
        return "df_skew_explains_skew_but_tail_misaligned"
    if not skew_aligned:
        return "skew_exceeds_df_reference"
    return "tail_misaligned"


def summarize_statistic_distribution_shape(
    rows: pd.DataFrame,
    *,
    group_columns: Sequence[str] = DEFAULT_GROUP_COLUMNS,
    alpha: float = 0.05,
    tail_tolerance: float = 0.02,
    skew_tolerance: float = 0.35,
    min_rows: int = 30,
) -> pd.DataFrame:
    """Summarize statistic shape by family, context, and df bin."""
    if rows.empty:
        return pd.DataFrame()
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError(f"alpha must lie in (0, 1); got {alpha!r}.")
    if float(tail_tolerance) < 0.0:
        raise ValueError(f"tail_tolerance must be non-negative; got {tail_tolerance!r}.")
    if float(skew_tolerance) < 0.0:
        raise ValueError(f"skew_tolerance must be non-negative; got {skew_tolerance!r}.")
    if int(min_rows) <= 0:
        raise ValueError(f"min_rows must be positive; got {min_rows!r}.")
    missing = set(group_columns) - set(rows.columns)
    if missing:
        raise ValueError(f"group_columns are missing from rows: {sorted(missing)!r}.")

    summaries: list[dict[str, object]] = []
    for key, group in rows.groupby(list(group_columns), dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        statistic = pd.to_numeric(group["test_statistic"], errors="coerce")
        degrees = pd.to_numeric(group["degrees_of_freedom"], errors="coerce")
        per_df = pd.to_numeric(group["statistic_per_df"], errors="coerce")
        residual = pd.to_numeric(group["chi_square_centered_residual"], errors="coerce")
        p_values = pd.to_numeric(group["chi_square_p_value"], errors="coerce")
        cdf_values = pd.to_numeric(group["chi_square_cdf_value"], errors="coerce")
        alternate_degrees = pd.to_numeric(
            group["alternate_degrees_of_freedom"],
            errors="coerce",
        )
        alternate_scale = pd.to_numeric(
            group["alternate_reference_scale"],
            errors="coerce",
        )
        alternate_p_values = pd.to_numeric(
            group["alternate_chi_square_p_value"],
            errors="coerce",
        )
        alternate_cdf_values = pd.to_numeric(
            group["alternate_chi_square_cdf_value"],
            errors="coerce",
        )
        n_rows = int(group.shape[0])
        n_alternate_rows = int(alternate_degrees.notna().sum())
        empirical_skew = _sample_skew(statistic)
        expected_skew = _expected_chi_square_mixture_skew(degrees)
        alternate_expected_skew = _expected_chi_square_mixture_skew(
            alternate_degrees.dropna()
        )
        ks_statistic = math.nan
        ks_p_value = math.nan
        if n_rows >= int(min_rows) and cdf_values.notna().all():
            ks_result = kstest(cdf_values.to_numpy(dtype=float), "uniform")
            ks_statistic = float(ks_result.statistic)
            ks_p_value = float(ks_result.pvalue)
        alternate_ks_statistic = math.nan
        alternate_ks_p_value = math.nan
        if n_alternate_rows >= int(min_rows):
            finite_alternate_cdf = alternate_cdf_values.dropna()
            if finite_alternate_cdf.shape[0] == n_alternate_rows:
                alternate_ks_result = kstest(finite_alternate_cdf.to_numpy(dtype=float), "uniform")
                alternate_ks_statistic = float(alternate_ks_result.statistic)
                alternate_ks_p_value = float(alternate_ks_result.pvalue)
        alpha_tail_rate = float((p_values <= float(alpha)).mean()) if n_rows else math.nan
        alternate_alpha_tail_rate = (
            float((alternate_p_values.dropna() <= float(alpha)).mean())
            if n_alternate_rows
            else math.nan
        )
        row = dict(zip(group_columns, key))
        row.update(
            {
                "n_rows": n_rows,
                "n_alternate_df_rows": n_alternate_rows,
                "degrees_of_freedom_mean": float(degrees.mean()),
                "degrees_of_freedom_min": float(degrees.min()),
                "degrees_of_freedom_max": float(degrees.max()),
                "alternate_degrees_of_freedom_mean": (
                    float(alternate_degrees.mean()) if n_alternate_rows else math.nan
                ),
                "alternate_reference_scale_mean": (
                    float(alternate_scale[alternate_degrees.notna()].mean())
                    if n_alternate_rows
                    else math.nan
                ),
                "df_difference_alternate_minus_current_mean": (
                    float(
                        pd.to_numeric(
                            group["df_difference_alternate_minus_current"],
                            errors="coerce",
                        ).mean()
                    )
                    if n_alternate_rows
                    else math.nan
                ),
                "df_ratio_alternate_to_current_q50": (
                    float(
                        pd.to_numeric(
                            group["df_ratio_alternate_to_current"],
                            errors="coerce",
                        ).quantile(0.50)
                    )
                    if n_alternate_rows
                    else math.nan
                ),
                "statistic_mean": float(statistic.mean()),
                "statistic_q50": float(statistic.quantile(0.50)),
                "statistic_q90": float(statistic.quantile(0.90)),
                "statistic_per_df_q50": float(per_df.quantile(0.50)),
                "statistic_per_df_q90": float(per_df.quantile(0.90)),
                "empirical_statistic_skew": empirical_skew,
                "expected_df_mixture_chi_square_skew": expected_skew,
                "skew_excess_over_df_reference": (
                    float(empirical_skew - expected_skew)
                    if math.isfinite(empirical_skew) and math.isfinite(expected_skew)
                    else math.nan
                ),
                "centered_residual_mean": float(residual.mean()),
                "centered_residual_std": float(residual.std(ddof=1)),
                "centered_residual_skew": _sample_skew(residual),
                "chi_square_p_value_q50": float(p_values.quantile(0.50)),
                "chi_square_p_value_q05": float(p_values.quantile(0.05)),
                "chi_square_tail_rate_at_alpha": alpha_tail_rate,
                "chi_square_tail_delta_from_alpha": float(alpha_tail_rate - float(alpha)),
                "chi_square_pit_ks_statistic": ks_statistic,
                "chi_square_pit_ks_p_value": ks_p_value,
                "alternate_expected_df_mixture_chi_square_skew": alternate_expected_skew,
                "alternate_skew_excess_over_df_reference": (
                    float(empirical_skew - alternate_expected_skew)
                    if math.isfinite(empirical_skew)
                    and math.isfinite(alternate_expected_skew)
                    else math.nan
                ),
                "alternate_chi_square_p_value_q50": (
                    float(alternate_p_values.quantile(0.50))
                    if n_alternate_rows
                    else math.nan
                ),
                "alternate_chi_square_p_value_q05": (
                    float(alternate_p_values.quantile(0.05))
                    if n_alternate_rows
                    else math.nan
                ),
                "alternate_chi_square_tail_rate_at_alpha": alternate_alpha_tail_rate,
                "alternate_chi_square_tail_delta_from_alpha": (
                    float(alternate_alpha_tail_rate - float(alpha))
                    if math.isfinite(alternate_alpha_tail_rate)
                    else math.nan
                ),
                "alternate_chi_square_pit_ks_statistic": alternate_ks_statistic,
                "alternate_chi_square_pit_ks_p_value": alternate_ks_p_value,
                "distribution_shape_status": _shape_status(
                    n_rows=n_rows,
                    alpha_tail_rate=alpha_tail_rate,
                    alpha=float(alpha),
                    tail_tolerance=float(tail_tolerance),
                    empirical_skew=empirical_skew,
                    expected_skew=expected_skew,
                    skew_tolerance=float(skew_tolerance),
                    ks_p_value=ks_p_value,
                    min_rows=int(min_rows),
                ),
                "study_role": STUDY_ROLE,
            }
        )
        summaries.append(row)
    return pd.DataFrame.from_records(summaries)


def run_statistic_distribution_shape_panel(
    *,
    records_path: Path,
    output_dir: Path,
    statistic_column: str = "test_statistic",
    degrees_of_freedom_column: str = "degrees_of_freedom",
    context_role_column: str = "statistic_context_role",
    test_family_column: str | None = "test_family",
    alternate_degrees_of_freedom_column: str | None = None,
    alternate_reference_scale_column: str | None = None,
    alpha: float = 0.05,
    min_rows: int = 30,
) -> dict[str, Path]:
    """Run the statistic distribution-shape panel from a CSV file."""
    source = pd.read_csv(records_path)
    mapped = pd.DataFrame(
        {
            "test_statistic": source[statistic_column],
            "degrees_of_freedom": source[degrees_of_freedom_column],
            "statistic_context_role": source[context_role_column],
            "test_family": (
                source[test_family_column]
                if test_family_column and test_family_column in source.columns
                else "unknown"
            ),
        }
    )
    if alternate_degrees_of_freedom_column:
        mapped["alternate_degrees_of_freedom"] = source[alternate_degrees_of_freedom_column]
    if alternate_reference_scale_column:
        mapped["alternate_reference_scale"] = source[alternate_reference_scale_column]
    for column in (
        "record_id",
        "case_id",
        "replicate_id",
        "parent_id",
        "child_id",
        "node_id",
        "edge_path_open",
        "barycentric_balance",
        "edge_action",
        "projection_dimension",
        "sibling_projection_dimension",
        "parent_sample_size",
        "child_sample_size",
    ):
        if column in source.columns:
            mapped[column] = source[column]
    rows = normalize_statistic_distribution_records(mapped)
    summary = summarize_statistic_distribution_shape(
        rows,
        alpha=alpha,
        min_rows=min_rows,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "statistic_distribution_shape_rows.csv"
    summary_path = output_dir / "statistic_distribution_shape_summary.csv"
    manifest_path = output_dir / "manifest.json"
    rows.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)
    manifest = {
        "created_at_utc": format_timestamp_utc(),
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "records_path": str(records_path),
        "statistic_column": statistic_column,
        "degrees_of_freedom_column": degrees_of_freedom_column,
        "context_role_column": context_role_column,
        "test_family_column": test_family_column,
        "alternate_degrees_of_freedom_column": alternate_degrees_of_freedom_column,
        "alternate_reference_scale_column": alternate_reference_scale_column,
        "outputs": {
            "rows": str(rows_path),
            "summary": str(summary_path),
        },
        "interpretation": (
            "Diagnostic distribution-shape panel. It compares empirical statistic "
            "skew to the chi-square skew implied by the observed degrees-of-freedom "
            "mixture and checks chi-square PIT/tail alignment."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"rows": rows_path, "summary": summary_path, "manifest": manifest_path}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--statistic-column", default="test_statistic")
    parser.add_argument("--degrees-of-freedom-column", default="degrees_of_freedom")
    parser.add_argument("--context-role-column", default="statistic_context_role")
    parser.add_argument("--test-family-column", default="test_family")
    parser.add_argument("--alternate-degrees-of-freedom-column", default=None)
    parser.add_argument("--alternate-reference-scale-column", default=None)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-rows", type=int, default=30)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    outputs = run_statistic_distribution_shape_panel(
        records_path=args.records,
        output_dir=args.output_dir,
        statistic_column=args.statistic_column,
        degrees_of_freedom_column=args.degrees_of_freedom_column,
        context_role_column=args.context_role_column,
        test_family_column=args.test_family_column,
        alternate_degrees_of_freedom_column=args.alternate_degrees_of_freedom_column,
        alternate_reference_scale_column=args.alternate_reference_scale_column,
        alpha=args.alpha,
        min_rows=args.min_rows,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "STUDY_ROLE",
    "infer_satterthwaite_reference_from_eigenvalues",
    "normalize_statistic_distribution_records",
    "run_statistic_distribution_shape_panel",
    "summarize_statistic_distribution_shape",
]
