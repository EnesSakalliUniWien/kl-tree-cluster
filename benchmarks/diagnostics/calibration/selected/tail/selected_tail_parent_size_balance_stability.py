"""Parent-size-stable selected-tail law diagnostic with barycentric balance.

This diagnostic tests whether a selected-tail context can transfer across
parent-size bins after conditioning on feature family, projection dimension,
edge-action bin, and predeclared barycentric-balance bin. It also reports
simulation-level c-hat precision metadata.

The output is diagnostic-only. It does not install an external calibration
path.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.diagnostics.calibration.selected.hierarchy.selected_hierarchy_geometry_covariates import (
    EDGE_ACTION_BIN_LABELS,
    EDGE_ACTION_BINS,
    SIMULATION_ID_COLUMN,
)
from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "diagnostic_parent_size_balance_tail_law_not_calibration"
REQUIRED_COLUMNS = {
    "source_family",
    "feature_family",
    "parent_size_bin",
    "sibling_projection_dimension",
    "negative_log10_min_child_edge_bh_p_value",
    "parent_sample_size",
    "left_child_sample_size",
    "right_child_sample_size",
    "selected_hierarchy_ratio",
    SIMULATION_ID_COLUMN,
}
BALANCE_BINS = (0.0, 0.1, 0.25, 0.4, 0.5000000001)
BALANCE_BIN_LABELS = (
    "balance_0_0.1",
    "balance_0.1_0.25",
    "balance_0.25_0.4",
    "balance_0.4_0.5",
)
CONTEXT_COLUMNS = (
    "source_family",
    "feature_family",
    "sibling_projection_dimension",
    "edge_action_bin",
    "barycentric_balance_bin",
)


def _positive_numeric(values: pd.Series, *, column_name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise").astype(float)
    if bool(((numeric <= 0.0) | ~np.isfinite(numeric)).any()):
        bad_index = numeric[((numeric <= 0.0) | ~np.isfinite(numeric))].index[0]
        raise ValueError(
            f"{column_name} must contain finite positive values; "
            f"row={int(bad_index)}, value={float(numeric.loc[bad_index])!r}."
        )
    return numeric


def _edge_action_bin(edge_action: float) -> str:
    for lower, upper, label in zip(
        EDGE_ACTION_BINS,
        EDGE_ACTION_BINS[1:],
        EDGE_ACTION_BIN_LABELS,
    ):
        if lower <= edge_action < upper:
            return str(label)
    raise ValueError(f"edge action did not match a predeclared bin: {edge_action!r}.")


def _balance_bin(balance: float) -> str:
    if not np.isfinite(balance) or balance <= 0.0 or balance > 0.5:
        raise ValueError(f"barycentric balance must lie in (0, 0.5]; got {balance!r}.")
    for lower, upper, label in zip(
        BALANCE_BINS,
        BALANCE_BINS[1:],
        BALANCE_BIN_LABELS,
    ):
        if lower < balance <= upper or (lower == 0.0 and lower < balance < upper):
            return label
    raise ValueError(f"barycentric balance did not match a bin: {balance!r}.")


def prepare_parent_size_balance_records(records: pd.DataFrame) -> pd.DataFrame:
    """Add edge-action and barycentric-balance bins to selected records."""
    missing = REQUIRED_COLUMNS - set(records.columns)
    if missing:
        raise ValueError(
            f"Parent-size balance stability records are missing columns: {sorted(missing)!r}."
        )
    table = records.copy()
    parent = _positive_numeric(table["parent_sample_size"], column_name="parent_sample_size")
    left = _positive_numeric(
        table["left_child_sample_size"],
        column_name="left_child_sample_size",
    )
    right = _positive_numeric(
        table["right_child_sample_size"],
        column_name="right_child_sample_size",
    )
    child_size_error = np.abs((left + right) - parent)
    if bool((child_size_error > 1e-8).any()):
        bad_index = child_size_error[child_size_error > 1e-8].index[0]
        raise ValueError(
            "left_child_sample_size + right_child_sample_size must equal "
            f"parent_sample_size; row={int(bad_index)}."
        )

    beta_left = left / parent
    beta_right = right / parent
    balance = np.minimum(beta_left, beta_right)
    table["left_barycentric_weight"] = beta_left
    table["barycentric_balance"] = balance
    table["log_barycentric_leverage"] = np.log(np.maximum(beta_left, beta_right) / balance)
    table["barycentric_balance_bin"] = [_balance_bin(float(value)) for value in balance]

    edge_action = pd.to_numeric(
        table["negative_log10_min_child_edge_bh_p_value"],
        errors="raise",
    ).astype(float)
    if bool(((edge_action < 0.0) | ~np.isfinite(edge_action)).any()):
        bad_index = edge_action[((edge_action < 0.0) | ~np.isfinite(edge_action))].index[0]
        raise ValueError(
            "negative_log10_min_child_edge_bh_p_value must be finite and "
            f"non-negative; row={int(bad_index)}."
        )
    derived_edge_bins = pd.Series(
        [_edge_action_bin(float(value)) for value in edge_action],
        index=table.index,
    )
    if "edge_action_bin" in table.columns:
        provided = table["edge_action_bin"].astype(str)
        stale = provided.ne(derived_edge_bins.astype(str))
        if bool(stale.any()):
            bad_index = stale[stale].index[0]
            raise ValueError(
                "Selected records contain stale edge_action_bin; "
                f"row={int(bad_index)}, provided={provided.loc[bad_index]!r}, "
                f"derived={derived_edge_bins.loc[bad_index]!r}."
            )
    table["edge_action_bin"] = derived_edge_bins
    table["selected_hierarchy_ratio"] = _positive_numeric(
        table["selected_hierarchy_ratio"],
        column_name="selected_hierarchy_ratio",
    )
    table["sibling_projection_dimension"] = pd.to_numeric(
        table["sibling_projection_dimension"],
        errors="raise",
    ).astype(int)
    table[SIMULATION_ID_COLUMN] = table[SIMULATION_ID_COLUMN].astype(str)
    return table


def _simulation_relative_c_se(group: pd.DataFrame) -> float:
    simulation_means = group.groupby(SIMULATION_ID_COLUMN)["selected_hierarchy_ratio"].mean()
    c_hat = float(group["selected_hierarchy_ratio"].mean())
    if simulation_means.shape[0] <= 1 or c_hat <= 0.0:
        return np.nan
    se = float(simulation_means.std(ddof=1) / np.sqrt(simulation_means.shape[0]))
    return float(se / c_hat)


def _fold_status(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    min_train_simulations: int,
    min_train_records: int,
    min_test_records: int,
) -> str:
    if test.shape[0] < min_test_records:
        return "insufficient_test_records"
    if train[SIMULATION_ID_COLUMN].nunique() < min_train_simulations:
        return "insufficient_train_simulations"
    if train.shape[0] < min_train_records:
        return "insufficient_train_records"
    return "ok"


def _evaluate_parent_size_folds(
    group: pd.DataFrame,
    *,
    alpha: float,
    min_train_simulations: int,
    min_train_records: int,
    min_test_records: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    fold_rows: list[dict[str, object]] = []
    exceedances: list[np.ndarray] = []
    thresholds: list[float] = []
    parent_bins = tuple(sorted(str(value) for value in group["parent_size_bin"].unique()))
    for parent_bin in parent_bins:
        test = group[group["parent_size_bin"].astype(str).eq(parent_bin)]
        train = group[~group["parent_size_bin"].astype(str).eq(parent_bin)]
        status = _fold_status(
            train=train,
            test=test,
            min_train_simulations=min_train_simulations,
            min_train_records=min_train_records,
            min_test_records=min_test_records,
        )
        threshold = np.nan
        exceedance_rate = np.nan
        exceedance_abs_error = np.nan
        exceedance_se = np.nan
        if status == "ok":
            threshold = float(
                np.quantile(
                    train["selected_hierarchy_ratio"].to_numpy(dtype=float),
                    1.0 - alpha,
                )
            )
            fold_exceedances = test["selected_hierarchy_ratio"].to_numpy(dtype=float) > threshold
            exceedances.append(fold_exceedances)
            thresholds.append(threshold)
            exceedance_rate = float(np.mean(fold_exceedances))
            exceedance_abs_error = float(abs(exceedance_rate - alpha))
            exceedance_se = float(
                np.sqrt(
                    exceedance_rate * (1.0 - exceedance_rate) / max(fold_exceedances.shape[0], 1)
                )
            )
        fold_rows.append(
            {
                "heldout_parent_size_bin": parent_bin,
                "n_train_records": int(train.shape[0]),
                "n_test_records": int(test.shape[0]),
                "n_train_simulations": int(train[SIMULATION_ID_COLUMN].nunique()),
                "n_test_simulations": int(test[SIMULATION_ID_COLUMN].nunique()),
                "parent_size_threshold": threshold,
                "fold_heldout_exceedance_rate": exceedance_rate,
                "fold_heldout_exceedance_absolute_error": exceedance_abs_error,
                "fold_heldout_exceedance_standard_error": exceedance_se,
                "fold_status": status,
            }
        )

    if not exceedances:
        return fold_rows, {
            "parent_size_bin_count": len(parent_bins),
            "parent_size_bins": ";".join(parent_bins),
            "n_parent_size_folds": len(parent_bins),
            "n_used_parent_size_folds": 0,
            "parent_size_threshold_mean": np.nan,
            "parent_size_threshold_median": np.nan,
            "parent_size_heldout_exceedance_rate": np.nan,
            "parent_size_heldout_exceedance_absolute_error": np.nan,
            "parent_size_heldout_exceedance_standard_error": np.nan,
            "parent_size_holdout_status": "no_valid_parent_size_folds",
        }
    all_exceedances = np.concatenate(exceedances)
    exceedance_rate = float(np.mean(all_exceedances))
    exceedance_se = float(
        np.sqrt(exceedance_rate * (1.0 - exceedance_rate) / all_exceedances.shape[0])
    )
    return fold_rows, {
        "parent_size_bin_count": len(parent_bins),
        "parent_size_bins": ";".join(parent_bins),
        "n_parent_size_folds": len(parent_bins),
        "n_used_parent_size_folds": len(exceedances),
        "parent_size_threshold_mean": float(np.mean(thresholds)),
        "parent_size_threshold_median": float(np.median(thresholds)),
        "parent_size_heldout_exceedance_rate": exceedance_rate,
        "parent_size_heldout_exceedance_absolute_error": float(abs(exceedance_rate - alpha)),
        "parent_size_heldout_exceedance_standard_error": exceedance_se,
        "parent_size_holdout_status": "parent_size_holdout_evaluated",
    }


def _context_decision(
    *,
    support_contract_met: bool,
    c_hat_precision_met: bool,
    parent_size_holdout_valid: bool,
    parent_size_abs_error_met: bool,
    parent_size_precision_met: bool,
) -> str:
    if not support_contract_met:
        return "undefined_support_failure"
    if not parent_size_holdout_valid:
        return "undefined_parent_size_holdout"
    if c_hat_precision_met and parent_size_abs_error_met and parent_size_precision_met:
        return "parent_size_balance_external_candidate"
    return "diagnostic_only_parent_size_unstable"


def _failure_reasons(
    *,
    support_contract_met: bool,
    c_hat_precision_met: bool,
    parent_size_holdout_valid: bool,
    parent_size_abs_error_met: bool,
    parent_size_precision_met: bool,
) -> str:
    reasons: list[str] = []
    if not support_contract_met:
        reasons.append("support_contract_failed")
    if not c_hat_precision_met:
        reasons.append("relative_c_simulation_se_failed")
    if not parent_size_holdout_valid:
        reasons.append("parent_size_holdout_unavailable")
    if not parent_size_abs_error_met:
        reasons.append("parent_size_abs_error_failed")
    if not parent_size_precision_met:
        reasons.append("parent_size_precision_failed")
    return ";".join(reasons)


def evaluate_parent_size_balance_stability(
    records: pd.DataFrame,
    *,
    alpha: float = 0.01,
    min_train_simulations: int = 20,
    min_train_records: int = 20,
    min_test_records: int = 10,
    required_min_matching_simulations: int = 499,
    required_min_matched_records: int = 499,
    max_relative_c_simulation_se: float = 0.05,
    max_exceedance_standard_error: float = 0.002,
    max_parent_size_abs_error: float = 0.005,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate balance-scoped selected-tail transfer across parent sizes."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1).")
    table = prepare_parent_size_balance_records(records)
    context_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []

    for context_values, group in table.groupby(list(CONTEXT_COLUMNS), dropna=False):
        if not isinstance(context_values, tuple):
            context_values = (context_values,)
        context = {column: value for column, value in zip(CONTEXT_COLUMNS, context_values)}
        context_fold_rows, parent_size_summary = _evaluate_parent_size_folds(
            group,
            alpha=alpha,
            min_train_simulations=min_train_simulations,
            min_train_records=min_train_records,
            min_test_records=min_test_records,
        )
        for row in context_fold_rows:
            row.update(context)
            fold_rows.append(row)

        n_records = int(group.shape[0])
        n_matching_simulations = int(group[SIMULATION_ID_COLUMN].nunique())
        c_hat = float(group["selected_hierarchy_ratio"].mean())
        relative_c_se = _simulation_relative_c_se(group)
        support_contract_met = (
            n_matching_simulations >= required_min_matching_simulations
            and n_records >= required_min_matched_records
        )
        c_hat_precision_met = bool(
            np.isfinite(relative_c_se) and relative_c_se <= float(max_relative_c_simulation_se)
        )
        parent_size_holdout_valid = bool(parent_size_summary["n_used_parent_size_folds"] >= 2)
        parent_size_abs_error = float(
            parent_size_summary["parent_size_heldout_exceedance_absolute_error"]
        )
        parent_size_se = float(parent_size_summary["parent_size_heldout_exceedance_standard_error"])
        parent_size_abs_error_met = bool(
            np.isfinite(parent_size_abs_error)
            and parent_size_abs_error <= max_parent_size_abs_error
        )
        parent_size_precision_met = bool(
            np.isfinite(parent_size_se) and parent_size_se <= max_exceedance_standard_error
        )
        decision = _context_decision(
            support_contract_met=support_contract_met,
            c_hat_precision_met=c_hat_precision_met,
            parent_size_holdout_valid=parent_size_holdout_valid,
            parent_size_abs_error_met=parent_size_abs_error_met,
            parent_size_precision_met=parent_size_precision_met,
        )
        output = dict(context)
        output.update(
            {
                "study_role": STUDY_ROLE,
                "alpha": float(alpha),
                "n_records": n_records,
                "n_matching_simulations": n_matching_simulations,
                "required_min_matching_simulations": int(required_min_matching_simulations),
                "required_min_matched_records": int(required_min_matched_records),
                "support_contract_met": support_contract_met,
                "selected_hierarchy_c_hat": c_hat,
                "selected_hierarchy_c_hat_relative_simulation_se": relative_c_se,
                "required_max_relative_c_simulation_se": float(max_relative_c_simulation_se),
                "c_hat_precision_met": c_hat_precision_met,
                "max_exceedance_standard_error": float(max_exceedance_standard_error),
                "max_parent_size_abs_error": float(max_parent_size_abs_error),
                "parent_size_holdout_valid": parent_size_holdout_valid,
                "parent_size_abs_error_met": parent_size_abs_error_met,
                "parent_size_precision_met": parent_size_precision_met,
                "stability_decision": decision,
                "stability_failure_reasons": _failure_reasons(
                    support_contract_met=support_contract_met,
                    c_hat_precision_met=c_hat_precision_met,
                    parent_size_holdout_valid=parent_size_holdout_valid,
                    parent_size_abs_error_met=parent_size_abs_error_met,
                    parent_size_precision_met=parent_size_precision_met,
                ),
            }
        )
        output.update(parent_size_summary)
        context_rows.append(output)

    contexts = pd.DataFrame.from_records(context_rows)
    folds = pd.DataFrame.from_records(fold_rows)
    if not folds.empty:
        folds = folds[
            [
                *CONTEXT_COLUMNS,
                "heldout_parent_size_bin",
                "n_train_records",
                "n_test_records",
                "n_train_simulations",
                "n_test_simulations",
                "parent_size_threshold",
                "fold_heldout_exceedance_rate",
                "fold_heldout_exceedance_absolute_error",
                "fold_heldout_exceedance_standard_error",
                "fold_status",
            ]
        ]
    summary = summarize_parent_size_balance_stability(contexts)
    return contexts.sort_values(list(CONTEXT_COLUMNS)).reset_index(drop=True), folds, summary


def summarize_parent_size_balance_stability(contexts: pd.DataFrame) -> pd.DataFrame:
    """Summarize context decisions."""
    rows: list[dict[str, object]] = []
    if contexts.empty:
        return pd.DataFrame()
    for decision, group in contexts.groupby("stability_decision", dropna=False):
        rows.append(
            {
                "stability_decision": str(decision),
                "n_contexts": int(group.shape[0]),
                "n_support_contract_met": int(group["support_contract_met"].sum()),
                "n_c_hat_precision_met": int(group["c_hat_precision_met"].sum()),
                "n_parent_size_holdout_valid": int(group["parent_size_holdout_valid"].sum()),
                "n_parent_size_abs_error_met": int(group["parent_size_abs_error_met"].sum()),
                "n_parent_size_precision_met": int(group["parent_size_precision_met"].sum()),
                "study_role": STUDY_ROLE,
            }
        )
    return pd.DataFrame.from_records(rows).sort_values("stability_decision")


def run_parent_size_balance_stability_diagnostic(
    *,
    records_path: Path,
    output_dir: Path,
    alpha: float = 0.01,
    min_train_simulations: int = 20,
    min_train_records: int = 20,
    min_test_records: int = 10,
    required_min_matching_simulations: int = 499,
    required_min_matched_records: int = 499,
    max_relative_c_simulation_se: float = 0.05,
    max_exceedance_standard_error: float = 0.002,
    max_parent_size_abs_error: float = 0.005,
) -> dict[str, Path]:
    """Run the parent-size balance stability diagnostic from row-level records."""
    records = pd.read_csv(records_path)
    contexts, folds, summary = evaluate_parent_size_balance_stability(
        records,
        alpha=alpha,
        min_train_simulations=min_train_simulations,
        min_train_records=min_train_records,
        min_test_records=min_test_records,
        required_min_matching_simulations=required_min_matching_simulations,
        required_min_matched_records=required_min_matched_records,
        max_relative_c_simulation_se=max_relative_c_simulation_se,
        max_exceedance_standard_error=max_exceedance_standard_error,
        max_parent_size_abs_error=max_parent_size_abs_error,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    contexts_path = output_dir / "parent_size_balance_contexts.csv"
    folds_path = output_dir / "parent_size_balance_parent_folds.csv"
    summary_path = output_dir / "parent_size_balance_summary.csv"
    manifest_path = output_dir / "manifest.json"
    contexts.to_csv(contexts_path, index=False)
    folds.to_csv(folds_path, index=False)
    summary.to_csv(summary_path, index=False)
    manifest = {
        "created_at_utc": format_timestamp_utc(),
        "study_role": STUDY_ROLE,
        "records_path": str(records_path),
        "alpha": float(alpha),
        "required_min_matching_simulations": int(required_min_matching_simulations),
        "required_min_matched_records": int(required_min_matched_records),
        "max_relative_c_simulation_se": float(max_relative_c_simulation_se),
        "max_exceedance_standard_error": float(max_exceedance_standard_error),
        "max_parent_size_abs_error": float(max_parent_size_abs_error),
        "context_columns": list(CONTEXT_COLUMNS),
        "balance_bins": list(BALANCE_BIN_LABELS),
        "outputs": {
            "contexts": str(contexts_path),
            "parent_folds": str(folds_path),
            "summary": str(summary_path),
        },
        "interpretation": (
            "Diagnostic-only parent-size transfer gate scoped by edge action, "
            "projection dimension, feature family, source family, and "
            "barycentric balance. Candidate rows are not production calibration "
            "rules without a separate promotion gate."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "contexts": contexts_path,
        "parent_folds": folds_path,
        "summary": summary_path,
        "manifest": manifest_path,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--min-train-simulations", type=int, default=20)
    parser.add_argument("--min-train-records", type=int, default=20)
    parser.add_argument("--min-test-records", type=int, default=10)
    parser.add_argument("--required-min-matching-simulations", type=int, default=499)
    parser.add_argument("--required-min-matched-records", type=int, default=499)
    parser.add_argument("--max-relative-c-simulation-se", type=float, default=0.05)
    parser.add_argument("--max-exceedance-standard-error", type=float, default=0.002)
    parser.add_argument("--max-parent-size-abs-error", type=float, default=0.005)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    outputs = run_parent_size_balance_stability_diagnostic(
        records_path=args.records,
        output_dir=args.output_dir,
        alpha=float(args.alpha),
        min_train_simulations=int(args.min_train_simulations),
        min_train_records=int(args.min_train_records),
        min_test_records=int(args.min_test_records),
        required_min_matching_simulations=int(args.required_min_matching_simulations),
        required_min_matched_records=int(args.required_min_matched_records),
        max_relative_c_simulation_se=float(args.max_relative_c_simulation_se),
        max_exceedance_standard_error=float(args.max_exceedance_standard_error),
        max_parent_size_abs_error=float(args.max_parent_size_abs_error),
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "BALANCE_BIN_LABELS",
    "CONTEXT_COLUMNS",
    "STUDY_ROLE",
    "evaluate_parent_size_balance_stability",
    "prepare_parent_size_balance_records",
    "run_parent_size_balance_stability_diagnostic",
    "summarize_parent_size_balance_stability",
]
