"""Topology-aware selected-tail context refinement diagnostic.

This diagnostic tests whether selected-tail contexts that are too coarse under
the base contract become more tail-homogeneous after adding topology, edge-path,
and spectral-alignment bins. It is descriptive evidence only. It does not add a
production external calibration rule or fallback.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.diagnostics.calibration.selected.hierarchy.selected_hierarchy_geometry_covariates import (
    EDGE_ACTION_BIN_LABELS,
    SIMULATION_ID_COLUMN,
    TAIL_LAW_CONTEXT_COLUMNS,
    _edge_action_bin,
    evaluate_selected_ratio_tail_law,
)
from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "descriptive_topology_refined_selected_tail_not_calibration"
PREDECLARED_CONTEXT_MODE = "predeclared_base_context"
DATA_ADAPTIVE_CONTEXT_MODE = "data_adaptive_refinement_context"
DATA_ADAPTIVE_PRODUCTION_BLOCK_REASON = (
    "data_adaptive_refinement_context_not_predeclared_production_contract"
)

BASE_CONTEXT_COLUMNS = TAIL_LAW_CONTEXT_COLUMNS

BIN_SPECS: tuple[tuple[str, str, int], ...] = (
    ("child_balance", "child_balance_bin", 3),
    ("subtree_colless_normalized", "subtree_colless_bin", 3),
    ("subtree_sackin_mean_depth", "subtree_sackin_depth_bin", 3),
    ("subtree_branch_length_condition_ratio", "subtree_branch_condition_bin", 3),
    ("subtree_branch_length_cv", "subtree_branch_cv_bin", 3),
    ("branch_length_asymmetry", "sibling_branch_asymmetry_bin", 3),
    ("negative_log10_min_child_edge_bh_p_value", "edge_action_quantile_bin", 3),
    ("eigenvalue_effective_rank", "effective_rank_bin", 3),
    ("selected_eigenvalue_over_mp_upper_bound", "eigenvalue_mp_ratio_bin", 3),
    ("selected_subspace_cos2", "selected_subspace_cos2_bin", 3),
)

REFINEMENT_CONTEXTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("base", ()),
    ("balance", ("child_balance_bin",)),
    (
        "topology",
        (
            "child_balance_bin",
            "subtree_colless_bin",
            "subtree_sackin_depth_bin",
        ),
    ),
    (
        "merge_persistence",
        (
            "subtree_branch_condition_bin",
            "subtree_branch_cv_bin",
            "sibling_branch_asymmetry_bin",
        ),
    ),
    ("edge_path", ("edge_action_quantile_bin",)),
    (
        "spectral_alignment",
        (
            "effective_rank_bin",
            "eigenvalue_mp_ratio_bin",
            "selected_subspace_cos2_bin",
        ),
    ),
    (
        "compact_topology_edge_spectral",
        (
            "child_balance_bin",
            "subtree_sackin_depth_bin",
            "edge_action_quantile_bin",
            "eigenvalue_mp_ratio_bin",
            "selected_subspace_cos2_bin",
        ),
    ),
)

REQUIRED_RECORD_COLUMNS = {
    *(column for column in BASE_CONTEXT_COLUMNS if column != "edge_action_bin"),
    SIMULATION_ID_COLUMN,
    "selected_hierarchy_ratio",
    "feature_dimension",
    "parent_sample_size",
    "left_child_sample_size",
    "right_child_sample_size",
    *(source for source, _target, _bins in BIN_SPECS),
}


@dataclass(frozen=True)
class RecordsInput:
    """Named selected-geometry record table."""

    run_id: str
    path: Path


def parse_records_specs(raw_specs: list[str]) -> tuple[RecordsInput, ...]:
    """Parse CLI specs of the form run_id=/path/to/selected_geometry_records.csv."""
    records_inputs: list[RecordsInput] = []
    seen_run_ids: set[str] = set()
    for raw_spec in raw_specs:
        if "=" not in raw_spec:
            raise ValueError(
                f"Selected-geometry record specs must use run_id=csv_path; got {raw_spec!r}."
            )
        run_id, raw_path = raw_spec.split("=", maxsplit=1)
        run_id = run_id.strip()
        raw_path = raw_path.strip()
        if not run_id:
            raise ValueError(f"Record spec has empty run_id: {raw_spec!r}.")
        if not raw_path:
            raise ValueError(f"Record spec has empty path: {raw_spec!r}.")
        if run_id in seen_run_ids:
            raise ValueError(f"Duplicate selected-geometry run_id {run_id!r}.")
        seen_run_ids.add(run_id)
        records_inputs.append(RecordsInput(run_id=run_id, path=Path(raw_path)))
    if not records_inputs:
        raise ValueError("At least one selected-geometry record spec is required.")
    return tuple(records_inputs)


def _derived_edge_action_bins(table: pd.DataFrame) -> pd.Series:
    return pd.Series(
        [
            _edge_action_bin(float(value))
            for value in table["negative_log10_min_child_edge_bh_p_value"]
        ],
        index=table.index,
        dtype=object,
    )


def load_selected_geometry_records(records_inputs: Sequence[RecordsInput]) -> pd.DataFrame:
    """Load and combine selected-geometry records with explicit run ids."""
    tables: list[pd.DataFrame] = []
    for records_input in records_inputs:
        table = pd.read_csv(records_input.path)
        missing = REQUIRED_RECORD_COLUMNS - set(table.columns)
        if missing:
            raise ValueError(
                f"Selected-geometry records {records_input.path} are missing "
                f"required columns: {sorted(missing)!r}."
            )
        table = table.copy()
        derived_edge_action_bins = _derived_edge_action_bins(table)
        if "edge_action_bin" not in table.columns:
            table["edge_action_bin"] = derived_edge_action_bins
        else:
            provided_bins = table["edge_action_bin"].astype(str)
            stale_mask = provided_bins.ne(derived_edge_action_bins.astype(str))
            if bool(stale_mask.any()):
                first_stale_index = stale_mask[stale_mask].index[0]
                raise ValueError(
                    "Selected-geometry records contain stale edge_action_bin "
                    f"at row {int(first_stale_index)}: provided "
                    f"{provided_bins.loc[first_stale_index]!r}, derived "
                    f"{derived_edge_action_bins.loc[first_stale_index]!r} from "
                    "negative_log10_min_child_edge_bh_p_value."
                )
            table["edge_action_bin"] = derived_edge_action_bins
        table[SIMULATION_ID_COLUMN] = [
            f"{records_input.run_id}:{simulation_id}"
            for simulation_id in table[SIMULATION_ID_COLUMN].astype(str)
        ]
        table.insert(0, "run_id", records_input.run_id)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def _safe_value_label(value: object) -> str:
    text = str(value)
    output = []
    for char in text:
        output.append(char if char.isalnum() else "_")
    return "".join(output).strip("_") or "empty"


def _bin_finite_values(values: pd.Series, *, n_bins: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise").astype(float)
    labels = pd.Series("missing", index=values.index, dtype=object)
    finite_mask = np.isfinite(numeric)
    finite = numeric.loc[finite_mask]
    if finite.empty:
        return labels
    unique_values = np.asarray(sorted(float(value) for value in finite.unique()))
    if unique_values.shape[0] == 1:
        labels.loc[finite.index] = "constant"
        return labels
    if unique_values.shape[0] <= n_bins:
        rank_by_value = {
            value: f"value_{rank}_{_safe_value_label(value)}"
            for rank, value in enumerate(unique_values, start=1)
        }
        labels.loc[finite.index] = [
            rank_by_value[float(value)] for value in finite.to_numpy(dtype=float)
        ]
        return labels

    binned = pd.qcut(
        finite,
        q=int(n_bins),
        labels=False,
        duplicates="drop",
    )
    labels.loc[finite.index] = [f"q{int(value) + 1}" for value in binned]
    return labels


def add_refinement_bins(
    records: pd.DataFrame,
    *,
    base_context_columns: Sequence[str] = BASE_CONTEXT_COLUMNS,
    bin_specs: Sequence[tuple[str, str, int]] = BIN_SPECS,
) -> pd.DataFrame:
    """Add within-base-context bins for topology and geometry variables."""
    table = records.copy()
    missing_context = [column for column in base_context_columns if column not in table.columns]
    if missing_context:
        raise KeyError(f"Missing base context column(s): {missing_context!r}.")
    for source_column, target_column, n_bins in bin_specs:
        if source_column not in table.columns:
            raise KeyError(f"Missing refinement source column {source_column!r}.")
        table[target_column] = "missing"
        for _context_values, group in table.groupby(list(base_context_columns), dropna=False):
            table.loc[group.index, target_column] = _bin_finite_values(
                group[source_column],
                n_bins=int(n_bins),
            )
    return table


def _append_failure_reason(existing: object, reason: str) -> str:
    text = "" if pd.isna(existing) else str(existing)
    return reason if not text else f"{text};{reason}"


def _apply_context_definition_contract(
    tail_law: pd.DataFrame,
    *,
    context_family: str,
) -> pd.DataFrame:
    """Separate diagnostic support checks from production admissibility.

    The base context is predeclared by the selected-tail support contract.
    Refined contexts are built from within-panel bins, so they are useful for
    debugging tail heterogeneity but cannot become production calibration laws.
    """
    table = tail_law.copy()
    diagnostic_passed = table["production_tail_law_admissible"].astype(bool)
    diagnostic_failures = table["tail_law_admissibility_failure_reasons"].astype(str)
    table["diagnostic_tail_law_contract_passed"] = diagnostic_passed
    table["diagnostic_tail_law_contract_failure_reasons"] = diagnostic_failures

    if context_family == "base":
        table["context_definition_mode"] = PREDECLARED_CONTEXT_MODE
        table["production_tail_law_failure_reasons"] = diagnostic_failures
        return table

    table["context_definition_mode"] = DATA_ADAPTIVE_CONTEXT_MODE
    table["production_tail_law_admissible"] = False
    table["production_tail_law_failure_reasons"] = [
        _append_failure_reason(reason, DATA_ADAPTIVE_PRODUCTION_BLOCK_REASON)
        for reason in diagnostic_failures
    ]
    return table


def _tail_precision_pass_mask(table: pd.DataFrame) -> pd.Series:
    return table["heldout_exceedance_standard_error"] <= table["max_exceedance_standard_error"]


def evaluate_topology_refined_tail_laws(
    records: pd.DataFrame,
    *,
    refinement_contexts: Sequence[tuple[str, tuple[str, ...]]] = REFINEMENT_CONTEXTS,
    base_context_columns: Sequence[str] = BASE_CONTEXT_COLUMNS,
    alpha: float = 0.01,
    n_folds: int = 5,
    min_train_simulations: int = 20,
    min_train_records: int = 20,
    required_min_matching_simulations: int = 499,
    required_min_matched_records: int = 499,
    max_exceedance_standard_error: float = 0.002,
) -> pd.DataFrame:
    """Evaluate base and refined selected-tail contexts."""
    table = add_refinement_bins(records, base_context_columns=base_context_columns)
    tail_laws: list[pd.DataFrame] = []
    for context_family, additional_columns in refinement_contexts:
        context_columns = tuple(base_context_columns) + tuple(additional_columns)
        tail_law = evaluate_selected_ratio_tail_law(
            table,
            alpha=alpha,
            n_folds=n_folds,
            min_train_simulations=min_train_simulations,
            min_train_records=min_train_records,
            required_min_matching_simulations=required_min_matching_simulations,
            required_min_matched_records=required_min_matched_records,
            max_exceedance_standard_error=max_exceedance_standard_error,
            context_columns=context_columns,
        )
        tail_law = _apply_context_definition_contract(
            tail_law,
            context_family=context_family,
        )
        tail_law.insert(0, "context_family", context_family)
        tail_law.insert(1, "context_columns", ",".join(context_columns))
        tail_law.insert(2, "additional_context_columns", ",".join(additional_columns))
        tail_law["study_role"] = STUDY_ROLE
        tail_laws.append(tail_law)
    return pd.concat(tail_laws, ignore_index=True)


def summarize_refinement_families(tail_laws: pd.DataFrame) -> pd.DataFrame:
    """Summarize support and held-out precision by refinement family."""
    rows: list[dict[str, object]] = []
    for context_family, group in tail_laws.groupby("context_family", dropna=False):
        descriptive = group[group["tail_law_status"].eq("descriptive_holdout_tail_law")]
        precision_pass = descriptive[_tail_precision_pass_mask(descriptive)]
        rows.append(
            {
                "context_family": context_family,
                "context_columns": str(group["context_columns"].iloc[0]),
                "n_contexts": int(group.shape[0]),
                "n_descriptive_contexts": int(descriptive.shape[0]),
                "n_production_admissible_contexts": int(
                    group["production_tail_law_admissible"].sum()
                ),
                "n_diagnostic_tail_law_contract_passed_contexts": int(
                    group["diagnostic_tail_law_contract_passed"].sum()
                ),
                "n_precision_pass_contexts": int(precision_pass.shape[0]),
                "max_matching_simulations": int(group["n_matching_simulations"].max()),
                "max_records": int(group["n_records"].max()),
                "min_heldout_exceedance_standard_error": float(
                    descriptive["heldout_exceedance_standard_error"].min()
                )
                if not descriptive.empty
                else np.nan,
                "median_heldout_exceedance_standard_error": float(
                    descriptive["heldout_exceedance_standard_error"].median()
                )
                if not descriptive.empty
                else np.nan,
                "median_heldout_exceedance_absolute_error": float(
                    descriptive["heldout_exceedance_absolute_error"].median()
                )
                if not descriptive.empty
                else np.nan,
                "study_role": STUDY_ROLE,
            }
        )
    return pd.DataFrame.from_records(rows).sort_values("context_family").reset_index(drop=True)


def _base_context_key(
    row: pd.Series,
    *,
    base_context_columns: Sequence[str],
) -> tuple[object, ...]:
    return tuple(row[column] for column in base_context_columns)


def compare_refinements_to_base(
    tail_laws: pd.DataFrame,
    *,
    base_context_columns: Sequence[str] = BASE_CONTEXT_COLUMNS,
) -> pd.DataFrame:
    """Compare each refinement family against the corresponding base context."""
    base = tail_laws[tail_laws["context_family"].eq("base")].copy()
    if base.empty:
        raise ValueError("Refinement comparison requires base context rows.")
    base_by_key = {
        _base_context_key(row, base_context_columns=base_context_columns): row
        for _index, row in base.iterrows()
    }
    rows: list[dict[str, object]] = []
    for context_family, group in tail_laws[~tail_laws["context_family"].eq("base")].groupby(
        "context_family", dropna=False
    ):
        for context_key, refined_group in group.groupby(list(base_context_columns), dropna=False):
            if not isinstance(context_key, tuple):
                context_key = (context_key,)
            base_row = base_by_key[context_key]
            descriptive = refined_group[
                refined_group["tail_law_status"].eq("descriptive_holdout_tail_law")
            ]
            admissible = refined_group[refined_group["production_tail_law_admissible"].astype(bool)]
            diagnostic_passed = refined_group[
                refined_group["diagnostic_tail_law_contract_passed"].astype(bool)
            ]
            precision_pass = descriptive[_tail_precision_pass_mask(descriptive)]
            base_se = float(base_row["heldout_exceedance_standard_error"])
            best_refined_se = (
                float(descriptive["heldout_exceedance_standard_error"].min())
                if not descriptive.empty
                else np.nan
            )
            if not diagnostic_passed.empty:
                interpretation = "refinement_has_diagnostic_contract_pass_not_production"
            elif descriptive.empty:
                interpretation = "refinement_fragmented_no_valid_tail_law"
            elif not precision_pass.empty:
                interpretation = "refinement_has_precision_pass_without_required_support"
            elif (
                np.isfinite(base_se) and np.isfinite(best_refined_se) and best_refined_se < base_se
            ):
                interpretation = "refinement_reduces_best_se_not_production"
            else:
                interpretation = "no_tail_precision_improvement"

            row = {column: value for column, value in zip(base_context_columns, context_key)}
            row.update(
                {
                    "context_family": context_family,
                    "base_n_matching_simulations": int(base_row["n_matching_simulations"]),
                    "base_n_records": int(base_row["n_records"]),
                    "base_heldout_exceedance_rate": float(base_row["heldout_exceedance_rate"]),
                    "base_heldout_exceedance_standard_error": base_se,
                    "base_production_tail_law_admissible": bool(
                        base_row["production_tail_law_admissible"]
                    ),
                    "n_refined_contexts": int(refined_group.shape[0]),
                    "n_refined_descriptive_contexts": int(descriptive.shape[0]),
                    "n_refined_precision_pass_contexts": int(precision_pass.shape[0]),
                    "n_refined_diagnostic_tail_law_contract_passed_contexts": int(
                        diagnostic_passed.shape[0]
                    ),
                    "n_refined_production_admissible_contexts": int(admissible.shape[0]),
                    "best_refined_heldout_exceedance_standard_error": best_refined_se,
                    "best_refined_standard_error_delta": (
                        float(base_se - best_refined_se)
                        if np.isfinite(base_se) and np.isfinite(best_refined_se)
                        else np.nan
                    ),
                    "interpretation": interpretation,
                    "study_role": STUDY_ROLE,
                }
            )
            rows.append(row)
    return (
        pd.DataFrame.from_records(rows)
        .sort_values(["context_family", *base_context_columns])
        .reset_index(drop=True)
    )


def run_topology_refinement_diagnostic(
    *,
    records_inputs: Sequence[RecordsInput],
    output_dir: Path,
    alpha: float = 0.01,
    n_folds: int = 5,
    min_train_simulations: int = 20,
    min_train_records: int = 20,
    required_min_matching_simulations: int = 499,
    required_min_matched_records: int = 499,
    max_exceedance_standard_error: float = 0.002,
) -> dict[str, pd.DataFrame]:
    """Write topology-refined selected-tail diagnostic tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_selected_geometry_records(records_inputs)
    refined_tail_law = evaluate_topology_refined_tail_laws(
        records,
        alpha=alpha,
        n_folds=n_folds,
        min_train_simulations=min_train_simulations,
        min_train_records=min_train_records,
        required_min_matching_simulations=required_min_matching_simulations,
        required_min_matched_records=required_min_matched_records,
        max_exceedance_standard_error=max_exceedance_standard_error,
    )
    refinement_summary = summarize_refinement_families(refined_tail_law)
    base_context_refinement_comparison = compare_refinements_to_base(
        refined_tail_law,
        base_context_columns=BASE_CONTEXT_COLUMNS,
    )
    outputs = {
        "refined_tail_law": refined_tail_law,
        "refinement_summary": refinement_summary,
        "base_context_refinement_comparison": base_context_refinement_comparison,
    }
    for name, table in outputs.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
    manifest = {
        "diagnostic": "selected_tail_topology_refinement",
        "study_role": STUDY_ROLE,
        "records_inputs": [
            {"run_id": records_input.run_id, "path": str(records_input.path)}
            for records_input in records_inputs
        ],
        "base_context_columns": list(BASE_CONTEXT_COLUMNS),
        "refinement_contexts": {
            context_family: list(additional_columns)
            for context_family, additional_columns in REFINEMENT_CONTEXTS
        },
        "bin_specs": [
            {"source_column": source, "target_column": target, "n_bins": int(n_bins)}
            for source, target, n_bins in BIN_SPECS
        ],
        "alpha": float(alpha),
        "n_folds": int(n_folds),
        "min_train_simulations": int(min_train_simulations),
        "min_train_records": int(min_train_records),
        "required_min_matching_simulations": int(required_min_matching_simulations),
        "required_min_matched_records": int(required_min_matched_records),
        "max_exceedance_standard_error": float(max_exceedance_standard_error),
        "edge_action_bin_labels": list(EDGE_ACTION_BIN_LABELS),
        "outputs": {name: str(output_dir / f"{name}.csv") for name in outputs},
        "note": (
            "Diagnostic-only comparison of base selected-tail contexts against "
            "topology, merge-persistence, edge-path, and spectral-alignment "
            "refinements. Refined contexts use data-adaptive bins learned from "
            "the diagnostic panel. They can pass diagnostic support checks but "
            "are never production-admissible calibration contexts."
        ),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether topology-aware context refinements improve "
            "held-out selected-tail homogeneity."
        )
    )
    parser.add_argument(
        "--records-csv",
        action="append",
        required=True,
        help="Record spec in the form run_id=/path/to/selected_geometry_records.csv.",
    )
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--min-train-simulations", type=int, default=20)
    parser.add_argument("--min-train-records", type=int, default=20)
    parser.add_argument("--required-min-matching-simulations", type=int, default=499)
    parser.add_argument("--required-min-matched-records", type=int, default=499)
    parser.add_argument("--max-exceedance-standard-error", type=float, default=0.002)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            Path("benchmarks")
            / "results"
            / f"selected_tail_topology_refinement_{format_timestamp_utc()}"
        )
    outputs = run_topology_refinement_diagnostic(
        records_inputs=parse_records_specs(list(args.records_csv)),
        output_dir=output_dir,
        alpha=float(args.alpha),
        n_folds=int(args.n_folds),
        min_train_simulations=int(args.min_train_simulations),
        min_train_records=int(args.min_train_records),
        required_min_matching_simulations=int(args.required_min_matching_simulations),
        required_min_matched_records=int(args.required_min_matched_records),
        max_exceedance_standard_error=float(args.max_exceedance_standard_error),
    )
    print(outputs["refinement_summary"].to_string(index=False))
    print(outputs["base_context_refinement_comparison"].to_string(index=False))
    print(f"Wrote topology-refinement diagnostic outputs to {output_dir}")


if __name__ == "__main__":
    main()
