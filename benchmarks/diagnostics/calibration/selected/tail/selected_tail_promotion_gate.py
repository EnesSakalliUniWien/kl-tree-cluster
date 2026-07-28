"""Selected-tail external calibration promotion gate.

This diagnostic combines the Q1/Q5/Q7/Q8 evidence surfaces:

* context-level selected-tail support/admissibility rows,
* Q5 barycentric selected-tail validation,
* optional external c-hat precision metadata.

It writes an explicit promotion decision table. The script is diagnostic-only:
it does not enable a production external calibration branch.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.diagnostics.calibration.selected.tail.selected_tail_admissibility_domain import (
    SelectedTailRun,
    parse_run_specs,
)
from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "diagnostic_selected_tail_external_promotion_gate_not_calibration"
DEFAULT_Q5_MODEL_ID = "q5_barycentric_edge_spectral"
DEFAULT_MAX_Q5_SPLIT_RESIDUAL_TAIL_ABS_ERROR = 0.005
DEFAULT_MAX_CONTEXT_HELDOUT_TAIL_ABS_ERROR = 0.005
DEFAULT_MAX_RELATIVE_C_SIMULATION_SE = 0.05
CONTEXT_COLUMNS = (
    "source_family",
    "feature_family",
    "parent_size_bin",
    "sibling_projection_dimension",
    "edge_action_bin",
)
TAIL_REQUIRED_COLUMNS = set(CONTEXT_COLUMNS) | {
    "alpha",
    "n_records",
    "n_matching_simulations",
    "required_min_matching_simulations",
    "required_min_matched_records",
    "max_exceedance_standard_error",
    "production_tail_law_admissible",
    "tail_law_admissibility_failure_reasons",
    "heldout_exceedance_rate",
    "heldout_exceedance_absolute_error",
    "heldout_exceedance_standard_error",
    "tail_law_status",
}
Q5_REQUIRED_COLUMNS = {
    "model_id",
    "split_strategy",
    "model_status",
    "residual_tail_exceedance_absolute_error",
    "residual_tail_exceedance_standard_error",
}


@dataclass(frozen=True)
class Q5PromotionGate:
    """Global Q5 selected-tail model promotion gate."""

    model_id: str
    max_split_residual_tail_abs_error: float
    q5_global_gate_pass: bool
    failure_reasons: tuple[str, ...]


def _validate_tail_law_table(table: pd.DataFrame, *, path: Path) -> None:
    missing = TAIL_REQUIRED_COLUMNS - set(table.columns)
    if missing:
        raise ValueError(f"Selected-tail law table {path} is missing columns: {sorted(missing)!r}.")


def _load_tail_law_runs(runs: tuple[SelectedTailRun, ...]) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    for run in runs:
        table = pd.read_csv(run.path)
        _validate_tail_law_table(table, path=run.path)
        table = table.copy()
        table.insert(0, "run_id", run.run_id)
        table.insert(1, "selected_tail_law_csv", str(run.path))
        tables.append(table)
    if not tables:
        raise ValueError("At least one selected-tail law run is required.")
    return pd.concat(tables, ignore_index=True)


def evaluate_q5_promotion_gate(
    q5_validation: pd.DataFrame,
    *,
    model_id: str = DEFAULT_Q5_MODEL_ID,
    max_split_residual_tail_abs_error: float = (DEFAULT_MAX_Q5_SPLIT_RESIDUAL_TAIL_ABS_ERROR),
) -> tuple[pd.DataFrame, Q5PromotionGate]:
    """Return split-level Q5 promotion diagnostics and the global gate."""
    missing = Q5_REQUIRED_COLUMNS - set(q5_validation.columns)
    if missing:
        raise ValueError(f"Q5 validation table is missing columns: {sorted(missing)!r}.")
    if max_split_residual_tail_abs_error < 0.0:
        raise ValueError("max_split_residual_tail_abs_error cannot be negative.")

    rows = q5_validation[q5_validation["model_id"].astype(str).eq(model_id)].copy()
    if rows.empty:
        gate = Q5PromotionGate(
            model_id=model_id,
            max_split_residual_tail_abs_error=float(max_split_residual_tail_abs_error),
            q5_global_gate_pass=False,
            failure_reasons=(f"missing_q5_model:{model_id}",),
        )
        return pd.DataFrame(), gate

    rows["residual_tail_exceedance_absolute_error"] = pd.to_numeric(
        rows["residual_tail_exceedance_absolute_error"],
        errors="coerce",
    )
    rows["q5_split_gate_pass"] = (
        rows["model_status"].astype(str).str.startswith("diagnostic_holdout")
        & np.isfinite(rows["residual_tail_exceedance_absolute_error"])
        & (
            rows["residual_tail_exceedance_absolute_error"]
            <= float(max_split_residual_tail_abs_error)
        )
    )
    rows["q5_split_failure_reason"] = ""
    bad_status = ~rows["model_status"].astype(str).str.startswith("diagnostic_holdout")
    rows.loc[bad_status, "q5_split_failure_reason"] = "no_valid_holdout_split"
    bad_error = np.isfinite(rows["residual_tail_exceedance_absolute_error"]) & (
        rows["residual_tail_exceedance_absolute_error"] > float(max_split_residual_tail_abs_error)
    )
    rows.loc[bad_error, "q5_split_failure_reason"] = "residual_tail_abs_error_above_gate"
    missing_error = ~np.isfinite(rows["residual_tail_exceedance_absolute_error"])
    rows.loc[missing_error, "q5_split_failure_reason"] = "missing_residual_tail_abs_error"
    failure_reasons = tuple(
        f"{row.split_strategy}:{row.q5_split_failure_reason}"
        for row in rows.itertuples(index=False)
        if not bool(row.q5_split_gate_pass)
    )
    gate = Q5PromotionGate(
        model_id=model_id,
        max_split_residual_tail_abs_error=float(max_split_residual_tail_abs_error),
        q5_global_gate_pass=not failure_reasons,
        failure_reasons=failure_reasons,
    )
    output_columns = [
        "model_id",
        "split_strategy",
        "model_status",
        "residual_tail_exceedance_rate",
        "residual_tail_exceedance_absolute_error",
        "residual_tail_exceedance_standard_error",
        "holdout_tail_auc_from_linear_score",
        "holdout_log_ratio_r_squared",
        "q5_split_gate_pass",
        "q5_split_failure_reason",
    ]
    return rows[[column for column in output_columns if column in rows.columns]], gate


def _support_contract_met(row: pd.Series) -> bool:
    return bool(
        int(row["n_matching_simulations"]) >= int(row["required_min_matching_simulations"])
        and int(row["n_records"]) >= int(row["required_min_matched_records"])
    )


def _context_tail_precision_met(row: pd.Series) -> bool:
    heldout_se = float(row["heldout_exceedance_standard_error"])
    max_se = float(row["max_exceedance_standard_error"])
    return bool(np.isfinite(heldout_se) and heldout_se <= max_se)


def _context_tail_abs_error_met(
    row: pd.Series,
    *,
    max_context_heldout_tail_abs_error: float,
) -> bool:
    absolute_error = float(row["heldout_exceedance_absolute_error"])
    return bool(
        np.isfinite(absolute_error) and absolute_error <= float(max_context_heldout_tail_abs_error)
    )


def _relative_c_precision_status(
    row: pd.Series,
    *,
    require_c_hat_precision: bool,
    max_relative_c_simulation_se: float,
) -> tuple[bool, str]:
    if not require_c_hat_precision:
        return True, "not_required"
    column = "selected_hierarchy_c_hat_relative_simulation_se"
    if column not in row.index:
        return False, "missing_relative_c_simulation_se"
    value = float(row[column])
    if not np.isfinite(value):
        return False, "missing_relative_c_simulation_se"
    if value > float(max_relative_c_simulation_se):
        return False, "relative_c_simulation_se_above_gate"
    return True, "relative_c_simulation_se_passed"


def _failure_reasons(
    *,
    support_contract_met: bool,
    context_tail_precision_met: bool,
    context_tail_abs_error_met: bool,
    production_tail_law_admissible: bool,
    q5_gate: Q5PromotionGate,
    c_hat_precision_met: bool,
    c_hat_precision_status: str,
) -> str:
    reasons: list[str] = []
    if not support_contract_met:
        reasons.append("selected_tail_support_contract_failed")
    if not context_tail_precision_met:
        reasons.append("selected_tail_precision_contract_failed")
    if not context_tail_abs_error_met:
        reasons.append("selected_tail_abs_error_contract_failed")
    if not production_tail_law_admissible:
        reasons.append("context_tail_law_not_admissible")
    if not q5_gate.q5_global_gate_pass:
        reasons.append("q5_global_gate_failed")
        reasons.extend(q5_gate.failure_reasons)
    if not c_hat_precision_met:
        reasons.append(c_hat_precision_status)
    return ";".join(dict.fromkeys(reasons))


def evaluate_selected_tail_promotion_gate(
    tail_law: pd.DataFrame,
    q5_validation: pd.DataFrame,
    *,
    q5_model_id: str = DEFAULT_Q5_MODEL_ID,
    max_q5_split_residual_tail_abs_error: float = (DEFAULT_MAX_Q5_SPLIT_RESIDUAL_TAIL_ABS_ERROR),
    max_context_heldout_tail_abs_error: float = (DEFAULT_MAX_CONTEXT_HELDOUT_TAIL_ABS_ERROR),
    require_c_hat_precision: bool = True,
    max_relative_c_simulation_se: float = DEFAULT_MAX_RELATIVE_C_SIMULATION_SE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate external selected-tail promotion decisions."""
    q5_gate_table, q5_gate = evaluate_q5_promotion_gate(
        q5_validation,
        model_id=q5_model_id,
        max_split_residual_tail_abs_error=max_q5_split_residual_tail_abs_error,
    )
    table = tail_law.copy()
    for column in (
        "n_records",
        "n_matching_simulations",
        "required_min_matching_simulations",
        "required_min_matched_records",
    ):
        table[column] = pd.to_numeric(table[column], errors="raise").astype(int)
    for column in (
        "heldout_exceedance_rate",
        "heldout_exceedance_absolute_error",
        "heldout_exceedance_standard_error",
        "max_exceedance_standard_error",
    ):
        table[column] = pd.to_numeric(table[column], errors="coerce").astype(float)

    rows: list[dict[str, object]] = []
    for row in table.itertuples(index=False):
        series = pd.Series(row._asdict())
        support_met = _support_contract_met(series)
        precision_met = _context_tail_precision_met(series)
        abs_error_met = _context_tail_abs_error_met(
            series,
            max_context_heldout_tail_abs_error=max_context_heldout_tail_abs_error,
        )
        production_tail_law_admissible = bool(series["production_tail_law_admissible"])
        c_hat_met, c_hat_status = _relative_c_precision_status(
            series,
            require_c_hat_precision=require_c_hat_precision,
            max_relative_c_simulation_se=max_relative_c_simulation_se,
        )
        if not support_met:
            decision = "undefined_support_failure"
        elif (
            production_tail_law_admissible
            and precision_met
            and abs_error_met
            and q5_gate.q5_global_gate_pass
            and c_hat_met
        ):
            decision = "external_admissible"
        else:
            decision = "external_diagnostic_only"
        failure_reasons = _failure_reasons(
            support_contract_met=support_met,
            context_tail_precision_met=precision_met,
            context_tail_abs_error_met=abs_error_met,
            production_tail_law_admissible=production_tail_law_admissible,
            q5_gate=q5_gate,
            c_hat_precision_met=c_hat_met,
            c_hat_precision_status=c_hat_status,
        )
        output = series.to_dict()
        output.update(
            {
                "support_contract_met": support_met,
                "context_tail_precision_met": precision_met,
                "context_tail_abs_error_met": abs_error_met,
                "q5_model_id": q5_gate.model_id,
                "q5_global_gate_pass": q5_gate.q5_global_gate_pass,
                "q5_global_failure_reasons": ";".join(q5_gate.failure_reasons),
                "require_c_hat_precision": bool(require_c_hat_precision),
                "c_hat_precision_met": c_hat_met,
                "c_hat_precision_status": c_hat_status,
                "promotion_decision": decision,
                "promotion_failure_reasons": failure_reasons,
                "study_role": STUDY_ROLE,
            }
        )
        rows.append(output)
    promotion = pd.DataFrame.from_records(rows)
    summary = summarize_promotion_decisions(promotion)
    return promotion, q5_gate_table, summary


def summarize_promotion_decisions(promotion: pd.DataFrame) -> pd.DataFrame:
    """Summarize promotion gate decisions."""
    grouped = promotion.groupby("promotion_decision", dropna=False)
    rows = []
    for decision, group in grouped:
        rows.append(
            {
                "promotion_decision": str(decision),
                "n_contexts": int(group.shape[0]),
                "n_support_contract_met": int(group["support_contract_met"].sum()),
                "n_context_tail_law_admissible": int(
                    group["production_tail_law_admissible"].astype(bool).sum()
                ),
                "n_q5_global_gate_pass": int(group["q5_global_gate_pass"].sum()),
                "n_c_hat_precision_met": int(group["c_hat_precision_met"].sum()),
                "study_role": STUDY_ROLE,
            }
        )
    return pd.DataFrame.from_records(rows).sort_values("promotion_decision")


def run_selected_tail_promotion_gate(
    *,
    runs: tuple[SelectedTailRun, ...],
    q5_validation_path: Path,
    output_dir: Path,
    q5_model_id: str = DEFAULT_Q5_MODEL_ID,
    max_q5_split_residual_tail_abs_error: float = (DEFAULT_MAX_Q5_SPLIT_RESIDUAL_TAIL_ABS_ERROR),
    max_context_heldout_tail_abs_error: float = (DEFAULT_MAX_CONTEXT_HELDOUT_TAIL_ABS_ERROR),
    require_c_hat_precision: bool = True,
    max_relative_c_simulation_se: float = DEFAULT_MAX_RELATIVE_C_SIMULATION_SE,
) -> dict[str, Path]:
    """Write selected-tail promotion gate outputs."""
    tail_law = _load_tail_law_runs(runs)
    q5_validation = pd.read_csv(q5_validation_path)
    promotion, q5_gate_table, summary = evaluate_selected_tail_promotion_gate(
        tail_law,
        q5_validation,
        q5_model_id=q5_model_id,
        max_q5_split_residual_tail_abs_error=max_q5_split_residual_tail_abs_error,
        max_context_heldout_tail_abs_error=max_context_heldout_tail_abs_error,
        require_c_hat_precision=require_c_hat_precision,
        max_relative_c_simulation_se=max_relative_c_simulation_se,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    context_path = output_dir / "selected_tail_promotion_contexts.csv"
    q5_path = output_dir / "selected_tail_q5_promotion_gate.csv"
    summary_path = output_dir / "selected_tail_promotion_summary.csv"
    manifest_path = output_dir / "manifest.json"
    promotion.to_csv(context_path, index=False)
    q5_gate_table.to_csv(q5_path, index=False)
    summary.to_csv(summary_path, index=False)
    manifest: Mapping[str, object] = {
        "created_at_utc": format_timestamp_utc(),
        "study_role": STUDY_ROLE,
        "run_specs": [
            {"run_id": run.run_id, "selected_tail_law_csv": str(run.path)} for run in runs
        ],
        "q5_validation_path": str(q5_validation_path),
        "q5_model_id": q5_model_id,
        "max_q5_split_residual_tail_abs_error": float(max_q5_split_residual_tail_abs_error),
        "max_context_heldout_tail_abs_error": float(max_context_heldout_tail_abs_error),
        "require_c_hat_precision": bool(require_c_hat_precision),
        "max_relative_c_simulation_se": float(max_relative_c_simulation_se),
        "outputs": {
            "contexts": str(context_path),
            "q5_gate": str(q5_path),
            "summary": str(summary_path),
        },
        "interpretation": (
            "Diagnostic promotion gate only. Rows with external_admissible would "
            "identify contexts eligible for a future external CalibrationDecision "
            "branch; this script does not change production calibration."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "contexts": context_path,
        "q5_gate": q5_path,
        "summary": summary_path,
        "manifest": manifest_path,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-spec",
        action="append",
        required=True,
        help="Run spec in the form run_id=/path/to/selected_ratio_tail_law.csv.",
    )
    parser.add_argument("--q5-validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--q5-model-id", default=DEFAULT_Q5_MODEL_ID)
    parser.add_argument(
        "--max-q5-split-residual-tail-abs-error",
        type=float,
        default=DEFAULT_MAX_Q5_SPLIT_RESIDUAL_TAIL_ABS_ERROR,
    )
    parser.add_argument(
        "--max-context-heldout-tail-abs-error",
        type=float,
        default=DEFAULT_MAX_CONTEXT_HELDOUT_TAIL_ABS_ERROR,
    )
    parser.add_argument(
        "--allow-missing-c-hat-precision",
        action="store_true",
        help=(
            "Do not block promotion when selected_hierarchy_c_hat_relative_"
            "simulation_se is absent. This should only be used for sensitivity "
            "diagnostics, not the production promotion gate."
        ),
    )
    parser.add_argument(
        "--max-relative-c-simulation-se",
        type=float,
        default=DEFAULT_MAX_RELATIVE_C_SIMULATION_SE,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    outputs = run_selected_tail_promotion_gate(
        runs=parse_run_specs(list(args.run_spec)),
        q5_validation_path=args.q5_validation,
        output_dir=args.output_dir,
        q5_model_id=str(args.q5_model_id),
        max_q5_split_residual_tail_abs_error=float(args.max_q5_split_residual_tail_abs_error),
        max_context_heldout_tail_abs_error=float(args.max_context_heldout_tail_abs_error),
        require_c_hat_precision=not bool(args.allow_missing_c_hat_precision),
        max_relative_c_simulation_se=float(args.max_relative_c_simulation_se),
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_Q5_MODEL_ID",
    "STUDY_ROLE",
    "evaluate_q5_promotion_gate",
    "evaluate_selected_tail_promotion_gate",
    "run_selected_tail_promotion_gate",
    "summarize_promotion_decisions",
]
