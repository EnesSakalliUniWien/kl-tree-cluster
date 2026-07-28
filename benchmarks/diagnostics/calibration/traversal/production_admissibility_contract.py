"""Production-admissibility contract diagnostic.

This module combines diagnostic component statuses into explicit production
decisions. It is intentionally conservative: missing or diagnostic-only
components remain fail-closed unless every required component is marked
production-ready by predeclared criteria.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "diagnostic_production_admissibility_contract_not_calibration"
SCHEMA_VERSION = "production_admissibility_contract/v1"
REQUIRED_COLUMNS = {
    "contract_id",
    "component_id",
    "component_type",
    "component_status",
    "required_for_production",
}
PRODUCTION_READY_STATUSES = frozenset(
    {
        "production_ready",
        "within_nominal_tolerance",
        "passes_internal_support_thresholds",
        "external_admissible",
        "guard_validation_candidate",
    }
)
FAIL_CLOSED_STATUSES = frozenset(
    {
        "fail_closed_undefined",
        "above_nominal_tolerance",
        "below_nominal_tolerance",
        "below_internal_support_thresholds",
        "data_independent_gate_insufficient_rows",
        "data_independent_gate_null_inflated",
        "data_independent_gate_penalty_insufficient_coverage",
        "data_independent_gate_penalty_null_inflated",
        "data_independent_gate_penalty_signal_weak",
        "data_independent_gate_signal_weak",
        "data_independent_traversal_insufficient_rows",
        "data_independent_traversal_null_inflated",
        "data_independent_traversal_signal_weak",
        "data_independent_traversal_transfer_confidence_insufficient",
        "data_independent_traversal_transfer_confidence_null_uncertain",
        "data_independent_traversal_transfer_confidence_signal_uncertain",
        "data_independent_traversal_transfer_insufficient_coverage",
        "data_independent_traversal_transfer_null_inflated",
        "data_independent_traversal_transfer_signal_weak",
        "external_selected_tail_fail_closed",
        "fixed_profile_adaptive_projection_detected",
        "fixed_profile_confidence_null_uncertain",
        "fixed_profile_confidence_signal_uncertain",
        "fixed_profile_insufficient_coverage",
        "fixed_profile_null_inflated",
        "fixed_profile_signal_weak",
        "undefined_external_not_admissible",
        "undefined_sparse_context",
        "insufficient_null_support",
        "undefined_no_matched_null_context",
        "df_skew_explains_skew_but_tail_misaligned",
        "nonsmooth_selection_geometry",
        "null_law_adaptive_projection_tail_inflated",
        "null_law_fixed_projection_tail_misaligned",
        "null_law_insufficient_rows",
        "projection_unstable",
        "regularized_boundary_unstable",
        "regularized_insufficient_rows",
        "regularized_projection_unstable",
        "regularized_selected_tree_not_yet_calibrated",
        "regularized_tail_misaligned",
        "selection_coupled",
        "skew_exceeds_df_reference",
        "tail_misaligned",
        "tail_misaligned_after_differential_checks",
        "wald_metric_boundary_unstable",
        "whitening_unstable",
    }
)
DIAGNOSTIC_ONLY_STATUSES = frozenset(
    {
        "chi_square_shape_candidate",
        "conditional_empirical_p_value",
        "data_independent_gate_null_candidate",
        "data_independent_gate_penalty_transfer_candidate",
        "data_independent_gate_signal_retained",
        "data_independent_traversal_null_candidate",
        "data_independent_traversal_signal_retained",
        "data_independent_traversal_transfer_confidence_candidate",
        "data_independent_traversal_transfer_candidate",
        "diagnostic_only",
        "diagnostic_only_guard",
        "fixed_subspace_candidate",
        "null_law_fixed_projection_candidate",
        "regularized_fixed_tree_candidate",
        "laplacian_connected_covariance",
        "laplacian_diagonal_covariance",
        "laplacian_disconnected_covariance",
        "laplacian_invalid_spectral_context",
        "laplacian_missing_spectral_context",
        "laplacian_near_disconnected_covariance",
        "laplacian_skipped_high_dimension",
        "laplacian_unavailable",
        "signal_retention_descriptive",
        "selected_nonnull_retention_descriptive",
        "external_selected_tail_candidate_descriptive",
        "fixed_profile_adaptive_projection_avoided",
        "fixed_profile_confidence_candidate",
        "fixed_profile_null_candidate",
        "fixed_profile_signal_retained",
        "fixed_profile_transfer_candidate",
        "insufficient_rows",
        "insufficient_flagged_rows",
        "supported_context",
    }
)


def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    lowered = values.astype(str).str.strip().str.lower()
    return lowered.isin({"1", "true", "t", "yes"})


def evaluate_production_admissibility_components(components: pd.DataFrame) -> pd.DataFrame:
    """Normalize component statuses for production-admissibility decisions."""
    missing = REQUIRED_COLUMNS - set(components.columns)
    if missing:
        raise ValueError(
            f"Production-admissibility contract is missing columns: {sorted(missing)!r}."
        )
    table = components.copy()
    status = table["component_status"].astype(str)
    known = PRODUCTION_READY_STATUSES | FAIL_CLOSED_STATUSES | DIAGNOSTIC_ONLY_STATUSES
    unknown_statuses = sorted(set(status) - known)
    if unknown_statuses:
        raise ValueError(f"component_status contains unknown statuses: {unknown_statuses!r}.")
    required = _bool_series(table["required_for_production"])
    normalized = pd.DataFrame(
        {
            "contract_id": table["contract_id"].astype(str),
            "component_id": table["component_id"].astype(str),
            "component_type": table["component_type"].astype(str),
            "component_status": status,
            "required_for_production": required,
            "component_ready_for_production": status.isin(PRODUCTION_READY_STATUSES),
            "component_fail_closed": status.isin(FAIL_CLOSED_STATUSES),
            "component_diagnostic_only": status.isin(DIAGNOSTIC_ONLY_STATUSES),
            "study_role": STUDY_ROLE,
        }
    )
    for column in ("evidence_path", "notes"):
        if column in table.columns:
            normalized[column] = table[column].astype(str)
    return normalized


def summarize_production_admissibility_contracts(rows: pd.DataFrame) -> pd.DataFrame:
    """Return one explicit production decision per contract id."""
    if rows.empty:
        return pd.DataFrame()
    summaries: list[dict[str, object]] = []
    for contract_id, group in rows.groupby("contract_id", sort=True):
        required = group["required_for_production"].astype(bool)
        required_group = group[required]
        missing_required = required_group.empty
        required_fail_closed = required_group["component_fail_closed"].astype(bool)
        required_diagnostic = required_group["component_diagnostic_only"].astype(bool)
        required_ready = required_group["component_ready_for_production"].astype(bool)
        if missing_required:
            decision = "fail_closed_undefined"
            reasons = ("no_required_components",)
        elif bool(required_fail_closed.any()):
            decision = "fail_closed_undefined"
            reasons = tuple(
                required_group.loc[
                    required_fail_closed,
                    "component_id",
                ].astype(str)
            )
        elif bool(required_diagnostic.any()):
            decision = "diagnostic_only"
            reasons = tuple(
                required_group.loc[
                    required_diagnostic,
                    "component_id",
                ].astype(str)
            )
        elif bool(required_ready.all()):
            decision = "production_admissible"
            reasons = ()
        else:
            decision = "fail_closed_undefined"
            reasons = ("unclassified_required_component",)
        summaries.append(
            {
                "contract_id": contract_id,
                "n_components": int(group.shape[0]),
                "n_required_components": int(required.sum()),
                "n_required_ready": int(required_ready.sum()),
                "n_required_fail_closed": int(required_fail_closed.sum()),
                "n_required_diagnostic_only": int(required_diagnostic.sum()),
                "production_decision": decision,
                "blocking_component_ids": ";".join(reasons),
                "study_role": STUDY_ROLE,
            }
        )
    return pd.DataFrame.from_records(summaries)


def run_production_admissibility_contract(
    *,
    components_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Run the production-admissibility contract diagnostic from a CSV file."""
    components = pd.read_csv(components_path)
    rows = evaluate_production_admissibility_components(components)
    summary = summarize_production_admissibility_contracts(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "production_admissibility_components.csv"
    summary_path = output_dir / "production_admissibility_summary.csv"
    manifest_path = output_dir / "manifest.json"
    rows.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)
    manifest = {
        "created_at_utc": format_timestamp_utc(),
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "components_path": str(components_path),
        "outputs": {
            "components": str(rows_path),
            "summary": str(summary_path),
        },
        "interpretation": (
            "Diagnostic production-admissibility contract. A production decision "
            "requires every required component to be production-ready; diagnostic "
            "or failed components remain non-production and fail closed."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "components": rows_path,
        "summary": summary_path,
        "manifest": manifest_path,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    outputs = run_production_admissibility_contract(
        components_path=args.components,
        output_dir=args.output_dir,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "STUDY_ROLE",
    "evaluate_production_admissibility_components",
    "run_production_admissibility_contract",
    "summarize_production_admissibility_contracts",
]
