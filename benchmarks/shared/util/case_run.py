"""Single-case orchestration helper for benchmark pipeline runs."""

from __future__ import annotations

import os
from pathlib import Path

from benchmarks.shared.audit_utils import (
    export_case_and_method_matrix_audits,
    export_decomposition_audit,
)
from benchmarks.shared.logging import log_test_case_start as _log_test_case_start
from benchmarks.shared.results import BenchmarkResultRow, ComputedResultRecord
from benchmarks.shared.runners.method_registry import METHOD_SPECS
from benchmarks.shared.util.case_inputs import prepare_case_inputs
from benchmarks.shared.util.method_execution import run_single_method_once


def run_single_case(
    *,
    tc: dict[str, object],
    case_position: int,
    total_cases: int,
    selected_methods: list[str],
    param_sets: dict[str, list[dict[str, object]]],
    significance_level: float,
    output_pdf: Path | None,
    plots_root: Path,
    matrix_audit: bool,
    verbose: bool,
) -> tuple[list[BenchmarkResultRow], list[ComputedResultRecord]]:
    """Run one benchmark case across all selected methods and params."""
    case_idx = tc.get("test_case_num", case_position)
    case_name = tc.get("name", f"Case {case_idx}")
    if verbose:
        _log_test_case_start(case_idx, total_cases, case_name)

    if output_pdf:
        audit_root_inc = output_pdf.parent
        if audit_root_inc.name == "plots":
            audit_root_inc = audit_root_inc.parent
    else:
        audit_root_inc = plots_root.parent if plots_root.name == "plots" else plots_root
    previous_audit_root = os.environ.get("KL_TE_MATRIX_AUDIT_ROOT")
    if matrix_audit:
        os.environ["KL_TE_MATRIX_AUDIT_ROOT"] = str(audit_root_inc)
    try:
        # Generate and process data/distances.
        (
            data_t,
            y_t,
            x_original,
            meta,
            distance_condensed,
            distance_matrix,
            precomputed_distance_condensed,
        ) = prepare_case_inputs(tc, selected_methods)

        method_audits: list[tuple[str, dict[str, object]]] = []
        case_result_rows: list[BenchmarkResultRow] = []
        case_computed_results: list[ComputedResultRecord] = []
        for method_id in selected_methods:
            spec = METHOD_SPECS[method_id]
            params_list = param_sets[method_id]
            for params in params_list:
                result_row, computed_result, method_audit = run_single_method_once(
                    method_id=method_id,
                    spec=spec,
                    params=params,
                    case_idx=case_idx,
                    case_name=case_name,
                    tc_seed=tc.get("seed"),
                    significance_level=significance_level,
                    data_t=data_t,
                    y_t=y_t,
                    x_original=x_original,
                    meta=meta,
                    distance_matrix=distance_matrix,
                    distance_condensed=distance_condensed,
                    precomputed_distance_condensed=precomputed_distance_condensed,
                    matrix_audit=matrix_audit,
                )
                case_result_rows.append(result_row)
                if computed_result is not None:
                    case_computed_results.append(computed_result)
                if method_audit is not None:
                    method_audits.append(method_audit)

        # Export detailed audit logs incrementally so each case is durable.
        if matrix_audit:
            export_case_and_method_matrix_audits(
                case_idx=case_idx,
                case_name=case_name,
                data_matrix=data_t.values,
                y_true=y_t,
                x_original=x_original,
                distance_matrix=distance_matrix,
                distance_condensed=distance_condensed,
                meta=meta,
                method_audits=method_audits,
                output_root=audit_root_inc,
                verbose=verbose,
            )

        export_decomposition_audit(
            computed_results=case_computed_results,
            output_root=audit_root_inc,
            verbose=verbose,
        )

        return case_result_rows, case_computed_results
    finally:
        if matrix_audit:
            if previous_audit_root is None:
                os.environ.pop("KL_TE_MATRIX_AUDIT_ROOT", None)
            else:
                os.environ["KL_TE_MATRIX_AUDIT_ROOT"] = previous_audit_root


__all__ = ["run_single_case"]
