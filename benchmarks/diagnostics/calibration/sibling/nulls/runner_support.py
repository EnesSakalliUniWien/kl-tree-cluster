"""Shared preparation for sibling-null diagnostic runners."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd
from tree_break_selection.hierarchy_analysis.decomposition.gates.annotation_bundle import (
    GateAnnotationBundle,
)
from tree_break_selection.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_SIBLING_ALPHA,
)

from benchmarks.diagnostics.calibration.sibling.nulls.sibling_inflation_diagnostic import (
    SiblingInflationDiagnosticTables,
    build_sibling_inflation_diagnostic_tables,
    collect_sibling_inflation_inputs,
)
from benchmarks.diagnostics.oracle.gate_path_trace import (
    build_prepared_gate_path_trace,
    prepare_gate_path_case,
)
from benchmarks.shared.tbs_tree_context import TbsTreeContext


@dataclass(frozen=True)
class PreparedSiblingDiagnostic:
    """Canonical tree, gate annotations, and diagnostic tables for one case."""

    context: TbsTreeContext
    gate_annotation_bundle: GateAnnotationBundle
    tables: SiblingInflationDiagnosticTables


def prepare_sibling_diagnostic(
    case: dict[str, object],
    classification: Mapping[str, object],
    *,
    max_contributors: int,
    contributors_for_all_crossings: bool,
) -> PreparedSiblingDiagnostic:
    """Build the shared tree/gate/oracle context used by sibling-null studies."""

    gate_path_case = prepare_gate_path_case(case)
    context = gate_path_case.context
    gate_annotation_bundle = gate_path_case.gate_annotation_bundle
    trace_df = build_prepared_gate_path_trace(
        gate_path_case,
        classification,
        sibling_inflation_trace_by_parent={},
    )

    inputs = collect_sibling_inflation_inputs(
        context.tree,
        gate_annotation_bundle,
        feature_space=context.feature_space,
    )
    if inputs.model is None:
        raise ValueError(
            f"Case {context.metadata['name']!r} has no focal sibling records to diagnose."
        )
    raw_tables = build_sibling_inflation_diagnostic_tables(
        records=inputs.records,
        model=inputs.model,
        trace_df=trace_df,
        sibling_alpha=DEFAULT_SIBLING_ALPHA,
        max_contributors=max_contributors,
        contributors_for_all_crossings=contributors_for_all_crossings,
    )
    tables = SiblingInflationDiagnosticTables(
        targets=_with_tree_geometry(raw_tables.targets, context),
        contributors=_with_tree_geometry(raw_tables.contributors, context),
        summary=_with_tree_geometry(raw_tables.summary, context),
    )
    return PreparedSiblingDiagnostic(
        context=context,
        gate_annotation_bundle=gate_annotation_bundle,
        tables=tables,
    )


def select_sibling_target_rows(
    targets: pd.DataFrame,
    *,
    target_mode: str,
) -> pd.DataFrame:
    """Select blocker, currently blocked, or all focal sibling rows."""

    if target_mode == "blockers":
        selected = targets[targets["blocker_candidate"].astype(bool)].copy()
    elif target_mode == "current_blocks":
        selected = targets[targets["current_blocks_at_alpha"].astype(bool)].copy()
    elif target_mode == "all_focal":
        selected = targets.copy()
    else:
        raise ValueError(f"Unknown target_mode={target_mode!r}.")
    if selected.empty:
        raise ValueError(f"No sibling targets matched target_mode={target_mode!r}.")
    return selected


def standardized_contrast_dimension(context: TbsTreeContext) -> int:
    """Return the active contrast dimension for a prepared benchmark tree."""

    if context.feature_space is None:
        return int(context.data.shape[1])
    return int(context.feature_space.contrast_dimension)


def _with_tree_geometry(frame: pd.DataFrame, context: TbsTreeContext) -> pd.DataFrame:
    enriched = frame.copy()
    if enriched.empty:
        return enriched
    enriched.insert(2, "tree_distance_metric", context.tree_distance_metric)
    enriched.insert(3, "tree_distance_source", context.tree_distance_source)
    enriched.insert(4, "tree_linkage_method", context.tree_linkage_method)
    return enriched


__all__ = [
    "PreparedSiblingDiagnostic",
    "prepare_sibling_diagnostic",
    "select_sibling_target_rows",
    "standardized_contrast_dimension",
]
