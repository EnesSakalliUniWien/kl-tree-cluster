"""Root binary-resolution strength diagnostic.

Tree-Break Selection always builds a binary hierarchy, so the root is always represented as

    r -> (L, R).

The statistical question is narrower: is that first binary split a calibrated
root event, or only one selected refinement of a weak/unresolved top-level
configuration? This panel records a descriptive root-resolution coordinate

    rho_r = T * min(A, E),

where T is selected tie-rank fraction, A is selected-ratio/action log1p, and E
is edge-margin/action log1p. The coordinate is not a p-value and does not rescue
splits. It is joined to the selected-root spectral-tail/equation status so the
method can fail closed when the binary root is uncalibrated.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from benchmarks.diagnostics.calibration.reporting import (
    print_diagnostic_output_paths,
    write_diagnostic_bundle,
)
from benchmarks.diagnostics.calibration.root.root_tail_values import finite_float, string_value
from benchmarks.diagnostics.calibration.root.selected.root_selected_spectral_tail_law_panel import (
    DEFAULT_RESULT_ROOT,
)

SCHEMA_VERSION = "root_selected_binary_resolution_panel/v1"
STUDY_ROLE = "diagnostic_root_selected_binary_resolution_not_calibration"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration.root.selected.root_selected_binary_resolution_panel"
)

DEFAULT_TAIL_ROWS = (
    DEFAULT_RESULT_ROOT
    / "root_selected_spectral_tail_law_deformed_hu_mild_replay_v3_smoke"
    / "root_selected_spectral_tail_law_rows.csv"
)
DEFAULT_EXTERNAL_LAW_EQUATION_ROWS = (
    DEFAULT_RESULT_ROOT
    / "root_selected_external_law_equation_mild_replay_v3_smoke"
    / "root_selected_external_law_equation_rows.csv"
)

ROWS_OUTPUT = "root_selected_binary_resolution_rows.csv"
SUMMARY_OUTPUT = "root_selected_binary_resolution_summary.csv"
MANIFEST_OUTPUT = "manifest.json"

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "target_case_id",
    "binary_root_model",
    "root_resolution_equation",
    "t_selected_tie_rank_fraction",
    "a_selected_ratio_action_log1p",
    "e_edge_margin_action_log1p",
    "root_action_edge_bottleneck_log",
    "root_binary_resolution_strength",
    "latent_multifurcation_risk_score",
    "root_binary_resolution_band",
    "b_bandwidth_topology_status",
    "h_u_population_law_status",
    "s_root_deformed_excess_log",
    "root_tail_inference_status",
    "external_law_equation_status",
    "root_binary_resolution_inference_status",
    "method_action",
    "mathematical_interpretation",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "target_count",
    "weak_resolution_count",
    "transition_resolution_count",
    "strong_resolution_count",
    "existing_tail_support_count",
    "new_spectral_support_required_count",
    "moment_only_reweighting_possible_count",
    "fail_closed_count",
    "median_root_binary_resolution_strength",
    "summary_status",
)


@dataclass(frozen=True)
class RootSelectedBinaryResolutionConfig:
    """Input/output contract for root binary-resolution rows."""

    output_dir: Path
    tail_rows_path: Path = DEFAULT_TAIL_ROWS
    external_law_equation_rows_path: Path = DEFAULT_EXTERNAL_LAW_EQUATION_ROWS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tail-rows-path", type=Path, default=DEFAULT_TAIL_ROWS)
    parser.add_argument(
        "--external-law-equation-rows-path",
        type=Path,
        default=DEFAULT_EXTERNAL_LAW_EQUATION_ROWS,
    )
    return parser.parse_args()


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{label} missing required columns: {sorted(missing)!r}.")


def _lookup_by_case(rows: pd.DataFrame) -> dict[str, pd.Series]:
    if rows.empty:
        return {}
    _require_columns(rows, {"target_case_id"}, "case rows")
    return {str(row["target_case_id"]): row for _, row in rows.iterrows()}


def _resolution_band(strength: float) -> str:
    if not math.isfinite(strength):
        return "root_resolution_missing"
    if strength < 3.5:
        return "weak_binary_resolution"
    if strength < 6.0:
        return "transition_binary_resolution"
    return "strong_binary_resolution"


def _resolution_status(
    *,
    tail_status: str,
    equation_status: str,
    resolution_band: str,
) -> tuple[str, str, str]:
    if tail_status == "calibrated_selected_root_spectral_tail_available":
        return (
            "binary_root_tail_calibrated",
            "use_existing_conservative_root_tail",
            "The binary root split has same-stratum tail support; do not infer a new law.",
        )
    if equation_status == "moment_equation_feasible_but_still_diagnostic_only":
        return (
            "binary_root_geometry_match_underpowered_tail",
            "fail_closed_until_tail_support_count_sufficient",
            "The required root geometry can be matched by current support, but tail support is still diagnostic-only.",
        )
    if equation_status == "requires_new_same_stratum_nonzero_s_h_u_support":
        return (
            "binary_root_selected_resolution_requires_new_spectral_law",
            "fail_closed_generate_same_geometry_nonzero_s_h_u_support",
            "The forced binary root is selected in this geometry, but current support has no positive deformed spectral tail mass.",
        )
    if resolution_band == "weak_binary_resolution":
        return (
            "weak_binary_root_refinement_unproven",
            "fail_closed_do_not_force_root_split_without_selected_tail_support",
            "The first binary split is a weak selected refinement candidate; ordinary sibling logic is not valid at the root.",
        )
    return (
        "binary_root_resolution_unproven",
        "fail_closed_pending_selected_root_law",
        "The first binary split may be structurally strong, but selected-root spectral calibration is missing.",
    )


def build_root_selected_binary_resolution_rows(
    *,
    tail_rows: pd.DataFrame,
    external_law_equation_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Return root binary-resolution diagnostic rows."""
    _require_columns(
        tail_rows,
        {
            "target_case_id",
            "t_selected_tie_rank_fraction",
            "a_selected_ratio_action_log1p",
            "e_edge_margin_action_log1p",
            "b_bandwidth_topology_status",
            "h_u_population_law_status",
            "s_root_deformed_excess_log",
            "root_tail_inference_status",
        },
        "tail rows",
    )
    equation_lookup = _lookup_by_case(external_law_equation_rows)
    records: list[dict[str, object]] = []
    for _, tail in tail_rows.sort_values("target_case_id").iterrows():
        case_id = string_value(tail, "target_case_id")
        t_rank = finite_float(tail.get("t_selected_tie_rank_fraction", math.nan))
        action = finite_float(tail.get("a_selected_ratio_action_log1p", math.nan))
        edge = finite_float(tail.get("e_edge_margin_action_log1p", math.nan))
        bottleneck = (
            min(action, edge) if math.isfinite(action) and math.isfinite(edge) else math.nan
        )
        strength = (
            float(t_rank * bottleneck)
            if math.isfinite(t_rank) and math.isfinite(bottleneck)
            else math.nan
        )
        risk = float(math.exp(-strength)) if math.isfinite(strength) else math.nan
        band = _resolution_band(strength)
        equation = equation_lookup.get(case_id)
        equation_status = (
            string_value(equation, "external_law_equation_status")
            if equation is not None
            else "external_law_equation_missing"
        )
        tail_status = string_value(tail, "root_tail_inference_status")
        inference_status, method_action, interpretation = _resolution_status(
            tail_status=tail_status,
            equation_status=equation_status,
            resolution_band=band,
        )
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "target_case_id": case_id,
                "binary_root_model": "observed_tree_is_always_binary_root_split",
                "root_resolution_equation": "rho_r = T * min(A,E)",
                "t_selected_tie_rank_fraction": t_rank,
                "a_selected_ratio_action_log1p": action,
                "e_edge_margin_action_log1p": edge,
                "root_action_edge_bottleneck_log": bottleneck,
                "root_binary_resolution_strength": strength,
                "latent_multifurcation_risk_score": risk,
                "root_binary_resolution_band": band,
                "b_bandwidth_topology_status": string_value(
                    tail,
                    "b_bandwidth_topology_status",
                ),
                "h_u_population_law_status": string_value(
                    tail,
                    "h_u_population_law_status",
                ),
                "s_root_deformed_excess_log": finite_float(
                    tail.get("s_root_deformed_excess_log", math.nan)
                ),
                "root_tail_inference_status": tail_status,
                "external_law_equation_status": equation_status,
                "root_binary_resolution_inference_status": inference_status,
                "method_action": method_action,
                "mathematical_interpretation": interpretation,
            }
        )
    return pd.DataFrame.from_records(records, columns=ROW_COLUMNS)


def summarize_root_selected_binary_resolution_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize binary-root resolution and selected-tail status."""
    if rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    band = rows["root_binary_resolution_band"].astype(str)
    status = rows["root_binary_resolution_inference_status"].astype(str)
    strength = pd.to_numeric(rows["root_binary_resolution_strength"], errors="coerce")
    fail_closed = rows["method_action"].astype(str).str.startswith("fail_closed")
    return pd.DataFrame.from_records(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "target_count": int(rows.shape[0]),
                "weak_resolution_count": int(band.eq("weak_binary_resolution").sum()),
                "transition_resolution_count": int(band.eq("transition_binary_resolution").sum()),
                "strong_resolution_count": int(band.eq("strong_binary_resolution").sum()),
                "existing_tail_support_count": int(status.eq("binary_root_tail_calibrated").sum()),
                "new_spectral_support_required_count": int(
                    status.eq("binary_root_selected_resolution_requires_new_spectral_law").sum()
                ),
                "moment_only_reweighting_possible_count": int(
                    status.eq("binary_root_geometry_match_underpowered_tail").sum()
                ),
                "fail_closed_count": int(fail_closed.sum()),
                "median_root_binary_resolution_strength": float(strength.median())
                if strength.notna().any()
                else math.nan,
                "summary_status": (
                    "selected_root_binary_resolution_law_incomplete"
                    if bool(fail_closed.any())
                    else "all_binary_root_rows_have_tail_support"
                ),
            }
        ],
        columns=SUMMARY_COLUMNS,
    )


def evaluate_root_selected_binary_resolution_panel(
    config: RootSelectedBinaryResolutionConfig,
) -> dict[str, pd.DataFrame]:
    tail = pd.read_csv(config.tail_rows_path, low_memory=False)
    equations = pd.read_csv(config.external_law_equation_rows_path, low_memory=False)
    rows = build_root_selected_binary_resolution_rows(
        tail_rows=tail,
        external_law_equation_rows=equations,
    )
    summary = summarize_root_selected_binary_resolution_rows(rows)
    return {"rows": rows, "summary": summary}


def run_root_selected_binary_resolution_panel(
    config: RootSelectedBinaryResolutionConfig,
) -> dict[str, Path]:
    tables = evaluate_root_selected_binary_resolution_panel(config)
    return write_diagnostic_bundle(
        output_dir=config.output_dir,
        tables=tables,
        filenames={
            "rows": ROWS_OUTPUT,
            "summary": SUMMARY_OUTPUT,
        },
        manifest={
            "schema_version": SCHEMA_VERSION,
            "study_role": STUDY_ROLE,
            "generated_by": GENERATED_BY,
            "config": config,
        },
        manifest_filename=MANIFEST_OUTPUT,
    )


def main() -> None:
    args = parse_args()
    outputs = run_root_selected_binary_resolution_panel(
        RootSelectedBinaryResolutionConfig(
            output_dir=args.output_dir,
            tail_rows_path=args.tail_rows_path,
            external_law_equation_rows_path=args.external_law_equation_rows_path,
        )
    )
    print_diagnostic_output_paths(outputs)


if __name__ == "__main__":
    main()
