"""Audit bandwidth/neighborhood joins for root tie-rank proposal rows.

The coupling-equation panel separates pure spectral-action coupling from
measured selected-neighborhood support. This diagnostic localizes why the
measured-neighborhood factor is missing: no topology-frontier row exists, a
frontier row exists but was not joined before feasibility, or the proposal
matrix itself is unavailable for replay.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA_VERSION = "root_tie_rank_neighborhood_join_audit/v1"
STUDY_ROLE = "diagnostic_root_tie_rank_neighborhood_join_audit_not_calibration"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration.root_tie_rank_neighborhood_join_audit"
)

DEFAULT_RESULT_ROOT = Path(
    "raw/assets/benchmark-results/specific_small_method_benchmark_20260615"
)
DEFAULT_PROPOSAL_ROOT = (
    DEFAULT_RESULT_ROOT / "root_tie_rank_null_proposal_frontier_two_case_smoke"
)
DEFAULT_PROPOSAL_FEASIBILITY_ROWS = (
    DEFAULT_PROPOSAL_ROOT / "root_tie_rank_null_proposal_combined_feasibility_rows.csv"
)
DEFAULT_PROPOSAL_MIXED_ROWS = (
    DEFAULT_PROPOSAL_ROOT / "root_tie_rank_null_proposal_mixed_rows.csv"
)
DEFAULT_GENERATED_MATRIX_DIR = DEFAULT_PROPOSAL_ROOT / "generated_proposal_matrices"
DEFAULT_TOPOLOGY_FRONTIER_ROWS = (
    DEFAULT_RESULT_ROOT
    / "selected_neighborhood_topology_frontier_overlap_expanded_candidates"
    / "selected_neighborhood_topology_frontier_rows.csv"
)

ROWS_OUTPUT = "root_tie_rank_neighborhood_join_audit_rows.csv"
SUMMARY_OUTPUT = "root_tie_rank_neighborhood_join_audit_summary.csv"
MANIFEST_OUTPUT = "manifest.json"

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "base_case_id",
    "data_role",
    "calibration_role",
    "proposal_family",
    "root_bandwidth_reopen_band",
    "feasibility_root_bandwidth_reopen_count",
    "mixed_root_frontier_row_count",
    "mixed_root_bandwidth_reopen_count",
    "mixed_root_hybrid_strict_support_count",
    "mixed_root_bandwidth_locality_status",
    "topology_frontier_root_row_count",
    "topology_frontier_reopen_count",
    "topology_frontier_direct_positive_reopen_count",
    "topology_frontier_effective_support_pass_count",
    "topology_frontier_hybrid_strict_support_count",
    "generated_matrix_path",
    "generated_matrix_exists",
    "bandwidth_join_status",
    "neighborhood_replay_next_action",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "proposal_family",
    "row_count",
    "bandwidth_measured_available_count",
    "bandwidth_missing_count",
    "topology_frontier_root_case_count",
    "topology_frontier_join_needed_count",
    "generated_matrix_exists_count",
    "generated_replay_needed_count",
    "generated_matrix_missing_count",
    "summary_status",
)


@dataclass(frozen=True)
class RootTieRankNeighborhoodJoinAuditConfig:
    """Input/output paths for the neighborhood join audit."""

    output_dir: Path
    proposal_feasibility_rows_path: Path = DEFAULT_PROPOSAL_FEASIBILITY_ROWS
    proposal_mixed_rows_path: Path | None = DEFAULT_PROPOSAL_MIXED_ROWS
    topology_frontier_rows_path: Path | None = DEFAULT_TOPOLOGY_FRONTIER_ROWS
    generated_matrix_dir: Path | None = DEFAULT_GENERATED_MATRIX_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--proposal-feasibility-rows-path",
        type=Path,
        default=DEFAULT_PROPOSAL_FEASIBILITY_ROWS,
    )
    parser.add_argument(
        "--proposal-mixed-rows-path",
        type=Path,
        default=DEFAULT_PROPOSAL_MIXED_ROWS,
    )
    parser.add_argument(
        "--topology-frontier-rows-path",
        type=Path,
        default=DEFAULT_TOPOLOGY_FRONTIER_ROWS,
    )
    parser.add_argument(
        "--generated-matrix-dir",
        type=Path,
        default=DEFAULT_GENERATED_MATRIX_DIR,
    )
    return parser.parse_args()


def _json_default(value: object) -> object:
    if isinstance(value, RootTieRankNeighborhoodJoinAuditConfig):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{label} missing required columns: {sorted(missing)!r}.")


def _read_optional_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _finite_float(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return math.nan
    return numeric if math.isfinite(numeric) else math.nan


def _int_or_zero(value: object) -> int:
    numeric = _finite_float(value)
    return int(numeric) if math.isfinite(numeric) else 0


def _bool_value(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _bool_sum(values: pd.Series) -> int:
    if values.empty:
        return 0
    return int(values.map(_bool_value).sum())


def _string_value(row: pd.Series | dict[str, object], column: str, default: str = "") -> str:
    if column not in row:
        return default
    value = row[column]
    if pd.isna(value):
        return default
    return str(value)


def _mixed_lookup(mixed_rows: pd.DataFrame) -> dict[str, dict[str, object]]:
    if mixed_rows.empty:
        return {}
    _require_columns(mixed_rows, {"case_id"}, "proposal mixed rows")
    return {str(row["case_id"]): dict(row) for _, row in mixed_rows.iterrows()}


def _frontier_lookup(topology_frontier_rows: pd.DataFrame) -> dict[str, dict[str, object]]:
    if topology_frontier_rows.empty:
        return {}
    _require_columns(topology_frontier_rows, {"case_id"}, "topology frontier rows")
    rows = topology_frontier_rows.copy()
    if "candidate_scope" in rows.columns:
        scope = rows["candidate_scope"].astype(str)
        root_mask = scope.eq("root_non_direct")
        if "parent_id" in rows.columns:
            root_mask = root_mask | (
                scope.eq("direct_measurable")
                & rows["parent_id"].fillna("").astype(str).eq("")
            )
        if "depth" in rows.columns:
            depth = pd.to_numeric(rows["depth"], errors="coerce")
            root_mask = root_mask | (scope.eq("direct_measurable") & depth.eq(0))
        rows = rows[root_mask].copy()
    records: dict[str, dict[str, object]] = {}
    for case_id, group in rows.groupby("case_id", sort=True):
        records[str(case_id)] = {
            "topology_frontier_root_row_count": int(group.shape[0]),
            "topology_frontier_reopen_count": _bool_sum(
                group.get("bandwidth_reference_reopens", pd.Series(dtype=object))
            ),
            "topology_frontier_direct_positive_reopen_count": _bool_sum(
                group.get(
                    "bandwidth_reference_direct_positive_reopens",
                    pd.Series(dtype=object),
                )
            ),
            "topology_frontier_effective_support_pass_count": _bool_sum(
                group.get(
                    "interpolation_effective_support_pass",
                    pd.Series(dtype=object),
                )
            ),
            "topology_frontier_hybrid_strict_support_count": _bool_sum(
                group.get("hybrid_strict_support", pd.Series(dtype=object))
            ),
        }
    return records


def _matrix_path_for_case(
    *,
    case_id: str,
    generated_matrix_dir: Path | None,
) -> tuple[str, bool]:
    if generated_matrix_dir is None:
        return "", False
    path = Path(generated_matrix_dir) / f"{case_id}.csv"
    return str(path), path.exists()


def _join_status_and_action(
    *,
    bandwidth_band: str,
    proposal_family: str,
    mixed_locality_status: str,
    frontier_count: int,
    frontier_reopen_count: int,
    frontier_hybrid_count: int,
    generated_matrix_exists: bool,
) -> tuple[str, str]:
    if bandwidth_band != "bandwidth_reopen_missing":
        return (
            "bandwidth_measured_available",
            "carry_bandwidth_into_conditioning_stratum",
        )
    if frontier_count > 0:
        if frontier_hybrid_count > 0:
            return (
                "topology_frontier_rows_present_hybrid_support_not_joined",
                "join_topology_frontier_rows_before_feasibility",
            )
        if frontier_reopen_count > 0:
            return (
                "topology_frontier_rows_present_reopen_not_joined",
                "join_topology_frontier_rows_before_feasibility",
            )
        return (
            "topology_frontier_rows_present_no_reopen_not_joined",
            "join_topology_frontier_rows_before_feasibility",
        )
    if proposal_family == "observed_target":
        return (
            "observed_target_topology_frontier_absent",
            "check_observed_topology_frontier_inputs",
        )
    if generated_matrix_exists:
        suffix = (
            "after_mixed_frontier_not_joined"
            if mixed_locality_status == "topology_frontier_not_joined"
            else "after_missing_frontier_rows"
        )
        return (
            f"generated_matrix_available_topology_frontier_absent_{suffix}",
            "run_generated_measurability_and_topology_frontier_replay",
        )
    return (
        "generated_matrix_missing_topology_frontier_absent",
        "regenerate_proposal_matrix_or_point_to_matrix_dir",
    )


def build_root_tie_rank_neighborhood_join_audit_rows(
    *,
    proposal_feasibility_rows: pd.DataFrame,
    proposal_mixed_rows: pd.DataFrame | None = None,
    topology_frontier_rows: pd.DataFrame | None = None,
    generated_matrix_dir: Path | None = None,
) -> pd.DataFrame:
    """Return row-level audit of root bandwidth/neighborhood join coverage."""
    _require_columns(
        proposal_feasibility_rows,
        {
            "case_id",
            "proposal_family",
            "root_bandwidth_reopen_band",
            "root_bandwidth_reopen_count",
        },
        "proposal feasibility rows",
    )
    mixed_by_case = _mixed_lookup(
        proposal_mixed_rows if proposal_mixed_rows is not None else pd.DataFrame()
    )
    frontier_by_case = _frontier_lookup(
        topology_frontier_rows
        if topology_frontier_rows is not None
        else pd.DataFrame()
    )
    records: list[dict[str, object]] = []
    for _, row in proposal_feasibility_rows.sort_values("case_id").iterrows():
        case_id = str(row["case_id"])
        proposal_family = _string_value(row, "proposal_family")
        mixed = mixed_by_case.get(case_id, {})
        frontier = frontier_by_case.get(case_id, {})
        matrix_path, matrix_exists = _matrix_path_for_case(
            case_id=case_id,
            generated_matrix_dir=generated_matrix_dir,
        )
        bandwidth_band = _string_value(row, "root_bandwidth_reopen_band")
        mixed_locality_status = _string_value(
            mixed,
            "root_bandwidth_locality_status",
        )
        frontier_count = _int_or_zero(
            frontier.get("topology_frontier_root_row_count", 0)
        )
        frontier_reopen_count = _int_or_zero(
            frontier.get("topology_frontier_reopen_count", 0)
        )
        frontier_hybrid_count = _int_or_zero(
            frontier.get("topology_frontier_hybrid_strict_support_count", 0)
        )
        status, next_action = _join_status_and_action(
            bandwidth_band=bandwidth_band,
            proposal_family=proposal_family,
            mixed_locality_status=mixed_locality_status,
            frontier_count=frontier_count,
            frontier_reopen_count=frontier_reopen_count,
            frontier_hybrid_count=frontier_hybrid_count,
            generated_matrix_exists=matrix_exists,
        )
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "case_id": case_id,
                "base_case_id": _string_value(row, "base_case_id", case_id),
                "data_role": _string_value(row, "data_role"),
                "calibration_role": _string_value(row, "calibration_role"),
                "proposal_family": proposal_family,
                "root_bandwidth_reopen_band": bandwidth_band,
                "feasibility_root_bandwidth_reopen_count": _finite_float(
                    row.get("root_bandwidth_reopen_count", math.nan)
                ),
                "mixed_root_frontier_row_count": _int_or_zero(
                    mixed.get("root_frontier_row_count", 0)
                ),
                "mixed_root_bandwidth_reopen_count": _int_or_zero(
                    mixed.get("root_bandwidth_reopen_count", 0)
                ),
                "mixed_root_hybrid_strict_support_count": _int_or_zero(
                    mixed.get("root_hybrid_strict_support_count", 0)
                ),
                "mixed_root_bandwidth_locality_status": mixed_locality_status,
                "topology_frontier_root_row_count": frontier_count,
                "topology_frontier_reopen_count": frontier_reopen_count,
                "topology_frontier_direct_positive_reopen_count": _int_or_zero(
                    frontier.get(
                        "topology_frontier_direct_positive_reopen_count",
                        0,
                    )
                ),
                "topology_frontier_effective_support_pass_count": _int_or_zero(
                    frontier.get(
                        "topology_frontier_effective_support_pass_count",
                        0,
                    )
                ),
                "topology_frontier_hybrid_strict_support_count": frontier_hybrid_count,
                "generated_matrix_path": matrix_path,
                "generated_matrix_exists": matrix_exists,
                "bandwidth_join_status": status,
                "neighborhood_replay_next_action": next_action,
            }
        )
    return pd.DataFrame.from_records(records, columns=ROW_COLUMNS)


def summarize_root_tie_rank_neighborhood_join_audit_rows(
    audit_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize neighborhood join coverage by proposal family."""
    if audit_rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    records: list[dict[str, object]] = []
    for family, group in audit_rows.groupby("proposal_family", sort=True):
        measured_count = int(
            group["bandwidth_join_status"].astype(str).eq(
                "bandwidth_measured_available"
            ).sum()
        )
        missing_count = int(group.shape[0] - measured_count)
        frontier_case_count = int((group["topology_frontier_root_row_count"] > 0).sum())
        join_needed_count = int(
            group["neighborhood_replay_next_action"].astype(str).eq(
                "join_topology_frontier_rows_before_feasibility"
            ).sum()
        )
        matrix_count = int(group["generated_matrix_exists"].map(_bool_value).sum())
        replay_needed_count = int(
            group["neighborhood_replay_next_action"].astype(str).eq(
                "run_generated_measurability_and_topology_frontier_replay"
            ).sum()
        )
        matrix_missing_count = int(
            group["neighborhood_replay_next_action"].astype(str).eq(
                "regenerate_proposal_matrix_or_point_to_matrix_dir"
            ).sum()
        )
        if missing_count == 0:
            status = "bandwidth_join_complete"
        elif replay_needed_count > 0:
            status = "generated_topology_frontier_replay_needed"
        elif join_needed_count > 0:
            status = "topology_frontier_join_needed"
        elif matrix_missing_count > 0:
            status = "generated_matrix_missing"
        else:
            status = "bandwidth_support_missing_inputs"
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "proposal_family": str(family),
                "row_count": int(group.shape[0]),
                "bandwidth_measured_available_count": measured_count,
                "bandwidth_missing_count": missing_count,
                "topology_frontier_root_case_count": frontier_case_count,
                "topology_frontier_join_needed_count": join_needed_count,
                "generated_matrix_exists_count": matrix_count,
                "generated_replay_needed_count": replay_needed_count,
                "generated_matrix_missing_count": matrix_missing_count,
                "summary_status": status,
            }
        )
    return pd.DataFrame.from_records(records, columns=SUMMARY_COLUMNS)


def evaluate_root_tie_rank_neighborhood_join_audit(
    config: RootTieRankNeighborhoodJoinAuditConfig,
) -> dict[str, pd.DataFrame]:
    """Read inputs and return neighborhood join audit tables."""
    feasibility = pd.read_csv(config.proposal_feasibility_rows_path, low_memory=False)
    mixed = _read_optional_csv(config.proposal_mixed_rows_path)
    topology = _read_optional_csv(config.topology_frontier_rows_path)
    rows = build_root_tie_rank_neighborhood_join_audit_rows(
        proposal_feasibility_rows=feasibility,
        proposal_mixed_rows=mixed,
        topology_frontier_rows=topology,
        generated_matrix_dir=config.generated_matrix_dir,
    )
    summary = summarize_root_tie_rank_neighborhood_join_audit_rows(rows)
    return {"rows": rows, "summary": summary}


def run_root_tie_rank_neighborhood_join_audit(
    config: RootTieRankNeighborhoodJoinAuditConfig,
) -> dict[str, Path]:
    """Run the neighborhood join audit and write outputs."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = evaluate_root_tie_rank_neighborhood_join_audit(config)
    paths = {
        "rows": output_dir / ROWS_OUTPUT,
        "summary": output_dir / SUMMARY_OUTPUT,
    }
    for key, path in paths.items():
        tables[key].to_csv(path, index=False)
    manifest_path = output_dir / MANIFEST_OUTPUT
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at": datetime.now(UTC).isoformat(),
        "config": config,
        "row_counts": {key: int(table.shape[0]) for key, table in tables.items()},
        "outputs": paths,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    paths["manifest"] = manifest_path
    return paths


def main() -> None:
    args = parse_args()
    outputs = run_root_tie_rank_neighborhood_join_audit(
        RootTieRankNeighborhoodJoinAuditConfig(
            output_dir=args.output_dir,
            proposal_feasibility_rows_path=args.proposal_feasibility_rows_path,
            proposal_mixed_rows_path=args.proposal_mixed_rows_path,
            topology_frontier_rows_path=args.topology_frontier_rows_path,
            generated_matrix_dir=args.generated_matrix_dir,
        )
    )
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
