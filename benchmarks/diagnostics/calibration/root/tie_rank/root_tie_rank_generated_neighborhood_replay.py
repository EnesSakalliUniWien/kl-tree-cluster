"""Replay generated proposal matrices through selected-neighborhood diagnostics.

The root tie-rank proposal frontier stores generated matrices for proposal
families, but the first coupling panels did not replay those matrices through
the selected-neighborhood measurability and topology-frontier stack. This
module fills that operational gap. The output is diagnostic-only and does not
change production traversal.
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
from scipy.spatial.distance import pdist
from tree_break_selection.tree.feature_space import (
    bernoulli_feature_space_from_columns,
)

from benchmarks.diagnostics.calibration.selected.family.selected_family_traversal_panel import (
    _build_node_decisions,
    _method_label,
    _profile_id_for_method,
)
from benchmarks.diagnostics.calibration.selected.neighborhood.selected_neighborhood_distribution_panel import (
    build_selected_neighborhood_distribution_rows,
)
from benchmarks.diagnostics.calibration.selected.neighborhood.selected_neighborhood_measurability_law import (
    build_measurability_law_rows,
    enrich_measurability_input_rows,
    summarize_measurability_law_rows,
)
from benchmarks.diagnostics.calibration.selected.neighborhood.selected_neighborhood_pvalue_interpolation_comparison import (
    build_pvalue_interpolation_comparison_rows,
    summarize_pvalue_interpolation_cases,
    summarize_pvalue_interpolation_comparison,
    summarize_tau_s_sensitivity,
)
from benchmarks.diagnostics.calibration.selected.neighborhood.selected_neighborhood_topology_frontier import (
    build_topology_frontier_rows,
    summarize_topology_frontier_rows,
)
from benchmarks.shared.runners.tbs_runner import _run_tbs_method

SCHEMA_VERSION = "root_tie_rank_generated_neighborhood_replay/v1"
STUDY_ROLE = "diagnostic_root_tie_rank_generated_neighborhood_replay_not_calibration"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration.root.tie_rank.root_tie_rank_generated_neighborhood_replay"
)

DEFAULT_RESULT_ROOT = Path("raw/assets/benchmark-results/specific_small_method_benchmark_20260615")
DEFAULT_PROPOSAL_ROOT = DEFAULT_RESULT_ROOT / "root_tie_rank_null_proposal_frontier_two_case_smoke"
DEFAULT_PROPOSAL_FEASIBILITY_ROWS = (
    DEFAULT_PROPOSAL_ROOT / "root_tie_rank_null_proposal_combined_feasibility_rows.csv"
)
DEFAULT_GENERATED_MATRIX_DIR = DEFAULT_PROPOSAL_ROOT / "generated_proposal_matrices"
DEFAULT_METHOD_ID = "fixed_coordinate_conditional_topology_diagnostic_v1"

RUN_ROWS_OUTPUT = "root_tie_rank_generated_neighborhood_replay_rows.csv"
NODE_DECISIONS_OUTPUT = "generated_multiscale_node_decisions.csv"
DISTRIBUTION_ROWS_OUTPUT = "generated_selected_neighborhood_distribution_rows.csv"
PVALUE_ROWS_OUTPUT = "generated_selected_neighborhood_pvalue_interpolation_rows.csv"
PVALUE_SUMMARY_OUTPUT = "generated_selected_neighborhood_pvalue_interpolation_summary.csv"
PVALUE_CASE_SUMMARY_OUTPUT = "generated_selected_neighborhood_pvalue_interpolation_case_summary.csv"
PVALUE_TAU_S_OUTPUT = "generated_selected_neighborhood_pvalue_tau_s_sensitivity.csv"
MEASURABILITY_ROWS_OUTPUT = "generated_selected_neighborhood_measurability_rows.csv"
MEASURABILITY_SUMMARY_OUTPUT = "generated_selected_neighborhood_measurability_summary.csv"
TOPOLOGY_ROWS_OUTPUT = "generated_selected_neighborhood_topology_frontier_rows.csv"
TOPOLOGY_SUMMARY_OUTPUT = "generated_selected_neighborhood_topology_frontier_summary.csv"
MANIFEST_OUTPUT = "manifest.json"

PROPOSAL_METADATA_COLUMNS = (
    "base_case_id",
    "calibration_role",
    "proposal_family",
    "proposal_calibration_status",
    "root_mixed_region_component",
    "root_tie_rank_band",
    "root_edge_margin_band",
    "root_spectral_ratio_band",
    "root_bandwidth_reopen_band",
    "root_conditioning_stratum_key",
    "root_sibling_selected_ratio",
    "root_tie_rank_median_fraction",
    "root_edge_path_statistic_margin",
    "root_selected_eigenvalue_over_mp_upper_bound",
    "root_bandwidth_reopen_count",
    "null_seed",
    "generated_matrix_density",
)

RUN_ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "base_case_id",
    "data_role",
    "calibration_role",
    "proposal_family",
    "method_id",
    "profile_id",
    "sibling_gate_method",
    "edge_alpha",
    "sibling_alpha",
    "replicate",
    "data_seed",
    "matrix_path",
    "matrix_exists",
    "n_samples",
    "n_features",
    "matrix_density",
    "run_status",
    "skip_reason",
    "found_clusters",
    "stable_boundary_count",
    "selected_root_blocked_count",
    "selected_family_blocked_count",
    "unstable_passthrough_zone_count",
    "accepted_internal_split_count",
    "leaf_fragment_count",
    "root_frontier_row_count",
    "root_bandwidth_reference_reopen_count",
    "root_hybrid_strict_support_count",
    "replay_status",
)


@dataclass(frozen=True)
class RootTieRankGeneratedNeighborhoodReplayConfig:
    """Input/output contract for generated-neighborhood replay."""

    output_dir: Path
    proposal_feasibility_rows_path: Path = DEFAULT_PROPOSAL_FEASIBILITY_ROWS
    generated_matrix_dir: Path = DEFAULT_GENERATED_MATRIX_DIR
    method_id: str = DEFAULT_METHOD_ID
    sibling_alpha: float = 0.01
    edge_alpha: float = 0.001
    tree_linkage_method: str = "average"
    max_cases: int | None = None
    case_ids: tuple[str, ...] = ()
    candidate_only: bool = False
    reference_tau_s: float = 20.0
    require_spectral_flow: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--proposal-feasibility-rows-path",
        type=Path,
        default=DEFAULT_PROPOSAL_FEASIBILITY_ROWS,
    )
    parser.add_argument(
        "--generated-matrix-dir",
        type=Path,
        default=DEFAULT_GENERATED_MATRIX_DIR,
    )
    parser.add_argument("--method-id", default=DEFAULT_METHOD_ID)
    parser.add_argument("--sibling-alpha", type=float, default=0.01)
    parser.add_argument("--edge-alpha", type=float, default=0.001)
    parser.add_argument("--tree-linkage-method", default="average")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--case-id", dest="case_ids", action="append", default=[])
    parser.add_argument("--candidate-only", action="store_true")
    parser.add_argument("--reference-tau-s", type=float, default=20.0)
    parser.add_argument("--require-spectral-flow", action="store_true")
    return parser.parse_args()


def _json_default(value: object) -> object:
    if isinstance(value, RootTieRankGeneratedNeighborhoodReplayConfig):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _finite_float(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return math.nan
    return numeric if math.isfinite(numeric) else math.nan


def _finite_int(value: object, default: int = 0) -> int:
    numeric = _finite_float(value)
    return int(numeric) if math.isfinite(numeric) else int(default)


def _string_value(
    row: pd.Series | dict[str, object],
    column: str,
    default: str = "",
) -> str:
    if column not in row:
        return default
    value = row[column]
    if pd.isna(value):
        return default
    return str(value)


def _generated_proposal_rows(
    proposal_rows: pd.DataFrame,
    *,
    case_ids: tuple[str, ...] = (),
    max_cases: int | None = None,
) -> pd.DataFrame:
    required = {"case_id", "data_role", "proposal_family"}
    missing = required - set(proposal_rows.columns)
    if missing:
        raise ValueError(f"proposal rows missing required columns: {sorted(missing)!r}.")
    rows = proposal_rows.loc[
        ~proposal_rows["proposal_family"].astype(str).eq("observed_target")
    ].copy()
    if case_ids:
        wanted = {str(case_id) for case_id in case_ids}
        rows = rows.loc[rows["case_id"].astype(str).isin(wanted)].copy()
    rows = rows.sort_values(["case_id", "proposal_family"]).reset_index(drop=True)
    if max_cases is not None:
        rows = rows.head(int(max_cases)).copy()
    return rows


def _matrix_path(row: pd.Series, generated_matrix_dir: Path) -> Path:
    return Path(generated_matrix_dir) / f"{_string_value(row, 'case_id')}.csv"


def _metadata_lookup(proposal_rows: pd.DataFrame) -> pd.DataFrame:
    columns = ["case_id"]
    columns.extend(
        column for column in PROPOSAL_METADATA_COLUMNS if column in proposal_rows.columns
    )
    return proposal_rows.loc[:, columns].drop_duplicates("case_id", keep="first")


def _attach_metadata(rows: pd.DataFrame, proposal_rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty or "case_id" not in rows.columns:
        return rows.copy()
    metadata = _metadata_lookup(proposal_rows)
    value_columns = [column for column in metadata.columns if column != "case_id"]
    drop_columns = [column for column in value_columns if column in rows.columns]
    base = rows.drop(columns=drop_columns).copy() if drop_columns else rows.copy()
    return base.merge(metadata, on="case_id", how="left")


def _matrix_density(data: pd.DataFrame) -> float:
    if data.empty:
        return math.nan
    return float(data.to_numpy(dtype=float).mean())


def _replay_one_matrix(
    *,
    row: pd.Series,
    matrix_path: Path,
    method_id: str,
    sibling_alpha: float,
    edge_alpha: float,
    tree_linkage_method: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    case_id = _string_value(row, "case_id")
    data_role = _string_value(row, "data_role", "diagnostic_proposal")
    replicate = _finite_int(row.get("replicate", 0))
    data_seed = _finite_int(row.get("null_seed", row.get("data_seed", replicate)))

    if not matrix_path.exists():
        run_row = _base_run_row(
            row=row,
            matrix_path=matrix_path,
            method_id=method_id,
            sibling_alpha=sibling_alpha,
            edge_alpha=edge_alpha,
            run_status="matrix_missing",
            skip_reason="generated_matrix_missing",
        )
        return run_row, pd.DataFrame()

    data = pd.read_csv(matrix_path, index_col=0)
    data = data.apply(pd.to_numeric, errors="raise")
    feature_space = bernoulli_feature_space_from_columns(tuple(data.columns))
    distance = pdist(data.to_numpy(dtype=float), metric="hamming")

    try:
        result = _run_tbs_method(
            data,
            distance,
            sibling_significance_level=float(sibling_alpha),
            tree_linkage_method=str(tree_linkage_method),
            edge_alpha=float(edge_alpha),
            feature_space=feature_space,
            sibling_gate_profile=_profile_id_for_method(method_id),
        )
        run_status = str(result.status)
        skip_reason = "" if result.skip_reason is None else str(result.skip_reason)
        if run_status != "ok":
            run_row = _base_run_row(
                row=row,
                matrix_path=matrix_path,
                method_id=method_id,
                sibling_alpha=sibling_alpha,
                edge_alpha=edge_alpha,
                n_samples=int(data.shape[0]),
                n_features=int(data.shape[1]),
                matrix_density=_matrix_density(data),
                run_status=run_status,
                skip_reason=skip_reason,
            )
            return run_row, pd.DataFrame()
        node_decisions = _build_node_decisions(
            case_id=case_id,
            data_role=data_role,
            method_id=method_id,
            replicate=replicate,
            data_seed=data_seed,
            result=result,
        )
        node_decisions = node_decisions.copy()
        node_decisions["base_case_id"] = _string_value(row, "base_case_id", case_id)
        node_decisions["calibration_role"] = _string_value(row, "calibration_role")
        node_decisions["proposal_family"] = _string_value(row, "proposal_family")
        run_row = _run_summary_row(
            row=row,
            matrix_path=matrix_path,
            method_id=method_id,
            sibling_alpha=sibling_alpha,
            edge_alpha=edge_alpha,
            n_samples=int(data.shape[0]),
            n_features=int(data.shape[1]),
            matrix_density=_matrix_density(data),
            result=result,
            node_decisions=node_decisions,
        )
        return run_row, node_decisions
    except Exception as exc:  # pragma: no cover - exercised through integration paths.
        run_row = _base_run_row(
            row=row,
            matrix_path=matrix_path,
            method_id=method_id,
            sibling_alpha=sibling_alpha,
            edge_alpha=edge_alpha,
            n_samples=int(data.shape[0]),
            n_features=int(data.shape[1]),
            matrix_density=_matrix_density(data),
            run_status="error",
            skip_reason=f"{type(exc).__name__}: {exc}",
        )
        return run_row, pd.DataFrame()


def _base_run_row(
    *,
    row: pd.Series,
    matrix_path: Path,
    method_id: str,
    sibling_alpha: float,
    edge_alpha: float,
    run_status: str,
    skip_reason: str,
    n_samples: int | float = math.nan,
    n_features: int | float = math.nan,
    matrix_density: float = math.nan,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "case_id": _string_value(row, "case_id"),
        "base_case_id": _string_value(row, "base_case_id", _string_value(row, "case_id")),
        "data_role": _string_value(row, "data_role"),
        "calibration_role": _string_value(row, "calibration_role"),
        "proposal_family": _string_value(row, "proposal_family"),
        "method_id": method_id,
        "profile_id": _profile_id_for_method(method_id) or "",
        "sibling_gate_method": _method_label(method_id),
        "edge_alpha": float(edge_alpha),
        "sibling_alpha": float(sibling_alpha),
        "replicate": _finite_int(row.get("replicate", 0)),
        "data_seed": _finite_int(row.get("null_seed", row.get("data_seed", 0))),
        "matrix_path": str(matrix_path),
        "matrix_exists": bool(matrix_path.exists()),
        "n_samples": n_samples,
        "n_features": n_features,
        "matrix_density": matrix_density,
        "run_status": run_status,
        "skip_reason": skip_reason,
        "found_clusters": math.nan,
        "stable_boundary_count": 0,
        "selected_root_blocked_count": 0,
        "selected_family_blocked_count": 0,
        "unstable_passthrough_zone_count": 0,
        "accepted_internal_split_count": 0,
        "leaf_fragment_count": 0,
        "root_frontier_row_count": 0,
        "root_bandwidth_reference_reopen_count": 0,
        "root_hybrid_strict_support_count": 0,
        "replay_status": "replay_not_available",
    }


def _run_summary_row(
    *,
    row: pd.Series,
    matrix_path: Path,
    method_id: str,
    sibling_alpha: float,
    edge_alpha: float,
    n_samples: int,
    n_features: int,
    matrix_density: float,
    result,
    node_decisions: pd.DataFrame,
) -> dict[str, object]:
    run_row = _base_run_row(
        row=row,
        matrix_path=matrix_path,
        method_id=method_id,
        sibling_alpha=sibling_alpha,
        edge_alpha=edge_alpha,
        n_samples=n_samples,
        n_features=n_features,
        matrix_density=matrix_density,
        run_status=str(result.status),
        skip_reason="" if result.skip_reason is None else str(result.skip_reason),
    )
    run_row.update(
        {
            "found_clusters": int(result.found_clusters),
            "stable_boundary_count": int(
                node_decisions["decision_class"].eq("stable_boundary").sum()
            ),
            "selected_root_blocked_count": int(
                node_decisions["decision_class"].eq("selected_root_blocked").sum()
            ),
            "selected_family_blocked_count": int(
                node_decisions["decision_class"].eq("selected_family_blocked").sum()
            ),
            "unstable_passthrough_zone_count": int(
                node_decisions["decision_class"].eq("unstable_passthrough_zone").sum()
            ),
            "accepted_internal_split_count": int(
                node_decisions["decision_class"].eq("accepted_internal_split").sum()
            ),
            "leaf_fragment_count": int(node_decisions["decision_class"].eq("leaf_fragment").sum()),
            "replay_status": "tbs_replay_completed",
        }
    )
    return run_row


def _update_run_rows_with_frontier(
    run_rows: pd.DataFrame,
    topology_rows: pd.DataFrame,
) -> pd.DataFrame:
    if run_rows.empty or topology_rows.empty:
        return run_rows.copy()
    root_rows = topology_rows.loc[
        topology_rows["candidate_scope"].astype(str).eq("root_non_direct")
    ].copy()
    if root_rows.empty:
        return run_rows.copy()
    summary = (
        root_rows.groupby("case_id", dropna=False)
        .agg(
            root_frontier_row_count=("case_id", "size"),
            root_bandwidth_reference_reopen_count=(
                "bandwidth_reference_reopens",
                lambda values: int(pd.Series(values).astype(bool).sum()),
            ),
            root_hybrid_strict_support_count=(
                "hybrid_strict_support",
                lambda values: int(pd.Series(values).astype(bool).sum()),
            ),
        )
        .reset_index()
    )
    enriched = run_rows.drop(
        columns=[
            "root_frontier_row_count",
            "root_bandwidth_reference_reopen_count",
            "root_hybrid_strict_support_count",
        ],
        errors="ignore",
    ).merge(summary, on="case_id", how="left")
    for column in (
        "root_frontier_row_count",
        "root_bandwidth_reference_reopen_count",
        "root_hybrid_strict_support_count",
    ):
        enriched[column] = pd.to_numeric(enriched[column], errors="coerce").fillna(0).astype(int)
    return enriched.loc[:, RUN_ROW_COLUMNS].copy()


def build_generated_neighborhood_replay_tables(
    *,
    proposal_rows: pd.DataFrame,
    generated_matrix_dir: Path,
    method_id: str = DEFAULT_METHOD_ID,
    sibling_alpha: float = 0.01,
    edge_alpha: float = 0.001,
    tree_linkage_method: str = "average",
    max_cases: int | None = None,
    case_ids: tuple[str, ...] = (),
    candidate_only: bool = False,
    reference_tau_s: float = 20.0,
    require_spectral_flow: bool = False,
) -> dict[str, pd.DataFrame]:
    """Return generated proposal replay tables without writing files."""
    selected_rows = _generated_proposal_rows(
        proposal_rows,
        case_ids=tuple(case_ids),
        max_cases=max_cases,
    )
    run_records: list[dict[str, object]] = []
    node_frames: list[pd.DataFrame] = []
    for _, row in selected_rows.iterrows():
        run_row, node_decisions = _replay_one_matrix(
            row=row,
            matrix_path=_matrix_path(row, generated_matrix_dir),
            method_id=str(method_id),
            sibling_alpha=float(sibling_alpha),
            edge_alpha=float(edge_alpha),
            tree_linkage_method=str(tree_linkage_method),
        )
        run_records.append(run_row)
        if not node_decisions.empty:
            node_frames.append(node_decisions)

    run_rows = pd.DataFrame.from_records(run_records, columns=RUN_ROW_COLUMNS)
    node_decisions = pd.concat(node_frames, ignore_index=True) if node_frames else pd.DataFrame()
    if node_decisions.empty:
        empty = pd.DataFrame()
        return {
            "run_rows": run_rows,
            "node_decisions": empty,
            "distribution_rows": empty,
            "pvalue_rows": empty,
            "pvalue_summary": empty,
            "pvalue_case_summary": empty,
            "pvalue_tau_s_sensitivity": empty,
            "measurability_rows": empty,
            "measurability_summary": empty,
            "topology_rows": empty,
            "topology_summary": empty,
        }

    distribution_rows = build_selected_neighborhood_distribution_rows(
        node_decisions=node_decisions,
    )
    pvalue_rows = build_pvalue_interpolation_comparison_rows(
        distribution_rows,
        candidate_only=bool(candidate_only),
    )
    pvalue_summary = summarize_pvalue_interpolation_comparison(pvalue_rows)
    pvalue_case_summary = summarize_pvalue_interpolation_cases(pvalue_rows)
    pvalue_tau_s_sensitivity = summarize_tau_s_sensitivity(pvalue_rows)
    enriched_input = enrich_measurability_input_rows(
        distribution_rows,
        pvalue_rows=pvalue_rows,
    )
    measurability_rows = build_measurability_law_rows(enriched_input)
    measurability_summary = summarize_measurability_law_rows(measurability_rows)
    topology_rows = build_topology_frontier_rows(
        measurability_rows,
        root_selected_region_summary=proposal_rows,
        reference_tau_s=float(reference_tau_s),
        require_spectral_flow=bool(require_spectral_flow),
    )
    topology_summary = summarize_topology_frontier_rows(topology_rows)
    run_rows = _update_run_rows_with_frontier(run_rows, topology_rows)

    return {
        "run_rows": run_rows,
        "node_decisions": _attach_metadata(node_decisions, proposal_rows),
        "distribution_rows": _attach_metadata(distribution_rows, proposal_rows),
        "pvalue_rows": _attach_metadata(pvalue_rows, proposal_rows),
        "pvalue_summary": pvalue_summary,
        "pvalue_case_summary": _attach_metadata(pvalue_case_summary, proposal_rows),
        "pvalue_tau_s_sensitivity": pvalue_tau_s_sensitivity,
        "measurability_rows": _attach_metadata(measurability_rows, proposal_rows),
        "measurability_summary": measurability_summary,
        "topology_rows": _attach_metadata(topology_rows, proposal_rows),
        "topology_summary": topology_summary,
    }


def run_root_tie_rank_generated_neighborhood_replay(
    config: RootTieRankGeneratedNeighborhoodReplayConfig,
) -> dict[str, Path]:
    """Run generated-neighborhood replay and write diagnostic artifacts."""
    proposal_rows = pd.read_csv(config.proposal_feasibility_rows_path, low_memory=False)
    tables = build_generated_neighborhood_replay_tables(
        proposal_rows=proposal_rows,
        generated_matrix_dir=config.generated_matrix_dir,
        method_id=config.method_id,
        sibling_alpha=float(config.sibling_alpha),
        edge_alpha=float(config.edge_alpha),
        tree_linkage_method=config.tree_linkage_method,
        max_cases=config.max_cases,
        case_ids=tuple(config.case_ids),
        candidate_only=bool(config.candidate_only),
        reference_tau_s=float(config.reference_tau_s),
        require_spectral_flow=bool(config.require_spectral_flow),
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "run_rows": config.output_dir / RUN_ROWS_OUTPUT,
        "node_decisions": config.output_dir / NODE_DECISIONS_OUTPUT,
        "distribution_rows": config.output_dir / DISTRIBUTION_ROWS_OUTPUT,
        "pvalue_rows": config.output_dir / PVALUE_ROWS_OUTPUT,
        "pvalue_summary": config.output_dir / PVALUE_SUMMARY_OUTPUT,
        "pvalue_case_summary": config.output_dir / PVALUE_CASE_SUMMARY_OUTPUT,
        "pvalue_tau_s_sensitivity": config.output_dir / PVALUE_TAU_S_OUTPUT,
        "measurability_rows": config.output_dir / MEASURABILITY_ROWS_OUTPUT,
        "measurability_summary": config.output_dir / MEASURABILITY_SUMMARY_OUTPUT,
        "topology_rows": config.output_dir / TOPOLOGY_ROWS_OUTPUT,
        "topology_summary": config.output_dir / TOPOLOGY_SUMMARY_OUTPUT,
    }
    for key, path in outputs.items():
        tables[key].to_csv(path, index=False)

    manifest_path = config.output_dir / MANIFEST_OUTPUT
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "created_at": datetime.now(UTC).isoformat(),
        "config": config,
        "outputs": {key: str(path) for key, path in outputs.items()},
        "rows": {key: int(value.shape[0]) for key, value in tables.items()},
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    outputs["manifest"] = manifest_path
    return outputs


def main() -> None:
    args = parse_args()
    run_root_tie_rank_generated_neighborhood_replay(
        RootTieRankGeneratedNeighborhoodReplayConfig(
            output_dir=args.output_dir,
            proposal_feasibility_rows_path=args.proposal_feasibility_rows_path,
            generated_matrix_dir=args.generated_matrix_dir,
            method_id=args.method_id,
            sibling_alpha=args.sibling_alpha,
            edge_alpha=args.edge_alpha,
            tree_linkage_method=args.tree_linkage_method,
            max_cases=args.max_cases,
            case_ids=tuple(args.case_ids),
            candidate_only=args.candidate_only,
            reference_tau_s=args.reference_tau_s,
            require_spectral_flow=args.require_spectral_flow,
        )
    )


if __name__ == "__main__":
    main()
