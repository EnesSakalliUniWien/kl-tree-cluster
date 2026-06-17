"""Join conditioned coherent topology replay into spectral generator targets.

The conditioned coherent spectral-lift sweep writes generated root rows before
selected-neighborhood topology replay. This diagnostic attaches replayed
root-frontier evidence to those rows, appends them to the existing proposal
feasibility artifact, and reruns the selected spectral generator target panel.

The output is diagnostic-only. Joined proposal rows remain external-null
unsupported unless their calibration role already says otherwise.
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

from benchmarks.diagnostics.calibration.root_selected_mixed_region_law import (
    _aggregate_root_frontier,
)
from benchmarks.diagnostics.calibration.root_tie_rank_selected_spectral_generator_target_panel import (
    DEFAULT_PROPOSAL_FEASIBILITY_ROWS,
    ROWS_OUTPUT,
    SUMMARY_OUTPUT,
    build_selected_spectral_generator_target_rows,
    summarize_selected_spectral_generator_target_rows,
)

SCHEMA_VERSION = "root_tie_rank_conditioned_coherent_topology_join/v1"
STUDY_ROLE = "diagnostic_root_tie_rank_conditioned_coherent_topology_join"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration."
    "root_tie_rank_conditioned_coherent_topology_join"
)

DEFAULT_RESULT_ROOT = Path(
    "raw/assets/benchmark-results/specific_small_method_benchmark_20260615"
)
DEFAULT_CONDITIONED_GENERATED_ROWS = (
    DEFAULT_RESULT_ROOT
    / "root_tie_rank_spectral_lift_parameter_sweep_overlap_mod6_conditioned_coherent_capped_smoke"
    / "root_tie_rank_spectral_lift_sweep_generated_rows.csv"
)
DEFAULT_CONDITIONED_TOPOLOGY_ROWS = (
    DEFAULT_RESULT_ROOT
    / "root_tie_rank_generated_neighborhood_replay_conditioned_coherent_capped_smoke"
    / "generated_selected_neighborhood_topology_frontier_rows.csv"
)
DEFAULT_OBSERVED_ROOT_SUMMARY_ROWS = (
    DEFAULT_RESULT_ROOT
    / "root_selected_region_margins_overlap_case_family"
    / "root_selected_region_summary.csv"
)

JOINED_ROWS_OUTPUT = "conditioned_coherent_joined_feasibility_rows.csv"
MANIFEST_OUTPUT = "manifest.json"

OBSERVED_ROOT_SPECTRAL_COLUMNS = (
    "root_raw_mp_signal_count",
    "root_effective_independent_rows",
    "root_mp_threshold_rows",
    "root_active_feature_count",
    "root_full_eigenvalue_count",
    "root_full_component_eigenvalues_json",
    "root_projected_eigenvalues_json",
    "root_eigenvalue_effective_rank",
    "root_top_eigenvalue_share",
    "root_selected_eigenvalue_mass_fraction",
    "root_mp_upper_bound",
    "root_selected_eigenvalue_over_mp_upper_bound",
)


@dataclass(frozen=True)
class RootTieRankConditionedCoherentTopologyJoinConfig:
    """Input/output contract for conditioned coherent topology joins."""

    output_dir: Path
    base_feasibility_rows_path: Path = DEFAULT_PROPOSAL_FEASIBILITY_ROWS
    conditioned_generated_rows_path: Path = DEFAULT_CONDITIONED_GENERATED_ROWS
    conditioned_topology_rows_path: Path = DEFAULT_CONDITIONED_TOPOLOGY_ROWS
    observed_root_summary_rows_path: Path | None = DEFAULT_OBSERVED_ROOT_SUMMARY_ROWS
    min_action_edge_fraction: float = 0.95
    min_tie_fraction_floor: float = 0.70


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--base-feasibility-rows-path",
        type=Path,
        default=DEFAULT_PROPOSAL_FEASIBILITY_ROWS,
    )
    parser.add_argument(
        "--conditioned-generated-rows-path",
        type=Path,
        default=DEFAULT_CONDITIONED_GENERATED_ROWS,
    )
    parser.add_argument(
        "--conditioned-topology-rows-path",
        type=Path,
        default=DEFAULT_CONDITIONED_TOPOLOGY_ROWS,
    )
    parser.add_argument(
        "--observed-root-summary-rows-path",
        type=Path,
        default=DEFAULT_OBSERVED_ROOT_SUMMARY_ROWS,
    )
    parser.add_argument("--min-action-edge-fraction", type=float, default=0.95)
    parser.add_argument("--min-tie-fraction-floor", type=float, default=0.70)
    return parser.parse_args()


def _json_default(value: object) -> object:
    if isinstance(value, RootTieRankConditionedCoherentTopologyJoinConfig):
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


def _finite_float(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return math.nan
    return numeric if math.isfinite(numeric) else math.nan


def _bandwidth_reopen_band(*, locality_status: str, reopen_count: float) -> str:
    if str(locality_status) == "topology_frontier_not_joined":
        return "bandwidth_reopen_missing"
    if not math.isfinite(float(reopen_count)):
        return "bandwidth_reopen_missing"
    if float(reopen_count) <= 0.0:
        return "bandwidth_no_root_reopen"
    return "bandwidth_root_reopen_observed"


def _apply_root_frontier_join(
    *,
    generated_rows: pd.DataFrame,
    topology_rows: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(generated_rows, {"case_id"}, "conditioned generated rows")
    root_frontier = _aggregate_root_frontier(topology_rows)
    joined = generated_rows.copy()
    for column in ("root_frontier_data_roles", "root_bandwidth_locality_status"):
        if column not in joined.columns:
            joined[column] = ""
        else:
            joined[column] = joined[column].fillna("").astype(str)
    for index, row in joined.iterrows():
        case_id = str(row["case_id"])
        frontier = root_frontier.get(case_id, {})
        frontier_row_count = int(frontier.get("root_frontier_row_count", 0))
        reopen_count = int(frontier.get("root_bandwidth_reopen_count", 0))
        hybrid_count = int(frontier.get("root_hybrid_strict_support_count", 0))
        if frontier_row_count <= 0:
            locality_status = "topology_frontier_not_joined"
        elif hybrid_count > 0:
            locality_status = "hybrid_support_observed_diagnostic_only"
        elif reopen_count > 0:
            locality_status = "bandwidth_reopens_without_root_law_support"
        else:
            locality_status = "bandwidth_does_not_reopen_root_rows"
        updates = {
            "root_frontier_row_count": frontier_row_count,
            "root_frontier_data_roles": str(
                frontier.get("root_frontier_data_roles", "")
            ),
            "root_bandwidth_reopen_count": reopen_count,
            "root_bandwidth_direct_positive_reopen_count": int(
                frontier.get("root_bandwidth_direct_positive_reopen_count", 0)
            ),
            "root_structural_proxy_pass_count": int(
                frontier.get("root_structural_proxy_pass_count", 0)
            ),
            "root_hybrid_strict_support_count": hybrid_count,
            "root_frontier_min_best_case_tau_s": _finite_float(
                frontier.get("root_frontier_min_best_case_tau_s", math.nan)
            ),
            "root_frontier_median_best_case_tau_s": _finite_float(
                frontier.get("root_frontier_median_best_case_tau_s", math.nan)
            ),
            "root_bandwidth_locality_status": locality_status,
            "root_bandwidth_reopen_band": _bandwidth_reopen_band(
                locality_status=locality_status,
                reopen_count=reopen_count,
            ),
        }
        for column, value in updates.items():
            joined.at[index, column] = value
    return joined


def _enrich_observed_root_summary(
    *,
    feasibility_rows: pd.DataFrame,
    observed_root_summary_rows: pd.DataFrame | None,
) -> pd.DataFrame:
    """Attach observed-root spectral capture fields to observed target rows."""
    if observed_root_summary_rows is None or observed_root_summary_rows.empty:
        return feasibility_rows
    _require_columns(
        observed_root_summary_rows,
        {"case_id"},
        "observed root summary rows",
    )
    summary_by_case = {
        str(row["case_id"]): row for _, row in observed_root_summary_rows.iterrows()
    }
    enriched = feasibility_rows.copy()
    if "data_role" not in enriched.columns:
        return enriched
    observed_mask = enriched["data_role"].astype(str).eq("observed_target")
    for index, row in enriched.loc[observed_mask].iterrows():
        case_id = str(row["case_id"])
        summary = summary_by_case.get(case_id)
        if summary is None:
            continue
        for column in OBSERVED_ROOT_SPECTRAL_COLUMNS:
            if column in summary:
                enriched.at[index, column] = summary[column]
    return enriched


def build_conditioned_coherent_joined_feasibility_rows(
    *,
    base_feasibility_rows: pd.DataFrame,
    conditioned_generated_rows: pd.DataFrame,
    conditioned_topology_rows: pd.DataFrame,
    observed_root_summary_rows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Append topology-joined generated/support rows to feasibility rows."""
    _require_columns(
        base_feasibility_rows,
        {"case_id", "data_role", "calibration_role", "proposal_family"},
        "base feasibility rows",
    )
    _require_columns(
        conditioned_generated_rows,
        {"case_id", "data_role", "calibration_role", "proposal_family"},
        "conditioned generated rows",
    )
    generated_only = conditioned_generated_rows.loc[
        ~conditioned_generated_rows["proposal_family"].astype(str).eq(
            "observed_target"
        )
        & ~conditioned_generated_rows["data_role"].astype(str).eq("observed_target")
    ].copy()
    joined_generated = _apply_root_frontier_join(
        generated_rows=generated_only,
        topology_rows=conditioned_topology_rows,
    )
    combined = pd.concat(
        [base_feasibility_rows.copy(), joined_generated],
        ignore_index=True,
        sort=False,
    )
    deduplicated = combined.drop_duplicates("case_id", keep="last")
    return _enrich_observed_root_summary(
        feasibility_rows=deduplicated,
        observed_root_summary_rows=observed_root_summary_rows,
    )


def evaluate_conditioned_coherent_topology_join(
    config: RootTieRankConditionedCoherentTopologyJoinConfig,
) -> dict[str, pd.DataFrame]:
    """Return joined rows plus refreshed spectral generator target tables."""
    base_rows = pd.read_csv(config.base_feasibility_rows_path, low_memory=False)
    generated_rows = pd.read_csv(
        config.conditioned_generated_rows_path,
        low_memory=False,
    )
    topology_rows = pd.read_csv(
        config.conditioned_topology_rows_path,
        low_memory=False,
    )
    observed_root_summary_rows = (
        pd.read_csv(config.observed_root_summary_rows_path, low_memory=False)
        if config.observed_root_summary_rows_path is not None
        and Path(config.observed_root_summary_rows_path).exists()
        else None
    )
    joined_rows = build_conditioned_coherent_joined_feasibility_rows(
        base_feasibility_rows=base_rows,
        conditioned_generated_rows=generated_rows,
        conditioned_topology_rows=topology_rows,
        observed_root_summary_rows=observed_root_summary_rows,
    )
    target_rows = build_selected_spectral_generator_target_rows(
        joined_rows,
        min_action_edge_fraction=float(config.min_action_edge_fraction),
        min_tie_fraction_floor=float(config.min_tie_fraction_floor),
    )
    summary = summarize_selected_spectral_generator_target_rows(target_rows)
    return {
        "joined_rows": joined_rows,
        "rows": target_rows,
        "summary": summary,
    }


def run_conditioned_coherent_topology_join(
    config: RootTieRankConditionedCoherentTopologyJoinConfig,
) -> dict[str, Path]:
    """Run the join and write joined feasibility and target-panel outputs."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    tables = evaluate_conditioned_coherent_topology_join(config)
    paths = {
        "joined_rows": config.output_dir / JOINED_ROWS_OUTPUT,
        "rows": config.output_dir / ROWS_OUTPUT,
        "summary": config.output_dir / SUMMARY_OUTPUT,
    }
    for key, path in paths.items():
        tables[key].to_csv(path, index=False)
    manifest_path = config.output_dir / MANIFEST_OUTPUT
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
    outputs = run_conditioned_coherent_topology_join(
        RootTieRankConditionedCoherentTopologyJoinConfig(
            output_dir=args.output_dir,
            base_feasibility_rows_path=args.base_feasibility_rows_path,
            conditioned_generated_rows_path=args.conditioned_generated_rows_path,
            conditioned_topology_rows_path=args.conditioned_topology_rows_path,
            observed_root_summary_rows_path=args.observed_root_summary_rows_path,
            min_action_edge_fraction=float(args.min_action_edge_fraction),
            min_tie_fraction_floor=float(args.min_tie_fraction_floor),
        )
    )
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
