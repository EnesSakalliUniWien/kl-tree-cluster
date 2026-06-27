"""Population-law requirement diagnostic for selected-root spectral tails.

Action-dominating support can match T,E,B,H_u and exceed the target action A,
but still sit below the observed spectral excess S_root. This diagnostic asks:
how much would the local MP reference edge have to move for the available
support to cover the target spectral excess?

The required multiplier is

    kappa_H = exp(S_target - S_support_max)

where S_support_max is the largest supported spectral excess in the relevant
diagnostic support set. Values above one mean identity MP would need a larger
local reference edge for the target to stop looking more extreme than support.
This is a diagnostic H_u requirement, not a calibration p-value.
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

from benchmarks.diagnostics.calibration.root_selected_spectral_tail_law_panel import (
    DEFAULT_RESULT_ROOT,
    _finite_float,
)

SCHEMA_VERSION = "root_selected_population_law_requirement_panel/v1"
STUDY_ROLE = "diagnostic_root_selected_population_law_requirement_not_calibration"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration.root_selected_population_law_requirement_panel"
)

DEFAULT_ACTION_DOMINANCE_ROWS = (
    DEFAULT_RESULT_ROOT
    / "root_selected_action_dominance_tail_mild_accumulated"
    / "root_selected_action_dominance_tail_rows.csv"
)

ROWS_OUTPUT = "root_selected_population_law_requirement_rows.csv"
SUMMARY_OUTPUT = "root_selected_population_law_requirement_summary.csv"
MANIFEST_OUTPUT = "manifest.json"

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "target_case_id",
    "target_s_root_spectral_excess_log",
    "support_reference_status",
    "support_count",
    "support_s_root_max_log",
    "spectral_excess_gap_to_support_max",
    "required_mp_edge_multiplier",
    "required_mp_edge_log_multiplier",
    "required_population_law_status",
    "production_inference_status",
    "next_mathematical_step",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "target_count",
    "requirement_available_count",
    "exact_no_requirement_count",
    "modest_requirement_count",
    "substantial_requirement_count",
    "large_requirement_count",
    "missing_support_count",
    "summary_status",
)


@dataclass(frozen=True)
class RootSelectedPopulationLawRequirementConfig:
    """Input/output contract for population-law requirement diagnostics."""

    output_dir: Path
    action_dominance_rows_path: Path = DEFAULT_ACTION_DOMINANCE_ROWS
    modest_multiplier_threshold: float = 1.5
    substantial_multiplier_threshold: float = 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--action-dominance-rows-path",
        type=Path,
        default=DEFAULT_ACTION_DOMINANCE_ROWS,
    )
    parser.add_argument("--modest-multiplier-threshold", type=float, default=1.5)
    parser.add_argument("--substantial-multiplier-threshold", type=float, default=3.0)
    return parser.parse_args()


def _json_default(value: object) -> object:
    if isinstance(value, RootSelectedPopulationLawRequirementConfig):
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


def _string_value(row: pd.Series | dict[str, object], column: str, default: str = "") -> str:
    if column not in row:
        return default
    value = row[column]
    if pd.isna(value):
        return default
    return str(value)


def _requirement_status(
    multiplier: float,
    *,
    modest_threshold: float,
    substantial_threshold: float,
) -> str:
    if not math.isfinite(multiplier):
        return "population_law_requirement_missing_support"
    if multiplier == 1.0:
        return "exact_tail_support_available_no_population_law_requirement"
    if multiplier <= 1.0:
        return "identity_mp_support_exceeds_target"
    if multiplier <= float(modest_threshold):
        return "modest_population_law_multiplier_required"
    if multiplier <= float(substantial_threshold):
        return "substantial_population_law_multiplier_required"
    return "large_population_law_multiplier_required"


def build_root_selected_population_law_requirement_rows(
    *,
    action_dominance_rows: pd.DataFrame,
    modest_multiplier_threshold: float = 1.5,
    substantial_multiplier_threshold: float = 3.0,
) -> pd.DataFrame:
    """Build required MP-edge multiplier rows from action-dominance diagnostics."""
    _require_columns(
        action_dominance_rows,
        {
            "target_case_id",
            "target_s_root_spectral_excess_log",
            "exact_support_count",
            "action_dominating_support_count",
            "best_action_dominating_support_s_root_log",
        },
        "action dominance rows",
    )
    records: list[dict[str, object]] = []
    for _, row in action_dominance_rows.sort_values("target_case_id").iterrows():
        target_s = _finite_float(row["target_s_root_spectral_excess_log"])
        exact_count = int(_finite_float(row.get("exact_support_count", 0)))
        action_count = int(_finite_float(row.get("action_dominating_support_count", 0)))
        support_count = action_count if action_count > 0 else exact_count
        support_s = _finite_float(row.get("best_action_dominating_support_s_root_log", math.nan))
        if support_count > 0 and math.isfinite(target_s) and math.isfinite(support_s):
            gap = max(float(target_s - support_s), 0.0)
            multiplier = float(math.exp(gap))
            log_multiplier = gap
        elif exact_count > 0:
            gap = 0.0
            multiplier = 1.0
            log_multiplier = 0.0
        else:
            gap = math.nan
            multiplier = math.nan
            log_multiplier = math.nan
        status = _requirement_status(
            multiplier,
            modest_threshold=modest_multiplier_threshold,
            substantial_threshold=substantial_multiplier_threshold,
        )
        if action_count > 0 and exact_count <= 0:
            support_status = "action_dominating_support"
        elif exact_count > 0:
            support_status = "exact_tail_support"
        else:
            support_status = "support_missing"
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "target_case_id": _string_value(row, "target_case_id"),
                "target_s_root_spectral_excess_log": target_s,
                "support_reference_status": support_status,
                "support_count": support_count,
                "support_s_root_max_log": support_s,
                "spectral_excess_gap_to_support_max": gap,
                "required_mp_edge_multiplier": multiplier,
                "required_mp_edge_log_multiplier": log_multiplier,
                "required_population_law_status": status,
                "production_inference_status": (
                    "exact_support_available_defer_to_root_tail_panel"
                    if exact_count > 0
                    else "fail_closed_population_law_requirement_diagnostic_only"
                ),
                "next_mathematical_step": (
                    "estimate_local_H_u_or_selected_spectral_tail_law"
                    if support_count > 0 and exact_count <= 0
                    else "use_exact_root_tail_panel"
                    if exact_count > 0
                    else "generate_support_before_estimating_H_u"
                ),
            }
        )
    return pd.DataFrame.from_records(records, columns=ROW_COLUMNS)


def summarize_root_selected_population_law_requirement_rows(
    rows: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize required population-law multipliers."""
    if rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    status = rows["required_population_law_status"].astype(str)
    missing = int(status.eq("population_law_requirement_missing_support").sum())
    exact_no_requirement = int(
        status.eq("exact_tail_support_available_no_population_law_requirement").sum()
    )
    modest = int(status.eq("modest_population_law_multiplier_required").sum())
    substantial = int(status.eq("substantial_population_law_multiplier_required").sum())
    large = int(status.eq("large_population_law_multiplier_required").sum())
    available = int(rows.shape[0] - missing)
    return pd.DataFrame.from_records(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "target_count": int(rows.shape[0]),
                "requirement_available_count": available,
                "exact_no_requirement_count": exact_no_requirement,
                "modest_requirement_count": modest,
                "substantial_requirement_count": substantial,
                "large_requirement_count": large,
                "missing_support_count": missing,
                "summary_status": (
                    "population_law_requirement_quantified_for_supported_contexts"
                    if available
                    else "population_law_requirement_missing_support"
                ),
            }
        ],
        columns=SUMMARY_COLUMNS,
    )


def evaluate_root_selected_population_law_requirement_panel(
    config: RootSelectedPopulationLawRequirementConfig,
) -> dict[str, pd.DataFrame]:
    action_rows = pd.read_csv(config.action_dominance_rows_path)
    rows = build_root_selected_population_law_requirement_rows(
        action_dominance_rows=action_rows,
        modest_multiplier_threshold=config.modest_multiplier_threshold,
        substantial_multiplier_threshold=config.substantial_multiplier_threshold,
    )
    summary = summarize_root_selected_population_law_requirement_rows(rows)
    return {"rows": rows, "summary": summary}


def run_root_selected_population_law_requirement_panel(
    config: RootSelectedPopulationLawRequirementConfig,
) -> dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    tables = evaluate_root_selected_population_law_requirement_panel(config)
    paths = {
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
    outputs = run_root_selected_population_law_requirement_panel(
        RootSelectedPopulationLawRequirementConfig(
            output_dir=args.output_dir,
            action_dominance_rows_path=args.action_dominance_rows_path,
            modest_multiplier_threshold=float(args.modest_multiplier_threshold),
            substantial_multiplier_threshold=float(args.substantial_multiplier_threshold),
        )
    )
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
