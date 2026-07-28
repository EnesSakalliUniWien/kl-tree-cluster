"""Coordinate-gap panel for root tie-rank proposal frontiers.

The proposal frontier can generate large root action, sparse spectral spikes,
or coupled perturbations, but target-stratum hits require the coordinates to
co-occur. This panel compares observed root strata against generated proposal
rows and reports which coordinates remain unmatched.
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

SCHEMA_VERSION = "root_tie_rank_proposal_gap_panel/v1"
STUDY_ROLE = "diagnostic_root_tie_rank_proposal_gap_panel_not_calibration"
GENERATED_BY = "benchmarks.diagnostics.calibration.root.tie_rank.root_tie_rank_proposal_gap_panel"

DEFAULT_RESULT_ROOT = Path("raw/assets/benchmark-results/specific_small_method_benchmark_20260615")
DEFAULT_PROPOSAL_FEASIBILITY_ROWS = (
    DEFAULT_RESULT_ROOT
    / "root_tie_rank_null_proposal_frontier_two_case_smoke"
    / "root_tie_rank_null_proposal_combined_feasibility_rows.csv"
)

ROWS_OUTPUT = "root_tie_rank_proposal_gap_rows.csv"
SUMMARY_OUTPUT = "root_tie_rank_proposal_gap_summary.csv"
MANIFEST_OUTPUT = "manifest.json"

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "target_case_id",
    "proposal_family",
    "best_generated_case_id",
    "target_root_conditioning_stratum_key",
    "generated_root_conditioning_stratum_key",
    "exact_stratum_hit",
    "target_root_tie_rank_band",
    "generated_root_tie_rank_band",
    "tie_band_match",
    "target_root_edge_margin_band",
    "generated_root_edge_margin_band",
    "edge_band_match",
    "target_root_spectral_ratio_band",
    "generated_root_spectral_ratio_band",
    "spectral_band_match",
    "target_root_bandwidth_reopen_band",
    "generated_root_bandwidth_reopen_band",
    "bandwidth_band_match",
    "bandwidth_gap_status",
    "target_root_sibling_selected_ratio",
    "generated_root_sibling_selected_ratio",
    "selected_ratio_exceeds_target",
    "target_root_edge_path_statistic_margin",
    "generated_root_edge_path_statistic_margin",
    "target_root_selected_eigenvalue_over_mp_upper_bound",
    "generated_root_selected_eigenvalue_over_mp_upper_bound",
    "tie_fraction_gap",
    "selected_ratio_log_gap",
    "edge_log_gap",
    "spectral_log_gap",
    "band_mismatch_count",
    "joint_tie_edge_spectral_band_match",
    "action_edge_without_spectral",
    "spectral_without_edge_action",
    "best_gap_score",
    "gap_pattern",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "proposal_family",
    "target_count",
    "exact_stratum_hit_count",
    "target_case_selected_ratio_exceed_count",
    "edge_band_match_count",
    "spectral_band_match_count",
    "joint_tie_edge_spectral_band_match_count",
    "bandwidth_missing_count",
    "action_edge_without_spectral_count",
    "spectral_without_edge_action_count",
    "median_best_gap_score",
    "min_best_gap_score",
    "summary_status",
)


@dataclass(frozen=True)
class RootTieRankProposalGapPanelConfig:
    """Input/output paths for the root tie-rank proposal gap panel."""

    output_dir: Path
    proposal_feasibility_rows_path: Path = DEFAULT_PROPOSAL_FEASIBILITY_ROWS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--proposal-feasibility-rows-path",
        type=Path,
        default=DEFAULT_PROPOSAL_FEASIBILITY_ROWS,
    )
    return parser.parse_args()


def _json_default(value: object) -> object:
    if isinstance(value, RootTieRankProposalGapPanelConfig):
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


def _positive_log1p(value: object) -> float:
    numeric = _finite_float(value)
    if not math.isfinite(numeric):
        return math.nan
    return float(math.log1p(max(numeric, 0.0)))


def _positive_log(value: object) -> float:
    numeric = _finite_float(value)
    if not math.isfinite(numeric):
        return math.nan
    return float(math.log(max(numeric, 1e-12)))


def _abs_gap(left: float, right: float, *, missing_value: float = 10.0) -> float:
    if not (math.isfinite(left) and math.isfinite(right)):
        return float(missing_value)
    return float(abs(left - right))


def _string_value(row: pd.Series, column: str, default: str = "") -> str:
    if column not in row:
        return default
    value = row[column]
    if pd.isna(value):
        return default
    return str(value)


def _is_observed_target(row: pd.Series) -> bool:
    family = _string_value(row, "proposal_family", "")
    role = _string_value(row, "calibration_role", "")
    data_role = _string_value(row, "data_role", "")
    return (
        family == "observed_target"
        or role == "observed_target_not_null_support"
        or data_role == "observed_target"
    )


def _bandwidth_gap_status(*, target_band: str, generated_band: str) -> str:
    if target_band == generated_band:
        return "bandwidth_band_match"
    if generated_band == "bandwidth_reopen_missing":
        return "generated_bandwidth_unmeasured"
    if target_band == "bandwidth_reopen_missing":
        return "target_bandwidth_unmeasured"
    return "bandwidth_band_mismatch"


def _gap_pattern(
    *,
    exact_hit: bool,
    tie_match: bool,
    edge_match: bool,
    spectral_match: bool,
    bandwidth_status: str,
    ratio_exceeds: bool,
) -> str:
    if exact_hit:
        return "exact_target_stratum_hit"
    if edge_match and ratio_exceeds and not spectral_match:
        return "action_edge_without_spectral"
    if spectral_match and not (edge_match and ratio_exceeds):
        return "spectral_without_edge_action"
    if tie_match and edge_match and spectral_match:
        if bandwidth_status == "generated_bandwidth_unmeasured":
            return "tie_edge_spectral_match_bandwidth_unmeasured"
        return "tie_edge_spectral_match_bandwidth_mismatch"
    if bandwidth_status == "generated_bandwidth_unmeasured":
        return "partial_match_bandwidth_unmeasured"
    return "partial_or_no_coordinate_match"


def _row_gap_record(target: pd.Series, generated: pd.Series) -> dict[str, object]:
    target_stratum = _string_value(target, "root_conditioning_stratum_key")
    generated_stratum = _string_value(generated, "root_conditioning_stratum_key")
    target_tie_band = _string_value(target, "root_tie_rank_band")
    generated_tie_band = _string_value(generated, "root_tie_rank_band")
    target_edge_band = _string_value(target, "root_edge_margin_band")
    generated_edge_band = _string_value(generated, "root_edge_margin_band")
    target_spectral_band = _string_value(target, "root_spectral_ratio_band")
    generated_spectral_band = _string_value(generated, "root_spectral_ratio_band")
    target_bandwidth_band = _string_value(target, "root_bandwidth_reopen_band")
    generated_bandwidth_band = _string_value(generated, "root_bandwidth_reopen_band")

    tie_match = target_tie_band == generated_tie_band
    edge_match = target_edge_band == generated_edge_band
    spectral_match = target_spectral_band == generated_spectral_band
    bandwidth_match = target_bandwidth_band == generated_bandwidth_band
    bandwidth_status = _bandwidth_gap_status(
        target_band=target_bandwidth_band,
        generated_band=generated_bandwidth_band,
    )
    exact_hit = target_stratum == generated_stratum
    target_ratio = _finite_float(target.get("root_sibling_selected_ratio", math.nan))
    generated_ratio = _finite_float(generated.get("root_sibling_selected_ratio", math.nan))
    ratio_exceeds = (
        math.isfinite(target_ratio)
        and math.isfinite(generated_ratio)
        and generated_ratio >= target_ratio
    )
    tie_fraction_gap = _abs_gap(
        _finite_float(target.get("root_tie_rank_median_fraction", math.nan)),
        _finite_float(generated.get("root_tie_rank_median_fraction", math.nan)),
        missing_value=1.0,
    )
    selected_ratio_log_gap = _abs_gap(
        _positive_log1p(target_ratio),
        _positive_log1p(generated_ratio),
    )
    edge_log_gap = _abs_gap(
        _positive_log1p(target.get("root_edge_path_statistic_margin", math.nan)),
        _positive_log1p(generated.get("root_edge_path_statistic_margin", math.nan)),
    )
    spectral_log_gap = _abs_gap(
        _positive_log(target.get("root_selected_eigenvalue_over_mp_upper_bound", math.nan)),
        _positive_log(generated.get("root_selected_eigenvalue_over_mp_upper_bound", math.nan)),
    )
    band_mismatch_count = int(
        sum(
            [
                not tie_match,
                not edge_match,
                not spectral_match,
                not bandwidth_match,
            ]
        )
    )
    score = (
        100.0 * float(band_mismatch_count)
        + tie_fraction_gap
        + min(selected_ratio_log_gap, 10.0)
        + min(edge_log_gap, 10.0)
        + min(spectral_log_gap, 10.0)
    )
    joint_match = bool(tie_match and edge_match and spectral_match)
    action_edge_without_spectral = bool(edge_match and ratio_exceeds and not spectral_match)
    spectral_without_edge_action = bool(spectral_match and not (edge_match and ratio_exceeds))
    pattern = _gap_pattern(
        exact_hit=exact_hit,
        tie_match=tie_match,
        edge_match=edge_match,
        spectral_match=spectral_match,
        bandwidth_status=bandwidth_status,
        ratio_exceeds=ratio_exceeds,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "target_case_id": _string_value(target, "case_id"),
        "proposal_family": _string_value(generated, "proposal_family"),
        "best_generated_case_id": _string_value(generated, "case_id"),
        "target_root_conditioning_stratum_key": target_stratum,
        "generated_root_conditioning_stratum_key": generated_stratum,
        "exact_stratum_hit": exact_hit,
        "target_root_tie_rank_band": target_tie_band,
        "generated_root_tie_rank_band": generated_tie_band,
        "tie_band_match": tie_match,
        "target_root_edge_margin_band": target_edge_band,
        "generated_root_edge_margin_band": generated_edge_band,
        "edge_band_match": edge_match,
        "target_root_spectral_ratio_band": target_spectral_band,
        "generated_root_spectral_ratio_band": generated_spectral_band,
        "spectral_band_match": spectral_match,
        "target_root_bandwidth_reopen_band": target_bandwidth_band,
        "generated_root_bandwidth_reopen_band": generated_bandwidth_band,
        "bandwidth_band_match": bandwidth_match,
        "bandwidth_gap_status": bandwidth_status,
        "target_root_sibling_selected_ratio": target_ratio,
        "generated_root_sibling_selected_ratio": generated_ratio,
        "selected_ratio_exceeds_target": ratio_exceeds,
        "target_root_edge_path_statistic_margin": _finite_float(
            target.get("root_edge_path_statistic_margin", math.nan)
        ),
        "generated_root_edge_path_statistic_margin": _finite_float(
            generated.get("root_edge_path_statistic_margin", math.nan)
        ),
        "target_root_selected_eigenvalue_over_mp_upper_bound": _finite_float(
            target.get("root_selected_eigenvalue_over_mp_upper_bound", math.nan)
        ),
        "generated_root_selected_eigenvalue_over_mp_upper_bound": _finite_float(
            generated.get("root_selected_eigenvalue_over_mp_upper_bound", math.nan)
        ),
        "tie_fraction_gap": tie_fraction_gap,
        "selected_ratio_log_gap": selected_ratio_log_gap,
        "edge_log_gap": edge_log_gap,
        "spectral_log_gap": spectral_log_gap,
        "band_mismatch_count": band_mismatch_count,
        "joint_tie_edge_spectral_band_match": joint_match,
        "action_edge_without_spectral": action_edge_without_spectral,
        "spectral_without_edge_action": spectral_without_edge_action,
        "best_gap_score": float(score),
        "gap_pattern": pattern,
    }


def build_root_tie_rank_proposal_gap_rows(
    combined_feasibility_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Return best generated proposal row for each target and family."""
    _require_columns(
        combined_feasibility_rows,
        {
            "case_id",
            "calibration_role",
            "proposal_family",
            "root_conditioning_stratum_key",
            "root_tie_rank_band",
            "root_edge_margin_band",
            "root_spectral_ratio_band",
            "root_bandwidth_reopen_band",
            "root_sibling_selected_ratio",
            "root_tie_rank_median_fraction",
            "root_edge_path_statistic_margin",
            "root_selected_eigenvalue_over_mp_upper_bound",
        },
        "combined feasibility rows",
    )
    rows = combined_feasibility_rows.copy()
    target_mask = rows.apply(_is_observed_target, axis=1)
    targets = rows[target_mask].copy()
    generated = rows[~target_mask].copy()
    records: list[dict[str, object]] = []
    for _, target in targets.sort_values("case_id").iterrows():
        for family, family_rows in generated.groupby("proposal_family", sort=True):
            candidate_records = [
                _row_gap_record(target, generated_row)
                for _, generated_row in family_rows.iterrows()
            ]
            if not candidate_records:
                continue
            best = min(
                candidate_records,
                key=lambda record: (
                    float(record["best_gap_score"]),
                    str(record["best_generated_case_id"]),
                ),
            )
            best["proposal_family"] = str(family)
            records.append(best)
    return pd.DataFrame.from_records(records, columns=ROW_COLUMNS)


def summarize_root_tie_rank_proposal_gap_rows(gap_rows: pd.DataFrame) -> pd.DataFrame:
    """Return proposal-family summary over best target gaps."""
    if gap_rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    records: list[dict[str, object]] = []
    for family, group in gap_rows.groupby("proposal_family", sort=True):
        exact_hits = int(group["exact_stratum_hit"].sum())
        action_edge = int(group["action_edge_without_spectral"].sum())
        spectral_only = int(group["spectral_without_edge_action"].sum())
        joint = int(group["joint_tie_edge_spectral_band_match"].sum())
        bandwidth_missing = int(
            group["bandwidth_gap_status"].eq("generated_bandwidth_unmeasured").sum()
        )
        if exact_hits > 0:
            status = "proposal_hits_observed_target_strata"
        elif action_edge > 0 and spectral_only > 0 and joint == 0:
            status = "separable_action_and_spectral_no_joint_match"
        elif action_edge > 0:
            status = "action_edge_without_spectral"
        elif spectral_only > 0:
            status = "spectral_without_edge_action"
        elif bandwidth_missing == int(group.shape[0]):
            status = "bandwidth_unmeasured_no_coordinate_match"
        else:
            status = "no_target_coordinate_match"
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "proposal_family": str(family),
                "target_count": int(group.shape[0]),
                "exact_stratum_hit_count": exact_hits,
                "target_case_selected_ratio_exceed_count": int(
                    group["selected_ratio_exceeds_target"].sum()
                ),
                "edge_band_match_count": int(group["edge_band_match"].sum()),
                "spectral_band_match_count": int(group["spectral_band_match"].sum()),
                "joint_tie_edge_spectral_band_match_count": joint,
                "bandwidth_missing_count": bandwidth_missing,
                "action_edge_without_spectral_count": action_edge,
                "spectral_without_edge_action_count": spectral_only,
                "median_best_gap_score": float(group["best_gap_score"].median()),
                "min_best_gap_score": float(group["best_gap_score"].min()),
                "summary_status": status,
            }
        )
    return pd.DataFrame.from_records(records, columns=SUMMARY_COLUMNS)


def evaluate_root_tie_rank_proposal_gap_panel(
    config: RootTieRankProposalGapPanelConfig,
) -> dict[str, pd.DataFrame]:
    """Read proposal frontier rows and return gap tables."""
    combined = pd.read_csv(config.proposal_feasibility_rows_path)
    rows = build_root_tie_rank_proposal_gap_rows(combined)
    summary = summarize_root_tie_rank_proposal_gap_rows(rows)
    return {"rows": rows, "summary": summary}


def run_root_tie_rank_proposal_gap_panel(
    config: RootTieRankProposalGapPanelConfig,
) -> dict[str, Path]:
    """Run the gap panel and write outputs."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = evaluate_root_tie_rank_proposal_gap_panel(config)
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
    outputs = run_root_tie_rank_proposal_gap_panel(
        RootTieRankProposalGapPanelConfig(
            output_dir=args.output_dir,
            proposal_feasibility_rows_path=args.proposal_feasibility_rows_path,
        )
    )
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
