"""Selected-root validity replay diagnostic.

The selected-root spectral-tail panels answer a conditional question:

    given the selected root topology, is the spectral tail supported?

This panel keeps that question separate from root validity. It joins selected
root-tail rows to root replay evidence such as feature-subsample root stability,
selected-root permutation p-values, or explicit topology-family split matches.
The output is diagnostic-only and fail-closed: a root is usable only when its
validity is replay-supported and its selected-root spectral tail is calibrated.
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

from benchmarks.diagnostics.calibration.root.root_tail_values import finite_float, string_value
from benchmarks.diagnostics.calibration.root.selected.root_selected_spectral_tail_law_panel import (
    DEFAULT_RESULT_ROOT,
)

SCHEMA_VERSION = "root_selected_validity_replay_panel/v1"
STUDY_ROLE = "diagnostic_root_selected_validity_replay_not_calibration"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration.root.selected.root_selected_validity_replay_panel"
)

DEFAULT_TAIL_ROWS = (
    DEFAULT_RESULT_ROOT
    / "root_selected_spectral_tail_law_deformed_hu_mild_replay_v3_smoke"
    / "root_selected_spectral_tail_law_rows.csv"
)

ROWS_OUTPUT = "root_selected_validity_replay_rows.csv"
SUMMARY_OUTPUT = "root_selected_validity_replay_summary.csv"
MANIFEST_OUTPUT = "manifest.json"

TAIL_CALIBRATED_STATUS = "calibrated_selected_root_spectral_tail_available"

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "target_case_id",
    "root_validity_problem",
    "root_tail_problem",
    "selected_root_family_condition",
    "root_tail_inference_status",
    "selected_null_support_count",
    "s_root_spectral_excess_log",
    "t_selected_tie_rank_fraction",
    "a_selected_ratio_action_log1p",
    "e_edge_margin_action_log1p",
    "b_bandwidth_topology_status",
    "h_u_population_law_status",
    "root_replay_source",
    "root_replay_count",
    "root_replay_mean_ari_to_observed",
    "root_replay_median_ari_to_observed",
    "root_replay_q10_ari_to_observed",
    "root_replay_min_ari_to_observed",
    "root_stability_threshold",
    "plausible_alternative_root_count",
    "plausible_alternative_root_fraction",
    "root_stability_guard_blocked",
    "root_selective_permutation_p_value",
    "root_selective_permutation_alpha",
    "root_selective_permutation_guard_blocked",
    "root_selective_permutation_guard_would_block",
    "root_validity_status",
    "selected_root_usability_status",
    "method_action",
    "mathematical_interpretation",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "target_count",
    "root_validity_supported_count",
    "root_validity_failed_count",
    "root_validity_unmeasured_count",
    "tail_calibrated_count",
    "usable_selected_root_count",
    "valid_but_tail_missing_count",
    "tail_calibrated_but_root_invalid_or_unmeasured_count",
    "plausible_alternative_root_count",
    "median_root_replay_mean_ari_to_observed",
    "summary_status",
)


@dataclass(frozen=True)
class RootSelectedValidityReplayConfig:
    """Input/output contract for selected-root validity replay rows."""

    output_dir: Path
    tail_rows_path: Path = DEFAULT_TAIL_ROWS
    root_replay_rows_path: Path | None = None
    root_stability_threshold: float = 0.24
    root_selective_permutation_alpha: float = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tail-rows-path", type=Path, default=DEFAULT_TAIL_ROWS)
    parser.add_argument("--root-replay-rows-path", type=Path, default=None)
    parser.add_argument("--root-stability-threshold", type=float, default=0.24)
    parser.add_argument(
        "--root-selective-permutation-alpha",
        type=float,
        default=0.01,
    )
    return parser.parse_args()


def _json_default(value: object) -> object:
    if isinstance(value, RootSelectedValidityReplayConfig):
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


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if pd.isna(value):
        return False
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _first_finite(row: pd.Series, columns: tuple[str, ...], default: float) -> float:
    for column in columns:
        if column not in row:
            continue
        value = finite_float(row.get(column, math.nan))
        if math.isfinite(value):
            return value
    return float(default)


def _numeric_values(group: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    for column in columns:
        if column not in group.columns:
            continue
        values = pd.to_numeric(group[column], errors="coerce").dropna().to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            return values
    return np.asarray([], dtype=float)


def _any_bool(group: pd.DataFrame, columns: tuple[str, ...]) -> bool:
    for column in columns:
        if column not in group.columns:
            continue
        return bool(any(_bool_value(value) for value in group[column]))
    return False


def _root_replay_case_lookup(root_replay_rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if root_replay_rows.empty:
        return {}
    if "case_id" not in root_replay_rows.columns:
        raise ValueError("root replay rows missing required column: 'case_id'.")
    return {
        str(case_id): group.copy()
        for case_id, group in root_replay_rows.groupby("case_id", sort=True)
    }


def _root_replay_ari_values(group: pd.DataFrame) -> tuple[np.ndarray, str]:
    values = _numeric_values(
        group,
        (
            "root_partition_ari_to_observed",
            "root_replay_ari_to_observed",
            "root_replay_split_ari_to_observed",
        ),
    )
    if values.size:
        return values, "explicit_root_topology_family_replay"

    values = _numeric_values(
        group,
        (
            "root_stability_subsample_mean_ari",
            "mean_root_stability_subsample_mean_ari",
        ),
    )
    if values.size:
        return values, "profile_root_stability_summary"
    return values, "root_replay_missing"


def _root_replay_summary(
    group: pd.DataFrame | None,
    *,
    configured_stability_threshold: float,
    configured_selective_alpha: float,
) -> dict[str, object]:
    if group is None or group.empty:
        return {
            "root_replay_source": "root_replay_missing",
            "root_replay_count": 0,
            "root_replay_mean_ari_to_observed": math.nan,
            "root_replay_median_ari_to_observed": math.nan,
            "root_replay_q10_ari_to_observed": math.nan,
            "root_replay_min_ari_to_observed": math.nan,
            "root_stability_threshold": float(configured_stability_threshold),
            "plausible_alternative_root_count": 0,
            "plausible_alternative_root_fraction": math.nan,
            "root_stability_guard_blocked": False,
            "root_selective_permutation_p_value": math.nan,
            "root_selective_permutation_alpha": float(configured_selective_alpha),
            "root_selective_permutation_guard_blocked": False,
            "root_selective_permutation_guard_would_block": False,
        }

    ari_values, replay_source = _root_replay_ari_values(group)
    threshold_values = _numeric_values(
        group,
        (
            "observed_root_stability_guard_threshold",
            "root_stability_guard_threshold",
        ),
    )
    stability_threshold = (
        float(np.nanmax(threshold_values))
        if threshold_values.size
        else float(configured_stability_threshold)
    )
    alternative_count = int(np.sum(ari_values < stability_threshold))
    replay_count = int(ari_values.size)
    p_values = _numeric_values(
        group,
        (
            "root_selective_permutation_p_value",
            "root_selective_p_value",
        ),
    )
    alpha_values = _numeric_values(
        group,
        (
            "observed_root_selective_permutation_guard_alpha",
            "root_selective_permutation_guard_alpha",
            "root_selective_permutation_alpha",
        ),
    )
    selective_alpha = (
        float(np.nanmin(alpha_values)) if alpha_values.size else float(configured_selective_alpha)
    )
    return {
        "root_replay_source": replay_source,
        "root_replay_count": replay_count,
        "root_replay_mean_ari_to_observed": float(np.mean(ari_values))
        if replay_count
        else math.nan,
        "root_replay_median_ari_to_observed": float(np.median(ari_values))
        if replay_count
        else math.nan,
        "root_replay_q10_ari_to_observed": float(np.quantile(ari_values, 0.10))
        if replay_count
        else math.nan,
        "root_replay_min_ari_to_observed": float(np.min(ari_values)) if replay_count else math.nan,
        "root_stability_threshold": stability_threshold,
        "plausible_alternative_root_count": alternative_count,
        "plausible_alternative_root_fraction": (
            float(alternative_count / replay_count) if replay_count else math.nan
        ),
        "root_stability_guard_blocked": _any_bool(
            group,
            (
                "root_stability_guard_blocked",
                "Root_Stability_Guard_Blocked",
            ),
        ),
        "root_selective_permutation_p_value": float(np.nanmean(p_values))
        if p_values.size
        else math.nan,
        "root_selective_permutation_alpha": selective_alpha,
        "root_selective_permutation_guard_blocked": _any_bool(
            group,
            (
                "root_selective_permutation_guard_blocked",
                "Root_Selective_Permutation_Guard_Blocked",
            ),
        ),
        "root_selective_permutation_guard_would_block": _any_bool(
            group,
            (
                "root_selective_permutation_guard_would_block",
                "Root_Selective_Permutation_Guard_Would_Block",
            ),
        ),
    }


def _root_validity_status(summary: dict[str, object]) -> tuple[str, str, str]:
    replay_count = int(summary["root_replay_count"])
    mean_ari = float(summary["root_replay_mean_ari_to_observed"])
    stability_threshold = float(summary["root_stability_threshold"])
    selective_p = float(summary["root_selective_permutation_p_value"])
    selective_alpha = float(summary["root_selective_permutation_alpha"])
    stability_blocked = bool(summary["root_stability_guard_blocked"])
    selective_blocked = bool(summary["root_selective_permutation_guard_blocked"])
    selective_would_block = bool(summary["root_selective_permutation_guard_would_block"])
    has_stability = bool(replay_count > 0 and math.isfinite(mean_ari))
    has_selective = math.isfinite(selective_p)

    if stability_blocked or (has_stability and mean_ari < stability_threshold):
        return (
            "root_validity_failed_feature_subsample_replay",
            "fail_closed_root_unstable_under_topology_replay",
            "The selected root is not stable across plausible feature/topology replays.",
        )
    if (
        selective_blocked
        or selective_would_block
        or (has_selective and selective_p > selective_alpha)
    ):
        return (
            "root_validity_failed_selected_root_permutation",
            "fail_closed_root_not_selectively_significant",
            "The selected root is not significant after replaying null-compatible root selection.",
        )
    if has_stability and has_selective:
        return (
            "root_validity_supported_by_stability_and_selection",
            "allow_selected_root_tail_test",
            "The selected root persists under perturbation and remains selectively significant.",
        )
    if has_stability:
        return (
            "root_validity_supported_by_stability_only",
            "allow_tail_test_but_measure_selected_root_permutation",
            "The selected root is stable under perturbation, but selected-null root replay is missing.",
        )
    if has_selective:
        return (
            "root_validity_supported_by_selected_permutation_only",
            "allow_tail_test_but_measure_topology_stability",
            "The selected root is significant under selected-root permutation, but topology stability is missing.",
        )
    return (
        "root_validity_unmeasured",
        "fail_closed_measure_root_validity_replay",
        "No replay evidence shows that the selected root is the right object to test.",
    )


def _selected_root_usability(
    *,
    root_validity_status: str,
    tail_status: str,
    validity_action: str,
) -> tuple[str, str, str]:
    root_supported = root_validity_status.startswith("root_validity_supported")
    tail_calibrated = tail_status == TAIL_CALIBRATED_STATUS
    if root_supported and tail_calibrated:
        return (
            "usable_selected_root_tail_after_validity_replay",
            "use_selected_root_tail_with_validity_annotation",
            "Both required layers are present: root validity and selected-root tail support.",
        )
    if root_supported:
        return (
            "fail_closed_valid_root_tail_support_missing",
            "fail_closed_generate_tail_support_for_valid_root",
            "The root looks valid, but the selected spectral-tail support is still missing.",
        )
    if root_validity_status == "root_validity_unmeasured":
        return (
            "fail_closed_root_validity_unmeasured",
            validity_action,
            "The selected-root spectral question is not usable until root validity is measured.",
        )
    return (
        "fail_closed_root_validity_failed",
        validity_action,
        "The selected-root tail question is calibrated only inside a root event that failed validity replay.",
    )


def build_root_selected_validity_replay_rows(
    *,
    tail_rows: pd.DataFrame,
    root_replay_rows: pd.DataFrame | None = None,
    root_stability_threshold: float = 0.24,
    root_selective_permutation_alpha: float = 0.01,
) -> pd.DataFrame:
    """Join selected-root tail rows to replay evidence for root validity."""
    _require_columns(
        tail_rows,
        {
            "target_case_id",
            "root_tail_inference_status",
            "s_root_spectral_excess_log",
            "t_selected_tie_rank_fraction",
            "a_selected_ratio_action_log1p",
            "e_edge_margin_action_log1p",
            "b_bandwidth_topology_status",
            "h_u_population_law_status",
        },
        "tail rows",
    )
    replay_rows = pd.DataFrame() if root_replay_rows is None else root_replay_rows.copy()
    replay_lookup = _root_replay_case_lookup(replay_rows)
    records: list[dict[str, object]] = []
    for _, tail in tail_rows.sort_values("target_case_id").iterrows():
        case_id = string_value(tail, "target_case_id")
        replay_summary = _root_replay_summary(
            replay_lookup.get(case_id),
            configured_stability_threshold=float(root_stability_threshold),
            configured_selective_alpha=float(root_selective_permutation_alpha),
        )
        validity_status, validity_action, validity_interpretation = _root_validity_status(
            replay_summary
        )
        tail_status = string_value(tail, "root_tail_inference_status")
        usability_status, method_action, usability_interpretation = _selected_root_usability(
            root_validity_status=validity_status,
            tail_status=tail_status,
            validity_action=validity_action,
        )
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "target_case_id": case_id,
                "root_validity_problem": ("was_the_selected_root_bifurcation_stable_and_coherent"),
                "root_tail_problem": ("given_a_valid_selected_root_is_the_spectral_tail_supported"),
                "selected_root_family_condition": (
                    "compare_observed_root_to_topology_replay_perturbation_and_selected_null_roots"
                ),
                "root_tail_inference_status": tail_status,
                "selected_null_support_count": int(
                    _first_finite(tail, ("selected_null_support_count",), 0.0)
                ),
                "s_root_spectral_excess_log": finite_float(
                    tail.get("s_root_spectral_excess_log", math.nan)
                ),
                "t_selected_tie_rank_fraction": finite_float(
                    tail.get("t_selected_tie_rank_fraction", math.nan)
                ),
                "a_selected_ratio_action_log1p": finite_float(
                    tail.get("a_selected_ratio_action_log1p", math.nan)
                ),
                "e_edge_margin_action_log1p": finite_float(
                    tail.get("e_edge_margin_action_log1p", math.nan)
                ),
                "b_bandwidth_topology_status": string_value(
                    tail,
                    "b_bandwidth_topology_status",
                ),
                "h_u_population_law_status": string_value(
                    tail,
                    "h_u_population_law_status",
                ),
                **replay_summary,
                "root_validity_status": validity_status,
                "selected_root_usability_status": usability_status,
                "method_action": method_action,
                "mathematical_interpretation": (
                    f"{validity_interpretation} {usability_interpretation}"
                ),
            }
        )
    return pd.DataFrame.from_records(records, columns=ROW_COLUMNS)


def summarize_root_selected_validity_replay_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize selected-root validity replay outcomes."""
    if rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    validity = rows["root_validity_status"].astype(str)
    usability = rows["selected_root_usability_status"].astype(str)
    tail = rows["root_tail_inference_status"].astype(str)
    replay_mean = pd.to_numeric(
        rows["root_replay_mean_ari_to_observed"],
        errors="coerce",
    )
    return pd.DataFrame.from_records(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "target_count": int(rows.shape[0]),
                "root_validity_supported_count": int(
                    validity.str.startswith("root_validity_supported").sum()
                ),
                "root_validity_failed_count": int(
                    validity.str.startswith("root_validity_failed").sum()
                ),
                "root_validity_unmeasured_count": int(
                    validity.eq("root_validity_unmeasured").sum()
                ),
                "tail_calibrated_count": int(tail.eq(TAIL_CALIBRATED_STATUS).sum()),
                "usable_selected_root_count": int(
                    usability.eq("usable_selected_root_tail_after_validity_replay").sum()
                ),
                "valid_but_tail_missing_count": int(
                    usability.eq("fail_closed_valid_root_tail_support_missing").sum()
                ),
                "tail_calibrated_but_root_invalid_or_unmeasured_count": int(
                    (
                        tail.eq(TAIL_CALIBRATED_STATUS)
                        & ~usability.eq("usable_selected_root_tail_after_validity_replay")
                    ).sum()
                ),
                "plausible_alternative_root_count": int(
                    pd.to_numeric(
                        rows["plausible_alternative_root_count"],
                        errors="coerce",
                    )
                    .fillna(0)
                    .sum()
                ),
                "median_root_replay_mean_ari_to_observed": float(replay_mean.median())
                if replay_mean.notna().any()
                else math.nan,
                "summary_status": (
                    "all_selected_roots_valid_and_tail_calibrated"
                    if usability.eq("usable_selected_root_tail_after_validity_replay").all()
                    else "selected_root_validity_or_tail_law_incomplete"
                ),
            }
        ],
        columns=SUMMARY_COLUMNS,
    )


def evaluate_root_selected_validity_replay_panel(
    config: RootSelectedValidityReplayConfig,
) -> dict[str, pd.DataFrame]:
    tail_rows = pd.read_csv(config.tail_rows_path, low_memory=False)
    root_replay_rows = (
        pd.DataFrame()
        if config.root_replay_rows_path is None
        else pd.read_csv(config.root_replay_rows_path, low_memory=False)
    )
    rows = build_root_selected_validity_replay_rows(
        tail_rows=tail_rows,
        root_replay_rows=root_replay_rows,
        root_stability_threshold=float(config.root_stability_threshold),
        root_selective_permutation_alpha=float(config.root_selective_permutation_alpha),
    )
    summary = summarize_root_selected_validity_replay_rows(rows)
    return {"rows": rows, "summary": summary}


def run_root_selected_validity_replay_panel(
    config: RootSelectedValidityReplayConfig,
) -> dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    tables = evaluate_root_selected_validity_replay_panel(config)
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
    outputs = run_root_selected_validity_replay_panel(
        RootSelectedValidityReplayConfig(
            output_dir=args.output_dir,
            tail_rows_path=args.tail_rows_path,
            root_replay_rows_path=args.root_replay_rows_path,
            root_stability_threshold=float(args.root_stability_threshold),
            root_selective_permutation_alpha=float(args.root_selective_permutation_alpha),
        )
    )
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
