"""Compare current and full legacy KL behavior over an edge-alpha grid.

This panel is a diagnostic comparator, not a calibration claim. It keeps the
edge-alpha dimension explicit so legacy signal gains cannot be summarized
without the paired selected-null leakage at the same alpha.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.diagnostics.calibration import (
    legacy_c2ef9a69_method_comparison_panel as legacy_panel,
)
from benchmarks.validation.selected_edge_type1_geometry import (
    DEFAULT_EDGE_ALPHA_GRID,
    parse_alpha_grid,
    parse_names,
)

SCHEMA_VERSION = "legacy_c2ef9a69_edge_alpha_comparison_panel/v1"
STUDY_ROLE = "diagnostic_legacy_c2ef9a69_edge_alpha_comparison_not_calibration"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration."
    "legacy_c2ef9a69_edge_alpha_comparison_panel"
)

DEFAULT_ROOT_TAIL_OVERLAP_CASE_NAMES = (
    "overlap_extreme_4c",
    "overlap_heavy_4c_small_feat",
    "overlap_mod_4c_small",
    "overlap_mod_6c_med",
    "overlap_part_4c_small",
    "overlap_unbal_4c_small",
    "overlap_unbal_6c_med",
)

ROWS_OUTPUT = "legacy_c2ef9a69_edge_alpha_comparison_rows.csv"
PAIRWISE_OUTPUT = "legacy_c2ef9a69_edge_alpha_comparison_pairwise.csv"
ALPHA_SUMMARY_OUTPUT = "legacy_c2ef9a69_edge_alpha_comparison_alpha_summary.csv"
TRADEOFF_SUMMARY_OUTPUT = (
    "legacy_c2ef9a69_edge_alpha_comparison_tradeoff_summary.csv"
)
MANIFEST_OUTPUT = "manifest.json"


@dataclass(frozen=True)
class LegacyC2ef9a69EdgeAlphaComparisonConfig:
    """Configuration for current-vs-full-legacy edge-alpha comparison."""

    output_dir: Path
    suite: str = "full"
    case_names: tuple[str, ...] = DEFAULT_ROOT_TAIL_OVERLAP_CASE_NAMES
    data_roles: tuple[str, ...] = ("null", "signal")
    replicates: int = 1
    base_seed: int = 20260613
    edge_alphas: tuple[float, ...] = DEFAULT_EDGE_ALPHA_GRID
    sibling_alpha: float = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", default="full")
    parser.add_argument(
        "--case-names",
        type=parse_names,
        default=DEFAULT_ROOT_TAIL_OVERLAP_CASE_NAMES,
    )
    parser.add_argument("--data-roles", type=parse_names, default=("null", "signal"))
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=20260613)
    parser.add_argument(
        "--edge-alphas",
        type=parse_alpha_grid,
        default=DEFAULT_EDGE_ALPHA_GRID,
    )
    parser.add_argument("--sibling-alpha", type=float, default=0.01)
    return parser.parse_args()


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _validate_edge_alphas(edge_alphas: tuple[float, ...]) -> tuple[float, ...]:
    values = tuple(float(alpha) for alpha in edge_alphas)
    if not values:
        raise ValueError("edge_alphas must contain at least one value.")
    invalid = [alpha for alpha in values if not 0.0 < alpha < 1.0]
    if invalid:
        raise ValueError(f"edge_alphas must lie in (0, 1): {invalid!r}.")
    return values


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].fillna(False).astype(bool)


def _status_for_pairwise_row(row: pd.Series) -> str:
    data_role = str(row["data_role"])
    delta_ari = float(row["delta_ari_legacy_minus_current"])
    current_false_split = bool(row["current_false_split"])
    legacy_false_split = bool(row["legacy_false_split"])
    if data_role == "selected_null":
        if legacy_false_split and not current_false_split:
            return "legacy_extra_selected_null_false_split"
        if current_false_split and not legacy_false_split:
            return "current_extra_selected_null_false_split"
        if legacy_false_split and current_false_split:
            return "both_methods_selected_null_false_split"
        return "no_selected_null_false_split"
    if delta_ari > 1e-12:
        return "legacy_signal_gain_observed"
    if delta_ari < -1e-12:
        return "legacy_signal_loss_observed"
    return "no_signal_ari_change"


def annotate_pairwise_rows(pairwise: pd.DataFrame) -> pd.DataFrame:
    """Add no-shortcut classification columns to pairwise alpha rows."""
    if pairwise.empty:
        return pairwise.copy()
    annotated = pairwise.copy()
    delta_ari = pd.to_numeric(
        annotated["delta_ari_legacy_minus_current"],
        errors="coerce",
    )
    data_role = annotated["data_role"].astype(str)
    annotated["legacy_extra_selected_null_false_split"] = (
        data_role.eq("selected_null")
        & _bool_series(annotated, "legacy_false_split")
        & ~_bool_series(annotated, "current_false_split")
    )
    annotated["legacy_signal_gain"] = data_role.eq("signal") & (delta_ari > 1e-12)
    annotated["legacy_signal_loss"] = data_role.eq("signal") & (delta_ari < -1e-12)
    annotated["comparison_interpretation"] = annotated.apply(
        _status_for_pairwise_row,
        axis=1,
    )
    return annotated


def run_edge_alpha_comparison_rows(
    config: LegacyC2ef9a69EdgeAlphaComparisonConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run current and legacy comparisons for each edge alpha."""
    edge_alphas = _validate_edge_alphas(config.edge_alphas)
    all_rows: list[pd.DataFrame] = []
    all_pairwise: list[pd.DataFrame] = []
    for edge_alpha in edge_alphas:
        alpha_config = legacy_panel.LegacyC2ef9a69MethodComparisonConfig(
            output_dir=config.output_dir,
            suite=config.suite,
            case_names=config.case_names,
            data_roles=config.data_roles,
            replicates=int(config.replicates),
            base_seed=int(config.base_seed),
            edge_alpha=float(edge_alpha),
            sibling_alpha=float(config.sibling_alpha),
        )
        rows, pairwise = legacy_panel.run_comparison_rows(alpha_config)
        rows = rows.copy()
        pairwise = pairwise.copy()
        rows["schema_version"] = SCHEMA_VERSION
        rows["study_role"] = STUDY_ROLE
        rows["grid_edge_alpha"] = float(edge_alpha)
        pairwise["schema_version"] = SCHEMA_VERSION
        pairwise["study_role"] = STUDY_ROLE
        pairwise["edge_alpha"] = float(edge_alpha)
        all_rows.append(rows)
        all_pairwise.append(pairwise)
    rows_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    pairwise_df = (
        pd.concat(all_pairwise, ignore_index=True) if all_pairwise else pd.DataFrame()
    )
    return rows_df, annotate_pairwise_rows(pairwise_df)


def summarize_by_alpha(pairwise: pd.DataFrame) -> pd.DataFrame:
    """Summarize current-vs-legacy behavior by alpha and data role."""
    if pairwise.empty:
        return pd.DataFrame(
            [
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_role": STUDY_ROLE,
                    "edge_alpha": np.nan,
                    "data_role": "all",
                    "row_count": 0,
                    "alpha_role_status": "no_pairwise_rows",
                }
            ]
        )
    summaries: list[dict[str, object]] = []
    for (edge_alpha, data_role), group in pairwise.groupby(
        ["edge_alpha", "data_role"],
        dropna=False,
    ):
        delta_ari = pd.to_numeric(
            group["delta_ari_legacy_minus_current"],
            errors="coerce",
        )
        delta_clusters = pd.to_numeric(
            group["delta_clusters_legacy_minus_current"],
            errors="coerce",
        )
        legacy_extra_null = int(
            group.get("legacy_extra_selected_null_false_split", False).sum()
        )
        legacy_signal_gain = int(group.get("legacy_signal_gain", False).sum())
        if str(data_role) == "selected_null" and legacy_extra_null:
            alpha_role_status = "legacy_alpha_leaks_selected_null"
        elif str(data_role) == "signal" and legacy_signal_gain:
            alpha_role_status = "legacy_alpha_has_signal_gain"
        else:
            alpha_role_status = "no_legacy_gain_or_extra_null_leak"
        summaries.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "edge_alpha": float(edge_alpha),
                "data_role": str(data_role),
                "row_count": int(len(group)),
                "completed_pair_count": int(
                    (
                        group["current_status"].astype(str).eq("ok")
                        & group["legacy_status"].astype(str).eq("ok")
                    ).sum()
                ),
                "current_false_split_count": int(
                    _bool_series(group, "current_false_split").sum()
                ),
                "legacy_false_split_count": int(
                    _bool_series(group, "legacy_false_split").sum()
                ),
                "legacy_extra_selected_null_false_split_count": legacy_extra_null,
                "current_under_split_count": int(
                    _bool_series(group, "current_under_split").sum()
                ),
                "legacy_under_split_count": int(
                    _bool_series(group, "legacy_under_split").sum()
                ),
                "legacy_signal_gain_count": legacy_signal_gain,
                "legacy_signal_loss_count": int(
                    group.get("legacy_signal_loss", False).sum()
                ),
                "mean_delta_ari_legacy_minus_current": float(delta_ari.mean()),
                "min_delta_ari_legacy_minus_current": float(delta_ari.min()),
                "max_delta_ari_legacy_minus_current": float(delta_ari.max()),
                "mean_delta_clusters_legacy_minus_current": float(
                    delta_clusters.mean()
                ),
                "mean_partition_ari_between_variants": float(
                    pd.to_numeric(
                        group["partition_ari_between_variants"],
                        errors="coerce",
                    ).mean()
                ),
                "alpha_role_status": alpha_role_status,
            }
        )
    return pd.DataFrame.from_records(summaries)


def summarize_alpha_tradeoff(alpha_summary: pd.DataFrame) -> pd.DataFrame:
    """Join selected-null leakage and signal gains into one alpha decision table."""
    if alpha_summary.empty or "edge_alpha" not in alpha_summary:
        return pd.DataFrame(
            [
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_role": STUDY_ROLE,
                    "edge_alpha": np.nan,
                    "tradeoff_status": "no_alpha_summary_rows",
                }
            ]
        )
    rows: list[dict[str, object]] = []
    for edge_alpha, group in alpha_summary.groupby("edge_alpha", dropna=False):
        by_role = group.set_index("data_role")
        selected_null = (
            by_role.loc["selected_null"] if "selected_null" in by_role.index else None
        )
        signal = by_role.loc["signal"] if "signal" in by_role.index else None
        legacy_extra_null = (
            int(selected_null["legacy_extra_selected_null_false_split_count"])
            if selected_null is not None
            else 0
        )
        current_null_false_split = (
            int(selected_null["current_false_split_count"])
            if selected_null is not None
            else 0
        )
        legacy_signal_gain = (
            int(signal["legacy_signal_gain_count"]) if signal is not None else 0
        )
        legacy_signal_loss = (
            int(signal["legacy_signal_loss_count"]) if signal is not None else 0
        )
        signal_mean_delta = (
            float(signal["mean_delta_ari_legacy_minus_current"])
            if signal is not None
            else np.nan
        )
        signal_max_delta = (
            float(signal["max_delta_ari_legacy_minus_current"])
            if signal is not None
            else np.nan
        )
        if legacy_extra_null and legacy_signal_gain:
            tradeoff_status = "legacy_power_not_admissible_extra_null_leak"
        elif legacy_extra_null:
            tradeoff_status = "legacy_rejected_extra_null_leak"
        elif legacy_signal_gain:
            tradeoff_status = "legacy_candidate_gain_without_extra_null_leak"
        else:
            tradeoff_status = "no_legacy_edge_alpha_power_gain"
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "edge_alpha": float(edge_alpha),
                "selected_null_legacy_extra_false_split_count": legacy_extra_null,
                "selected_null_current_false_split_count": current_null_false_split,
                "signal_legacy_gain_count": legacy_signal_gain,
                "signal_legacy_loss_count": legacy_signal_loss,
                "signal_mean_delta_ari_legacy_minus_current": signal_mean_delta,
                "signal_max_delta_ari_legacy_minus_current": signal_max_delta,
                "tradeoff_status": tradeoff_status,
            }
        )
    return pd.DataFrame.from_records(rows)


def run_legacy_c2ef9a69_edge_alpha_comparison_panel(
    config: LegacyC2ef9a69EdgeAlphaComparisonConfig,
) -> dict[str, Path]:
    """Run the edge-alpha comparison and write CSV outputs plus manifest."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows, pairwise = run_edge_alpha_comparison_rows(config)
    alpha_summary = summarize_by_alpha(pairwise)
    tradeoff_summary = summarize_alpha_tradeoff(alpha_summary)

    rows_path = config.output_dir / ROWS_OUTPUT
    pairwise_path = config.output_dir / PAIRWISE_OUTPUT
    alpha_summary_path = config.output_dir / ALPHA_SUMMARY_OUTPUT
    tradeoff_summary_path = config.output_dir / TRADEOFF_SUMMARY_OUTPUT
    manifest_path = config.output_dir / MANIFEST_OUTPUT

    rows.to_csv(rows_path, index=False)
    pairwise.to_csv(pairwise_path, index=False)
    alpha_summary.to_csv(alpha_summary_path, index=False)
    tradeoff_summary.to_csv(tradeoff_summary_path, index=False)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at": datetime.now(UTC).isoformat(),
        "parameters": {
            "suite": config.suite,
            "case_names": list(config.case_names),
            "data_roles": list(config.data_roles),
            "replicates": int(config.replicates),
            "base_seed": int(config.base_seed),
            "edge_alphas": list(_validate_edge_alphas(config.edge_alphas)),
            "sibling_alpha": float(config.sibling_alpha),
        },
        "outputs": {
            "rows": rows_path,
            "pairwise": pairwise_path,
            "alpha_summary": alpha_summary_path,
            "tradeoff_summary": tradeoff_summary_path,
            "manifest": manifest_path,
        },
        "n_rows": int(len(rows)),
        "n_pairwise_rows": int(len(pairwise)),
        "n_alpha_summary_rows": int(len(alpha_summary)),
        "n_tradeoff_summary_rows": int(len(tradeoff_summary)),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    return {
        "rows": rows_path,
        "pairwise": pairwise_path,
        "alpha_summary": alpha_summary_path,
        "tradeoff_summary": tradeoff_summary_path,
        "manifest": manifest_path,
    }


def main() -> None:
    args = parse_args()
    run_legacy_c2ef9a69_edge_alpha_comparison_panel(
        LegacyC2ef9a69EdgeAlphaComparisonConfig(
            output_dir=args.output_dir,
            suite=args.suite,
            case_names=tuple(args.case_names),
            data_roles=tuple(args.data_roles),
            replicates=int(args.replicates),
            base_seed=int(args.base_seed),
            edge_alphas=tuple(args.edge_alphas),
            sibling_alpha=float(args.sibling_alpha),
        )
    )


if __name__ == "__main__":
    main()
