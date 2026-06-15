"""Traversal-geometry guard validation panel.

This module evaluates diagnostic high-action angular-shell guard thresholds on
precomputed traversal geometry rows. It is diagnostic-only and does not install
a production traversal guard.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "diagnostic_traversal_guard_validation_panel_not_calibration"
SCHEMA_VERSION = "traversal_guard_validation_panel/v1"
TRAVERSAL_CONTEXT_ROLES = (
    "pure_fragment",
    "mixed_signal",
    "true_signal",
)
PURE_FRAGMENT_ROLES = frozenset({"pure_fragment"})
SIGNAL_ROLES = frozenset({"mixed_signal", "true_signal"})
REQUIRED_COLUMNS = {
    "traversal_context_role",
    "action_budget_proxy_capped",
    "geometry_angle_to_leading_axis_deg",
    "geometry_independent_radius_fraction",
}


def _finite_numeric(table: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(table[column], errors="raise").astype(float)
    invalid = ~np.isfinite(values)
    if bool(invalid.any()):
        bad_index = invalid[invalid].index[0]
        raise ValueError(
            f"{column} must contain finite values; "
            f"row={int(bad_index)}, value={float(values.loc[bad_index])!r}."
        )
    return values


def _context_id(table: pd.DataFrame) -> pd.Series:
    if "traversal_context_id" in table.columns:
        return table["traversal_context_id"].astype(str)
    if "context_id" in table.columns:
        return table["context_id"].astype(str)
    return pd.Series(
        [f"traversal_context_{index}" for index in table.index],
        index=table.index,
    )


def evaluate_traversal_guard_rows(
    panel: pd.DataFrame,
    *,
    action_thresholds: tuple[float, ...] = (0.5, 0.75, 0.9),
    angle_thresholds: tuple[float, ...] = (45.0, 60.0, 75.0),
    independent_thresholds: tuple[float, ...] = (0.85,),
    mixed_context_cost_ratio: float = 2.0,
) -> pd.DataFrame:
    """Return one guard-decision row per input row and threshold tuple."""
    if float(mixed_context_cost_ratio) < 0.0:
        raise ValueError(
            "mixed_context_cost_ratio must be non-negative; "
            f"got {mixed_context_cost_ratio!r}."
        )
    missing = REQUIRED_COLUMNS - set(panel.columns)
    if missing:
        raise ValueError(
            f"Traversal guard panel is missing columns: {sorted(missing)!r}."
        )
    table = panel.copy()
    role = table["traversal_context_role"].astype(str)
    unknown_roles = sorted(set(role) - set(TRAVERSAL_CONTEXT_ROLES))
    if unknown_roles:
        raise ValueError(
            "traversal_context_role contains unknown roles: "
            f"{unknown_roles!r}; expected {list(TRAVERSAL_CONTEXT_ROLES)!r}."
        )
    action = _finite_numeric(table, "action_budget_proxy_capped")
    angle = _finite_numeric(table, "geometry_angle_to_leading_axis_deg")
    independent = _finite_numeric(table, "geometry_independent_radius_fraction")

    base = pd.DataFrame(
        {
            "traversal_context_id": _context_id(table),
            "traversal_context_role": role,
            "is_pure_fragment_context": role.isin(PURE_FRAGMENT_ROLES),
            "is_signal_context": role.isin(SIGNAL_ROLES),
            "action_budget_proxy_capped": action,
            "geometry_angle_to_leading_axis_deg": angle,
            "geometry_independent_radius_fraction": independent,
        }
    )
    passthrough_columns = [
        column
        for column in (
            "case_id",
            "replicate_id",
            "parent_id",
            "parent_sample_size",
            "barycentric_balance",
            "feature_family",
        )
        if column in table.columns
    ]
    for column in passthrough_columns:
        base[column] = table[column].to_numpy()

    rows: list[pd.DataFrame] = []
    for action_threshold in action_thresholds:
        for angle_threshold in angle_thresholds:
            for independent_threshold in independent_thresholds:
                flagged = (
                    action.ge(float(action_threshold))
                    & angle.ge(float(angle_threshold))
                    & independent.ge(float(independent_threshold))
                )
                guard_id = (
                    f"action_ge_{action_threshold:g}__angle_ge_"
                    f"{angle_threshold:g}__ind_ge_{independent_threshold:g}"
                )
                chunk = base.copy()
                chunk["guard_id"] = guard_id
                chunk["action_threshold"] = float(action_threshold)
                chunk["angle_threshold_deg"] = float(angle_threshold)
                chunk["independent_fraction_threshold"] = float(independent_threshold)
                chunk["guard_flagged"] = flagged
                chunk["mixed_context_cost_ratio"] = float(mixed_context_cost_ratio)
                chunk["row_utility"] = np.where(
                    flagged & base["is_pure_fragment_context"],
                    1.0,
                    np.where(
                        flagged & base["is_signal_context"],
                        -float(mixed_context_cost_ratio),
                        0.0,
                    ),
                )
                chunk["study_role"] = STUDY_ROLE
                rows.append(chunk)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _guard_status(
    *,
    n_flagged: int,
    pure_precision: float,
    signal_flag_rate: float,
    net_utility: float,
    min_flagged: int,
    min_pure_precision: float,
    max_signal_flag_rate: float,
) -> str:
    if n_flagged < min_flagged:
        return "insufficient_flagged_rows"
    if (
        math.isfinite(pure_precision)
        and pure_precision >= min_pure_precision
        and math.isfinite(signal_flag_rate)
        and signal_flag_rate <= max_signal_flag_rate
        and net_utility > 0.0
    ):
        return "guard_validation_candidate"
    return "diagnostic_only_guard"


def summarize_traversal_guard_rows(
    rows: pd.DataFrame,
    *,
    min_flagged: int = 10,
    min_pure_precision: float = 0.7,
    max_signal_flag_rate: float = 0.1,
) -> pd.DataFrame:
    """Summarize guard validation rows by threshold tuple."""
    if rows.empty:
        return pd.DataFrame()
    summaries: list[dict[str, object]] = []
    for guard_id, group in rows.groupby("guard_id", sort=True):
        flagged = group["guard_flagged"].astype(bool)
        pure = group["is_pure_fragment_context"].astype(bool)
        signal = group["is_signal_context"].astype(bool)
        n_flagged = int(flagged.sum())
        n_pure_flagged = int((flagged & pure).sum())
        n_signal_flagged = int((flagged & signal).sum())
        pure_precision = float(n_pure_flagged / n_flagged) if n_flagged else math.nan
        signal_flag_rate = (
            float(n_signal_flagged / signal.sum()) if bool(signal.any()) else math.nan
        )
        net_utility = float(group["row_utility"].sum())
        first = group.iloc[0]
        summaries.append(
            {
                "guard_id": guard_id,
                "action_threshold": float(first["action_threshold"]),
                "angle_threshold_deg": float(first["angle_threshold_deg"]),
                "independent_fraction_threshold": float(
                    first["independent_fraction_threshold"]
                ),
                "n_rows": int(group.shape[0]),
                "n_flagged": n_flagged,
                "n_pure_fragment_contexts": int(pure.sum()),
                "n_signal_contexts": int(signal.sum()),
                "n_pure_flagged": n_pure_flagged,
                "n_signal_flagged": n_signal_flagged,
                "pure_fragment_precision": pure_precision,
                "pure_fragment_flag_rate": (
                    float(n_pure_flagged / pure.sum()) if bool(pure.any()) else math.nan
                ),
                "signal_context_flag_rate": signal_flag_rate,
                "net_utility": net_utility,
                "mixed_context_cost_ratio": float(first["mixed_context_cost_ratio"]),
                "guard_validation_status": _guard_status(
                    n_flagged=n_flagged,
                    pure_precision=pure_precision,
                    signal_flag_rate=signal_flag_rate,
                    net_utility=net_utility,
                    min_flagged=int(min_flagged),
                    min_pure_precision=float(min_pure_precision),
                    max_signal_flag_rate=float(max_signal_flag_rate),
                ),
                "min_flagged": int(min_flagged),
                "min_pure_precision": float(min_pure_precision),
                "max_signal_flag_rate": float(max_signal_flag_rate),
                "study_role": STUDY_ROLE,
            }
        )
    return pd.DataFrame.from_records(summaries)


def run_traversal_guard_validation_panel(
    *,
    panel_path: Path,
    output_dir: Path,
    mixed_context_cost_ratio: float = 2.0,
    min_flagged: int = 10,
    min_pure_precision: float = 0.7,
    max_signal_flag_rate: float = 0.1,
) -> dict[str, Path]:
    """Run the traversal guard validation diagnostic from a CSV panel."""
    panel = pd.read_csv(panel_path)
    rows = evaluate_traversal_guard_rows(
        panel,
        mixed_context_cost_ratio=mixed_context_cost_ratio,
    )
    summary = summarize_traversal_guard_rows(
        rows,
        min_flagged=min_flagged,
        min_pure_precision=min_pure_precision,
        max_signal_flag_rate=max_signal_flag_rate,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "traversal_guard_validation_rows.csv"
    summary_path = output_dir / "traversal_guard_validation_summary.csv"
    manifest_path = output_dir / "manifest.json"
    rows.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)
    manifest = {
        "created_at_utc": format_timestamp_utc(),
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "panel_path": str(panel_path),
        "mixed_context_cost_ratio": float(mixed_context_cost_ratio),
        "outputs": {
            "rows": str(rows_path),
            "summary": str(summary_path),
        },
        "interpretation": (
            "Diagnostic traversal-geometry guard validation. Candidate guard "
            "rows are not production traversal rules until independently "
            "validated on predeclared panels."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "rows": rows_path,
        "summary": summary_path,
        "manifest": manifest_path,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mixed-context-cost-ratio", type=float, default=2.0)
    parser.add_argument("--min-flagged", type=int, default=10)
    parser.add_argument("--min-pure-precision", type=float, default=0.7)
    parser.add_argument("--max-signal-flag-rate", type=float, default=0.1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    outputs = run_traversal_guard_validation_panel(
        panel_path=args.panel,
        output_dir=args.output_dir,
        mixed_context_cost_ratio=args.mixed_context_cost_ratio,
        min_flagged=args.min_flagged,
        min_pure_precision=args.min_pure_precision,
        max_signal_flag_rate=args.max_signal_flag_rate,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "STUDY_ROLE",
    "TRAVERSAL_CONTEXT_ROLES",
    "evaluate_traversal_guard_rows",
    "run_traversal_guard_validation_panel",
    "summarize_traversal_guard_rows",
]
