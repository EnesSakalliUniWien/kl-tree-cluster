"""Sibling null-weight rule validation diagnostic.

This module scores predeclared sibling null-weight rules from child-parent
edge p-values. It can be run on selected-geometry records now and reused on
future mixed null/signal simulations. The output is diagnostic-only; it does
not change the production weight rule.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "diagnostic_sibling_null_weight_rule_not_calibration"
EPS = 1e-300

REQUIRED_COLUMNS = {
    "left_edge_bh_p_value",
    "right_edge_bh_p_value",
    "selected_hierarchy_ratio",
}

WEIGHT_RULE_IDS = (
    "current_product_bh_p",
    "min_edge_bh_p",
    "geometric_mean_bh_p",
    "fisher_combined_null_p",
    "hard_null_indicator_0_05",
)


def _probability_series(values: pd.Series, *, column_name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise").astype(float)
    if bool(((numeric < 0.0) | (numeric > 1.0) | ~np.isfinite(numeric)).any()):
        bad_index = numeric[((numeric < 0.0) | (numeric > 1.0) | ~np.isfinite(numeric))].index[
            0
        ]
        raise ValueError(
            f"{column_name} must contain finite probabilities in [0, 1]; "
            f"row={int(bad_index)}, value={float(numeric.loc[bad_index])!r}."
        )
    return numeric


def _effective_sample_size(weights: np.ndarray) -> float:
    positive = weights[weights > 0.0]
    if positive.size == 0:
        return 0.0
    return float(np.sum(positive) ** 2 / np.sum(positive**2))


def _max_weight_share(weights: np.ndarray) -> float:
    positive = weights[weights > 0.0]
    if positive.size == 0:
        return 0.0
    return float(np.max(positive) / np.sum(positive))


def add_sibling_null_weight_rule_columns(records: pd.DataFrame) -> pd.DataFrame:
    """Add predeclared sibling null-weight rule columns to a record table."""
    missing = REQUIRED_COLUMNS - set(records.columns)
    if missing:
        raise ValueError(f"Weight-rule records are missing columns: {sorted(missing)!r}.")
    table = records.copy()
    left = _probability_series(table["left_edge_bh_p_value"], column_name="left_edge_bh_p_value")
    right = _probability_series(
        table["right_edge_bh_p_value"],
        column_name="right_edge_bh_p_value",
    )
    table["current_product_bh_p"] = np.clip(left, EPS, 1.0) * np.clip(right, EPS, 1.0)
    table["min_edge_bh_p"] = np.minimum(left, right)
    table["geometric_mean_bh_p"] = np.sqrt(np.clip(left, EPS, 1.0) * np.clip(right, EPS, 1.0))
    fisher_statistic = -2.0 * (np.log(np.clip(left, EPS, 1.0)) + np.log(np.clip(right, EPS, 1.0)))
    table["fisher_combined_null_p"] = chi2.sf(fisher_statistic, df=4)
    table["hard_null_indicator_0_05"] = ((left > 0.05) & (right > 0.05)).astype(float)
    return table


def _support_status(table: pd.DataFrame) -> str:
    if {"is_null_like", "is_edge_blocked"} <= set(table.columns):
        support_mask = table["is_null_like"].astype(bool) | table["is_edge_blocked"].astype(bool)
        if bool(support_mask.any()):
            return "has_internal_support_labels"
        return "selected_nonnull_only_by_labels"
    return "support_labels_unavailable"


def _labeled_support_masks(table: pd.DataFrame) -> tuple[pd.Series, pd.Series] | None:
    if not {"is_null_like", "is_edge_blocked"} <= set(table.columns):
        return None
    support_mask = table["is_null_like"].astype(bool) | table["is_edge_blocked"].astype(bool)
    selected_nonnull_mask = ~support_mask
    return support_mask, selected_nonnull_mask


def _weighted_mean_or_nan(values: np.ndarray, weights: np.ndarray) -> float:
    if float(np.sum(weights)) <= 0.0:
        return np.nan
    return float(np.sum(weights * values) / np.sum(weights))


def evaluate_sibling_null_weight_rules(records: pd.DataFrame) -> pd.DataFrame:
    """Summarize weight-rule behavior on a sibling-record table."""
    table = add_sibling_null_weight_rule_columns(records)
    ratios = pd.to_numeric(table["selected_hierarchy_ratio"], errors="raise").astype(float)
    if bool(((ratios < 0.0) | ~np.isfinite(ratios)).any()):
        bad_index = ratios[((ratios < 0.0) | ~np.isfinite(ratios))].index[0]
        raise ValueError(
            "selected_hierarchy_ratio must be finite and non-negative; "
            f"row={int(bad_index)}, value={float(ratios.loc[bad_index])!r}."
        )

    rows: list[dict[str, object]] = []
    support_status = _support_status(table)
    label_masks = _labeled_support_masks(table)
    ratio_values = ratios.to_numpy(dtype=float)
    for rule_id in WEIGHT_RULE_IDS:
        weights = table[rule_id].to_numpy(dtype=float)
        positive = weights > 0.0
        weighted_ratio = _weighted_mean_or_nan(ratio_values, weights)
        labeled_metrics: dict[str, object]
        if label_masks is None:
            labeled_metrics = {
                "n_supported_records_by_label": 0,
                "n_selected_nonnull_records_by_label": 0,
                "n_supported_positive_weight_records": 0,
                "n_selected_nonnull_positive_weight_records": 0,
                "supported_weight_share": np.nan,
                "selected_nonnull_weight_share": np.nan,
                "supported_weighted_selected_ratio_mean": np.nan,
                "selected_nonnull_weighted_selected_ratio_mean": np.nan,
            }
        else:
            support_mask, selected_nonnull_mask = label_masks
            support = support_mask.to_numpy(dtype=bool)
            selected_nonnull = selected_nonnull_mask.to_numpy(dtype=bool)
            weight_sum = float(np.sum(weights))
            support_weight_sum = float(np.sum(weights[support]))
            selected_nonnull_weight_sum = float(np.sum(weights[selected_nonnull]))
            labeled_metrics = {
                "n_supported_records_by_label": int(np.sum(support)),
                "n_selected_nonnull_records_by_label": int(np.sum(selected_nonnull)),
                "n_supported_positive_weight_records": int(
                    np.sum(positive & support)
                ),
                "n_selected_nonnull_positive_weight_records": int(
                    np.sum(positive & selected_nonnull)
                ),
                "supported_weight_share": (
                    support_weight_sum / weight_sum if weight_sum > 0.0 else np.nan
                ),
                "selected_nonnull_weight_share": (
                    selected_nonnull_weight_sum / weight_sum
                    if weight_sum > 0.0
                    else np.nan
                ),
                "supported_weighted_selected_ratio_mean": _weighted_mean_or_nan(
                    ratio_values[support],
                    weights[support],
                ),
                "selected_nonnull_weighted_selected_ratio_mean": _weighted_mean_or_nan(
                    ratio_values[selected_nonnull],
                    weights[selected_nonnull],
                ),
            }
        rows.append(
            {
                "weight_rule_id": rule_id,
                "n_records": int(table.shape[0]),
                "n_positive_weight_records": int(np.sum(positive)),
                "positive_weight_fraction": float(np.mean(positive)),
                "effective_sample_size": _effective_sample_size(weights),
                "max_weight_share": _max_weight_share(weights),
                "weighted_selected_ratio_mean": weighted_ratio,
                "unweighted_selected_ratio_mean": float(np.mean(ratios)),
                "support_status": support_status,
                "study_role": STUDY_ROLE,
                **labeled_metrics,
            }
        )
    return pd.DataFrame.from_records(rows)


def run_sibling_null_weight_rule_validation(
    *,
    records_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Run the sibling null-weight rule diagnostic from a CSV file."""
    records = pd.read_csv(records_path)
    rule_summary = evaluate_sibling_null_weight_rules(records)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "sibling_null_weight_rule_summary.csv"
    manifest_path = output_dir / "manifest.json"
    rule_summary.to_csv(summary_path, index=False)
    manifest = {
        "created_at_utc": format_timestamp_utc(),
        "study_role": STUDY_ROLE,
        "records_path": str(records_path),
        "outputs": {"summary": str(summary_path)},
        "interpretation": (
            "Diagnostic comparison of sibling null-weight rules. The output "
            "does not validate or install a production replacement rule."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"summary": summary_path, "manifest": manifest_path}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    outputs = run_sibling_null_weight_rule_validation(
        records_path=args.records,
        output_dir=args.output_dir,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "STUDY_ROLE",
    "WEIGHT_RULE_IDS",
    "add_sibling_null_weight_rule_columns",
    "evaluate_sibling_null_weight_rules",
    "run_sibling_null_weight_rule_validation",
]
