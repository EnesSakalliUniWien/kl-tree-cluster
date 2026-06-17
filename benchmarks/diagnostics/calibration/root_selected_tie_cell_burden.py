"""Discrete tie-cell burden diagnostic for root selected regions.

The smooth root selected-region margin diagnostic identifies binary overlap
roots as discrete tie-cell geometry. This panel quantifies that tie cell by
summing the log multiplicity of tied minimum merge choices across root-child
construction steps.

The resulting burden is diagnostic-only. It is not a calibrated p-value and it
does not change traversal.
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
from scipy import stats

SCHEMA_VERSION = "root_selected_tie_cell_burden/v1"
STUDY_ROLE = "diagnostic_root_selected_tie_cell_burden_not_calibration"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration.root_selected_tie_cell_burden"
)

DEFAULT_RESULT_ROOT = Path(
    "raw/assets/benchmark-results/specific_small_method_benchmark_20260615"
)
DEFAULT_ROOT_SELECTED_REGION_DIR = (
    DEFAULT_RESULT_ROOT / "root_selected_region_margins_overlap_case_family"
)
DEFAULT_ROOT_SELECTED_REGION_SUMMARY = (
    DEFAULT_ROOT_SELECTED_REGION_DIR / "root_selected_region_summary.csv"
)
DEFAULT_ROOT_SELECTED_REGION_MERGE_MARGINS = (
    DEFAULT_ROOT_SELECTED_REGION_DIR / "root_selected_region_merge_margins.csv"
)
DEFAULT_NEAR_ZERO_TOLERANCE = 1e-12

ROWS_OUTPUT = "root_selected_tie_cell_burden_rows.csv"
RELATIONSHIPS_OUTPUT = "root_selected_tie_cell_burden_relationships.csv"
MANIFEST_OUTPUT = "manifest.json"

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "root_selected_region_law_status",
    "root_sibling_selected_ratio",
    "root_child_balance",
    "root_child_construction_merge_count",
    "root_tie_step_count",
    "root_tie_step_fraction",
    "root_near_zero_margin_count",
    "root_near_zero_margin_fraction",
    "root_tie_cell_log_burden",
    "root_tie_cell_bits_burden",
    "root_tie_cell_mean_log_multiplicity",
    "root_tie_cell_geometric_mean_multiplicity",
    "root_tie_cell_max_multiplicity",
    "root_tie_cell_median_multiplicity",
    "root_tie_rank_log_burden",
    "root_tie_rank_mean_fraction",
    "root_tie_rank_median_fraction",
    "root_tie_rank_max_fraction",
    "root_left_tie_cell_log_burden",
    "root_right_tie_cell_log_burden",
    "root_tie_cell_side_log_burden_asymmetry",
    "root_candidate_pair_log_burden",
    "root_tie_to_candidate_log_fraction",
    "root_tie_cell_status",
)

RELATIONSHIP_COLUMNS = (
    "schema_version",
    "study_role",
    "target",
    "covariate",
    "relationship_status",
    "valid_pair_count",
    "unique_covariate_count",
    "spearman_r",
    "spearman_p_value",
    "log_log_pearson_r",
)

RELATIONSHIP_COVARIATES = (
    "root_tie_step_fraction",
    "root_near_zero_margin_fraction",
    "root_tie_cell_log_burden",
    "root_tie_cell_mean_log_multiplicity",
    "root_tie_cell_geometric_mean_multiplicity",
    "root_tie_cell_max_multiplicity",
    "root_tie_rank_log_burden",
    "root_tie_rank_mean_fraction",
    "root_tie_rank_median_fraction",
    "root_tie_rank_max_fraction",
    "root_tie_cell_side_log_burden_asymmetry",
    "root_tie_to_candidate_log_fraction",
)


@dataclass(frozen=True)
class RootSelectedTieCellBurdenConfig:
    """Input paths and numerical knobs for the tie-cell burden panel."""

    output_dir: Path
    root_selected_region_summary_path: Path = DEFAULT_ROOT_SELECTED_REGION_SUMMARY
    root_selected_region_merge_margins_path: Path = (
        DEFAULT_ROOT_SELECTED_REGION_MERGE_MARGINS
    )
    near_zero_tolerance: float = DEFAULT_NEAR_ZERO_TOLERANCE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--root-selected-region-summary-path",
        type=Path,
        default=DEFAULT_ROOT_SELECTED_REGION_SUMMARY,
    )
    parser.add_argument(
        "--root-selected-region-merge-margins-path",
        type=Path,
        default=DEFAULT_ROOT_SELECTED_REGION_MERGE_MARGINS,
    )
    parser.add_argument(
        "--near-zero-tolerance",
        type=float,
        default=DEFAULT_NEAR_ZERO_TOLERANCE,
    )
    return parser.parse_args()


def _json_default(value: object) -> object:
    if isinstance(value, RootSelectedTieCellBurdenConfig):
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


def _finite_median(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    return float(numeric.median()) if not numeric.empty else math.nan


def _safe_fraction(numerator: int, denominator: int) -> float:
    if int(denominator) <= 0:
        return math.nan
    return float(numerator) / float(denominator)


def _safe_exp(value: float) -> float:
    if not math.isfinite(value):
        return math.nan
    return float(math.exp(min(value, 700.0)))


def _side_log_burden(construction: pd.DataFrame, side: str) -> float:
    side_rows = construction[construction["root_child_side"].astype(str).eq(side)]
    if side_rows.empty:
        return 0.0
    multiplicity = (
        pd.to_numeric(side_rows["tied_minimum_pair_count"], errors="coerce")
        .fillna(1.0)
        .clip(lower=1.0)
        .to_numpy(dtype=float)
    )
    return float(np.log(multiplicity).sum())


def _summary_lookup(root_summary: pd.DataFrame) -> dict[str, dict[str, object]]:
    if root_summary.empty:
        return {}
    _require_columns(root_summary, {"case_id"}, "root summary")
    return {str(row["case_id"]): dict(row) for _, row in root_summary.iterrows()}


def build_root_selected_tie_cell_burden_rows(
    *,
    root_summary: pd.DataFrame,
    merge_margins: pd.DataFrame,
    near_zero_tolerance: float = DEFAULT_NEAR_ZERO_TOLERANCE,
) -> pd.DataFrame:
    """Return one tie-cell burden row per case."""
    _require_columns(
        merge_margins,
        {
            "case_id",
            "root_region_role",
            "root_child_side",
            "tied_minimum_pair_count",
            "merge_margin_to_nearest_competitor",
            "candidate_pair_count",
        },
        "merge margins",
    )
    summary_by_case = _summary_lookup(root_summary)
    records: list[dict[str, object]] = []
    for case_id, case_rows in merge_margins.groupby("case_id", sort=True):
        construction = case_rows[
            case_rows["root_region_role"].astype(str).eq("root_child_construction")
        ].copy()
        if construction.empty:
            raise ValueError(f"Case {case_id!r} has no root-child construction rows.")

        multiplicity = (
            pd.to_numeric(
                construction["tied_minimum_pair_count"],
                errors="coerce",
            )
            .fillna(1.0)
            .clip(lower=1.0)
            .to_numpy(dtype=float)
        )
        candidate_pair_count = (
            pd.to_numeric(construction["candidate_pair_count"], errors="coerce")
            .fillna(1.0)
            .clip(lower=1.0)
            .to_numpy(dtype=float)
        )
        if "selected_tie_rank_lexicographic" in construction:
            selected_tie_rank = (
                pd.to_numeric(
                    construction["selected_tie_rank_lexicographic"],
                    errors="coerce",
                )
                .fillna(1.0)
                .clip(lower=1.0)
                .to_numpy(dtype=float)
            )
        else:
            selected_tie_rank = np.ones_like(multiplicity)
        if "selected_tie_rank_fraction" in construction:
            selected_tie_rank_fraction = (
                pd.to_numeric(
                    construction["selected_tie_rank_fraction"],
                    errors="coerce",
                )
                .fillna(1.0)
                .clip(lower=0.0, upper=1.0)
                .to_numpy(dtype=float)
            )
        else:
            selected_tie_rank_fraction = selected_tie_rank / multiplicity
        margins = pd.to_numeric(
            construction["merge_margin_to_nearest_competitor"],
            errors="coerce",
        ).to_numpy(dtype=float)
        tie_mask = multiplicity > 1.0
        near_zero_mask = np.isfinite(margins) & (
            np.abs(margins) <= float(near_zero_tolerance)
        )
        construction_count = int(construction.shape[0])
        tie_step_count = int(np.count_nonzero(tie_mask))
        near_zero_count = int(np.count_nonzero(near_zero_mask))
        log_multiplicity = np.log(multiplicity)
        log_selected_tie_rank = np.log(selected_tie_rank)
        log_burden = float(log_multiplicity.sum())
        rank_log_burden = float(log_selected_tie_rank.sum())
        candidate_log_burden = float(np.log(candidate_pair_count).sum())
        left_log_burden = _side_log_burden(construction, "left")
        right_log_burden = _side_log_burden(construction, "right")
        side_denominator = left_log_burden + right_log_burden
        side_asymmetry = (
            abs(left_log_burden - right_log_burden) / side_denominator
            if side_denominator > 0.0
            else math.nan
        )
        summary = summary_by_case.get(str(case_id), {})
        law_status = str(
            summary.get("root_selected_region_law_status", "root_summary_not_joined")
        )
        if law_status == "discrete_tie_cell_geometry_required" and tie_step_count > 0:
            tie_cell_status = "discrete_tie_burden_observed_diagnostic_only"
        elif tie_step_count > 0:
            tie_cell_status = "tie_burden_observed_without_root_law_status"
        else:
            tie_cell_status = "no_tie_cell_burden_observed"

        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "case_id": str(case_id),
                "root_selected_region_law_status": law_status,
                "root_sibling_selected_ratio": _finite_float(
                    summary.get("root_sibling_selected_ratio", math.nan)
                ),
                "root_child_balance": _finite_float(
                    summary.get("root_child_balance", math.nan)
                ),
                "root_child_construction_merge_count": construction_count,
                "root_tie_step_count": tie_step_count,
                "root_tie_step_fraction": _safe_fraction(
                    tie_step_count,
                    construction_count,
                ),
                "root_near_zero_margin_count": near_zero_count,
                "root_near_zero_margin_fraction": _safe_fraction(
                    near_zero_count,
                    construction_count,
                ),
                "root_tie_cell_log_burden": log_burden,
                "root_tie_cell_bits_burden": log_burden / math.log(2.0),
                "root_tie_cell_mean_log_multiplicity": (
                    log_burden / float(construction_count)
                ),
                "root_tie_cell_geometric_mean_multiplicity": _safe_exp(
                    log_burden / float(construction_count)
                ),
                "root_tie_cell_max_multiplicity": float(np.max(multiplicity)),
                "root_tie_cell_median_multiplicity": _finite_median(
                    pd.Series(multiplicity)
                ),
                "root_tie_rank_log_burden": rank_log_burden,
                "root_tie_rank_mean_fraction": float(
                    np.mean(selected_tie_rank_fraction)
                ),
                "root_tie_rank_median_fraction": _finite_median(
                    pd.Series(selected_tie_rank_fraction)
                ),
                "root_tie_rank_max_fraction": float(
                    np.max(selected_tie_rank_fraction)
                ),
                "root_left_tie_cell_log_burden": left_log_burden,
                "root_right_tie_cell_log_burden": right_log_burden,
                "root_tie_cell_side_log_burden_asymmetry": side_asymmetry,
                "root_candidate_pair_log_burden": candidate_log_burden,
                "root_tie_to_candidate_log_fraction": (
                    log_burden - candidate_log_burden
                ),
                "root_tie_cell_status": tie_cell_status,
            }
        )
    return pd.DataFrame.from_records(records, columns=ROW_COLUMNS)


def summarize_tie_cell_relationships(rows: pd.DataFrame) -> pd.DataFrame:
    """Relate tie-cell burden variables to root selected ratio."""
    if rows.empty:
        return pd.DataFrame(columns=RELATIONSHIP_COLUMNS)
    _require_columns(rows, {"root_sibling_selected_ratio"}, "tie burden rows")
    target = pd.to_numeric(rows["root_sibling_selected_ratio"], errors="coerce")
    records: list[dict[str, object]] = []
    for covariate in RELATIONSHIP_COVARIATES:
        if covariate not in rows.columns:
            raise KeyError(f"Missing tie-cell covariate {covariate!r}.")
        values = pd.to_numeric(rows[covariate], errors="coerce")
        valid = np.isfinite(values.to_numpy(dtype=float)) & np.isfinite(
            target.to_numpy(dtype=float)
        )
        valid &= target.to_numpy(dtype=float) > 0.0
        valid_count = int(np.count_nonzero(valid))
        unique_count = int(np.unique(values.to_numpy(dtype=float)[valid]).shape[0])
        if valid_count < 3:
            status = "insufficient_valid_pairs"
            spearman_r = math.nan
            spearman_p = math.nan
            log_log_pearson = math.nan
        elif unique_count < 2:
            status = "constant_covariate"
            spearman_r = math.nan
            spearman_p = math.nan
            log_log_pearson = math.nan
        else:
            status = "evaluated"
            spearman = stats.spearmanr(values[valid], target[valid])
            spearman_r = float(spearman.statistic)
            spearman_p = float(spearman.pvalue)
            positive_covariate = values.to_numpy(dtype=float)[valid] > 0.0
            if bool(np.all(positive_covariate)):
                log_log_pearson = float(
                    np.corrcoef(
                        np.log(values.to_numpy(dtype=float)[valid]),
                        np.log(target.to_numpy(dtype=float)[valid]),
                    )[0, 1]
                )
            else:
                log_log_pearson = math.nan
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "target": "log_root_sibling_selected_ratio",
                "covariate": covariate,
                "relationship_status": status,
                "valid_pair_count": valid_count,
                "unique_covariate_count": unique_count,
                "spearman_r": spearman_r,
                "spearman_p_value": spearman_p,
                "log_log_pearson_r": log_log_pearson,
            }
        )
    return pd.DataFrame.from_records(records, columns=RELATIONSHIP_COLUMNS)


def evaluate_root_selected_tie_cell_burden(
    config: RootSelectedTieCellBurdenConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read inputs and return tie-cell rows plus relationship summaries."""
    root_summary = pd.read_csv(config.root_selected_region_summary_path)
    merge_margins = pd.read_csv(config.root_selected_region_merge_margins_path)
    rows = build_root_selected_tie_cell_burden_rows(
        root_summary=root_summary,
        merge_margins=merge_margins,
        near_zero_tolerance=float(config.near_zero_tolerance),
    )
    relationships = summarize_tie_cell_relationships(rows)
    return rows, relationships


def run_root_selected_tie_cell_burden(
    config: RootSelectedTieCellBurdenConfig,
) -> dict[str, Path]:
    """Write tie-cell burden outputs."""
    rows, relationships = evaluate_root_selected_tie_cell_burden(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_out = output_dir / ROWS_OUTPUT
    relationships_out = output_dir / RELATIONSHIPS_OUTPUT
    manifest_out = output_dir / MANIFEST_OUTPUT
    rows.to_csv(rows_out, index=False)
    relationships.to_csv(relationships_out, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at": datetime.now(UTC).isoformat(),
        "config": config,
        "row_count": int(rows.shape[0]),
        "relationship_row_count": int(relationships.shape[0]),
        "outputs": {
            "rows": rows_out,
            "relationships": relationships_out,
        },
    }
    manifest_out.write_text(
        json.dumps(manifest, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return {
        "rows": rows_out,
        "relationships": relationships_out,
        "manifest": manifest_out,
    }


def main() -> None:
    args = parse_args()
    run_root_selected_tie_cell_burden(
        RootSelectedTieCellBurdenConfig(
            output_dir=args.output_dir,
            root_selected_region_summary_path=args.root_selected_region_summary_path,
            root_selected_region_merge_margins_path=(
                args.root_selected_region_merge_margins_path
            ),
            near_zero_tolerance=float(args.near_zero_tolerance),
        )
    )


if __name__ == "__main__":
    main()
