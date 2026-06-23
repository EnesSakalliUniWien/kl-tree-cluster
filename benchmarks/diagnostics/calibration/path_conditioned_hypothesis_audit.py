"""Path-conditioned traversal hypothesis audit.

This diagnostic consumes existing benchmark and traversal audit artifacts. It
does not run clustering methods, change traversal decisions, or select methods
adaptively. Its purpose is to make the branch-length hypothesis falsifiable by
joining outcome deltas to path-conditioned traversal burden and by recording how
recent candidate methods connect to that hypothesis.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.shared.util.time import format_timestamp_utc

SCHEMA_VERSION = "path_conditioned_hypothesis_audit/v1"
STUDY_ROLE = "diagnostic_hypothesis_audit_not_policy"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration.path_conditioned_hypothesis_audit"
)

DEFAULT_OUTPUT_DIR = Path(
    "raw/assets/benchmark-results/path_conditioned_hypothesis_audit_20260618"
)
DEFAULT_PATH_AUDIT_DIR = Path(
    "raw/assets/benchmark-results/path_conditioned_traversal_audit_20260618"
)
DEFAULT_PROMOTION_AUDIT_DIR = Path(
    "raw/assets/benchmark-results/branch_length_candidate_promotion_audit_20260618"
)
DEFAULT_FULL_BIG_DIR = Path(
    "raw/assets/benchmark-results/branch_length_candidate_full_big_20260618"
)
DEFAULT_MANUAL_GUARDED_DIR = Path(
    "raw/assets/benchmark-results/manual_guarded_benchmark_run_direct_20260617"
)

CURRENT_METHOD = "tbs"
BRANCH_METHOD = "tbs_internal_filter_branch_length_v1"

BRANCH_CURRENT_TABLE = "branch_current_path_hypothesis_table.csv"
METHOD_CASE_TABLE = "method_case_path_hypothesis_table.csv"
BURDEN_SUMMARY = "path_burden_outcome_summary.csv"
RECENT_METHOD_CONNECTIONS = "recent_method_connection_summary.csv"
HYPOTHESIS_SOLUTIONS = "hypothesis_solution_matrix.csv"
REPORT_OUTPUT = "path_conditioned_hypothesis_report.md"
MANIFEST_OUTPUT = "manifest.json"
RUN_LOG_OUTPUT = "run.log"
VERIFICATION_LOG_OUTPUT = "verification.log"


@dataclass(frozen=True)
class PathConditionedHypothesisAuditConfig:
    """Configuration for the path-conditioned hypothesis audit."""

    output_dir: Path = DEFAULT_OUTPUT_DIR
    path_audit_dir: Path = DEFAULT_PATH_AUDIT_DIR
    promotion_audit_dir: Path = DEFAULT_PROMOTION_AUDIT_DIR
    full_big_dir: Path = DEFAULT_FULL_BIG_DIR
    manual_guarded_dir: Path = DEFAULT_MANUAL_GUARDED_DIR
    delta_epsilon: float = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--path-audit-dir", type=Path, default=DEFAULT_PATH_AUDIT_DIR)
    parser.add_argument(
        "--promotion-audit-dir", type=Path, default=DEFAULT_PROMOTION_AUDIT_DIR
    )
    parser.add_argument("--full-big-dir", type=Path, default=DEFAULT_FULL_BIG_DIR)
    parser.add_argument(
        "--manual-guarded-dir", type=Path, default=DEFAULT_MANUAL_GUARDED_DIR
    )
    parser.add_argument("--delta-epsilon", type=float, default=1e-9)
    return parser.parse_args()


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _log(log_path: Path, message: str) -> None:
    line = f"[{format_timestamp_utc()}] {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _is_true_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def _finite_median(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.median()) if not numeric.empty else math.nan


def _finite_min(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.min()) if not numeric.empty else math.nan


def _finite_max(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.max()) if not numeric.empty else math.nan


def _status_label(status: object) -> str:
    if status is None or pd.isna(status):
        return ""
    return str(status)


def _outcome_class(
    *,
    current_status: str,
    branch_status: str,
    branch_minus_current_ari: object,
    branch_stacked_pass_through_rows: float,
    branch_pass_through_rows: float,
    branch_mixed_boundary_rows: float,
    epsilon: float,
) -> str:
    current_ok = current_status == "ok"
    branch_ok = branch_status == "ok"
    if current_ok and not branch_ok:
        return "current_ok_branch_skip"
    if branch_ok and not current_ok:
        return "branch_ok_current_skip"
    if not current_ok and not branch_ok:
        return "both_skip_or_non_ok"

    try:
        delta = float(branch_minus_current_ari)
    except (TypeError, ValueError):
        return "unclassified_missing_delta"
    if not math.isfinite(delta):
        return "unclassified_missing_delta"

    stacked = float(branch_stacked_pass_through_rows or 0.0)
    pass_through = float(branch_pass_through_rows or 0.0)
    mixed = float(branch_mixed_boundary_rows or 0.0)

    if delta > epsilon:
        if stacked > 0:
            return "branch_gain_despite_stacked_pass_through"
        if pass_through > 0:
            return "branch_gain_with_isolated_pass_through"
        return "branch_gain_without_pass_through_burden"
    if delta < -epsilon:
        if stacked > 0:
            return "branch_loss_with_stacked_pass_through"
        if mixed > 0:
            return "branch_loss_with_mixed_boundary"
        if pass_through > 0:
            return "branch_loss_with_isolated_pass_through"
        return "branch_loss_without_path_burden"
    return "ari_tie_or_negligible_delta"


def _status_mismatch_class(
    *,
    current_status: object,
    branch_status: object,
    current_path_status: object,
    branch_path_status: object,
) -> str:
    current = _status_label(current_status)
    branch = _status_label(branch_status)
    current_path = _status_label(current_path_status)
    branch_path = _status_label(branch_path_status)
    if not current_path and not branch_path:
        return "path_audit_status_missing"
    mismatches = []
    if current_path and current != current_path:
        mismatches.append(f"current_{current}_path_{current_path}")
    if branch_path and branch != branch_path:
        mismatches.append(f"branch_{branch}_path_{branch_path}")
    if not mismatches:
        return "status_consistent"
    return "status_mismatch__" + "__".join(mismatches)


def _sign_with_epsilon(value: object, epsilon: float) -> int | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    if numeric > epsilon:
        return 1
    if numeric < -epsilon:
        return -1
    return 0


def _delta_connection_class(
    *,
    promotion_delta: object,
    path_delta: object,
    epsilon: float,
) -> str:
    promotion_sign = _sign_with_epsilon(promotion_delta, epsilon)
    path_sign = _sign_with_epsilon(path_delta, epsilon)
    if promotion_sign is None or path_sign is None:
        return "delta_missing_in_one_source"
    try:
        difference = abs(float(path_delta) - float(promotion_delta))
    except (TypeError, ValueError):
        return "delta_missing_in_one_source"
    if difference <= epsilon:
        return "delta_exact_match"
    if promotion_sign == path_sign:
        return "delta_direction_consistent"
    return "delta_direction_mismatch"


def aggregate_tuple_burden(tuple_rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate path-conditioned tuple burden by method and case."""
    if tuple_rows.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    actual_visited = _is_true_series(tuple_rows["actual_visited"])
    live = tuple_rows[actual_visited].copy()
    live["ancestor_pass_through_count"] = pd.to_numeric(
        live["ancestor_pass_through_count"], errors="coerce"
    ).fillna(0)
    live["ancestor_sibling_closed_edge_open_count"] = pd.to_numeric(
        live["ancestor_sibling_closed_edge_open_count"], errors="coerce"
    ).fillna(0)

    for (method, case_id), group in live.groupby(["method", "case_id"], sort=False):
        pass_rows = group[group["actual_decision"] == "pass_through"]
        boundary_rows = group[group["actual_decision"] == "boundary"]
        stacked_pass = pass_rows[pass_rows["ancestor_pass_through_count"] > 0]
        edge_open_closed_boundaries = boundary_rows[
            (boundary_rows["edge_traversal_action"] == "continue")
            & (~_is_true_series(boundary_rows["sibling_gate_open"]))
        ]
        truth_coherent_desc = (
            pd.to_numeric(
                pass_rows.get(
                    "descendant_truth_coherent_live_split_count",
                    pd.Series(dtype=float),
                ),
                errors="coerce",
            ).fillna(0)
            > 0
        )
        rows.append(
            {
                "method": method,
                "case_id": case_id,
                "live_rows_from_tuples": int(len(group)),
                "pass_through_rows_from_tuples": int(len(pass_rows)),
                "stacked_pass_through_rows_from_tuples": int(len(stacked_pass)),
                "max_ancestor_pass_through_count": _finite_max(
                    pass_rows["ancestor_pass_through_count"]
                ),
                "max_ancestor_sibling_closed_edge_open_count": _finite_max(
                    pass_rows["ancestor_sibling_closed_edge_open_count"]
                ),
                "median_pass_through_incoming_branch_length": _finite_median(
                    pass_rows["incoming_branch_length"]
                ),
                "min_pass_through_incoming_branch_length": _finite_min(
                    pass_rows["incoming_branch_length"]
                ),
                "max_pass_through_depth": _finite_max(pass_rows["depth"]),
                "pass_through_with_truth_coherent_descendant_rows": int(
                    truth_coherent_desc.sum()
                ),
                "truth_coherent_pass_through_rows": int(
                    (pass_rows["truth_audit_label"] == "truth_coherent_split_path").sum()
                ),
                "boundary_rows_from_tuples": int(len(boundary_rows)),
                "pure_boundary_rows_from_tuples": int(
                    (boundary_rows["truth_audit_label"] == "pure_boundary").sum()
                ),
                "mixed_boundary_rows_from_tuples": int(
                    (boundary_rows["truth_audit_label"] == "mixed_boundary").sum()
                ),
                "edge_open_closed_sibling_boundary_rows_from_tuples": int(
                    len(edge_open_closed_boundaries)
                ),
                "median_boundary_ancestor_pass_through_count": _finite_median(
                    boundary_rows["ancestor_pass_through_count"]
                ),
                "median_boundary_incoming_branch_length": _finite_median(
                    boundary_rows["incoming_branch_length"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _wide_case_metrics(method_case: pd.DataFrame) -> pd.DataFrame:
    selected = [
        "case_id",
        "method",
        "status",
        "ari",
        "found_clusters",
        "true_clusters",
        "path_conditioned_pass_through_rows",
        "stacked_pass_through_rows",
        "mixed_boundary_rows",
        "pure_boundary_rows",
        "edge_open_closed_sibling_boundary_rows",
        "median_pass_through_ancestor_pass_through_count",
        "median_pass_through_incoming_branch_length",
        "pass_through_with_truth_coherent_descendant_split_rows",
        "max_ancestor_pass_through_count",
        "max_ancestor_sibling_closed_edge_open_count",
        "min_pass_through_incoming_branch_length",
        "max_pass_through_depth",
    ]
    available = [column for column in selected if column in method_case.columns]
    wide = method_case[available].pivot(index="case_id", columns="method")
    wide.columns = [f"{method}_{column}" for column, method in wide.columns]
    return wide.reset_index()


def build_method_case_table(
    *,
    case_summary: pd.DataFrame,
    tuple_burden: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> pd.DataFrame:
    """Return one row per audited method/case with outcome and path burden."""
    method_case = case_summary.merge(
        tuple_burden,
        on=["method", "case_id"],
        how="left",
        validate="one_to_one",
    )
    pairwise_columns = [
        "case_id",
        "source_family",
        "feature_representation",
        "branch_minus_current_ari",
        "branch_vs_current_relation",
        "branch_vs_legacy_relation",
        f"{CURRENT_METHOD}_status",
        f"{BRANCH_METHOD}_status",
        "tbs_legacy_c2ef9a69_status",
        f"{CURRENT_METHOD}_found_clusters",
        f"{BRANCH_METHOD}_found_clusters",
        "tbs_legacy_c2ef9a69_found_clusters",
        CURRENT_METHOD,
        BRANCH_METHOD,
        "tbs_legacy_c2ef9a69",
    ]
    available = [column for column in pairwise_columns if column in pairwise.columns]
    method_case = method_case.merge(
        pairwise[available],
        on="case_id",
        how="left",
        validate="many_to_one",
    )
    method_case["schema_version"] = SCHEMA_VERSION
    method_case["study_role"] = STUDY_ROLE
    return method_case


def build_branch_current_table(
    *,
    method_case: pd.DataFrame,
    pairwise: pd.DataFrame,
    epsilon: float,
) -> pd.DataFrame:
    """Return one row per audited case comparing current and branch methods."""
    wide = _wide_case_metrics(method_case)
    pairwise_subset = pairwise[pairwise["case_id"].isin(wide["case_id"])].copy()
    table = pairwise_subset.merge(
        wide,
        on="case_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_path_audit"),
    )

    if f"{BRANCH_METHOD}_ari" in table.columns and f"{CURRENT_METHOD}_ari" in table.columns:
        table["path_audit_branch_minus_current_ari"] = (
            pd.to_numeric(table[f"{BRANCH_METHOD}_ari"], errors="coerce")
            - pd.to_numeric(table[f"{CURRENT_METHOD}_ari"], errors="coerce")
        )
    table["outcome_path_delta_connection"] = [
        _delta_connection_class(
            promotion_delta=promotion_delta,
            path_delta=path_delta,
            epsilon=epsilon,
        )
        for promotion_delta, path_delta in zip(
            table.get("branch_minus_current_ari", pd.Series(math.nan, index=table.index)),
            table.get(
                "path_audit_branch_minus_current_ari",
                pd.Series(math.nan, index=table.index),
            ),
            strict=False,
        )
    ]

    branch_stacked = table.get(
        f"{BRANCH_METHOD}_stacked_pass_through_rows", pd.Series(0, index=table.index)
    ).fillna(0)
    branch_pass = table.get(
        f"{BRANCH_METHOD}_path_conditioned_pass_through_rows",
        pd.Series(0, index=table.index),
    ).fillna(0)
    branch_mixed = table.get(
        f"{BRANCH_METHOD}_mixed_boundary_rows", pd.Series(0, index=table.index)
    ).fillna(0)
    current_status = table.get(
        f"{CURRENT_METHOD}_status", pd.Series("", index=table.index)
    )
    branch_status = table.get(
        f"{BRANCH_METHOD}_status", pd.Series("", index=table.index)
    )
    current_path_status = table.get(
        f"{CURRENT_METHOD}_status_path_audit", pd.Series("", index=table.index)
    )
    branch_path_status = table.get(
        f"{BRANCH_METHOD}_status_path_audit", pd.Series("", index=table.index)
    )
    table["outcome_path_status_connection"] = [
        _status_mismatch_class(
            current_status=current,
            branch_status=branch,
            current_path_status=current_path,
            branch_path_status=branch_path,
        )
        for current, branch, current_path, branch_path in zip(
            current_status,
            branch_status,
            current_path_status,
            branch_path_status,
            strict=False,
        )
    ]
    table["promotion_outcome_hypothesis_class"] = [
        _outcome_class(
            current_status=_status_label(current_status),
            branch_status=_status_label(branch_status),
            branch_minus_current_ari=delta,
            branch_stacked_pass_through_rows=stacked,
            branch_pass_through_rows=pass_rows,
            branch_mixed_boundary_rows=mixed,
            epsilon=epsilon,
        )
        for current_status, branch_status, delta, stacked, pass_rows, mixed in zip(
            current_status,
            branch_status,
            table.get("branch_minus_current_ari", pd.Series(math.nan, index=table.index)),
            branch_stacked,
            branch_pass,
            branch_mixed,
            strict=False,
        )
    ]
    table["path_audit_hypothesis_class"] = [
        _outcome_class(
            current_status=_status_label(current_path),
            branch_status=_status_label(branch_path),
            branch_minus_current_ari=delta,
            branch_stacked_pass_through_rows=stacked,
            branch_pass_through_rows=pass_rows,
            branch_mixed_boundary_rows=mixed,
            epsilon=epsilon,
        )
        for current_path, branch_path, delta, stacked, pass_rows, mixed in zip(
            current_path_status,
            branch_path_status,
            table.get(
                "path_audit_branch_minus_current_ari",
                pd.Series(math.nan, index=table.index),
            ),
            branch_stacked,
            branch_pass,
            branch_mixed,
            strict=False,
        )
    ]
    table["path_hypothesis_class"] = np.where(
        table["outcome_path_status_connection"] == "status_consistent",
        table["promotion_outcome_hypothesis_class"],
        "outcome_path_status_mismatch",
    )
    table["schema_version"] = SCHEMA_VERSION
    table["study_role"] = STUDY_ROLE
    return table


def summarize_burden_outcomes(branch_current_table: pd.DataFrame) -> pd.DataFrame:
    """Summarize branch/current outcome classes against path burden."""
    if branch_current_table.empty:
        return pd.DataFrame()

    branch_stacked_column = f"{BRANCH_METHOD}_stacked_pass_through_rows"
    branch_pass_column = f"{BRANCH_METHOD}_path_conditioned_pass_through_rows"
    branch_mixed_column = f"{BRANCH_METHOD}_mixed_boundary_rows"
    current_stacked_column = f"{CURRENT_METHOD}_stacked_pass_through_rows"
    rows: list[dict[str, object]] = []
    for label, group in branch_current_table.groupby("path_hypothesis_class", sort=True):
        delta = pd.to_numeric(group["branch_minus_current_ari"], errors="coerce")
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "path_hypothesis_class": label,
                "cases": int(len(group)),
                "mean_branch_minus_current_ari": (
                    float(delta.mean()) if delta.notna().any() else math.nan
                ),
                "median_branch_minus_current_ari": (
                    float(delta.median()) if delta.notna().any() else math.nan
                ),
                "branch_stacked_pass_through_rows_sum": int(
                    pd.to_numeric(group.get(branch_stacked_column, 0), errors="coerce")
                    .fillna(0)
                    .sum()
                ),
                "current_stacked_pass_through_rows_sum": int(
                    pd.to_numeric(group.get(current_stacked_column, 0), errors="coerce")
                    .fillna(0)
                    .sum()
                ),
                "branch_pass_through_rows_sum": int(
                    pd.to_numeric(group.get(branch_pass_column, 0), errors="coerce")
                    .fillna(0)
                    .sum()
                ),
                "branch_mixed_boundary_rows_sum": int(
                    pd.to_numeric(group.get(branch_mixed_column, 0), errors="coerce")
                    .fillna(0)
                    .sum()
                ),
                "case_ids": ";".join(str(case_id) for case_id in group["case_id"]),
            }
        )
    return pd.DataFrame(rows)


def _method_connection_text(method: str) -> tuple[str, str]:
    mapping = {
        "tbs": (
            "current_reference",
            "Baseline guarded TBS; path-conditioned burden measures what the current traversal already avoids or tolerates.",
        ),
        "tbs_current": (
            "current_reference_alias",
            "Direct-run alias for current TBS in the guarded smoke; should match tbs unless dispatch configuration differs.",
        ),
        "tbs_legacy_c2ef9a69": (
            "legacy_power_comparator",
            "Comparator with high completion power but no guarded fail-closed behavior; useful witness, not production rule.",
        ),
        "tbs_internal_filter_v1": (
            "internal_support_filter",
            "Tests the internal-barycenter support idea without branch-length state; separates support filtering from branch-length conditioning.",
        ),
        BRANCH_METHOD: (
            "branch_length_candidate",
            "Fixed candidate under this audit; branch length is evaluated only through path-conditioned support and not as a router.",
        ),
        "tbs_bandwidth_context_v1": (
            "neighborhood_bandwidth_context",
            "Regional bandwidth support regularizer; connects to selected-neighborhood evidence but matched current in the smoke.",
        ),
        "tbs_rescued_legacy_v1": (
            "combined_guarded_legacy_candidate",
            "Combined branch-length, support, passthrough, and bandwidth components; smoke result did not justify promotion.",
        ),
    }
    return mapping.get(method, ("other", "No specific connection annotation."))


def build_recent_method_connections(
    *,
    full_summary: pd.DataFrame,
    manual_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize how recent method variants connect to the hypothesis."""
    rows: list[dict[str, object]] = []
    for source, summary in (
        ("full_big_branch_length_candidate", full_summary),
        ("manual_guarded_six_method_smoke", manual_summary),
    ):
        for row in summary.itertuples(index=False):
            method = str(row.method)
            role, interpretation = _method_connection_text(method)
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_role": STUDY_ROLE,
                    "source": source,
                    "method": method,
                    "method_hypothesis_role": role,
                    "rows": int(getattr(row, "rows", 0)),
                    "ok": int(getattr(row, "ok", 0)),
                    "skip": int(getattr(row, "skip", 0)),
                    "fail": int(getattr(row, "fail", 0)),
                    "exact_k": int(getattr(row, "exact_k", 0)),
                    "mean_ari": float(getattr(row, "mean_ari", math.nan)),
                    "mean_found_clusters": float(
                        getattr(row, "mean_found_clusters", math.nan)
                    ),
                    "connection_to_path_hypothesis": interpretation,
                }
            )
    return pd.DataFrame(rows)


def build_hypothesis_solution_matrix() -> pd.DataFrame:
    rows = [
        {
            "hypothesis_step": "isolated_pass_through",
            "diagnostic_signal": "pass_through_rows > 0 and stacked_pass_through_rows == 0",
            "solution": "Keep as diagnostic compression unless a future run shows truth-coherent descendant splits below isolated pass-throughs.",
            "admissibility_constraint": "No runtime truth labels and no outcome-conditioned method selection.",
            "connection_to_recent_methods": "Current TBS, bandwidth context, and branch-length can all produce isolated pass-throughs; the audit checks whether these are benign.",
        },
        {
            "hypothesis_step": "stacked_pass_through",
            "diagnostic_signal": "ancestor_pass_through_count > 0 under sibling_gate_open=False and incoming_edge_open=True",
            "solution": "Treat as a candidate failure marker; if predictive across reruns, encode only as a predeclared guard in a new fixed method variant.",
            "admissibility_constraint": "The guard must be calibrated or predeclared before benchmarking; do not switch methods after observing outcomes.",
            "connection_to_recent_methods": "This is the strongest branch-length traversal-specific burden and separates branch-length from rescued-legacy/bandwidth components.",
        },
        {
            "hypothesis_step": "branch_length_ambiguity",
            "diagnostic_signal": "short incoming branch lengths appear in stacked phylogenetic chains but long isolated pass-throughs also exist",
            "solution": "Use branch length only conditioned on path state: upper-chain sibling-closed continuation plus path quantiles.",
            "admissibility_constraint": "Raw branch length alone is not a valid decision rule.",
            "connection_to_recent_methods": "Branch-length internal filter uses branch_length_state; bandwidth context uses regional support and should not be conflated with branch length.",
        },
        {
            "hypothesis_step": "mixed_boundary_or_root_stop",
            "diagnostic_signal": "mixed_boundary_rows > 0, especially without pass-through ancestry",
            "solution": "Handle as calibration/support failure or conservative root stop; evaluate fail-closed behavior separately from pass-through chains.",
            "admissibility_constraint": "Truth-label boundary diagnosis is benchmark-only evidence and cannot enter runtime decisions.",
            "connection_to_recent_methods": "Hard-overlap fail-closed behavior belongs to guarded current/internal-filter methods, not legacy completion.",
        },
        {
            "hypothesis_step": "legacy_and_rescued_methods",
            "diagnostic_signal": "legacy completes unsupported cases; rescued legacy underperforms in guarded smoke",
            "solution": "Keep legacy as a comparator/power witness and rescued legacy as a negative control unless new support evidence changes the picture.",
            "admissibility_constraint": "Do not promote legacy-style completion when selected-null support is missing.",
            "connection_to_recent_methods": "Manual smoke showed rescued legacy weak mean ARI while branch-length internal filtering had the best smoke profile.",
        },
    ]
    out = pd.DataFrame(rows)
    out.insert(0, "schema_version", SCHEMA_VERSION)
    out.insert(1, "study_role", STUDY_ROLE)
    return out


def _format_count_table(df: pd.DataFrame, columns: list[str], limit: int = 20) -> str:
    if df.empty:
        return "(empty)"
    return df[columns].head(limit).to_markdown(index=False)


def build_report(
    *,
    branch_current: pd.DataFrame,
    burden_summary: pd.DataFrame,
    recent_connections: pd.DataFrame,
    solution_matrix: pd.DataFrame,
) -> str:
    branch_path_column = f"{BRANCH_METHOD}_stacked_pass_through_rows"
    status_mismatch_count = int(
        (branch_current["path_hypothesis_class"] == "outcome_path_status_mismatch").sum()
    )
    delta_exact_match_count = int(
        (branch_current["outcome_path_delta_connection"] == "delta_exact_match").sum()
    )
    selected_columns = [
        "case_id",
        "branch_vs_current_relation",
        "branch_minus_current_ari",
        "path_audit_branch_minus_current_ari",
        "outcome_path_status_connection",
        "outcome_path_delta_connection",
        "path_hypothesis_class",
        "path_audit_hypothesis_class",
        "promotion_outcome_hypothesis_class",
        f"{CURRENT_METHOD}_status",
        f"{BRANCH_METHOD}_status",
        f"{CURRENT_METHOD}_status_path_audit",
        f"{BRANCH_METHOD}_status_path_audit",
        f"{CURRENT_METHOD}_path_conditioned_pass_through_rows",
        f"{BRANCH_METHOD}_path_conditioned_pass_through_rows",
        f"{CURRENT_METHOD}_stacked_pass_through_rows",
        branch_path_column,
        f"{BRANCH_METHOD}_mixed_boundary_rows",
    ]
    selected_columns = [column for column in selected_columns if column in branch_current]
    lines = [
        "# Path-Conditioned Hypothesis Audit 2026-06-18",
        "",
        "## Summary",
        "",
        "This audit joins fixed branch-length/current outcomes to the path-conditioned traversal tuples. It is a falsification surface, not a production policy. The core hypothesis is that branch-length evidence is meaningful only when conditioned on upper-chain traversal state, especially repeated sibling-closed edge-open continuation.",
        "",
        f"This run has `{status_mismatch_count}` outcome/path status mismatch rows and `{delta_exact_match_count}` exact branch-minus-current delta matches.",
        "",
        "## Outcome Classes",
        "",
        _format_count_table(
            burden_summary,
            [
                "path_hypothesis_class",
                "cases",
                "mean_branch_minus_current_ari",
                "branch_stacked_pass_through_rows_sum",
                "branch_pass_through_rows_sum",
                "branch_mixed_boundary_rows_sum",
            ],
        ),
        "",
        "## Audited Case Table",
        "",
        _format_count_table(branch_current, selected_columns),
        "",
        "## Method Connections",
        "",
        _format_count_table(
            recent_connections,
            [
                "source",
                "method",
                "method_hypothesis_role",
                "rows",
                "ok",
                "skip",
                "exact_k",
                "mean_ari",
            ],
        ),
        "",
        "## Solution Matrix",
        "",
        _format_count_table(
            solution_matrix,
            [
                "hypothesis_step",
                "diagnostic_signal",
                "solution",
                "admissibility_constraint",
            ],
        ),
        "",
        "## Interpretation",
        "",
        "- Stacked pass-through is a candidate failure marker, not a method selector.",
        "- Branch length should be read through path state and support, not as a standalone threshold.",
        "- Cases with outcome/path status mismatch are connection failures between artifacts; they should be rerun under one runner before path burden is used to explain a full-suite delta.",
        "- Cases with delta-direction consistency but different delta magnitude are weaker evidence than exact matches, but still useful for sign-level hypothesis checks.",
        "- Legacy remains a power/completion comparator; rescued legacy and bandwidth-context results are connection evidence but not promotion evidence.",
        "- Any eventual guard must become a new fixed, predeclared method variant and then be rerun.",
    ]
    return "\n".join(lines) + "\n"


def run_path_conditioned_hypothesis_audit(
    config: PathConditionedHypothesisAuditConfig,
) -> dict[str, Path]:
    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_log = output_dir / RUN_LOG_OUTPUT
    run_log.write_text("", encoding="utf-8")

    _log(run_log, f"generated_by={GENERATED_BY}")
    _log(run_log, f"cwd={Path.cwd()}")
    _log(run_log, f"python={sys.version.split()[0]} platform={platform.platform()}")
    _log(run_log, f"argv={sys.argv}")

    path_audit_dir = config.path_audit_dir.expanduser().resolve()
    promotion_dir = config.promotion_audit_dir.expanduser().resolve()
    full_dir = config.full_big_dir.expanduser().resolve()
    manual_dir = config.manual_guarded_dir.expanduser().resolve()

    case_summary = _read_csv(path_audit_dir / "path_conditioned_traversal_case_summary.csv")
    tuple_rows = _read_csv(path_audit_dir / "path_conditioned_traversal_tuples.csv")
    pairwise = _read_csv(promotion_dir / "case_pairwise_audit.csv")
    full_summary = _read_csv(full_dir / "summary_by_method.csv")
    manual_summary = _read_csv(manual_dir / "summary_by_method.csv")

    _log(run_log, f"path_case_summary_rows={len(case_summary)}")
    _log(run_log, f"path_tuple_rows={len(tuple_rows)}")
    _log(run_log, f"pairwise_rows={len(pairwise)}")

    tuple_burden = aggregate_tuple_burden(tuple_rows)
    method_case = build_method_case_table(
        case_summary=case_summary,
        tuple_burden=tuple_burden,
        pairwise=pairwise,
    )
    branch_current = build_branch_current_table(
        method_case=method_case,
        pairwise=pairwise,
        epsilon=float(config.delta_epsilon),
    )
    burden_summary = summarize_burden_outcomes(branch_current)
    recent_connections = build_recent_method_connections(
        full_summary=full_summary,
        manual_summary=manual_summary,
    )
    solution_matrix = build_hypothesis_solution_matrix()
    report = build_report(
        branch_current=branch_current,
        burden_summary=burden_summary,
        recent_connections=recent_connections,
        solution_matrix=solution_matrix,
    )

    outputs = {
        "branch_current_path_hypothesis_table": output_dir / BRANCH_CURRENT_TABLE,
        "method_case_path_hypothesis_table": output_dir / METHOD_CASE_TABLE,
        "path_burden_outcome_summary": output_dir / BURDEN_SUMMARY,
        "recent_method_connection_summary": output_dir / RECENT_METHOD_CONNECTIONS,
        "hypothesis_solution_matrix": output_dir / HYPOTHESIS_SOLUTIONS,
        "path_conditioned_hypothesis_report": output_dir / REPORT_OUTPUT,
        "manifest": output_dir / MANIFEST_OUTPUT,
        "run_log": run_log,
        "verification_log": output_dir / VERIFICATION_LOG_OUTPUT,
    }
    branch_current.to_csv(outputs["branch_current_path_hypothesis_table"], index=False)
    method_case.to_csv(outputs["method_case_path_hypothesis_table"], index=False)
    burden_summary.to_csv(outputs["path_burden_outcome_summary"], index=False)
    recent_connections.to_csv(outputs["recent_method_connection_summary"], index=False)
    solution_matrix.to_csv(outputs["hypothesis_solution_matrix"], index=False)
    outputs["path_conditioned_hypothesis_report"].write_text(report, encoding="utf-8")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at_utc": format_timestamp_utc(),
        "output_dir": str(output_dir),
        "inputs": {
            "path_audit_dir": str(path_audit_dir),
            "promotion_audit_dir": str(promotion_dir),
            "full_big_dir": str(full_dir),
            "manual_guarded_dir": str(manual_dir),
        },
        "outputs": {key: str(path) for key, path in outputs.items()},
        "counts": {
            "branch_current_cases": int(len(branch_current)),
            "method_case_rows": int(len(method_case)),
            "burden_summary_rows": int(len(burden_summary)),
            "recent_method_connection_rows": int(len(recent_connections)),
            "solution_rows": int(len(solution_matrix)),
        },
        "environment": {
            "cwd": str(Path.cwd()),
            "python": sys.version,
            "platform": platform.platform(),
        },
        "argv": sys.argv,
    }
    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    _log(run_log, f"wrote_outputs={outputs}")
    return outputs


def main() -> None:
    args = parse_args()
    config = PathConditionedHypothesisAuditConfig(
        output_dir=args.output_dir,
        path_audit_dir=args.path_audit_dir,
        promotion_audit_dir=args.promotion_audit_dir,
        full_big_dir=args.full_big_dir,
        manual_guarded_dir=args.manual_guarded_dir,
        delta_epsilon=float(args.delta_epsilon),
    )
    run_path_conditioned_hypothesis_audit(config)


if __name__ == "__main__":
    main()
