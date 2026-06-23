"""Cached diagnostics for the path-conditioned barycentric action equation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCHEMA_VERSION = "path_conditioned_barycentric_action/v1"
STUDY_ROLE = "diagnostic_path_conditioned_barycentric_action_not_calibration"
LABEL_PROVENANCE = "posthoc_final_assignment_parent_cluster_count"

GEOMETRY_COLUMNS = [
    "weighting",
    "block_name",
    "parent_size",
    "balance_fraction",
    "sibling_separation_to_parent_radius_ratio",
    "sibling_separation_to_child_radius_ratio",
    "parent_centroid_norm",
    "parent_centroid_angle_to_leading_axis_deg",
    "parent_centroid_independent_fraction",
    "abs_sibling_common_axis_mean_delta",
    "parent_dominant_cluster_fraction",
    "parent_cluster_count",
]

VALIDATION_COLUMNS = [
    "model_id",
    "holdout_run_id",
    "n_train",
    "n_test",
    "test_pure_rate",
    "auc_for_pure_fragment",
    "study_role",
]


def _require_columns(table: pd.DataFrame, columns: list[str], table_name: str) -> None:
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def _finite_median(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    return float(values.median()) if values.notna().any() else math.nan


def _finite_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    return float(values.mean()) if values.notna().any() else math.nan


def _status_count(table: pd.DataFrame, column: str, value: object) -> int:
    if column not in table.columns:
        return 0
    return int(table[column].eq(value).sum())


def summarize_kak_geometry(geometry: pd.DataFrame) -> pd.DataFrame:
    """Summarize cached KAK internal-tree geometry by weighting/block run id."""
    _require_columns(geometry, GEOMETRY_COLUMNS, "kak internal geometry")
    table = geometry.copy()
    table["run_id"] = table["weighting"].astype(str) + "__" + table["block_name"].astype(str)

    rows: list[dict[str, object]] = []
    for run_id, group in table.groupby("run_id", sort=True):
        row = {
            "run_id": run_id,
            "weighting": str(group["weighting"].iloc[0]),
            "block_name": str(group["block_name"].iloc[0]),
            "n_internal_nodes": int(len(group)),
            "parent_size_median": _finite_median(group["parent_size"]),
            "balance_fraction_median": _finite_median(group["balance_fraction"]),
            "parent_radius_median": _finite_median(group["parent_centroid_norm"]),
            "angle_to_leading_axis_deg_median": _finite_median(
                group["parent_centroid_angle_to_leading_axis_deg"]
            ),
            "independent_fraction_median": _finite_median(
                group["parent_centroid_independent_fraction"]
            ),
            "sibling_separation_parent_ratio_median": _finite_median(
                group["sibling_separation_to_parent_radius_ratio"]
            ),
            "sibling_separation_child_ratio_median": _finite_median(
                group["sibling_separation_to_child_radius_ratio"]
            ),
            "abs_common_axis_gap_median": _finite_median(
                group["abs_sibling_common_axis_mean_delta"]
            ),
            "parent_dominant_cluster_fraction_median": _finite_median(
                group["parent_dominant_cluster_fraction"]
            ),
            "parent_cluster_count_median": _finite_median(group["parent_cluster_count"]),
            "study_role": STUDY_ROLE,
        }
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def summarize_kak_validation(validation: pd.DataFrame) -> pd.DataFrame:
    """Summarize cached leave-one-block-out KAK fragmentation validation rows."""
    _require_columns(validation, VALIDATION_COLUMNS, "kak fragmentation validation")
    rows: list[dict[str, object]] = []
    for model_id, group in validation.groupby("model_id", sort=True):
        auc = pd.to_numeric(group["auc_for_pure_fragment"], errors="coerce")
        rows.append(
            {
                "model_id": model_id,
                "n_holdouts": int(len(group)),
                "median_auc": float(auc.median()) if auc.notna().any() else math.nan,
                "mean_auc": float(auc.mean()) if auc.notna().any() else math.nan,
                "min_auc": float(auc.min()) if auc.notna().any() else math.nan,
                "max_auc": float(auc.max()) if auc.notna().any() else math.nan,
                "median_test_pure_rate": _finite_median(group["test_pure_rate"]),
                "study_role": STUDY_ROLE,
            }
        )
    return pd.DataFrame.from_records(rows)


def annotate_kak_radius_angle_action(geometry: pd.DataFrame) -> pd.DataFrame:
    """Add diagnostic-only radius/angle/action annotations to KAK merge rows."""
    _require_columns(geometry, GEOMETRY_COLUMNS, "kak internal geometry")
    table = geometry.copy()
    table["run_id"] = table["weighting"].astype(str) + "__" + table["block_name"].astype(str)
    balance = pd.to_numeric(table["balance_fraction"], errors="coerce").clip(0.0, 0.5)
    separation = pd.to_numeric(
        table["sibling_separation_to_parent_radius_ratio"],
        errors="coerce",
    )
    action_proxy = balance * (1.0 - balance) * separation.pow(2)
    angle = pd.to_numeric(
        table["parent_centroid_angle_to_leading_axis_deg"],
        errors="coerce",
    ).fillna(0.0)
    independent = pd.to_numeric(
        table["parent_centroid_independent_fraction"],
        errors="coerce",
    ).fillna(0.0)
    parent_cluster_count = pd.to_numeric(
        table["parent_cluster_count"],
        errors="coerce",
    )

    annotated = pd.DataFrame(
        {
            "run_id": table["run_id"],
            "weighting": table["weighting"].astype(str),
            "block_name": table["block_name"].astype(str),
            "parent_linkage_id": table.get("parent_linkage_id", pd.NA),
            "parent_size": pd.to_numeric(table["parent_size"], errors="coerce"),
            "balance_fraction": balance,
            "geometry_parent_radius": pd.to_numeric(
                table["parent_centroid_norm"],
                errors="coerce",
            ),
            "geometry_angle_to_leading_axis_deg": angle,
            "geometry_independent_radius_fraction": independent,
            "geometry_sibling_separation_parent_ratio": separation,
            "geometry_sibling_separation_child_ratio": pd.to_numeric(
                table["sibling_separation_to_child_radius_ratio"],
                errors="coerce",
            ),
            "geometry_abs_common_axis_gap": pd.to_numeric(
                table["abs_sibling_common_axis_mean_delta"],
                errors="coerce",
            ),
            "action_budget_proxy": action_proxy,
            "action_budget_proxy_capped": action_proxy.clip(upper=1.0),
            "angular_shell_risk_score": (
                action_proxy.clip(upper=1.0) * (angle / 90.0).clip(0.0, 1.0) * independent
            ),
            "parent_dominant_cluster_fraction": pd.to_numeric(
                table["parent_dominant_cluster_fraction"],
                errors="coerce",
            ),
            "parent_cluster_count": parent_cluster_count,
            "is_null_context": parent_cluster_count.le(1).fillna(False),
            "is_signal_context": parent_cluster_count.gt(1).fillna(False),
            "geometry_label_provenance": LABEL_PROVENANCE,
            "study_role": STUDY_ROLE,
        }
    )
    return annotated


def evaluate_action_budget_guard_panel(
    annotations: pd.DataFrame,
    *,
    action_thresholds: tuple[float, ...] = (0.5, 0.75, 0.9),
    angle_thresholds: tuple[float, ...] = (45.0, 60.0, 75.0),
    independent_thresholds: tuple[float, ...] = (0.85,),
) -> pd.DataFrame:
    """Evaluate diagnostic high-action angular-shell guard thresholds."""
    required = [
        "action_budget_proxy_capped",
        "geometry_angle_to_leading_axis_deg",
        "geometry_independent_radius_fraction",
        "is_null_context",
        "is_signal_context",
    ]
    _require_columns(annotations, required, "KAK radius/angle annotations")
    null_mask = annotations["is_null_context"].astype(bool)
    signal_mask = annotations["is_signal_context"].astype(bool)
    base_null_rate = float(null_mask.mean()) if len(annotations) else math.nan
    n_pure_contexts = int(null_mask.sum())
    n_mixed_contexts = int(signal_mask.sum())
    rows: list[dict[str, object]] = []
    for action_threshold in action_thresholds:
        for angle_threshold in angle_thresholds:
            for independent_threshold in independent_thresholds:
                flag = (
                    pd.to_numeric(
                        annotations["action_budget_proxy_capped"],
                        errors="coerce",
                    ).ge(action_threshold)
                    & pd.to_numeric(
                        annotations["geometry_angle_to_leading_axis_deg"],
                        errors="coerce",
                    ).ge(angle_threshold)
                    & pd.to_numeric(
                        annotations["geometry_independent_radius_fraction"],
                        errors="coerce",
                    ).ge(independent_threshold)
                )
                flagged = int(flag.sum())
                pure_flagged = int((flag & null_mask).sum())
                mixed_flagged = int((flag & signal_mask).sum())
                precision = float(pure_flagged / flagged) if flagged else math.nan
                rows.append(
                    {
                        "guard_id": (
                            f"action_ge_{action_threshold:g}__angle_ge_"
                            f"{angle_threshold:g}__ind_ge_{independent_threshold:g}"
                        ),
                        "action_threshold": float(action_threshold),
                        "angle_threshold_deg": float(angle_threshold),
                        "independent_fraction_threshold": float(independent_threshold),
                        "n_rows": int(len(annotations)),
                        "n_pure_contexts": n_pure_contexts,
                        "n_mixed_contexts": n_mixed_contexts,
                        "n_flagged": flagged,
                        "n_pure_flagged": pure_flagged,
                        "n_mixed_flagged": mixed_flagged,
                        "flag_rate": float(flag.mean()) if len(flag) else math.nan,
                        "pure_fragment_flag_rate": (
                            float((flag & null_mask).sum() / null_mask.sum())
                            if null_mask.any()
                            else math.nan
                        ),
                        "mixed_context_flag_rate": (
                            float((flag & signal_mask).sum() / signal_mask.sum())
                            if signal_mask.any()
                            else math.nan
                        ),
                        "diagnostic_precision_for_pure_fragment": precision,
                        "diagnostic_lift_over_base_pure_rate": (
                            float(precision / base_null_rate)
                            if math.isfinite(precision)
                            and math.isfinite(base_null_rate)
                            and base_null_rate > 0
                            else math.nan
                        ),
                        "false_guard_signal_fraction": (
                            float(mixed_flagged / flagged) if flagged else math.nan
                        ),
                        "label_provenance": LABEL_PROVENANCE,
                        "study_role": STUDY_ROLE,
                        "recursive_decision": (
                            "candidate_guard_validation_panel"
                            if math.isfinite(precision)
                            and precision >= 0.65
                            and (
                                not signal_mask.any()
                                or (mixed_flagged / max(int(signal_mask.sum()), 1)) <= 0.20
                            )
                            else "diagnostic_only"
                        ),
                    }
                )
    return pd.DataFrame.from_records(rows)


def evaluate_action_budget_guard_utility(
    guard_panel: pd.DataFrame,
    *,
    mixed_context_cost_ratios: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 5.0),
) -> pd.DataFrame:
    """Score guard usefulness under explicit mixed-context false-guard costs."""
    required = [
        "guard_id",
        "n_rows",
        "n_flagged",
        "n_pure_flagged",
        "n_mixed_flagged",
        "diagnostic_precision_for_pure_fragment",
    ]
    _require_columns(guard_panel, required, "action-budget guard panel")

    rows: list[dict[str, object]] = []
    for guard in guard_panel.itertuples(index=False):
        n_rows = int(getattr(guard, "n_rows"))
        n_flagged = int(getattr(guard, "n_flagged"))
        n_pure_flagged = int(getattr(guard, "n_pure_flagged"))
        n_mixed_flagged = int(getattr(guard, "n_mixed_flagged"))
        break_even = (
            float(n_pure_flagged / n_mixed_flagged)
            if n_mixed_flagged > 0
            else math.inf
        )
        for cost_ratio in mixed_context_cost_ratios:
            net_utility = float(n_pure_flagged - cost_ratio * n_mixed_flagged)
            rows.append(
                {
                    "guard_id": str(getattr(guard, "guard_id")),
                    "mixed_context_cost_ratio": float(cost_ratio),
                    "n_rows": n_rows,
                    "n_flagged": n_flagged,
                    "n_pure_flagged": n_pure_flagged,
                    "n_mixed_flagged": n_mixed_flagged,
                    "net_utility": net_utility,
                    "net_utility_per_row": (
                        float(net_utility / n_rows) if n_rows else math.nan
                    ),
                    "net_utility_per_flag": (
                        float(net_utility / n_flagged) if n_flagged else math.nan
                    ),
                    "break_even_mixed_context_cost_ratio": break_even,
                    "utility_positive": bool(net_utility > 0),
                    "diagnostic_precision_for_pure_fragment": float(
                        getattr(guard, "diagnostic_precision_for_pure_fragment")
                    ),
                    "pure_fragment_flag_rate": float(
                        getattr(guard, "pure_fragment_flag_rate")
                    ),
                    "mixed_context_flag_rate": float(
                        getattr(guard, "mixed_context_flag_rate")
                    ),
                    "label_provenance": LABEL_PROVENANCE,
                    "study_role": STUDY_ROLE,
                }
            )
    return pd.DataFrame.from_records(rows)


def summarize_benchmark(benchmark: pd.DataFrame | None) -> dict[str, Any]:
    """Return compact benchmark evidence used by the missing-equation panel."""
    if benchmark is None or benchmark.empty:
        return {
            "available": False,
            "n_rows": 0,
            "n_tbs_rows": 0,
            "n_tbs_skips": 0,
            "n_tbs_calibration_support_skips": 0,
            "n_tbs_under_split_rows": 0,
            "tbs_mean_ari_ok": math.nan,
        }

    method = benchmark.get("method", pd.Series("", index=benchmark.index)).astype(str)
    tbs = benchmark[method.eq("tbs")].copy()
    status = tbs.get("status", pd.Series("", index=tbs.index)).astype(str)
    skip_reason = tbs.get("skip_reason", pd.Series("", index=tbs.index)).astype(str)
    ari = pd.to_numeric(tbs.get("ari", pd.Series(np.nan, index=tbs.index)), errors="coerce")
    ok = status.eq("ok")
    calibration_skips = (
        status.eq("skip")
        & skip_reason.str.contains("calibration|support|inflation", case=False, na=False)
    )
    under_split = pd.to_numeric(
        tbs.get("under_split", pd.Series(0, index=tbs.index)),
        errors="coerce",
    ).fillna(0)
    return {
        "available": True,
        "n_rows": int(len(benchmark)),
        "n_tbs_rows": int(len(tbs)),
        "n_tbs_skips": _status_count(tbs, "status", "skip"),
        "n_tbs_calibration_support_skips": int(calibration_skips.sum()),
        "n_tbs_under_split_rows": int((under_split > 0).sum()),
        "tbs_mean_ari_ok": float(ari[ok].mean()) if ok.any() else math.nan,
    }


def build_candidate_panel(
    *,
    validation_summary: pd.DataFrame,
    geometry_summary: pd.DataFrame,
    guard_panel: pd.DataFrame,
    benchmark_summary: dict[str, Any],
) -> pd.DataFrame:
    """Rank missing equation candidates without promoting production calibration."""
    radius_rows = validation_summary[
        validation_summary["model_id"].eq("radius_angle_action")
    ]
    radius_auc = (
        float(radius_rows["median_auc"].iloc[0]) if not radius_rows.empty else math.nan
    )
    radius_evidence = math.isfinite(radius_auc) and radius_auc >= 0.85
    median_angle = _finite_median(geometry_summary["angle_to_leading_axis_deg_median"])
    median_independent = _finite_median(geometry_summary["independent_fraction_median"])
    support_skips = int(benchmark_summary.get("n_tbs_calibration_support_skips", 0))
    under_splits = int(benchmark_summary.get("n_tbs_under_split_rows", 0))
    guard_candidates = (
        guard_panel["recursive_decision"].eq("candidate_guard_validation_panel").sum()
        if "recursive_decision" in guard_panel.columns
        else 0
    )

    rows = [
        {
            "candidate_path": "barycentric_edge_sibling_identity",
            "evidence_role": "exact_identity",
            "recursive_decision": "keep_diagnostic",
            "priority": 1,
            "evidence_summary": (
                "Exact local algebra couples child-parent edge and sibling "
                "directions in the default no-branch-scaling path."
            ),
        },
        {
            "candidate_path": "angular_shell_kak_radius_angle_action",
            "evidence_role": (
                "diagnostic_association" if radius_evidence else "validation_required"
            ),
            "recursive_decision": (
                "promote_to_validation_panel" if radius_evidence else "keep_diagnostic"
            ),
            "priority": 2 if radius_evidence else 4,
            "evidence_summary": (
                f"radius_angle_action median AUC={radius_auc:.6g}; "
                f"median block angle={median_angle:.6g}; "
                f"median independent fraction={median_independent:.6g}."
            ),
        },
        {
            "candidate_path": "action_budget_parallel_axis",
            "evidence_role": (
                "diagnostic_association" if guard_candidates else "validation_required"
            ),
            "recursive_decision": "promote_to_validation_panel",
            "priority": 3,
            "evidence_summary": (
                "Parallel-axis split action is the probable missing equation for "
                "testing split energy against remaining parent inertia. "
                f"Candidate high-action angular-shell guard rows={int(guard_candidates)}."
            ),
        },
        {
            "candidate_path": "external_selected_tail_support",
            "evidence_role": "support_gap" if support_skips else "validation_required",
            "recursive_decision": "needs_new_data" if support_skips else "keep_diagnostic",
            "priority": 4 if support_skips else 6,
            "evidence_summary": (
                f"TBS calibration-support skip count={support_skips}; external selected-tail "
                "calibration remains fail-closed without support validation."
            ),
        },
        {
            "candidate_path": "traversal_survival_path",
            "evidence_role": "diagnostic_association",
            "recursive_decision": "promote_to_validation_panel",
            "priority": 5,
            "evidence_summary": (
                "Traversal must be modeled as a reached-and-split path event, "
                f"with observed TBS under-split rows={under_splits}."
            ),
        },
        {
            "candidate_path": "selected_region_tangent_cone",
            "evidence_role": "validation_required",
            "recursive_decision": "keep_diagnostic",
            "priority": 6,
            "evidence_summary": (
                "Merge inequalities, discrete tie cells, and tangent cones remain "
                "separate selected-region geometry objects."
            ),
        },
        {
            "candidate_path": "projection_selection_basis_law",
            "evidence_role": "validation_required",
            "recursive_decision": "keep_diagnostic",
            "priority": 7,
            "evidence_summary": (
                "Fixed chi-square references require validated selected-basis "
                "conditions; raw MP or selected PCA rules are not production laws."
            ),
        },
    ]
    return pd.DataFrame.from_records(rows).sort_values("priority").reset_index(drop=True)


def _write_report(
    *,
    output_dir: Path,
    validation_summary: pd.DataFrame,
    geometry_summary: pd.DataFrame,
    guard_panel: pd.DataFrame,
    guard_utility: pd.DataFrame,
    benchmark_summary: dict[str, Any],
    candidate_panel: pd.DataFrame,
) -> None:
    radius_rows = validation_summary[
        validation_summary["model_id"].eq("radius_angle_action")
    ]
    radius_auc = (
        float(radius_rows["median_auc"].iloc[0]) if not radius_rows.empty else math.nan
    )
    top_candidate = str(candidate_panel.iloc[0]["candidate_path"])
    candidate_guards = guard_panel[
        guard_panel["recursive_decision"].eq("candidate_guard_validation_panel")
    ]
    best_guard = (
        candidate_guards.sort_values(
            ["diagnostic_precision_for_pure_fragment", "mixed_context_flag_rate"],
            ascending=[False, True],
        ).iloc[0]
        if not candidate_guards.empty
        else None
    )
    lines = [
        "# Path-Conditioned Barycentric Action Diagnostic",
        "",
        f"- schema_version: `{SCHEMA_VERSION}`",
        f"- study_role: `{STUDY_ROLE}`",
        f"- n_kak_blocks: `{len(geometry_summary)}`",
        f"- radius_angle_action_median_auc: `{radius_auc:.6f}`"
        if math.isfinite(radius_auc)
        else "- radius_angle_action_median_auc: `nan`",
        f"- benchmark_available: `{bool(benchmark_summary.get('available'))}`",
        f"- top_candidate_path: `{top_candidate}`",
        "",
        "## Interpretation",
        "",
        (
            "The cached KAK radius/angle/action signal is traversal diagnostic "
            "evidence, not calibration evidence. It should be used to design a "
            "mixed null/signal validation panel, not as a production sibling "
            "inflation fallback."
        ),
        "",
        "## Equation Trace",
        "",
        "- `theta_u = beta theta_L + (1 - beta) theta_R`",
        "- `theta_L - theta_u = (1 - beta)(theta_L - theta_R)`",
        "- `theta_R - theta_u = -beta(theta_L - theta_R)`",
        "- `T_u` must be read under the selected path: tree, edge opening, projection, traversal.",
        "- `split_action_B = n_u beta(1 - beta) ||P_B(theta_L - theta_R)||^2`",
        "- `action_fraction_B = split_action_B / parent_inertia_B`",
        "",
        "## Candidate Paths",
        "",
    ]
    for row in candidate_panel.itertuples(index=False):
        lines.append(
            f"- `{row.candidate_path}`: `{row.evidence_role}`, "
            f"`{row.recursive_decision}`. {row.evidence_summary}"
        )
    lines.extend(
        [
            "",
            "## Action-Budget Guard Panel",
            "",
            (
                "The guard panel is diagnostic-only. It uses post-hoc final "
                "assignment labels to separate pure-fragment contexts from mixed "
                "contexts, then evaluates high-action angular-shell risk flags."
            ),
            "",
        ]
    )
    if best_guard is not None:
        lines.extend(
            [
                f"- best_candidate_guard: `{best_guard.guard_id}`",
                (
                    "- diagnostic_precision_for_pure_fragment: "
                    f"`{best_guard.diagnostic_precision_for_pure_fragment:.6f}`"
                ),
                f"- mixed_context_flag_rate: `{best_guard.mixed_context_flag_rate:.6f}`",
                f"- pure_fragment_flag_rate: `{best_guard.pure_fragment_flag_rate:.6f}`",
                f"- label_provenance: `{LABEL_PROVENANCE}`",
                "",
            ]
        )
    else:
        lines.extend(["- best_candidate_guard: `none`", ""])
    if not guard_utility.empty:
        best_cost_one = guard_utility[
            guard_utility["mixed_context_cost_ratio"].eq(1.0)
        ].sort_values("net_utility", ascending=False)
        best_cost_two = guard_utility[
            guard_utility["mixed_context_cost_ratio"].eq(2.0)
        ].sort_values("net_utility", ascending=False)
        if not best_cost_one.empty:
            row = best_cost_one.iloc[0]
            lines.extend(
                [
                    "## Guard Utility",
                    "",
                    (
                        "Utility assumes each prevented pure fragment is worth `+1` "
                        "and each delayed mixed context costs the listed ratio."
                    ),
                    "",
                    f"- best_guard_cost_1: `{row.guard_id}`",
                    f"- net_utility_cost_1: `{row.net_utility:.6f}`",
                    f"- net_utility_per_row_cost_1: `{row.net_utility_per_row:.6f}`",
                    (
                        "- break_even_mixed_context_cost_ratio_cost_1_guard: "
                        f"`{row.break_even_mixed_context_cost_ratio:.6f}`"
                    ),
                ]
            )
        if not best_cost_two.empty:
            row = best_cost_two.iloc[0]
            lines.extend(
                [
                    f"- best_guard_cost_2: `{row.guard_id}`",
                    f"- net_utility_cost_2: `{row.net_utility:.6f}`",
                    f"- net_utility_per_row_cost_2: `{row.net_utility_per_row:.6f}`",
                    "",
                ]
            )
    lines.extend(
        [
            "## Benchmark Evidence",
            "",
            f"- TBS rows: `{benchmark_summary.get('n_tbs_rows', 0)}`",
            f"- TBS skips: `{benchmark_summary.get('n_tbs_skips', 0)}`",
            (
                "- TBS calibration-support skips: "
                f"`{benchmark_summary.get('n_tbs_calibration_support_skips', 0)}`"
            ),
            f"- TBS under-split rows: `{benchmark_summary.get('n_tbs_under_split_rows', 0)}`",
            "",
        ]
    )
    output_dir.joinpath("path_conditioned_barycentric_action_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def run_path_conditioned_barycentric_action_diagnostic(
    *,
    output_dir: Path,
    kak_internal_geometry_csv: Path,
    kak_fragmentation_validation_csv: Path,
    benchmark_comparison_csv: Path | None = None,
) -> dict[str, Any]:
    """Run cached path-conditioned barycentric action diagnostics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry = pd.read_csv(kak_internal_geometry_csv)
    validation = pd.read_csv(kak_fragmentation_validation_csv)
    benchmark = (
        pd.read_csv(benchmark_comparison_csv)
        if benchmark_comparison_csv is not None and benchmark_comparison_csv.exists()
        else None
    )

    geometry_summary = summarize_kak_geometry(geometry)
    validation_summary = summarize_kak_validation(validation)
    annotations = annotate_kak_radius_angle_action(geometry)
    guard_panel = evaluate_action_budget_guard_panel(annotations)
    guard_utility = evaluate_action_budget_guard_utility(guard_panel)
    benchmark_summary = summarize_benchmark(benchmark)
    candidate_panel = build_candidate_panel(
        validation_summary=validation_summary,
        geometry_summary=geometry_summary,
        guard_panel=guard_panel,
        benchmark_summary=benchmark_summary,
    )

    validation_summary.to_csv(
        output_dir / "kak_radius_angle_action_summary.csv",
        index=False,
    )
    geometry_summary.to_csv(
        output_dir / "kak_internal_geometry_block_summary.csv",
        index=False,
    )
    annotations.to_csv(
        output_dir / "kak_radius_angle_action_annotations.csv",
        index=False,
    )
    guard_panel.to_csv(
        output_dir / "action_budget_guard_panel.csv",
        index=False,
    )
    guard_utility.to_csv(
        output_dir / "action_budget_guard_utility_curve.csv",
        index=False,
    )
    candidate_panel.to_csv(
        output_dir / "missing_equation_candidate_panel.csv",
        index=False,
    )
    _write_report(
        output_dir=output_dir,
        validation_summary=validation_summary,
        geometry_summary=geometry_summary,
        guard_panel=guard_panel,
        guard_utility=guard_utility,
        benchmark_summary=benchmark_summary,
        candidate_panel=candidate_panel,
    )

    radius_rows = validation_summary[
        validation_summary["model_id"].eq("radius_angle_action")
    ]
    radius_auc = (
        float(radius_rows["median_auc"].iloc[0]) if not radius_rows.empty else math.nan
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "n_internal_geometry_rows": int(len(geometry)),
        "n_kak_blocks": int(len(geometry_summary)),
        "n_guard_candidates": int(
            guard_panel["recursive_decision"]
            .eq("candidate_guard_validation_panel")
            .sum()
        ),
        "radius_angle_action_median_auc": radius_auc,
        "benchmark_summary": benchmark_summary,
        "top_candidate_path": str(candidate_panel.iloc[0]["candidate_path"]),
        "best_guard_utility_cost_1": (
            guard_utility[guard_utility["mixed_context_cost_ratio"].eq(1.0)]
            .sort_values("net_utility", ascending=False)
            .head(1)
            .to_dict(orient="records")
        ),
        "best_guard_utility_cost_2": (
            guard_utility[guard_utility["mixed_context_cost_ratio"].eq(2.0)]
            .sort_values("net_utility", ascending=False)
            .head(1)
            .to_dict(orient="records")
        ),
        "outputs": {
            "kak_radius_angle_action_summary": "kak_radius_angle_action_summary.csv",
            "kak_internal_geometry_block_summary": "kak_internal_geometry_block_summary.csv",
            "kak_radius_angle_action_annotations": "kak_radius_angle_action_annotations.csv",
            "action_budget_guard_panel": "action_budget_guard_panel.csv",
            "action_budget_guard_utility_curve": "action_budget_guard_utility_curve.csv",
            "missing_equation_candidate_panel": "missing_equation_candidate_panel.csv",
            "report": "path_conditioned_barycentric_action_report.md",
        },
    }
    (output_dir / "path_conditioned_barycentric_action_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kak-internal-geometry-csv", required=True, type=Path)
    parser.add_argument("--kak-fragmentation-validation-csv", required=True, type=Path)
    parser.add_argument("--benchmark-comparison-csv", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    summary = run_path_conditioned_barycentric_action_diagnostic(
        output_dir=args.output_dir,
        kak_internal_geometry_csv=args.kak_internal_geometry_csv,
        kak_fragmentation_validation_csv=args.kak_fragmentation_validation_csv,
        benchmark_comparison_csv=args.benchmark_comparison_csv,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
