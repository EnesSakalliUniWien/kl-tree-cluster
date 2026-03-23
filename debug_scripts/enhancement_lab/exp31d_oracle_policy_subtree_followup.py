#!/usr/bin/env python3
"""
Experiment 31d: leakage-free oracle policy follow-up.

This is a follow-up to exp31 / exp31b that:
- keeps the original row exports untouched
- derives a predictive equation without multiplying by the oracle label
- evaluates the Gaussian high-gap / zero-edge guard at subtree level
- compares offline suppression vs downranking using the leakage-free score

Inputs:
- exp31 row exports already written to disk

Outputs:
- equation coefficient / CV summary
- subtree-aware guard summaries
- guarded anchor rows
- top-20 false-global rankings under baseline vs downranked policy
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from debug_scripts.enhancement_lab.lab_helpers import build_tree_and_data


_ROOT = Path(__file__).resolve().parents[2]
_FEATURE_COLUMNS = [
    "log_n_parent",
    "log_k",
    "edge_weight",
    "log_branch",
    "depth",
    "stability",
    "gap_log",
]


@dataclass(frozen=True)
class ModeSpec:
    name: str
    rows_csv: str


_DEFAULT_MODES: tuple[ModeSpec, ...] = (
    ModeSpec(
        name="baseline",
        rows_csv="debug_scripts/enhancement_lab/_oracle_policy_rows.csv",
    ),
    ModeSpec(
        name="one_active_1d",
        rows_csv="debug_scripts/enhancement_lab/_oracle_policy_rows_one_active_1d.csv",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--equation-summary-output-csv",
        default="debug_scripts/enhancement_lab/_oracle_policy_followup_equation_summary.csv",
    )
    parser.add_argument(
        "--equation-coefficients-output-csv",
        default="debug_scripts/enhancement_lab/_oracle_policy_followup_equation_coefficients.csv",
    )
    parser.add_argument(
        "--guard-summary-output-csv",
        default="debug_scripts/enhancement_lab/_oracle_policy_followup_guard_summary.csv",
    )
    parser.add_argument(
        "--guard-subfamily-output-csv",
        default="debug_scripts/enhancement_lab/_oracle_policy_followup_guard_subfamilies.csv",
    )
    parser.add_argument(
        "--guard-anchor-output-csv",
        default="debug_scripts/enhancement_lab/_oracle_policy_followup_guard_anchors.csv",
    )
    parser.add_argument(
        "--top20-output-csv",
        default="debug_scripts/enhancement_lab/_oracle_policy_followup_top20.csv",
    )
    parser.add_argument(
        "--summary-markdown",
        default="debug_scripts/enhancement_lab/_oracle_policy_followup_summary.md",
    )
    parser.add_argument("--guard-max-depth", type=int, default=1)
    parser.add_argument("--guard-gap-threshold", type=float, default=4.0)
    parser.add_argument("--guard-edge-eps", type=float, default=1e-9)
    return parser.parse_args()


def _usable_feature_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    kept: list[str] = []
    for column in columns:
        if column not in frame.columns:
            continue
        values = frame[column].dropna().astype(np.float64)
        if values.empty:
            continue
        if float(values.max()) == float(values.min()):
            continue
        kept.append(column)
    return kept


def _prepare_design_matrix(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    kept = _usable_feature_columns(frame, columns)
    predictors = frame[kept].copy()
    return sm.add_constant(predictors, has_constant="add")


def _fit_statsmodels_logit(
    frame: pd.DataFrame,
    *,
    target_column: str,
) -> sm.discrete.discrete_model.BinaryResultsWrapper | None:
    focal = frame.loc[~frame["is_null_like"]].copy()
    positives = int(focal[target_column].sum())
    negatives = int((~focal[target_column]).sum())
    if positives == 0 or negatives == 0:
        return None
    predictors = _prepare_design_matrix(focal, _FEATURE_COLUMNS)
    target = focal[target_column].astype(int)
    model = sm.Logit(target, predictors)
    try:
        return model.fit(disp=False)
    except Exception:
        return model.fit_regularized(disp=False, alpha=1e-4)


def _grouped_logit_cv_summary(
    frame: pd.DataFrame,
    *,
    target_column: str,
    group_column: str,
) -> dict[str, float | int]:
    focal = frame.loc[~frame["is_null_like"]].copy()
    positives = int(focal[target_column].sum())
    negatives = int((~focal[target_column]).sum())
    if positives == 0 or negatives == 0 or focal[group_column].nunique() < 2:
        return {
            "n_focal": int(len(focal)),
            "n_positive": positives,
            "n_negative": negatives,
            "mean_balanced_accuracy": math.nan,
            "mean_roc_auc": math.nan,
        }

    feature_names = _usable_feature_columns(focal, _FEATURE_COLUMNS)
    if not feature_names:
        return {
            "n_focal": int(len(focal)),
            "n_positive": positives,
            "n_negative": negatives,
            "mean_balanced_accuracy": math.nan,
            "mean_roc_auc": math.nan,
        }

    x = focal[feature_names].to_numpy(dtype=np.float64)
    y = focal[target_column].astype(int).to_numpy(dtype=np.int64)
    groups = focal[group_column].to_numpy()

    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    ba_scores: list[float] = []
    roc_scores: list[float] = []
    for train_idx, test_idx in splitter.split(x, y, groups):
        x_train = x[train_idx]
        y_train = y[train_idx]
        x_test = x[test_idx]
        y_test = y[test_idx]
        if len(np.unique(y_train)) < 2:
            continue
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "logit",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )
        model.fit(x_train, y_train)
        prob = model.predict_proba(x_test)[:, 1]
        pred = (prob >= 0.5).astype(np.int64)
        ba_scores.append(float(balanced_accuracy_score(y_test, pred)))
        if len(np.unique(y_test)) > 1:
            roc_scores.append(float(roc_auc_score(y_test, prob)))

    return {
        "n_focal": int(len(focal)),
        "n_positive": positives,
        "n_negative": negatives,
        "mean_balanced_accuracy": float(np.mean(ba_scores)) if ba_scores else math.nan,
        "mean_roc_auc": float(np.mean(roc_scores)) if roc_scores else math.nan,
    }


def _predictive_score(frame: pd.DataFrame) -> pd.Series:
    base_probability = _heldout_probability(frame)
    return base_probability * (1.0 + frame["gap_log"].to_numpy(dtype=np.float64))


def _heldout_probability(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["gaussian_subfamily_probability"]
        .fillna(frame["family_portability_probability"])
        .fillna(0.0)
        .astype(np.float64)
    )


def _heldout_probability_source(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.select(
            [
                frame["gaussian_subfamily_probability"].notna(),
                frame["family_portability_probability"].notna(),
            ],
            [
                "gaussian_subfamily_probability",
                "family_portability_probability",
            ],
            default="none",
        ),
        index=frame.index,
        dtype="object",
    )


def _assign_driver_type(
    frame: pd.DataFrame, *, edge_eps: float, gap_threshold: float, max_depth: int
) -> pd.Series:
    return pd.Series(
        np.select(
            [
                (frame["depth"] <= max_depth)
                & (frame["gap_log"] >= gap_threshold)
                & (frame["edge_weight"] <= edge_eps)
                & (frame["k"] == 1),
                (frame["depth"] <= max_depth)
                & (frame["gap_log"] >= gap_threshold)
                & (frame["edge_weight"] <= edge_eps),
                (frame["depth"] <= max_depth) & (frame["gap_log"] >= gap_threshold),
                (frame["depth"] >= 3) & (frame["n_parent"] <= 4),
                frame["edge_weight"] <= edge_eps,
                frame["k"] == 1,
            ],
            [
                "early_high_gap_zero_edge_k1",
                "early_high_gap_zero_edge",
                "early_high_gap",
                "tiny_deep_node",
                "zero_edge_weight",
                "k1_pair",
            ],
            default="mid_tree_gap",
        ),
        index=frame.index,
    )


def _rank_false_global_rows(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.loc[frame["oracle_prefers_global_bh"]].copy()
    if ranked.empty:
        return ranked
    ranked = ranked.assign(
        heldout_probability=lambda df: _heldout_probability(df),
        heldout_probability_source=lambda df: _heldout_probability_source(df),
        predictive_rank=lambda df: df["predictive_score"]
        .rank(method="first", ascending=False)
        .astype(np.int64),
    )
    return ranked.sort_values(
        ["predictive_score", "heldout_probability", "p_global"],
        ascending=[False, False, True],
        kind="stable",
    )


def _rank_false_global_rows_downranked(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.loc[frame["oracle_prefers_global_bh"]].copy()
    if ranked.empty:
        return ranked
    ranked = ranked.assign(
        heldout_probability=lambda df: _heldout_probability(df),
        heldout_probability_source=lambda df: _heldout_probability_source(df),
        predictive_rank=lambda df: df["predictive_score"]
        .rank(method="first", ascending=False)
        .astype(np.int64),
    )
    return ranked.sort_values(
        [
            "in_guarded_subtree",
            "predictive_score",
            "heldout_probability",
            "p_global",
        ],
        ascending=[True, False, False, True],
        kind="stable",
    )


def _iter_case_subtree_nodes(case_name: str, anchor_nodes: Iterable[str]) -> set[str]:
    tree, _data_df, _y_true, _tc = build_tree_and_data(case_name)
    covered: set[str] = set()
    for node in anchor_nodes:
        covered.add(str(node))
        covered.update(str(child) for child in nx.descendants(tree, node))
    return covered


def _collect_guard_anchors(
    frame: pd.DataFrame,
    *,
    mode_name: str,
    max_depth: int,
    gap_threshold: float,
    edge_eps: float,
) -> tuple[pd.DataFrame, set[tuple[str, str]]]:
    hits = frame.loc[frame["guard_hit"]].copy()
    if hits.empty:
        return pd.DataFrame(), set()

    covered_keys: set[tuple[str, str]] = set()
    rows: list[dict[str, object]] = []
    for case_name, case_hits in hits.groupby("case_name", sort=True):
        anchors = [str(node) for node in case_hits["parent"].tolist()]
        covered_nodes = _iter_case_subtree_nodes(str(case_name), anchors)
        case_rows = frame.loc[frame["case_name"] == case_name].copy().set_index("parent", drop=False)
        covered_case = case_rows.loc[case_rows.index.intersection(covered_nodes)].copy()
        for parent, row in case_hits.set_index("parent", drop=False).iterrows():
            subtree_nodes = _iter_case_subtree_nodes(str(case_name), [str(parent)])
            subtree_rows = case_rows.loc[case_rows.index.intersection(subtree_nodes)].copy()
            rows.append(
                {
                    "mode": mode_name,
                    "case_name": str(case_name),
                    "case_subfamily": str(row["case_subfamily"]),
                    "guard_parent": str(parent),
                    "guard_depth": int(row["depth"]),
                    "guard_n_parent": int(row["n_parent"]),
                    "guard_k": int(row["k"]),
                    "guard_gap_log": float(row["gap_log"]),
                    "guard_edge_weight": float(row["edge_weight"]),
                    "guard_false_global": bool(row["oracle_prefers_global_bh"]),
                    "guard_predictive_score": float(row["predictive_score"]),
                    "subtree_rows": int(len(subtree_rows)),
                    "subtree_false_global": int(subtree_rows["oracle_prefers_global_bh"].sum()),
                    "subtree_predictive_mass": float(subtree_rows["predictive_score"].sum()),
                    "covered_case_rows": int(len(covered_case)),
                    "covered_case_false_global": int(covered_case["oracle_prefers_global_bh"].sum()),
                    "covered_case_predictive_mass": float(covered_case["predictive_score"].sum()),
                    "guard_definition": (
                        f"depth<={max_depth} & gap>={gap_threshold:g} & edge<={edge_eps:.1e} & k=1"
                    ),
                }
            )
        covered_keys.update((str(case_name), str(node)) for node in covered_nodes)
    return pd.DataFrame(rows), covered_keys


def _with_top20_flag(frame: pd.DataFrame, ranked_false_global: pd.DataFrame) -> pd.DataFrame:
    marked = frame.copy()
    marked["in_top20_false_global"] = False
    if ranked_false_global.empty:
        return marked
    top = ranked_false_global.head(20).copy()
    keys = set(zip(top["case_name"].astype(str), top["parent"].astype(str)))
    marked["in_top20_false_global"] = [
        (str(case_name), str(parent)) in keys
        for case_name, parent in zip(marked["case_name"], marked["parent"])
    ]
    return marked


def _summary_row(frame: pd.DataFrame, *, mode_name: str, policy_name: str) -> dict[str, object]:
    gaussian = frame.loc[(frame["case_family"] == "gaussian") & (~frame["is_null_like"])].copy()
    extreme = gaussian.loc[gaussian["case_subfamily"] == "gaussian_extreme_noise"].copy()
    other = gaussian.loc[gaussian["case_subfamily"] != "gaussian_extreme_noise"].copy()
    top20_false_global = frame.loc[frame["in_top20_false_global"] & frame["oracle_prefers_global_bh"]].copy()
    return {
        "mode": mode_name,
        "policy": policy_name,
        "gaussian_focal_rows": int(len(gaussian)),
        "gaussian_false_global": int(gaussian["oracle_prefers_global_bh"].sum()),
        "gaussian_predictive_mass": float(gaussian["predictive_score"].sum()),
        "extreme_noise_focal_rows": int(len(extreme)),
        "extreme_noise_false_global": int(extreme["oracle_prefers_global_bh"].sum()),
        "extreme_noise_predictive_mass": float(extreme["predictive_score"].sum()),
        "other_gaussian_focal_rows": int(len(other)),
        "other_gaussian_false_global": int(other["oracle_prefers_global_bh"].sum()),
        "other_gaussian_predictive_mass": float(other["predictive_score"].sum()),
        "top20_false_global_rows": int(len(top20_false_global)),
        "top20_extreme_noise_rows": int((top20_false_global["case_subfamily"] == "gaussian_extreme_noise").sum()),
        "top20_other_gaussian_rows": int(
            ((top20_false_global["case_family"] == "gaussian") & (top20_false_global["case_subfamily"] != "gaussian_extreme_noise")).sum()
        ),
    }


def _subfamily_summary(frame: pd.DataFrame, *, mode_name: str, policy_name: str) -> pd.DataFrame:
    gaussian = frame.loc[(frame["case_family"] == "gaussian") & (~frame["is_null_like"])].copy()
    if gaussian.empty:
        return pd.DataFrame()
    summary = (
        gaussian.groupby("case_subfamily", as_index=False)
        .agg(
            cases=("case_name", "nunique"),
            rows=("case_name", "size"),
            false_global=("oracle_prefers_global_bh", "sum"),
            predictive_mass=("predictive_score", "sum"),
            top20_rows=("in_top20_false_global", "sum"),
        )
        .sort_values(["false_global", "predictive_mass", "case_subfamily"], ascending=[False, False, True], kind="stable")
        .reset_index(drop=True)
    )
    summary.insert(0, "policy", policy_name)
    summary.insert(0, "mode", mode_name)
    return summary


def _equation_outputs(frame: pd.DataFrame, *, mode_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_column = "oracle_prefers_global_bh"
    model = _fit_statsmodels_logit(frame, target_column=target_column)
    cv = _grouped_logit_cv_summary(frame, target_column=target_column, group_column="case_name")

    summary = pd.DataFrame(
        [
            {
                "mode": mode_name,
                "target_column": target_column,
                **cv,
            }
        ]
    )

    coefficient_rows: list[dict[str, object]] = []
    if model is not None:
        for name, value in model.params.items():
            coefficient_rows.append(
                {
                    "mode": mode_name,
                    "target_column": target_column,
                    "term": str(name),
                    "coefficient": float(value),
                }
            )
    coefficients = pd.DataFrame(coefficient_rows)
    return summary, coefficients


def _analyze_mode(
    mode: ModeSpec,
    *,
    max_depth: int,
    gap_threshold: float,
    edge_eps: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv((_ROOT / mode.rows_csv).resolve()).copy()
    frame["heldout_probability"] = _heldout_probability(frame)
    frame["heldout_probability_source"] = _heldout_probability_source(frame)
    frame["predictive_score"] = _predictive_score(frame)
    frame["driver_type_v2"] = _assign_driver_type(
        frame,
        edge_eps=edge_eps,
        gap_threshold=gap_threshold,
        max_depth=max_depth,
    )
    frame["guard_hit"] = (
        (frame["case_family"] == "gaussian")
        & (~frame["is_null_like"])
        & (frame["depth"] <= max_depth)
        & (frame["gap_log"] >= gap_threshold)
        & (frame["edge_weight"] <= edge_eps)
        & (frame["k"] == 1)
    )

    equation_summary, equation_coefficients = _equation_outputs(frame, mode_name=mode.name)
    anchors, covered_keys = _collect_guard_anchors(
        frame,
        mode_name=mode.name,
        max_depth=max_depth,
        gap_threshold=gap_threshold,
        edge_eps=edge_eps,
    )
    frame["in_guarded_subtree"] = [
        (str(case_name), str(parent)) in covered_keys
        for case_name, parent in zip(frame["case_name"], frame["parent"])
    ]

    baseline_ranked = _rank_false_global_rows(frame)
    baseline_marked = _with_top20_flag(frame, baseline_ranked)

    downrank_ranked = _rank_false_global_rows_downranked(frame)
    downrank_marked = _with_top20_flag(frame, downrank_ranked)

    suppressed = frame.loc[~frame["in_guarded_subtree"]].copy()
    suppress_ranked = _rank_false_global_rows(suppressed)
    suppress_marked = _with_top20_flag(suppressed, suppress_ranked)

    guard_summary = pd.DataFrame(
        [
            _summary_row(baseline_marked, mode_name=mode.name, policy_name="baseline"),
            _summary_row(downrank_marked, mode_name=mode.name, policy_name="downrank_subtree"),
            _summary_row(suppress_marked, mode_name=mode.name, policy_name="suppress_subtree"),
        ]
    )
    subfamilies = pd.concat(
        [
            _subfamily_summary(baseline_marked, mode_name=mode.name, policy_name="baseline"),
            _subfamily_summary(downrank_marked, mode_name=mode.name, policy_name="downrank_subtree"),
            _subfamily_summary(suppress_marked, mode_name=mode.name, policy_name="suppress_subtree"),
        ],
        ignore_index=True,
    )

    top20 = pd.concat(
        [
            baseline_ranked.head(20).assign(mode=mode.name, policy="baseline"),
            downrank_ranked.head(20).assign(mode=mode.name, policy="downrank_subtree"),
            suppress_ranked.head(20).assign(mode=mode.name, policy="suppress_subtree"),
        ],
        ignore_index=True,
    )
    return equation_summary, equation_coefficients, guard_summary, subfamilies, anchors, top20


def _write_markdown(
    equation_summary: pd.DataFrame,
    equation_coefficients: pd.DataFrame,
    guard_summary: pd.DataFrame,
    anchors: pd.DataFrame,
    output_path: Path,
) -> None:
    lines: list[str] = [
        "# Experiment 31d: Leakage-Free Oracle Policy Follow-Up",
        "",
    ]

    lines.extend(["## Equation Summary", ""])
    with pd.option_context("display.max_columns", None, "display.width", 180):
        lines.append(equation_summary.to_string(index=False))
    lines.append("")

    lines.extend(["## Coefficients", ""])
    with pd.option_context("display.max_columns", None, "display.width", 180):
        lines.append(equation_coefficients.to_string(index=False))
    lines.append("")

    lines.extend(["## Guard Summary", ""])
    with pd.option_context("display.max_columns", None, "display.width", 180):
        lines.append(guard_summary.to_string(index=False))
    lines.append("")

    if not anchors.empty:
        lines.extend(["## Guard Anchors", ""])
        with pd.option_context("display.max_columns", None, "display.width", 180):
            lines.append(anchors.to_string(index=False))
        lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    equation_summaries: list[pd.DataFrame] = []
    equation_coefficients: list[pd.DataFrame] = []
    guard_summaries: list[pd.DataFrame] = []
    guard_subfamilies: list[pd.DataFrame] = []
    guard_anchors: list[pd.DataFrame] = []
    top20_frames: list[pd.DataFrame] = []

    for mode in _DEFAULT_MODES:
        (
            equation_summary,
            equation_coeff,
            guard_summary,
            subfamily_summary,
            anchors,
            top20,
        ) = _analyze_mode(
            mode,
            max_depth=args.guard_max_depth,
            gap_threshold=args.guard_gap_threshold,
            edge_eps=args.guard_edge_eps,
        )
        equation_summaries.append(equation_summary)
        equation_coefficients.append(equation_coeff)
        guard_summaries.append(guard_summary)
        guard_subfamilies.append(subfamily_summary)
        guard_anchors.append(anchors)
        top20_frames.append(top20)

    equation_summary_df = pd.concat(equation_summaries, ignore_index=True)
    equation_coeff_df = pd.concat(equation_coefficients, ignore_index=True)
    guard_summary_df = pd.concat(guard_summaries, ignore_index=True)
    guard_subfamily_df = pd.concat(guard_subfamilies, ignore_index=True)
    guard_anchor_df = pd.concat(guard_anchors, ignore_index=True)
    top20_df = pd.concat(top20_frames, ignore_index=True)

    equation_summary_path = (_ROOT / args.equation_summary_output_csv).resolve()
    equation_coeff_path = (_ROOT / args.equation_coefficients_output_csv).resolve()
    guard_summary_path = (_ROOT / args.guard_summary_output_csv).resolve()
    guard_subfamily_path = (_ROOT / args.guard_subfamily_output_csv).resolve()
    guard_anchor_path = (_ROOT / args.guard_anchor_output_csv).resolve()
    top20_path = (_ROOT / args.top20_output_csv).resolve()
    markdown_path = (_ROOT / args.summary_markdown).resolve()

    equation_summary_df.to_csv(equation_summary_path, index=False)
    equation_coeff_df.to_csv(equation_coeff_path, index=False)
    guard_summary_df.to_csv(guard_summary_path, index=False)
    guard_subfamily_df.to_csv(guard_subfamily_path, index=False)
    guard_anchor_df.to_csv(guard_anchor_path, index=False)
    top20_df.to_csv(top20_path, index=False)
    _write_markdown(
        equation_summary_df,
        equation_coeff_df,
        guard_summary_df,
        guard_anchor_df,
        markdown_path,
    )

    print(f"Equation summary CSV: {equation_summary_path}")
    print(f"Equation coefficients CSV: {equation_coeff_path}")
    print(f"Guard summary CSV: {guard_summary_path}")
    print(f"Guard subfamily CSV: {guard_subfamily_path}")
    print(f"Guard anchor CSV: {guard_anchor_path}")
    print(f"Top20 CSV: {top20_path}")
    print(f"Summary markdown: {markdown_path}")


if __name__ == "__main__":
    main()
