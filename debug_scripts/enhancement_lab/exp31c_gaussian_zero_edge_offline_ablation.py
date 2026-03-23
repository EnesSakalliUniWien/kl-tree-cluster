#!/usr/bin/env python3
"""
Experiment 31c: Offline suppression/downranking of early gaussian zero-edge splits.

Purpose:
- identify gaussian focal rows that look like early high-gap / zero-edge anchors
- estimate how much downstream false-global burden sits under those anchors
- compare two offline policies:
  1. suppress the anchor subtree(s) entirely
  2. down-rank those subtree rows in the false-global ranking without removing them

This does not modify the live clustering path. It operates only on exported exp31 rows.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd

from debug_scripts.enhancement_lab.lab_helpers import build_tree_and_data


_ROOT = Path(__file__).resolve().parents[2]


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline ablation of early gaussian high-gap/zero-edge anchors from exp31 rows."
    )
    parser.add_argument(
        "--summary-output-csv",
        default="debug_scripts/enhancement_lab/_oracle_policy_zero_edge_ablation_summary.csv",
    )
    parser.add_argument(
        "--subfamily-output-csv",
        default="debug_scripts/enhancement_lab/_oracle_policy_zero_edge_ablation_subfamilies.csv",
    )
    parser.add_argument(
        "--anchor-output-csv",
        default="debug_scripts/enhancement_lab/_oracle_policy_zero_edge_ablation_anchors.csv",
    )
    parser.add_argument(
        "--top20-output-csv",
        default="debug_scripts/enhancement_lab/_oracle_policy_zero_edge_ablation_top20.csv",
    )
    parser.add_argument(
        "--summary-markdown",
        default="debug_scripts/enhancement_lab/_oracle_policy_zero_edge_ablation_summary.md",
    )
    return parser.parse_args()


def _compute_risk_score(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["gaussian_subfamily_probability"]
        .fillna(frame["family_portability_probability"])
        .fillna(frame["statsmodels_fitted_probability"])
        .fillna(0.0)
        * (1.0 + frame["gap_log"].to_numpy(dtype=np.float64))
        * (1.0 + frame["oracle_prefers_global_bh"].astype(np.float64))
    )


def _target_mask(frame: pd.DataFrame, *, max_depth: int) -> pd.Series:
    return (
        (frame["case_family"] == "gaussian")
        & (~frame["is_null_like"])
        & (frame["depth"] <= max_depth)
        & (frame["gap_log"] >= 4.0)
        & (frame["edge_weight"] <= 1e-9)
    )


def _rank_false_global_rows(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.loc[frame["oracle_prefers_global_bh"]].copy()
    if ranked.empty:
        return ranked
    ranked = ranked.assign(
        family_probability_rank=lambda df: df["family_portability_probability"]
        .rank(method="first", ascending=False)
        .astype(np.int64),
        probability_gap=lambda df: df["family_portability_probability"]
        - df["statsmodels_fitted_probability"],
    )
    return ranked.sort_values(
        ["family_portability_probability", "statsmodels_fitted_probability", "p_global"],
        ascending=[False, False, True],
        kind="stable",
    )


def _rank_false_global_rows_downranked(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.loc[frame["oracle_prefers_global_bh"]].copy()
    if ranked.empty:
        return ranked
    ranked = ranked.assign(
        family_probability_rank=lambda df: df["family_portability_probability"]
        .rank(method="first", ascending=False)
        .astype(np.int64),
        probability_gap=lambda df: df["family_portability_probability"]
        - df["statsmodels_fitted_probability"],
    )
    return ranked.sort_values(
        [
            "in_targeted_subtree",
            "family_portability_probability",
            "statsmodels_fitted_probability",
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


def _collect_anchor_table(frame: pd.DataFrame, *, mode_name: str, scope_name: str) -> tuple[pd.DataFrame, set[tuple[str, str]]]:
    target_rows = frame.loc[frame["is_target_anchor"]].copy()
    if target_rows.empty:
        return pd.DataFrame(), set()

    covered_keys: set[tuple[str, str]] = set()
    anchor_rows: list[dict[str, object]] = []
    for case_name, case_targets in target_rows.groupby("case_name", sort=True):
        anchors = [str(node) for node in case_targets["parent"].tolist()]
        covered_nodes = _iter_case_subtree_nodes(str(case_name), anchors)
        case_rows = frame.loc[frame["case_name"] == case_name].copy()
        case_rows = case_rows.set_index("parent", drop=False)
        covered_case = case_rows.loc[case_rows.index.intersection(covered_nodes)].copy()
        for parent, row in case_targets.set_index("parent", drop=False).iterrows():
            subtree_nodes = _iter_case_subtree_nodes(str(case_name), [str(parent)])
            subtree_rows = case_rows.loc[case_rows.index.intersection(subtree_nodes)].copy()
            anchor_rows.append(
                {
                    "mode": mode_name,
                    "scope": scope_name,
                    "case_name": str(case_name),
                    "case_subfamily": str(row["case_subfamily"]),
                    "anchor_parent": str(parent),
                    "anchor_depth": int(row["depth"]),
                    "anchor_n_parent": int(row["n_parent"]),
                    "anchor_gap_log": float(row["gap_log"]),
                    "anchor_edge_weight": float(row["edge_weight"]),
                    "anchor_false_global": bool(row["oracle_prefers_global_bh"]),
                    "subtree_rows": int(len(subtree_rows)),
                    "subtree_false_global": int(subtree_rows["oracle_prefers_global_bh"].sum()),
                    "subtree_risk_sum": float(subtree_rows["risk_score"].sum()),
                    "covered_case_rows": int(len(covered_case)),
                    "covered_case_false_global": int(covered_case["oracle_prefers_global_bh"].sum()),
                    "covered_case_risk_sum": float(covered_case["risk_score"].sum()),
                }
            )
        covered_keys.update((str(case_name), str(node)) for node in covered_nodes)

    return pd.DataFrame(anchor_rows), covered_keys


def _subfamily_summary(frame: pd.DataFrame, *, mode_name: str, policy_name: str, scope_name: str) -> pd.DataFrame:
    gaussian = frame.loc[(frame["case_family"] == "gaussian") & (~frame["is_null_like"])].copy()
    if gaussian.empty:
        return pd.DataFrame()
    return (
        gaussian.groupby("case_subfamily", as_index=False)
        .agg(
            mode=("case_name", lambda _s: mode_name),
            policy=("case_name", lambda _s: policy_name),
            scope=("case_name", lambda _s: scope_name),
            cases=("case_name", "nunique"),
            rows=("case_name", "size"),
            false_global=("oracle_prefers_global_bh", "sum"),
            risk_sum=("risk_score", "sum"),
            top20_rows=("in_top20_false_global", "sum"),
        )
        .sort_values(["false_global", "risk_sum", "case_subfamily"], ascending=[False, False, True], kind="stable")
        .reset_index(drop=True)
    )


def _overall_summary(frame: pd.DataFrame, *, mode_name: str, policy_name: str, scope_name: str) -> dict[str, object]:
    gaussian = frame.loc[(frame["case_family"] == "gaussian") & (~frame["is_null_like"])].copy()
    extreme = gaussian.loc[gaussian["case_subfamily"] == "gaussian_extreme_noise"].copy()
    other = gaussian.loc[gaussian["case_subfamily"] != "gaussian_extreme_noise"].copy()
    false_global_top20 = frame.loc[frame["in_top20_false_global"] & frame["oracle_prefers_global_bh"]].copy()
    return {
        "mode": mode_name,
        "policy": policy_name,
        "scope": scope_name,
        "gaussian_focal_rows": int(len(gaussian)),
        "gaussian_false_global": int(gaussian["oracle_prefers_global_bh"].sum()),
        "gaussian_risk_sum": float(gaussian["risk_score"].sum()),
        "extreme_noise_focal_rows": int(len(extreme)),
        "extreme_noise_false_global": int(extreme["oracle_prefers_global_bh"].sum()),
        "extreme_noise_risk_sum": float(extreme["risk_score"].sum()),
        "other_gaussian_focal_rows": int(len(other)),
        "other_gaussian_false_global": int(other["oracle_prefers_global_bh"].sum()),
        "other_gaussian_risk_sum": float(other["risk_score"].sum()),
        "top20_false_global_rows": int(len(false_global_top20)),
        "top20_extreme_noise_rows": int((false_global_top20["case_subfamily"] == "gaussian_extreme_noise").sum()),
        "top20_other_gaussian_rows": int(
            ((false_global_top20["case_family"] == "gaussian") & (false_global_top20["case_subfamily"] != "gaussian_extreme_noise")).sum()
        ),
    }


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


def _write_markdown(summary: pd.DataFrame, anchors: pd.DataFrame, output_path: Path) -> None:
    lines: list[str] = [
        "# Experiment 31c: Gaussian Zero-Edge Offline Ablation",
        "",
    ]
    if summary.empty:
        lines.append("No results were produced.")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    for (mode_name, scope_name), block in summary.groupby(["mode", "scope"], sort=False):
        lines.extend([f"## {mode_name} / {scope_name}", ""])
        with pd.option_context("display.max_columns", None, "display.width", 180):
            lines.append(block.to_string(index=False))
        lines.append("")
        anchor_block = anchors.loc[(anchors["mode"] == mode_name) & (anchors["scope"] == scope_name)].copy()
        if not anchor_block.empty:
            lines.append("Anchors:")
            with pd.option_context("display.max_columns", None, "display.width", 180):
                lines.append(anchor_block.to_string(index=False))
            lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _analyze_mode(mode: ModeSpec, *, scope_name: str, max_depth: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv((_ROOT / mode.rows_csv).resolve()).copy()
    frame["risk_score"] = _compute_risk_score(frame)
    frame["is_target_anchor"] = _target_mask(frame, max_depth=max_depth)

    anchors, covered_keys = _collect_anchor_table(frame, mode_name=mode.name, scope_name=scope_name)
    frame["in_targeted_subtree"] = [
        (str(case_name), str(parent)) in covered_keys
        for case_name, parent in zip(frame["case_name"], frame["parent"])
    ]

    outputs_summary: list[dict[str, object]] = []
    outputs_subfamily: list[pd.DataFrame] = []
    outputs_top20: list[pd.DataFrame] = []

    baseline_ranked = _rank_false_global_rows(frame)
    baseline_marked = _with_top20_flag(frame, baseline_ranked)
    outputs_summary.append(_overall_summary(baseline_marked, mode_name=mode.name, policy_name="baseline", scope_name=scope_name))
    outputs_subfamily.append(_subfamily_summary(baseline_marked, mode_name=mode.name, policy_name="baseline", scope_name=scope_name))
    if not baseline_ranked.empty:
        top = baseline_ranked.head(20).copy()
        top.insert(0, "mode", mode.name)
        top.insert(1, "scope", scope_name)
        top.insert(2, "policy", "baseline")
        outputs_top20.append(top)

    downrank_ranked = _rank_false_global_rows_downranked(frame)
    downrank_marked = _with_top20_flag(frame, downrank_ranked)
    outputs_summary.append(_overall_summary(downrank_marked, mode_name=mode.name, policy_name="downrank_subtree", scope_name=scope_name))
    outputs_subfamily.append(_subfamily_summary(downrank_marked, mode_name=mode.name, policy_name="downrank_subtree", scope_name=scope_name))
    if not downrank_ranked.empty:
        top = downrank_ranked.head(20).copy()
        top.insert(0, "mode", mode.name)
        top.insert(1, "scope", scope_name)
        top.insert(2, "policy", "downrank_subtree")
        outputs_top20.append(top)

    suppressed = frame.loc[~frame["in_targeted_subtree"]].copy()
    suppress_ranked = _rank_false_global_rows(suppressed)
    suppress_marked = _with_top20_flag(suppressed, suppress_ranked)
    outputs_summary.append(_overall_summary(suppress_marked, mode_name=mode.name, policy_name="suppress_subtree", scope_name=scope_name))
    outputs_subfamily.append(_subfamily_summary(suppress_marked, mode_name=mode.name, policy_name="suppress_subtree", scope_name=scope_name))
    if not suppress_ranked.empty:
        top = suppress_ranked.head(20).copy()
        top.insert(0, "mode", mode.name)
        top.insert(1, "scope", scope_name)
        top.insert(2, "policy", "suppress_subtree")
        outputs_top20.append(top)

    return (
        pd.DataFrame(outputs_summary),
        pd.concat(outputs_subfamily, ignore_index=True) if outputs_subfamily else pd.DataFrame(),
        anchors,
        pd.concat(outputs_top20, ignore_index=True) if outputs_top20 else pd.DataFrame(),
    )


def main() -> None:
    args = _parse_args()
    all_summary: list[pd.DataFrame] = []
    all_subfamily: list[pd.DataFrame] = []
    all_anchors: list[pd.DataFrame] = []
    all_top20: list[pd.DataFrame] = []

    for mode in _DEFAULT_MODES:
        for scope_name, max_depth in (("root_only", 0), ("early_depth_le1", 1)):
            summary, subfamily, anchors, top20 = _analyze_mode(mode, scope_name=scope_name, max_depth=max_depth)
            all_summary.append(summary)
            all_subfamily.append(subfamily)
            all_anchors.append(anchors)
            all_top20.append(top20)

    summary_df = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    subfamily_df = pd.concat(all_subfamily, ignore_index=True) if all_subfamily else pd.DataFrame()
    anchor_df = pd.concat(all_anchors, ignore_index=True) if all_anchors else pd.DataFrame()
    top20_df = pd.concat(all_top20, ignore_index=True) if all_top20 else pd.DataFrame()

    summary_path = (_ROOT / args.summary_output_csv).resolve()
    subfamily_path = (_ROOT / args.subfamily_output_csv).resolve()
    anchor_path = (_ROOT / args.anchor_output_csv).resolve()
    top20_path = (_ROOT / args.top20_output_csv).resolve()
    markdown_path = (_ROOT / args.summary_markdown).resolve()

    summary_df.to_csv(summary_path, index=False)
    subfamily_df.to_csv(subfamily_path, index=False)
    anchor_df.to_csv(anchor_path, index=False)
    top20_df.to_csv(top20_path, index=False)
    _write_markdown(summary_df, anchor_df, markdown_path)

    print(f"Summary CSV: {summary_path}")
    print(f"Subfamily CSV: {subfamily_path}")
    print(f"Anchor CSV: {anchor_path}")
    print(f"Top20 CSV: {top20_path}")
    print(f"Summary markdown: {markdown_path}")


if __name__ == "__main__":
    main()
