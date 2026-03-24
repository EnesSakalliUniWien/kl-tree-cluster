#!/usr/bin/env python3
"""Export node-wise deflation features for case-to-case comparison.

Focuses on the adjusted-Wald calibration inputs that drive node-level
over-deflation:
- local node test features
- parent PCA eigenvalue summaries
- subtree ("downward") burden terms
- outside-subtree burden terms
- blocked / null-like / df=1 burden slices

Default comparison:
- overlap_unbal_4c_small
- binary_unbalanced_low
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("KL_TE_N_JOBS", "1")

from debug_scripts.enhancement_lab.lab_helpers import build_tree_and_data, compute_ari, temporary_config
from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.statistics.branch_length_utils import compute_mean_branch_length
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.k_estimators import effective_rank
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence import sibling_config
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.sibling_pair_collection import (
    SiblingPairRecord,
    collect_sibling_pair_records,
)


@dataclass
class Aggregate:
    count: int
    count_null_like: int
    count_focal: int
    count_blocked: int
    count_blocked_df1_null: int
    count_raw_p_lt_005: int
    count_raw_p_lt_001: int
    sum_weight: float
    sum_ratio_weight: float
    mean_ratio_weighted: float
    sum_n_parent: int
    sum_leaf_count: int
    min_raw_p: float
    median_raw_p: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["overlap_unbal_4c_small", "binary_unbalanced_low"],
        help="Case IDs to compare.",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/nodewise_deflation_case_compare",
        help="Directory for CSV/JSON outputs.",
    )
    return parser.parse_args()


def _safe_float(x: object) -> float:
    try:
        x = float(x)
    except Exception:
        return float("nan")
    return x if np.isfinite(x) else float("nan")


def _safe_int(x: object) -> int | None:
    try:
        return int(x)
    except Exception:
        return None


def _leaf_descendants(tree: nx.DiGraph, node: str, leaves: set[str]) -> list[str]:
    return [n for n in nx.descendants(tree, node) if n in leaves]


def _truth_mix(tree: nx.DiGraph, node: str, leaves: set[str], label_by_leaf: dict[str, int]) -> dict[int, float]:
    leaf_ids = _leaf_descendants(tree, node, leaves)
    if not leaf_ids:
        return {}
    counts = pd.Series([label_by_leaf[x] for x in leaf_ids]).value_counts().sort_index()
    total = float(counts.sum())
    return {int(k): float(v / total) for k, v in counts.items()}


def _record_ratio(record: SiblingPairRecord) -> float:
    if record.degrees_of_freedom <= 0 or not np.isfinite(record.stat):
        return float("nan")
    return float(record.stat / record.degrees_of_freedom)


def _aggregate(records: Iterable[SiblingPairRecord], leaf_count_by_node: dict[str, int]) -> Aggregate:
    recs = [r for r in records if r.degrees_of_freedom > 0 and np.isfinite(r.stat)]
    if not recs:
        return Aggregate(
            count=0,
            count_null_like=0,
            count_focal=0,
            count_blocked=0,
            count_blocked_df1_null=0,
            count_raw_p_lt_005=0,
            count_raw_p_lt_001=0,
            sum_weight=0.0,
            sum_ratio_weight=0.0,
            mean_ratio_weighted=float("nan"),
            sum_n_parent=0,
            sum_leaf_count=0,
            min_raw_p=float("nan"),
            median_raw_p=float("nan"),
        )

    ratios = np.array([_record_ratio(r) for r in recs], dtype=float)
    weights = np.array([float(r.edge_weight) for r in recs], dtype=float)
    raw_ps = np.array([float(r.p_value) for r in recs], dtype=float)
    sum_weight = float(np.sum(weights))
    sum_ratio_weight = float(np.sum(ratios * weights))
    mean_ratio_weighted = float(sum_ratio_weight / sum_weight) if sum_weight > 0 else float("nan")
    return Aggregate(
        count=len(recs),
        count_null_like=sum(int(r.is_null_like) for r in recs),
        count_focal=sum(int(not r.is_null_like) for r in recs),
        count_blocked=sum(int(r.is_gate2_blocked) for r in recs),
        count_blocked_df1_null=sum(
            int(r.is_null_like and r.is_gate2_blocked and r.degrees_of_freedom == 1) for r in recs
        ),
        count_raw_p_lt_005=int(np.sum(raw_ps < 0.05)),
        count_raw_p_lt_001=int(np.sum(raw_ps < 0.01)),
        sum_weight=sum_weight,
        sum_ratio_weight=sum_ratio_weight,
        mean_ratio_weighted=mean_ratio_weighted,
        sum_n_parent=int(np.sum([int(r.n_parent) for r in recs])),
        sum_leaf_count=int(np.sum([leaf_count_by_node.get(r.parent, 0) for r in recs])),
        min_raw_p=float(np.min(raw_ps)),
        median_raw_p=float(np.median(raw_ps)),
    )


def _top_burden_parent(tree: nx.DiGraph, root: str, records: list[SiblingPairRecord], top_n: int = 12) -> list[dict[str, object]]:
    valid = [r for r in records if r.degrees_of_freedom > 0 and np.isfinite(r.stat)]
    total_weight = float(np.sum([float(r.edge_weight) for r in valid]))
    rows = []
    for r in valid:
        contrib = (_record_ratio(r) * float(r.edge_weight) / total_weight) if total_weight > 0 else float("nan")
        rows.append(
            {
                "node": r.parent,
                "depth": int(nx.shortest_path_length(tree, root, r.parent)),
                "ratio": _record_ratio(r),
                "weight": float(r.edge_weight),
                "contrib_to_c_global": contrib,
                "blocked": bool(r.is_gate2_blocked),
                "null_like": bool(r.is_null_like),
                "df": int(r.degrees_of_freedom),
                "n_parent": int(r.n_parent),
            }
        )
    rows.sort(key=lambda x: float(x["contrib_to_c_global"]), reverse=True)
    return rows[:top_n]


def _case_features(case: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    tree, data_df, y_true, tc = build_tree_and_data(case)
    root = next(node for node, d in tree.in_degree() if d == 0)
    leaves = set(data_df.index.tolist())
    label_by_leaf = dict(zip(data_df.index.tolist(), y_true.tolist(), strict=False))

    with temporary_config(ONE_ACTIVE_1D_MODE="per_tree_load_guard"):
        decomp = tree.decompose(
            leaf_data=data_df,
            alpha_local=config.EDGE_ALPHA,
            sibling_alpha=config.SIBLING_ALPHA,
        )

    stats = tree.annotations_df
    mean_bl = compute_mean_branch_length(tree) if config.FELSENSTEIN_SCALING else None
    sibling_dims = sibling_config.derive_sibling_spectral_dims(tree, stats)
    pca_projections, pca_eigenvalues = sibling_config.derive_sibling_pca_projections(stats, sibling_dims)
    child_pca = sibling_config.derive_sibling_child_pca_projections(tree, stats, sibling_dims)
    records, _ = collect_sibling_pair_records(
        tree,
        stats,
        mean_bl,
        spectral_dims=sibling_dims,
        pca_projections=pca_projections,
        pca_eigenvalues=pca_eigenvalues,
        child_pca_projections=child_pca,
        whitening=config.SIBLING_WHITENING,
    )

    record_by_parent = {r.parent: r for r in records}
    internal_binary_nodes = [n for n in tree.nodes if len(list(tree.successors(n))) == 2]
    leaf_count_by_node = {
        n: int(stats.loc[n, "leaf_count"]) if "leaf_count" in stats.columns else len(_leaf_descendants(tree, n, leaves))
        for n in internal_binary_nodes
    }
    total_weight = float(np.sum([float(r.edge_weight) for r in records if r.degrees_of_freedom > 0 and np.isfinite(r.stat)]))
    total_agg = _aggregate(records, leaf_count_by_node)

    rows: list[dict[str, object]] = []
    for node in internal_binary_nodes:
        descendants = set(nx.descendants(tree, node)) | {node}
        down_records = [r for r in records if r.parent in descendants]
        out_records = [r for r in records if r.parent not in descendants]
        down = _aggregate(down_records, leaf_count_by_node)
        out = _aggregate(out_records, leaf_count_by_node)

        rec = record_by_parent.get(node)
        eig = None if pca_eigenvalues is None else pca_eigenvalues.get(node)
        eig_arr = np.asarray(eig, dtype=float) if eig is not None else np.array([], dtype=float)
        eig_arr = eig_arr[np.isfinite(eig_arr)]
        eig_rank = float(effective_rank(np.maximum(eig_arr, 0.0))) if eig_arr.size else float("nan")
        left, right = list(tree.successors(node))

        raw_ratio = _record_ratio(rec) if rec is not None else float("nan")
        contrib = (
            raw_ratio * float(rec.edge_weight) / total_weight
            if rec is not None and np.isfinite(raw_ratio) and total_weight > 0
            else float("nan")
        )

        row = {
            "case": case,
            "true_k": int(tc.get("n_clusters", -1)),
            "found_k": int(decomp["num_clusters"]),
            "ari": float(compute_ari(decomp, data_df, y_true)),
            "root": root,
            "node": node,
            "depth": int(nx.shortest_path_length(tree, root, node)),
            "is_root": bool(node == root),
            "leaf_count": leaf_count_by_node[node],
            "n_parent": None if rec is None else int(rec.n_parent),
            "branch_length_sum": None if rec is None else float(rec.branch_length_sum),
            "is_null_like": None if rec is None else bool(rec.is_null_like),
            "is_gate2_blocked": None if rec is None else bool(rec.is_gate2_blocked),
            "edge_weight": None if rec is None else float(rec.edge_weight),
            "raw_stat": None if rec is None else float(rec.stat),
            "raw_df": None if rec is None else int(rec.degrees_of_freedom),
            "raw_ratio": raw_ratio,
            "raw_p": None if rec is None else float(rec.p_value),
            "adj_stat": _safe_float(stats.loc[node, "Sibling_Test_Statistic"]) if "Sibling_Test_Statistic" in stats.columns else float("nan"),
            "adj_df": _safe_float(stats.loc[node, "Sibling_Degrees_of_Freedom"]) if "Sibling_Degrees_of_Freedom" in stats.columns else float("nan"),
            "adj_p": _safe_float(stats.loc[node, "Sibling_Divergence_P_Value"]) if "Sibling_Divergence_P_Value" in stats.columns else float("nan"),
            "adj_p_bh": _safe_float(stats.loc[node, "Sibling_Divergence_P_Value_Corrected"]) if "Sibling_Divergence_P_Value_Corrected" in stats.columns else float("nan"),
            "sib_diff": bool(stats.loc[node, "Sibling_BH_Different"]) if "Sibling_BH_Different" in stats.columns else None,
            "sib_same": bool(stats.loc[node, "Sibling_BH_Same"]) if "Sibling_BH_Same" in stats.columns else None,
            "sib_skip": bool(stats.loc[node, "Sibling_Divergence_Skipped"]) if "Sibling_Divergence_Skipped" in stats.columns else None,
            "left": left,
            "right": right,
            "left_sig": bool(stats.loc[left, "Child_Parent_Divergence_Significant"]),
            "right_sig": bool(stats.loc[right, "Child_Parent_Divergence_Significant"]),
            "left_tested": bool(stats.loc[left, "Child_Parent_Divergence_Tested"]) if "Child_Parent_Divergence_Tested" in stats.columns else None,
            "right_tested": bool(stats.loc[right, "Child_Parent_Divergence_Tested"]) if "Child_Parent_Divergence_Tested" in stats.columns else None,
            "left_blocked": bool(stats.loc[left, "Child_Parent_Divergence_Ancestor_Blocked"]) if "Child_Parent_Divergence_Ancestor_Blocked" in stats.columns else None,
            "right_blocked": bool(stats.loc[right, "Child_Parent_Divergence_Ancestor_Blocked"]) if "Child_Parent_Divergence_Ancestor_Blocked" in stats.columns else None,
            "left_edge_p_bh": _safe_float(stats.loc[left, "Child_Parent_Divergence_P_Value_BH"]) if "Child_Parent_Divergence_P_Value_BH" in stats.columns else float("nan"),
            "right_edge_p_bh": _safe_float(stats.loc[right, "Child_Parent_Divergence_P_Value_BH"]) if "Child_Parent_Divergence_P_Value_BH" in stats.columns else float("nan"),
            "spectral_k": None if sibling_dims is None else _safe_int(sibling_dims.get(node)),
            "pca_eig_count": int(eig_arr.size),
            "pca_eig_top1": float(eig_arr[0]) if eig_arr.size >= 1 else float("nan"),
            "pca_eig_top2": float(eig_arr[1]) if eig_arr.size >= 2 else float("nan"),
            "pca_eig_sum_top4": float(np.sum(eig_arr[:4])) if eig_arr.size else float("nan"),
            "pca_eig_sum_all": float(np.sum(eig_arr)) if eig_arr.size else float("nan"),
            "pca_effective_rank": eig_rank,
            "pca_eigenvalues_top8_json": json.dumps([float(x) for x in eig_arr[:8].tolist()]),
            "root_truth_mix_left_json": json.dumps(_truth_mix(tree, left, leaves, label_by_leaf)) if node == root else "",
            "root_truth_mix_right_json": json.dumps(_truth_mix(tree, right, leaves, label_by_leaf)) if node == root else "",
            "c_global": float(stats.attrs["sibling_divergence_audit"]["global_inflation_factor"]),
            "effective_n_global": float(stats.attrs["sibling_divergence_audit"]["diagnostics"]["effective_n"]),
            "contrib_to_c_global": contrib,
        }

        for prefix, agg in (("down", down), ("out", out), ("all", total_agg)):
            agg_dict = asdict(agg)
            for key, value in agg_dict.items():
                row[f"{prefix}_{key}"] = value

        rows.append(row)

    node_df = pd.DataFrame(rows)

    focus_nodes = [root]
    focus_nodes.extend(child for child in tree.successors(root) if child in set(node_df["node"]))
    top_nodes = _top_burden_parent(tree, root, records, top_n=12)
    focus_nodes.extend(x["node"] for x in top_nodes)
    focus_nodes = list(dict.fromkeys(focus_nodes))

    focus_df = node_df[node_df["node"].isin(focus_nodes)].copy()
    focus_df.insert(1, "focus_rank", range(1, len(focus_df) + 1))
    return node_df, focus_df


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_nodes: list[pd.DataFrame] = []
    all_focus: list[pd.DataFrame] = []

    for case in args.cases:
        node_df, focus_df = _case_features(case)
        all_nodes.append(node_df)
        all_focus.append(focus_df)

    nodes_df = pd.concat(all_nodes, ignore_index=True)
    focus_df = pd.concat(all_focus, ignore_index=True)

    case_summary = (
        nodes_df[nodes_df["is_root"]]
        .loc[
            :,
            [
                "case",
                "true_k",
                "found_k",
                "ari",
                "node",
                "raw_ratio",
                "adj_p",
                "adj_p_bh",
                "c_global",
                "effective_n_global",
                "all_count_null_like",
                "all_count_focal",
                "all_count_blocked",
                "all_count_blocked_df1_null",
                "all_sum_ratio_weight",
                "all_mean_ratio_weighted",
            ],
        ]
        .rename(columns={"node": "root"})
        .reset_index(drop=True)
    )

    nodes_path = out_dir / "node_features.csv"
    focus_path = out_dir / "focus_nodes.csv"
    summary_path = out_dir / "case_summary.csv"
    nodes_df.to_csv(nodes_path, index=False)
    focus_df.to_csv(focus_path, index=False)
    case_summary.to_csv(summary_path, index=False)

    print(summary_path)
    print(focus_path)
    print(nodes_path)


if __name__ == "__main__":
    main()
