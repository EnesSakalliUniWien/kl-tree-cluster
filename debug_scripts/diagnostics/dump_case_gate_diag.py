#!/usr/bin/env python3
"""Dump gate-level diagnostics for a single benchmark case as JSON."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import networkx as nx

REPO_ROOT = Path(os.environ.get("KL_TE_REPO_ROOT", Path(__file__).resolve().parents[2])).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LAB_ROOT = Path(os.environ.get("KL_TE_LAB_ROOT", REPO_ROOT / "debug_scripts" / "enhancement_lab")).resolve()
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

os.environ.setdefault("KL_TE_N_JOBS", "1")

from debug_scripts.enhancement_lab.lab_helpers import build_tree_and_data, compute_ari  # noqa: E402
from kl_clustering_analysis import config  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    """Run one case and write internal-node gate diagnostics to JSON."""
    args = parse_args()
    tree, data_df, y_true, _ = build_tree_and_data(args.case)
    decomp = tree.decompose(
        leaf_data=data_df,
        alpha_local=config.SIBLING_ALPHA,
        sibling_alpha=config.SIBLING_ALPHA,
    )
    stats = tree.stats_df.copy()
    root = next(node for node, degree in tree.in_degree() if degree == 0)
    depths = dict(nx.shortest_path_length(tree, root))

    rows: list[dict[str, object]] = []
    for node in tree.nodes():
        children = list(tree.successors(node))
        if len(children) != 2:
            continue
        left, right = children
        rows.append(
            {
                "node": node,
                "depth": int(depths[node]),
                "leaf_count": int(stats.loc[node, "leaf_count"]) if "leaf_count" in stats.columns else None,
                "left": left,
                "right": right,
                "left_sig": bool(stats.loc[left, "Child_Parent_Divergence_Significant"]),
                "right_sig": bool(stats.loc[right, "Child_Parent_Divergence_Significant"]),
                "g2_pass": bool(stats.loc[left, "Child_Parent_Divergence_Significant"])
                or bool(stats.loc[right, "Child_Parent_Divergence_Significant"]),
                "sib_skip": bool(stats.loc[node, "Sibling_Divergence_Skipped"]),
                "sib_diff": bool(stats.loc[node, "Sibling_BH_Different"]),
                "sib_same": bool(stats.loc[node, "Sibling_BH_Same"])
                if "Sibling_BH_Same" in stats.columns
                else None,
                "sib_p": float(stats.loc[node, "Sibling_Divergence_P_Value"])
                if "Sibling_Divergence_P_Value" in stats.columns
                else None,
                "sib_p_bh": float(stats.loc[node, "Sibling_Divergence_P_Value_Corrected"])
                if "Sibling_Divergence_P_Value_Corrected" in stats.columns
                else None,
                "sib_stat": float(stats.loc[node, "Sibling_Test_Statistic"])
                if "Sibling_Test_Statistic" in stats.columns
                else None,
                "sib_df": float(stats.loc[node, "Sibling_Degrees_of_Freedom"])
                if "Sibling_Degrees_of_Freedom" in stats.columns
                else None,
                "invalid": bool(stats.loc[node, "Sibling_Divergence_Invalid"])
                if "Sibling_Divergence_Invalid" in stats.columns
                else None,
                "test_method": str(stats.loc[node, "Sibling_Test_Method"])
                if "Sibling_Test_Method" in stats.columns
                else None,
            }
        )

    out = {
        "case": args.case,
        "found_k": int(decomp["num_clusters"]),
        "ari": float(compute_ari(decomp, data_df, y_true)),
        "root": root,
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(args.output)
    print(json.dumps({"found_k": out["found_k"], "ari": out["ari"], "root": root}, indent=2))
    print(
        "split_nodes",
        sum(1 for row in rows if row["g2_pass"] and row["sib_diff"] and not row["sib_skip"]),
    )


if __name__ == "__main__":
    main()
