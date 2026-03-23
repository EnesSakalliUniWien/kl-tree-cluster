#!/usr/bin/env python3
"""Compare Gate 2 FDR methods on the live KL decomposition path.

Runs the default benchmark case suite with:
- tree_bh
- level_wise
- flat

Outputs aggregate JSON and per-case CSV artifacts under /tmp by default so the
repo worktree stays clean apart from this helper script.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("KL_TE_N_JOBS", "1")

from benchmarks.shared.cases import get_default_test_cases  # noqa: E402
from benchmarks.shared.generators import generate_case_data  # noqa: E402
from debug_scripts.enhancement_lab.lab_helpers import compute_ari  # noqa: E402
from kl_clustering_analysis import config  # noqa: E402
from kl_clustering_analysis.tree.poset_tree import PosetTree  # noqa: E402

DEFAULT_METHODS = ("tree_bh", "level_wise", "flat")


def _true_k(case: dict[str, Any]) -> int | None:
    if case.get("n_clusters") is not None:
        return int(case["n_clusters"])
    if case.get("sizes") is not None:
        return int(len(case["sizes"]))
    return None


def _get_cases(selected_names: list[str] | None) -> list[dict[str, Any]]:
    all_cases = get_default_test_cases()
    if not selected_names:
        return all_cases

    index = {str(case["name"]): case for case in all_cases}
    missing = [name for name in selected_names if name not in index]
    if missing:
        raise KeyError(f"Unknown case(s): {missing}")
    return [index[name] for name in selected_names]


def _build_tree(case: dict[str, Any]) -> tuple[PosetTree, pd.DataFrame, np.ndarray | None]:
    data_df, y_true, _, _ = generate_case_data(case)
    Z = linkage(
        pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC),
        method=config.TREE_LINKAGE_METHOD,
    )
    tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())
    return tree, data_df, y_true


def _run_case(case: dict[str, Any], *, gate2_fdr_method: str) -> dict[str, Any]:
    tree, data_df, y_true = _build_tree(case)
    started = time.time()
    decomp = tree.decompose(
        leaf_data=data_df,
        alpha_local=config.EDGE_ALPHA,
        sibling_alpha=config.SIBLING_ALPHA,
        gate2_fdr_method=gate2_fdr_method,
    )
    elapsed = time.time() - started

    ari = float("nan")
    if y_true is not None:
        ari = float(compute_ari(decomp, data_df, y_true))

    return {
        "case": str(case["name"]),
        "category": str(case.get("category", "unknown")),
        "true_k": _true_k(case),
        "found_k": int(decomp["num_clusters"]),
        "ari": ari,
        "elapsed_sec": float(elapsed),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_rows = [row for row in rows if "error" not in row]
    ari_values = [row["ari"] for row in valid_rows if not math.isnan(row["ari"])]
    exact_k = sum(
        1
        for row in valid_rows
        if row["true_k"] is not None and int(row["found_k"]) == int(row["true_k"])
    )
    return {
        "cases": len(rows),
        "successful_cases": len(valid_rows),
        "errors": len(rows) - len(valid_rows),
        "mean_ari": float(np.mean(ari_values)) if ari_values else float("nan"),
        "median_ari": float(np.median(ari_values)) if ari_values else float("nan"),
        "exact_k": int(exact_k),
        "k_eq_1": int(sum(1 for row in valid_rows if int(row["found_k"]) == 1)),
        "mean_elapsed_sec": (
            float(np.mean([row["elapsed_sec"] for row in valid_rows])) if valid_rows else float("nan")
        ),
    }


def _is_sbm_case(row: dict[str, Any]) -> bool:
    category = str(row.get("category", ""))
    return category.startswith("sbm")


def _pairwise_deltas(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_map = {row["case"]: row for row in baseline_rows if "error" not in row}
    candidate_map = {row["case"]: row for row in candidate_rows if "error" not in row}
    common_cases = [case for case in baseline_map if case in candidate_map]

    rows: list[dict[str, Any]] = []
    for case_name in common_cases:
        baseline = baseline_map[case_name]
        candidate = candidate_map[case_name]
        delta_ari = float("nan")
        if not math.isnan(baseline["ari"]) and not math.isnan(candidate["ari"]):
            delta_ari = float(candidate["ari"] - baseline["ari"])
        rows.append(
            {
                "case": case_name,
                "category": baseline["category"],
                "baseline_found_k": int(baseline["found_k"]),
                "candidate_found_k": int(candidate["found_k"]),
                "delta_found_k": int(candidate["found_k"]) - int(baseline["found_k"]),
                "baseline_ari": baseline["ari"],
                "candidate_ari": candidate["ari"],
                "delta_ari": delta_ari,
            }
        )

    improved = sorted(
        [row for row in rows if not math.isnan(row["delta_ari"]) and row["delta_ari"] > 0.0],
        key=lambda row: row["delta_ari"],
        reverse=True,
    )
    regressed = sorted(
        [row for row in rows if not math.isnan(row["delta_ari"]) and row["delta_ari"] < 0.0],
        key=lambda row: row["delta_ari"],
    )
    return {
        "n_improved": len(improved),
        "n_regressed": len(regressed),
        "top_improved": improved[:15],
        "top_regressed": regressed[:15],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        choices=list(DEFAULT_METHODS),
        help="Gate 2 FDR methods to compare.",
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="Optional list of case names. Defaults to the full suite.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/kl_gate2_fdr_method_compare"),
        help="Artifact output directory.",
    )
    args = parser.parse_args()

    cases = _get_cases(args.cases)
    case_names = [str(case["name"]) for case in cases]
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "per_case.csv"
    out_json = out_dir / "summary.json"

    rows_by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in args.methods}
    started = time.time()

    for case_index, case in enumerate(cases, start=1):
        case_name = str(case["name"])
        for method in args.methods:
            print(
                f"[{case_index}/{len(cases)}] {method:<10} {case_name}",
                flush=True,
            )
            try:
                rows_by_method[method].append(_run_case(case, gate2_fdr_method=method))
            except Exception as err:  # pragma: no cover - diagnostic script
                rows_by_method[method].append(
                    {
                        "case": case_name,
                        "category": str(case.get("category", "unknown")),
                        "true_k": _true_k(case),
                        "found_k": None,
                        "ari": float("nan"),
                        "elapsed_sec": float("nan"),
                        "error": repr(err),
                    }
                )

    merged_rows: list[dict[str, Any]] = []
    by_case_method = {
        (row["case"], method): row
        for method, method_rows in rows_by_method.items()
        for row in method_rows
    }
    for case_name in case_names:
        case_row = {"case": case_name}
        case = next(case for case in cases if str(case["name"]) == case_name)
        case_row["category"] = str(case.get("category", "unknown"))
        case_row["true_k"] = _true_k(case)
        for method in args.methods:
            row = by_case_method[(case_name, method)]
            case_row[f"{method}_found_k"] = row.get("found_k")
            case_row[f"{method}_ari"] = row.get("ari")
            case_row[f"{method}_elapsed_sec"] = row.get("elapsed_sec")
            case_row[f"{method}_error"] = row.get("error")
        merged_rows.append(case_row)

    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(merged_rows[0].keys()))
        writer.writeheader()
        writer.writerows(merged_rows)

    summary = {
        "elapsed_seconds": float(time.time() - started),
        "config": {
            "edge_alpha": config.EDGE_ALPHA,
            "sibling_alpha": config.SIBLING_ALPHA,
            "tree_distance_metric": config.TREE_DISTANCE_METRIC,
            "tree_linkage_method": config.TREE_LINKAGE_METHOD,
        },
        "methods": {method: _summarize(rows) for method, rows in rows_by_method.items()},
        "sbm_only": {
            method: _summarize([row for row in rows if _is_sbm_case(row)])
            for method, rows in rows_by_method.items()
        },
        "pairwise_vs_tree_bh": {},
        "csv": str(out_csv),
    }
    if "tree_bh" in rows_by_method:
        baseline_rows = rows_by_method["tree_bh"]
        for method in args.methods:
            if method == "tree_bh":
                continue
            summary["pairwise_vs_tree_bh"][method] = _pairwise_deltas(
                baseline_rows,
                rows_by_method[method],
            )

    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"Wrote per-case CSV to {out_csv}")


if __name__ == "__main__":
    main()
