#!/usr/bin/env python3
"""Compare current Gate 3 sibling-dimension behavior against JL fallback.

Runs the full default benchmark case suite twice:
1. Current runtime behavior from sibling_config.py.
2. Reconstructed JL fallback by disabling sibling spectral overrides.

Outputs an aggregate JSON summary and a per-case CSV under /tmp so the repo
worktree stays clean apart from this helper script.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LAB_ROOT = REPO_ROOT / "debug_scripts" / "enhancement_lab"
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

os.environ.setdefault("KL_TE_N_JOBS", "1")

from benchmarks.shared.cases import get_default_test_cases  # noqa: E402
from debug_scripts.enhancement_lab.lab_helpers import (  # noqa: E402
    build_tree_and_data,
    compute_ari,
    temporary_attr,
)
from kl_clustering_analysis import config  # noqa: E402
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence import (  # noqa: E402
    sibling_config,
)


def run_case(case_name: str, *, use_jl_fallback: bool) -> dict[str, Any]:
    tree, data_df, y_true, test_case = build_tree_and_data(case_name)

    if use_jl_fallback:
        with temporary_attr(
            sibling_config,
            "derive_sibling_spectral_dims",
            lambda tree_obj, df: None,
        ):
            decomp = tree.decompose(
                leaf_data=data_df,
                alpha_local=config.SIBLING_ALPHA,
                sibling_alpha=config.SIBLING_ALPHA,
            )
    else:
        decomp = tree.decompose(
            leaf_data=data_df,
            alpha_local=config.SIBLING_ALPHA,
            sibling_alpha=config.SIBLING_ALPHA,
        )

    ari = float("nan")
    ari_error: str | None = None
    if y_true is not None:
        try:
            ari = float(compute_ari(decomp, data_df, y_true))
        except ValueError as err:
            ari_error = str(err)

    result = {
        "case": case_name,
        "true_k": test_case.get("n_clusters"),
        "found_k": int(decomp["num_clusters"]),
        "ari": ari,
    }
    if ari_error is not None:
        result["ari_error"] = ari_error
    return result


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ari_values = [row["ari"] for row in rows if not math.isnan(row["ari"])]
    exact_k = sum(
        1 for row in rows if row["true_k"] is not None and row["found_k"] == row["true_k"]
    )
    return {
        "cases": len(rows),
        "mean_ari": float(np.mean(ari_values)) if ari_values else float("nan"),
        "median_ari": float(np.median(ari_values)) if ari_values else float("nan"),
        "exact_k": exact_k,
        "k_eq_1": sum(1 for row in rows if row["found_k"] == 1),
    }


def main() -> None:
    case_names = [case["name"] for case in get_default_test_cases()]
    out_dir = Path("/tmp/kl_compare_current_vs_jl")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "per_case.csv"
    out_json = out_dir / "summary.json"

    current_rows: list[dict[str, Any]] = []
    jl_rows: list[dict[str, Any]] = []
    started = time.time()

    for index, case_name in enumerate(case_names, start=1):
        print(f"[{index}/{len(case_names)}] current      {case_name}", flush=True)
        current_rows.append(run_case(case_name, use_jl_fallback=False))

        print(f"[{index}/{len(case_names)}] jl-fallback  {case_name}", flush=True)
        jl_rows.append(run_case(case_name, use_jl_fallback=True))

    current_map = {row["case"]: row for row in current_rows}
    jl_map = {row["case"]: row for row in jl_rows}

    merged_rows: list[dict[str, Any]] = []
    for case_name in case_names:
        current_row = current_map[case_name]
        jl_row = jl_map[case_name]
        delta_ari = float("nan")
        if not math.isnan(current_row["ari"]) and not math.isnan(jl_row["ari"]):
            delta_ari = jl_row["ari"] - current_row["ari"]
        merged_rows.append(
            {
                "case": case_name,
                "true_k": current_row["true_k"],
                "current_found_k": current_row["found_k"],
                "current_ari": current_row["ari"],
                "jl_found_k": jl_row["found_k"],
                "jl_ari": jl_row["ari"],
                "delta_ari": delta_ari,
                "delta_found_k": jl_row["found_k"] - current_row["found_k"],
            }
        )

    improved = sorted(
        [row for row in merged_rows if not math.isnan(row["delta_ari"]) and row["delta_ari"] > 0.0],
        key=lambda row: row["delta_ari"],
        reverse=True,
    )
    regressed = sorted(
        [row for row in merged_rows if not math.isnan(row["delta_ari"]) and row["delta_ari"] < 0.0],
        key=lambda row: row["delta_ari"],
    )

    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(merged_rows[0].keys()))
        writer.writeheader()
        writer.writerows(merged_rows)

    payload = {
        "elapsed_seconds": time.time() - started,
        "current": summarize(current_rows),
        "jl_fallback": summarize(jl_rows),
        "n_improved": len(improved),
        "n_regressed": len(regressed),
        "top_improved": improved[:15],
        "top_regressed": regressed[:15],
        "csv": str(out_csv),
    }

    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n=== SUMMARY ===")
    print(json.dumps(payload, indent=2))
    print(f"Wrote per-case CSV to {out_csv}")


if __name__ == "__main__":
    main()
