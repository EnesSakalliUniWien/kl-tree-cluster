#!/usr/bin/env python3
"""Run a compact Gate 3 ablation battery and summarize results."""

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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LAB_ROOT = REPO_ROOT / "debug_scripts" / "enhancement_lab"
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

os.environ.setdefault("KL_TE_N_JOBS", "1")

from kl_clustering_analysis import config  # noqa: E402
from debug_scripts.enhancement_lab.lab_helpers import (  # noqa: E402
    build_tree_and_data,
    compute_ari,
    temporary_config,
)


BATTERY_CASES = [
    "binary_balanced_low_noise",
    "binary_balanced_low_noise__2",
    "binary_perfect_8c",
    "gauss_extreme_noise_3c",
    "gauss_extreme_noise_highd",
    "overlap_heavy_4c_small_feat",
    "gauss_overlap_4c_med",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label",
        default="dev_baseline",
        help="Artifact label used in output filenames.",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/kl_gate3_ablation_battery",
        help="Directory for per-case CSV and summary JSON.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=BATTERY_CASES,
        help="Case names to run. Defaults to the 7-case discriminative battery.",
    )
    parser.add_argument(
        "--sibling-method",
        default=None,
        help="Temporary override for config.SIBLING_TEST_METHOD.",
    )
    parser.add_argument(
        "--sibling-whitening",
        default=None,
        help="Temporary override for config.SIBLING_WHITENING.",
    )
    return parser.parse_args()


def run_case(case_name: str) -> dict[str, Any]:
    """Run one battery case and return summary metrics."""
    tree, data_df, y_true, test_case = build_tree_and_data(case_name)
    started = time.time()
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

    row: dict[str, Any] = {
        "case": case_name,
        "true_k": test_case.get("n_clusters"),
        "found_k": int(decomp["num_clusters"]),
        "ari": ari,
        "elapsed_seconds": time.time() - started,
    }
    if row["true_k"] is not None:
        row["delta_found_k"] = int(row["found_k"]) - int(row["true_k"])
    if ari_error is not None:
        row["ari_error"] = ari_error
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate summary metrics for one battery run."""
    ari_values = [float(row["ari"]) for row in rows if not math.isnan(float(row["ari"]))]
    exact_k = sum(
        1
        for row in rows
        if row["true_k"] is not None and int(row["found_k"]) == int(row["true_k"])
    )
    return {
        "cases": len(rows),
        "mean_ari": float(np.mean(ari_values)) if ari_values else float("nan"),
        "median_ari": float(np.median(ari_values)) if ari_values else float("nan"),
        "exact_k": exact_k,
        "k_eq_1": sum(1 for row in rows if int(row["found_k"]) == 1),
        "n_errors": sum(1 for row in rows if "ari_error" in row),
    }


def main() -> None:
    """Run the battery and write artifacts."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_csv = output_dir / f"{args.label}_per_case.csv"
    out_json = output_dir / f"{args.label}_summary.json"

    started = time.time()
    with temporary_config(
        SIBLING_TEST_METHOD=args.sibling_method or config.SIBLING_TEST_METHOD,
        SIBLING_WHITENING=args.sibling_whitening or config.SIBLING_WHITENING,
    ):
        rows: list[dict[str, Any]] = []
        for index, case_name in enumerate(args.cases, start=1):
            print(f"[{index}/{len(args.cases)}] {args.label:<16s} {case_name}", flush=True)
            rows.append(run_case(case_name))

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "label": args.label,
        "elapsed_seconds": time.time() - started,
        "cases": list(args.cases),
        "config": {
            "SIBLING_TEST_METHOD": args.sibling_method or config.SIBLING_TEST_METHOD,
            "SPECTRAL_DIMENSION_ESTIMATOR": "marchenko_pastur",
            "SIBLING_WHITENING": args.sibling_whitening or config.SIBLING_WHITENING,
        },
        "summary": summarize(rows),
        "csv": str(out_csv),
    }

    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n=== SUMMARY ===")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
