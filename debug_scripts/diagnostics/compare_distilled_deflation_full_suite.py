#!/usr/bin/env python3
"""Compare global-constant vs distilled-runtime deflation on the full KL suite.

Runs the default benchmark case suite with four configurations:
- current_off:      global_constant deflation, ONE_ACTIVE_1D_MODE=off
- distilled_off:    distilled_runtime deflation, ONE_ACTIVE_1D_MODE=off
- current_guard:    global_constant deflation, ONE_ACTIVE_1D_MODE=per_tree_load_guard
- distilled_guard:  distilled_runtime deflation, ONE_ACTIVE_1D_MODE=per_tree_load_guard

Outputs aggregate JSON and per-case CSV artifacts under /tmp by default.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
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


@dataclass(frozen=True)
class RunConfig:
    label: str
    deflation_mode: str
    one_active_mode: str


DEFAULT_CONFIGS: tuple[RunConfig, ...] = (
    RunConfig("current_off", "global_constant", "off"),
    RunConfig("distilled_off", "distilled_runtime", "off"),
    RunConfig("current_guard", "global_constant", "per_tree_load_guard"),
    RunConfig("distilled_guard", "distilled_runtime", "per_tree_load_guard"),
)


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
    z = linkage(
        pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC),
        method=config.TREE_LINKAGE_METHOD,
    )
    tree = PosetTree.from_linkage(z, leaf_names=data_df.index.tolist())
    return tree, data_df, y_true


def _run_case(case: dict[str, Any], *, run_config: RunConfig) -> dict[str, Any]:
    previous_deflation = config.SIBLING_DEFLATION_MODE
    previous_one_active = config.ONE_ACTIVE_1D_MODE
    previous_model_path = config.SIBLING_DISTILLED_DEFLATION_MODEL_PATH
    try:
        config.SIBLING_DEFLATION_MODE = run_config.deflation_mode
        config.ONE_ACTIVE_1D_MODE = run_config.one_active_mode

        tree, data_df, y_true = _build_tree(case)
        started = time.time()
        decomp = tree.decompose(
            leaf_data=data_df,
            alpha_local=config.EDGE_ALPHA,
            sibling_alpha=config.SIBLING_ALPHA,
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
            "deflation_mode": run_config.deflation_mode,
            "one_active_mode": run_config.one_active_mode,
        }
    finally:
        config.SIBLING_DEFLATION_MODE = previous_deflation
        config.ONE_ACTIVE_1D_MODE = previous_one_active
        config.SIBLING_DISTILLED_DEFLATION_MODEL_PATH = previous_model_path


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
        "--cases",
        nargs="*",
        default=None,
        help="Optional list of case names. Defaults to the full suite.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/kl_distilled_deflation_compare"),
        help="Artifact output directory.",
    )
    parser.add_argument(
        "--model-json",
        type=Path,
        default=Path("debug_scripts/enhancement_lab/_oracle_policy_distilled_deflation_coefficients.json"),
        help="Learned distilled-deflation model artifact used by distilled_runtime runs.",
    )
    args = parser.parse_args()

    cases = _get_cases(args.cases)
    case_names = [str(case["name"]) for case in cases]
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "per_case.csv"
    out_json = out_dir / "summary.json"

    rows_by_config: dict[str, list[dict[str, Any]]] = {cfg.label: [] for cfg in DEFAULT_CONFIGS}
    started = time.time()

    for case_index, case in enumerate(cases, start=1):
        case_name = str(case["name"])
        for cfg in DEFAULT_CONFIGS:
            print(f"[{case_index}/{len(cases)}] {cfg.label:<16} {case_name}", flush=True)
            try:
                if cfg.deflation_mode == "distilled_runtime":
                    config.SIBLING_DISTILLED_DEFLATION_MODEL_PATH = str(
                        (REPO_ROOT / args.model_json).resolve()
                    )
                else:
                    config.SIBLING_DISTILLED_DEFLATION_MODEL_PATH = None
                rows_by_config[cfg.label].append(_run_case(case, run_config=cfg))
            except Exception as err:  # pragma: no cover - diagnostic script
                rows_by_config[cfg.label].append(
                    {
                        "case": case_name,
                        "category": str(case.get("category", "unknown")),
                        "true_k": _true_k(case),
                        "found_k": None,
                        "ari": float("nan"),
                        "elapsed_sec": float("nan"),
                        "deflation_mode": cfg.deflation_mode,
                        "one_active_mode": cfg.one_active_mode,
                        "error": repr(err),
                    }
                )

    merged_rows: list[dict[str, Any]] = []
    by_case_config = {
        (row["case"], label): row
        for label, config_rows in rows_by_config.items()
        for row in config_rows
    }
    for case_name in case_names:
        case_row = {"case": case_name}
        case = next(case for case in cases if str(case["name"]) == case_name)
        case_row["category"] = str(case.get("category", "unknown"))
        case_row["true_k"] = _true_k(case)
        for cfg in DEFAULT_CONFIGS:
            row = by_case_config[(case_name, cfg.label)]
            case_row[f"{cfg.label}_found_k"] = row.get("found_k")
            case_row[f"{cfg.label}_ari"] = row.get("ari")
            case_row[f"{cfg.label}_error"] = row.get("error")
        merged_rows.append(case_row)

    pd.DataFrame(merged_rows).to_csv(out_csv, index=False)

    summary = {
        "cases": len(cases),
        "elapsed_sec": float(time.time() - started),
        "configs": {
            cfg.label: {
                "deflation_mode": cfg.deflation_mode,
                "one_active_mode": cfg.one_active_mode,
                **_summarize(rows_by_config[cfg.label]),
            }
            for cfg in DEFAULT_CONFIGS
        },
        "pairwise": {
            "off_distilled_vs_current": _pairwise_deltas(
                rows_by_config["current_off"],
                rows_by_config["distilled_off"],
            ),
            "guard_distilled_vs_current": _pairwise_deltas(
                rows_by_config["current_guard"],
                rows_by_config["distilled_guard"],
            ),
        },
    }

    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nWrote per-case results to {out_csv}")
    print(f"Wrote summary to {out_json}")


if __name__ == "__main__":
    main()
