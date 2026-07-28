#!/usr/bin/env python3
"""Diagnose empirical-null inflation on oracle-recoverable sibling blockers."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from tree_break_selection.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_SIBLING_ALPHA,
)

from benchmarks.diagnostics.calibration.sibling.nulls.runner_support import (
    prepare_sibling_diagnostic,
)
from benchmarks.diagnostics.oracle.oracle_tree_recoverability import (
    FAILURE_CLASS_GATE_UNDER_SPLIT,
)
from benchmarks.diagnostics.runner_support import (
    configure_serial_runtime,
    create_result_directory,
    parse_csv_values,
    resolve_classified_cases,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build target and calibration-contributor tables for the sibling "
            "empirical-null inflation estimator."
        )
    )
    parser.add_argument(
        "--suite",
        choices=("regression_gate", "full"),
        default="full",
        help="Benchmark suite containing the classified case names.",
    )
    parser.add_argument(
        "--classification-csv",
        type=Path,
        default=None,
        help=(
            "Oracle recoverability CSV with failure_class. Defaults to the latest "
            "oracle_tree_recoverability CSV containing failure_class."
        ),
    )
    parser.add_argument(
        "--failure-classes",
        default=FAILURE_CLASS_GATE_UNDER_SPLIT,
        help="Comma-separated failure classes to diagnose.",
    )
    parser.add_argument(
        "--case-names",
        default="",
        help="Optional comma-separated case names after failure-class filtering.",
    )
    parser.add_argument(
        "--max-contributors",
        type=int,
        default=10,
        help="Maximum local calibration contributor rows per target.",
    )
    parser.add_argument(
        "--contributors-for-all-crossings",
        action="store_true",
        help=(
            "Emit contributor rows for every focal test whose raw p-value crosses "
            "alpha after inflation or sibling FDR, not only oracle blocker nodes."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to "
            "benchmarks/results/sibling_inflation_diagnostic_<timestamp>/."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    configure_serial_runtime()

    selection = resolve_classified_cases(
        suite=args.suite,
        classification_csv=args.classification_csv,
        failure_classes=parse_csv_values(args.failure_classes),
        case_names=parse_csv_values(args.case_names),
        required_columns=("tbs_ari", "oracle_true_k_subtree_ari"),
    )
    output_dir = create_result_directory(
        args.output_dir,
        study_slug="sibling_inflation_diagnostic",
    )
    targets_csv = output_dir / "sibling_inflation_targets.csv"
    contributors_csv = output_dir / "sibling_inflation_contributors.csv"
    summary_csv = output_dir / "sibling_inflation_summary.csv"
    metadata_json = output_dir / "sibling_inflation_metadata.json"

    started_at = time.perf_counter()
    target_frames: list[pd.DataFrame] = []
    contributor_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    for index, selected_case in enumerate(selection.cases, start=1):
        case = selected_case.case
        classification = selected_case.classification
        print(
            f"[{index}/{len(selection.cases)}] "
            f"{case['name']} ({classification['failure_class']})",
            flush=True,
        )
        prepared = prepare_sibling_diagnostic(
            case,
            classification,
            max_contributors=int(args.max_contributors),
            contributors_for_all_crossings=bool(args.contributors_for_all_crossings),
        )
        target_frames.append(prepared.tables.targets)
        contributor_frames.append(prepared.tables.contributors)
        summary_frames.append(prepared.tables.summary)

    targets_df = pd.concat(target_frames, ignore_index=True)
    contributors_df = pd.concat(contributor_frames, ignore_index=True)
    summary_df = pd.concat(summary_frames, ignore_index=True)
    targets_df.to_csv(targets_csv, index=False)
    contributors_df.to_csv(contributors_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    elapsed_sec = time.perf_counter() - started_at
    metadata = {
        "suite": args.suite,
        "classification_csv": str(selection.classification_path),
        "failure_classes": list(parse_csv_values(args.failure_classes)),
        "case_names": [str(item.case["name"]) for item in selection.cases],
        "n_cases": len(selection.cases),
        "sibling_alpha": float(DEFAULT_SIBLING_ALPHA),
        "max_contributors": int(args.max_contributors),
        "contributors_for_all_crossings": bool(args.contributors_for_all_crossings),
        "elapsed_sec": round(elapsed_sec, 6),
        "targets_csv": str(targets_csv),
        "contributors_csv": str(contributors_csv),
        "summary_csv": str(summary_csv),
    }
    metadata_json.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    print(f"Sibling inflation diagnostic complete in {elapsed_sec:.2f}s")
    print(f"Targets: {targets_csv}")
    print(f"Contributors: {contributors_csv}")
    print(f"Summary: {summary_csv}")
    print(f"Metadata: {metadata_json}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
