#!/usr/bin/env python3
"""Run tests in documented purpose-based stages.

Usage:
  python scripts/run_tests_ordered.py            # all stages in order
  python scripts/run_tests_ordered.py --stage 1  # run a single stage
  python scripts/run_tests_ordered.py --list      # list stages
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Stage:
    index: int
    title: str
    tests: tuple[str, ...]


STAGES: tuple[Stage, ...] = (
    Stage(
        1,
        "Core structure + decomposition",
        (
            "tests/core/10_test_poset_tree.py",
            "tests/core/11_test_tree_decomposition_distance.py",
            "tests/core/12_test_cluster_assignments.py",
            "tests/core/13_test_cluster_decomposer_threshold.py",
            "tests/core/14_test_local_kl_utils.py",
        ),
    ),
    Stage(
        2,
        "Statistical engines + calibration",
        (
            "tests/statistics/20_test_clt_validity.py",
            "tests/statistics/21_test_random_projection.py",
            "tests/statistics/22_test_edge_branch_length_regression.py",
            "tests/statistics/23_test_weighted_calibration.py",
            "tests/statistics/24_test_weighted_calibration_diagnostic.py",
            "tests/statistics/25_test_per_test_projection_seeding.py",
            "tests/statistics/26_test_invalid_nonfinite_handling.py",
            "tests/statistics/27_test_categorical_distributions.py",
        ),
    ),
    Stage(
        3,
        "Localization + post-hoc merge behavior",
        (
            "tests/localization/30_test_signal_localization.py",
            "tests/localization/31_test_posthoc_merge.py",
            "tests/localization/32_test_posthoc_merge_calibration.py",
            "tests/localization/33_test_skip_reason_propagation_integration.py",
        ),
    ),
    Stage(
        4,
        "Cluster validation stack",
        (
            "tests/validation/40_test_cluster_validation_core.py",
            "tests/validation/41_test_independent_cluster_validation.py",
            "tests/validation/42_test_cluster_validation_integration.py",
            "tests/validation/43_test_result_status_validation.py",
        ),
    ),
    Stage(
        5,
        "Pipeline contracts + reporting artifacts",
        (
            "tests/pipeline/50_test_attention_pipeline.py",
            "tests/pipeline/51_test_dispatch_contract.py",
            "tests/pipeline/52_test_method_execution_index_alignment.py",
            "tests/pipeline/53_test_runner_contract_alignment.py",
            "tests/pipeline/54_test_compare_sibling_methods_contract.py",
            "tests/pipeline/55_test_case_run_audit_env_restore.py",
            "tests/pipeline/56_test_run_weighted_full_import_safety.py",
            "tests/pipeline/57_test_pdf_utils.py",
            "tests/pipeline/58_test_pipeline_pdf_behavior.py",
            "tests/pipeline/59_test_pipeline_pdf_naming.py",
        ),
    ),
    Stage(
        6,
        "Integration smoke + visualization",
        (
            "tests/integration/60_test_benchmark_methods_smoke.py",
            "tests/integration/61_test_sbm_integration.py",
            "tests/integration/62_test_phylogenetic_generator.py",
            "tests/visualization/70_test_cluster_tree_layout.py",
            "tests/visualization/71_test_cluster_tree_visualization.py",
        ),
    ),
)


def run_pytest(test_files: tuple[str, ...]) -> int:
    cmd = ["pytest", *test_files]
    completed = subprocess.run(cmd)
    return completed.returncode


def list_stages() -> None:
    for stage in STAGES:
        print(f"{stage.index}. {stage.title} ({len(stage.tests)} files)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ordered test stages.")
    parser.add_argument("--stage", type=int, help="Run only one stage index.")
    parser.add_argument("--list", action="store_true", help="List available stages.")
    args = parser.parse_args()

    if args.list:
        list_stages()
        return 0

    if args.stage is not None:
        matching = [stage for stage in STAGES if stage.index == args.stage]
        if not matching:
            print(f"Unknown stage: {args.stage}", file=sys.stderr)
            return 2
        stage = matching[0]
        print(f"\n== Stage {stage.index}: {stage.title} ==")
        return run_pytest(stage.tests)

    for stage in STAGES:
        print(f"\n== Stage {stage.index}: {stage.title} ==")
        code = run_pytest(stage.tests)
        if code != 0:
            print(f"Stopped at stage {stage.index} with exit code {code}.", file=sys.stderr)
            return code

    print("\nAll ordered stages completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
