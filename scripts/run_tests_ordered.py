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
            "tests/core",
            "tests/tree",
        ),
    ),
    Stage(
        2,
        "Statistical engines + calibration",
        (
            "tests/statistics",
        ),
    ),
    Stage(
        3,
        "Localization + post-hoc merge behavior",
        (
            "tests/localization",
        ),
    ),
    Stage(
        4,
        "Validation, calibration + diagnostic studies",
        (
            "tests/validation",
        ),
    ),
    Stage(
        5,
        "Pipeline + application contracts and reporting artifacts",
        (
            "tests/pipeline",
            "tests/applications",
        ),
    ),
    Stage(
        6,
        "Integration smoke + visualization",
        (
            "tests/integration",
            "tests/visualization",
        ),
    ),
    Stage(
        7,
        "Wiki memory contracts",
        (
            "tests/wiki",
        ),
    ),
)


def run_pytest(test_files: tuple[str, ...]) -> int:
    cmd = [sys.executable, "-m", "pytest", *test_files]
    completed = subprocess.run(cmd)
    return completed.returncode


def list_stages() -> None:
    for stage in STAGES:
        target_count = len(stage.tests)
        target_label = "target" if target_count == 1 else "targets"
        print(f"{stage.index}. {stage.title} ({target_count} test {target_label})")


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
