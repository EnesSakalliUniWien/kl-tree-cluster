"""AWS Batch wrapper for the tree-strategy semantic panel.

This runner is intentionally thin: it syncs precomputed diagnostic result CSVs
from S3, builds the semantic panel, and syncs the panel outputs back. It does
not rerun clustering and does not change production calibration.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import pandas as pd

from benchmarks.cloud.aws_alpha_grid_search import sync_path_to_s3, sync_s3_to_path
from benchmarks.cloud.aws_selected_tail_equation_study import current_git_state
from benchmarks.diagnostics.spectral.tree_strategy_semantic_panel import (
    DEFAULT_BLOB_RESULTS_DIR,
    DEFAULT_DIAGNOSTICS_RESULTS_DIR,
    SCHEMA_VERSION,
    build_semantic_panel,
    write_report,
)

AWS_TREE_STRATEGY_PANEL_ROLE = "aws_tree_strategy_semantic_panel"
MANIFEST_NAME = "aws_tree_strategy_semantic_panel_manifest.json"
PANEL_NAME = "tree_strategy_semantic_panel.csv"


@dataclass(frozen=True)
class AwsTreeStrategyPanelConfig:
    """Cloud execution contract for one semantic-panel run."""

    output_dir: Path
    input_s3_uri: str | None = None
    output_s3_uri: str | None = None
    input_work_dir: Path = Path("/tmp/tree-strategy-input")
    blob_results_dir: Path = DEFAULT_BLOB_RESULTS_DIR
    diagnostics_results_dir: Path = DEFAULT_DIAGNOSTICS_RESULTS_DIR
    alpha_summary_paths: tuple[Path, ...] = ()
    include_failed: bool = False


def _resolved_path(path: Path, input_work_dir: Path, *, use_synced_input: bool) -> Path:
    if path.is_absolute() or not use_synced_input:
        return path
    return input_work_dir / path


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_panel(config: AwsTreeStrategyPanelConfig) -> pd.DataFrame:
    started = perf_counter()
    if config.input_s3_uri is not None:
        sync_s3_to_path(config.input_s3_uri.rstrip("/"), config.input_work_dir)

    use_synced_input = config.input_s3_uri is not None
    blob_results_dir = _resolved_path(
        config.blob_results_dir,
        config.input_work_dir,
        use_synced_input=use_synced_input,
    )
    diagnostics_results_dir = _resolved_path(
        config.diagnostics_results_dir,
        config.input_work_dir,
        use_synced_input=use_synced_input,
    )
    alpha_summary_paths = [
        _resolved_path(path, config.input_work_dir, use_synced_input=use_synced_input)
        for path in config.alpha_summary_paths
    ]

    config.output_dir.mkdir(parents=True, exist_ok=True)
    panel = build_semantic_panel(
        blob_results_dir=blob_results_dir,
        diagnostics_results_dir=diagnostics_results_dir,
        alpha_summary_paths=alpha_summary_paths,
        include_failed=config.include_failed,
    )
    panel.to_csv(config.output_dir / PANEL_NAME, index=False)
    write_report(panel, config.output_dir)
    manifest = {
        "runner": "benchmarks.cloud.aws_tree_strategy_semantic_panel",
        "execution_role": AWS_TREE_STRATEGY_PANEL_ROLE,
        "schema_version": SCHEMA_VERSION,
        "input_s3_uri": config.input_s3_uri,
        "output_s3_uri": config.output_s3_uri,
        "input_work_dir": str(config.input_work_dir),
        "blob_results_dir": str(blob_results_dir),
        "diagnostics_results_dir": str(diagnostics_results_dir),
        "alpha_summary_paths": [str(path) for path in alpha_summary_paths],
        "include_failed": bool(config.include_failed),
        "n_rows": int(panel.shape[0]),
        "semantic_interpretation_counts": (
            panel["semantic_interpretation"].value_counts().sort_index().to_dict()
            if "semantic_interpretation" in panel.columns
            else {}
        ),
        "elapsed_sec": round(float(perf_counter() - started), 6),
        "outputs": {
            "panel": str(config.output_dir / PANEL_NAME),
            "report": str(config.output_dir / "tree_strategy_semantic_panel_report.md"),
        },
        "git": current_git_state(),
        "note": (
            "AWS output is a semantic diagnostic panel over precomputed result CSVs. "
            "It is not a production tree-selection or calibration rule."
        ),
    }
    _write_json(config.output_dir / MANIFEST_NAME, manifest)
    if config.output_s3_uri is not None:
        sync_path_to_s3(config.output_dir, config.output_s3_uri.rstrip("/"))
    return panel


def parse_alpha_summary_paths(raw: str | None) -> tuple[Path, ...]:
    if raw is None or not raw.strip():
        return ()
    return tuple(Path(item.strip()) for item in raw.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the tree-strategy semantic panel on AWS Batch."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--input-s3-uri", default=None)
    run_parser.add_argument("--output-s3-uri", default=None)
    run_parser.add_argument("--input-work-dir", type=Path, default=Path("/tmp/tree-strategy-input"))
    run_parser.add_argument("--blob-results-dir", type=Path, default=DEFAULT_BLOB_RESULTS_DIR)
    run_parser.add_argument(
        "--diagnostics-results-dir",
        type=Path,
        default=DEFAULT_DIAGNOSTICS_RESULTS_DIR,
    )
    run_parser.add_argument("--alpha-summary-paths", default=None)
    run_parser.add_argument("--include-failed", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> AwsTreeStrategyPanelConfig:
    return AwsTreeStrategyPanelConfig(
        output_dir=Path(args.output_dir),
        input_s3_uri=args.input_s3_uri,
        output_s3_uri=args.output_s3_uri,
        input_work_dir=Path(args.input_work_dir),
        blob_results_dir=Path(args.blob_results_dir),
        diagnostics_results_dir=Path(args.diagnostics_results_dir),
        alpha_summary_paths=parse_alpha_summary_paths(args.alpha_summary_paths),
        include_failed=bool(args.include_failed),
    )


def main() -> None:
    args = parse_args()
    if args.command != "run":
        raise ValueError(f"Unknown command: {args.command!r}.")
    config = config_from_args(args)
    run_panel(config)
    print(f"Wrote AWS tree strategy semantic panel to {config.output_dir}", flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "AWS_TREE_STRATEGY_PANEL_ROLE",
    "AwsTreeStrategyPanelConfig",
    "MANIFEST_NAME",
    "PANEL_NAME",
    "config_from_args",
    "parse_alpha_summary_paths",
    "run_panel",
]
