"""AWS Batch runner for the selected-tail equation study.

This module only distributes the diagnostic workload. It does not add a
calibration fallback or a production external-null model. Shards write raw
selected-hierarchy geometry records; the merge step recomputes every summary
from the combined row-level records so support counts and held-out tail-law
checks remain tied to explicit independent simulation ids.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import pandas as pd
from tree_break_selection.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_EDGE_ALPHA,
    DEFAULT_SIBLING_ALPHA,
)

from benchmarks.diagnostics.calibration.selected_hierarchy_geometry_covariates import (
    CANDIDATE_EQUATIONS,
    EDGE_ACTION_BIN_LABELS,
    EDGE_ACTION_BINS,
    MODEL_BLOCKS,
    RESPONSE_COLUMN,
    SIMULATION_ID_COLUMN,
    STUDY_ROLE,
    TAIL_LAW_CONTEXT_COLUMNS,
    TAIL_LAW_ROLE,
    evaluate_candidate_equation_holdout,
    evaluate_candidate_equations,
    evaluate_covariate_block_models,
    evaluate_covariate_relationships,
    evaluate_selected_ratio_tail_law,
    run_selected_hierarchy_geometry_covariate_study,
    summarize_selected_geometry_by_case,
)
from benchmarks.diagnostics.calibration.selected_hierarchy_null_audit import DEFAULT_CASE_NAMES
from benchmarks.shared.env import resolve_aws_batch_shard_index as resolve_shard_index

AWS_SELECTED_TAIL_STUDY_ROLE = "aws_distributed_selected_tail_equation_diagnostic"
SHARD_RECORDS_NAME = "selected_geometry_records.csv"
SHARD_MANIFEST_NAME = "aws_shard_manifest.json"
COMBINED_MANIFEST_NAME = "aws_selected_tail_equation_study_manifest.json"


@dataclass(frozen=True)
class AwsSelectedTailStudyConfig:
    """Cloud execution contract for one selected-tail study."""

    case_names: tuple[str, ...]
    output_dir: Path
    base_seed: int
    shard_count: int
    replicates_per_shard: int
    s3_uri: str | None = None


@dataclass(frozen=True)
class ShardSpec:
    """Concrete shard parameters after resolving AWS Batch array state."""

    shard_index: int
    shard_count: int
    seed: int
    output_dir: Path


def parse_case_names(raw_case_names: str) -> tuple[str, ...]:
    """Parse and validate the comma-separated benchmark case list."""
    case_names = tuple(
        item.strip()
        for item in str(raw_case_names).split(",")
        if item.strip()
    )
    if not case_names:
        raise ValueError("At least one case name is required.")
    return case_names


def shard_seed(*, base_seed: int, shard_index: int) -> int:
    """Return the deterministic seed assigned to one shard."""
    if shard_index < 0:
        raise ValueError(f"shard_index must be non-negative; got {shard_index!r}.")
    return int(base_seed) + int(shard_index) * 1_000_000_000


def validate_shard_contract(*, shard_index: int, shard_count: int, replicates_per_shard: int) -> None:
    """Validate shard dimensions before launching compute."""
    if shard_count <= 0:
        raise ValueError(f"shard_count must be positive; got {shard_count!r}.")
    if replicates_per_shard <= 0:
        raise ValueError(
            f"replicates_per_shard must be positive; got {replicates_per_shard!r}."
        )
    if not 0 <= shard_index < shard_count:
        raise ValueError(
            f"shard_index must satisfy 0 <= index < shard_count; got "
            f"{shard_index!r} with shard_count={shard_count!r}."
        )


def make_shard_spec(configured: AwsSelectedTailStudyConfig, shard_index: int) -> ShardSpec:
    """Build the concrete shard output path and seed."""
    validate_shard_contract(
        shard_index=shard_index,
        shard_count=configured.shard_count,
        replicates_per_shard=configured.replicates_per_shard,
    )
    return ShardSpec(
        shard_index=int(shard_index),
        shard_count=int(configured.shard_count),
        seed=shard_seed(base_seed=configured.base_seed, shard_index=shard_index),
        output_dir=configured.output_dir / "shards" / f"shard_{shard_index:04d}",
    )


def current_git_state() -> dict[str, object]:
    """Capture reproducibility state without failing the benchmark."""
    state: dict[str, object] = {
        "build_commit": os.environ.get("TBS_GIT_COMMIT", "unknown"),
        "build_branch": os.environ.get("TBS_GIT_BRANCH", "unknown"),
    }
    commands = {
        "commit": ("git", "rev-parse", "HEAD"),
        "branch": ("git", "branch", "--show-current"),
        "status_short": ("git", "status", "--short"),
    }
    for key, command in commands.items():
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            state[key] = f"unavailable:{exc}"
        else:
            state[key] = completed.stdout.strip()
    return state


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sync_path_to_s3(local_path: Path, s3_uri: str) -> None:
    """Sync a local shard or merged output directory to S3 with the AWS CLI."""
    subprocess.run(
        ("aws", "s3", "sync", str(local_path), s3_uri),
        check=True,
    )


def sync_s3_to_path(s3_uri: str, local_path: Path) -> None:
    """Sync S3 shard outputs into the merge job's local filesystem."""
    local_path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ("aws", "s3", "sync", s3_uri, str(local_path)),
        check=True,
    )


def run_shard(configured: AwsSelectedTailStudyConfig, shard_index: int) -> dict[str, pd.DataFrame]:
    """Run one AWS Batch array shard."""
    spec = make_shard_spec(configured, shard_index)
    started_at = perf_counter()
    outputs = run_selected_hierarchy_geometry_covariate_study(
        case_names=list(configured.case_names),
        output_dir=spec.output_dir,
        n_replicates=int(configured.replicates_per_shard),
        seed=int(spec.seed),
        write_selected_records=True,
    )
    manifest = {
        "runner": "benchmarks.cloud.aws_selected_tail_equation_study",
        "execution_role": AWS_SELECTED_TAIL_STUDY_ROLE,
        "diagnostic_role": STUDY_ROLE,
        "shard_index": int(spec.shard_index),
        "shard_count": int(spec.shard_count),
        "base_seed": int(configured.base_seed),
        "shard_seed": int(spec.seed),
        "replicates_per_shard": int(configured.replicates_per_shard),
        "case_names": list(configured.case_names),
        "selected_records_path": str(spec.output_dir / SHARD_RECORDS_NAME),
        "elapsed_sec": round(float(perf_counter() - started_at), 6),
        "git": current_git_state(),
        "note": (
            "Shard output is row-level diagnostic evidence only. Combine shards "
            "with the merge command before interpreting tail-law support."
        ),
    }
    _write_json(spec.output_dir / SHARD_MANIFEST_NAME, manifest)
    if configured.s3_uri is not None:
        sync_path_to_s3(spec.output_dir, f"{configured.s3_uri.rstrip('/')}/shards/shard_{shard_index:04d}")
    return outputs


def iter_expected_shard_dirs(output_dir: Path, shard_count: int) -> tuple[Path, ...]:
    """Return the canonical shard directories for a completed array job."""
    if shard_count <= 0:
        raise ValueError(f"shard_count must be positive; got {shard_count!r}.")
    return tuple(output_dir / "shards" / f"shard_{index:04d}" for index in range(shard_count))


def load_shard_records(shard_dir: Path) -> pd.DataFrame:
    """Load one shard's row-level records and namespace its simulation ids."""
    records_path = shard_dir / SHARD_RECORDS_NAME
    manifest_path = shard_dir / SHARD_MANIFEST_NAME
    if not records_path.exists():
        raise FileNotFoundError(f"Missing shard records: {records_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing shard manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    shard_index = int(manifest["shard_index"])
    shard_seed_value = int(manifest["shard_seed"])
    run_id = f"shard_{shard_index:04d}"
    records = pd.read_csv(records_path)
    if SIMULATION_ID_COLUMN not in records.columns:
        raise KeyError(
            f"Shard records {records_path} must contain {SIMULATION_ID_COLUMN!r}."
        )
    records = records.copy()
    records.insert(0, "aws_shard_id", run_id)
    records.insert(1, "aws_shard_index", shard_index)
    records.insert(2, "aws_shard_seed", shard_seed_value)
    records[SIMULATION_ID_COLUMN] = [
        f"{run_id}:{simulation_id}"
        for simulation_id in records[SIMULATION_ID_COLUMN].astype(str)
    ]
    return records


def load_combined_shard_records(shard_dirs: Iterable[Path]) -> pd.DataFrame:
    """Concatenate row-level selected-geometry records across shards."""
    record_tables: list[pd.DataFrame] = []
    seen_shard_ids: set[str] = set()
    for shard_dir in shard_dirs:
        table = load_shard_records(shard_dir)
        shard_ids = set(table["aws_shard_id"].astype(str).unique())
        if len(shard_ids) != 1:
            raise ValueError(f"Shard records {shard_dir} contain multiple shard ids: {shard_ids!r}.")
        shard_id = next(iter(shard_ids))
        if shard_id in seen_shard_ids:
            raise ValueError(f"Duplicate shard id in merge input: {shard_id!r}.")
        seen_shard_ids.add(shard_id)
        record_tables.append(table)
    if not record_tables:
        raise ValueError("At least one shard directory is required.")
    return pd.concat(record_tables, ignore_index=True)


def recompute_outputs_from_records(records: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Recompute every combined diagnostic table from row-level records."""
    return {
        "geometry_summary_by_case": summarize_selected_geometry_by_case(records),
        "covariate_relationships": evaluate_covariate_relationships(records),
        "covariate_block_models": evaluate_covariate_block_models(records),
        "candidate_equations": evaluate_candidate_equations(records),
        "candidate_equation_holdout": evaluate_candidate_equation_holdout(records),
        "selected_ratio_tail_law": evaluate_selected_ratio_tail_law(records),
        "selected_geometry_records": records,
    }


def write_combined_outputs(
    *,
    records: pd.DataFrame,
    output_dir: Path,
    shard_dirs: Sequence[Path],
    configured: AwsSelectedTailStudyConfig,
) -> dict[str, pd.DataFrame]:
    """Write merged outputs and the cloud reproducibility manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = perf_counter()
    outputs = recompute_outputs_from_records(records)
    for name, table in outputs.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)

    manifest = {
        "runner": "benchmarks.cloud.aws_selected_tail_equation_study",
        "execution_role": AWS_SELECTED_TAIL_STUDY_ROLE,
        "diagnostic_role": STUDY_ROLE,
        "tail_law_role": TAIL_LAW_ROLE,
        "case_names": list(configured.case_names),
        "base_seed": int(configured.base_seed),
        "shard_count": int(configured.shard_count),
        "replicates_per_shard": int(configured.replicates_per_shard),
        "requested_total_replicates_per_case": int(
            configured.shard_count * configured.replicates_per_shard
        ),
        "n_selected_records": int(records.shape[0]),
        "independent_simulation_id_column": SIMULATION_ID_COLUMN,
        "n_independent_simulations": int(records[SIMULATION_ID_COLUMN].nunique()),
        "response_column": RESPONSE_COLUMN,
        "edge_alpha": float(DEFAULT_EDGE_ALPHA),
        "sibling_alpha": float(DEFAULT_SIBLING_ALPHA),
        "source_shard_dirs": [str(path) for path in shard_dirs],
        "outputs": {name: str(output_dir / f"{name}.csv") for name in outputs},
        "covariate_blocks": {
            block_name: list(predictors) for block_name, predictors in MODEL_BLOCKS
        },
        "candidate_equations": {
            equation_id: {
                "equation": equation,
                "predictors": list(predictors),
            }
            for equation_id, equation, predictors in CANDIDATE_EQUATIONS
        },
        "selected_ratio_tail_law": {
            "role": TAIL_LAW_ROLE,
            "context_columns": list(TAIL_LAW_CONTEXT_COLUMNS),
            "edge_action_bins": [
                {
                    "label": label,
                    "lower": float(lower),
                    "upper": None if upper == float("inf") else float(upper),
                }
                for lower, upper, label in zip(
                    EDGE_ACTION_BINS,
                    EDGE_ACTION_BINS[1:],
                    EDGE_ACTION_BIN_LABELS,
                )
            ],
            "alpha": float(DEFAULT_SIBLING_ALPHA),
            "production_min_matching_simulations": 499,
            "production_min_matched_records": 499,
            "production_max_exceedance_standard_error": 0.002,
        },
        "git": current_git_state(),
        "elapsed_sec": round(float(perf_counter() - started_at), 6),
        "note": (
            "Merged cloud output recomputes diagnostic summaries from combined "
            "row-level records. It remains descriptive evidence for selected-tail "
            "law discovery and does not define a production external calibration "
            "fallback."
        ),
    }
    _write_json(output_dir / COMBINED_MANIFEST_NAME, manifest)
    return outputs


def merge_shards(configured: AwsSelectedTailStudyConfig) -> dict[str, pd.DataFrame]:
    """Merge all expected shard directories for a selected-tail cloud run."""
    if configured.s3_uri is not None:
        sync_s3_to_path(
            f"{configured.s3_uri.rstrip('/')}/shards",
            configured.output_dir / "shards",
        )
    shard_dirs = iter_expected_shard_dirs(configured.output_dir, configured.shard_count)
    records = load_combined_shard_records(shard_dirs)
    merged_dir = configured.output_dir / "merged"
    outputs = write_combined_outputs(
        records=records,
        output_dir=merged_dir,
        shard_dirs=shard_dirs,
        configured=configured,
    )
    if configured.s3_uri is not None:
        sync_path_to_s3(merged_dir, f"{configured.s3_uri.rstrip('/')}/merged")
    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run or merge the AWS-sharded selected-tail equation diagnostic. "
            "This is research infrastructure, not a production calibration path."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_shared_arguments(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--case-names", default=",".join(DEFAULT_CASE_NAMES))
        subparser.add_argument("--output-dir", type=Path, required=True)
        subparser.add_argument("--base-seed", type=int, default=20260603)
        subparser.add_argument("--shard-count", type=int, required=True)
        subparser.add_argument("--replicates-per-shard", type=int, required=True)
        subparser.add_argument("--s3-uri", default=None)

    shard_parser = subparsers.add_parser("run-shard")
    add_shared_arguments(shard_parser)
    shard_parser.add_argument("--shard-index", type=int, default=None)

    merge_parser = subparsers.add_parser("merge")
    add_shared_arguments(merge_parser)

    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> AwsSelectedTailStudyConfig:
    return AwsSelectedTailStudyConfig(
        case_names=parse_case_names(str(args.case_names)),
        output_dir=Path(args.output_dir),
        base_seed=int(args.base_seed),
        shard_count=int(args.shard_count),
        replicates_per_shard=int(args.replicates_per_shard),
        s3_uri=args.s3_uri,
    )


def main() -> None:
    args = _parse_args()
    configured = _config_from_args(args)
    if args.command == "run-shard":
        shard_index = resolve_shard_index(args.shard_index)
        run_shard(configured, shard_index)
        print(f"Wrote shard {shard_index} to {make_shard_spec(configured, shard_index).output_dir}")
    elif args.command == "merge":
        merge_shards(configured)
        print(f"Wrote merged selected-tail study to {configured.output_dir / 'merged'}")
    else:
        raise ValueError(f"Unknown command: {args.command!r}.")


if __name__ == "__main__":
    main()


__all__ = [
    "AWS_SELECTED_TAIL_STUDY_ROLE",
    "AwsSelectedTailStudyConfig",
    "COMBINED_MANIFEST_NAME",
    "SHARD_MANIFEST_NAME",
    "SHARD_RECORDS_NAME",
    "ShardSpec",
    "iter_expected_shard_dirs",
    "load_combined_shard_records",
    "load_shard_records",
    "make_shard_spec",
    "merge_shards",
    "parse_case_names",
    "recompute_outputs_from_records",
    "resolve_shard_index",
    "run_shard",
    "shard_seed",
    "sync_s3_to_path",
    "validate_shard_contract",
    "write_combined_outputs",
]
