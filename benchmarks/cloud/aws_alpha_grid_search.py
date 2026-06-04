"""AWS Batch wrapper for the alpha-grid benchmark diagnostic.

This module distributes alpha-pair benchmark evaluations across AWS Batch array
jobs. It is only execution infrastructure for validation evidence; it does not
change production alpha defaults or add a statistical fallback.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import pandas as pd

from benchmarks.validation.alpha_grid_search import (
    DEFAULT_EDGE_ALPHA_GRID,
    DEFAULT_SIBLING_ALPHA_GRID,
    AlphaGridConfig,
    build_alpha_pairs,
    current_git_state,
    parse_float_grid,
    run_alpha_grid_search,
)

AWS_ALPHA_GRID_ROLE = "aws_distributed_alpha_grid_diagnostic"
SHARD_MANIFEST_NAME = "aws_alpha_grid_shard_manifest.json"
COMBINED_MANIFEST_NAME = "aws_alpha_grid_manifest.json"
SUMMARY_NAME = "alpha_grid_summary.csv"
RESULTS_NAME = "alpha_grid_results.csv"


@dataclass(frozen=True)
class AwsAlphaGridConfig:
    """Cloud execution contract for one alpha-grid run."""

    suite: str
    case_names: tuple[str, ...]
    output_dir: Path
    edge_alphas: tuple[float, ...]
    sibling_alphas: tuple[float, ...]
    shard_count: int
    s3_uri: str | None = None
    resume: bool = True


@dataclass(frozen=True)
class AlphaGridShardSpec:
    """Concrete shard parameters after resolving AWS Batch array state."""

    shard_index: int
    shard_count: int
    output_dir: Path
    alpha_pairs: tuple[tuple[float, float], ...]
    alpha_pair_indices: tuple[int, ...]


def parse_case_names(raw_case_names: str | None) -> tuple[str, ...]:
    """Parse optional comma-separated benchmark case names."""
    if raw_case_names is None:
        return ()
    return tuple(item.strip() for item in raw_case_names.split(",") if item.strip())


def validate_shard_contract(*, shard_index: int, shard_count: int) -> None:
    """Validate shard dimensions before launching compute."""
    if shard_count <= 0:
        raise ValueError(f"shard_count must be positive; got {shard_count!r}.")
    if not 0 <= shard_index < shard_count:
        raise ValueError(
            f"shard_index must satisfy 0 <= index < shard_count; got "
            f"{shard_index!r} with shard_count={shard_count!r}."
        )


def resolve_shard_index(explicit_index: int | None, environ: dict[str, str] | None = None) -> int:
    """Resolve the zero-based shard index from CLI or AWS Batch environment."""
    if explicit_index is not None:
        if explicit_index < 0:
            raise ValueError(f"shard_index must be non-negative; got {explicit_index!r}.")
        return int(explicit_index)

    environment = os.environ if environ is None else environ
    raw_index = environment.get("AWS_BATCH_JOB_ARRAY_INDEX")
    if raw_index is None:
        raise ValueError(
            "Shard index is required. Pass --shard-index outside AWS Batch, or "
            "run as an AWS Batch array job with AWS_BATCH_JOB_ARRAY_INDEX."
        )
    shard_index = int(raw_index)
    if shard_index < 0:
        raise ValueError(f"AWS_BATCH_JOB_ARRAY_INDEX must be non-negative; got {raw_index!r}.")
    return shard_index


def make_shard_spec(configured: AwsAlphaGridConfig, shard_index: int) -> AlphaGridShardSpec:
    """Build the concrete shard output path and assigned alpha-pair subset."""
    validate_shard_contract(
        shard_index=shard_index,
        shard_count=configured.shard_count,
    )
    all_pairs = build_alpha_pairs(configured.edge_alphas, configured.sibling_alphas)
    selected = tuple(
        (index, pair)
        for index, pair in enumerate(all_pairs)
        if index % configured.shard_count == shard_index
    )
    if not selected:
        raise ValueError(
            f"Shard {shard_index} has no alpha pairs. "
            f"Use shard_count <= {len(all_pairs)} for this grid."
        )
    return AlphaGridShardSpec(
        shard_index=int(shard_index),
        shard_count=int(configured.shard_count),
        output_dir=configured.output_dir / "shards" / f"shard_{shard_index:04d}",
        alpha_pairs=tuple(pair for _index, pair in selected),
        alpha_pair_indices=tuple(index for index, _pair in selected),
    )


def sync_path_to_s3(local_path: Path, s3_uri: str) -> None:
    """Sync a local output directory to S3 with the AWS CLI."""
    subprocess.run(("aws", "s3", "sync", str(local_path), s3_uri), check=True)


def sync_s3_to_path(s3_uri: str, local_path: Path) -> None:
    """Sync S3 outputs into the local filesystem."""
    local_path.mkdir(parents=True, exist_ok=True)
    subprocess.run(("aws", "s3", "sync", s3_uri, str(local_path)), check=True)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_shard(configured: AwsAlphaGridConfig, shard_index: int) -> dict[str, object]:
    """Run one AWS Batch array shard."""
    spec = make_shard_spec(configured, shard_index)
    started = perf_counter()
    outputs = run_alpha_grid_search(
        AlphaGridConfig(
            suite=configured.suite,
            case_names=configured.case_names,
            edge_alphas=configured.edge_alphas,
            sibling_alphas=configured.sibling_alphas,
            output_dir=spec.output_dir,
            resume=configured.resume,
            alpha_pairs=spec.alpha_pairs,
        )
    )
    manifest = {
        "runner": "benchmarks.cloud.aws_alpha_grid_search",
        "execution_role": AWS_ALPHA_GRID_ROLE,
        "shard_index": int(spec.shard_index),
        "shard_count": int(spec.shard_count),
        "alpha_pair_indices": list(spec.alpha_pair_indices),
        "alpha_pairs": [
            {"edge_alpha": edge_alpha, "sibling_alpha": sibling_alpha}
            for edge_alpha, sibling_alpha in spec.alpha_pairs
        ],
        "suite": configured.suite,
        "case_names": list(configured.case_names),
        "edge_alphas": list(configured.edge_alphas),
        "sibling_alphas": list(configured.sibling_alphas),
        "elapsed_sec": round(float(perf_counter() - started), 6),
        "git": current_git_state(),
        "note": (
            "Shard output is alpha-grid benchmark evidence only. Merge all shards "
            "before interpreting the full grid."
        ),
    }
    _write_json(spec.output_dir / SHARD_MANIFEST_NAME, manifest)
    if configured.s3_uri is not None:
        sync_path_to_s3(
            spec.output_dir,
            f"{configured.s3_uri.rstrip('/')}/shards/shard_{shard_index:04d}",
        )
    return outputs


def iter_expected_shard_dirs(output_dir: Path, shard_count: int) -> tuple[Path, ...]:
    """Return the canonical shard directories for a completed array job."""
    if shard_count <= 0:
        raise ValueError(f"shard_count must be positive; got {shard_count!r}.")
    return tuple(output_dir / "shards" / f"shard_{index:04d}" for index in range(shard_count))


def load_shard_summary(shard_dir: Path) -> pd.DataFrame:
    """Load one shard summary and validate its manifest."""
    manifest_path = shard_dir / SHARD_MANIFEST_NAME
    summary_path = shard_dir / SUMMARY_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing alpha-grid shard manifest: {manifest_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing alpha-grid shard summary: {summary_path}")
    manifest = json.loads(manifest_path.read_text())
    table = pd.read_csv(summary_path)
    table.insert(0, "aws_shard_index", int(manifest["shard_index"]))
    return table


def load_shard_results(shard_dir: Path) -> pd.DataFrame:
    """Load one shard's per-case benchmark result rows."""
    tables: list[pd.DataFrame] = []
    for result_path in sorted(shard_dir.glob("edge_*__sibling_*.csv")):
        tables.append(pd.read_csv(result_path))
    if not tables:
        raise FileNotFoundError(f"No alpha-pair result CSVs found in {shard_dir}")
    return pd.concat(tables, ignore_index=True)


def merge_shards(configured: AwsAlphaGridConfig) -> dict[str, pd.DataFrame]:
    """Merge all expected alpha-grid shard directories."""
    if configured.s3_uri is not None:
        sync_s3_to_path(
            f"{configured.s3_uri.rstrip('/')}/shards",
            configured.output_dir / "shards",
        )

    shard_dirs = iter_expected_shard_dirs(configured.output_dir, configured.shard_count)
    summary = pd.concat(
        [load_shard_summary(shard_dir) for shard_dir in shard_dirs],
        ignore_index=True,
    )
    duplicated = summary.duplicated(subset=["edge_alpha", "sibling_alpha"], keep=False)
    if duplicated.any():
        duplicate_rows = summary.loc[duplicated, ["edge_alpha", "sibling_alpha"]]
        raise ValueError(f"Duplicate alpha-pair summaries in shard outputs: {duplicate_rows!r}")

    expected_pairs = set(build_alpha_pairs(configured.edge_alphas, configured.sibling_alphas))
    observed_pairs = set(
        zip(summary["edge_alpha"].astype(float), summary["sibling_alpha"].astype(float))
    )
    missing_pairs = sorted(expected_pairs - observed_pairs)
    if missing_pairs:
        raise ValueError(f"Missing alpha-pair summaries after merge: {missing_pairs!r}")

    results = pd.concat(
        [load_shard_results(shard_dir) for shard_dir in shard_dirs],
        ignore_index=True,
    )
    merged_dir = configured.output_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    summary.sort_values(["edge_alpha", "sibling_alpha"]).to_csv(
        merged_dir / SUMMARY_NAME,
        index=False,
    )
    results.to_csv(merged_dir / RESULTS_NAME, index=False)
    manifest = {
        "runner": "benchmarks.cloud.aws_alpha_grid_search",
        "execution_role": AWS_ALPHA_GRID_ROLE,
        "suite": configured.suite,
        "case_names": list(configured.case_names),
        "edge_alphas": list(configured.edge_alphas),
        "sibling_alphas": list(configured.sibling_alphas),
        "shard_count": int(configured.shard_count),
        "n_alpha_pairs": int(len(expected_pairs)),
        "n_result_rows": int(results.shape[0]),
        "outputs": {
            "summary": str(merged_dir / SUMMARY_NAME),
            "results": str(merged_dir / RESULTS_NAME),
        },
        "source_shard_dirs": [str(path) for path in shard_dirs],
        "git": current_git_state(),
        "note": (
            "Merged output is diagnostic benchmark evidence for alpha constants. "
            "It does not prove selected-tree Type-I error control."
        ),
    }
    _write_json(merged_dir / COMBINED_MANIFEST_NAME, manifest)
    if configured.s3_uri is not None:
        sync_path_to_s3(merged_dir, f"{configured.s3_uri.rstrip('/')}/merged")
    return {"summary": summary, "results": results}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run or merge the AWS-sharded alpha-grid benchmark diagnostic. "
            "This is validation infrastructure, not a production method path."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_shared_arguments(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--suite", default="full")
        subparser.add_argument("--case-names", default=None)
        subparser.add_argument(
            "--edge-alphas",
            default=",".join(str(value) for value in DEFAULT_EDGE_ALPHA_GRID),
        )
        subparser.add_argument(
            "--sibling-alphas",
            default=",".join(str(value) for value in DEFAULT_SIBLING_ALPHA_GRID),
        )
        subparser.add_argument("--output-dir", type=Path, required=True)
        subparser.add_argument("--shard-count", type=int, required=True)
        subparser.add_argument("--s3-uri", default=None)

    shard_parser = subparsers.add_parser("run-shard")
    add_shared_arguments(shard_parser)
    shard_parser.add_argument("--shard-index", type=int, default=None)
    shard_parser.add_argument("--no-resume", action="store_true")

    merge_parser = subparsers.add_parser("merge")
    add_shared_arguments(merge_parser)

    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> AwsAlphaGridConfig:
    return AwsAlphaGridConfig(
        suite=str(args.suite),
        case_names=parse_case_names(args.case_names),
        output_dir=Path(args.output_dir),
        edge_alphas=parse_float_grid(str(args.edge_alphas)),
        sibling_alphas=parse_float_grid(str(args.sibling_alphas)),
        shard_count=int(args.shard_count),
        s3_uri=args.s3_uri,
        resume=not bool(getattr(args, "no_resume", False)),
    )


def main() -> None:
    args = _parse_args()
    configured = _config_from_args(args)
    if args.command == "run-shard":
        shard_index = resolve_shard_index(args.shard_index)
        run_shard(configured, shard_index)
        print(f"Wrote alpha-grid shard {shard_index} to {make_shard_spec(configured, shard_index).output_dir}")
    elif args.command == "merge":
        merge_shards(configured)
        print(f"Wrote merged alpha-grid study to {configured.output_dir / 'merged'}")
    else:
        raise ValueError(f"Unknown command: {args.command!r}.")


if __name__ == "__main__":
    main()


__all__ = [
    "AWS_ALPHA_GRID_ROLE",
    "AlphaGridShardSpec",
    "AwsAlphaGridConfig",
    "COMBINED_MANIFEST_NAME",
    "SHARD_MANIFEST_NAME",
    "RESULTS_NAME",
    "SUMMARY_NAME",
    "iter_expected_shard_dirs",
    "load_shard_results",
    "load_shard_summary",
    "make_shard_spec",
    "merge_shards",
    "parse_case_names",
    "resolve_shard_index",
    "run_shard",
    "sync_s3_to_path",
    "sync_path_to_s3",
    "validate_shard_contract",
]
