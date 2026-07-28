"""AWS Batch wrapper for traversal-aligned sibling FDR null diagnostics."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import pandas as pd

from benchmarks.shared.env import resolve_aws_batch_shard_index as resolve_shard_index
from benchmarks.validation.statistics.traversal_sibling_fdr_null import (
    SIMULATION_OUTPUT_NAME,
    SUMMARY_OUTPUT_NAME,
    FdrLayer,
    TraversalSiblingFdrConfig,
    classify_fdr_outcome,
    estimate_fdr,
    parse_names,
    run_traversal_sibling_fdr_layers,
)

SHARD_MANIFEST_NAME = "aws_traversal_sibling_fdr_shard_manifest.json"
COMBINED_MANIFEST_NAME = "aws_traversal_sibling_fdr_manifest.json"
AWS_ROLE = "aws_distributed_traversal_sibling_fdr_null_diagnostic"


@dataclass(frozen=True)
class AwsTraversalSiblingFdrConfig:
    """Cloud execution contract for one traversal sibling-FDR layer."""

    layer: FdrLayer
    case_names: tuple[str, ...]
    output_dir: Path
    replicates: int
    alpha: float
    base_seed: int
    shard_count: int
    suite: str = "binary"
    edge_alpha: float = 0.001
    s3_uri: str | None = None


@dataclass(frozen=True)
class TraversalSiblingFdrShardSpec:
    """Concrete shard assignment."""

    shard_index: int
    shard_count: int
    output_dir: Path
    replicate_indices: tuple[int, ...]


def validate_shard_contract(*, shard_index: int, shard_count: int) -> None:
    """Validate shard dimensions before launching compute."""
    if shard_count <= 0:
        raise ValueError(f"shard_count must be positive; got {shard_count!r}.")
    if not 0 <= shard_index < shard_count:
        raise ValueError(
            f"shard_index must satisfy 0 <= index < shard_count; got "
            f"{shard_index!r} with shard_count={shard_count!r}."
        )


def make_shard_spec(
    configured: AwsTraversalSiblingFdrConfig,
    shard_index: int,
) -> TraversalSiblingFdrShardSpec:
    """Assign replicate indices to one shard by modulo partition."""
    validate_shard_contract(shard_index=shard_index, shard_count=configured.shard_count)
    replicate_indices = tuple(
        index
        for index in range(configured.replicates)
        if index % configured.shard_count == shard_index
    )
    if not replicate_indices:
        raise ValueError(
            f"Shard {shard_index} has no replicate indices. "
            f"Use shard_count <= {configured.replicates}."
        )
    return TraversalSiblingFdrShardSpec(
        shard_index=int(shard_index),
        shard_count=int(configured.shard_count),
        output_dir=configured.output_dir / "shards" / f"shard_{shard_index:04d}",
        replicate_indices=replicate_indices,
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


def _validation_config(configured: AwsTraversalSiblingFdrConfig) -> TraversalSiblingFdrConfig:
    return TraversalSiblingFdrConfig(
        layer=configured.layer,
        suite=configured.suite,
        case_names=configured.case_names,
        replicates=configured.replicates,
        alpha=configured.alpha,
        edge_alpha=configured.edge_alpha,
        base_seed=configured.base_seed,
    )


def run_shard(configured: AwsTraversalSiblingFdrConfig, shard_index: int) -> dict[str, object]:
    """Run one AWS Batch shard."""
    spec = make_shard_spec(configured, shard_index)
    started = perf_counter()
    source_manifest = run_traversal_sibling_fdr_layers(
        [_validation_config(configured)],
        output_dir=spec.output_dir,
        replicate_indices=spec.replicate_indices,
    )
    shard_manifest = {
        "runner": "benchmarks.cloud.aws_traversal_sibling_fdr_null",
        "execution_role": AWS_ROLE,
        "shard_index": int(spec.shard_index),
        "shard_count": int(spec.shard_count),
        "replicate_indices": list(spec.replicate_indices),
        "layer": str(configured.layer),
        "suite": configured.suite,
        "case_names": list(configured.case_names),
        "alpha": float(configured.alpha),
        "edge_alpha": float(configured.edge_alpha),
        "replicates": int(configured.replicates),
        "base_seed": int(configured.base_seed),
        "elapsed_sec": round(float(perf_counter() - started), 6),
        "source_manifest": source_manifest,
        "note": "Shard output is diagnostic evidence only; merge before interpretation.",
    }
    _write_json(spec.output_dir / SHARD_MANIFEST_NAME, shard_manifest)
    if configured.s3_uri is not None:
        sync_path_to_s3(
            spec.output_dir,
            f"{configured.s3_uri.rstrip('/')}/shards/shard_{shard_index:04d}",
        )
    return shard_manifest


def iter_expected_shard_dirs(output_dir: Path, shard_count: int) -> tuple[Path, ...]:
    """Return canonical shard directories for a completed array job."""
    if shard_count <= 0:
        raise ValueError(f"shard_count must be positive; got {shard_count!r}.")
    return tuple(output_dir / "shards" / f"shard_{index:04d}" for index in range(shard_count))


def _load_shard_manifest(shard_dir: Path) -> dict[str, object]:
    path = shard_dir / SHARD_MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"Missing traversal sibling FDR shard manifest: {path}")
    return json.loads(path.read_text())


def _load_shard_table(shard_dir: Path, filename: str) -> pd.DataFrame:
    path = shard_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing traversal sibling FDR shard table: {path}")
    return pd.read_csv(path)


def _validate_replicate_coverage(
    manifests: list[dict[str, object]],
    *,
    expected_replicates: int,
) -> None:
    observed: list[int] = []
    for manifest in manifests:
        observed.extend(int(index) for index in manifest["replicate_indices"])
    duplicated = sorted(index for index in set(observed) if observed.count(index) > 1)
    if duplicated:
        raise ValueError(f"Duplicate replicate index assignment after merge: {duplicated!r}")
    missing = sorted(set(range(expected_replicates)) - set(observed))
    if missing:
        raise ValueError(f"Missing replicate index assignment after merge: {missing!r}")


def _validate_simulation_coverage(
    simulations: pd.DataFrame,
    configured: AwsTraversalSiblingFdrConfig,
) -> None:
    expected = {
        (str(configured.layer), case_id, replicate_index)
        for case_id in configured.case_names
        for replicate_index in range(configured.replicates)
    }
    observed = {
        (str(row.layer), str(row.case_id), int(row.replicate_index))
        for row in simulations.itertuples(index=False)
    }
    missing = sorted(expected - observed)
    if missing:
        raise ValueError(f"Missing traversal sibling FDR simulation rows: {missing[:10]!r}")
    duplicated = simulations.duplicated(
        subset=["layer", "case_id", "replicate_index"],
        keep=False,
    )
    if duplicated.any():
        duplicate_rows = simulations.loc[
            duplicated,
            ["layer", "case_id", "replicate_index"],
        ]
        raise ValueError(
            "Duplicate traversal sibling FDR simulation rows: "
            f"{duplicate_rows.head(10).to_dict(orient='records')!r}"
        )


def _combined_summary(
    simulations: pd.DataFrame,
    configured: AwsTraversalSiblingFdrConfig,
) -> pd.DataFrame:
    rows = simulations.to_dict(orient="records")
    summary = estimate_fdr(rows)
    summary.update(
        {
            "schema_version": "traversal_sibling_fdr_null/v1",
            "layer": str(configured.layer),
            "suite": configured.suite,
            "case_names": ",".join(configured.case_names),
            "alpha": float(configured.alpha),
            "edge_alpha": float(configured.edge_alpha),
            "base_seed": int(configured.base_seed),
            "outcome": classify_fdr_outcome(
                layer=configured.layer,
                mean_fdp=float(summary["mean_fdp"]),
                alpha=float(configured.alpha),
                n_ok=int(summary["n_ok"]),
                n_support_failures=int(summary["n_support_failures"]),
            ),
        }
    )
    return pd.DataFrame.from_records([summary])


def merge_shards(configured: AwsTraversalSiblingFdrConfig) -> dict[str, pd.DataFrame]:
    """Merge all expected traversal sibling FDR shard outputs."""
    if configured.s3_uri is not None:
        sync_s3_to_path(
            f"{configured.s3_uri.rstrip('/')}/shards",
            configured.output_dir / "shards",
        )
    shard_dirs = iter_expected_shard_dirs(configured.output_dir, configured.shard_count)
    manifests = [_load_shard_manifest(shard_dir) for shard_dir in shard_dirs]
    _validate_replicate_coverage(manifests, expected_replicates=configured.replicates)
    simulations = pd.concat(
        [_load_shard_table(shard_dir, SIMULATION_OUTPUT_NAME) for shard_dir in shard_dirs],
        ignore_index=True,
    )
    _validate_simulation_coverage(simulations, configured)
    summary = _combined_summary(simulations, configured)

    merged_dir = configured.output_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    simulations.to_csv(merged_dir / SIMULATION_OUTPUT_NAME, index=False)
    summary.to_csv(merged_dir / SUMMARY_OUTPUT_NAME, index=False)
    combined_manifest = {
        "runner": "benchmarks.cloud.aws_traversal_sibling_fdr_null",
        "execution_role": AWS_ROLE,
        "layer": str(configured.layer),
        "suite": configured.suite,
        "case_names": list(configured.case_names),
        "alpha": float(configured.alpha),
        "edge_alpha": float(configured.edge_alpha),
        "replicates": int(configured.replicates),
        "base_seed": int(configured.base_seed),
        "shard_count": int(configured.shard_count),
        "n_simulation_rows": int(len(simulations)),
        "n_summary_rows": int(len(summary)),
        "source_shard_dirs": [str(path) for path in shard_dirs],
        "note": "Merged traversal sibling FDR null diagnostic evidence only.",
    }
    _write_json(merged_dir / COMBINED_MANIFEST_NAME, combined_manifest)
    if configured.s3_uri is not None:
        sync_path_to_s3(merged_dir, f"{configured.s3_uri.rstrip('/')}/merged")
    return {"simulations": simulations, "summary": summary}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run-shard", "merge"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--output-dir", type=Path, required=True)
        sub.add_argument("--layer", default=FdrLayer.SYNTHETIC_VALID_P.value)
        sub.add_argument("--suite", default="binary")
        sub.add_argument("--case-names", required=True)
        sub.add_argument("--replicates", type=int, required=True)
        sub.add_argument("--alpha", type=float, default=0.01)
        sub.add_argument("--edge-alpha", type=float, default=0.001)
        sub.add_argument("--base-seed", type=int, default=20260604)
        sub.add_argument("--shard-count", type=int, required=True)
        sub.add_argument("--s3-uri")
        if command == "run-shard":
            sub.add_argument("--shard-index", type=int)
    return parser


def _configured_from_args(args: argparse.Namespace) -> AwsTraversalSiblingFdrConfig:
    if args.replicates <= 0:
        raise ValueError("replicates must be positive.")
    return AwsTraversalSiblingFdrConfig(
        output_dir=args.output_dir,
        layer=FdrLayer(str(args.layer)),
        suite=str(args.suite),
        case_names=parse_names(args.case_names),
        replicates=int(args.replicates),
        alpha=float(args.alpha),
        edge_alpha=float(args.edge_alpha),
        base_seed=int(args.base_seed),
        shard_count=int(args.shard_count),
        s3_uri=args.s3_uri,
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    configured = _configured_from_args(args)
    if args.command == "run-shard":
        shard_index = resolve_shard_index(args.shard_index)
        print(json.dumps(run_shard(configured, shard_index), indent=2, sort_keys=True))
        return 0
    if args.command == "merge":
        outputs = merge_shards(configured)
        print(
            json.dumps(
                {name: int(len(table)) for name, table in outputs.items()},
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    raise ValueError(f"Unsupported command: {args.command!r}.")


if __name__ == "__main__":
    raise SystemExit(main())
