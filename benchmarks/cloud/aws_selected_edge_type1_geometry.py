"""AWS Batch wrapper for selected-edge Type-I geometry diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import pandas as pd

from benchmarks.validation.selected_edge_type1_geometry import (
    DEFAULT_EDGE_ALPHA_GRID,
    DEFAULT_MODES,
    SelectedEdgeGeometryConfig,
    parse_alpha_grid,
    parse_names,
    run_selected_edge_geometry_for_replicates,
    validate_modes,
)

SHARD_MANIFEST_NAME = "aws_selected_edge_geometry_shard_manifest.json"
COMBINED_MANIFEST_NAME = "aws_selected_edge_geometry_manifest.json"
EDGE_ROWS_NAME = "selected_edge_geometry_edges.csv"
SIBLING_ROWS_NAME = "selected_edge_geometry_siblings.csv"
FINAL_ROWS_NAME = "selected_edge_geometry_final.csv"
AWS_ROLE = "aws_distributed_selected_edge_type1_geometry_diagnostic"


@dataclass(frozen=True)
class AwsSelectedEdgeGeometryConfig:
    """Cloud execution contract for selected-edge geometry diagnostics."""

    output_dir: Path
    suite: str
    case_names: tuple[str, ...]
    modes: tuple[str, ...]
    edge_alphas: tuple[float, ...]
    sibling_alpha: float
    replicates: int
    base_seed: int
    shard_count: int
    s3_uri: str | None = None
    resume: bool = True


@dataclass(frozen=True)
class SelectedEdgeShardSpec:
    """Concrete shard assignment."""

    shard_index: int
    shard_count: int
    output_dir: Path
    replicate_indices: tuple[int, ...]


def validate_shard_contract(*, shard_index: int, shard_count: int) -> None:
    if shard_count <= 0:
        raise ValueError(f"shard_count must be positive; got {shard_count!r}.")
    if not 0 <= shard_index < shard_count:
        raise ValueError(
            f"shard_index must satisfy 0 <= index < shard_count; got "
            f"{shard_index!r} with shard_count={shard_count!r}."
        )


def resolve_shard_index(explicit_index: int | None, environ: dict[str, str] | None = None) -> int:
    """Resolve shard index from CLI or AWS Batch array environment."""
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


def make_shard_spec(
    configured: AwsSelectedEdgeGeometryConfig,
    shard_index: int,
) -> SelectedEdgeShardSpec:
    """Assign replicate indices to one shard by modulo partition."""
    validate_shard_contract(shard_index=shard_index, shard_count=configured.shard_count)
    replicate_indices = tuple(
        index for index in range(configured.replicates) if index % configured.shard_count == shard_index
    )
    if not replicate_indices:
        raise ValueError(
            f"Shard {shard_index} has no replicate indices. "
            f"Use shard_count <= {configured.replicates}."
        )
    return SelectedEdgeShardSpec(
        shard_index=int(shard_index),
        shard_count=int(configured.shard_count),
        output_dir=configured.output_dir / "shards" / f"shard_{shard_index:04d}",
        replicate_indices=replicate_indices,
    )


def sync_path_to_s3(local_path: Path, s3_uri: str) -> None:
    subprocess.run(("aws", "s3", "sync", str(local_path), s3_uri), check=True)


def sync_s3_to_path(s3_uri: str, local_path: Path) -> None:
    local_path.mkdir(parents=True, exist_ok=True)
    subprocess.run(("aws", "s3", "sync", s3_uri, str(local_path)), check=True)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _validation_config(
    configured: AwsSelectedEdgeGeometryConfig,
    output_dir: Path,
) -> SelectedEdgeGeometryConfig:
    return SelectedEdgeGeometryConfig(
        output_dir=output_dir,
        suite=configured.suite,
        case_names=configured.case_names,
        modes=configured.modes,
        edge_alphas=configured.edge_alphas,
        sibling_alpha=configured.sibling_alpha,
        replicates=configured.replicates,
        base_seed=configured.base_seed,
    )


def run_shard(configured: AwsSelectedEdgeGeometryConfig, shard_index: int) -> dict[str, object]:
    """Run one selected-edge geometry shard."""
    spec = make_shard_spec(configured, shard_index)
    started = perf_counter()
    manifest = run_selected_edge_geometry_for_replicates(
        _validation_config(configured, spec.output_dir),
        replicate_indices=spec.replicate_indices,
    )
    shard_manifest = {
        "runner": "benchmarks.cloud.aws_selected_edge_type1_geometry",
        "execution_role": AWS_ROLE,
        "shard_index": int(spec.shard_index),
        "shard_count": int(spec.shard_count),
        "replicate_indices": list(spec.replicate_indices),
        "suite": configured.suite,
        "case_names": list(configured.case_names),
        "modes": list(configured.modes),
        "edge_alphas": list(configured.edge_alphas),
        "sibling_alpha": float(configured.sibling_alpha),
        "replicates": int(configured.replicates),
        "base_seed": int(configured.base_seed),
        "elapsed_sec": round(float(perf_counter() - started), 6),
        "source_manifest": manifest,
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
    if shard_count <= 0:
        raise ValueError(f"shard_count must be positive; got {shard_count!r}.")
    return tuple(output_dir / "shards" / f"shard_{index:04d}" for index in range(shard_count))


def _load_shard_table(shard_dir: Path, filename: str) -> pd.DataFrame:
    path = shard_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing selected-edge shard table: {path}")
    return pd.read_csv(path)


def _load_shard_manifest(shard_dir: Path) -> dict[str, object]:
    path = shard_dir / SHARD_MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"Missing selected-edge shard manifest: {path}")
    return json.loads(path.read_text())


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


def merge_shards(configured: AwsSelectedEdgeGeometryConfig) -> dict[str, pd.DataFrame]:
    """Merge all selected-edge geometry shard outputs."""
    if configured.s3_uri is not None:
        sync_s3_to_path(
            f"{configured.s3_uri.rstrip('/')}/shards",
            configured.output_dir / "shards",
        )
    shard_dirs = iter_expected_shard_dirs(configured.output_dir, configured.shard_count)
    manifests = [_load_shard_manifest(shard_dir) for shard_dir in shard_dirs]
    _validate_replicate_coverage(manifests, expected_replicates=configured.replicates)

    edges = pd.concat(
        [_load_shard_table(shard_dir, EDGE_ROWS_NAME) for shard_dir in shard_dirs],
        ignore_index=True,
    )
    siblings = pd.concat(
        [_load_shard_table(shard_dir, SIBLING_ROWS_NAME) for shard_dir in shard_dirs],
        ignore_index=True,
    )
    final = pd.concat(
        [_load_shard_table(shard_dir, FINAL_ROWS_NAME) for shard_dir in shard_dirs],
        ignore_index=True,
    )
    merged_dir = configured.output_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    edges.to_csv(merged_dir / EDGE_ROWS_NAME, index=False)
    siblings.to_csv(merged_dir / SIBLING_ROWS_NAME, index=False)
    final.to_csv(merged_dir / FINAL_ROWS_NAME, index=False)
    combined_manifest = {
        "runner": "benchmarks.cloud.aws_selected_edge_type1_geometry",
        "execution_role": AWS_ROLE,
        "suite": configured.suite,
        "case_names": list(configured.case_names),
        "modes": list(configured.modes),
        "edge_alphas": list(configured.edge_alphas),
        "sibling_alpha": float(configured.sibling_alpha),
        "replicates": int(configured.replicates),
        "base_seed": int(configured.base_seed),
        "shard_count": int(configured.shard_count),
        "n_edge_rows": int(len(edges)),
        "n_sibling_rows": int(len(siblings)),
        "n_final_rows": int(len(final)),
        "source_shard_dirs": [str(path) for path in shard_dirs],
        "note": "Merged selected-edge Type-I geometry diagnostic evidence only.",
    }
    _write_json(merged_dir / COMBINED_MANIFEST_NAME, combined_manifest)
    if configured.s3_uri is not None:
        sync_path_to_s3(merged_dir, f"{configured.s3_uri.rstrip('/')}/merged")
    return {"edges": edges, "siblings": siblings, "final": final}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run-shard", "merge"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--output-dir", type=Path, required=True)
        sub.add_argument("--suite", default="binary")
        sub.add_argument("--case-names")
        sub.add_argument("--modes", default=",".join(DEFAULT_MODES))
        sub.add_argument(
            "--edge-alphas",
            default=",".join(str(value) for value in DEFAULT_EDGE_ALPHA_GRID),
        )
        sub.add_argument("--sibling-alpha", type=float, default=0.01)
        sub.add_argument("--replicates", type=int, default=10)
        sub.add_argument("--base-seed", type=int, default=20260604)
        sub.add_argument("--shard-count", type=int, required=True)
        sub.add_argument("--s3-uri")
        sub.add_argument("--resume", action="store_true", default=True)
        if command == "run-shard":
            sub.add_argument("--shard-index", type=int)
    return parser


def _configured_from_args(args: argparse.Namespace) -> AwsSelectedEdgeGeometryConfig:
    if args.replicates <= 0:
        raise ValueError("replicates must be positive.")
    return AwsSelectedEdgeGeometryConfig(
        output_dir=args.output_dir,
        suite=str(args.suite),
        case_names=parse_names(args.case_names),
        modes=validate_modes(parse_names(args.modes)),
        edge_alphas=parse_alpha_grid(str(args.edge_alphas)),
        sibling_alpha=float(args.sibling_alpha),
        replicates=int(args.replicates),
        base_seed=int(args.base_seed),
        shard_count=int(args.shard_count),
        s3_uri=args.s3_uri,
        resume=bool(args.resume),
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
