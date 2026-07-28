"""AWS Batch wrapper for KAK/cosine lens linkage alpha sweeps.

This module distributes diagnostic lens sweeps across AWS Batch array jobs.
Each shard owns one or more ``lens x linkage`` groups and runs the full alpha
grid for those groups. It is validation infrastructure only; it does not change
production tree construction or calibration.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter

import pandas as pd

from benchmarks.cloud.aws_alpha_grid_search import (
    sync_path_to_s3,
    sync_s3_to_path,
)
from benchmarks.cloud.aws_selected_tail_equation_study import current_git_state
from benchmarks.diagnostics.spectral.kak_lens_alpha_sweep import (
    DEFAULT_LENSES,
    LensSpec,
    parse_float_list,
    parse_lens,
    parse_linkage_methods,
)
from benchmarks.shared.env import resolve_aws_batch_shard_index as resolve_shard_index

AWS_KAK_LENS_SWEEP_ROLE = "aws_distributed_kak_lens_linkage_alpha_sweep"
SHARD_MANIFEST_NAME = "aws_kak_lens_linkage_alpha_sweep_shard_manifest.json"
COMBINED_MANIFEST_NAME = "aws_kak_lens_linkage_alpha_sweep_manifest.json"
SUMMARY_NAME = "kak_lens_linkage_alpha_sweep_summary.csv"

DEFAULT_EDGE_ALPHA_GRID = (0.0001, 0.0003, 0.001, 0.003, 0.01)
DEFAULT_SIBLING_ALPHA_GRID = (0.001, 0.003, 0.01, 0.03, 0.1)


@dataclass(frozen=True)
class LensLinkageGroup:
    """One reusable tree-geometry group for an alpha grid."""

    lens: LensSpec
    tree_linkage_method: str
    group_index: int

    @property
    def group_id(self) -> str:
        return f"{self.lens.lens_id}__{self.tree_linkage_method}"


@dataclass(frozen=True)
class AwsKakLensSweepConfig:
    """Cloud execution contract for one KAK/cosine lens sweep."""

    input_path: Path
    main_assignments: Path
    output_dir: Path
    edge_alphas: tuple[float, ...]
    sibling_alphas: tuple[float, ...]
    lenses: tuple[LensSpec, ...]
    tree_linkage_methods: tuple[str, ...]
    shard_count: int
    max_rank: int = 80
    min_segment_length: int = 4
    max_segments: int = 8
    diffusion_k_neighbors: int = 15
    diffusion_time: int = 3
    diffusion_components: int = 30
    adaptive_bandwidth_type: str = "-1/(d+2)"
    adaptive_epsilon: str = "median"
    adaptive_metric: str = "euclidean"
    input_s3_uri: str | None = None
    input_work_dir: Path = Path("/tmp/kak-lens-linkage-alpha-input")
    s3_uri: str | None = None


@dataclass(frozen=True)
class KakLensSweepShardSpec:
    """Concrete shard parameters after resolving AWS Batch array state."""

    shard_index: int
    shard_count: int
    output_dir: Path
    groups: tuple[LensLinkageGroup, ...]


def lens_to_arg(lens: LensSpec) -> str:
    return f"{lens.family}:{lens.weighting}:{lens.block_name}"


def parse_lenses(raw_lenses: str | None) -> tuple[LensSpec, ...]:
    if raw_lenses is None or not raw_lenses.strip():
        return DEFAULT_LENSES
    return tuple(parse_lens(item.strip()) for item in raw_lenses.split(",") if item.strip())


def build_groups(configured: AwsKakLensSweepConfig) -> tuple[LensLinkageGroup, ...]:
    groups: list[LensLinkageGroup] = []
    for lens in configured.lenses:
        for linkage_method in configured.tree_linkage_methods:
            groups.append(
                LensLinkageGroup(
                    lens=lens,
                    tree_linkage_method=linkage_method,
                    group_index=len(groups),
                )
            )
    return tuple(groups)


def validate_shard_contract(*, shard_index: int, shard_count: int, n_groups: int) -> None:
    if shard_count <= 0:
        raise ValueError(f"shard_count must be positive; got {shard_count!r}.")
    if not 0 <= shard_index < shard_count:
        raise ValueError(
            f"shard_index must satisfy 0 <= index < shard_count; got "
            f"{shard_index!r} with shard_count={shard_count!r}."
        )
    if shard_count > n_groups:
        raise ValueError(
            f"shard_count={shard_count} exceeds number of lens/linkage groups {n_groups}."
        )


def make_shard_spec(
    configured: AwsKakLensSweepConfig,
    shard_index: int,
) -> KakLensSweepShardSpec:
    groups = build_groups(configured)
    validate_shard_contract(
        shard_index=shard_index,
        shard_count=configured.shard_count,
        n_groups=len(groups),
    )
    selected = tuple(
        group for group in groups if group.group_index % configured.shard_count == shard_index
    )
    if not selected:
        raise ValueError(f"Shard {shard_index} has no lens/linkage groups.")
    return KakLensSweepShardSpec(
        shard_index=int(shard_index),
        shard_count=int(configured.shard_count),
        output_dir=configured.output_dir / "shards" / f"shard_{shard_index:04d}",
        groups=selected,
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _resolved_path(path: Path, input_work_dir: Path, *, use_synced_input: bool) -> Path:
    if path.is_absolute() or not use_synced_input:
        return path
    return input_work_dir / path


def resolve_synced_input_config(configured: AwsKakLensSweepConfig) -> AwsKakLensSweepConfig:
    """Sync repository-shaped S3 inputs and point relative paths at the bundle."""

    if configured.input_s3_uri is None:
        return configured
    sync_s3_to_path(configured.input_s3_uri.rstrip("/"), configured.input_work_dir)
    return replace(
        configured,
        input_path=_resolved_path(
            configured.input_path,
            configured.input_work_dir,
            use_synced_input=True,
        ),
        main_assignments=_resolved_path(
            configured.main_assignments,
            configured.input_work_dir,
            use_synced_input=True,
        ),
    )


def _group_output_dir(shard_dir: Path, group: LensLinkageGroup) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in group.group_id)
    return shard_dir / "groups" / f"{group.group_index:04d}_{safe}"


def _run_group(
    configured: AwsKakLensSweepConfig, group: LensLinkageGroup, output_dir: Path
) -> None:
    command = [
        sys.executable,
        "-m",
        "benchmarks.diagnostics.spectral.kak_lens_alpha_sweep",
        "--input",
        str(configured.input_path),
        "--main-assignments",
        str(configured.main_assignments),
        "--output-dir",
        str(output_dir),
        "--edge-alphas",
        ",".join(str(value) for value in configured.edge_alphas),
        "--sibling-alphas",
        ",".join(str(value) for value in configured.sibling_alphas),
        "--tree-linkage-methods",
        group.tree_linkage_method,
        "--lens",
        lens_to_arg(group.lens),
        "--max-rank",
        str(configured.max_rank),
        "--min-segment-length",
        str(configured.min_segment_length),
        "--max-segments",
        str(configured.max_segments),
        "--diffusion-k-neighbors",
        str(configured.diffusion_k_neighbors),
        "--diffusion-time",
        str(configured.diffusion_time),
        "--diffusion-components",
        str(configured.diffusion_components),
        f"--adaptive-bandwidth-type={configured.adaptive_bandwidth_type}",
        "--adaptive-epsilon",
        configured.adaptive_epsilon,
        "--adaptive-metric",
        configured.adaptive_metric,
    ]
    subprocess.run(command, check=True)


def run_shard(configured: AwsKakLensSweepConfig, shard_index: int) -> pd.DataFrame:
    spec = make_shard_spec(configured, shard_index)
    started = perf_counter()
    spec.output_dir.mkdir(parents=True, exist_ok=True)

    tables: list[pd.DataFrame] = []
    for group in spec.groups:
        group_dir = _group_output_dir(spec.output_dir, group)
        _run_group(configured, group, group_dir)
        summary_path = group_dir / "kak_lens_alpha_sweep_summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing group summary: {summary_path}")
        table = pd.read_csv(summary_path)
        table.insert(0, "aws_group_index", int(group.group_index))
        table.insert(1, "aws_group_id", group.group_id)
        tables.append(table)

    summary = pd.concat(tables, ignore_index=True)
    summary.to_csv(spec.output_dir / SUMMARY_NAME, index=False)
    manifest = {
        "runner": "benchmarks.cloud.aws_kak_lens_linkage_alpha_sweep",
        "execution_role": AWS_KAK_LENS_SWEEP_ROLE,
        "shard_index": int(spec.shard_index),
        "shard_count": int(spec.shard_count),
        "groups": [
            {
                "group_index": int(group.group_index),
                "group_id": group.group_id,
                "lens": lens_to_arg(group.lens),
                "tree_linkage_method": group.tree_linkage_method,
            }
            for group in spec.groups
        ],
        "input_path": str(configured.input_path),
        "main_assignments": str(configured.main_assignments),
        "input_s3_uri": configured.input_s3_uri,
        "input_work_dir": str(configured.input_work_dir),
        "edge_alphas": list(configured.edge_alphas),
        "sibling_alphas": list(configured.sibling_alphas),
        "elapsed_sec": round(float(perf_counter() - started), 6),
        "git": current_git_state(),
        "note": (
            "Shard output is KAK/cosine lens diagnostic evidence only. Merge all "
            "shards before interpreting average-complete-ward linkage behavior."
        ),
    }
    _write_json(spec.output_dir / SHARD_MANIFEST_NAME, manifest)
    if configured.s3_uri is not None:
        sync_path_to_s3(
            spec.output_dir,
            f"{configured.s3_uri.rstrip('/')}/shards/shard_{shard_index:04d}",
        )
    return summary


def iter_expected_shard_dirs(output_dir: Path, shard_count: int) -> tuple[Path, ...]:
    if shard_count <= 0:
        raise ValueError(f"shard_count must be positive; got {shard_count!r}.")
    return tuple(output_dir / "shards" / f"shard_{index:04d}" for index in range(shard_count))


def load_shard_summary(shard_dir: Path) -> pd.DataFrame:
    manifest_path = shard_dir / SHARD_MANIFEST_NAME
    summary_path = shard_dir / SUMMARY_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing KAK lens shard manifest: {manifest_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing KAK lens shard summary: {summary_path}")
    manifest = json.loads(manifest_path.read_text())
    table = pd.read_csv(summary_path)
    table.insert(0, "aws_shard_index", int(manifest["shard_index"]))
    return table


def merge_shards(configured: AwsKakLensSweepConfig) -> pd.DataFrame:
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
    expected_groups = {group.group_id for group in build_groups(configured)}
    observed_groups = set(summary["aws_group_id"].astype(str))
    missing_groups = sorted(expected_groups - observed_groups)
    if missing_groups:
        raise ValueError(f"Missing lens/linkage group summaries after merge: {missing_groups!r}")

    merged_dir = configured.output_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    summary.sort_values(
        ["aws_group_index", "edge_alpha", "sibling_alpha"],
        na_position="last",
    ).to_csv(merged_dir / SUMMARY_NAME, index=False)
    manifest = {
        "runner": "benchmarks.cloud.aws_kak_lens_linkage_alpha_sweep",
        "execution_role": AWS_KAK_LENS_SWEEP_ROLE,
        "input_path": str(configured.input_path),
        "main_assignments": str(configured.main_assignments),
        "input_s3_uri": configured.input_s3_uri,
        "input_work_dir": str(configured.input_work_dir),
        "edge_alphas": list(configured.edge_alphas),
        "sibling_alphas": list(configured.sibling_alphas),
        "lenses": [lens_to_arg(lens) for lens in configured.lenses],
        "tree_linkage_methods": list(configured.tree_linkage_methods),
        "shard_count": int(configured.shard_count),
        "n_groups": int(len(expected_groups)),
        "n_rows": int(summary.shape[0]),
        "outputs": {"summary": str(merged_dir / SUMMARY_NAME)},
        "source_shard_dirs": [str(path) for path in shard_dirs],
        "git": current_git_state(),
        "note": (
            "Merged output is diagnostic evidence for KAK/cosine lens linkage and "
            "alpha behavior. It is not a selected-subspace calibration proof."
        ),
    }
    _write_json(merged_dir / COMBINED_MANIFEST_NAME, manifest)
    if configured.s3_uri is not None:
        sync_path_to_s3(merged_dir, f"{configured.s3_uri.rstrip('/')}/merged")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or merge AWS-sharded KAK/cosine lens linkage alpha sweeps."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_shared_arguments(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--input",
            type=Path,
            default=Path("data/feature_matrices/feature_matrix_julia_GOCC_GOBP_GOMF_combined.tsv"),
        )
        subparser.add_argument(
            "--main-assignments",
            type=Path,
            default=Path(
                "benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/"
                "09_full_data_method_reference_comparison_20260610/tbs_diffusion_adaptive/"
                "cluster_assignments.csv"
            ),
        )
        subparser.add_argument(
            "--edge-alphas",
            default=",".join(str(value) for value in DEFAULT_EDGE_ALPHA_GRID),
        )
        subparser.add_argument(
            "--sibling-alphas",
            default=",".join(str(value) for value in DEFAULT_SIBLING_ALPHA_GRID),
        )
        subparser.add_argument("--lenses", default=None)
        subparser.add_argument("--tree-linkage-methods", default="average,complete,ward")
        subparser.add_argument("--output-dir", type=Path, required=True)
        subparser.add_argument("--input-s3-uri", default=None)
        subparser.add_argument(
            "--input-work-dir",
            type=Path,
            default=Path("/tmp/kak-lens-linkage-alpha-input"),
        )
        subparser.add_argument("--shard-count", type=int, required=True)
        subparser.add_argument("--s3-uri", default=None)
        subparser.add_argument("--max-rank", type=int, default=80)
        subparser.add_argument("--min-segment-length", type=int, default=4)
        subparser.add_argument("--max-segments", type=int, default=8)
        subparser.add_argument("--diffusion-k-neighbors", type=int, default=15)
        subparser.add_argument("--diffusion-time", type=int, default=3)
        subparser.add_argument("--diffusion-components", type=int, default=30)
        subparser.add_argument("--adaptive-bandwidth-type", default="-1/(d+2)")
        subparser.add_argument("--adaptive-epsilon", default="median")
        subparser.add_argument("--adaptive-metric", default="euclidean")

    shard_parser = subparsers.add_parser("run-shard")
    add_shared_arguments(shard_parser)
    shard_parser.add_argument("--shard-index", type=int, default=None)

    merge_parser = subparsers.add_parser("merge")
    add_shared_arguments(merge_parser)

    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> AwsKakLensSweepConfig:
    return AwsKakLensSweepConfig(
        input_path=Path(args.input),
        main_assignments=Path(args.main_assignments),
        output_dir=Path(args.output_dir),
        input_s3_uri=args.input_s3_uri,
        input_work_dir=Path(args.input_work_dir),
        edge_alphas=tuple(parse_float_list(str(args.edge_alphas))),
        sibling_alphas=tuple(parse_float_list(str(args.sibling_alphas))),
        lenses=parse_lenses(args.lenses),
        tree_linkage_methods=tuple(parse_linkage_methods(str(args.tree_linkage_methods))),
        shard_count=int(args.shard_count),
        max_rank=int(args.max_rank),
        min_segment_length=int(args.min_segment_length),
        max_segments=int(args.max_segments),
        diffusion_k_neighbors=int(args.diffusion_k_neighbors),
        diffusion_time=int(args.diffusion_time),
        diffusion_components=int(args.diffusion_components),
        adaptive_bandwidth_type=str(args.adaptive_bandwidth_type),
        adaptive_epsilon=str(args.adaptive_epsilon),
        adaptive_metric=str(args.adaptive_metric),
        s3_uri=args.s3_uri,
    )


def main() -> None:
    args = _parse_args()
    configured = _config_from_args(args)
    if args.command == "run-shard":
        configured = resolve_synced_input_config(configured)
        shard_index = resolve_shard_index(args.shard_index)
        run_shard(configured, shard_index)
        print(
            "Wrote KAK lens linkage alpha shard "
            f"{shard_index} to {make_shard_spec(configured, shard_index).output_dir}"
        )
    elif args.command == "merge":
        merge_shards(configured)
        print(f"Wrote merged KAK lens linkage alpha sweep to {configured.output_dir / 'merged'}")
    else:
        raise ValueError(f"Unknown command: {args.command!r}.")


if __name__ == "__main__":
    main()


__all__ = [
    "AWS_KAK_LENS_SWEEP_ROLE",
    "AwsKakLensSweepConfig",
    "KakLensSweepShardSpec",
    "LensLinkageGroup",
    "SHARD_MANIFEST_NAME",
    "SUMMARY_NAME",
    "build_groups",
    "iter_expected_shard_dirs",
    "lens_to_arg",
    "load_shard_summary",
    "make_shard_spec",
    "merge_shards",
    "parse_lenses",
    "resolve_shard_index",
    "run_shard",
    "validate_shard_contract",
]
