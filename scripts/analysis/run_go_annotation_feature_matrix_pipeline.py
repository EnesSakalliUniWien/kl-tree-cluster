#!/usr/bin/env python3
"""Run the canonical GO annotation feature-matrix analysis pipeline.

The pipeline keeps analysis levels explicit:

1. matrix quality diagnostics,
2. current adaptive-diffusion cosine-subspace KL trees,
3. optional method/tree-geometry candidate matrix for audit only,
4. analysis-level inventory tying the output folder back to the canonical
   contract.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

from scripts.analysis.audit_go_annotation_analysis_levels import audit_paths, write_audit_outputs


@dataclass(frozen=True)
class PipelineStage:
    stage_id: str
    analysis_level: str
    description: str
    output_dir: Path
    command: list[str]
    optional: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/feature_matrices/feature_matrix_julia_allGO_new.tsv"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dataset-label", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument("--skip-current-subspace", action="store_true")
    parser.add_argument(
        "--include-method-matrix",
        action="store_true",
        help="Also run the legacy/current x tree-geometry candidate matrix as an audit stage.",
    )
    parser.add_argument(
        "--audit-paths",
        nargs="*",
        type=Path,
        default=[],
        help="Additional existing result directories to include in the level audit.",
    )
    parser.add_argument("--edge-alpha", type=float, default=None)
    parser.add_argument("--sibling-alpha", type=float, default=None)
    parser.add_argument("--max-rank", type=int, default=80)
    parser.add_argument("--min-segment-length", type=int, default=4)
    parser.add_argument("--max-segments", type=int, default=8)
    parser.add_argument("--diffusion-k-neighbors", type=int, default=15)
    parser.add_argument("--diffusion-time", type=int, default=3)
    parser.add_argument("--diffusion-components", type=int, default=30)
    parser.add_argument("--adaptive-bandwidth-type", default="-1/(d+2)")
    parser.add_argument("--adaptive-epsilon", default="median")
    parser.add_argument("--adaptive-metric", default="euclidean")
    parser.add_argument(
        "--weightings",
        nargs="+",
        default=["binary", "tfidf"],
        choices=["binary", "tfidf"],
    )
    parser.add_argument("--block-names", nargs="*", default=None)
    parser.add_argument(
        "--method-versions",
        nargs="+",
        default=["legacy_c2ef", "current"],
        choices=["legacy_c2ef", "current"],
    )
    parser.add_argument(
        "--tree-geometries",
        nargs="+",
        default=[
            "whole_adaptive_diffusion",
            "raw_cosine_subspace",
            "adaptive_diffusion_cosine_subspace",
        ],
        choices=[
            "whole_adaptive_diffusion",
            "raw_cosine_subspace",
            "adaptive_diffusion_cosine_subspace",
        ],
    )
    return parser.parse_args()


def safe_name(value: object) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))


def matrix_slug(input_path: Path, dataset_label: str | None = None) -> str:
    raw = dataset_label or input_path.stem
    if raw.startswith("feature_matrix_"):
        raw = raw[len("feature_matrix_") :]
    return safe_name(raw).strip("_").lower() or "feature_matrix"


def default_output_dir(input_path: Path, dataset_label: str | None = None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results/analyses") / f"{matrix_slug(input_path, dataset_label)}_go_annotation_pipeline_{stamp}"


def _append_if_present(command: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def _append_many(command: list[str], flag: str, values: Sequence[object] | None) -> None:
    if values:
        command.append(flag)
        command.extend(str(value) for value in values)


def build_pipeline_plan(args: argparse.Namespace, output_dir: Path | None = None) -> list[PipelineStage]:
    repo_root = Path(__file__).resolve().parents[2]
    root = output_dir or args.output_dir or default_output_dir(args.input, args.dataset_label)
    artifact_prefix = matrix_slug(args.input, args.dataset_label)
    stages: list[PipelineStage] = []

    inventory_command = [
        args.python,
        str(repo_root / "scripts/analysis/inventory_go_annotation_datasets.py"),
        "--output-dir",
        str(root / "00_dataset_inventory"),
        "--extra-paths",
        str(args.input),
    ]
    stages.append(
        PipelineStage(
            stage_id="00_dataset_inventory",
            analysis_level="dataset_inventory_and_naming_preflight",
            description="Inventory GO feature matrices and check directory/naming conventions.",
            output_dir=root / "00_dataset_inventory",
            command=inventory_command,
        )
    )

    if not args.skip_quality:
        stages.append(
            PipelineStage(
                stage_id="05_matrix_quality",
                analysis_level="00_matrix_quality_only",
                description="Validate binary GO matrix quality before tree analysis.",
                output_dir=root / "05_matrix_quality",
                command=[
                    args.python,
                    str(repo_root / "scripts/analysis/feature_matrix_quality_analysis.py"),
                    "--input",
                    str(args.input),
                    "--output-dir",
                    str(root / "05_matrix_quality"),
                ],
            )
        )

    if not args.skip_current_subspace:
        command = [
            args.python,
            str(repo_root / "scripts/analysis/run_current_adaptive_diffusion_subspace_tree_experiment.py"),
            "--input",
            str(args.input),
            "--output-dir",
            str(root / "10_current_adaptive_diffusion_subspace_tree"),
            "--dataset-label",
            artifact_prefix,
            "--max-rank",
            str(args.max_rank),
            "--min-segment-length",
            str(args.min_segment_length),
            "--max-segments",
            str(args.max_segments),
            "--diffusion-k-neighbors",
            str(args.diffusion_k_neighbors),
            "--diffusion-time",
            str(args.diffusion_time),
            "--diffusion-components",
            str(args.diffusion_components),
            "--adaptive-bandwidth-type",
            str(args.adaptive_bandwidth_type),
            "--adaptive-epsilon",
            str(args.adaptive_epsilon),
            "--adaptive-metric",
            str(args.adaptive_metric),
        ]
        _append_if_present(command, "--edge-alpha", args.edge_alpha)
        _append_if_present(command, "--sibling-alpha", args.sibling_alpha)
        _append_many(command, "--weightings", args.weightings)
        _append_many(command, "--block-names", args.block_names)
        stages.append(
            PipelineStage(
                stage_id="10_current_adaptive_diffusion_subspace_tree",
                analysis_level="30_canonical_current_subspace_pipeline",
                description=(
                    "Run the canonical current KL adaptive-diffusion cosine-subspace "
                    "tree pipeline with connected PDFs, rankings, and manifests."
                ),
                output_dir=root / "10_current_adaptive_diffusion_subspace_tree",
                command=command,
            )
        )

    if args.include_method_matrix:
        command = [
            args.python,
            str(repo_root / "scripts/analysis/run_allgo_method_version_tree_matrix.py"),
            "--input",
            str(args.input),
            "--output-dir",
            str(root / "20_method_tree_matrix_audit"),
            "--max-rank",
            str(args.max_rank),
            "--min-segment-length",
            str(args.min_segment_length),
            "--max-segments",
            str(args.max_segments),
            "--diffusion-k-neighbors",
            str(args.diffusion_k_neighbors),
            "--diffusion-time",
            str(args.diffusion_time),
            "--diffusion-components",
            str(args.diffusion_components),
            "--adaptive-bandwidth-type",
            str(args.adaptive_bandwidth_type),
            "--adaptive-epsilon",
            str(args.adaptive_epsilon),
            "--adaptive-metric",
            str(args.adaptive_metric),
        ]
        _append_if_present(command, "--edge-alpha", args.edge_alpha)
        _append_if_present(command, "--sibling-alpha", args.sibling_alpha)
        _append_many(command, "--weightings", args.weightings)
        _append_many(command, "--block-names", args.block_names)
        _append_many(command, "--method-versions", args.method_versions)
        _append_many(command, "--tree-geometries", args.tree_geometries)
        stages.append(
            PipelineStage(
                stage_id="20_method_tree_matrix_audit",
                analysis_level="10_candidate_tree_generation",
                description=(
                    "Optional legacy/current x tree-geometry candidate matrix. "
                    "Use as an audit, not as the canonical reader PDF pipeline."
                ),
                output_dir=root / "20_method_tree_matrix_audit",
                command=command,
                optional=True,
            )
        )

    return stages


def _command_to_text(command: Sequence[str]) -> str:
    return " ".join(str(part) for part in command)


def write_pipeline_files(
    *,
    output_dir: Path,
    input_path: Path,
    stages: Sequence[PipelineStage],
    stage_results: Sequence[dict[str, object]],
    dry_run: bool,
    audit_outputs: dict[str, str] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "dry_run": dry_run,
        "canonical_level": "30_canonical_current_subspace_pipeline",
        "stages": [asdict(stage) for stage in stages],
        "stage_results": list(stage_results),
        "audit_outputs": audit_outputs or {},
    }
    (output_dir / "pipeline_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )
    lines = [
        "# GO Annotation Feature-Matrix Pipeline",
        "",
        f"Input: `{input_path}`",
        f"Dry run: `{dry_run}`",
        "",
        "Canonical interpretation level:",
        "- `30_canonical_current_subspace_pipeline`: current KL gates on adaptive-diffusion cosine subspace trees.",
        "",
        "## Stages",
        "",
    ]
    for stage in stages:
        lines.extend(
            [
                f"### {stage.stage_id}",
                f"- Analysis level: `{stage.analysis_level}`",
                f"- Output: `{stage.output_dir}`",
                f"- Optional: `{stage.optional}`",
                f"- Command: `{_command_to_text(stage.command)}`",
                "",
            ]
        )
    if audit_outputs:
        lines.extend(
            [
                "## Analysis-Level Audit",
                "",
                f"- CSV: `{audit_outputs['csv']}`",
                f"- Markdown: `{audit_outputs['markdown']}`",
                "",
            ]
        )
    (output_dir / "PIPELINE.md").write_text("\n".join(lines), encoding="utf-8")


def run_stage(stage: PipelineStage, *, repo_root: Path, dry_run: bool) -> dict[str, object]:
    stage.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = stage.output_dir / f"{stage.stage_id}.log"
    if dry_run:
        log_path.write_text(_command_to_text(stage.command) + "\n", encoding="utf-8")
        return {
            "stage_id": stage.stage_id,
            "status": "planned",
            "returncode": 0,
            "log_path": str(log_path),
        }
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            stage.command,
            cwd=repo_root,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    status = "ok" if completed.returncode == 0 else "failed"
    return {
        "stage_id": stage.stage_id,
        "status": status,
        "returncode": completed.returncode,
        "log_path": str(log_path),
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir or default_output_dir(args.input, args.dataset_label)
    stages = build_pipeline_plan(args, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_results = []
    for stage in stages:
        print(f"[{stage.stage_id}] {stage.description}", flush=True)
        result = run_stage(stage, repo_root=repo_root, dry_run=args.dry_run)
        stage_results.append(result)
        if result["status"] == "failed":
            write_pipeline_files(
                output_dir=output_dir,
                input_path=args.input,
                stages=stages,
                stage_results=stage_results,
                dry_run=args.dry_run,
            )
            raise SystemExit(int(result["returncode"]))

    audit_dir = output_dir / "30_analysis_level_audit"
    audit_targets = [] if args.dry_run else [stage.output_dir for stage in stages]
    audit_targets.extend(args.audit_paths)
    audit_frame = audit_paths(audit_targets)
    audit_outputs = write_audit_outputs(audit_frame, audit_dir)
    write_pipeline_files(
        output_dir=output_dir,
        input_path=args.input,
        stages=stages,
        stage_results=stage_results,
        dry_run=args.dry_run,
        audit_outputs=audit_outputs,
    )
    print(json.dumps({"output_dir": str(output_dir), "audit_outputs": audit_outputs}, indent=2))


if __name__ == "__main__":
    main()
