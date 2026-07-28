#!/usr/bin/env python3
"""Run the canonical GO annotation feature-matrix analysis pipeline.

The pipeline keeps analysis levels explicit:

1. matrix quality diagnostics,
2. current adaptive-diffusion cosine-subspace TBS trees,
3. optional method/tree-geometry candidate matrix for audit only,
4. analysis-level inventory tying the output folder back to the canonical
   contract.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

from applications.endotypes._shared import matrix_slug
from applications.endotypes.audit_go_annotation_analysis_levels import (
    audit_paths,
    write_audit_outputs,
)


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
        "--skip-subspace-package",
        action="store_true",
        help="Skip the radial-tree, annotation-PDF, cluster-audit, and upload-package stages.",
    )
    parser.add_argument(
        "--skip-life-science-lookups",
        action="store_true",
        help="Build the systematic subspace PDF without QuickGO/UniProt lookups.",
    )
    parser.add_argument(
        "--include-method-matrix",
        action="store_true",
        help="Also run the current x tree-geometry candidate matrix as an audit stage.",
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
        default=["current"],
        choices=["current"],
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


def default_output_dir(input_path: Path, dataset_label: str | None = None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        Path("results/analyses")
        / f"{matrix_slug(input_path, dataset_label)}_go_annotation_pipeline_{stamp}"
    )


def _append_if_present(command: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def _append_many(command: list[str], flag: str, values: Sequence[object] | None) -> None:
    if values:
        command.append(flag)
        command.extend(str(value) for value in values)


def current_subspace_dir(root: Path) -> Path:
    return root / "10_current_adaptive_diffusion_subspace_tree"


def annotation_package_dir(root: Path) -> Path:
    return current_subspace_dir(root) / "systematic_subspace_gene_annotations"


def build_pipeline_plan(
    args: argparse.Namespace, output_dir: Path | None = None
) -> list[PipelineStage]:
    repo_root = Path(__file__).resolve().parents[2]
    root = output_dir or args.output_dir or default_output_dir(args.input, args.dataset_label)
    artifact_prefix = matrix_slug(args.input, args.dataset_label)
    stages: list[PipelineStage] = []

    inventory_command = [
        args.python,
        str(repo_root / "applications/endotypes/inventory_go_annotation_datasets.py"),
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
                    str(repo_root / "applications/endotypes/feature_matrix_quality_analysis.py"),
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
            str(
                repo_root
                / "applications/endotypes/run_current_adaptive_diffusion_subspace_tree_experiment.py"
            ),
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
            f"--adaptive-bandwidth-type={args.adaptive_bandwidth_type}",
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
                    "Run the canonical current TBS adaptive-diffusion cosine-subspace "
                    "tree pipeline with connected PDFs, rankings, and manifests."
                ),
                output_dir=root / "10_current_adaptive_diffusion_subspace_tree",
                command=command,
            )
        )

    if args.include_method_matrix:
        command = [
            args.python,
            str(repo_root / "applications/endotypes/run_allgo_method_version_tree_matrix.py"),
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
            f"--adaptive-bandwidth-type={args.adaptive_bandwidth_type}",
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
                    "Optional current x tree-geometry candidate matrix. "
                    "Use as an audit, not as the canonical reader PDF pipeline."
                ),
                output_dir=root / "20_method_tree_matrix_audit",
                command=command,
                optional=True,
            )
        )

    if not args.skip_current_subspace and not args.skip_subspace_package:
        experiment_dir = current_subspace_dir(root)
        annotation_dir = annotation_package_dir(root)
        stages.append(
            PipelineStage(
                stage_id="40_subspace_rosters_radial_trees",
                analysis_level="40_systematic_subspace_annotation_package",
                description="Export per-subspace rosters, source artifacts, embeddings, and radial cluster trees.",
                output_dir=annotation_dir,
                command=[
                    args.python,
                    str(repo_root / "applications/endotypes/export_subspace_cluster_rosters.py"),
                    "--experiment-dir",
                    str(experiment_dir),
                    "--feature-matrix",
                    str(args.input),
                    "--dataset-label",
                    artifact_prefix,
                ],
            )
        )
        pdf_command = [
            args.python,
            str(repo_root / "applications/endotypes/build_subspace_gene_annotation_pdf.py"),
            "--feature-matrix",
            str(args.input),
            "--experiment-dir",
            str(experiment_dir),
            "--dataset-label",
            artifact_prefix,
        ]
        if args.skip_life_science_lookups:
            pdf_command.append("--skip-life-science-lookups")
        stages.append(
            PipelineStage(
                stage_id="45_subspace_annotation_pdf",
                analysis_level="40_systematic_subspace_annotation_package",
                description="Build the systematic two-page-per-subspace annotation PDF.",
                output_dir=annotation_dir,
                command=pdf_command,
            )
        )
        stages.append(
            PipelineStage(
                stage_id="50_cluster_meaningfulness_audit",
                analysis_level="40_systematic_subspace_annotation_package",
                description="Audit cluster feature enrichment and eigenband coherence against size-preserving nulls.",
                output_dir=annotation_dir / "cluster_meaningfulness_audit",
                command=[
                    args.python,
                    str(repo_root / "applications/endotypes/audit_go_cluster_meaningfulness.py"),
                    "--dataset-label",
                    artifact_prefix,
                    "--feature-matrix",
                    str(args.input),
                    "--annotation-root",
                    str(annotation_dir),
                ],
            )
        )
        stages.append(
            PipelineStage(
                stage_id="90_package_for_github",
                analysis_level="90_github_upload_package",
                description="Copy the input matrix, verify radial outputs, and write the upload run summary.",
                output_dir=root,
                command=[
                    "internal:finalize_go_annotation_results",
                    "--input",
                    str(args.input),
                    "--output-dir",
                    str(root),
                    "--dataset-label",
                    artifact_prefix,
                ],
            )
        )

    return stages


def _command_to_text(command: Sequence[str]) -> str:
    return " ".join(str(part) for part in command)


def _is_packaging_stage(stage: PipelineStage) -> bool:
    return stage.analysis_level in {
        "40_systematic_subspace_annotation_package",
        "90_github_upload_package",
    }


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
        "- `30_canonical_current_subspace_pipeline`: current TBS gates on adaptive-diffusion cosine subspace trees.",
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


def _read_matrix_shape(input_path: Path) -> tuple[int, int]:
    import pandas as pd

    frame = pd.read_csv(input_path, sep="\t")
    return int(frame.shape[0]), int(max(frame.shape[1] - 1, 0))


def _read_csv_if_exists(path: Path):
    import pandas as pd

    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _pdf_page_count(path: Path) -> int | None:
    try:
        from pypdf import PdfReader
    except Exception:
        return None
    try:
        return len(PdfReader(str(path)).pages)
    except Exception:
        return None


def _resolve_output_path(value: object, *, repo_root: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return repo_root / path


def _quickgo_obsolete_lookup(annotation_root: Path) -> dict[str, bool]:
    cache_path = annotation_root / "life_science_lookup_cache.json"
    if not cache_path.exists():
        return {}
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    lookup: dict[str, bool] = {}
    for go_id, record in cache.get("quickgo", {}).items():
        text = " ".join(str(record.get(key, "")) for key in ("name", "definition")).lower()
        lookup[str(go_id)] = "obsolete" in text
    return lookup


def write_annotation_recurrence_outputs(
    *,
    dataset_label: str,
    annotation_root: Path,
    audit_dir: Path,
) -> tuple[object, object]:
    import pandas as pd

    cluster_stats = _read_csv_if_exists(audit_dir / "cluster_meaningfulness.csv")
    subspace_summary = _read_csv_if_exists(audit_dir / "subspace_meaningfulness_summary.csv")
    columns = [
        "top_go_id",
        "top_term",
        "is_obsolete",
        "n_clusters",
        "n_subspaces",
        "min_q",
        "median_lift",
        "max_lift",
        "total_hits",
        "total_cluster_genes",
    ]
    if cluster_stats.empty:
        all_terms = pd.DataFrame(columns=columns)
    else:
        strong = cluster_stats[
            cluster_stats["meaningfulness_label"].eq("strong")
            & cluster_stats["top_term"].fillna("").astype(str).ne("")
        ].copy()
        obsolete_lookup = _quickgo_obsolete_lookup(annotation_root)
        if strong.empty:
            all_terms = pd.DataFrame(columns=columns)
        else:
            grouped = strong.groupby(["top_go_id", "top_term"], dropna=False)
            all_terms = grouped.agg(
                n_clusters=("cluster_id", "count"),
                n_subspaces=("run_id", "nunique"),
                min_q=("best_q_value", "min"),
                median_lift=("top_lift", "median"),
                max_lift=("top_lift", "max"),
                total_hits=("top_hits", "sum"),
                total_cluster_genes=("cluster_size", "sum"),
            ).reset_index()
            all_terms["top_go_id"] = all_terms["top_go_id"].fillna("").astype(str)
            all_terms["is_obsolete"] = all_terms["top_go_id"].map(
                lambda go_id: bool(obsolete_lookup.get(go_id, False))
            )
            all_terms = all_terms[columns].sort_values(
                ["n_clusters", "n_subspaces", "min_q"],
                ascending=[False, False, True],
            )
    nonobsolete = all_terms[~all_terms["is_obsolete"]].drop(
        columns=["is_obsolete"], errors="ignore"
    )
    all_terms.to_csv(audit_dir / "highest_occurring_annotations_all_terms.csv", index=False)
    nonobsolete.to_csv(audit_dir / "highest_occurring_nonobsolete_annotations.csv", index=False)

    lines = [
        f"# {dataset_label} Annotation Coherence Summary",
        "",
        f"Annotation root: `{annotation_root}`",
        "",
        "## Eigenband coherence",
        "",
    ]
    if subspace_summary.empty:
        lines.append("No subspace coherence summary was produced.")
    else:
        display_columns = [
            "display_rank",
            "weighting",
            "block_name",
            "n_clusters",
            "n_tested_clusters",
            "strong_fraction",
            "enriched_fraction",
            "empirical_p_enriched_fraction",
            "empirical_p_strong_fraction",
            "eigenband_coherence",
            "top_terms",
        ]
        lines.append(
            subspace_summary[
                [col for col in display_columns if col in subspace_summary]
            ].to_markdown(index=False)
        )
    lines.extend(["", "## Highest Recurring Non-Obsolete Strong Annotations", ""])
    if nonobsolete.empty:
        lines.append("No non-obsolete strong annotations were found.")
    else:
        lines.append(nonobsolete.head(25).to_markdown(index=False, floatfmt=".3g"))
    lines.extend(["", "## Highest Recurring Strong Annotations Including Obsolete Terms", ""])
    if all_terms.empty:
        lines.append("No strong annotations were found.")
    else:
        lines.append(all_terms.head(25).to_markdown(index=False, floatfmt=".3g"))
    (audit_dir / "coherence_and_top_annotations_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return all_terms, nonobsolete


def finalize_go_annotation_results(
    *,
    repo_root: Path,
    input_path: Path,
    output_dir: Path,
    dataset_label: str,
) -> dict[str, object]:
    import pandas as pd

    annotation_root = annotation_package_dir(output_dir)
    audit_dir = annotation_root / "cluster_meaningfulness_audit"
    report_pdf = annotation_root / f"{dataset_label}_systematic_subspace_gene_annotation_report.pdf"
    radial_pdf = annotation_root / f"{dataset_label}_radial_tree_clusters.pdf"
    required = [
        report_pdf,
        radial_pdf,
        annotation_root / "subspace_cluster_roster.csv",
        annotation_root / "subspace_gene_membership_long.csv",
        annotation_root / "subspace_cluster_status.csv",
        audit_dir / "cluster_meaningfulness.csv",
        audit_dir / "subspace_meaningfulness_summary.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required GO annotation package outputs: " + ", ".join(missing)
        )

    input_dir = output_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    copied_input = input_dir / input_path.name
    shutil.copy2(input_path, copied_input)

    status = pd.read_csv(annotation_root / "subspace_cluster_status.csv")
    radial_column = "radial_tree_clusters_png"
    if radial_column not in status:
        raise ValueError("subspace_cluster_status.csv is missing radial_tree_clusters_png.")
    radial_paths = [
        _resolve_output_path(value, repo_root=repo_root)
        for value in status[radial_column].dropna().astype(str)
        if value.strip()
    ]
    missing_radial = [str(path) for path in radial_paths if not path.exists()]
    if not radial_paths:
        raise ValueError("No radial tree files were recorded in subspace_cluster_status.csv.")
    if missing_radial:
        raise FileNotFoundError(
            "Missing recorded radial tree files: " + ", ".join(missing_radial[:10])
        )

    rows, features = _read_matrix_shape(input_path)
    membership = pd.read_csv(annotation_root / "subspace_gene_membership_long.csv")
    cluster_stats = pd.read_csv(audit_dir / "cluster_meaningfulness.csv")
    subspace_summary = pd.read_csv(audit_dir / "subspace_meaningfulness_summary.csv")
    assignment_counts = (
        status["assignment_source"].fillna("").astype(str).value_counts().sort_index().to_dict()
        if "assignment_source" in status
        else {}
    )
    coherence_counts = (
        subspace_summary["eigenband_coherence"]
        .fillna("")
        .astype(str)
        .value_counts()
        .sort_index()
        .to_dict()
        if "eigenband_coherence" in subspace_summary
        else {}
    )
    per_subspace_genes = (
        membership.groupby("run_id")["gene"].nunique().astype(int)
        if {"run_id", "gene"}.issubset(membership.columns)
        else pd.Series(dtype=int)
    )
    all_memberships_cover_input = bool(
        len(per_subspace_genes) and per_subspace_genes.eq(rows).all()
    )
    all_terms, nonobsolete = write_annotation_recurrence_outputs(
        dataset_label=dataset_label,
        annotation_root=annotation_root,
        audit_dir=audit_dir,
    )

    pages = _pdf_page_count(report_pdf) if report_pdf.exists() else None
    key_outputs = [
        annotation_root / f"{dataset_label}_systematic_subspace_gene_annotation_report.pdf",
        annotation_root / f"{dataset_label}_radial_tree_clusters.pdf",
        annotation_root / "subspace_cluster_roster.csv",
        annotation_root / "subspace_gene_membership_long.csv",
        annotation_root / "subspace_cluster_status.csv",
        audit_dir / "coherence_and_top_annotations_summary.md",
        audit_dir / "highest_occurring_nonobsolete_annotations.csv",
        audit_dir / "highest_occurring_annotations_all_terms.csv",
    ]
    lines = [
        f"# {dataset_label} GO Annotation Analysis",
        "",
        f"Input matrix: `{input_path}`",
        "",
        f"Dataset shape: `{rows}` genes x `{features}` features.",
        "",
        "Main output directory:",
        f"`{current_subspace_dir(output_dir)}`",
        "",
        "Key outputs:",
        "",
    ]
    for path in key_outputs:
        lines.append(f"- `{path.relative_to(output_dir)}`")
    lines.extend(
        [
            "",
            "Checks:",
            "",
            f"- Subspace assignment sources: `{assignment_counts}`.",
            f"- All subspace gene-membership tables assign `{rows}/{rows}` genes: `{all_memberships_cover_input}`.",
            f"- Radial tree PNGs recorded: `{len(radial_paths)}`.",
            f"- The report PDF has `{pages if pages is not None else 'unknown'}` pages.",
            f"- Cluster meaningfulness audit covers `{len(cluster_stats)}` clusters across `{len(subspace_summary)}` subspaces.",
            f"- Strict eigenband coherence counts: `{coherence_counts}`.",
            "",
            "Highest recurring non-obsolete strong annotations:",
            "",
        ]
    )
    if len(nonobsolete) == 0:
        lines.append("- None.")
    else:
        for index, row in enumerate(nonobsolete.head(10).itertuples(index=False), start=1):
            go_id = getattr(row, "top_go_id", "")
            suffix = f" (`{go_id}`)" if go_id else ""
            lines.append(f"{index}. {getattr(row, 'top_term')}{suffix}")
    obsolete = all_terms[all_terms["is_obsolete"]] if "is_obsolete" in all_terms else pd.DataFrame()
    if len(obsolete):
        lines.extend(
            ["", "Obsolete top terms excluded from the non-obsolete recurrence table:", ""]
        )
        for row in obsolete.head(10).itertuples(index=False):
            go_id = getattr(row, "top_go_id", "")
            suffix = f" (`{go_id}`)" if go_id else ""
            lines.append(f"- {getattr(row, 'top_term')}{suffix}")
    (output_dir / "RUN_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "copied_input": str(copied_input),
        "run_summary": str(output_dir / "RUN_SUMMARY.md"),
        "radial_trees": len(radial_paths),
        "clusters": len(cluster_stats),
        "subspaces": len(subspace_summary),
    }


def _parse_internal_finalize_command(command: Sequence[str]) -> tuple[Path, Path, str]:
    values = {
        str(command[index]): str(command[index + 1]) for index in range(1, len(command) - 1, 2)
    }
    return Path(values["--input"]), Path(values["--output-dir"]), values["--dataset-label"]


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
    if stage.command and stage.command[0] == "internal:finalize_go_annotation_results":
        with log_path.open("w", encoding="utf-8") as log_file:
            try:
                input_path, output_dir, dataset_label = _parse_internal_finalize_command(
                    stage.command
                )
                result = finalize_go_annotation_results(
                    repo_root=repo_root,
                    input_path=input_path,
                    output_dir=output_dir,
                    dataset_label=dataset_label,
                )
                log_file.write(json.dumps(result, indent=2) + "\n")
                return {
                    "stage_id": stage.stage_id,
                    "status": "ok",
                    "returncode": 0,
                    "log_path": str(log_path),
                    "outputs": result,
                }
            except Exception as exc:
                log_file.write(f"{type(exc).__name__}: {exc}\n")
                return {
                    "stage_id": stage.stage_id,
                    "status": "failed",
                    "returncode": 1,
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
    audit_targets = (
        []
        if args.dry_run
        else [stage.output_dir for stage in stages if not _is_packaging_stage(stage)]
    )
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
