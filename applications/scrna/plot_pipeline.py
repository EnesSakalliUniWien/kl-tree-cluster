"""Unified plotting entry point and plot manifest for pancreas scRNA analyses."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLOT_SUFFIXES = {".png", ".pdf"}
METADATA_COLUMNS = {"cell_id", "celltype", "umap1", "umap2"}
DATASET_OUTPUT_DIRS = {
    "pancreas": (
        PROJECT_ROOT
        / "raw"
        / "assets"
        / "benchmark-results"
        / "pancreas_scrna_cluster_benchmark_20260623"
    ),
    "goncalves": (
        PROJECT_ROOT
        / "raw"
        / "assets"
        / "benchmark-results"
        / "goncalves_fetal_pancreas_progenitor_benchmark_20260624"
    ),
}

MANIFEST_FIELDS = [
    "dataset",
    "plot_id",
    "path",
    "format",
    "stage",
    "role",
    "canonical",
    "alias_of",
    "generator_script",
    "input_tables",
    "intended_question",
    "status",
    "sha256",
    "bytes",
    "notes",
]


def _python_application(relative_path: str, *args: str) -> list[str]:
    script_path = PROJECT_ROOT / "applications" / "scrna" / relative_path
    return [sys.executable, str(script_path), *args]


def _r_application(relative_path: str, *args: str) -> list[str]:
    script_path = PROJECT_ROOT / "applications" / "scrna" / relative_path
    return ["Rscript", str(script_path), *args]


def pipeline_commands(dataset: str, output_dir: Path) -> list[list[str]]:
    output_arg = str(output_dir)
    output_r_arg = f"--output-dir={output_dir}"
    title = {
        "pancreas": "Pancreas scRNA clustering UMAPs",
        "goncalves": "Goncalves fetal pancreas clustering UMAPs",
    }[dataset]
    combo_title = {
        "pancreas": "Pancreas TBS UMAPs and full radial trees",
        "goncalves": "Goncalves fetal pancreas TBS UMAPs and full radial trees",
    }[dataset]

    commands = [
        _python_application(
            "plots/pancreas_all_method_umap_clusters.py",
            "--output-dir",
            output_arg,
            "--title",
            title,
        ),
        _r_application("plots/pancreas_radial_trees_ggtree.R", output_r_arg),
        _r_application(
            "plots/pancreas_umap_tree_combo_ggtree.R",
            output_r_arg,
            f"--title={combo_title}",
        ),
    ]
    if dataset == "pancreas":
        commands.insert(
            1,
            _python_application(
                "plots/pancreas_readable_umap_clusters.py",
                "--output-dir",
                output_arg,
            ),
        )
        commands.insert(
            3,
            _r_application("plots/pancreas_cluster_radial_trees_ggtree.R", output_r_arg),
        )
    if dataset == "goncalves":
        commands.extend(
            [
                _python_application(
                    "analysis/analyze_goncalves_progenitors.py",
                    "--output-dir",
                    output_arg,
                ),
                _r_application(
                    "plots/goncalves_progenitor_trees_ggtree.R",
                    output_r_arg,
                    "--pdf-width=30",
                    "--pdf-height=42",
                    "--page-width=30",
                    "--page-height=16",
                    "--png-dpi=180",
                ),
            ]
        )
    return commands


def current_assignment_columns(output_dir: Path) -> set[str]:
    path = output_dir / "method_assignments.csv"
    if not path.exists():
        return set()
    with path.open(newline="") as handle:
        header = next(csv.reader(handle))
    return set(header) - METADATA_COLUMNS


def file_digest(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def base_row(
    *,
    dataset: str,
    output_dir: Path,
    name: str,
    stage: str,
    role: str,
    canonical: bool,
    alias_of: str = "",
    generator_script: str,
    input_tables: str,
    intended_question: str,
    notes: str = "",
    status: str | None = None,
) -> dict[str, Any]:
    path = output_dir / name
    exists = path.exists()
    sha256 = ""
    size: int | str = ""
    if exists:
        sha256, size = file_digest(path)
    return {
        "dataset": dataset,
        "plot_id": path.stem,
        "path": str(path),
        "format": path.suffix.removeprefix("."),
        "stage": stage,
        "role": role,
        "canonical": "true" if canonical else "false",
        "alias_of": alias_of,
        "generator_script": generator_script,
        "input_tables": input_tables,
        "intended_question": intended_question,
        "status": status or ("present" if exists else "missing"),
        "sha256": sha256,
        "bytes": size,
        "notes": notes,
    }


def add_known_rows(dataset: str, output_dir: Path, rows: dict[str, dict[str, Any]]) -> None:
    def add(name: str, **kwargs: Any) -> None:
        rows[name] = base_row(dataset=dataset, output_dir=output_dir, name=name, **kwargs)

    common_inputs = "method_assignments.csv;method_metrics.csv"
    for name in ["qc_metric_distributions.png", "classical_umap_overview.png"]:
        add(
            name,
            stage="qc_embedding",
            role="canonical",
            canonical=True,
            generator_script="applications/scrna/pancreas_benchmark.py",
            input_tables="qc_cell_metrics.csv;manifest.json",
            intended_question="Check the analysis subset, QC distributions, and embedding context.",
        )
    for name in ["method_ari_barplot.png", "method_split_merge_diagnostic.png"]:
        add(
            name,
            stage="benchmark_metrics",
            role="canonical",
            canonical=True,
            generator_script="applications/scrna/pancreas_benchmark.py",
            input_tables="method_metrics.csv",
            intended_question="Compare clustering metrics and split/merge behavior.",
        )
    add(
        "method_umap_clusters.png",
        stage="benchmark_umap",
        role="derivative",
        canonical=False,
        generator_script="applications/scrna/pancreas_benchmark.py",
        input_tables=common_inputs,
        intended_question="Quick in-run UMAP check; superseded for review by all_methods_umap_clusters_all_colored.",
        notes="Derivative quick diagnostic, not the canonical all-method review surface.",
    )
    for suffix in ["png", "pdf"]:
        add(
            f"all_methods_umap_clusters_all_colored.{suffix}",
            stage="benchmark_umap",
            role="canonical",
            canonical=True,
            generator_script="applications/scrna/plots/pancreas_all_method_umap_clusters.py",
            input_tables=common_inputs,
            intended_question="Review all method cluster assignments on the same UMAP with every assigned cluster colored.",
        )
        add(
            f"tbs_umap_cluster_radial_tree_combo_scaled_umap_ggtree.{suffix}",
            stage="tbs_umap_tree_review",
            role="canonical",
            canonical=True,
            generator_script="applications/scrna/plots/pancreas_umap_tree_combo_ggtree.R",
            input_tables="method_assignments.csv;*_tree_edges.csv",
            intended_question="Review each TBS UMAP beside its full radial tree using the same cluster colors.",
        )
        add(
            f"tbs_umap_cluster_radial_tree_combo_all_clusters_ggtree.{suffix}",
            stage="tbs_umap_tree_review",
            role="alias",
            canonical=False,
            alias_of="tbs_umap_cluster_radial_tree_combo_scaled_umap_ggtree",
            generator_script="applications/scrna/plots/pancreas_umap_tree_combo_ggtree.R",
            input_tables="method_assignments.csv;*_tree_edges.csv",
            intended_question="Compatibility alias for the scaled all-cluster TBS UMAP/tree review plot.",
        )
        add(
            f"tbs_umap_cluster_radial_tree_combo_ggtree.{suffix}",
            stage="tbs_umap_tree_review",
            role="alias",
            canonical=False,
            alias_of="tbs_umap_cluster_radial_tree_combo_scaled_umap_ggtree",
            generator_script="applications/scrna/plots/pancreas_umap_tree_combo_ggtree.R",
            input_tables="method_assignments.csv;*_tree_edges.csv",
            intended_question="Legacy compatibility alias for the scaled all-cluster TBS UMAP/tree review plot.",
        )
    add(
        "tbs_branch_time_sensitivity.png",
        stage="tbs_diagnostics",
        role="canonical",
        canonical=True,
        generator_script="applications/scrna/pancreas_benchmark.py",
        input_tables="tbs_branch_time_sensitivity.csv",
        intended_question="Review supervised branch-time transform split/merge sensitivity.",
    )

    if dataset == "pancreas":
        for suffix in ["png", "pdf"]:
            add(
                f"tbs_readable_umap_clusters_all_colored.{suffix}",
                stage="tbs_umap_review",
                role="derivative",
                canonical=False,
                generator_script="applications/scrna/plots/pancreas_readable_umap_clusters.py",
                input_tables="method_assignments.csv",
                intended_question="Standalone TBS UMAP view with every final cluster colored.",
            )
            add(
                f"tbs_readable_umap_clusters_ge50.{suffix}",
                stage="tbs_umap_review",
                role="alias",
                canonical=False,
                alias_of="tbs_readable_umap_clusters_all_colored",
                generator_script="applications/scrna/plots/pancreas_readable_umap_clusters.py",
                input_tables="method_assignments.csv",
                intended_question="Compatibility alias; only labels, not colors, are thresholded at 50 cells.",
            )
        for stem in [
            "pancreas_progenitor_signature_comparison",
            "pancreas_progenitor_signature_umap",
            "tbs_adaptive_inner_node_progenitor_comparison",
            "tbs_adaptive_monophyletic_subtree_meeting_mixed_umap",
            "tbs_adaptive_two_three_cluster_junction_mixed_umap",
            "celltype_cluster_composition_comparison",
        ]:
            for suffix in ["png", "pdf"]:
                add(
                    f"{stem}.{suffix}",
                    stage="biology_progenitor_review",
                    role="derivative",
                    canonical=False,
                    generator_script="applications/scrna/analysis/analyze_pancreas_inner_nodes.py;applications/scrna/analysis/compare_pancreas_progenitor_signatures.py",
                    input_tables="method_assignments.csv;*_progenitor*.csv",
                    intended_question="Review adult pancreas lineage/progenitor interpretation diagnostics.",
                )

    if dataset == "goncalves":
        for stem in [
            "goncalves_progenitor_signature_umap",
            "goncalves_population_tbs_cluster_signature_heatmap",
            "goncalves_tbs_monophyletic_meeting_progenitor_plot",
        ]:
            for suffix in ["png", "pdf"]:
                add(
                    f"{stem}.{suffix}",
                    stage="biology_progenitor_review",
                    role="canonical",
                    canonical=True,
                    generator_script="applications/scrna/analysis/analyze_goncalves_progenitors.py",
                    input_tables="goncalves_*_progenitor_signature_scores.csv",
                    intended_question="Review Goncalves fetal progenitor signatures, TBS clusters, and exact meeting nodes.",
                )
        add(
            "goncalves_tbs_progenitor_umap_tree_pages_ggtree.pdf",
            stage="biology_progenitor_tree_review",
            role="canonical",
            canonical=True,
            generator_script="applications/scrna/plots/goncalves_progenitor_trees_ggtree.R",
            input_tables="method_assignments.csv;goncalves_tbs_*_progenitor_signature_scores.csv;*_tree_edges.csv",
            intended_question="Canonical multi-page UMAP/tree review for fetal population, progenitor fraction, and progenitor state.",
        )
        add(
            "goncalves_tbs_relation_umap_tree_pages_ggtree.pdf",
            stage="relation_overview",
            role="canonical",
            canonical=True,
            generator_script="applications/scrna/plots/goncalves_progenitor_trees_ggtree.R",
            input_tables="method_assignments.csv;goncalves_tbs_*_progenitor_signature_scores.csv;*_tree_edges.csv",
            intended_question="Canonical multi-page relation review: TBS clusters, fetal population, progenitor fraction, and progenitor state on matching UMAP/tree pages.",
        )
        for stem in [
            "goncalves_tbs_relation_umap_grid_ggtree",
            "goncalves_tbs_relation_tree_grid_ggtree",
        ]:
            for suffix in ["png", "pdf"]:
                add(
                    f"{stem}.{suffix}",
                    stage="relation_overview",
                    role="canonical",
                    canonical=True,
                    generator_script="applications/scrna/plots/goncalves_progenitor_trees_ggtree.R",
                    input_tables="method_assignments.csv;goncalves_tbs_*_progenitor_signature_scores.csv;*_tree_edges.csv",
                    intended_question="Compare adaptive-diffusion TBS final clusters with Goncalves fetal population and progenitor interpretation using one consistent plotting surface.",
                )
        for stem in [
            "goncalves_tbs_population_radial_tree_ggtree",
            "goncalves_tbs_progenitor_fraction_radial_tree_ggtree",
            "goncalves_tbs_progenitor_state_radial_tree_ggtree",
            "goncalves_tbs_progenitor_tree_panel_ggtree",
            "goncalves_tbs_progenitor_umap_tree_panel_wide_ggtree",
        ]:
            for suffix in ["png", "pdf"]:
                add(
                    f"{stem}.{suffix}",
                    stage="biology_progenitor_tree_review",
                    role="derivative",
                    canonical=False,
                    generator_script="applications/scrna/plots/goncalves_progenitor_trees_ggtree.R",
                    input_tables="method_assignments.csv;goncalves_tbs_*_progenitor_signature_scores.csv;*_tree_edges.csv",
                    intended_question="Derivative Goncalves progenitor tree or UMAP/tree preview.",
                )
        for suffix in ["png", "pdf"]:
            add(
                f"goncalves_tbs_cluster_radial_tree_ggtree.{suffix}",
                stage="relation_overview",
                role="derivative",
                canonical=False,
                generator_script="applications/scrna/plots/goncalves_progenitor_trees_ggtree.R",
                input_tables="method_assignments.csv;*_tree_edges.csv",
                intended_question="Standalone adaptive-diffusion TBS final-cluster tree used in the Goncalves relation review.",
            )
        add(
            "goncalves_tbs_cluster_umap_tree_page_ggtree.png",
            stage="relation_overview",
            role="derivative",
            canonical=False,
            generator_script="applications/scrna/plots/goncalves_progenitor_trees_ggtree.R",
            input_tables="method_assignments.csv;*_tree_edges.csv",
            intended_question="PNG preview of the TBS clustering-result page from the canonical Goncalves relation UMAP/tree PDF.",
        )
        for stem in [
            "goncalves_tbs_population_umap_tree_page_ggtree",
            "goncalves_tbs_progenitor_fraction_umap_tree_page_ggtree",
            "goncalves_tbs_progenitor_state_umap_tree_page_ggtree",
        ]:
            add(
                f"{stem}.png",
                stage="biology_progenitor_tree_review",
                role="derivative",
                canonical=False,
                generator_script="applications/scrna/plots/goncalves_progenitor_trees_ggtree.R",
                input_tables="method_assignments.csv;goncalves_tbs_*_progenitor_signature_scores.csv;*_tree_edges.csv",
                intended_question="PNG page preview for the canonical multi-page Goncalves progenitor UMAP/tree PDF.",
            )


def method_prefix(name: str) -> str:
    for suffix in [
        "_clusters_radial_tree_ggtree.png",
        "_clusters_radial_tree_ggtree.pdf",
        "_radial_tree_ggtree.png",
        "_radial_tree_ggtree.pdf",
        "_tree_dendrogram.png",
        "_branch_lengths.png",
    ]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return ""


def classify_extra_plot(
    *,
    dataset: str,
    output_dir: Path,
    path: Path,
    current_methods: set[str],
) -> dict[str, Any]:
    prefix = method_prefix(path.name)
    if path.name.endswith("_branch_lengths.png"):
        status = "present" if prefix in current_methods else "orphaned_existing_file"
        return base_row(
            dataset=dataset,
            output_dir=output_dir,
            name=path.name,
            stage="tbs_diagnostics",
            role="derivative" if status == "present" else "orphaned_previous_run",
            canonical=False,
            generator_script="applications/scrna/pancreas_benchmark.py",
            input_tables=f"{prefix}_tree_edges.csv;tbs_tree_branch_length_summary.csv",
            intended_question="Inspect branch-length distribution for one TBS run.",
            status=status,
            notes="Current assignment column found."
            if status == "present"
            else "No matching current method_assignments column.",
        )
    if path.name.endswith("_tree_dendrogram.png"):
        status = "present" if prefix in current_methods else "orphaned_existing_file"
        return base_row(
            dataset=dataset,
            output_dir=output_dir,
            name=path.name,
            stage="tbs_diagnostics",
            role="derivative" if status == "present" else "orphaned_previous_run",
            canonical=False,
            generator_script="applications/scrna/pancreas_benchmark.py",
            input_tables=f"{prefix}_tree_edges.csv",
            intended_question="Inspect a truncated dendrogram for one TBS run.",
            status=status,
            notes="Current assignment column found."
            if status == "present"
            else "No matching current method_assignments column.",
        )
    if path.name.endswith("_clusters_radial_tree_ggtree.png") or path.name.endswith(
        "_clusters_radial_tree_ggtree.pdf"
    ):
        status = "present" if prefix in current_methods else "orphaned_existing_file"
        return base_row(
            dataset=dataset,
            output_dir=output_dir,
            name=path.name,
            stage="tbs_tree_review",
            role="derivative" if status == "present" else "orphaned_previous_run",
            canonical=False,
            generator_script="applications/scrna/plots/pancreas_cluster_radial_trees_ggtree.R",
            input_tables=f"method_assignments.csv;{prefix}_tree_edges.csv",
            intended_question="Standalone full radial tree colored by final TBS clusters.",
            status=status,
            notes="Current assignment column found."
            if status == "present"
            else "No matching current method_assignments column.",
        )
    if path.name.endswith("_radial_tree_ggtree.png") or path.name.endswith(
        "_radial_tree_ggtree.pdf"
    ):
        status = "present" if prefix in current_methods else "orphaned_existing_file"
        return base_row(
            dataset=dataset,
            output_dir=output_dir,
            name=path.name,
            stage="tbs_tree_review",
            role="derivative" if status == "present" else "orphaned_previous_run",
            canonical=False,
            generator_script="applications/scrna/plots/pancreas_radial_trees_ggtree.R",
            input_tables=f"{prefix}_tree_edges.csv",
            intended_question="Standalone full radial tree colored by edge-gate state.",
            status=status,
            notes="Current assignment column found."
            if status == "present"
            else "No matching current method_assignments column.",
        )
    return base_row(
        dataset=dataset,
        output_dir=output_dir,
        name=path.name,
        stage="unclassified",
        role="review_needed",
        canonical=False,
        generator_script="unknown",
        input_tables="unknown",
        intended_question="Existing plot file not classified by the scRNA plot pipeline.",
        status="unclassified_existing_file",
    )


def validate_aliases(rows: list[dict[str, Any]]) -> None:
    target_by_key = {
        (Path(row["path"]).stem, row["format"]): row for row in rows if row["status"] == "present"
    }
    for row in rows:
        if row["role"] != "alias" or row["status"] != "present":
            continue
        target = target_by_key.get((row["alias_of"], row["format"]))
        if target is None:
            row["status"] = "alias_target_missing"
            row["notes"] = _append_note(row["notes"], "Alias target is missing from manifest.")
        elif row["sha256"] == target["sha256"]:
            row["notes"] = _append_note(row["notes"], "Alias hash matches target.")
        else:
            row["status"] = "alias_mismatch"
            row["notes"] = _append_note(row["notes"], "Alias hash differs from target.")


def _append_note(notes: str, extra: str) -> str:
    return f"{notes} {extra}".strip() if notes else extra


def build_manifest(dataset: str, output_dir: Path) -> list[dict[str, Any]]:
    rows_by_name: dict[str, dict[str, Any]] = {}
    add_known_rows(dataset, output_dir, rows_by_name)
    current_methods = current_assignment_columns(output_dir)

    for path in sorted(output_dir.iterdir()):
        if path.suffix.lower() not in PLOT_SUFFIXES:
            continue
        if path.name in rows_by_name:
            continue
        rows_by_name[path.name] = classify_extra_plot(
            dataset=dataset,
            output_dir=output_dir,
            path=path,
            current_methods=current_methods,
        )

    rows = [rows_by_name[name] for name in sorted(rows_by_name)]
    validate_aliases(rows)
    return rows


def write_manifest(
    *,
    dataset: str,
    output_dir: Path,
    rows: list[dict[str, Any]],
    commands: list[list[str]],
) -> dict[str, Any]:
    csv_path = output_dir / "plot_manifest.csv"
    json_path = output_dir / "plot_manifest.json"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    status_counts = Counter(str(row["status"]) for row in rows)
    role_counts = Counter(str(row["role"]) for row in rows)
    summary = {
        "dataset": dataset,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "plot_count": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "commands": commands,
        "manifest_csv": str(csv_path),
        "rows": rows,
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def run_commands(commands: list[list[str]]) -> None:
    for command in commands:
        print("+", " ".join(command))
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def strict_failures(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failure_statuses = {"missing", "alias_mismatch", "alias_target_missing"}
    return [
        row
        for row in rows
        if row["status"] in failure_statuses and row["role"] in {"canonical", "alias"}
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASET_OUTPUT_DIRS), required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-generators", action="store_true")
    parser.add_argument("--print-commands", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir or DATASET_OUTPUT_DIRS[args.dataset]
    output_dir.mkdir(parents=True, exist_ok=True)
    commands = pipeline_commands(args.dataset, output_dir)

    if args.print_commands:
        for command in commands:
            print(" ".join(command))

    if args.run_generators:
        run_commands(commands)

    rows = build_manifest(args.dataset, output_dir)
    summary = write_manifest(
        dataset=args.dataset, output_dir=output_dir, rows=rows, commands=commands
    )
    print(
        f"Wrote {summary['manifest_csv']} "
        f"({summary['plot_count']} plots; statuses: {summary['status_counts']})"
    )

    failures = strict_failures(rows)
    if args.strict and failures:
        details = "\n".join(f"- {Path(row['path']).name}: {row['status']}" for row in failures)
        raise SystemExit(f"Strict plot-manifest validation failed:\n{details}")


if __name__ == "__main__":
    main()
