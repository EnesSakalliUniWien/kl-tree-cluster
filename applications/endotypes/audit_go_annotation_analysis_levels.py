#!/usr/bin/env python3
"""Audit GO annotation feature-matrix analysis output levels.

The allGO work has several historical output shapes: input-matrix quality PDFs,
candidate tree matrices, mixed GO-IC reader reports, method-split PDFs, and the
new connected current-method subspace pipeline. This audit makes those levels
explicit so the artifacts are not compared as if they were the same analysis.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

DEFAULT_SEARCH_ROOTS = (
    Path("results/analyses"),
    Path("raw/assets/benchmark-results"),
)


@dataclass(frozen=True)
class AnalysisLevel:
    code: str
    name: str
    order: int
    is_canonical: bool
    interpretation: str
    inconsistency: str


LEVEL_MATRIX_QUALITY = AnalysisLevel(
    code="00_matrix_quality_only",
    name="matrix quality only",
    order=0,
    is_canonical=False,
    interpretation="Feature-matrix diagnostics before any tree or TBS gate is run.",
    inconsistency=(
        "Input-quality level only: no tree assignments, GO-IC ranking, or "
        "reader-facing tree pages."
    ),
)
LEVEL_PIPELINE_ORCHESTRATION = AnalysisLevel(
    code="05_pipeline_orchestration_root",
    name="pipeline orchestration root",
    order=5,
    is_canonical=False,
    interpretation="Pipeline manifest root that records commands and child analysis stages.",
    inconsistency=(
        "Orchestration level: do not interpret directly as an analysis report; "
        "inspect the child stage directories and analysis-level audit."
    ),
)
LEVEL_CANDIDATE_MATRIX = AnalysisLevel(
    code="10_candidate_tree_generation",
    name="candidate tree generation",
    order=10,
    is_canonical=False,
    interpretation="Tree-assignment generation across method/tree-geometry axes.",
    inconsistency=(
        "Candidate-generation level: produces assignments and cluster summaries "
        "but not a connected GO-IC reader report."
    ),
)
LEVEL_MIXED_READER = AnalysisLevel(
    code="20_mixed_method_go_ic_reader_report",
    name="mixed-method GO-IC reader report",
    order=20,
    is_canonical=False,
    interpretation="GO-IC scoring and PDF pages assembled from mixed method families.",
    inconsistency=(
        "Mixed reader level: ranks multiple method families and tree geometries "
        "together before method-specific isolation."
    ),
)
LEVEL_METHOD_SPLIT = AnalysisLevel(
    code="25_posthoc_method_split_report",
    name="post-hoc method split report",
    order=25,
    is_canonical=False,
    interpretation="Method-specific PDFs split from an upstream mixed GO-IC report.",
    inconsistency=(
        "Post-hoc split level: useful for reading one method at a time but still "
        "depends on the upstream mixed-ranking input contract."
    ),
)
LEVEL_CANONICAL_CURRENT = AnalysisLevel(
    code="30_canonical_current_subspace_pipeline",
    name="canonical current subspace pipeline",
    order=30,
    is_canonical=True,
    interpretation=(
        "Current TBS gates on adaptive-diffusion cosine subspace trees with "
        "connected rankings, manifests, per-subspace artifacts, and method PDFs."
    ),
    inconsistency="Canonical target: current method/version and one tree-geometry contract.",
)
LEVEL_UNCLASSIFIED = AnalysisLevel(
    code="90_unclassified_pdf_report",
    name="unclassified PDF/report output",
    order=90,
    is_canonical=False,
    interpretation="Report-like output that does not match a known allGO contract.",
    inconsistency="Unclassified output level: inspect manually before interpretation.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths",
        nargs="*",
        type=Path,
        default=None,
        help="Explicit output directories to audit. Defaults to allGO-like result roots.",
    )
    parser.add_argument(
        "--search-roots",
        nargs="*",
        type=Path,
        default=list(DEFAULT_SEARCH_ROOTS),
        help="Roots scanned when --paths is omitted.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _safe_rel(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _iter_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        return []
    return [candidate for candidate in path.rglob("*") if candidate.is_file()]


def _first_existing(path: Path, candidates: Iterable[str]) -> Path | None:
    for candidate in candidates:
        candidate_path = path / candidate
        if candidate_path.exists():
            return candidate_path
    return None


def _read_summary_values(path: Path) -> dict[str, str]:
    values: dict[str, set[str]] = {
        "method_versions": set(),
        "tree_geometries": set(),
        "method_families": set(),
    }
    summary_paths = [
        path / "method_tree_matrix_summary.csv",
        path / "go_ic_tree_ranking.csv",
        path / "allgo_new_quality_aware_go_ic_tree_ranking.csv",
        path / "rankings/current_adaptive_diffusion_subspace_tree_ranking.csv",
    ]
    summary_paths.extend(sorted(path.glob("*_tree_ranking.csv")))
    summary_paths.extend(sorted((path / "rankings").glob("*_ranking.csv")))
    summary_paths.extend(sorted(path.rglob("method_tree_matrix_summary.csv")))
    summary_paths.extend(sorted(path.rglob("*_tree_ranking.csv")))

    config_path = path / "experiment_config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            config = {}
        for config_key, value_key in (
            ("method_version", "method_versions"),
            ("tree_geometry", "tree_geometries"),
        ):
            value = config.get(config_key)
            if value:
                values[value_key].add(str(value))

    for summary_path in dict.fromkeys(summary_paths):
        if not summary_path.exists():
            continue
        try:
            frame = pd.read_csv(summary_path, nrows=500)
        except Exception:
            continue
        for column, key in (
            ("method_version", "method_versions"),
            ("tree_geometry", "tree_geometries"),
            ("family", "method_families"),
            ("method_family", "method_families"),
        ):
            if column in frame.columns:
                values[key].update(str(value) for value in frame[column].dropna().unique())

    return {key: ";".join(sorted(value)) for key, value in values.items()}


def infer_level(features: dict[str, object]) -> AnalysisLevel:
    if features["has_pipeline_manifest"]:
        return LEVEL_PIPELINE_ORCHESTRATION
    if features["has_connected_manifest"] and features["has_subspaces_dir"]:
        return LEVEL_CANONICAL_CURRENT
    if features["has_current_subspace_ranking"] and features["has_subspaces_dir"]:
        return LEVEL_CANONICAL_CURRENT
    if features["has_matrix_quality_summary"] and not features["has_assignments"]:
        return LEVEL_MATRIX_QUALITY
    if features["has_method_tree_matrix_summary"]:
        return LEVEL_CANDIDATE_MATRIX
    if features["has_go_ic_ranking"] and features["has_method_split_pdfs"]:
        return LEVEL_MIXED_READER
    if features["has_method_split_pdfs"]:
        return LEVEL_METHOD_SPLIT
    if int(features["pdf_count"]) > 0 or bool(features["has_go_ic_ranking"]):
        return LEVEL_UNCLASSIFIED
    return LEVEL_UNCLASSIFIED


def collect_features(path: Path) -> dict[str, object]:
    files = _iter_files(path)
    names = {file.name for file in files}
    rel_files = {_safe_rel(file) for file in files}
    pdf_count = sum(1 for file in files if file.suffix.lower() == ".pdf")
    assignment_count = sum(1 for file in files if "assignment" in file.name.lower())
    axis_term_table_count = sum(
        1
        for file in files
        if file.suffix.lower() == ".csv" and "axis" in file.name.lower() and "term" in file.name.lower()
    )
    has_method_split_pdfs = any(
        "_by_method" in rel_file and rel_file.endswith(".pdf") for rel_file in rel_files
    )
    has_go_ic_ranking = any(
        file.name in {"go_ic_tree_ranking.csv", "allgo_new_quality_aware_go_ic_tree_ranking.csv"}
        or file.name.endswith("_tree_ranking.csv")
        for file in files
    )
    summary_values = _read_summary_values(path)
    features: dict[str, object] = {
        "path": str(path),
        "exists": path.exists(),
        "pdf_count": pdf_count,
        "assignment_count": assignment_count,
        "axis_term_table_count": axis_term_table_count,
        "has_assignments": assignment_count > 0,
        "has_quality_report_pdf": "feature_matrix_quality_report.pdf" in names,
        "has_matrix_quality_summary": "matrix_quality_summary.csv" in names,
        "has_method_tree_matrix_summary": "method_tree_matrix_summary.csv" in names,
        "has_go_ic_ranking": has_go_ic_ranking,
        "has_current_subspace_ranking": (
            "current_adaptive_diffusion_subspace_tree_ranking.csv" in names
        ),
        "has_pipeline_manifest": "pipeline_manifest.json" in names,
        "has_connected_manifest": "connected_results_manifest.json" in names,
        "has_artifact_index": "ARTIFACT_INDEX.md" in names or "artifact_index.csv" in names,
        "has_subspaces_dir": (path / "subspaces").exists(),
        "has_method_split_pdfs": has_method_split_pdfs,
        **summary_values,
    }
    level = infer_level(features)
    features.update(
        {
            "level_code": level.code,
            "level_name": level.name,
            "level_order": level.order,
            "is_canonical": level.is_canonical,
            "interpretation": level.interpretation,
            "detected_inconsistency": level.inconsistency,
        }
    )
    return features


def default_candidate_paths(search_roots: Sequence[Path]) -> list[Path]:
    candidates: list[Path] = []
    needles = ("allgo", "go_ic", "feature_matrix_quality", "current_adaptive")
    for root in search_roots:
        if not root.exists():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            lowered = child.name.lower()
            if any(needle in lowered for needle in needles):
                candidates.append(child)
    return candidates


def audit_paths(paths: Sequence[Path]) -> pd.DataFrame:
    rows = [collect_features(path) for path in paths]
    if not rows:
        return pd.DataFrame(
            columns=[
                "path",
                "level_code",
                "level_name",
                "is_canonical",
                "detected_inconsistency",
            ]
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values(["level_order", "path"]).reset_index(drop=True)


def write_audit_outputs(frame: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "go_annotation_analysis_level_inventory.csv"
    json_path = output_dir / "go_annotation_analysis_level_inventory.json"
    md_path = output_dir / "go_annotation_analysis_level_summary.md"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(frame.to_dict(orient="records"), indent=2, default=str),
        encoding="utf-8",
    )

    level_counts = (
        frame.groupby(["level_code", "level_name"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("level_code")
    )
    lines = [
        "# GO Annotation Analysis Level Audit",
        "",
        "Canonical target:",
        (
            "- `30_canonical_current_subspace_pipeline`: current TBS gates on "
            "adaptive-diffusion cosine subspace trees with connected rankings, "
            "manifests, per-subspace artifacts, and method PDFs."
        ),
        "",
        "## Levels Found",
        "",
    ]
    if level_counts.empty:
        lines.append("No output directories were audited.")
    else:
        for row in level_counts.itertuples(index=False):
            lines.append(f"- `{row.level_code}` ({row.level_name}): {int(row.count)}")
    lines.extend(["", "## Inconsistent Levels", ""])
    for row in frame.itertuples(index=False):
        lines.extend(
            [
                f"### `{row.level_code}`",
                f"- Path: `{row.path}`",
                f"- Canonical: `{bool(row.is_canonical)}`",
                f"- Interpretation: {row.interpretation}",
                f"- Inconsistency: {row.detected_inconsistency}",
                (
                    "- Signals: "
                    f"pdfs={int(row.pdf_count)}, assignments={int(row.assignment_count)}, "
                    f"axis_term_tables={int(row.axis_term_table_count)}, "
                    f"methods=`{row.method_versions or row.method_families}`, "
                    f"tree_geometries=`{row.tree_geometries}`"
                ),
                "",
            ]
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "csv": str(csv_path),
        "json": str(json_path),
        "markdown": str(md_path),
    }


def main() -> None:
    args = parse_args()
    paths = args.paths if args.paths else default_candidate_paths(args.search_roots)
    frame = audit_paths(paths)
    outputs = write_audit_outputs(frame, args.output_dir)
    print(json.dumps({"audited_paths": len(frame), "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
