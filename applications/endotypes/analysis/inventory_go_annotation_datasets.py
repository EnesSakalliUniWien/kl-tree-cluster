#!/usr/bin/env python3
"""Inventory GO annotation feature matrices and result-root naming.

This is a lightweight preflight for the GO annotation analysis pipeline. It
finds available feature-matrix datasets, checks matrix validity, detects
duplicate copies, and records whether result roots follow the canonical naming
contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

CANONICAL_FEATURE_ROOT = Path("data/feature_matrices")
DEFAULT_RESULT_ROOTS = (Path("results/analyses"), Path("raw/assets/benchmark-results"))


@dataclass(frozen=True)
class ResultKind:
    code: str
    preferred_root: str
    description: str


RESULT_KINDS = (
    ResultKind(
        "go_annotation_pipeline",
        "results/analyses/<dataset_slug>_go_annotation_pipeline_<YYYYMMDD_HHMMSS>",
        "Canonical wrapper root with dataset inventory, quality, current subspace, optional audit, and level audit stages.",
    ),
    ResultKind(
        "current_adaptive_diffusion_subspace_tree",
        "results/analyses/<dataset_slug>_current_adaptive_diffusion_subspace_tree_<YYYYMMDD_HHMMSS>",
        "Direct current-method adaptive-diffusion cosine-subspace tree run.",
    ),
    ResultKind(
        "feature_matrix_quality",
        "raw/assets/benchmark-results/<dataset_slug>_feature_matrix_quality_<YYYYMMDD>",
        "Retained evidence for matrix-quality-only reports.",
    ),
    ResultKind(
        "method_version_tree_matrix",
        "raw/assets/benchmark-results/<dataset_slug>_method_version_tree_matrix_<suffix>_<YYYYMMDD>",
        "Candidate-generation or method/tree-geometry audit matrix.",
    ),
    ResultKind(
        "go_ic_tree_plots",
        "raw/assets/benchmark-results/<dataset_slug>_go_ic_tree_plots_<YYYYMMDD>",
        "Historical mixed-method GO-IC reader deck; not the canonical final method level.",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, default=CANONICAL_FEATURE_ROOT)
    parser.add_argument(
        "--extra-paths",
        nargs="*",
        type=Path,
        default=[],
        help="Additional matrix paths to inspect, such as root-level duplicates.",
    )
    parser.add_argument(
        "--result-roots",
        nargs="*",
        type=Path,
        default=list(DEFAULT_RESULT_ROOTS),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def safe_slug(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"\.[A-Za-z0-9]+$", "", text)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text.lower() or "dataset"


def dataset_slug_from_matrix_name(path: Path) -> str:
    stem = path.stem
    if stem == "feature_matrix":
        return "feature_matrix"
    if stem.startswith("feature_matrix_"):
        stem = stem[len("feature_matrix_") :]
    elif stem.endswith("_feature_matrix"):
        stem = stem[: -len("_feature_matrix")]
    elif "_feature_matrix_" in stem:
        prefix, suffix = stem.split("_feature_matrix_", 1)
        stem = f"{prefix}_{suffix}"
    return safe_slug(stem)


def canonical_matrix_name(path: Path) -> str:
    if path.name == "feature_matrix.tsv":
        return path.name
    if path.name.startswith("feature_matrix_") and re.search(r"[\s()]", path.name) is None:
        return path.name
    return f"feature_matrix_{dataset_slug_from_matrix_name(path)}.tsv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tracked_paths(paths: Sequence[Path]) -> set[str]:
    if not paths:
        return set()
    try:
        repo_root = Path(
            subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except Exception:
        repo_root = Path.cwd()
    rel_paths = []
    rel_lookup: dict[str, str] = {}
    for path in paths:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(repo_root)
        except ValueError:
            continue
        rel = str(relative)
        rel_paths.append(rel)
        rel_lookup[rel] = str(path)
    if not rel_paths:
        return set()
    try:
        completed = subprocess.run(
            ["git", "ls-files", "--", *rel_paths],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return set()
    return {
        rel_lookup.get(line.strip(), line.strip())
        for line in completed.stdout.splitlines()
        if line.strip()
    }


def matrix_paths(feature_root: Path, extra_paths: Sequence[Path]) -> list[Path]:
    paths = []
    if feature_root.exists():
        paths.extend(sorted(feature_root.glob("*.tsv")))
    paths.extend(path for path in extra_paths if path.exists())
    return list(dict.fromkeys(paths))


def matrix_inventory(feature_root: Path, extra_paths: Sequence[Path]) -> pd.DataFrame:
    paths = matrix_paths(feature_root, extra_paths)
    tracked = tracked_paths(paths)
    rows: list[dict[str, object]] = []
    digest_to_paths: dict[str, list[Path]] = defaultdict(list)

    for path in paths:
        digest = sha256_file(path)
        digest_to_paths[digest].append(path)
        try:
            data = pd.read_csv(path, sep="\t", index_col=0)
            numeric = data.apply(pd.to_numeric, errors="raise")
            values = numeric.to_numpy()
            finite_values = values[~pd.isna(values)]
            binary = bool(np.isin(finite_values, [0, 1]).all())
            missing = int(pd.isna(values).sum())
            zero_rows = int((numeric.sum(axis=1) == 0).sum())
            zero_cols = int((numeric.sum(axis=0) == 0).sum())
            rows_count, columns_count = numeric.shape
            density = float(numeric.to_numpy(dtype=float).mean()) if numeric.size else 0.0
            load_error = ""
        except Exception as exc:  # noqa: BLE001 - inventory should record bad inputs.
            binary = False
            missing = -1
            zero_rows = -1
            zero_cols = -1
            rows_count = -1
            columns_count = -1
            density = float("nan")
            load_error = repr(exc)

        canonical_name = canonical_matrix_name(path)
        in_feature_root = path.parent == feature_root
        if in_feature_root and path.name == canonical_name:
            naming_status = "canonical"
        elif in_feature_root:
            naming_status = "legacy_tracked_name"
        else:
            naming_status = "outside_feature_root"

        rows.append(
            {
                "path": str(path),
                "dataset_slug": dataset_slug_from_matrix_name(path),
                "recommended_filename": canonical_name,
                "recommended_path": str(feature_root / canonical_name),
                "n_rows": rows_count,
                "n_features": columns_count,
                "binary_0_1": binary,
                "missing_values": missing,
                "zero_rows": zero_rows,
                "zero_columns": zero_cols,
                "density": density,
                "sha256": digest,
                "tracked_by_git": str(path) in tracked,
                "naming_status": naming_status,
                "load_error": load_error,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    duplicate_counts = {
        digest: len(paths_for_digest) for digest, paths_for_digest in digest_to_paths.items()
    }
    frame["duplicate_file_count"] = frame["sha256"].map(duplicate_counts).astype(int)
    frame["has_duplicate_copy"] = frame["duplicate_file_count"] > 1
    frame["usable_for_go_pipeline"] = (
        frame["binary_0_1"]
        & frame["missing_values"].eq(0)
        & frame["zero_rows"].eq(0)
        & frame["zero_columns"].eq(0)
    )
    return frame.sort_values(["naming_status", "dataset_slug", "path"]).reset_index(drop=True)


def duplicate_inventory(matrix_frame: pd.DataFrame) -> pd.DataFrame:
    if matrix_frame.empty or "sha256" not in matrix_frame:
        return pd.DataFrame(columns=["sha256", "paths", "canonical_candidates", "n_copies"])
    rows = []
    for digest, group in matrix_frame.groupby("sha256"):
        if len(group) <= 1:
            continue
        rows.append(
            {
                "sha256": digest,
                "n_copies": int(len(group)),
                "paths": ";".join(group["path"].astype(str)),
                "canonical_candidates": ";".join(group["recommended_path"].astype(str).unique()),
            }
        )
    return pd.DataFrame(rows)


def infer_result_kind(path: Path) -> str:
    name = path.name
    for kind in RESULT_KINDS:
        if kind.code in name:
            return kind.code
    if (path / "pipeline_manifest.json").exists():
        return "go_annotation_pipeline"
    if (path / "connected_results_manifest.json").exists():
        return "current_adaptive_diffusion_subspace_tree"
    if (path / "matrix_quality_summary.csv").exists():
        return "feature_matrix_quality"
    if (path / "method_tree_matrix_summary.csv").exists():
        return "method_version_tree_matrix"
    if list(path.rglob("*tree_ranking.csv")):
        return "go_ic_tree_plots"
    return "unclassified"


def infer_dataset_slug_from_result(path: Path, kind: str) -> str:
    name = path.name
    suffixes = [
        f"_{kind}",
        "_current_adaptive_diffusion_subspace_tree",
        "_go_annotation_pipeline",
        "_feature_matrix_quality",
        "_method_version_tree_matrix",
        "_go_ic_tree_plots",
        "_c2ef_validate_cosine_subspace_split_legacy",
    ]
    no_date = re.sub(r"_20\d{6}(?:_\d{6})?$", "", name)
    for suffix in suffixes:
        if suffix in no_date:
            return safe_slug(no_date.split(suffix, 1)[0])
    config_path = path / "experiment_config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            config = {}
        input_path = config.get("input")
        if input_path:
            return dataset_slug_from_matrix_name(Path(str(input_path)))
    return safe_slug(no_date)


def result_root_candidates(result_roots: Sequence[Path]) -> list[Path]:
    candidates: list[Path] = []
    needles = (
        "allgo",
        "go_annotation",
        "go_ic",
        "feature_matrix_quality",
        "method_version_tree_matrix",
        "current_adaptive_diffusion_subspace_tree",
    )
    for root in result_roots:
        if not root.exists():
            continue
        for child in sorted(root.iterdir()):
            lowered = child.name.lower()
            if "analysis_level_audit" in lowered or "dataset_inventory" in lowered:
                continue
            if child.is_dir() and any(needle in lowered for needle in needles):
                candidates.append(child)
    return list(dict.fromkeys(candidates))


def result_root_inventory(result_roots: Sequence[Path]) -> pd.DataFrame:
    rows = []
    for path in result_root_candidates(result_roots):
        kind = infer_result_kind(path)
        dataset_slug = infer_dataset_slug_from_result(path, kind)
        if kind in {"go_annotation_pipeline", "current_adaptive_diffusion_subspace_tree"}:
            expected_prefix = f"{dataset_slug}_{kind}_"
            preferred_base = "results/analyses"
        else:
            expected_prefix = f"{dataset_slug}_{kind}_"
            preferred_base = "raw/assets/benchmark-results"
        rows.append(
            {
                "path": str(path),
                "result_kind": kind,
                "dataset_slug": dataset_slug,
                "expected_prefix": expected_prefix,
                "prefix_matches": path.name.startswith(expected_prefix),
                "preferred_base": preferred_base,
                "under_preferred_base": str(path).startswith(preferred_base),
                "has_pipeline_manifest": (path / "pipeline_manifest.json").exists(),
                "has_connected_manifest": (path / "connected_results_manifest.json").exists(),
                "has_subspaces": (path / "subspaces").exists(),
                "has_rankings": (path / "rankings").exists(),
                "has_method_pdfs": any(
                    "_by_method" in str(file) and file.suffix == ".pdf"
                    for file in path.rglob("*.pdf")
                ),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["result_kind", "dataset_slug", "path"]).reset_index(drop=True)


def write_summary(
    output_dir: Path,
    matrix_frame: pd.DataFrame,
    duplicate_frame: pd.DataFrame,
    result_frame: pd.DataFrame,
) -> Path:
    summary_path = output_dir / "go_annotation_directory_naming_summary.md"
    usable = int(matrix_frame["usable_for_go_pipeline"].sum()) if not matrix_frame.empty else 0
    duplicate_count = len(duplicate_frame)
    result_mismatches = (
        int((~result_frame["prefix_matches"] | ~result_frame["under_preferred_base"]).sum())
        if not result_frame.empty
        else 0
    )
    lines = [
        "# GO Annotation Dataset and Directory Naming Inventory",
        "",
        "## Dataset Location Contract",
        "",
        "- Canonical reusable feature matrices live under `data/feature_matrices/`.",
        "- New GO annotation matrices should use `feature_matrix_<dataset_slug>.tsv`.",
        "- `<dataset_slug>` is lowercase snake case and contains no spaces, parentheses, or date suffixes.",
        "- Root-level copies and files with download suffixes such as `(1)` are non-canonical.",
        "- Large/private external inputs should first be captured or documented, then promoted to `data/feature_matrices/` before canonical reruns.",
        "",
        "## Result Directory Contract",
        "",
        "- Canonical wrapper runs: `results/analyses/<dataset_slug>_go_annotation_pipeline_<YYYYMMDD_HHMMSS>/`.",
        "- Direct current-method runs: `results/analyses/<dataset_slug>_current_adaptive_diffusion_subspace_tree_<YYYYMMDD_HHMMSS>/`.",
        "- Retained historical evidence under `raw/assets/benchmark-results/` should keep `<dataset_slug>_<analysis_kind>_<YYYYMMDD>/`.",
        "- The canonical current stage contains `rankings/`, `plots/`, `subspaces/<weighting>/<block_name>/`, `connected_results_manifest.json`, `ARTIFACT_INDEX.md`, `<dataset_slug>_quality_aware_go_ic_by_method/`, and `<dataset_slug>_quality_aware_go_ic_plots/`.",
        "- Reader-facing names should use `adaptive_diffusion_cosine_subspace` or `raw_cosine_subspace`, not internal `kak` labels.",
        "",
        "## Inventory Counts",
        "",
        f"- Matrices inspected: `{len(matrix_frame)}`",
        f"- Matrices directly usable for the GO pipeline: `{usable}`",
        f"- Duplicate-content groups: `{duplicate_count}`",
        f"- Result roots inspected: `{len(result_frame)}`",
        f"- Result roots with naming/base mismatches: `{result_mismatches}`",
        "",
        "## Matrices",
        "",
    ]
    if matrix_frame.empty:
        lines.append("No matrices found.")
    else:
        cols = [
            "path",
            "dataset_slug",
            "recommended_path",
            "n_rows",
            "n_features",
            "usable_for_go_pipeline",
            "tracked_by_git",
            "naming_status",
            "has_duplicate_copy",
        ]
        lines.extend(["```text", matrix_frame[cols].to_string(index=False), "```"])
    lines.extend(["", "## Duplicate Copies", ""])
    if duplicate_frame.empty:
        lines.append("No duplicate-content matrix groups found.")
    else:
        lines.extend(["```text", duplicate_frame.to_string(index=False), "```"])
    lines.extend(["", "## Result Roots", ""])
    if result_frame.empty:
        lines.append("No result roots found.")
    else:
        cols = [
            "path",
            "result_kind",
            "dataset_slug",
            "prefix_matches",
            "under_preferred_base",
            "has_connected_manifest",
            "has_method_pdfs",
        ]
        lines.extend(["```text", result_frame[cols].to_string(index=False), "```"])
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix_frame = matrix_inventory(args.feature_root, args.extra_paths)
    duplicate_frame = duplicate_inventory(matrix_frame)
    result_frame = result_root_inventory(args.result_roots)

    matrix_frame.to_csv(args.output_dir / "go_annotation_dataset_inventory.csv", index=False)
    duplicate_frame.to_csv(args.output_dir / "go_annotation_dataset_duplicates.csv", index=False)
    result_frame.to_csv(args.output_dir / "go_annotation_result_root_inventory.csv", index=False)
    summary_path = write_summary(args.output_dir, matrix_frame, duplicate_frame, result_frame)

    print(
        json.dumps(
            {
                "matrices": int(len(matrix_frame)),
                "duplicates": int(len(duplicate_frame)),
                "result_roots": int(len(result_frame)),
                "summary": str(summary_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
