"""Build a semantic panel for Julia tree and lens strategies.

This diagnostic joins the existing Julia context-quality, lens-refinement,
alpha-sweep, feature-axis, p-value-continuity, and radius/angle/action outputs.
It is an interpretation surface only; it does not change clustering, traversal,
or calibration behavior.
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA_VERSION = "tree_strategy_semantic_panel/v1"

DEFAULT_BLOB_RESULTS_DIR = Path(
    "benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis"
)
DEFAULT_DIAGNOSTICS_RESULTS_DIR = Path("benchmarks/results/diagnostics")

PANEL_COLUMNS = [
    "tree_strategy",
    "lens_family",
    "linkage",
    "alpha",
    "n_clusters",
    "singleton_fraction",
    "medium_contexts",
    "GO Jaccard",
    "enrichment_fraction",
    "main_context_refinement",
    "axis_feature_bridge",
    "edge/sibling p-value continuity",
    "radius_angle_action",
]

EXTRA_COLUMNS = [
    "semantic_interpretation",
    "source",
    "status",
    "lens",
    "lens_id",
    "schema_version",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join existing Julia tree/lens diagnostics into one semantic panel."
    )
    parser.add_argument("--blob-results-dir", type=Path, default=DEFAULT_BLOB_RESULTS_DIR)
    parser.add_argument(
        "--diagnostics-results-dir",
        type=Path,
        default=DEFAULT_DIAGNOSTICS_RESULTS_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--alpha-summary",
        action="append",
        type=Path,
        default=None,
        help="Optional additional KAK lens alpha/linkage summary CSV, e.g. AWS merged output.",
    )
    parser.add_argument("--include-failed", action="store_true")
    return parser.parse_args()


def default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    return Path("benchmarks/results/diagnostics") / f"tree_strategy_semantic_panel_{stamp}"


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def finite_float(value: object) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(parsed) if np.isfinite(parsed) else math.nan


def format_float(value: object, digits: int = 6) -> str:
    parsed = finite_float(value)
    return "" if not math.isfinite(parsed) else f"{parsed:.{digits}g}"


def method_to_lens(method: str) -> tuple[str, str, str]:
    if method == "full_adaptive_diffusion":
        return "main_diffusion", "full_adaptive_diffusion", "full_adaptive_diffusion"
    if method == "full_fixed_diffusion":
        return "main_diffusion", "full_fixed_diffusion", "full_fixed_diffusion"
    if method == "paper_cosine_k20":
        return "paper_cosine", "k20", "paper_cosine_k20"
    if method.startswith("raw_kak_"):
        tail = method.removeprefix("raw_kak_")
        weighting, start, end = tail.split("_")
        lens = f"{weighting}__adaptive_modes_{start}_{end}"
        return "raw_kak", lens, f"raw_kak__{lens}"
    if method.startswith("sep_fixed_"):
        tail = method.removeprefix("sep_fixed_")
        weighting, start, end = tail.split("_")
        lens = f"{weighting}__adaptive_modes_{start}_{end}"
        return "sep_fixed_diffusion", lens, f"sep_fixed_diffusion__{lens}"
    if method.startswith("sep_adaptive_"):
        tail = method.removeprefix("sep_adaptive_")
        weighting, start, end = tail.split("_")
        lens = f"{weighting}__adaptive_modes_{start}_{end}"
        return "sep_adaptive_diffusion", lens, f"sep_adaptive_diffusion__{lens}"
    return "unknown", method, method


def lens_id_from_parts(lens_family: object, weighting: object, block_name: object) -> str:
    family = str(lens_family)
    return f"{family}__{weighting}__{block_name}"


def lens_id_from_lens(family: object, lens: object) -> str:
    return f"{family}__{lens}"


def lens_key_from_id(lens_id: object) -> tuple[str, str]:
    text = str(lens_id)
    parts = text.split("__", 1)
    if len(parts) != 2:
        return text, text
    return parts[0], parts[1]


def build_refinement_lookup(table: pd.DataFrame) -> dict[tuple[str, str], float]:
    if table.empty:
        return {}
    required = {"lens_family", "lens", "lens_refinement_score"}
    if not required.issubset(table.columns):
        return {}
    return {
        (str(row.lens_family), str(row.lens)): finite_float(row.lens_refinement_score)
        for row in table.itertuples(index=False)
    }


def build_axis_bridge_lookup(table: pd.DataFrame) -> dict[str, str]:
    if table.empty or "lens_id" not in table.columns:
        return {}
    lookup: dict[str, str] = {}
    for row in table.itertuples(index=False):
        feature = getattr(row, "top_axis_connection_feature", "")
        score = format_float(getattr(row, "top_axis_connection_score", math.nan), digits=4)
        if feature:
            suffix = f" score={score}" if score else ""
            lookup[str(row.lens_id)] = f"{feature}{suffix}"
    return lookup


def build_pvalue_continuity(summary: pd.DataFrame) -> str:
    if summary.empty or "summary_id" not in summary.columns:
        return ""
    values = {
        str(row.summary_id): finite_float(row.value)
        for row in summary.itertuples(index=False)
        if getattr(row, "scope", "") == "global"
    }
    pieces = []
    for key, label in [
        ("edge_parent_sibling_pvalue_coupling", "edge_parent"),
        ("recursive_sibling_pvalue_continuity", "recursive_sibling"),
        ("raw_recursive_sibling_pvalue_continuity", "raw_recursive_sibling"),
    ]:
        if key in values and math.isfinite(values[key]):
            pieces.append(f"{label}={values[key]:.6f}")
    return "; ".join(pieces)


def build_radius_action_lookup(
    block_geometry: pd.DataFrame,
    action_summary: pd.DataFrame,
) -> dict[str, str]:
    radius_auc = math.nan
    if not action_summary.empty and {"model_id", "median_auc"}.issubset(action_summary.columns):
        match = action_summary[action_summary["model_id"].eq("radius_angle_action")]
        if not match.empty:
            radius_auc = finite_float(match.iloc[0]["median_auc"])

    lookup: dict[str, str] = {}
    if block_geometry.empty or "run_id" not in block_geometry.columns:
        return lookup
    for row in block_geometry.itertuples(index=False):
        run_id = str(row.run_id)
        pieces = []
        if math.isfinite(radius_auc):
            pieces.append(f"auc={radius_auc:.6f}")
        angle = format_float(getattr(row, "angle_to_leading_axis_deg_median", math.nan), digits=5)
        independent = format_float(getattr(row, "independent_fraction_median", math.nan), digits=5)
        separation = format_float(
            getattr(row, "sibling_separation_parent_ratio_median", math.nan),
            digits=5,
        )
        if angle:
            pieces.append(f"angle_q50={angle}")
        if independent:
            pieces.append(f"independent_q50={independent}")
        if separation:
            pieces.append(f"sep_parent_q50={separation}")
        lookup[run_id] = "; ".join(pieces)
    return lookup


def semantic_interpretation(row: dict[str, object]) -> str:
    status = str(row.get("status", "ok"))
    if status and status != "ok":
        return "fail_closed_or_unusable"
    singleton = finite_float(row.get("singleton_fraction"))
    clusters = finite_float(row.get("n_clusters"))
    refinement = finite_float(row.get("main_context_refinement"))
    jaccard = finite_float(row.get("GO Jaccard"))
    enrichment = finite_float(row.get("enrichment_fraction"))
    linkage = str(row.get("linkage", ""))
    family = str(row.get("lens_family", ""))

    if linkage in {"complete", "ward"} and singleton >= 0.5:
        return "fragmentation_lens"
    if clusters >= 250 and singleton >= 0.1:
        return "fragmentation_lens"
    if refinement >= 0.9 and singleton < 0.1:
        return "fine_lens"
    if family == "main_diffusion" and jaccard >= 0.12 and enrichment >= 0.9:
        return "main_context_tree"
    if family in {"raw_kak", "sep_fixed_diffusion", "sep_adaptive_diffusion"}:
        return "diagnostic_lens"
    if family == "paper_cosine":
        return "coarse_baseline"
    return "comparison_tree"


def row_with_defaults(**values: object) -> dict[str, object]:
    row = {column: "" for column in PANEL_COLUMNS + EXTRA_COLUMNS}
    row.update(values)
    row["schema_version"] = SCHEMA_VERSION
    row["semantic_interpretation"] = semantic_interpretation(row)
    return row


def context_rows(
    context_quality: pd.DataFrame,
    *,
    refinement_lookup: dict[tuple[str, str], float],
    axis_bridge_lookup: dict[str, str],
    pvalue_continuity: str,
    radius_action_lookup: dict[str, str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if context_quality.empty:
        return rows
    for record in context_quality.itertuples(index=False):
        method = str(record.method)
        family, lens, lens_id = method_to_lens(method)
        radius_key = lens.replace("__", "__")
        rows.append(
            row_with_defaults(
                tree_strategy=method,
                lens_family=family,
                linkage="method_default",
                alpha="default",
                n_clusters=int(getattr(record, "n_clusters")),
                singleton_fraction=finite_float(getattr(record, "gene_singleton_fraction")),
                medium_contexts=int(getattr(record, "medium_context_clusters_size_5_80")),
                **{
                    "GO Jaccard": finite_float(
                        getattr(record, "mean_non_singleton_jaccard_gene_weighted")
                    ),
                    "enrichment_fraction": finite_float(
                        getattr(record, "enriched_non_singleton_fraction_q05")
                    ),
                    "main_context_refinement": refinement_lookup.get((family, lens), math.nan),
                    "axis_feature_bridge": axis_bridge_lookup.get(lens_id, ""),
                    "edge/sibling p-value continuity": pvalue_continuity,
                    "radius_angle_action": radius_action_lookup.get(radius_key, ""),
                },
                source="context_quality",
                status="ok",
                lens=lens,
                lens_id=lens_id,
            )
        )
    return rows


def alpha_rows(
    alpha_summary: pd.DataFrame,
    *,
    source: str,
    refinement_lookup: dict[tuple[str, str], float],
    axis_bridge_lookup: dict[str, str],
    pvalue_continuity: str,
    radius_action_lookup: dict[str, str],
    include_failed: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if alpha_summary.empty:
        return rows
    for record in alpha_summary.itertuples(index=False):
        status = str(getattr(record, "status", getattr(record, "row_status", "ok")))
        if status != "ok" and not include_failed:
            continue
        lens_id = str(getattr(record, "lens_id", ""))
        family, lens = lens_key_from_id(lens_id)
        linkage = str(getattr(record, "tree_linkage_method", "average"))
        if linkage == "nan":
            linkage = "average"
        alpha = (
            f"edge={format_float(getattr(record, 'edge_alpha', math.nan), digits=4)}; "
            f"sibling={format_float(getattr(record, 'sibling_alpha', math.nan), digits=4)}"
        )
        score = finite_float(getattr(record, "alpha_lens_score", math.nan))
        if not math.isfinite(score):
            score = refinement_lookup.get((family, lens), math.nan)
        run_id = ""
        if hasattr(record, "weighting") and hasattr(record, "block_name"):
            run_id = f"{record.weighting}__{record.block_name}"
        elif "__" in lens:
            run_id = lens
        rows.append(
            row_with_defaults(
                tree_strategy="lens_alpha_sweep",
                lens_family=family,
                linkage=linkage,
                alpha=alpha,
                n_clusters=finite_float(getattr(record, "n_clusters", math.nan)),
                singleton_fraction=finite_float(
                    getattr(record, "gene_singleton_fraction", math.nan)
                ),
                medium_contexts="",
                **{
                    "GO Jaccard": math.nan,
                    "enrichment_fraction": math.nan,
                    "main_context_refinement": score,
                    "axis_feature_bridge": axis_bridge_lookup.get(lens_id, ""),
                    "edge/sibling p-value continuity": pvalue_continuity,
                    "radius_angle_action": radius_action_lookup.get(run_id, ""),
                },
                source=source,
                status=status,
                lens=lens,
                lens_id=lens_id,
            )
        )
    return rows


def feature_axis_rows(
    feature_axis: pd.DataFrame,
    *,
    refinement_lookup: dict[tuple[str, str], float],
    pvalue_continuity: str,
    radius_action_lookup: dict[str, str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if feature_axis.empty:
        return rows
    for record in feature_axis.itertuples(index=False):
        lens_id = str(record.lens_id)
        family, lens = lens_key_from_id(lens_id)
        run_id = lens
        bridge = build_axis_bridge_lookup(feature_axis).get(lens_id, "")
        rows.append(
            row_with_defaults(
                tree_strategy="feature_axis_clustered_lens",
                lens_family=family,
                linkage=str(getattr(record, "tree_linkage_method", "average")),
                alpha=(
                    f"edge={format_float(getattr(record, 'edge_alpha', math.nan), digits=4)}; "
                    f"sibling={format_float(getattr(record, 'sibling_alpha', math.nan), digits=4)}"
                ),
                n_clusters=int(getattr(record, "n_clusters")),
                singleton_fraction=finite_float(getattr(record, "singleton_gene_fraction")),
                medium_contexts="",
                **{
                    "GO Jaccard": math.nan,
                    "enrichment_fraction": math.nan,
                    "main_context_refinement": refinement_lookup.get((family, lens), math.nan),
                    "axis_feature_bridge": bridge,
                    "edge/sibling p-value continuity": pvalue_continuity,
                    "radius_angle_action": radius_action_lookup.get(run_id, ""),
                },
                source="feature_axis_debug",
                status="ok",
                lens=lens,
                lens_id=lens_id,
            )
        )
    return rows


def default_alpha_paths(blob_results_dir: Path) -> list[Path]:
    return [
        blob_results_dir / "15_kak_lens_alpha_sweep_20260610" / "kak_lens_alpha_sweep_summary.csv",
        blob_results_dir
        / "18_kak_lens_linkage_complete_smoke_20260611"
        / "kak_lens_alpha_sweep_summary.csv",
    ]


def build_semantic_panel(
    *,
    blob_results_dir: Path = DEFAULT_BLOB_RESULTS_DIR,
    diagnostics_results_dir: Path = DEFAULT_DIAGNOSTICS_RESULTS_DIR,
    alpha_summary_paths: list[Path] | None = None,
    include_failed: bool = False,
) -> pd.DataFrame:
    context_quality = read_table(
        blob_results_dir
        / "12_context_quality_tree_comparison_20260610"
        / "context_method_quality_summary.csv"
    )
    lens_refinement = read_table(
        blob_results_dir
        / "13_subspace_lens_vs_main_adaptive_contexts_20260610"
        / "subspace_lens_vs_main_context_summary.csv"
    )
    feature_axis = read_table(
        blob_results_dir
        / "21_kak_lens_feature_axis_clustering_debug_20260612"
        / "kak_lens_feature_axis_clustering_summary.csv"
    )
    pvalue_summary = read_table(
        diagnostics_results_dir
        / "recursive_pvalue_geometry_full_20260606"
        / "recursive_pvalue_geometry_summary.csv"
    )
    radius_summary = read_table(
        diagnostics_results_dir
        / "path_conditioned_barycentric_action_20260606_guard_utility"
        / "kak_radius_angle_action_summary.csv"
    )
    radius_blocks = read_table(
        diagnostics_results_dir
        / "path_conditioned_barycentric_action_20260606_guard_utility"
        / "kak_internal_geometry_block_summary.csv"
    )

    refinement_lookup = build_refinement_lookup(lens_refinement)
    axis_bridge_lookup = build_axis_bridge_lookup(feature_axis)
    pvalue_continuity = build_pvalue_continuity(pvalue_summary)
    radius_action_lookup = build_radius_action_lookup(radius_blocks, radius_summary)

    rows: list[dict[str, object]] = []
    rows.extend(
        context_rows(
            context_quality,
            refinement_lookup=refinement_lookup,
            axis_bridge_lookup=axis_bridge_lookup,
            pvalue_continuity=pvalue_continuity,
            radius_action_lookup=radius_action_lookup,
        )
    )
    paths = default_alpha_paths(blob_results_dir)
    if alpha_summary_paths:
        paths.extend(alpha_summary_paths)
    for path in paths:
        rows.extend(
            alpha_rows(
                read_table(path),
                source=str(path),
                refinement_lookup=refinement_lookup,
                axis_bridge_lookup=axis_bridge_lookup,
                pvalue_continuity=pvalue_continuity,
                radius_action_lookup=radius_action_lookup,
                include_failed=include_failed,
            )
        )
    rows.extend(
        feature_axis_rows(
            feature_axis,
            refinement_lookup=refinement_lookup,
            pvalue_continuity=pvalue_continuity,
            radius_action_lookup=radius_action_lookup,
        )
    )
    panel = pd.DataFrame.from_records(rows, columns=PANEL_COLUMNS + EXTRA_COLUMNS)
    if panel.empty:
        return panel
    panel = panel.drop_duplicates(
        subset=["tree_strategy", "lens_id", "linkage", "alpha"],
        keep="first",
    )
    return panel.sort_values(
        ["semantic_interpretation", "lens_family", "n_clusters", "alpha"],
        na_position="last",
    ).reset_index(drop=True)


def write_report(panel: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "# Tree Strategy Semantic Panel",
        "",
        "This diagnostic joins existing Julia strategy outputs into one interpretation surface.",
        "It is diagnostic-only and does not promote any tree, lens, or calibration rule.",
        "",
        f"- rows: `{len(panel)}`",
        f"- schema_version: `{SCHEMA_VERSION}`",
        "",
    ]
    if not panel.empty:
        counts = panel["semantic_interpretation"].value_counts().sort_index()
        lines.extend(["## Semantic Roles", "", counts.to_markdown(), ""])
        requested = panel[PANEL_COLUMNS].copy()
        display = requested.head(25)
        lines.extend(["## First Rows", "", display.to_markdown(index=False), ""])

        ranked = panel.copy()
        ranked["singleton_fraction_numeric"] = pd.to_numeric(
            ranked["singleton_fraction"],
            errors="coerce",
        )
        ranked["refinement_numeric"] = pd.to_numeric(
            ranked["main_context_refinement"],
            errors="coerce",
        )
        top_refinement = ranked.sort_values(
            ["refinement_numeric", "singleton_fraction_numeric"],
            ascending=[False, True],
            na_position="last",
        ).head(10)
        lines.extend(
            [
                "## Top Refinement Rows",
                "",
                top_refinement[
                    [
                        "tree_strategy",
                        "lens_id",
                        "linkage",
                        "alpha",
                        "n_clusters",
                        "singleton_fraction",
                        "main_context_refinement",
                        "semantic_interpretation",
                    ]
                ].to_markdown(index=False),
                "",
            ]
        )
    (output_dir / "tree_strategy_semantic_panel_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = build_semantic_panel(
        blob_results_dir=args.blob_results_dir,
        diagnostics_results_dir=args.diagnostics_results_dir,
        alpha_summary_paths=args.alpha_summary,
        include_failed=bool(args.include_failed),
    )
    panel.to_csv(output_dir / "tree_strategy_semantic_panel.csv", index=False)
    write_report(panel, output_dir)
    print(f"Wrote tree strategy semantic panel: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
