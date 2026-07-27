"""Decompose edge and sibling p-values into functionals of the same z-vector."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from tree_break_selection.hierarchy_analysis.statistics.contrast_covariance import (
    compute_whitened_wald_contrast,
)
from tree_break_selection.hierarchy_analysis.statistics.projection.projected_wald.projected_wald_kernel import (
    run_projected_wald_kernel,
)
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.fixed_subspace_annotation import (
    fixed_subspace_sibling_p_value,
)
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.projection.gate_inputs.projection_dimensions import (
    derive_sibling_projection_dimensions_from_child_edge_comparisons,
)
from tree_break_selection.tree.feature_space import FeatureSpace

from benchmarks.diagnostics.calibration.selected_family_traversal_panel import (
    NULL_OUTPUT_ROLE,
    _generate_data_with_truth,
    _output_data_role,
    _profile_id_for_method,
    validate_methods,
)
from benchmarks.shared.runners.tbs_runner import _run_tbs_method
from benchmarks.validation.selected_edge_type1_geometry import (
    _case_contract,
    _select_cases,
    parse_names,
)

SCHEMA_VERSION = "same_z_functional_decomposition/v1"
STUDY_ROLE = "diagnostic_same_z_functional_decomposition_not_calibration"

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "method_id",
    "replicate",
    "data_seed",
    "parent_node_id",
    "left_child_id",
    "right_child_id",
    "n_left",
    "n_right",
    "n_parent",
    "sibling_z_l2_statistic",
    "sibling_z_max_abs",
    "sibling_z_effective_coordinate_count",
    "fixed_coordinate_bh_p_value",
    "fixed_global_chi_square_p_value",
    "fixed_block_bh_p_value",
    "projected_sibling_statistic",
    "projected_sibling_df",
    "projected_sibling_p_value",
    "projection_dimension",
    "parent_projection_dimension",
    "left_edge_statistic",
    "left_edge_df",
    "left_edge_p_value",
    "left_edge_bh_p_value",
    "right_edge_statistic",
    "right_edge_df",
    "right_edge_p_value",
    "right_edge_bh_p_value",
    "min_edge_bh_p_value",
    "max_edge_neglog10_bh",
    "projected_sibling_neglog10",
    "fixed_coordinate_neglog10",
    "fixed_global_neglog10",
    "edge_projected_sibling_neglog10_delta",
    "projected_fixed_coordinate_neglog10_delta",
    "fixed_global_fixed_coordinate_neglog10_delta",
    "scaling_class",
    "strategy_note",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "method_id",
    "replicate",
    "n_rows",
    "median_max_edge_neglog10_bh",
    "median_projected_sibling_neglog10",
    "median_fixed_coordinate_neglog10",
    "median_fixed_global_neglog10",
    "median_edge_projected_sibling_delta",
    "median_projected_fixed_coordinate_delta",
    "median_fixed_global_fixed_coordinate_delta",
    "dominant_scaling_class",
)


def _neglog10(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return math.nan
    if not math.isfinite(numeric):
        return math.nan
    return float(-math.log10(min(max(numeric, np.nextafter(0.0, 1.0)), 1.0)))


def _annotation_float(annotations: pd.DataFrame, node: object, column: str) -> float:
    if column not in annotations.columns or node not in annotations.index:
        return math.nan
    value = annotations.at[node, column]
    return math.nan if pd.isna(value) else float(value)


def _children_pair(tree: nx.DiGraph, parent: object) -> tuple[object, object] | None:
    children = list(tree.successors(parent))
    if len(children) != 2:
        return None
    return children[0], children[1]


def _node_leaf_count(tree: nx.DiGraph, node: object) -> int:
    return int(tree.nodes[node]["leaf_count"])


def _effective_coordinate_count(z: np.ndarray) -> float:
    energy = np.asarray(z, dtype=float) ** 2
    total = float(np.sum(energy))
    if total <= 0.0:
        return 0.0
    weights = energy / total
    return float(1.0 / np.sum(weights**2))


def classify_scaling_row(
    *,
    edge_projected_delta: float,
    projected_fixed_coordinate_delta: float,
    fixed_global_fixed_coordinate_delta: float,
) -> tuple[str, str]:
    """Return an interpretable label for the p-value scaling gap."""
    if (
        math.isfinite(edge_projected_delta)
        and abs(edge_projected_delta) <= 1.0
        and math.isfinite(projected_fixed_coordinate_delta)
        and projected_fixed_coordinate_delta >= 3.0
    ):
        return (
            "selected_projection_energy_vs_coordinate_bh",
            "Edge and projected sibling are on the same scale; fixed-coordinate BH is much weaker, indicating dense/directed energy rather than a raw contrast mismatch.",
        )
    if (
        math.isfinite(fixed_global_fixed_coordinate_delta)
        and fixed_global_fixed_coordinate_delta >= 3.0
    ):
        return (
            "l2_aggregation_vs_coordinate_bh",
            "Full fixed L2 energy is much stronger than coordinate BH, so the sibling contrast is diffuse across coordinates.",
        )
    if math.isfinite(edge_projected_delta) and edge_projected_delta >= 3.0:
        return (
            "edge_stronger_than_projected_sibling",
            "Outgoing edge evidence is stronger than the same-z projected sibling statistic; tree correction or projection policy is contributing additional scale.",
        )
    if math.isfinite(projected_fixed_coordinate_delta) and projected_fixed_coordinate_delta >= 1.0:
        return (
            "moderate_projection_coordinate_gap",
            "Projected sibling evidence is stronger than fixed-coordinate BH, but the gap is moderate.",
        )
    return (
        "same_scale_or_weak_gap",
        "No large decomposition gap is visible among edge, projected sibling, and fixed-coordinate sibling scales.",
    )


def same_z_rows_for_result(
    *,
    case_id: str,
    data_role: str,
    method_id: str,
    replicate: int,
    data_seed: int,
    result,
    feature_space: FeatureSpace,
) -> pd.DataFrame:
    """Return one row per binary parent comparing same-z p-value functionals."""
    tree = result.extra["tree"]
    annotations = result.extra["annotations"]
    edge_gate_result = result.extra["gate_bundle"].edge_gate_result
    spectral_context = edge_gate_result.spectral_context
    sibling_dimensions = derive_sibling_projection_dimensions_from_child_edge_comparisons(
        tree,
        spectral_context=spectral_context,
    )
    records: list[dict[str, object]] = []

    for parent in tree.nodes:
        children = _children_pair(tree, parent)
        if children is None:
            continue
        left, right = children
        z = compute_whitened_wald_contrast(
            np.asarray(tree.nodes[left]["distribution"], dtype=float),
            np.asarray(tree.nodes[right]["distribution"], dtype=float),
            float(_node_leaf_count(tree, left)),
            float(_node_leaf_count(tree, right)),
            comparison="sibling",
            feature_space=feature_space,
        )
        projection_dimension = int(sibling_dimensions[parent])
        parent_projection = spectral_context.principal_component_projections_by_node[parent]
        parent_eigenvalues = spectral_context.principal_component_eigenvalues_by_node[
            parent
        ]
        projected = run_projected_wald_kernel(
            z,
            spectral_k=projection_dimension,
            pca_projection=parent_projection,
            pca_eigenvalues=parent_eigenvalues,
        )

        fixed_coordinate = fixed_subspace_sibling_p_value(
            z,
            feature_space,
            method="fixed_coordinate_bh",
        )
        fixed_global = fixed_subspace_sibling_p_value(
            z,
            feature_space,
            method="fixed_global_chi_square",
        )
        fixed_block = fixed_subspace_sibling_p_value(
            z,
            feature_space,
            method="fixed_block_bh",
        )

        left_edge_bh = _annotation_float(
            annotations,
            left,
            "Child_Parent_Divergence_P_Value_BH",
        )
        right_edge_bh = _annotation_float(
            annotations,
            right,
            "Child_Parent_Divergence_P_Value_BH",
        )
        max_edge_neglog = max(_neglog10(left_edge_bh), _neglog10(right_edge_bh))
        projected_neglog = _neglog10(projected.p_value)
        fixed_coordinate_neglog = _neglog10(fixed_coordinate)
        fixed_global_neglog = _neglog10(fixed_global)
        edge_projected_delta = max_edge_neglog - projected_neglog
        projected_fixed_delta = projected_neglog - fixed_coordinate_neglog
        fixed_global_fixed_delta = fixed_global_neglog - fixed_coordinate_neglog
        scaling_class, strategy_note = classify_scaling_row(
            edge_projected_delta=edge_projected_delta,
            projected_fixed_coordinate_delta=projected_fixed_delta,
            fixed_global_fixed_coordinate_delta=fixed_global_fixed_delta,
        )

        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "case_id": case_id,
                "data_role": _output_data_role(data_role),
                "method_id": method_id,
                "replicate": int(replicate),
                "data_seed": int(data_seed),
                "parent_node_id": str(parent),
                "left_child_id": str(left),
                "right_child_id": str(right),
                "n_left": _node_leaf_count(tree, left),
                "n_right": _node_leaf_count(tree, right),
                "n_parent": _node_leaf_count(tree, parent),
                "sibling_z_l2_statistic": float(np.dot(z, z)),
                "sibling_z_max_abs": float(np.max(np.abs(z))) if z.size else 0.0,
                "sibling_z_effective_coordinate_count": _effective_coordinate_count(z),
                "fixed_coordinate_bh_p_value": fixed_coordinate,
                "fixed_global_chi_square_p_value": fixed_global,
                "fixed_block_bh_p_value": fixed_block,
                "projected_sibling_statistic": float(projected.statistic),
                "projected_sibling_df": float(projected.degrees_of_freedom),
                "projected_sibling_p_value": float(projected.p_value),
                "projection_dimension": projection_dimension,
                "parent_projection_dimension": int(parent_projection.shape[0]),
                "left_edge_statistic": _annotation_float(
                    annotations,
                    left,
                    "Child_Parent_Divergence_Test_Statistic",
                ),
                "left_edge_df": _annotation_float(
                    annotations,
                    left,
                    "Child_Parent_Divergence_df",
                ),
                "left_edge_p_value": _annotation_float(
                    annotations,
                    left,
                    "Child_Parent_Divergence_P_Value",
                ),
                "left_edge_bh_p_value": left_edge_bh,
                "right_edge_statistic": _annotation_float(
                    annotations,
                    right,
                    "Child_Parent_Divergence_Test_Statistic",
                ),
                "right_edge_df": _annotation_float(
                    annotations,
                    right,
                    "Child_Parent_Divergence_df",
                ),
                "right_edge_p_value": _annotation_float(
                    annotations,
                    right,
                    "Child_Parent_Divergence_P_Value",
                ),
                "right_edge_bh_p_value": right_edge_bh,
                "min_edge_bh_p_value": min(left_edge_bh, right_edge_bh),
                "max_edge_neglog10_bh": max_edge_neglog,
                "projected_sibling_neglog10": projected_neglog,
                "fixed_coordinate_neglog10": fixed_coordinate_neglog,
                "fixed_global_neglog10": fixed_global_neglog,
                "edge_projected_sibling_neglog10_delta": edge_projected_delta,
                "projected_fixed_coordinate_neglog10_delta": projected_fixed_delta,
                "fixed_global_fixed_coordinate_neglog10_delta": fixed_global_fixed_delta,
                "scaling_class": scaling_class,
                "strategy_note": strategy_note,
            }
        )
    return pd.DataFrame.from_records(records, columns=ROW_COLUMNS)


def summarize_same_z_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize the functional decomposition by case, role, method, replicate."""
    summary_rows: list[dict[str, object]] = []
    group_columns = ["case_id", "data_role", "method_id", "replicate"]
    for key, group in rows.groupby(group_columns, dropna=False):
        case_id, data_role, method_id, replicate = key
        class_counts = group["scaling_class"].value_counts()
        dominant = str(class_counts.index[0]) if not class_counts.empty else ""
        summary_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "case_id": case_id,
                "data_role": data_role,
                "method_id": method_id,
                "replicate": int(replicate),
                "n_rows": int(group.shape[0]),
                "median_max_edge_neglog10_bh": float(
                    group["max_edge_neglog10_bh"].median()
                ),
                "median_projected_sibling_neglog10": float(
                    group["projected_sibling_neglog10"].median()
                ),
                "median_fixed_coordinate_neglog10": float(
                    group["fixed_coordinate_neglog10"].median()
                ),
                "median_fixed_global_neglog10": float(
                    group["fixed_global_neglog10"].median()
                ),
                "median_edge_projected_sibling_delta": float(
                    group["edge_projected_sibling_neglog10_delta"].median()
                ),
                "median_projected_fixed_coordinate_delta": float(
                    group["projected_fixed_coordinate_neglog10_delta"].median()
                ),
                "median_fixed_global_fixed_coordinate_delta": float(
                    group["fixed_global_fixed_coordinate_neglog10_delta"].median()
                ),
                "dominant_scaling_class": dominant,
            }
        )
    return pd.DataFrame.from_records(summary_rows, columns=SUMMARY_COLUMNS)


def plot_same_z_functional_decomposition(
    rows: pd.DataFrame,
    *,
    output_path: Path,
    title: str = "Same-z Edge and Sibling Functional Decomposition",
) -> None:
    """Write a static plot exposing edge/sibling scale relationships."""
    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), constrained_layout=True)
    colors = {"selected_null": "#4c78a8", "signal": "#f58518"}

    plotted = rows[
        rows["max_edge_neglog10_bh"].notna()
        & rows["projected_sibling_neglog10"].notna()
        & rows["fixed_coordinate_neglog10"].notna()
    ].copy()
    for role, group in plotted.groupby("data_role"):
        color = colors.get(str(role), "#54a24b")
        axes[0, 0].scatter(
            group["projected_sibling_neglog10"],
            group["max_edge_neglog10_bh"],
            s=28,
            alpha=0.45,
            color=color,
            label=str(role),
        )
        axes[0, 1].scatter(
            group["fixed_coordinate_neglog10"],
            group["projected_sibling_neglog10"],
            s=28,
            alpha=0.45,
            color=color,
            label=str(role),
        )

    if not plotted.empty:
        max_identity = float(
            np.nanmax(
                plotted[["projected_sibling_neglog10", "max_edge_neglog10_bh"]].to_numpy()
            )
        )
        axes[0, 0].plot([0.0, max_identity], [0.0, max_identity], color="#222", lw=1)
    axes[0, 0].set_title("Projected sibling vs strongest outgoing edge")
    axes[0, 0].set_xlabel("projected sibling -log10 p")
    axes[0, 0].set_ylabel("max outgoing edge -log10 BH p")
    axes[0, 0].legend(fontsize=9)

    axes[0, 1].set_title("Selected projection vs fixed coordinate BH")
    axes[0, 1].set_xlabel("fixed-coordinate sibling -log10 p")
    axes[0, 1].set_ylabel("projected sibling -log10 p")
    axes[0, 1].legend(fontsize=9)

    functional_values = [
        ("edge_max_bh", rows["max_edge_neglog10_bh"]),
        ("projected_sibling", rows["projected_sibling_neglog10"]),
        ("fixed_global", rows["fixed_global_neglog10"]),
        ("fixed_coordinate", rows["fixed_coordinate_neglog10"]),
    ]
    box_values = [series.dropna().to_numpy(dtype=float) for _name, series in functional_values]
    box_labels = [name for name, _series in functional_values]
    box_positions = np.arange(1, len(box_values) + 1, dtype=float)
    boxes = axes[1, 0].boxplot(
        box_values,
        positions=box_positions,
        showfliers=False,
        patch_artist=True,
    )
    for patch, color in zip(boxes["boxes"], ["#4c78a8", "#72b7b2", "#54a24b", "#e45756"], strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
    axes[1, 0].set_title("Same-z p-value strength by functional")
    axes[1, 0].set_ylabel("-log10 p")
    axes[1, 0].set_xticks(box_positions, box_labels)
    axes[1, 0].tick_params(axis="x", labelrotation=20)

    class_counts = rows["scaling_class"].value_counts()
    axes[1, 1].barh(class_counts.index.astype(str), class_counts.to_numpy(dtype=float), color="#4c78a8")
    axes[1, 1].set_title("Dominant scaling explanations")
    axes[1, 1].set_xlabel("parent rows")

    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_same_z_html_report(
    *,
    output_path: Path,
    plot_path: Path,
    summary: pd.DataFrame,
    title: str = "Same-z Edge and Sibling Functional Decomposition",
) -> None:
    """Write a lightweight accessible report for same-z decomposition outputs."""
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45; margin: 2rem; color: #1f2933; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #d8dee9; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
    th, td {{ border: 1px solid #d8dee9; padding: 0.35rem 0.5rem; }}
    th {{ background: #f1f5f9; text-align: left; }}
    code {{ background: #f1f5f9; padding: 0.1rem 0.25rem; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p>
    Each row starts from the same parent sibling z-vector and compares four
    p-value functionals: outgoing edge projected evidence, projected sibling
    evidence, fixed-global sibling evidence, and fixed-coordinate BH sibling
    evidence.
  </p>
  <img src="{html.escape(plot_path.name)}" alt="Four-panel same-z diagnostic:
  projected sibling against edge evidence, projected sibling against
  fixed-coordinate evidence, p-value strength distributions by functional, and
  counts of scaling explanation classes.">
  <h2>Summary</h2>
  {summary.to_html(index=False, escape=True)}
</body>
</html>
"""
    output_path.write_text(body, encoding="utf-8")


def run_same_z_functional_decomposition(
    *,
    output_dir: Path,
    suite: str,
    case_names: tuple[str, ...],
    methods: tuple[str, ...],
    data_roles: tuple[str, ...],
    sibling_alpha: float,
    edge_alpha: float,
    replicates: int,
    base_seed: int,
) -> dict[str, Path]:
    """Run the same-z decomposition over a selected-family benchmark slice."""
    validated_methods = validate_methods(methods)
    cases = _select_cases(suite=suite, case_names=case_names)
    all_rows: list[pd.DataFrame] = []
    failure_rows: list[dict[str, object]] = []

    for case in cases:
        (
            case_id,
            source_family,
            feature_representation,
            n_samples,
            n_features,
            n_categories,
        ) = _case_contract(case)
        for replicate in range(int(replicates)):
            data_seed = int(base_seed) + int(replicate)
            for data_role in data_roles:
                data, feature_space, _truth_labels, _true_clusters = _generate_data_with_truth(
                    case=case,
                    case_id=case_id,
                    source_family=source_family,
                    feature_representation=feature_representation,
                    n_samples=n_samples,
                    n_features=n_features,
                    n_categories=n_categories,
                    data_role=data_role,
                    seed=data_seed,
                )
                distance = pdist(data.to_numpy(dtype=float), metric="hamming")
                for method_id in validated_methods:
                    try:
                        result = _run_tbs_method(
                            data,
                            distance,
                            sibling_significance_level=float(sibling_alpha),
                            tree_linkage_method="average",
                            edge_alpha=float(edge_alpha),
                            feature_space=feature_space,
                            sibling_gate_profile=_profile_id_for_method(method_id),
                        )
                    except Exception as exc:  # pragma: no cover - diagnostic capture
                        failure_rows.append(
                            {
                                "schema_version": SCHEMA_VERSION,
                                "study_role": STUDY_ROLE,
                                "case_id": case_id,
                                "data_role": _output_data_role(data_role),
                                "method_id": method_id,
                                "replicate": int(replicate),
                                "data_seed": data_seed,
                                "error_type": type(exc).__name__,
                                "error_message": str(exc),
                            }
                        )
                        continue
                    all_rows.append(
                        same_z_rows_for_result(
                            case_id=case_id,
                            data_role=data_role,
                            method_id=method_id,
                            replicate=replicate,
                            data_seed=data_seed,
                            result=result,
                            feature_space=feature_space,
                        )
                    )

    rows = (
        pd.concat(all_rows, ignore_index=True)
        if all_rows
        else pd.DataFrame(columns=ROW_COLUMNS)
    )
    summary = summarize_same_z_rows(rows) if not rows.empty else pd.DataFrame(
        columns=SUMMARY_COLUMNS
    )
    failures = pd.DataFrame.from_records(failure_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "same_z_functional_rows.csv"
    summary_path = output_dir / "same_z_functional_summary.csv"
    failures_path = output_dir / "same_z_functional_failures.csv"
    plot_path = output_dir / "same_z_functional_decomposition.png"
    html_path = output_dir / "same_z_functional_decomposition.html"
    manifest_path = output_dir / "manifest.json"
    rows.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)
    failures.to_csv(failures_path, index=False)
    if not rows.empty:
        plot_same_z_functional_decomposition(rows, output_path=plot_path)
        write_same_z_html_report(
            output_path=html_path,
            plot_path=plot_path,
            summary=summary,
        )
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "suite": suite,
                "case_names": list(case_names),
                "methods": list(validated_methods),
                "data_roles": [
                    NULL_OUTPUT_ROLE if role == "null" else str(role)
                    for role in data_roles
                ],
                "replicates": int(replicates),
                "base_seed": int(base_seed),
                "outputs": {
                    "rows": str(rows_path),
                    "summary": str(summary_path),
                    "failures": str(failures_path),
                    "plot": str(plot_path),
                    "html_report": str(html_path),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return {
        "rows": rows_path,
        "summary": summary_path,
        "failures": failures_path,
        "plot": plot_path,
        "html_report": html_path,
        "manifest": manifest_path,
    }


def _parse_csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--suite", default="binary")
    parser.add_argument("--case-names", default="")
    parser.add_argument(
        "--methods",
        default="fixed_coordinate_guarded_v1",
        help="Comma-separated selected-family method ids.",
    )
    parser.add_argument("--data-roles", default="null,signal")
    parser.add_argument("--sibling-alpha", type=float, default=0.01)
    parser.add_argument("--edge-alpha", type=float, default=0.001)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=20260619)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    outputs = run_same_z_functional_decomposition(
        output_dir=args.output_dir,
        suite=str(args.suite),
        case_names=parse_names(args.case_names),
        methods=_parse_csv_tuple(args.methods),
        data_roles=_parse_csv_tuple(args.data_roles),
        sibling_alpha=float(args.sibling_alpha),
        edge_alpha=float(args.edge_alpha),
        replicates=int(args.replicates),
        base_seed=int(args.base_seed),
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
