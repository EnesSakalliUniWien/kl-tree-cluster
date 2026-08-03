"""Plot edge and sibling p-value relationships from node-decision diagnostics."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarks.diagnostics.calibration.reporting import print_diagnostic_output_paths

STUDY_ROLE = "diagnostic_edge_sibling_pvalue_relationship_not_calibration"
SCHEMA_VERSION = "edge_sibling_pvalue_relationship/v1"

PAIR_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "method_id",
    "replicate",
    "data_seed",
    "parent_node_id",
    "decision_class",
    "traversal_decision",
    "visited",
    "left_child_id",
    "right_child_id",
    "left_edge_p_value",
    "left_edge_bh_p_value",
    "right_edge_p_value",
    "right_edge_bh_p_value",
    "min_edge_bh_p_value",
    "max_edge_bh_p_value",
    "geomean_edge_bh_p_value",
    "left_edge_neglog10_bh",
    "right_edge_neglog10_bh",
    "edge_pair_max_neglog10_bh",
    "edge_pair_mean_neglog10_bh",
    "edge_pair_abs_log10_ratio",
    "sibling_p_value",
    "sibling_p_value_corrected",
    "sibling_neglog10_corrected",
    "sibling_open",
    "sibling_test_method",
    "sibling_gate_p_value_calibration",
    "sibling_sparse_p_value",
    "sibling_sparse_method",
    "sibling_sparse_neglog10",
    "sibling_dense_p_value",
    "sibling_dense_method",
    "sibling_dense_neglog10",
    "sibling_projection_dimension",
    "sibling_tested",
    "sibling_to_edge_max_neglog10_ratio",
    "sparse_sibling_to_edge_max_neglog10_ratio",
    "dense_sibling_to_edge_max_neglog10_ratio",
)

LONG_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "method_id",
    "replicate",
    "data_seed",
    "parent_node_id",
    "decision_class",
    "test_type",
    "p_value",
    "neglog10_p_value",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "method_id",
    "replicate",
    "n_parent_sibling_rows",
    "n_sibling_tested",
    "n_sibling_open",
    "min_edge_bh_p_value",
    "min_sibling_corrected_p_value",
    "median_edge_pair_max_neglog10_bh",
    "median_sibling_neglog10_corrected",
    "median_sparse_sibling_neglog10",
    "median_dense_sibling_neglog10",
    "median_sibling_to_edge_max_neglog10_ratio",
    "median_sparse_sibling_to_edge_max_neglog10_ratio",
    "median_dense_sibling_to_edge_max_neglog10_ratio",
    "spearman_edge_max_vs_sibling_neglog10",
)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _string(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series("", index=frame.index, dtype=object)
    return frame[column].fillna("").astype(str)


def _bool(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    values = frame[column]
    if values.dtype == bool:
        return values.fillna(False).astype(bool)
    return values.fillna(False).astype(str).str.lower().isin({"1", "true", "yes"})


def _neglog10(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    clipped = numeric.clip(lower=np.nextafter(0.0, 1.0), upper=1.0)
    return -np.log10(clipped)


def build_edge_sibling_pair_table(node_decisions: pd.DataFrame) -> pd.DataFrame:
    """Return one row per binary parent linking sibling tests to outgoing edges."""
    n_children = _numeric(node_decisions, "n_children")
    left_child = _string(node_decisions, "outgoing_left_child_id")
    right_child = _string(node_decisions, "outgoing_right_child_id")
    binary = n_children.eq(2) | (left_child.ne("") & right_child.ne(""))
    rows = node_decisions.loc[binary].copy()

    left_edge_p = _numeric(rows, "outgoing_left_edge_p_value")
    right_edge_p = _numeric(rows, "outgoing_right_edge_p_value")
    left_edge_bh = _numeric(rows, "outgoing_left_edge_bh_p_value")
    right_edge_bh = _numeric(rows, "outgoing_right_edge_bh_p_value")
    sibling_p = _numeric(rows, "sibling_p_value")
    sibling_corrected = _numeric(rows, "sibling_p_value_corrected")
    sibling_effective = sibling_corrected.where(sibling_corrected.notna(), sibling_p)
    sibling_sparse = _numeric(rows, "sibling_sparse_p_value")
    sibling_dense = _numeric(rows, "sibling_dense_p_value")

    left_neglog = _neglog10(left_edge_bh)
    right_neglog = _neglog10(right_edge_bh)
    edge_pair_max = pd.concat([left_neglog, right_neglog], axis=1).max(axis=1)
    edge_pair_mean = pd.concat([left_neglog, right_neglog], axis=1).mean(axis=1)
    edge_pair_ratio = (left_neglog - right_neglog).abs()
    sibling_neglog = _neglog10(sibling_effective)
    sibling_sparse_neglog = _neglog10(sibling_sparse)
    sibling_dense_neglog = _neglog10(sibling_dense)

    min_edge_bh = pd.concat([left_edge_bh, right_edge_bh], axis=1).min(axis=1)
    max_edge_bh = pd.concat([left_edge_bh, right_edge_bh], axis=1).max(axis=1)
    geomean_edge_bh = np.sqrt(left_edge_bh * right_edge_bh)
    sibling_to_edge = sibling_neglog / edge_pair_max.replace(0.0, np.nan)
    sparse_sibling_to_edge = sibling_sparse_neglog / edge_pair_max.replace(0.0, np.nan)
    dense_sibling_to_edge = sibling_dense_neglog / edge_pair_max.replace(0.0, np.nan)

    records = pd.DataFrame(
        {
            "schema_version": SCHEMA_VERSION,
            "study_role": STUDY_ROLE,
            "case_id": _string(rows, "case_id"),
            "data_role": _string(rows, "data_role"),
            "method_id": _string(rows, "method_id"),
            "replicate": _numeric(rows, "replicate").fillna(-1).astype(int),
            "data_seed": _numeric(rows, "data_seed").fillna(-1).astype(int),
            "parent_node_id": _string(rows, "node_id"),
            "decision_class": _string(rows, "decision_class"),
            "traversal_decision": _string(rows, "traversal_decision"),
            "visited": _bool(rows, "visited"),
            "left_child_id": left_child.loc[rows.index],
            "right_child_id": right_child.loc[rows.index],
            "left_edge_p_value": left_edge_p,
            "left_edge_bh_p_value": left_edge_bh,
            "right_edge_p_value": right_edge_p,
            "right_edge_bh_p_value": right_edge_bh,
            "min_edge_bh_p_value": min_edge_bh,
            "max_edge_bh_p_value": max_edge_bh,
            "geomean_edge_bh_p_value": geomean_edge_bh,
            "left_edge_neglog10_bh": left_neglog,
            "right_edge_neglog10_bh": right_neglog,
            "edge_pair_max_neglog10_bh": edge_pair_max,
            "edge_pair_mean_neglog10_bh": edge_pair_mean,
            "edge_pair_abs_log10_ratio": edge_pair_ratio,
            "sibling_p_value": sibling_p,
            "sibling_p_value_corrected": sibling_corrected,
            "sibling_neglog10_corrected": sibling_neglog,
            "sibling_open": _bool(rows, "sibling_open"),
            "sibling_test_method": _string(rows, "sibling_test_method"),
            "sibling_gate_p_value_calibration": _string(
                rows,
                "sibling_gate_p_value_calibration",
            ),
            "sibling_sparse_p_value": sibling_sparse,
            "sibling_sparse_method": _string(rows, "sibling_sparse_method"),
            "sibling_sparse_neglog10": sibling_sparse_neglog,
            "sibling_dense_p_value": sibling_dense,
            "sibling_dense_method": _string(rows, "sibling_dense_method"),
            "sibling_dense_neglog10": sibling_dense_neglog,
            "sibling_projection_dimension": _numeric(
                rows,
                "sibling_projection_dimension",
            ),
            "sibling_tested": sibling_effective.notna(),
            "sibling_to_edge_max_neglog10_ratio": sibling_to_edge,
            "sparse_sibling_to_edge_max_neglog10_ratio": sparse_sibling_to_edge,
            "dense_sibling_to_edge_max_neglog10_ratio": dense_sibling_to_edge,
        }
    )
    return records.loc[:, PAIR_COLUMNS]


def build_long_distribution_table(pair_table: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy p-value table for distribution plotting."""
    records: list[dict[str, object]] = []
    shared_columns = (
        "case_id",
        "data_role",
        "method_id",
        "replicate",
        "data_seed",
        "parent_node_id",
        "decision_class",
    )
    test_columns = {
        "left_edge_bh": "left_edge_bh_p_value",
        "right_edge_bh": "right_edge_bh_p_value",
        "sibling_active_corrected": "sibling_p_value_corrected",
        "sibling_sparse": "sibling_sparse_p_value",
        "sibling_dense": "sibling_dense_p_value",
    }
    for _, row in pair_table.iterrows():
        shared = {column: row[column] for column in shared_columns}
        for test_type, p_column in test_columns.items():
            p_value = row[p_column]
            if pd.isna(p_value):
                continue
            records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_role": STUDY_ROLE,
                    **shared,
                    "test_type": test_type,
                    "p_value": float(p_value),
                    "neglog10_p_value": float(
                        -np.log10(np.clip(float(p_value), np.nextafter(0.0, 1.0), 1.0))
                    ),
                }
            )
    return pd.DataFrame.from_records(records, columns=LONG_COLUMNS)


def summarize_edge_sibling_pairs(pair_table: pd.DataFrame) -> pd.DataFrame:
    """Summarize p-value scaling by case, role, method, and replicate."""
    rows: list[dict[str, object]] = []
    group_columns = ["case_id", "data_role", "method_id", "replicate"]
    for group_key, group in pair_table.groupby(group_columns, dropna=False):
        case_id, data_role, method_id, replicate = group_key
        finite = group[
            group["edge_pair_max_neglog10_bh"].notna() & group["sibling_neglog10_corrected"].notna()
        ]
        spearman = (
            float(
                finite["edge_pair_max_neglog10_bh"].corr(
                    finite["sibling_neglog10_corrected"],
                    method="spearman",
                )
            )
            if finite.shape[0] >= 2
            else np.nan
        )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "case_id": case_id,
                "data_role": data_role,
                "method_id": method_id,
                "replicate": int(replicate),
                "n_parent_sibling_rows": int(group.shape[0]),
                "n_sibling_tested": int(group["sibling_tested"].sum()),
                "n_sibling_open": int(group["sibling_open"].sum()),
                "min_edge_bh_p_value": float(group["min_edge_bh_p_value"].min()),
                "min_sibling_corrected_p_value": float(group["sibling_p_value_corrected"].min()),
                "median_edge_pair_max_neglog10_bh": float(
                    group["edge_pair_max_neglog10_bh"].median()
                ),
                "median_sibling_neglog10_corrected": float(
                    group["sibling_neglog10_corrected"].median()
                ),
                "median_sparse_sibling_neglog10": float(group["sibling_sparse_neglog10"].median()),
                "median_dense_sibling_neglog10": float(group["sibling_dense_neglog10"].median()),
                "median_sibling_to_edge_max_neglog10_ratio": float(
                    group["sibling_to_edge_max_neglog10_ratio"].median()
                ),
                "median_sparse_sibling_to_edge_max_neglog10_ratio": float(
                    group["sparse_sibling_to_edge_max_neglog10_ratio"].median()
                ),
                "median_dense_sibling_to_edge_max_neglog10_ratio": float(
                    group["dense_sibling_to_edge_max_neglog10_ratio"].median()
                ),
                "spearman_edge_max_vs_sibling_neglog10": spearman,
            }
        )
    return pd.DataFrame.from_records(rows, columns=SUMMARY_COLUMNS)


def plot_edge_sibling_relationship(
    pair_table: pd.DataFrame,
    long_table: pd.DataFrame,
    *,
    output_path: Path,
    title: str,
) -> None:
    """Write a static plot of sibling p-values against related edge p-values."""
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

    scatter = pair_table[
        pair_table["edge_pair_max_neglog10_bh"].notna()
        & pair_table["sibling_neglog10_corrected"].notna()
    ].copy()
    colors = {"selected_null": "#4c78a8", "signal": "#f58518"}
    markers = ["o", "s", "^", "D", "P", "X"]
    methods = sorted(scatter["method_id"].dropna().unique())
    marker_by_method = {
        method: markers[index % len(markers)] for index, method in enumerate(methods)
    }
    for (data_role, method_id), group in scatter.groupby(["data_role", "method_id"]):
        sizes = np.where(group["sibling_open"].astype(bool), 110, 42)
        axes[0, 0].scatter(
            group["edge_pair_max_neglog10_bh"],
            group["sibling_neglog10_corrected"],
            s=sizes,
            alpha=0.72,
            color=colors.get(str(data_role), "#54a24b"),
            marker=marker_by_method.get(method_id, "o"),
            label=f"{data_role} / {method_id}",
        )
    axes[0, 0].set_title("Sibling test vs strongest outgoing edge")
    axes[0, 0].set_xlabel("max(-log10 left/right edge BH p)")
    axes[0, 0].set_ylabel("-log10 corrected sibling p")
    axes[0, 0].legend(fontsize=8, loc="best")

    test_types = [
        "left_edge_bh",
        "right_edge_bh",
        "sibling_active_corrected",
        "sibling_sparse",
        "sibling_dense",
    ]
    roles = sorted(long_table["data_role"].dropna().unique())
    positions: list[float] = []
    labels: list[str] = []
    arrays: list[np.ndarray] = []
    box_colors: list[str] = []
    for test_index, test_type in enumerate(test_types):
        for role_index, role in enumerate(roles):
            values = long_table.loc[
                long_table["test_type"].eq(test_type) & long_table["data_role"].eq(role),
                "neglog10_p_value",
            ].dropna()
            if values.empty:
                continue
            position = test_index * (len(roles) + 1) + role_index
            positions.append(float(position))
            labels.append(f"{test_type}\n{role}")
            arrays.append(values.to_numpy(dtype=float))
            box_colors.append(colors.get(str(role), "#54a24b"))
    boxes = axes[0, 1].boxplot(arrays, positions=positions, showfliers=False, patch_artist=True)
    for patch, color in zip(boxes["boxes"], box_colors, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
    for position, values, color in zip(positions, arrays, box_colors, strict=False):
        jitter = np.linspace(-0.16, 0.16, num=min(len(values), 200))
        sample = values[: len(jitter)]
        axes[0, 1].scatter(
            np.full_like(sample, position, dtype=float) + jitter,
            sample,
            color=color,
            alpha=0.2,
            s=8,
        )
    axes[0, 1].set_xticks(positions, labels, rotation=25, ha="right")
    axes[0, 1].set_title("Distribution by test type")
    axes[0, 1].set_ylabel("-log10 p")

    for test_type, group in long_table.groupby("test_type"):
        values = np.sort(group["neglog10_p_value"].dropna().to_numpy(dtype=float))
        if values.size == 0:
            continue
        y = np.arange(1, values.size + 1, dtype=float) / float(values.size)
        axes[1, 0].plot(values, y, label=str(test_type), linewidth=2)
    axes[1, 0].set_title("Empirical distribution of p-value strength")
    axes[1, 0].set_xlabel("-log10 p")
    axes[1, 0].set_ylabel("fraction <= x")
    axes[1, 0].legend(fontsize=9)

    ratio_rows = pair_table[
        pair_table["sibling_to_edge_max_neglog10_ratio"]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .notna()
    ].copy()
    ratio_groups = list(ratio_rows.groupby(["case_id", "data_role"]))
    ratio_arrays = [
        group["sibling_to_edge_max_neglog10_ratio"].dropna().to_numpy(dtype=float)
        for _, group in ratio_groups
    ]
    ratio_positions = np.arange(len(ratio_arrays), dtype=float)
    ratio_labels = [f"{case}\n{role}" for (case, role), _ in ratio_groups]
    if ratio_arrays:
        ratio_boxes = axes[1, 1].boxplot(
            ratio_arrays,
            positions=ratio_positions,
            showfliers=False,
            patch_artist=True,
        )
        for patch, ((_, role), _) in zip(ratio_boxes["boxes"], ratio_groups, strict=False):
            patch.set_facecolor(colors.get(str(role), "#54a24b"))
            patch.set_alpha(0.45)
        for position, values, ((_, role), _) in zip(
            ratio_positions,
            ratio_arrays,
            ratio_groups,
            strict=False,
        ):
            sample = values[: min(len(values), 200)]
            jitter = np.linspace(-0.16, 0.16, num=len(sample))
            axes[1, 1].scatter(
                np.full_like(sample, position, dtype=float) + jitter,
                sample,
                color=colors.get(str(role), "#54a24b"),
                alpha=0.2,
                s=8,
            )
        axes[1, 1].set_xticks(ratio_positions, ratio_labels, rotation=20, ha="right")
    axes[1, 1].set_title("Active sibling p-value scale relative to edge p-value scale")
    axes[1, 1].set_ylabel("sibling -log10 p / max edge -log10 p")

    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _html_table(frame: pd.DataFrame, *, max_rows: int = 50) -> str:
    display = frame.head(max_rows).copy()
    return display.to_html(index=False, escape=True, classes="data-table")


def write_html_report(
    *,
    output_path: Path,
    plot_path: Path,
    pair_path: Path,
    long_path: Path,
    summary: pd.DataFrame,
    title: str,
) -> None:
    """Write a lightweight report with plot alt text and tables."""
    relative_plot = html.escape(plot_path.name)
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
    This diagnostic joins each parent-keyed sibling test to the left and right
    outgoing child-parent edge tests used by that sibling decision. The plot
    uses <code>-log10(p)</code> so stronger evidence appears higher or farther
    right.
  </p>
  <img src="{relative_plot}" alt="Four-panel p-value diagnostic: sibling tests
  versus strongest outgoing edge, p-value distributions by test type, empirical
  cumulative distributions, and sibling-to-edge scale ratios.">
  <h2>Summary</h2>
  {_html_table(summary)}
  <h2>Artifacts</h2>
  <ul>
    <li>Parent-level pairs: <code>{html.escape(pair_path.name)}</code></li>
    <li>Tidy distribution table: <code>{html.escape(long_path.name)}</code></li>
    <li>Plot image: <code>{html.escape(plot_path.name)}</code></li>
  </ul>
</body>
</html>
"""
    output_path.write_text(body, encoding="utf-8")


def run_edge_sibling_pvalue_relationship(
    *,
    node_decisions_path: Path,
    output_dir: Path,
    title: str = "Edge and Sibling P-Value Relationship",
) -> dict[str, Path]:
    """Build relationship tables and plots from a node-decision CSV."""
    node_decisions = pd.read_csv(node_decisions_path)
    pair_table = build_edge_sibling_pair_table(node_decisions)
    long_table = build_long_distribution_table(pair_table)
    summary = summarize_edge_sibling_pairs(pair_table)

    output_dir.mkdir(parents=True, exist_ok=True)
    pair_path = output_dir / "edge_sibling_pvalue_pairs.csv"
    long_path = output_dir / "edge_sibling_pvalue_long.csv"
    summary_path = output_dir / "edge_sibling_pvalue_summary.csv"
    plot_path = output_dir / "edge_sibling_pvalue_relationship.png"
    html_path = output_dir / "edge_sibling_pvalue_relationship.html"
    manifest_path = output_dir / "manifest.json"

    pair_table.to_csv(pair_path, index=False)
    long_table.to_csv(long_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_edge_sibling_relationship(pair_table, long_table, output_path=plot_path, title=title)
    write_html_report(
        output_path=html_path,
        plot_path=plot_path,
        pair_path=pair_path,
        long_path=long_path,
        summary=summary,
        title=title,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "node_decisions_path": str(node_decisions_path),
        "outputs": {
            "pair_table": str(pair_path),
            "long_table": str(long_path),
            "summary": str(summary_path),
            "plot": str(plot_path),
            "html_report": str(html_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "pair_table": pair_path,
        "long_table": long_path,
        "summary": summary_path,
        "plot": plot_path,
        "html_report": html_path,
        "manifest": manifest_path,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--node-decisions",
        required=True,
        type=Path,
        help="Path to multiscale_node_decisions.csv.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for relationship tables and plots.",
    )
    parser.add_argument(
        "--title",
        default="Edge and Sibling P-Value Relationship",
        help="Report and figure title.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    outputs = run_edge_sibling_pvalue_relationship(
        node_decisions_path=args.node_decisions,
        output_dir=args.output_dir,
        title=args.title,
    )
    print_diagnostic_output_paths(outputs)


if __name__ == "__main__":
    main()
