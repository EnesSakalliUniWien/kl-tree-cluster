"""Interactive Plotly MNIST TBS analysis pages from the saved PCA50 sweep."""

from __future__ import annotations

import hashlib
import html
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import umap
from benchmarks.experiments.mnist.run import load_mnist_subset
from benchmarks.experiments.mnist.run_higher_categories import (
    create_plotly_higher_category_plot_3d,
)

ROOT = Path("/Users/berksakalli/Projects/kl-te-cluster")
SOURCE_DIR = ROOT / "benchmarks/results/experiments/mnist"
SWEEP_DIR = SOURCE_DIR / "alpha_sweep_continuous_pca50_20260605"
OUT_DIR = ROOT / "raw/assets/benchmark-results/mnist_tbs_analysis_20260624_plotly"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_INDEX = OUT_DIR / "index.html"
OUT_CSV = OUT_DIR / "mnist_tbs_analysis_summary.csv"
OUT_UMAP_COORDS = OUT_DIR / "mnist_pca50_umap_coordinates.csv"
OUT_MANIFEST = OUT_DIR / "manifest.json"
OUT_UMAP3D_HTML = OUT_DIR / "00b_mnist_umap3d_true_digit_best_tbs_cluster.html"
OUT_UMAP3D_DIGIT_LABELS_HTML = OUT_DIR / "00c_mnist_umap3d_visible_digit_labels.html"
OUT_UMAP3D_DOCSTYLE_HTML = OUT_DIR / "00d_mnist_umap3d_docstyle_digit_colorbar.html"

DIGIT_COLORS = {
    0: "#4e79a7",
    1: "#f28e2b",
    2: "#e15759",
    3: "#76b7b2",
    4: "#59a14f",
    5: "#edc948",
    6: "#b07aa1",
    7: "#ff9da7",
    8: "#9c755f",
    9: "#bab0ab",
}


def _alpha_label(value: float) -> str:
    return f"{value:g}"


def _assignment_key(linkage: str, edge_alpha: float, sibling_alpha: float) -> str:
    return f"{linkage}_e{_alpha_label(edge_alpha)}_s{_alpha_label(sibling_alpha)}"


def _best_rows(summary: pd.DataFrame) -> pd.DataFrame:
    idx = summary.groupby("linkage")["ARI"].idxmax()
    return summary.loc[idx].sort_values("ARI", ascending=False).reset_index(drop=True)


def _parse_digit_counts(value: str) -> dict[int, int]:
    counts: dict[int, int] = {}
    for digit, count in re.findall(r"(\d+):(\d+)", str(value)):
        counts[int(digit)] = int(count)
    return counts


def _base_layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left"},
        template="plotly_white",
        font={"family": "Arial, sans-serif", "size": 13},
        margin={"l": 75, "r": 40, "t": 86, "b": 86},
        width=1050,
        height=720,
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.22, "xanchor": "center", "x": 0.5},
    )
    return fig


def _write(fig: go.Figure, filename: str, title: str) -> Path:
    path = OUT_DIR / filename
    _base_layout(fig, title).write_html(path, include_plotlyjs=True, full_html=True)
    return path


def best_summary_figure(summary: pd.DataFrame) -> go.Figure:
    best = _best_rows(summary)
    labels = [
        f"{row.linkage}<br>edge={row.edge_alpha:g}, sibling={row.sibling_alpha:g}<br>{int(row.n_clusters)} clusters"
        for row in best.itertuples()
    ]
    fig = go.Figure()
    fig.add_bar(
        x=labels,
        y=best["ARI"],
        name="ARI",
        marker_color="#2563eb",
        text=[f"{v:.3f}" for v in best["ARI"]],
        textposition="outside",
        customdata=best[["linkage", "edge_alpha", "sibling_alpha", "n_clusters", "NMI"]].to_numpy(),
        hovertemplate=(
            "linkage=%{customdata[0]}<br>edge=%{customdata[1]:g}<br>"
            "sibling=%{customdata[2]:g}<br>clusters=%{customdata[3]}<br>"
            "ARI=%{y:.4f}<br>NMI=%{customdata[4]:.4f}<extra></extra>"
        ),
    )
    fig.add_bar(
        x=labels,
        y=best["NMI"],
        name="NMI",
        marker_color="#059669",
        text=[f"{v:.3f}" for v in best["NMI"]],
        textposition="outside",
        customdata=best[["linkage", "edge_alpha", "sibling_alpha", "n_clusters", "ARI"]].to_numpy(),
        hovertemplate=(
            "linkage=%{customdata[0]}<br>edge=%{customdata[1]:g}<br>"
            "sibling=%{customdata[2]:g}<br>clusters=%{customdata[3]}<br>"
            "NMI=%{y:.4f}<br>ARI=%{customdata[4]:.4f}<extra></extra>"
        ),
    )
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="score", range=[0, 0.76], gridcolor="#e5e7eb")
    fig.update_xaxes(title_text="best setting per linkage")
    return fig


def heatmap_figure(summary: pd.DataFrame, linkage: str, value: str, title: str) -> go.Figure:
    frame = summary[summary["linkage"] == linkage].copy()
    edges = sorted(frame["edge_alpha"].unique())
    siblings = sorted(frame["sibling_alpha"].unique())
    matrix: list[list[float]] = []
    text: list[list[str]] = []
    for sibling in siblings:
        row_values = []
        row_text = []
        for edge in edges:
            selected = frame[(frame["edge_alpha"] == edge) & (frame["sibling_alpha"] == sibling)]
            metric = float(selected.iloc[0][value])
            row_values.append(metric)
            row_text.append(f"{metric:.3f}" if value != "n_clusters" else str(int(metric)))
        matrix.append(row_values)
        text.append(row_text)
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=[_alpha_label(v) for v in edges],
            y=[_alpha_label(v) for v in siblings],
            text=text,
            texttemplate="%{text}",
            colorscale="Viridis" if value != "n_clusters" else "Magma",
            colorbar={"title": value},
            hovertemplate="edge alpha=%{x}<br>sibling alpha=%{y}<br>" + value + "=%{z}<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="edge alpha")
    fig.update_yaxes(title_text="sibling alpha")
    fig.update_layout(title={"text": title, "x": 0.02, "xanchor": "left"})
    return fig


def composition_figure(composition: pd.DataFrame) -> go.Figure:
    frame = composition.sort_values("size", ascending=True).copy()
    y_labels = [
        f"C{int(row.cluster)} | n={int(row.size)} | top={int(row.dominant_digit)} ({row.purity:.0%})"
        for row in frame.itertuples()
    ]
    fig = go.Figure()
    parsed = frame["digit_counts"].map(_parse_digit_counts)
    for digit in range(10):
        values = [counts.get(digit, 0) for counts in parsed]
        fig.add_bar(
            x=values,
            y=y_labels,
            orientation="h",
            name=str(digit),
            marker_color=DIGIT_COLORS[digit],
            hovertemplate=f"digit={digit}<br>samples=%{{x}}<br>%{{y}}<extra></extra>",
        )
    fig.update_layout(barmode="stack", height=760)
    fig.update_xaxes(title_text="samples", gridcolor="#e5e7eb")
    fig.update_yaxes(title_text="best-run cluster")
    return fig


def baseline_figure(binary: pd.DataFrame, continuous: pd.DataFrame, summary: pd.DataFrame) -> go.Figure:
    best_sweep = summary.sort_values("ARI", ascending=False).iloc[0]
    best_continuous = continuous.loc[continuous["ARI"].idxmax()]
    best_binary = binary.loc[binary["ARI"].idxmax()]
    rows = pd.DataFrame(
        [
            {
                "run": "best alpha sweep<br>continuous PCA50 Ward",
                "ARI": best_sweep["ARI"],
                "NMI": best_sweep["NMI"],
                "clusters": int(best_sweep["n_clusters"]),
            },
            {
                "run": "best fixed continuous<br>PCA50 complete",
                "ARI": best_continuous["ARI"],
                "NMI": best_continuous["NMI"],
                "clusters": int(best_continuous["n_clusters"]),
            },
            {
                "run": "old binary run<br>jaccard/dice single",
                "ARI": best_binary["ARI"],
                "NMI": best_binary["NMI"],
                "clusters": int(best_binary["n_clusters"]),
            },
        ]
    )
    labels = [f"{row.run}<br>{row.clusters} clusters" for row in rows.itertuples()]
    fig = go.Figure()
    fig.add_bar(x=labels, y=rows["ARI"], name="ARI", marker_color="#2563eb", text=[f"{v:.3f}" for v in rows["ARI"]], textposition="outside")
    fig.add_bar(x=labels, y=rows["NMI"], name="NMI", marker_color="#059669", text=[f"{v:.3f}" for v in rows["NMI"]], textposition="outside")
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="score", range=[0, 0.76], gridcolor="#e5e7eb")
    fig.update_xaxes(title_text="MNIST result source")
    return fig


def _write_index(pages: list[tuple[str, str]], generated_at: str) -> None:
    links = "\n".join(
        f'<li><a href="{html.escape(path)}">{html.escape(title)}</a></li>' for title, path in pages
    )
    OUT_INDEX.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>MNIST TBS Plotly Analysis</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #111827; }}
    li {{ margin: 10px 0; }}
    .note {{ color: #374151; max-width: 920px; line-height: 1.45; }}
  </style>
</head>
<body>
  <h1>MNIST TBS Plotly Analysis</h1>
  <p class="note"><strong>Generated at:</strong> {html.escape(generated_at)}</p>
  <p class="note">Saved run: 2,000 normalized MNIST images, PCA50 representation
  with 83.13% variance retained, continuous TBS alpha sweep over complete,
  weighted, and Ward linkage. The UMAP pages use the same sample/PCA50 setup
  and join visualization coordinates to the saved best-run TBS labels;
  clustering labels and metrics are not rerun here. The 3D page is generated
  with the existing MNIST Plotly 3D writer. The visible-label 3D page draws
  each point with its true MNIST digit as text. The docs-style 3D page follows
  the UMAP basic-usage convention: one embedding, colored by true digit with a
  digit colorbar. UMAP hover labels start with the true digit number. Each linked
  page contains one Plotly figure.</p>
  <ol>
    {links}
  </ol>
</body>
</html>
""",
        encoding="utf-8",
    )


def _artifact_record(path: Path) -> dict[str, object]:
    record: dict[str, object] = {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    if path.suffix == ".csv":
        frame = pd.read_csv(path)
        record["rows"] = int(len(frame))
        record["columns"] = int(len(frame.columns))
        if "generated_at" in frame.columns:
            record["generated_at_values"] = sorted(frame["generated_at"].astype(str).unique())
    return record


def _write_manifest(
    pages: list[tuple[str, str]],
    *,
    generated_at: str,
    best_key: str,
) -> None:
    artifact_paths = [OUT_INDEX, OUT_CSV, OUT_UMAP_COORDS] + [OUT_DIR / page for _, page in pages]
    manifest = {
        "manifest_schema_version": "static_artifact_provenance/v1",
        "bundle": "mnist_tbs_analysis_20260624_plotly",
        "generated_at": generated_at,
        "source_script": "scripts/plot_mnist_tbs_analysis_plotly.py",
        "source_inputs": [
            str((SWEEP_DIR / "alpha_sweep_summary.csv").relative_to(ROOT)),
            str((SWEEP_DIR / "alpha_sweep_assignments.csv").relative_to(ROOT)),
            str((SWEEP_DIR / f"{best_key}_top_cluster_digit_composition.csv").relative_to(ROOT)),
            str((SOURCE_DIR / "mnist_benchmark_summary.csv").relative_to(ROOT)),
            str((SOURCE_DIR / "mnist_continuous_pca50_summary.csv").relative_to(ROOT)),
        ],
        "artifacts": [_artifact_record(path) for path in artifact_paths],
        "notes": [
            "The HTML pages were generated by the existing MNIST Plotly report script.",
            "The UMAP coordinate CSV may be reused from cache when its sample, digit, and TBS assignment columns match the saved alpha-sweep assignments.",
        ],
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _load_or_create_umap(assignments: pd.DataFrame, best_key: str) -> pd.DataFrame:
    if best_key not in assignments.columns:
        raise KeyError(f"Missing saved assignment column: {best_key}")

    if OUT_UMAP_COORDS.exists():
        cached = pd.read_csv(OUT_UMAP_COORDS)
        required_columns = {"sample", "true_digit", "umap1", "umap2", "best_tbs_cluster"}
        if required_columns.issubset(cached.columns) and len(cached) == len(assignments):
            cache_matches_assignments = (
                cached["sample"].tolist() == assignments["sample"].tolist()
                and cached["true_digit"].astype(int).tolist()
                == assignments["true_digit"].astype(int).tolist()
                and cached["best_tbs_cluster"].astype(int).tolist()
                == assignments[best_key].astype(int).tolist()
            )
            if cache_matches_assignments:
                return cached

    pca50, y_subset = load_mnist_subset(
        n_samples=2000,
        seed=42,
        use_pca=True,
        n_components=50,
    )
    reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(pca50)
    frame = pd.DataFrame(
        {
            "sample": [f"Sample_{index}" for index in range(len(pca50))],
            "true_digit": y_subset.astype(int),
            "umap1": embedding[:, 0],
            "umap2": embedding[:, 1],
        }
    )
    frame = frame.merge(assignments[["sample", best_key]], on="sample", how="left")
    frame = frame.rename(columns={best_key: "best_tbs_cluster"})
    if frame["best_tbs_cluster"].isna().any():
        raise ValueError("UMAP samples did not all join to saved TBS assignments")
    frame.to_csv(OUT_UMAP_COORDS, index=False)
    return frame


def _write_umap3d_with_existing_writer(
    assignments: pd.DataFrame,
    best_key: str,
    *,
    output_path: Path,
    visible_digit_labels: bool = False,
) -> Path:
    if best_key not in assignments.columns:
        raise KeyError(f"Missing saved assignment column: {best_key}")

    pca50, y_subset = load_mnist_subset(
        n_samples=2000,
        seed=42,
        use_pca=True,
        n_components=50,
    )
    cluster_labels = assignments[best_key].to_numpy()
    if len(cluster_labels) != len(y_subset):
        raise ValueError("Saved TBS assignments do not match the MNIST sample size")

    create_plotly_higher_category_plot_3d(
        feature_matrix=pca50,
        digit_labels=y_subset,
        cluster_labels=cluster_labels,
        true_higher_category_names=np.array([f"digit {int(digit)}" for digit in y_subset]),
        predicted_higher_category_names=np.array(
            [f"TBS C{int(cluster)}" for cluster in cluster_labels]
        ),
        output_path=output_path,
        visible_digit_labels=visible_digit_labels,
    )
    replacements = {
        "MNIST Higher Categories - Interactive 3D UMAP": "MNIST PCA50 saved TBS labels - Interactive 3D UMAP",
        "3D UMAP - True Higher Categories": "3D UMAP - True digits",
        "3D UMAP - Predicted Higher Categories": "3D UMAP - Best TBS clusters",
        "Raw Tree Clusters": "Saved TBS clusters",
        "True Higher Category": "True digit",
        "Pred Higher Category": "Best TBS cluster",
        "Tree Cluster": "TBS cluster",
    }
    html_text = output_path.read_text(encoding="utf-8")
    for old, new in replacements.items():
        html_text = html_text.replace(old, new)
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def _write_umap3d_docstyle(assignments: pd.DataFrame, best_key: str) -> Path:
    if best_key not in assignments.columns:
        raise KeyError(f"Missing saved assignment column: {best_key}")

    pca50, y_subset = load_mnist_subset(
        n_samples=2000,
        seed=42,
        use_pca=True,
        n_components=50,
    )
    cluster_labels = assignments[best_key].to_numpy()
    if len(cluster_labels) != len(y_subset):
        raise ValueError("Saved TBS assignments do not match the MNIST sample size")

    reducer = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(pca50)
    customdata = np.column_stack(
        [
            [f"Sample_{index}" for index in range(len(y_subset))],
            y_subset.astype(int),
            cluster_labels.astype(int),
        ]
    )
    hovertext = [
        f"Digit number: {int(digit)}<br>Sample: Sample_{index}<br>"
        f"Best TBS cluster: {int(cluster)}"
        for index, (digit, cluster) in enumerate(zip(y_subset, cluster_labels, strict=True))
    ]
    hovertemplate = "%{hovertext}<br>UMAP1=%{x:.3f}<br>UMAP2=%{y:.3f}<br>UMAP3=%{z:.3f}<extra></extra>"
    fig = go.Figure(
        go.Scatter3d(
            x=embedding[:, 0],
            y=embedding[:, 1],
            z=embedding[:, 2],
            mode="markers+text",
            text=[str(int(digit)) for digit in y_subset],
            textposition="middle center",
            textfont={"size": 8, "color": "#111827"},
            customdata=customdata,
            marker={
                "size": 7,
                "opacity": 0.42,
                "color": y_subset.astype(int),
                "colorscale": "Spectral",
                "cmin": -0.5,
                "cmax": 9.5,
                "colorbar": {
                    "title": "true digit",
                    "tickmode": "array",
                    "tickvals": list(range(10)),
                    "ticktext": [str(digit) for digit in range(10)],
                },
            },
            hovertext=hovertext,
            hovertemplate=hovertemplate,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=embedding[:, 0],
            y=embedding[:, 1],
            z=embedding[:, 2],
            mode="markers",
            name="hover target",
            showlegend=False,
            customdata=customdata,
            hovertext=hovertext,
            marker={"size": 24, "opacity": 0.01, "color": "#111827"},
            hovertemplate=hovertemplate,
        )
    )
    fig.update_layout(
        title={"text": "MNIST PCA50 3D UMAP: digits as labels", "x": 0.02, "xanchor": "left"},
        template="plotly_white",
        font={"family": "Arial, sans-serif", "size": 13},
        width=1050,
        height=760,
        margin={"l": 10, "r": 10, "t": 70, "b": 10},
        scene={
            "xaxis_title": "UMAP-1",
            "yaxis_title": "UMAP-2",
            "zaxis_title": "UMAP-3",
        },
    )
    fig.write_html(OUT_UMAP3D_DOCSTYLE_HTML, include_plotlyjs=True, full_html=True)
    return OUT_UMAP3D_DOCSTYLE_HTML


def umap_figure(frame: pd.DataFrame) -> go.Figure:
    true_digits = sorted(frame["true_digit"].unique())
    clusters = sorted(frame["best_tbs_cluster"].unique())
    fig = go.Figure()
    for digit in true_digits:
        group = frame[frame["true_digit"] == digit]
        hovertext = [
            f"Digit number: {int(row.true_digit)}<br>Sample: {row.sample}<br>"
            f"Best TBS cluster: {int(row.best_tbs_cluster)}"
            for row in group.itertuples(index=False)
        ]
        fig.add_scattergl(
            x=group["umap1"],
            y=group["umap2"],
            mode="markers",
            name=f"digit {digit}",
            marker={"size": 5, "opacity": 0.78},
            customdata=group[["sample", "true_digit", "best_tbs_cluster"]].to_numpy(),
            hovertext=hovertext,
            hovertemplate=(
                "%{hovertext}<br>UMAP1=%{x:.3f}<br>UMAP2=%{y:.3f}<extra></extra>"
            ),
            visible=True,
            legendgroup="true",
        )
    for cluster in clusters:
        group = frame[frame["best_tbs_cluster"] == cluster]
        hovertext = [
            f"Digit number: {int(row.true_digit)}<br>Sample: {row.sample}<br>"
            f"Best TBS cluster: {int(row.best_tbs_cluster)}"
            for row in group.itertuples(index=False)
        ]
        fig.add_scattergl(
            x=group["umap1"],
            y=group["umap2"],
            mode="markers",
            name=f"C{int(cluster)}",
            marker={"size": 5, "opacity": 0.78},
            customdata=group[["sample", "true_digit", "best_tbs_cluster"]].to_numpy(),
            hovertext=hovertext,
            hovertemplate=(
                "%{hovertext}<br>UMAP1=%{x:.3f}<br>UMAP2=%{y:.3f}<extra></extra>"
            ),
            visible=False,
            legendgroup="cluster",
        )

    n_true = len(true_digits)
    n_cluster = len(clusters)
    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0.02,
                "y": 1.12,
                "buttons": [
                    {
                        "label": "Color by true digit",
                        "method": "update",
                        "args": [
                            {"visible": [True] * n_true + [False] * n_cluster},
                            {"legend": {"title": {"text": "true digit"}}},
                        ],
                    },
                    {
                        "label": "Color by best TBS cluster",
                        "method": "update",
                        "args": [
                            {"visible": [False] * n_true + [True] * n_cluster},
                            {"legend": {"title": {"text": "TBS cluster"}}},
                        ],
                    },
                ],
            }
        ],
        legend={"title": {"text": "true digit"}},
    )
    fig.update_xaxes(title_text="UMAP1")
    fig.update_yaxes(title_text="UMAP2", scaleanchor="x", scaleratio=1)
    return fig


def main() -> None:
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    summary = pd.read_csv(SWEEP_DIR / "alpha_sweep_summary.csv")
    binary = pd.read_csv(SOURCE_DIR / "mnist_benchmark_summary.csv")
    continuous = pd.read_csv(SOURCE_DIR / "mnist_continuous_pca50_summary.csv")
    assignments = pd.read_csv(SWEEP_DIR / "alpha_sweep_assignments.csv")
    best = summary.sort_values("ARI", ascending=False).iloc[0]
    best_key = _assignment_key(str(best["linkage"]), float(best["edge_alpha"]), float(best["sibling_alpha"]))
    composition = pd.read_csv(SWEEP_DIR / f"{best_key}_top_cluster_digit_composition.csv")
    umap_frame = _load_or_create_umap(assignments, best_key)
    umap3d_path = _write_umap3d_with_existing_writer(
        assignments,
        best_key,
        output_path=OUT_UMAP3D_HTML,
    )
    umap3d_digit_labels_path = _write_umap3d_with_existing_writer(
        assignments,
        best_key,
        output_path=OUT_UMAP3D_DIGIT_LABELS_HTML,
        visible_digit_labels=True,
    )
    umap3d_docstyle_path = _write_umap3d_docstyle(assignments, best_key)

    analysis_summary = _best_rows(summary)
    analysis_summary.insert(0, "generated_at", generated_at)
    analysis_summary.to_csv(OUT_CSV, index=False)
    pages = [
        (
            "MNIST UMAP: true digit and best TBS cluster",
            _write(
                umap_figure(umap_frame),
                "00_mnist_umap_true_digit_best_tbs_cluster.html",
                "MNIST PCA50 UMAP: true digit and best TBS cluster",
            ).name,
        ),
        (
            "MNIST 3D UMAP: true digit and best TBS cluster",
            umap3d_path.name,
        ),
        (
            "MNIST 3D UMAP: visible digit labels",
            umap3d_digit_labels_path.name,
        ),
        (
            "MNIST 3D UMAP: docs-style digit colorbar",
            umap3d_docstyle_path.name,
        ),
        (
            "Best alpha setting per linkage",
            _write(
                best_summary_figure(summary),
                "01_best_alpha_setting_per_linkage.html",
                "MNIST continuous PCA50 TBS: best alpha setting per linkage",
            ).name,
        ),
        (
            "Ward ARI reaction to alpha",
            _write(
                heatmap_figure(summary, "ward", "ARI", "Ward linkage: ARI reaction to alpha"),
                "02_ward_ari_alpha_heatmap.html",
                "Ward linkage: ARI reaction to alpha",
            ).name,
        ),
        (
            "Ward cluster-count reaction to alpha",
            _write(
                heatmap_figure(summary, "ward", "n_clusters", "Ward linkage: cluster-count reaction to alpha"),
                "03_ward_cluster_count_alpha_heatmap.html",
                "Ward linkage: cluster-count reaction to alpha",
            ).name,
        ),
        (
            "Best-run cluster composition by true digit",
            _write(
                composition_figure(composition),
                "04_best_ward_cluster_digit_composition.html",
                "Best MNIST TBS run: cluster composition by true digit",
            ).name,
        ),
        (
            "Result source comparison",
            _write(
                baseline_figure(binary, continuous, summary),
                "05_result_source_comparison.html",
                "Which MNIST result should be trusted?",
            ).name,
        ),
    ]
    _write_index(pages, generated_at)
    _write_manifest(pages, generated_at=generated_at, best_key=best_key)
    print(OUT_INDEX)
    print(OUT_CSV)
    print(OUT_UMAP_COORDS)
    print(OUT_MANIFEST)
    print(OUT_UMAP3D_HTML)
    print(OUT_UMAP3D_DIGIT_LABELS_HTML)
    print(OUT_UMAP3D_DOCSTYLE_HTML)
    for _title, page in pages:
        print(OUT_DIR / page)


if __name__ == "__main__":
    main()
