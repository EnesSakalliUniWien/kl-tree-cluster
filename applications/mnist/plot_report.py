"""Clean MNIST TBS analysis plots from the saved continuous PCA50 alpha sweep."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from applications.mnist._shared import best_rows, parse_digit_counts

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "benchmarks/results/experiments/mnist"
SWEEP_DIR = SOURCE_DIR / "alpha_sweep_continuous_pca50_20260605"
OUT_DIR = ROOT / "raw/assets/benchmark-results/mnist_tbs_analysis_20260624"

OUT_PDF = OUT_DIR / "mnist_tbs_analysis_one_plot_per_page.pdf"
OUT_MAIN = OUT_DIR / "mnist_tbs_analysis_summary.png"
OUT_CSV = OUT_DIR / "mnist_tbs_analysis_summary.csv"

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


def _add_generated_at(fig: plt.Figure, generated_at: str) -> None:
    fig.text(
        0.99,
        0.01,
        f"Generated at: {generated_at}",
        ha="right",
        va="bottom",
        fontsize=7,
        color="#6b7280",
    )


def _plot_best_summary(ax, summary: pd.DataFrame) -> None:
    best = best_rows(summary)
    labels = [
        f"{row.linkage}\ne={row.edge_alpha:g}, s={row.sibling_alpha:g}\n{int(row.n_clusters)} clusters"
        for row in best.itertuples()
    ]
    x = np.arange(len(best))
    width = 0.36
    ax.bar(x - width / 2, best["ARI"], width, label="ARI", color="#2563eb")
    ax.bar(x + width / 2, best["NMI"], width, label="NMI", color="#059669")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.0, max(0.75, float(best[["ARI", "NMI"]].max().max()) + 0.08))
    ax.set_ylabel("score")
    ax.set_title("MNIST continuous PCA50 TBS: best alpha setting per linkage", weight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#e5e7eb", lw=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for index, row in enumerate(best.itertuples()):
        ax.text(index - width / 2, row.ARI + 0.015, f"{row.ARI:.3f}", ha="center", fontsize=8)
        ax.text(index + width / 2, row.NMI + 0.015, f"{row.NMI:.3f}", ha="center", fontsize=8)


def _heatmap_matrix(summary: pd.DataFrame, linkage: str, value: str) -> tuple[np.ndarray, list[float], list[float]]:
    frame = summary[summary["linkage"] == linkage].copy()
    edges = sorted(frame["edge_alpha"].unique())
    siblings = sorted(frame["sibling_alpha"].unique())
    matrix = np.full((len(siblings), len(edges)), np.nan, dtype=float)
    for row in frame.itertuples():
        i = siblings.index(row.sibling_alpha)
        j = edges.index(row.edge_alpha)
        matrix[i, j] = float(getattr(row, value))
    return matrix, edges, siblings


def _plot_heatmap(ax, summary: pd.DataFrame, linkage: str, value: str, title: str, cmap: str) -> None:
    matrix, edges, siblings = _heatmap_matrix(summary, linkage, value)
    image = ax.imshow(matrix, cmap=cmap, origin="lower", aspect="auto")
    ax.set_xticks(np.arange(len(edges)))
    ax.set_xticklabels([_alpha_label(v) for v in edges], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(siblings)))
    ax.set_yticklabels([_alpha_label(v) for v in siblings])
    ax.set_xlabel("edge alpha")
    ax.set_ylabel("sibling alpha")
    ax.set_title(title, weight="bold")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                text = f"{matrix[i, j]:.3f}" if value != "n_clusters" else f"{int(matrix[i, j])}"
                ax.text(j, i, text, ha="center", va="center", fontsize=8, color="black")
    cbar = plt.colorbar(image, ax=ax, fraction=0.047, pad=0.03)
    cbar.set_label(value)


def _plot_best_composition(ax, composition: pd.DataFrame) -> None:
    frame = composition.sort_values("size", ascending=True).copy()
    y = np.arange(len(frame))
    left = np.zeros(len(frame), dtype=float)
    for digit in range(10):
        values = []
        for counts in frame["digit_counts"].map(parse_digit_counts):
            values.append(counts.get(digit, 0))
        ax.barh(y, values, left=left, color=DIGIT_COLORS[digit], label=str(digit), edgecolor="none")
        left += np.asarray(values, dtype=float)
    labels = [
        f"C{int(row.cluster)}  n={int(row.size)}  top={int(row.dominant_digit)} ({row.purity:.0%})"
        for row in frame.itertuples()
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("samples")
    ax.set_title("Best MNIST TBS run: cluster composition by true digit", weight="bold")
    ax.legend(title="digit", ncol=10, bbox_to_anchor=(0.5, -0.11), loc="upper center", frameon=False)
    ax.grid(axis="x", color="#e5e7eb", lw=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_failure_baseline(ax, binary: pd.DataFrame, continuous: pd.DataFrame, summary: pd.DataFrame) -> None:
    best_sweep = summary.sort_values("ARI", ascending=False).iloc[0]
    rows = [
        {
            "run": "best alpha sweep\ncontinuous PCA50 Ward",
            "ARI": best_sweep["ARI"],
            "NMI": best_sweep["NMI"],
            "clusters": int(best_sweep["n_clusters"]),
        },
        {
            "run": "best fixed continuous\nPCA50 complete",
            "ARI": continuous["ARI"].max(),
            "NMI": continuous.loc[continuous["ARI"].idxmax(), "NMI"],
            "clusters": int(continuous.loc[continuous["ARI"].idxmax(), "n_clusters"]),
        },
        {
            "run": "old binary run\njaccard/dice single",
            "ARI": binary["ARI"].max(),
            "NMI": binary.loc[binary["ARI"].idxmax(), "NMI"],
            "clusters": int(binary.loc[binary["ARI"].idxmax(), "n_clusters"]),
        },
    ]
    frame = pd.DataFrame(rows)
    x = np.arange(len(frame))
    width = 0.36
    ax.bar(x - width / 2, frame["ARI"], width, color="#2563eb", label="ARI")
    ax.bar(x + width / 2, frame["NMI"], width, color="#059669", label="NMI")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{row.run}\n{row.clusters} clusters" for row in frame.itertuples()],
        fontsize=9,
    )
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("score")
    ax.set_title("Which MNIST result should be trusted?", weight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#e5e7eb", lw=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    summary = pd.read_csv(SWEEP_DIR / "alpha_sweep_summary.csv")
    binary = pd.read_csv(SOURCE_DIR / "mnist_benchmark_summary.csv")
    continuous = pd.read_csv(SOURCE_DIR / "mnist_continuous_pca50_summary.csv")
    best = summary.sort_values("ARI", ascending=False).iloc[0]
    best_key = _assignment_key(str(best["linkage"]), float(best["edge_alpha"]), float(best["sibling_alpha"]))
    composition = pd.read_csv(SWEEP_DIR / f"{best_key}_top_cluster_digit_composition.csv")

    analysis_summary = best_rows(summary)
    analysis_summary.insert(0, "generated_at", generated_at)
    analysis_summary.to_csv(OUT_CSV, index=False)

    fig, ax = plt.subplots(figsize=(9, 6.2), facecolor="white")
    _plot_best_summary(ax, summary)
    fig.text(
        0.01,
        0.01,
        "Saved MNIST run: 2,000 normalized images, PCA50 (83.13% variance), TBS continuous alpha sweep.",
        fontsize=8.5,
        color="#374151",
    )
    _add_generated_at(fig, generated_at)
    fig.savefig(OUT_MAIN, dpi=220, bbox_inches="tight")
    plt.close(fig)

    with PdfPages(
        OUT_PDF,
        metadata={
            "Title": "MNIST TBS static analysis report",
            "Subject": f"Generated at: {generated_at}",
        },
    ) as pdf:
        for plotter in [
            lambda ax: _plot_best_summary(ax, summary),
            lambda ax: _plot_heatmap(ax, summary, "ward", "ARI", "Ward linkage: ARI reaction to alpha", "viridis"),
            lambda ax: _plot_heatmap(
                ax,
                summary,
                "ward",
                "n_clusters",
                "Ward linkage: cluster-count reaction to alpha",
                "magma",
            ),
            lambda ax: _plot_best_composition(ax, composition),
            lambda ax: _plot_failure_baseline(ax, binary, continuous, summary),
        ]:
            fig, ax = plt.subplots(figsize=(9, 6.4), facecolor="white")
            plotter(ax)
            fig.tight_layout()
            _add_generated_at(fig, generated_at)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(OUT_PDF)
    print(OUT_MAIN)
    print(OUT_CSV)


def run_cli() -> None:
    """Validate command-line arguments before generating retained figures."""

    argparse.ArgumentParser(description=__doc__).parse_args()
    main()


if __name__ == "__main__":
    run_cli()
