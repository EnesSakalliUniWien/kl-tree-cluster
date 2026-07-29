"""Shared helpers for static scRNA report artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a generated artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(
    path: Path,
    *,
    role: str,
    relative_to: Path,
    generated_at: str | None = None,
) -> dict[str, object]:
    """Return the shared provenance record for one generated artifact."""

    record: dict[str, object] = {
        "path": str(path.relative_to(relative_to)),
        "role": role,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.suffix == ".csv":
        table = pd.read_csv(path)
        record["rows"] = len(table)
        record["columns"] = len(table.columns)
        if generated_at is not None:
            record["generated_at"] = generated_at
        if "generated_at" in table.columns:
            record["generated_at_values"] = sorted(
                table["generated_at"].dropna().unique().tolist()
            )
    return record


def plot_cluster_size_bars(
    ax,
    summary: pd.DataFrame,
    cluster_palette: dict[int, str],
    title: str,
    *,
    max_rows: int | None = None,
    ytick_fontsize: float | None = None,
) -> None:
    """Draw cluster-size bars with dominant cell-type labels."""

    summary = summary.sort_values("n_cells_assignment", ascending=False).copy()
    if max_rows is not None and len(summary) > max_rows:
        head = summary.head(max_rows - 1).copy()
        tail = summary.iloc[max_rows - 1 :]
        other = {
            "cluster": -1,
            "n_cells_assignment": int(tail["n_cells_assignment"].sum()),
            "top_celltype_assignment": f"remaining {len(tail)} clusters",
            "top_celltype_fraction_assignment": 1.0,
        }
        summary = pd.concat([head, pd.DataFrame([other])], ignore_index=True)

    summary = summary.sort_values("n_cells_assignment", ascending=True)
    labels = []
    colors = []
    for row in summary.itertuples(index=False):
        cluster = int(row.cluster)
        if cluster < 0:
            labels.append(str(row.top_celltype_assignment))
            colors.append("#9ca3af")
        else:
            labels.append(
                f"C{cluster}  {str(row.top_celltype_assignment)[:22]} "
                f"({row.top_celltype_fraction_assignment:.0%})"
            )
            colors.append(cluster_palette[cluster])

    y = list(range(len(summary)))
    ax.barh(
        y,
        summary["n_cells_assignment"],
        color=colors,
        edgecolor="none",
        alpha=0.92,
    )
    ax.set_yticks(y)
    if ytick_fontsize is None:
        ytick_fontsize = 5.6 if len(summary) > 30 else 7
    ax.set_yticklabels(labels, fontsize=ytick_fontsize)
    ax.set_xlabel("cells", fontsize=8)
    ax.set_title(title, fontsize=10, weight="bold", pad=6)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", color="#e5e7eb", lw=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
