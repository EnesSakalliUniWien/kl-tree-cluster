"""Plot Goncalves progenitor coherence scores for selected structures."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from applications.scrna.plots.report_helpers import artifact_record

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = (
    ROOT / "raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624"
)
OUT_PNG = OUT_DIR / "goncalves_progenitor_coherence_score.png"
OUT_PDF = OUT_DIR / "goncalves_progenitor_coherence_score.pdf"
OUT_CSV = OUT_DIR / "goncalves_progenitor_coherence_score.csv"
OUT_MANIFEST = OUT_DIR / "goncalves_progenitor_coherence_score_manifest.json"

SOURCES = {
    "cluster": OUT_DIR / "goncalves_tbs_cluster_progenitor_signature_scores.csv",
    "inner_node": OUT_DIR / "goncalves_tbs_inner_node_progenitor_signature_scores.csv",
    "meeting_node": OUT_DIR / "goncalves_tbs_monophyletic_meeting_progenitor_signature_scores.csv",
}

SELECTED = [
    ("C6", "cluster", "clear progenitor structure"),
    ("C1", "cluster", "clear progenitor structure"),
    ("N2851", "inner_node", "clear progenitor structure"),
    ("N2876", "meeting_node", "clear progenitor structure"),
    ("N2910", "meeting_node", "clear progenitor structure"),
    ("C22", "cluster", "broad mixed neighborhood"),
    ("N2907", "meeting_node", "broad mixed neighborhood"),
]


def _load_rows(generated_at: str) -> pd.DataFrame:
    tables = {key: pd.read_csv(path) for key, path in SOURCES.items()}
    records = []
    for label, source_key, class_name in SELECTED:
        table = tables[source_key]
        rows = table.loc[table["label"].eq(label)]
        if len(rows) != 1:
            raise ValueError(f"Expected one row for {label} in {source_key}, found {len(rows)}")
        row = rows.iloc[0]
        dominant_fraction = float(row["dominant_population_fraction"])
        progenitor_fraction = float(row["progenitor_population_fraction"])
        records.append(
            {
                "generated_at": generated_at,
                "label": label,
                "source": source_key,
                "class": class_name,
                "n_cells": int(row["n_cells"]),
                "dominant_population": row["dominant_population"],
                "dominant_population_fraction": dominant_fraction,
                "progenitor_population_fraction": progenitor_fraction,
                "coherence_score": (dominant_fraction + progenitor_fraction) / 2.0,
                "top_populations": row["top_populations"],
            }
        )
    return pd.DataFrame(records)


def _plot(score_table: pd.DataFrame, generated_at: str) -> None:
    ordered = score_table.sort_values("coherence_score", ascending=True).reset_index(drop=True)
    colors = ordered["class"].map(
        {
            "clear progenitor structure": "#2f9e44",
            "broad mixed neighborhood": "#6c757d",
        }
    )
    y = range(len(ordered))
    fig, ax = plt.subplots(figsize=(10.5, 5.8), facecolor="white")
    ax.barh(y, ordered["coherence_score"], color=colors, alpha=0.9, edgecolor="none")
    ax.scatter(
        ordered["dominant_population_fraction"],
        list(y),
        s=34,
        color="#0b7285",
        label="dominant population fraction",
        zorder=3,
    )
    ax.scatter(
        ordered["progenitor_population_fraction"],
        list(y),
        s=34,
        color="#e67700",
        label="progenitor population fraction",
        zorder=3,
    )

    clear_mean = score_table.loc[
        score_table["class"].eq("clear progenitor structure"), "coherence_score"
    ].mean()
    broad_mean = score_table.loc[
        score_table["class"].eq("broad mixed neighborhood"), "coherence_score"
    ].mean()
    ax.axvline(clear_mean, color="#2f9e44", ls="--", lw=1.2)
    ax.axvline(broad_mean, color="#6c757d", ls="--", lw=1.2)
    ax.text(
        clear_mean + 0.01,
        1.01,
        f"clear mean {clear_mean:.3f}",
        color="#2f9e44",
        fontsize=10,
        transform=ax.get_xaxis_transform(),
    )
    ax.text(
        broad_mean + 0.01,
        1.01,
        f"broad mean {broad_mean:.3f}",
        color="#495057",
        fontsize=10,
        transform=ax.get_xaxis_transform(),
    )

    labels = [
        f"{row.label}  {row.dominant_population}  n={row.n_cells}"
        for row in ordered.itertuples(index=False)
    ]
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("coherence score = mean(dominant population fraction, progenitor fraction)")
    ax.set_title(
        "Goncalves TBS progenitor coherence score",
        fontsize=14,
        weight="bold",
        pad=12,
    )
    ax.grid(axis="x", color="#e5e7eb", lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False, fontsize=9)

    for ypos, row in enumerate(ordered.itertuples(index=False)):
        ax.text(
            min(row.coherence_score + 0.015, 1.015),
            ypos,
            f"{row.coherence_score:.3f}",
            va="center",
            fontsize=9,
            color="#111827",
        )

    fig.text(
        0.01,
        0.01,
        f"Generated at: {generated_at}. Source rows are exact selected TBS clusters/nodes from the Goncalves progenitor analysis.",
        fontsize=8,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))
    fig.savefig(OUT_PNG, dpi=220)
    fig.savefig(OUT_PDF)
    plt.close(fig)


def _write_manifest(generated_at: str) -> None:
    artifacts = []
    for path, role in [
        (OUT_PNG, "Goncalves progenitor coherence score PNG"),
        (OUT_PDF, "Goncalves progenitor coherence score PDF"),
        (OUT_CSV, "Goncalves progenitor coherence score table"),
    ]:
        artifacts.append(
            artifact_record(path, role=role, relative_to=ROOT, generated_at=generated_at)
        )
    OUT_MANIFEST.write_text(
        json.dumps(
            {
                "manifest_schema_version": "static_artifact_provenance/v1",
                "generated_at": generated_at,
                "source_script": "applications/scrna/plots/goncalves_progenitor_coherence.py",
                "score_definition": "mean(dominant_population_fraction, progenitor_population_fraction)",
                "source_tables": [str(path.relative_to(ROOT)) for path in SOURCES.values()],
                "artifacts": artifacts,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    score_table = _load_rows(generated_at)
    score_table.to_csv(OUT_CSV, index=False)
    _plot(score_table, generated_at)
    _write_manifest(generated_at)
    print(OUT_PNG)
    print(OUT_PDF)
    print(OUT_CSV)
    print(OUT_MANIFEST)


if __name__ == "__main__":
    main()
