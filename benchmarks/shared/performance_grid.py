"""Write clear benchmark performance grids from canonical result rows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CASE_COLUMNS = [
    "test_case",
    "case_id",
    "case_category",
    "source_family",
    "feature_representation",
    "samples",
    "features",
    "true_clusters",
]

METRIC_COLUMNS = ["ari", "nmi", "purity"]
SUMMARY_METRICS = [
    "mean_ari",
    "median_ari",
    "mean_nmi",
    "mean_purity",
    "exact_k_rate",
    "mean_cluster_count_abs_error",
]


@dataclass(frozen=True)
class BenchmarkPerformanceGridArtifacts:
    """Paths emitted by the performance-grid writer."""

    report_md: Path
    summary_csv: Path
    ari_grid_csv: Path
    nmi_grid_csv: Path
    purity_grid_csv: Path
    status_grid_csv: Path


def _require_columns(results: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required.difference(results.columns))
    if missing:
        raise ValueError(f"Benchmark performance grid is missing result columns: {missing}")


def _normalize_grid_frame(results: pd.DataFrame) -> pd.DataFrame:
    required = set(CASE_COLUMNS) | {
        "method",
        "run_id",
        "benchmark_class",
        "benchmark_grid",
        "benchmark_repeat",
        "params",
        "found_clusters",
        "status",
        "skip_reason",
        "cluster_count_abs_error",
        *METRIC_COLUMNS,
    }
    _require_columns(results, required)

    frame = results.copy()
    for column in CASE_COLUMNS + [
        "method",
        "run_id",
        "benchmark_class",
        "benchmark_grid",
        "params",
        "status",
        "skip_reason",
    ]:
        frame[column] = frame[column].fillna("").astype(str)

    for column in [
        "benchmark_repeat",
        "found_clusters",
        "true_clusters",
        "cluster_count_abs_error",
        *METRIC_COLUMNS,
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    duplicated = frame.duplicated(["case_id", "run_id"], keep=False)
    if duplicated.any():
        duplicate_rows = frame.loc[duplicated, ["case_id", "run_id"]].drop_duplicates()
        examples = duplicate_rows.head(5).to_dict(orient="records")
        raise ValueError(
            "Benchmark performance grid requires unique case_id/run_id rows; "
            f"duplicates include {examples}."
        )
    return frame


def _case_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame[CASE_COLUMNS]
        .drop_duplicates()
        .sort_values(["test_case", "case_id"])
        .reset_index(drop=True)
    )


def _metric_grid(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    metadata = _case_metadata(frame)
    values = frame.pivot(index="case_id", columns="run_id", values=metric)
    values = values.reindex(metadata["case_id"])
    values = values.reindex(sorted(values.columns), axis=1)
    values = values.reset_index(drop=True)
    return pd.concat([metadata, values], axis=1)


def _status_grid(frame: pd.DataFrame) -> pd.DataFrame:
    metadata = _case_metadata(frame)
    values = frame.pivot(index="case_id", columns="run_id", values="status")
    values = values.reindex(metadata["case_id"])
    values = values.reindex(sorted(values.columns), axis=1)
    values = values.reset_index(drop=True)
    return pd.concat([metadata, values], axis=1)


def _first_nonempty(values: pd.Series) -> str:
    nonempty = values.dropna().astype(str)
    nonempty = nonempty[nonempty.str.len() > 0]
    return "" if nonempty.empty else str(nonempty.iloc[0])


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["ok"] = working["status"].eq("ok")
    working["exact_k"] = np.where(
        working["true_clusters"].notna() & working["found_clusters"].notna(),
        (working["true_clusters"] == working["found_clusters"]).astype(float),
        np.nan,
    )

    grouped = working.groupby("run_id", sort=False, dropna=False)
    summary = grouped.agg(
        method=("method", _first_nonempty),
        benchmark_class=("benchmark_class", _first_nonempty),
        benchmark_grid=("benchmark_grid", _first_nonempty),
        benchmark_repeat=("benchmark_repeat", "first"),
        params=("params", _first_nonempty),
        rows=("case_id", "size"),
        ok_rows=("ok", "sum"),
        skipped_rows=("ok", lambda values: int((~values).sum())),
        mean_ari=("ari", "mean"),
        median_ari=("ari", "median"),
        mean_nmi=("nmi", "mean"),
        mean_purity=("purity", "mean"),
        exact_k_rate=("exact_k", "mean"),
        mean_cluster_count_abs_error=("cluster_count_abs_error", "mean"),
    )
    summary = summary.reset_index()
    summary["skip_rate"] = np.where(
        summary["rows"] > 0,
        summary["skipped_rows"] / summary["rows"],
        np.nan,
    )
    summary = summary.sort_values(
        ["skip_rate", "mean_ari", "mean_nmi", "ok_rows", "run_id"],
        ascending=[True, False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    summary.insert(0, "rank_skip_rate_then_mean_ari", np.arange(1, len(summary) + 1))
    return summary


def _markdown_table(frame: pd.DataFrame, *, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    visible = frame.head(max_rows).copy()
    for column in SUMMARY_METRICS + ["skip_rate"]:
        if column in visible.columns:
            visible[column] = visible[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.4f}"
            )
    columns = list(visible.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _column in columns) + " |",
    ]
    for row in visible.itertuples(index=False):
        values = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_benchmark_performance_grid(
    results: pd.DataFrame,
    output_dir: Path | str,
    *,
    source_path: Path | str | None = None,
) -> BenchmarkPerformanceGridArtifacts:
    """Write case-by-run metric grids and ranked run-cell performance summaries."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = _normalize_grid_frame(results)

    summary = _summary(frame)
    ari_grid = _metric_grid(frame, "ari")
    nmi_grid = _metric_grid(frame, "nmi")
    purity_grid = _metric_grid(frame, "purity")
    status_grid = _status_grid(frame)

    summary_csv = output / "benchmark_performance_grid_summary.csv"
    ari_grid_csv = output / "benchmark_performance_grid_ari.csv"
    nmi_grid_csv = output / "benchmark_performance_grid_nmi.csv"
    purity_grid_csv = output / "benchmark_performance_grid_purity.csv"
    status_grid_csv = output / "benchmark_performance_grid_status.csv"
    report_md = output / "benchmark_performance_grid.md"

    summary.to_csv(summary_csv, index=False)
    ari_grid.to_csv(ari_grid_csv, index=False)
    nmi_grid.to_csv(nmi_grid_csv, index=False)
    purity_grid.to_csv(purity_grid_csv, index=False)
    status_grid.to_csv(status_grid_csv, index=False)

    source_line = f"- Source rows: `{source_path}`\n" if source_path is not None else ""
    report_md.write_text(
        "\n".join(
            [
                "# Benchmark Performance Grid",
                "",
                source_line.rstrip(),
                f"- Result rows: `{len(frame)}`",
                f"- Benchmark cases: `{frame['case_id'].nunique()}`",
                f"- Run cells: `{frame['run_id'].nunique()}`",
                "",
                "## Artifacts",
                "",
                f"- Summary by run cell: `{summary_csv.name}`",
                f"- ARI case grid: `{ari_grid_csv.name}`",
                f"- NMI case grid: `{nmi_grid_csv.name}`",
                f"- Purity case grid: `{purity_grid_csv.name}`",
                f"- Status case grid: `{status_grid_csv.name}`",
                "",
                "## Ranked Run Cells",
                "",
                "Sorted by skip rate first, then mean ARI on available rows.",
                "",
                _markdown_table(summary),
                "",
            ]
        ),
        encoding="utf-8",
    )

    return BenchmarkPerformanceGridArtifacts(
        report_md=report_md,
        summary_csv=summary_csv,
        ari_grid_csv=ari_grid_csv,
        nmi_grid_csv=nmi_grid_csv,
        purity_grid_csv=purity_grid_csv,
        status_grid_csv=status_grid_csv,
    )


__all__ = [
    "BenchmarkPerformanceGridArtifacts",
    "write_benchmark_performance_grid",
]
