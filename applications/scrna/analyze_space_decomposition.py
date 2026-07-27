#!/usr/bin/env python3
"""Analyze scRNA benchmark PCA space by invariant and equivariant axes.

This is a diagnostic layer over the saved scRNA benchmark subsets. It treats
the leading standardized-PCA spectral axis as a common/invariant axis and the
next orthogonal axes as equivariant variation. This is intentionally a
PCA/spectral diagnostic, not a formal Cartan decomposition claim.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tree_break_selection.space_separation import (
    decompose_invariant_equivariant_space,
)

SCHEMA_VERSION = "scrna_space_decomposition/v1"
STUDY_ROLE = "diagnostic_scrna_invariant_equivariant_axis_decomposition"
GENERATED_BY = "applications/scrna/analyze_space_decomposition.py"
DEFAULT_OUTPUT_DIR = Path(
    "raw/assets/benchmark-results/scrna_space_decomposition_20260627"
)
DEFAULT_DATASETS = (
    (
        "adult_pancreas",
        Path("raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623"),
    ),
    (
        "goncalves_fetal_pancreas",
        Path(
            "raw/assets/benchmark-results/"
            "goncalves_fetal_pancreas_progenitor_benchmark_20260624"
        ),
    ),
)


@dataclass(frozen=True)
class DatasetConfig:
    """Input and output contract for one saved scRNA benchmark dataset."""

    dataset_id: str
    input_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run invariant/equivariant spectral-space diagnostics on saved "
            "adult and Goncalves scRNA benchmark PCA subsets."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV, PNG, and manifest outputs.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help=(
            "Optional dataset specs as dataset_id=/path/to/benchmark_dir. "
            "Defaults to the adult pancreas and Goncalves benchmark outputs."
        ),
    )
    parser.add_argument(
        "--equivariant-dim",
        type=int,
        default=5,
        help="Number of orthogonal axes after the invariant axis to summarize.",
    )
    return parser.parse_args()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _dataset_configs(raw_specs: list[str] | None) -> list[DatasetConfig]:
    if not raw_specs:
        return [DatasetConfig(dataset_id, path) for dataset_id, path in DEFAULT_DATASETS]
    configs: list[DatasetConfig] = []
    for spec in raw_specs:
        if "=" not in spec:
            raise ValueError(
                "Dataset specs must have form dataset_id=/path/to/benchmark_dir; "
                f"got {spec!r}."
            )
        dataset_id, raw_path = spec.split("=", 1)
        dataset_id = dataset_id.strip()
        if not dataset_id:
            raise ValueError(f"Dataset id is empty in spec {spec!r}.")
        configs.append(DatasetConfig(dataset_id, Path(raw_path)))
    return configs


def _load_pca(input_dir: Path) -> pd.DataFrame:
    path = input_dir / "benchmark_subset_pca.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    table = pd.read_csv(path)
    id_column = table.columns[0]
    table = table.rename(columns={id_column: "cell_id"})
    pc_columns = [column for column in table.columns if str(column).startswith("PC")]
    if len(pc_columns) < 3:
        raise ValueError(f"{path} must contain at least three PC columns.")
    table["cell_id"] = table["cell_id"].astype(str)
    return table.set_index("cell_id")[pc_columns].astype(float)


def _load_assignments(input_dir: Path, cell_ids: pd.Index) -> pd.DataFrame:
    path = input_dir / "method_assignments.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    table = pd.read_csv(path)
    required = {"cell_id", "celltype"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)!r}")
    table["cell_id"] = table["cell_id"].astype(str)
    table = table.set_index("cell_id", drop=False)
    missing_cells = cell_ids.difference(table.index)
    if len(missing_cells):
        raise ValueError(
            f"{path} is missing {len(missing_cells)} benchmark cells; "
            f"first missing ids: {missing_cells[:5].tolist()!r}."
        )
    return table.loc[cell_ids].reset_index(drop=True)


def _safe_quantile(values: pd.Series | np.ndarray, q: float) -> float:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float(numeric.quantile(q))


def _between_fraction(values: pd.Series, labels: pd.Series) -> float:
    data = pd.DataFrame(
        {
            "value": pd.to_numeric(values, errors="coerce"),
            "label": labels.astype(str),
        }
    ).dropna()
    if data.empty or data["label"].nunique() <= 1:
        return float("nan")
    overall = float(data["value"].mean())
    total = float(((data["value"] - overall) ** 2).sum())
    if total <= 0.0:
        return float("nan")
    grouped = data.groupby("label", dropna=False)["value"]
    between = sum(
        len(group) * float((group.mean() - overall) ** 2) for _, group in grouped
    )
    return float(between / total)


def _geometry_table(
    *,
    dataset_id: str,
    pca: pd.DataFrame,
    assignments: pd.DataFrame,
    coords: np.ndarray,
    energy_fraction: np.ndarray,
) -> pd.DataFrame:
    invariant = coords[:, 0]
    equivariant = coords[:, 1:]
    equivariant_radius = (
        np.linalg.norm(equivariant, axis=1)
        if equivariant.shape[1]
        else np.zeros(len(invariant), dtype=float)
    )
    total_radius = np.linalg.norm(coords, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        signed_cos = np.divide(
            invariant,
            total_radius,
            out=np.zeros_like(invariant, dtype=float),
            where=total_radius > 0.0,
        )
        equivariant_fraction = np.divide(
            equivariant_radius**2,
            total_radius**2,
            out=np.zeros_like(equivariant_radius, dtype=float),
            where=total_radius > 0.0,
        )
    angle = np.degrees(np.arccos(np.clip(np.abs(signed_cos), 0.0, 1.0)))
    table = pd.DataFrame(
        {
            "dataset_id": dataset_id,
            "cell_id": pca.index.astype(str),
            "celltype": assignments["celltype"].astype(str).to_numpy(),
            "invariant_axis_score": invariant,
            "abs_invariant_axis_score": np.abs(invariant),
            "equivariant_radius": equivariant_radius,
            "total_radius": total_radius,
            "signed_cosine_to_invariant_axis": signed_cos,
            "angle_to_invariant_axis_deg": angle,
            "equivariant_energy_fraction_cell": equivariant_fraction,
            "global_invariant_axis_energy_fraction": float(energy_fraction[0]),
            "global_equivariant_axes_energy_fraction": float(energy_fraction[1:].sum()),
        }
    )
    for axis_index in range(coords.shape[1]):
        role = "invariant" if axis_index == 0 else "equivariant"
        table[f"{role}_axis_{axis_index + 1}_score"] = coords[:, axis_index]
    return table


def _celltype_summary(geometry: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for celltype, group in geometry.groupby("celltype", dropna=False):
        rows.append(
            {
                "dataset_id": str(group["dataset_id"].iloc[0]),
                "celltype": celltype,
                "n_cells": int(len(group)),
                "invariant_axis_mean": float(group["invariant_axis_score"].mean()),
                "abs_invariant_axis_median": _safe_quantile(
                    group["abs_invariant_axis_score"], 0.5
                ),
                "equivariant_radius_mean": float(group["equivariant_radius"].mean()),
                "equivariant_radius_median": _safe_quantile(
                    group["equivariant_radius"], 0.5
                ),
                "angle_to_invariant_axis_deg_median": _safe_quantile(
                    group["angle_to_invariant_axis_deg"], 0.5
                ),
                "equivariant_energy_fraction_cell_median": _safe_quantile(
                    group["equivariant_energy_fraction_cell"], 0.5
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset_id", "n_cells"], ascending=[True, False])


def _method_columns(assignments: pd.DataFrame) -> list[str]:
    excluded = {"cell_id", "celltype", "umap1", "umap2"}
    return [column for column in assignments.columns if column not in excluded]


def _assignment_key_from_label(label: str) -> str:
    """Mirror MethodConfig.assignment_key from the scRNA benchmark runner."""

    return (
        str(label)
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("=", "")
        .replace(".", "p")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
    )


def _cluster_summary(
    *,
    dataset_id: str,
    geometry: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    base = geometry.copy()
    for method in _method_columns(assignments):
        labels = assignments[method]
        if labels.isna().all():
            continue
        working = base.copy()
        working["cluster_id"] = labels.astype(str).to_numpy()
        for cluster_id, group in working.groupby("cluster_id", dropna=False):
            composition = group["celltype"].astype(str).value_counts()
            dominant = str(composition.index[0])
            purity = float(composition.iloc[0] / len(group))
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "method": method,
                    "cluster_id": cluster_id,
                    "n_cells": int(len(group)),
                    "dominant_celltype": dominant,
                    "dominant_celltype_purity": purity,
                    "invariant_axis_mean": float(group["invariant_axis_score"].mean()),
                    "invariant_axis_q25": _safe_quantile(
                        group["invariant_axis_score"], 0.25
                    ),
                    "invariant_axis_q75": _safe_quantile(
                        group["invariant_axis_score"], 0.75
                    ),
                    "equivariant_radius_mean": float(group["equivariant_radius"].mean()),
                    "equivariant_radius_median": _safe_quantile(
                        group["equivariant_radius"], 0.5
                    ),
                    "angle_to_invariant_axis_deg_median": _safe_quantile(
                        group["angle_to_invariant_axis_deg"], 0.5
                    ),
                    "equivariant_energy_fraction_cell_median": _safe_quantile(
                        group["equivariant_energy_fraction_cell"], 0.5
                    ),
                }
            )
    return pd.DataFrame(rows)


def _method_summary(
    *,
    dataset_id: str,
    geometry: pd.DataFrame,
    assignments: pd.DataFrame,
    metrics: pd.DataFrame | None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in _method_columns(assignments):
        labels = assignments[method]
        if labels.isna().all():
            continue
        label_strings = labels.astype(str)
        rows.append(
            {
                "dataset_id": dataset_id,
                "method": method,
                "n_clusters": int(label_strings.nunique()),
                "cluster_between_fraction_invariant_axis": _between_fraction(
                    geometry["invariant_axis_score"],
                    label_strings,
                ),
                "cluster_between_fraction_equivariant_radius": _between_fraction(
                    geometry["equivariant_radius"],
                    label_strings,
                ),
                "celltype_between_fraction_invariant_axis": _between_fraction(
                    geometry["invariant_axis_score"],
                    geometry["celltype"],
                ),
                "celltype_between_fraction_equivariant_radius": _between_fraction(
                    geometry["equivariant_radius"],
                    geometry["celltype"],
                ),
            }
        )
    summary = pd.DataFrame(rows)
    if metrics is not None and not metrics.empty:
        metric_subset = metrics[
            ["method", "method_id", "n_clusters", "ari", "v_measure"]
        ].copy()
        metric_subset = metric_subset.rename(
            columns={
                "method": "method_label",
                "method_id": "benchmark_method_id",
            }
        )
        metric_subset["method"] = metric_subset["method_label"].map(
            _assignment_key_from_label
        )
        duplicate_keys = metric_subset["method"][metric_subset["method"].duplicated()]
        if not duplicate_keys.empty:
            raise ValueError(
                "method_metrics.csv does not map uniquely to assignment columns; "
                f"duplicate assignment keys: {sorted(duplicate_keys.unique())!r}."
            )
        summary = summary.merge(
            metric_subset,
            on="method",
            how="left",
            suffixes=("", "_benchmark"),
        )
    return summary


def _load_metrics(input_dir: Path) -> pd.DataFrame | None:
    path = input_dir / "method_metrics.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _axis_loadings(
    *,
    dataset_id: str,
    columns: pd.Index,
    axes: np.ndarray,
    singular_values: np.ndarray,
    energy_fraction: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for axis_index, vector in enumerate(axes):
        role = "invariant" if axis_index == 0 else "equivariant"
        for feature, loading in zip(columns, vector, strict=True):
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "axis_index": int(axis_index + 1),
                    "axis_role": role,
                    "feature": str(feature),
                    "loading": float(loading),
                    "abs_loading": float(abs(loading)),
                    "singular_value": float(singular_values[axis_index]),
                    "axis_energy_fraction": float(energy_fraction[axis_index]),
                }
            )
    return pd.DataFrame(rows)


def _plot_dataset_space(geometry: pd.DataFrame, path: Path) -> None:
    celltypes = geometry["celltype"].astype(str)
    categories = celltypes.astype("category")
    codes = categories.cat.codes.to_numpy()
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    scatter = ax.scatter(
        geometry["invariant_axis_score"],
        geometry["equivariant_radius"],
        c=codes,
        cmap="tab20",
        s=12,
        alpha=0.72,
        linewidths=0,
    )
    ax.set_title(f"{geometry['dataset_id'].iloc[0]} space decomposition")
    ax.set_xlabel("Invariant axis score")
    ax.set_ylabel("Equivariant radius")
    handles, _ = scatter.legend_elements(num=min(12, len(categories.cat.categories)))
    labels = list(categories.cat.categories[: len(handles)])
    if handles:
        ax.legend(
            handles,
            labels,
            title="celltype",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=7,
        )
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_method_summary(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        return
    table = summary.sort_values(
        "cluster_between_fraction_equivariant_radius",
        ascending=False,
    )
    fig, ax = plt.subplots(figsize=(10.5, max(4.0, 0.42 * len(table))))
    y = np.arange(len(table))
    ax.barh(
        y - 0.18,
        table["cluster_between_fraction_invariant_axis"],
        height=0.34,
        label="invariant axis",
        color="#2563eb",
    )
    ax.barh(
        y + 0.18,
        table["cluster_between_fraction_equivariant_radius"],
        height=0.34,
        label="equivariant radius",
        color="#dc2626",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(table["method"], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Between-cluster variance fraction")
    ax.set_title(f"{table['dataset_id'].iloc[0]} method geometry summary")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.22)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze_dataset(
    config: DatasetConfig,
    *,
    output_dir: Path,
    equivariant_dim: int,
) -> dict[str, object]:
    pca = _load_pca(config.input_dir)
    assignments = _load_assignments(config.input_dir, pca.index)
    decomposition = decompose_invariant_equivariant_space(
        pca.to_numpy(dtype=float),
        equivariant_dim=equivariant_dim,
    )
    coords = decomposition.coordinates
    axes = decomposition.axes
    singular_values = decomposition.singular_values
    energy_fraction = decomposition.energy_fraction
    dataset_output = output_dir / config.dataset_id
    dataset_output.mkdir(parents=True, exist_ok=True)

    geometry = _geometry_table(
        dataset_id=config.dataset_id,
        pca=pca,
        assignments=assignments,
        coords=coords,
        energy_fraction=energy_fraction,
    )
    celltype_summary = _celltype_summary(geometry)
    cluster_summary = _cluster_summary(
        dataset_id=config.dataset_id,
        geometry=geometry,
        assignments=assignments,
    )
    method_summary = _method_summary(
        dataset_id=config.dataset_id,
        geometry=geometry,
        assignments=assignments,
        metrics=_load_metrics(config.input_dir),
    )
    loadings = _axis_loadings(
        dataset_id=config.dataset_id,
        columns=pca.columns,
        axes=axes,
        singular_values=singular_values,
        energy_fraction=energy_fraction,
    )

    geometry_path = dataset_output / "cell_space_decomposition.csv"
    celltype_path = dataset_output / "celltype_space_summary.csv"
    cluster_path = dataset_output / "method_cluster_space_summary.csv"
    method_path = dataset_output / "method_space_summary.csv"
    loadings_path = dataset_output / "axis_loadings.csv"
    geometry.to_csv(geometry_path, index=False)
    celltype_summary.to_csv(celltype_path, index=False)
    cluster_summary.to_csv(cluster_path, index=False)
    method_summary.to_csv(method_path, index=False)
    loadings.to_csv(loadings_path, index=False)
    _plot_dataset_space(geometry, dataset_output / "cell_space_decomposition.png")
    _plot_method_summary(method_summary, dataset_output / "method_space_summary.png")

    return {
        "dataset_id": config.dataset_id,
        "input_dir": config.input_dir,
        "n_cells": int(len(geometry)),
        "n_pca_features": int(pca.shape[1]),
        "equivariant_dim": int(equivariant_dim),
        "invariant_axis_energy_fraction": float(energy_fraction[0]),
        "equivariant_axes_energy_fraction": float(energy_fraction[1:].sum()),
        "celltype_between_fraction_invariant_axis": _between_fraction(
            geometry["invariant_axis_score"],
            geometry["celltype"],
        ),
        "celltype_between_fraction_equivariant_radius": _between_fraction(
            geometry["equivariant_radius"],
            geometry["celltype"],
        ),
        "outputs": {
            "cell_space_decomposition": geometry_path,
            "celltype_space_summary": celltype_path,
            "method_cluster_space_summary": cluster_path,
            "method_space_summary": method_path,
            "axis_loadings": loadings_path,
            "cell_space_plot": dataset_output / "cell_space_decomposition.png",
            "method_space_plot": dataset_output / "method_space_summary.png",
        },
    }


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(UTC).isoformat()
    dataset_summaries = [
        analyze_dataset(
            config,
            output_dir=output_dir,
            equivariant_dim=args.equivariant_dim,
        )
        for config in _dataset_configs(args.datasets)
    ]
    combined_summary = pd.DataFrame(
        [
            {
                key: value
                for key, value in summary.items()
                if key not in {"outputs", "input_dir"}
            }
            | {"input_dir": str(summary["input_dir"])}
            for summary in dataset_summaries
        ]
    )
    combined_summary.to_csv(output_dir / "scrna_space_decomposition_summary.csv", index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at": generated_at,
        "output_dir": str(output_dir),
        "datasets": dataset_summaries,
        "config": {
            "equivariant_dim": int(args.equivariant_dim),
            "axis_contract": (
                "standardize saved benchmark PCA coordinates; decompose centered "
                "cell-by-PC matrix by SVD; axis 1 is the common/invariant axis; "
                "axes 2..k are the orthogonal/equivariant axes"
            ),
            "formal_cartan_claim": False,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
