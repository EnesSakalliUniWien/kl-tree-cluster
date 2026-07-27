"""Goncalves fetal pancreas progenitor scRNA clustering benchmark.

The input is the processed UCSC Cell Browser matrix from Goncalves et al.
human fetal pancreas development. The script downloads or reads the matrix,
builds a conventional Scanpy post-count workflow, then reuses the existing
pancreas benchmark methods for classical clustering and TBS comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/kl_te_cluster_matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/kl_te_cluster_numba")

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from scipy import sparse

from applications.scrna.pancreas_benchmark import (
    _json_default,
    _run_benchmarks,
    _scanpy_module,
    _scanpy_version,
    _write_plots,
    _write_qc_outputs,
)

DATASET_ID = "goncalves_human_pancreas_dev_fetal_pancreas"
EXPR_URL = "https://cells.ucsc.edu/human-pancreas-dev/fetal-pancreas/exprMatrix.tsv.gz"
META_URL = "https://cells.ucsc.edu/human-pancreas-dev/fetal-pancreas/meta.tsv"
CELLTYPE_CANDIDATES = (
    "population",
    "Population",
    "celltype",
    "Celltype",
    "cell_type",
    "Cell Type",
    "cluster",
    "Cluster",
    "annotation",
    "Annotation",
    "identity",
    "Identity",
)
BATCH_CANDIDATES = (
    "sample",
    "Sample",
    "donor",
    "Donor",
    "age",
    "Age",
    "stage",
    "Stage",
    "batch",
    "Batch",
)


def _anndata_module() -> Any:
    import anndata as ad

    return ad


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _download_file(url: str, path: Path, *, timeout: int = 120) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    request = urllib.request.Request(url, headers={"User-Agent": "kl-te-cluster-benchmark/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle)
    except (urllib.error.URLError, TimeoutError) as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download {url}. Provide the file at {path} and rerun with "
            "--skip-download, or rerun where outbound HTTPS/DNS is available."
        ) from exc
    temporary.replace(path)


def _ensure_inputs(expr_path: Path, meta_path: Path, *, skip_download: bool) -> None:
    if skip_download:
        missing = [str(path) for path in (expr_path, meta_path) if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing required local input(s) with --skip-download: " + ", ".join(missing)
            )
        return
    if not expr_path.exists():
        _download_file(EXPR_URL, expr_path)
    if not meta_path.exists():
        _download_file(META_URL, meta_path)


def _first_present(columns: pd.Index, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    lower_to_original = {str(column).lower(): str(column) for column in columns}
    for candidate in candidates:
        matched = lower_to_original.get(candidate.lower())
        if matched is not None:
            return matched
    return None


def _read_ucsc_expression(expr_path: Path) -> tuple[pd.DataFrame, str]:
    expression = pd.read_csv(expr_path, sep="\t", index_col=0)
    expression.index = expression.index.astype(str)
    expression.columns = expression.columns.astype(str)
    if expression.shape[0] < expression.shape[1]:
        # UCSC matrices are normally genes x cells. Keep this guard for already
        # transposed local exports.
        return expression, "cells_by_genes"
    return expression.T, "genes_by_cells_transposed"


def _read_metadata(meta_path: Path, cell_ids: pd.Index) -> pd.DataFrame:
    metadata = pd.read_csv(meta_path, sep="\t", index_col=0)
    metadata.index = metadata.index.astype(str)
    missing = cell_ids.difference(metadata.index)
    if len(missing):
        raise ValueError(
            "Expression cells are missing from metadata. First missing ids: "
            + ", ".join(missing[:5].astype(str))
        )
    metadata = metadata.loc[cell_ids].copy()

    celltype_column = _first_present(metadata.columns, CELLTYPE_CANDIDATES)
    if celltype_column is None:
        metadata["celltype"] = "unknown"
        celltype_column = "celltype"
    metadata["celltype"] = metadata[celltype_column].astype(str).replace({"nan": "unknown"})

    batch_column = _first_present(metadata.columns, BATCH_CANDIDATES)
    if batch_column is None:
        metadata["batch"] = "unknown"
        metadata["sample"] = "unknown"
    else:
        metadata["batch"] = metadata[batch_column].astype(str).replace({"nan": "unknown"})
        metadata["sample"] = metadata["batch"]

    return metadata


def _build_adata(expr_path: Path, meta_path: Path, output_dir: Path) -> tuple[Any, str]:
    ad = _anndata_module()
    sc = _scanpy_module()
    matrix, orientation = _read_ucsc_expression(expr_path)
    metadata = _read_metadata(meta_path, matrix.index)
    values = matrix.to_numpy(dtype=np.float32, copy=False)
    finite_values = values[np.isfinite(values)]
    min_value = float(finite_values.min()) if finite_values.size else 0.0
    max_value = float(finite_values.max()) if finite_values.size else 0.0
    count_like = bool(
        min_value >= 0.0
        and finite_values.size
        and np.allclose(finite_values[: min(250_000, finite_values.size)] % 1.0, 0.0)
    )
    expression = sparse.csr_matrix(values) if count_like else values
    adata = ad.AnnData(
        X=expression,
        obs=metadata,
        var=pd.DataFrame(index=matrix.columns.astype(str)),
    )
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    adata.layers["input_expression"] = adata.X.copy()
    if count_like:
        adata.layers["counts"] = adata.X.copy()
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    if count_like:
        sc.pp.calculate_qc_metrics(
            adata,
            qc_vars=["mt"] if bool(adata.var["mt"].any()) else [],
            percent_top=None,
            inplace=True,
        )
    else:
        if "nFeature_RNA" in adata.obs:
            adata.obs["n_genes_by_counts"] = pd.to_numeric(
                adata.obs["nFeature_RNA"],
                errors="coerce",
            )
        else:
            adata.obs["n_genes_by_counts"] = np.count_nonzero(values, axis=1)
        if "nCount_RNA" in adata.obs:
            adata.obs["total_counts"] = pd.to_numeric(adata.obs["nCount_RNA"], errors="coerce")
        else:
            adata.obs["total_counts"] = np.nan
    if "pct_counts_mt" not in adata.obs:
        adata.obs["pct_counts_mt"] = np.nan
    adata.obs["qc_n_genes_by_counts"] = adata.obs["n_genes_by_counts"]
    adata.obs["qc_total_counts"] = adata.obs["total_counts"]
    adata.uns["input_orientation"] = orientation
    adata.uns["input_expression_kind"] = "count_like" if count_like else "processed_scaled"
    adata.uns["input_expression_min"] = min_value
    adata.uns["input_expression_max"] = max_value
    output_dir.mkdir(parents=True, exist_ok=True)
    return adata, orientation


def _prepare_classical_workflow(
    adata: Any,
    output_dir: Path,
    *,
    n_pcs: int,
    n_hvgs: int,
) -> tuple[Any, int]:
    sc = _scanpy_module()
    expression_kind = str(adata.uns.get("input_expression_kind", "unknown"))
    if expression_kind == "count_like":
        sc.pp.normalize_total(adata, target_sum=10_000)
        sc.pp.log1p(adata)
        adata.raw = adata
        if adata.n_vars > n_hvgs:
            sc.pp.highly_variable_genes(adata, n_top_genes=n_hvgs, flavor="seurat")
            adata = adata[:, adata.var["highly_variable"]].copy()
    else:
        adata.raw = adata
        if adata.n_vars > n_hvgs:
            values = np.asarray(adata.X, dtype=np.float32)
            variances = np.nanvar(values, axis=0)
            variances[~np.isfinite(variances)] = -np.inf
            selected = np.argsort(variances)[-int(n_hvgs) :]
            selected.sort()
            adata.var["highly_variable"] = False
            adata.var.iloc[selected, adata.var.columns.get_loc("highly_variable")] = True
            adata = adata[:, adata.var["highly_variable"]].copy()
    effective_pcs = max(2, min(int(n_pcs), adata.n_obs - 1, adata.n_vars - 1))
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=effective_pcs, svd_solver="arpack", random_state=0)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=effective_pcs, random_state=0)
    sc.tl.umap(adata, random_state=0)
    sc.tl.leiden(adata, resolution=1.0, random_state=0, key_added="classical_leiden")
    adata.write_h5ad(output_dir / "goncalves_fetal_pancreas_classical_pipeline.h5ad")
    return adata, effective_pcs


def _write_dataset_report(
    *,
    output_dir: Path,
    expr_path: Path,
    meta_path: Path,
    adata: Any,
    qc_summary: dict[str, object],
    results: pd.DataFrame,
    max_cells: int,
    n_pcs: int,
    effective_pcs: int,
    seed: int,
    elapsed_sec: float,
    generated_at: str,
) -> None:
    label_counts = adata.obs["celltype"].astype(str).value_counts()
    top_counts = "\n".join(f"- {label}: {count}" for label, count in label_counts.items())
    expression_kind = str(adata.uns.get("input_expression_kind", "unknown"))
    has_raw_counts = "counts" in adata.layers
    if expression_kind == "count_like":
        workflow_note = (
            "The script preserves the downloaded count matrix in `layers[\"counts\"]`, "
            "computes standard cell-level QC metrics, normalizes to 10,000 counts per "
            "cell, applies `log1p`, selects highly variable genes when needed, and "
            "recomputes PCA, neighbors, UMAP, and Leiden clustering."
        )
    else:
        workflow_note = (
            "The UCSC expression matrix is processed/scaled expression rather than raw "
            "counts, so the script preserves it in `layers[\"input_expression\"]`, uses "
            "metadata `nCount_RNA` and `nFeature_RNA` for count QC summaries, skips "
            "count normalization/log1p, selects high-variance genes, and recomputes "
            "PCA, neighbors, UMAP, and Leiden clustering."
        )
    result_columns = [
        "method",
        "status",
        "significance_level",
        "edge_alpha",
        "n_clusters",
        "overcluster_ratio",
        "weighted_cluster_purity",
        "weighted_label_dominant_cluster_recall",
        "merge_error_rate",
        "split_error_rate",
        "v_measure",
        "nmi",
        "ari",
        "silhouette",
        "elapsed_sec",
        "skip_reason",
    ]
    available_columns = [column for column in result_columns if column in results.columns]
    results_table = results[available_columns].to_markdown(index=False, floatfmt=".4f")
    report = f"""# Goncalves fetal pancreas progenitor benchmark

Generated at: {generated_at}

## Data

- Dataset: Goncalves et al. human fetal pancreas development, UCSC Cell Browser
  fetal-pancreas matrix.
- Expression source: `{EXPR_URL}`.
- Metadata source: `{META_URL}`.
- Local expression path: `{expr_path}`.
- Local metadata path: `{meta_path}`.
- Shape: {adata.n_obs} cells x {adata.n_vars} genes after feature selection.
- Input expression kind: `{expression_kind}`.
- Cell-type labels: {qc_summary["celltype_count"]}.
- Batches/samples: {qc_summary["batches"]}.

Label counts:

{top_counts}

## Workflow

{workflow_note} The benchmark then runs the same classical and TBS method set
used for the adult pancreas comparison.

Requested PCs: {n_pcs}. Effective PCs: {effective_pcs}. Maximum benchmark cells:
{max_cells}. Seed: {seed}. Elapsed seconds: {elapsed_sec:.2f}.

TBS settings are sibling alpha `0.01`, edge alpha `0.001`, local adaptive
projected-Wald dimensions at 90% contrast energy, and separate topology-only,
recomputed-NNLS branch-time, and raw-linkage diagnostic rows.

## QC Notes

- Raw counts preserved in `layers["counts"]`: {str(has_raw_counts).lower()}.
- Input expression preserved in `layers["input_expression"]`: true.
- Raw/log-normalized snapshot present: {qc_summary["raw_layer_present"]}.
- Mitochondrial genes detected by `MT-` prefix: {qc_summary["mitochondrial_gene_count"]}.
- Doublet status: {qc_summary["scdblfinder_status"]}.
- Ambient RNA status: {qc_summary["ambient_rna_status"]}.

## Results

{results_table}

## Files

- `goncalves_fetal_pancreas_classical_pipeline.h5ad`
- `qc_cell_metrics.csv`
- `qc_metric_distributions.png`
- `method_metrics.csv`
- `method_assignments.csv`
- `celltype_fragmentation_by_method.csv`
- `cluster_composition_by_method.csv`
- `method_umap_clusters.png`
- `method_split_merge_diagnostic.png`
- `edge_gate_distance_time_model_analysis.md`
- `tbs_tree_branch_length_summary.csv`
- `*_tree_edges.csv`
- `*_traversal_trace.csv`
- `*_full_edge_traversal_trace.csv`
- `manifest.json`
"""
    (output_dir / "summary.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    default_input_dir = (
        _project_root() / "raw" / "inbox" / "goncalves_human_pancreas_dev" / "fetal-pancreas"
    )
    parser.add_argument("--expr-matrix", type=Path, default=default_input_dir / "exprMatrix.tsv.gz")
    parser.add_argument("--metadata", type=Path, default=default_input_dir / "meta.tsv")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--max-cells", type=int, default=2500)
    parser.add_argument("--n-pcs", type=int, default=30)
    parser.add_argument("--n-hvgs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            _project_root()
            / "raw"
            / "assets"
            / "benchmark-results"
            / "goncalves_fetal_pancreas_progenitor_benchmark_20260624"
        ),
    )
    args = parser.parse_args()

    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    start = time.perf_counter()
    try:
        _ensure_inputs(args.expr_matrix, args.metadata, skip_download=args.skip_download)
    except (FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    if args.download_only:
        print(f"Downloaded inputs to {args.expr_matrix.parent}")
        return

    adata, orientation = _build_adata(args.expr_matrix, args.metadata, args.output_dir)
    adata, effective_pcs = _prepare_classical_workflow(
        adata,
        args.output_dir,
        n_pcs=args.n_pcs,
        n_hvgs=args.n_hvgs,
    )
    qc_summary = _write_qc_outputs(adata, args.output_dir)
    (
        results,
        assignments,
        tree_summaries,
        adaptive_metadata,
        length_sensitivity_rows,
    ) = _run_benchmarks(
        adata,
        args.output_dir,
        max_cells=args.max_cells,
        n_pcs=effective_pcs,
        seed=args.seed,
        generated_at=generated_at,
    )
    _write_plots(adata, results, assignments, args.output_dir)

    elapsed_sec = time.perf_counter() - start
    manifest: dict[str, Any] = {
        "generated_at": generated_at,
        "dataset_id": DATASET_ID,
        "expression_url": EXPR_URL,
        "metadata_url": META_URL,
        "expr_matrix": str(args.expr_matrix),
        "metadata": str(args.metadata),
        "output_dir": str(args.output_dir),
        "input_orientation": orientation,
        "input_expression_kind": adata.uns.get("input_expression_kind"),
        "max_cells": args.max_cells,
        "requested_n_pcs": args.n_pcs,
        "effective_n_pcs": effective_pcs,
        "n_hvgs": args.n_hvgs,
        "seed": args.seed,
        "elapsed_sec": elapsed_sec,
        "python": platform.python_version(),
        "scanpy": _scanpy_version(),
        "qc_summary": qc_summary,
        "adaptive_diffusion_metadata": adaptive_metadata,
        "tbs_tree_summaries": tree_summaries,
        "tbs_branch_time_sensitivity_rows": length_sensitivity_rows,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_default)
    )
    _write_dataset_report(
        output_dir=args.output_dir,
        expr_path=args.expr_matrix,
        meta_path=args.metadata,
        adata=adata,
        qc_summary=qc_summary,
        results=results,
        max_cells=args.max_cells,
        n_pcs=args.n_pcs,
        effective_pcs=effective_pcs,
        seed=args.seed,
        elapsed_sec=elapsed_sec,
        generated_at=generated_at,
    )
    print(results.to_string(index=False))
    print(f"\nWrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
