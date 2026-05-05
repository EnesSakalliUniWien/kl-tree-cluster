#!/usr/bin/env python3
"""Build KAK-inspired MP regime trees and run the project TreeDecomposition gates."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score

from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.cluster_assignments import (
    build_sample_cluster_assignments,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree
from scripts.adaptive_cosine_spectral_blocks import adaptive_spectral_blocks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use KAK-inspired Marchenko-Pastur regimes as tree topologies, "
            "then run the existing TreeDecomposition gates without bypassing them."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Full GO feature matrix TSV.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha-local", type=float, default=float(config.EDGE_ALPHA))
    parser.add_argument("--sibling-alpha", type=float, default=float(config.SIBLING_ALPHA))
    parser.add_argument(
        "--blocks",
        nargs="+",
        default=["signal_adaptive"],
        choices=["signal", "signal_adaptive", "bulk", "full"],
        help="KAK MP regimes to convert into trees.",
    )
    parser.add_argument("--min-segment-length", type=int, default=4)
    parser.add_argument("--max-segments", type=int, default=10)
    return parser.parse_args()


def load_matrix(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, sep="\t", index_col=0)
    data = data.apply(pd.to_numeric, errors="raise").astype(float)
    data = data.loc[:, data.sum(axis=0) > 0]
    if data.empty:
        raise ValueError(f"No non-empty features in {path}")
    return data


def mp_bounds(n_samples: int, n_features: int) -> tuple[float, float]:
    q = float(n_features) / float(n_samples)
    root_q = math.sqrt(q)
    return max(1.0 - root_q, 0.0) ** 2, (1.0 + root_q) ** 2


def standardize_active_features(data: pd.DataFrame) -> tuple[np.ndarray, int]:
    values = data.to_numpy(dtype=float)
    std = values.std(axis=0, ddof=1)
    active = std > 1e-12
    values = values[:, active]
    std = values.std(axis=0, ddof=1)
    standardized = (values - values.mean(axis=0)) / std
    return np.nan_to_num(standardized), int(active.sum())


def kak_mp_gene_scores(standardized: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return nonzero dual-correlation eigenvalues and gene score basis.

    Nonzero feature-correlation eigenvalues equal eigenvalues of
    ``Z Z.T / (n - 1)``. The columns of ``U`` give the gene-side coordinates.
    """
    n_samples = standardized.shape[0]
    gram = (standardized @ standardized.T) / float(n_samples - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    keep = eigenvalues > 1e-10
    return eigenvalues[keep], eigenvectors[:, keep]


def cluster_sizes(assignments: pd.DataFrame) -> pd.DataFrame:
    return (
        assignments.groupby("cluster_id", as_index=False)
        .size()
        .rename(columns={"size": "n_genes"})
        .sort_values(["n_genes", "cluster_id"], ascending=[False, True])
    )


def top_terms_for_assignments(
    data: pd.DataFrame,
    labels: pd.Series,
    *,
    max_terms_per_cluster: int = 10,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cluster_id in sorted(labels.unique()):
        mask = labels == cluster_id
        if int(mask.sum()) == 0 or int((~mask).sum()) == 0:
            continue
        cluster_prevalence = data.loc[mask].mean(axis=0)
        rest_prevalence = data.loc[~mask].mean(axis=0)
        delta = (cluster_prevalence - rest_prevalence).sort_values(ascending=False)
        for term, value in delta.head(max_terms_per_cluster).items():
            rows.append(
                {
                    "cluster_id": int(cluster_id),
                    "cluster_size": int(mask.sum()),
                    "term": term,
                    "cluster_prevalence": float(cluster_prevalence[term]),
                    "rest_prevalence": float(rest_prevalence[term]),
                    "prevalence_delta": float(value),
                }
            )
    return pd.DataFrame(rows)


def build_regimes(
    eigenvalues: np.ndarray,
    *,
    mp_lower: float,
    mp_upper: float,
    requested_blocks: set[str],
    min_segment_length: int,
    max_segments: int,
) -> list[dict[str, object]]:
    signal = np.where(eigenvalues > mp_upper)[0]
    bulk = np.where((eigenvalues <= mp_upper) & (eigenvalues >= mp_lower))[0]
    regimes: list[dict[str, object]] = []
    if "signal_adaptive" in requested_blocks and len(signal):
        signal_eigenvalues = eigenvalues[signal]
        signal_blocks, _ = adaptive_spectral_blocks(
            signal_eigenvalues,
            min_segment_length=min_segment_length,
            max_segments=max_segments,
        )
        for block in signal_blocks:
            start = int(block["block_start"]) - 1
            end = int(block["block_end"])
            indices = signal[start:end]
            regimes.append(
                {
                    "run_id": (
                        f"kak_signal_adaptive_modes_{indices[0] + 1:03d}_"
                        f"{indices[-1] + 1:03d}"
                    ),
                    "block_type": f"above_mp_signal__{block['block_type']}",
                    "indices": indices,
                }
            )
    if "signal" in requested_blocks and len(signal):
        regimes.append(
            {
                "run_id": f"kak_signal_above_mp_001_{len(signal):03d}",
                "block_type": "above_mp_signal",
                "indices": signal,
            }
        )
    if "bulk" in requested_blocks and len(bulk):
        regimes.append(
            {
                "run_id": f"kak_bulk_inside_mp_{bulk[0] + 1:03d}_{bulk[-1] + 1:03d}",
                "block_type": "inside_mp_bulk",
                "indices": bulk,
            }
        )
    if "full" in requested_blocks:
        regimes.append(
            {
                "run_id": f"kak_full_nonzero_001_{len(eigenvalues):03d}",
                "block_type": "full_nonzero_rank",
                "indices": np.arange(len(eigenvalues)),
            }
        )
    return regimes


def run_regime(
    *,
    regime: dict[str, object],
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    data: pd.DataFrame,
    output_dir: Path,
    alpha_local: float,
    sibling_alpha: float,
) -> dict[str, object]:
    indices = np.asarray(regime["indices"], dtype=int)
    run_id = str(regime["run_id"])
    block_type = str(regime["block_type"])
    start = time.time()

    coords = eigenvectors[:, indices] * np.sqrt(np.maximum(eigenvalues[indices], 0.0))
    coords = np.nan_to_num(coords)
    distances = pdist(coords, metric="euclidean")
    if not np.isfinite(distances).all() or np.allclose(distances, 0.0):
        raise ValueError(f"{run_id}: degenerate KAK MP distances")

    linkage_matrix = linkage(distances, method="average")
    tree = PosetTree.from_linkage(linkage_matrix, leaf_names=data.index.tolist())

    decomposition = tree.decompose(
        leaf_data=data,
        alpha_local=alpha_local,
        sibling_alpha=sibling_alpha,
    )
    assignments = build_sample_cluster_assignments(decomposition).loc[data.index]
    assignments.index.name = "gene"
    assignments.to_csv(output_dir / "assignments" / f"{run_id}_cluster_assignments.csv")

    sizes = cluster_sizes(assignments)
    sizes.to_csv(output_dir / "cluster_sizes" / f"{run_id}_cluster_sizes.csv", index=False)

    labels = assignments["cluster_id"].astype(int)
    top_terms = top_terms_for_assignments(data, labels)
    top_terms.to_csv(output_dir / "top_terms" / f"{run_id}_top_terms.csv", index=False)

    silhouette = math.nan
    if 1 < labels.nunique() < len(labels):
        silhouette = float(silhouette_score(coords, labels, metric="euclidean"))

    return {
        "run_id": run_id,
        "block_type": block_type,
        "component_start": int(indices[0] + 1),
        "component_end": int(indices[-1] + 1),
        "subspace_dimensions": int(len(indices)),
        "first_eigenvalue": float(eigenvalues[indices[0]]),
        "last_eigenvalue": float(eigenvalues[indices[-1]]),
        "energy_fraction": float(eigenvalues[indices].sum() / eigenvalues.sum()),
        "n_clusters": int(labels.nunique()),
        "largest_cluster_size": int(sizes.iloc[0]["n_genes"]),
        "largest_cluster_fraction": float(sizes.iloc[0]["n_genes"] / len(data)),
        "silhouette_in_kak_subspace": silhouette,
        "runtime_seconds": float(time.time() - start),
    }


def main() -> None:
    args = parse_args()
    out = args.output_dir
    (out / "assignments").mkdir(parents=True, exist_ok=True)
    (out / "cluster_sizes").mkdir(exist_ok=True)
    (out / "top_terms").mkdir(exist_ok=True)

    data = load_matrix(args.input)
    standardized, n_active_features = standardize_active_features(data)
    n_genes = int(standardized.shape[0])
    mp_lower, mp_upper = mp_bounds(n_genes, n_active_features)

    eigenvalues, eigenvectors = kak_mp_gene_scores(standardized)
    regimes = build_regimes(
        eigenvalues,
        mp_lower=mp_lower,
        mp_upper=mp_upper,
        requested_blocks=set(args.blocks),
        min_segment_length=args.min_segment_length,
        max_segments=args.max_segments,
    )

    spectrum = pd.DataFrame(
        {
            "component": np.arange(1, len(eigenvalues) + 1),
            "correlation_eigenvalue": eigenvalues,
            "mp_lower": mp_lower,
            "mp_upper": mp_upper,
            "mp_regime": np.where(eigenvalues > mp_upper, "above_mp_signal", "inside_mp_bulk"),
        }
    )
    spectrum.to_csv(out / "kak_feature_correlation_spectrum.csv", index=False)

    rows = []
    for regime in regimes:
        print(
            f"Running {regime['run_id']} with {len(regime['indices'])} dimensions "
            "through TreeDecomposition gates...",
            flush=True,
        )
        rows.append(
            run_regime(
                regime=regime,
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
                data=data,
                output_dir=out,
                alpha_local=args.alpha_local,
                sibling_alpha=args.sibling_alpha,
            )
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "kak_mp_tree_method_summary.csv", index=False)

    readme = [
        "# KAK MP Tree Method Test",
        "",
        f"Input: `{args.input}`",
        f"Genes: `{data.shape[0]}`",
        f"Active GO features after variance filter: `{n_active_features}`",
        "",
        "Method: KAK-inspired Marchenko-Pastur regimes generate tree topology; "
        "the existing `TreeDecomposition` gates make the split decisions.",
        "",
        f"Split thresholds: `EDGE_ALPHA={args.alpha_local}`, "
        f"`SIBLING_ALPHA={args.sibling_alpha}`",
        f"MP bounds: lower `{mp_lower:.6g}`, upper `{mp_upper:.6g}`.",
        "",
        "Because `features > genes`, the MP lower bound is zero in this orientation; "
        "the default test therefore uses the above-MP signal regime, then splits "
        "that coupled regime by adaptive log-eigenvalue decay before applying our gates.",
        "",
        "## Summary",
        "",
        summary.to_string(index=False) if not summary.empty else "No completed regimes.",
        "",
    ]
    (out / "README.md").write_text("\n".join(readme))
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
