#!/usr/bin/env python3
"""Dump sibling-pair calibration summaries for a few sentinel cases."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LAB_ROOT = REPO_ROOT / "debug_scripts" / "enhancement_lab"
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from lab_helpers import build_tree_and_data, run_decomposition  # noqa: E402


from kl_clustering_analysis import config  # noqa: E402
from kl_clustering_analysis.hierarchy_analysis.statistics.branch_length_utils import (  # noqa: E402
    compute_mean_branch_length,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.sibling_pair_collection import (  # noqa: E402
    collect_sibling_pair_records,
)
from debug_scripts._shared.sibling_child_pca import derive_sibling_child_pca_projections  # noqa: E402
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_config import (  # noqa: E402
    derive_sibling_pca_projections,
    derive_sibling_spectral_dims,
)

DEFAULT_OUTPUT = REPO_ROOT / "debug_scripts" / "diagnostics" / "results" / "calibration_dump.txt"
DEFAULT_CASES = ["gauss_moderate_3c", "binary_perfect_8c", "gauss_null_small"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as handle:
        for case_name in args.cases:
            tree, data_df, _, test_case = build_tree_and_data(case_name)
            decomp = run_decomposition(tree, data_df)
            annotations_df = tree.annotations_df
            spectral_dims = derive_sibling_spectral_dims(tree, annotations_df)
            pca_projections, pca_eigenvalues = derive_sibling_pca_projections(
                annotations_df,
                spectral_dims,
            )
            child_pca_projections = derive_sibling_child_pca_projections(
                tree,
                annotations_df,
                spectral_dims,
            )
            mean_branch_length = (
                compute_mean_branch_length(tree) if config.FELSENSTEIN_SCALING else None
            )
            records, _ = collect_sibling_pair_records(
                tree,
                annotations_df,
                mean_branch_length,
                spectral_dims=spectral_dims,
                pca_projections=pca_projections,
                pca_eigenvalues=pca_eigenvalues,
                whitening=config.SIBLING_WHITENING,
            )

            valid = [
                (
                    record.parent,
                    record.stat / record.degrees_of_freedom,
                    record.sibling_null_prior_from_edge_pvalue,
                    record.is_null_like,
                    record.n_parent,
                )
                for record in records
                if np.isfinite(record.stat) and record.degrees_of_freedom > 0
            ]
            valid.sort(key=lambda row: -row[2])
            ratios = np.array([row[1] for row in valid])
            weights = np.array([row[2] for row in valid])
            c_hat = float(np.average(ratios, weights=weights)) if weights.sum() > 0 else 1.0
            null_ratios = np.array([row[1] for row in valid if row[3]])
            focal_ratios = np.array([row[1] for row in valid if not row[3]])

            handle.write(f"\n{'=' * 60}\n")
            handle.write(
                f"CASE: {case_name} (true K={test_case.get('n_clusters')}, found K={decomp['num_clusters']})\n"
            )
            handle.write(f"{'=' * 60}\n")
            handle.write(
                f"Valid pairs: {len(valid)}, Null-like: {len(null_ratios)}, Focal: {len(focal_ratios)}\n"
            )
            if len(null_ratios) > 0:
                handle.write(
                    f"Null T/k: mean={np.mean(null_ratios):.3f}, med={np.median(null_ratios):.3f}, "
                    f"max={np.max(null_ratios):.3f}, min={np.min(null_ratios):.3f}\n"
                )
            if len(focal_ratios) > 0:
                handle.write(
                    f"Focal T/k: mean={np.mean(focal_ratios):.3f}, med={np.median(focal_ratios):.3f}, "
                    f"max={np.max(focal_ratios):.3f}\n"
                )
            handle.write(f"c_hat (weighted mean): {c_hat:.3f}\n")
            handle.write(f"Max ratio: {np.max(ratios):.3f}\n")
            handle.write(f"Weight sum: {weights.sum():.4f}, Mean weight: {np.mean(weights):.4f}\n")

            handle.write("\nAll pairs sorted by sibling_null_prior (most null-like first):\n")
            handle.write(f"{'Parent':<8} {'T/k':>7} {'prior':>7} {'null':>5} {'nP':>4}\n")
            handle.write("-" * 38 + "\n")
            for parent, ratio, sibling_null_prior, is_null_like, n_parent in valid:
                handle.write(
                    f"{parent:<8} {ratio:>7.3f} {sibling_null_prior:>7.4f} {str(is_null_like):>5} {n_parent:>4}\n"
                )

            contributions = weights * ratios
            top_indices = np.argsort(-contributions)[:8]
            handle.write("\nTop 8 contributors (prior * T/k):\n")
            for index in top_indices:
                parent, ratio, sibling_null_prior, is_null_like, n_parent = valid[index]
                handle.write(
                    f"  {parent:<8} T/k={ratio:.3f} x prior={sibling_null_prior:.4f} = {sibling_null_prior * ratio:.4f} "
                    f"null={is_null_like} nP={n_parent}\n"
                )

    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
