"""Edge-gate annotation wrapper (Gate 2)."""

from __future__ import annotations

import pandas as pd

from kl_clustering_analysis import config

from ...statistics.kl_tests.edge_significance import annotate_child_parent_divergence


def annotate_edge_gate(
    tree,
    results_df: pd.DataFrame,
    *,
    significance_level_alpha: float = config.ALPHA_LOCAL,
    leaf_data: pd.DataFrame | None = None,
    spectral_method: str | None = None,
    min_k: int | None = None,
    fdr_method: str = "tree_bh",
) -> pd.DataFrame:
    """Annotate child-parent divergence for decomposition gate logic."""
    return annotate_child_parent_divergence(
        tree,
        results_df,
        significance_level_alpha=significance_level_alpha,
        fdr_method=fdr_method,
        leaf_data=leaf_data,
        spectral_method=spectral_method,
        min_k=min_k,
    )


__all__ = ["annotate_edge_gate"]

