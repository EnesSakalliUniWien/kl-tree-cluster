"""Compute statistical gate annotations for tree decomposition.

This module provides :func:`compute_gate_annotations`, which runs Gate 2
(child-parent divergence) and Gate 3 (sibling divergence) annotation on a
results DataFrame.  The function is idempotent — if the required columns
already exist, the DataFrame is returned unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from .. import config
from .statistics import annotate_child_parent_divergence, annotate_sibling_divergence

if TYPE_CHECKING:
    from ..tree.poset_tree import PosetTree


def compute_gate_annotations(
    tree: "PosetTree",
    results_df: pd.DataFrame,
    *,
    alpha_local: float = config.ALPHA_LOCAL,
    sibling_alpha: float = config.SIBLING_ALPHA,
    leaf_data: pd.DataFrame | None = None,
    spectral_method: str | None = None,
) -> pd.DataFrame:
    """Compute Gate 2 and Gate 3 statistical annotations on *results_df*.

    If the required gate columns (``Child_Parent_Divergence_Significant``,
    ``Sibling_BH_Different``, ``Sibling_Divergence_Skipped``) already exist,
    the DataFrame is returned unchanged.  Otherwise the annotation pipeline
    is executed using the configured alpha levels and sibling test method.

    Parameters
    ----------
    tree
        Directed hierarchy (typically a :class:`~tree.poset_tree.PosetTree`).
    results_df
        DataFrame indexed by node id.  May already contain the annotation
        columns (pre-annotated path) or only base KL/distribution columns
        (raw path).
    alpha_local
        Significance level for the child-parent divergence gate (Gate 2).
    sibling_alpha
        Significance level for the sibling divergence gate (Gate 3).
    leaf_data
        Raw binary data matrix (samples × features).  Required for per-node
        spectral dimension estimation.
    spectral_method
        Per-node projection dimension estimator.  See ``config.SPECTRAL_METHOD``.

    Returns
    -------
    pd.DataFrame
        The annotated DataFrame, ready for gate evaluation.
    """
    required = {
        "Child_Parent_Divergence_Significant",
        "Sibling_BH_Different",
        "Sibling_Divergence_Skipped",
    }
    if required.issubset(results_df.columns):
        return results_df

    # -- Gate 2: child-parent divergence --
    results_df = annotate_child_parent_divergence(
        tree,
        results_df,
        significance_level_alpha=alpha_local,
        leaf_data=leaf_data,
        spectral_method=spectral_method,
    )

    # NOTE: spectral dimensions and PCA projections are NOT passed to sibling tests.
    # The edge test benefits from spectral dims because child ⊂ parent (subset structure),
    # but the sibling z-vector represents the DIFFERENCE between children — spectral_k
    # (effective rank) is much smaller than JL dimension near leaves, giving the sibling
    # χ² test fewer degrees of freedom and less statistical power. Sibling tests keep
    # the JL-based projection dimension which provides better power for detecting
    # differences between siblings.

    # -- Gate 3: sibling divergence (method selected via config) --
    if config.SIBLING_TEST_METHOD == "cousin_ftest":
        from .statistics.sibling_divergence import annotate_sibling_divergence_cousin

        results_df = annotate_sibling_divergence_cousin(
            tree,
            results_df,
            significance_level_alpha=sibling_alpha,
        )
    elif config.SIBLING_TEST_METHOD == "cousin_adjusted_wald":
        from .statistics.sibling_divergence import annotate_sibling_divergence_adjusted

        results_df = annotate_sibling_divergence_adjusted(
            tree,
            results_df,
            significance_level_alpha=sibling_alpha,
        )
    elif config.SIBLING_TEST_METHOD == "cousin_tree_guided":
        from .statistics.sibling_divergence import annotate_sibling_divergence_tree_guided

        results_df = annotate_sibling_divergence_tree_guided(
            tree,
            results_df,
            significance_level_alpha=sibling_alpha,
        )
    elif config.SIBLING_TEST_METHOD == "cousin_weighted_wald":
        from .statistics.sibling_divergence import annotate_sibling_divergence_weighted

        results_df = annotate_sibling_divergence_weighted(
            tree,
            results_df,
            significance_level_alpha=sibling_alpha,
        )
    else:
        results_df = annotate_sibling_divergence(
            tree,
            results_df,
            significance_level_alpha=sibling_alpha,
        )

    return results_df
