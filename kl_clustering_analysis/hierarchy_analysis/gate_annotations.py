"""Compute statistical gate annotations for tree decomposition.

This module provides :func:`compute_gate_annotations`, which runs Gate 2
(child-parent divergence) and Gate 3 (sibling divergence) annotation on a
results DataFrame.  Annotations are always recomputed from scratch so that
config changes (alpha levels, sibling test method, etc.) take effect
immediately.

Extracted from ``TreeDecomposition._prepare_annotations`` (roadmap 3.2) so
that the annotation logic is a standalone, stateless function that can be
tested and reused independently of the decomposer class.
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
    min_k: int | None = None,
) -> pd.DataFrame:
    """Compute Gate 2 and Gate 3 statistical annotations on *results_df*.

    Annotations are always recomputed from scratch using the current alpha
    levels and sibling test method (``config.SIBLING_TEST_METHOD``).  Any
    pre-existing gate columns are overwritten.

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
    min_k
        Minimum projection dimension (floor).  When ``None``, uses
        ``config.PROJECTION_MIN_K`` (resolved to int).  Pass an explicit int
        to override, e.g. from :func:`resolve_min_k` with ``"auto"``.

    Returns
    -------
    pd.DataFrame
        The annotated DataFrame, ready for gate evaluation.
    """
    # Always recompute annotations from scratch.  Previous versions had an
    # early-return guard that skipped recomputation when the gate columns
    # already existed — this caused stale results when alpha levels,
    # SIBLING_TEST_METHOD, or other config values changed between calls.
    # The annotation functions initialise their output columns from scratch,
    # so rerunning them on a DataFrame that already has those columns is safe.

    # -- Gate 2: child-parent divergence --
    results_df = annotate_child_parent_divergence(
        tree,
        results_df,
        significance_level_alpha=alpha_local,
        leaf_data=leaf_data,
        spectral_method=spectral_method,
        min_k=min_k,
    )

    # -- Sibling spectral dimensions: use JL-based dimension --
    # The sibling z-vector z = (θ_L − θ_R)/√Var has d components with
    # signal spread across many directions.  Power scales as √k, so the
    # projection dimension should be as large as the data supports.
    # Both within-cluster erank and parent overall erank give low k
    # (limited by n ≪ d), destroying power.  The JL-based dimension
    # k ≈ 8·ln(n)/ε² with information cap gives adequate power while
    # the calibration model (weighted Wald) corrects any inflation.
    #
    # NOTE: spectral dimensions and PCA projections are NOT passed to sibling
    # tests.  The edge test benefits from spectral dims because child ⊂ parent
    # (subset structure), but the sibling z-vector represents the DIFFERENCE
    # between children — spectral_k (effective rank) is much smaller than JL
    # dimension near leaves, giving the sibling χ² test fewer degrees of
    # freedom and less statistical power.
    _sibling_spectral_dims = None  # JL fallback in sibling_divergence_test

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
            spectral_dims=_sibling_spectral_dims,
            pca_projections=None,  # No PCA for siblings — use random projection
            pca_eigenvalues=None,
        )
    else:
        results_df = annotate_sibling_divergence(
            tree,
            results_df,
            significance_level_alpha=sibling_alpha,
        )

    # -- Post-hoc Edge Calibration (uses sibling p-values as weights) --
    if config.EDGE_CALIBRATION:
        from .statistics.kl_tests.edge_calibration import calibrate_edges_from_sibling_neighborhood

        results_df = calibrate_edges_from_sibling_neighborhood(
            tree,
            results_df,
            alpha=alpha_local,
        )

    return results_df
