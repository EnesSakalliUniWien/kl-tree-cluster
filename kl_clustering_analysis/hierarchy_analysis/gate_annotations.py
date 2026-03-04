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
from .decomposition.gates.orchestrator import run_gate_annotation_pipeline

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

    Compatibility entrypoint that delegates to the decomposition orchestrator.
    The public signature is intentionally stable for external callers.

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
    bundle = run_gate_annotation_pipeline(
        tree,
        results_df,
        alpha_local=alpha_local,
        sibling_alpha=sibling_alpha,
        leaf_data=leaf_data,
        spectral_method=spectral_method,
        min_k=min_k,
        sibling_method=config.SIBLING_TEST_METHOD,
        # Preserve legacy Gate 2 correction default.
        fdr_method="tree_bh",
        # Preserve legacy sibling behavior: JL fallback, no PCA siblings.
        sibling_spectral_dims=None,
        sibling_pca_projections=None,
        sibling_pca_eigenvalues=None,
        # Preserve legacy config-controlled edge calibration behavior.
        edge_calibration=None,
    )
    return bundle.annotated_df
