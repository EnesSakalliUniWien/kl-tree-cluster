"""Top-level gate annotation orchestration wrapper."""

from __future__ import annotations

import pandas as pd

from kl_clustering_analysis import config

from ..core.contracts import GateAnnotationBundle
from ...gate_annotations import compute_gate_annotations


def run_gate_annotation_pipeline(
    tree,
    results_df: pd.DataFrame,
    *,
    alpha_local: float = config.ALPHA_LOCAL,
    sibling_alpha: float = config.SIBLING_ALPHA,
    leaf_data: pd.DataFrame | None = None,
    spectral_method: str | None = None,
    min_k: int | None = None,
) -> GateAnnotationBundle:
    """Run the current gate-annotation flow and return a typed bundle."""
    annotated_df = compute_gate_annotations(
        tree,
        results_df,
        alpha_local=alpha_local,
        sibling_alpha=sibling_alpha,
        leaf_data=leaf_data,
        spectral_method=spectral_method,
        min_k=min_k,
    )
    return GateAnnotationBundle(annotated_df=annotated_df)


__all__ = ["run_gate_annotation_pipeline"]

