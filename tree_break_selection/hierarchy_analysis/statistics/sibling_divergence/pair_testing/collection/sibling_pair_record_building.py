"""Construction helpers for sibling pair records."""

from __future__ import annotations

import numpy as np

from ..types.sibling_pair_record import SiblingPairRecord


def build_sibling_pair_record(
    *,
    parent_node_id: object,
    left_child_id: object,
    right_child_id: object,
    test_statistic: float,
    reference_scale: float,
    degrees_of_freedom: float,
    p_value: float,
    branch_length_sum: float,
    parent_sample_size: int,
    is_null_like: bool,
    is_edge_blocked: bool,
    sibling_null_weight: float,
    sibling_projection_dimension: float,
    parent_spectral_eigenvalue_count: float = 0.0,
    parent_positive_eigenvalue_count: float = 0.0,
    parent_spectral_rank: float = 0.0,
    parent_eigenvalue_sum: float = 0.0,
    parent_top_eigenvalue: float = 0.0,
    parent_top_eigenvalue_share: float = 0.0,
    parent_spectral_entropy: float = 0.0,
    parent_effective_rank: float = 0.0,
    parent_retained_eigenvalue_sum: float = 0.0,
    parent_retained_eigenvalue_share: float = 0.0,
    parent_top_spectral_gap: float = 0.0,
    parent_eigengap_at_projection_dimension: float = 0.0,
    parent_spectral_gap_at_projection_dimension: float = 0.0,
    parent_spectral_pseudodeterminant: float = 0.0,
    parent_spectral_log_pseudodeterminant: float = 0.0,
    parent_spectral_geometric_mean: float = 0.0,
    feature_family: str = "bernoulli",
) -> SiblingPairRecord:
    """Construct a sibling-pair record from resolved statistical inputs."""
    if not np.isfinite(test_statistic):
        raise ValueError(
            f"Sibling pair record requires a finite test statistic; parent={parent_node_id!r}."
        )
    if not np.isfinite(reference_scale) or reference_scale <= 0:
        raise ValueError(
            "Sibling pair record requires a finite positive reference_scale; "
            f"parent={parent_node_id!r}."
        )
    if not np.isfinite(degrees_of_freedom) or degrees_of_freedom < 0:
        raise ValueError(
            "Sibling pair record requires finite non-negative degrees of freedom; "
            f"parent={parent_node_id!r}."
        )
    if not np.isfinite(p_value) or p_value < 0.0 or p_value > 1.0:
        raise ValueError(
            f"Sibling pair record requires a finite p-value in [0, 1]; parent={parent_node_id!r}."
        )
    if not np.isfinite(sibling_null_weight) or not 0.0 <= sibling_null_weight <= 1.0:
        raise ValueError(
            "Sibling pair record requires a finite sibling_null_weight in [0, 1]; "
            f"parent={parent_node_id!r}."
        )
    if not np.isfinite(sibling_projection_dimension) or sibling_projection_dimension < 0.0:
        raise ValueError(
            "Sibling pair record requires a finite non-negative sibling_projection_dimension; "
            f"parent={parent_node_id!r}."
        )
    spectral_values = {
        "parent_spectral_eigenvalue_count": parent_spectral_eigenvalue_count,
        "parent_positive_eigenvalue_count": parent_positive_eigenvalue_count,
        "parent_spectral_rank": parent_spectral_rank,
        "parent_eigenvalue_sum": parent_eigenvalue_sum,
        "parent_top_eigenvalue": parent_top_eigenvalue,
        "parent_top_eigenvalue_share": parent_top_eigenvalue_share,
        "parent_spectral_entropy": parent_spectral_entropy,
        "parent_effective_rank": parent_effective_rank,
        "parent_retained_eigenvalue_sum": parent_retained_eigenvalue_sum,
        "parent_retained_eigenvalue_share": parent_retained_eigenvalue_share,
        "parent_top_spectral_gap": parent_top_spectral_gap,
        "parent_eigengap_at_projection_dimension": parent_eigengap_at_projection_dimension,
        "parent_spectral_gap_at_projection_dimension": parent_spectral_gap_at_projection_dimension,
        "parent_spectral_pseudodeterminant": parent_spectral_pseudodeterminant,
        "parent_spectral_geometric_mean": parent_spectral_geometric_mean,
    }
    for name, value in spectral_values.items():
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                f"Sibling pair record requires finite non-negative {name}; "
                f"parent={parent_node_id!r}, value={value!r}."
            )
    if not np.isfinite(parent_spectral_log_pseudodeterminant):
        raise ValueError(
            "Sibling pair record requires finite parent_spectral_log_pseudodeterminant; "
            f"parent={parent_node_id!r}, value={parent_spectral_log_pseudodeterminant!r}."
        )
    if feature_family not in {"bernoulli", "categorical", "continuous", "mixed"}:
        raise ValueError(
            "Sibling pair record requires feature_family to be 'bernoulli', "
            f"'categorical', 'continuous', or 'mixed'; got {feature_family!r} for "
            f"parent={parent_node_id!r}."
        )
    return SiblingPairRecord(
        parent=parent_node_id,
        left=left_child_id,
        right=right_child_id,
        stat=test_statistic,
        reference_scale=float(reference_scale),
        degrees_of_freedom=float(degrees_of_freedom),
        p_value=p_value,
        branch_length_sum=branch_length_sum,
        n_parent=parent_sample_size,
        is_null_like=is_null_like,
        is_edge_blocked=is_edge_blocked,
        sibling_null_weight=sibling_null_weight,
        sibling_projection_dimension=sibling_projection_dimension,
        parent_spectral_eigenvalue_count=float(parent_spectral_eigenvalue_count),
        parent_positive_eigenvalue_count=float(parent_positive_eigenvalue_count),
        parent_spectral_rank=float(parent_spectral_rank),
        parent_eigenvalue_sum=float(parent_eigenvalue_sum),
        parent_top_eigenvalue=float(parent_top_eigenvalue),
        parent_top_eigenvalue_share=float(parent_top_eigenvalue_share),
        parent_spectral_entropy=float(parent_spectral_entropy),
        parent_effective_rank=float(parent_effective_rank),
        parent_retained_eigenvalue_sum=float(parent_retained_eigenvalue_sum),
        parent_retained_eigenvalue_share=float(parent_retained_eigenvalue_share),
        parent_top_spectral_gap=float(parent_top_spectral_gap),
        parent_eigengap_at_projection_dimension=float(parent_eigengap_at_projection_dimension),
        parent_spectral_gap_at_projection_dimension=float(
            parent_spectral_gap_at_projection_dimension
        ),
        parent_spectral_pseudodeterminant=float(parent_spectral_pseudodeterminant),
        parent_spectral_log_pseudodeterminant=float(
            parent_spectral_log_pseudodeterminant
        ),
        parent_spectral_geometric_mean=float(parent_spectral_geometric_mean),
        feature_family=feature_family,
    )


__all__ = ["build_sibling_pair_record"]
