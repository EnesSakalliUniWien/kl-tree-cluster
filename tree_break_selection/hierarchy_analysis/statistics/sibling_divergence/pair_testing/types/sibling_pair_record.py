"""Sibling pair record dataclass for the pair_testing package."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SiblingPairRecord:
    """Raw per-parent sibling-test record used by calibration pipelines."""

    parent: object
    left: object
    right: object
    stat: float
    reference_scale: float
    """Scale in the projected quadratic reference law; unit for orthonormal PCA."""
    degrees_of_freedom: float
    p_value: float
    branch_length_sum: float
    n_parent: int
    is_null_like: bool
    is_edge_blocked: bool = False
    sibling_null_weight: float = 0.0
    """Weight that this sibling pair represents empirical-null structure."""
    sibling_projection_dimension: float = 0.0
    """Projection dimension used by the sibling test and inflation context."""
    parent_spectral_eigenvalue_count: float = 0.0
    """Number of finite parent eigenvalues available for the sibling test context."""
    parent_positive_eigenvalue_count: float = 0.0
    """Number of positive parent eigenvalues after finite/non-positive filtering."""
    parent_spectral_rank: float = 0.0
    """Rank of the positive parent spectrum used for spectral diagnostics."""
    parent_eigenvalue_sum: float = 0.0
    """Sum of positive parent eigenvalues in the available spectral context."""
    parent_top_eigenvalue: float = 0.0
    """Largest positive parent eigenvalue, or zero when no positive spectrum exists."""
    parent_top_eigenvalue_share: float = 0.0
    """Largest positive eigenvalue divided by the positive eigenvalue sum."""
    parent_spectral_entropy: float = 0.0
    """Entropy of positive normalized parent eigenvalues."""
    parent_effective_rank: float = 0.0
    """Exponential spectral entropy of positive normalized parent eigenvalues."""
    parent_retained_eigenvalue_sum: float = 0.0
    """Sum of eigenvalues retained by the sibling projection dimension."""
    parent_retained_eigenvalue_share: float = 0.0
    """Retained eigenvalue sum divided by the positive eigenvalue sum."""
    parent_top_spectral_gap: float = 0.0
    """Largest eigenvalue minus second-largest eigenvalue; zero when unavailable."""
    parent_eigengap_at_projection_dimension: float = 0.0
    """Eigenvalue ratio lambda_k / lambda_{k+1}; zero when unavailable."""
    parent_spectral_gap_at_projection_dimension: float = 0.0
    """Eigenvalue difference lambda_k - lambda_{k+1}; zero when unavailable."""
    parent_spectral_pseudodeterminant: float = 0.0
    """Product of positive parent eigenvalues; zero when no positive spectrum exists."""
    parent_spectral_log_pseudodeterminant: float = 0.0
    """Log product of positive parent eigenvalues; zero when no positive spectrum exists."""
    parent_spectral_geometric_mean: float = 0.0
    """Rank-root of the pseudodeterminant; zero when no positive spectrum exists."""
    feature_family: str = "bernoulli"
    """Feature-space family label for the sibling contrast covariance model."""


__all__ = ["SiblingPairRecord"]
