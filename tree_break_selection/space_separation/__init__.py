"""Reusable space-separation methods used across applications and benchmarks."""

from .adaptive_cosine import (
    AdaptiveCosineSpace,
    SpectralBlock,
    adaptive_spectral_blocks,
    coordinates_for_block,
    cosine_eigendecomposition,
    separate_adaptive_cosine_space,
    weight_feature_matrix,
)
from .diffusion import (
    DiffusionGeometry,
    adaptive_diffusion_geometry,
    block_adaptive_diffusion_geometry,
    block_diffusion_geometry,
    compute_diffusion_coordinates,
    hamming_knn_diffusion_geometry,
    resolve_adaptive_epsilon,
    resolve_adaptive_epsilon_from_scaled_sq,
    resolve_neighbor_search_k,
)
from .invariant_equivariant import (
    InvariantEquivariantSpace,
    decompose_invariant_equivariant_space,
    standardize_columns,
)

__all__ = [
    "AdaptiveCosineSpace",
    "DiffusionGeometry",
    "InvariantEquivariantSpace",
    "SpectralBlock",
    "adaptive_spectral_blocks",
    "adaptive_diffusion_geometry",
    "block_adaptive_diffusion_geometry",
    "block_diffusion_geometry",
    "compute_diffusion_coordinates",
    "coordinates_for_block",
    "cosine_eigendecomposition",
    "decompose_invariant_equivariant_space",
    "hamming_knn_diffusion_geometry",
    "separate_adaptive_cosine_space",
    "resolve_adaptive_epsilon",
    "resolve_adaptive_epsilon_from_scaled_sq",
    "resolve_neighbor_search_k",
    "standardize_columns",
    "weight_feature_matrix",
]
