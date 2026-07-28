from __future__ import annotations

import numpy as np
import pandas as pd
from benchmarks.shared.runners.tbs_diffusion_runner import (
    GRAPHTOOLS_ADAPTIVE_NEIGHBOR_PROFILE_CONNECTIVITY_MINIMUM,
    GRAPHTOOLS_ADAPTIVE_NEIGHBOR_PROFILE_FRAGMENTATION_GUARD,
    _require_adaptive_diffusion_family_contract,
    _resolve_graphtools_neighbor_search_k,
)
from tree_break_selection.space_separation import adaptive_diffusion_geometry
from tree_break_selection.tree.feature_space import (
    continuous_feature_space_from_columns,
    infer_feature_space_from_columns,
)


def test_build_adaptive_diffusion_geometry_returns_finite_condensed_matrix():
    rng = np.random.default_rng(42)
    data = pd.DataFrame(
        rng.integers(0, 2, size=(12, 18)),
        index=[f"s{i}" for i in range(12)],
        columns=[f"f{j}" for j in range(18)],
    )

    geometry = adaptive_diffusion_geometry(
        data,
        k_neighbors=10,
        diffusion_time=2,
        n_components=5,
        metric="hamming",
        bandwidth_type="-1/(d+2)",
        epsilon="median",
    )

    expected_size = data.shape[0] * (data.shape[0] - 1) // 2
    assert geometry.distance_condensed.shape == (expected_size,)
    assert geometry.coordinates.shape == (12, 5)
    assert np.isfinite(geometry.distance_condensed).all()
    assert geometry.metadata["backend"] == "pydiffmap"
    assert 2 <= geometry.metadata["neighbor_search_k"] < data.shape[0]
    assert geometry.metadata["epsilon"] > 0


def test_adaptive_diffusion_rejects_hamming_for_continuous_feature_space() -> None:
    data = pd.DataFrame(
        [[0.0, 0.1], [1.0, 1.1], [2.0, 2.1]],
        columns=["x", "y"],
    )

    with np.testing.assert_raises_regex(ValueError, "requires metric='euclidean'"):
        _require_adaptive_diffusion_family_contract(
            data,
            continuous_feature_space_from_columns(data.columns),
            metric="hamming",
        )


def test_adaptive_diffusion_accepts_hamming_for_categorical_blocks() -> None:
    data = pd.DataFrame(
        [[1, 0, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1]],
        columns=["F0_c0", "F0_c1", "F1_c0", "F1_c1"],
    )

    family = _require_adaptive_diffusion_family_contract(
        data,
        infer_feature_space_from_columns(data.columns),
        metric="hamming",
    )

    assert family == "categorical"


def test_graphtools_adaptive_neighbor_resolver_selects_connected_candidate():
    X = np.array([[0.0], [0.1], [0.2], [10.0], [10.1], [10.2]])

    selected_k, metadata = _resolve_graphtools_neighbor_search_k(
        X,
        k_neighbors=1,
        metric="euclidean",
        adaptive_neighbor_profile=GRAPHTOOLS_ADAPTIVE_NEIGHBOR_PROFILE_CONNECTIVITY_MINIMUM,
        adaptive_neighbor_grid=(1, 3),
    )

    assert selected_k == 3
    assert metadata["adaptive_neighbor_status"] == "connected"
    assert metadata["adaptive_neighbor_component_counts"] == {"2": 2, "3": 1}
    assert metadata["adaptive_neighbor_selected_k"] == 3


def test_graphtools_adaptive_neighbor_resolver_is_duplicate_aware():
    X = np.array([[0.0], [0.0], [0.0], [10.0]])

    selected_k, metadata = _resolve_graphtools_neighbor_search_k(
        X,
        k_neighbors=1,
        metric="euclidean",
        adaptive_neighbor_profile=GRAPHTOOLS_ADAPTIVE_NEIGHBOR_PROFILE_CONNECTIVITY_MINIMUM,
        adaptive_neighbor_grid=(1,),
    )

    assert selected_k == 3
    assert metadata["adaptive_neighbor_max_duplicate_count"] == 3
    assert metadata["adaptive_neighbor_duplicate_aware_min_k"] == 3
    assert metadata["adaptive_neighbor_component_counts"] == {"3": 1}


def test_graphtools_fragmentation_guard_preserves_stable_components():
    X = np.array([[0.0], [0.1], [0.2], [10.0], [10.1], [10.2]])

    selected_k, metadata = _resolve_graphtools_neighbor_search_k(
        X,
        k_neighbors=2,
        metric="euclidean",
        adaptive_neighbor_profile=GRAPHTOOLS_ADAPTIVE_NEIGHBOR_PROFILE_FRAGMENTATION_GUARD,
        adaptive_neighbor_grid=(2, 3),
    )

    assert selected_k == 2
    assert metadata["adaptive_neighbor_status"] == "stable_components"
    assert metadata["adaptive_neighbor_component_counts"] == {"2": 2}
    assert metadata["adaptive_neighbor_component_min_sizes"] == {"2": 3}
