from __future__ import annotations

import numpy as np
import pandas as pd
from benchmarks.shared.runners.kl_diffusion_runner import _build_adaptive_diffusion_distance


def test_build_adaptive_diffusion_distance_returns_finite_condensed_matrix():
    rng = np.random.default_rng(42)
    data = pd.DataFrame(
        rng.integers(0, 2, size=(12, 18)),
        index=[f"s{i}" for i in range(12)],
        columns=[f"f{j}" for j in range(18)],
    )

    condensed, metadata = _build_adaptive_diffusion_distance(
        data,
        k_neighbors=None,
        diffusion_time=2,
        n_components=5,
        metric="hamming",
        bandwidth_type="-1/(d+2)",
        epsilon="median",
        return_metadata=True,
    )

    expected_size = data.shape[0] * (data.shape[0] - 1) // 2
    assert condensed.shape == (expected_size,)
    assert np.isfinite(condensed).all()
    assert metadata["backend"] in {"pydiffmap", "local_scaling_fallback"}
    assert 2 <= metadata["neighbor_search_k"] < data.shape[0]
    assert metadata["epsilon"] > 0
