from __future__ import annotations

from kl_clustering_analysis.hierarchy_analysis.decomposition.backends.random_projection_backend import (
    derive_projection_seed_backend as derive_projection_seed,
)


def test_derive_projection_seed_is_deterministic_and_unique() -> None:
    seed_a_1 = derive_projection_seed(42, "edge:root->A")
    seed_a_2 = derive_projection_seed(42, "edge:root->A")
    seed_b = derive_projection_seed(42, "edge:root->B")

    assert seed_a_1 == seed_a_2
    assert seed_a_1 != seed_b
