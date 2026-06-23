"""Phylogenetic continuous null generators."""

from __future__ import annotations

import numpy as np

from benchmarks.shared.generators.case_data_contracts import (
    CaseDataResult,
    case_metadata,
    continuous_dataframe_and_metadata,
    require_case_value,
)


def generate_phylogenetic_brownian_continuous(
    test_case: dict,
    seed: int | None,
) -> CaseDataResult:
    """Generate taxon-correlated continuous features under a Brownian-like null."""
    n_taxa = int(require_case_value(test_case, "n_taxa", "Phylogenetic Brownian"))
    samples_per_taxon = int(
        require_case_value(test_case, "samples_per_taxon", "Phylogenetic Brownian")
    )
    n_features = int(require_case_value(test_case, "n_features", "Phylogenetic Brownian"))
    sigma2 = float(test_case.get("sigma2", 1.0))
    rng = np.random.default_rng(seed)

    n_samples = n_taxa * samples_per_taxon
    labels = np.repeat(np.arange(n_taxa, dtype=int), samples_per_taxon)
    root_effect = rng.normal(scale=np.sqrt(sigma2), size=n_features)
    taxon_effects = np.empty((n_taxa, n_features), dtype=float)
    for taxon in range(n_taxa):
        depth_scale = 1.0 + (taxon.bit_count() / max(n_taxa.bit_length(), 1))
        taxon_effects[taxon] = root_effect + rng.normal(
            scale=np.sqrt(sigma2 * depth_scale),
            size=n_features,
        )
    matrix = taxon_effects[labels] + rng.normal(
        scale=np.sqrt(0.15 * sigma2),
        size=(n_samples, n_features),
    )

    data_df, feature_space, distance_condensed = continuous_dataframe_and_metadata(
        matrix,
        [f"T{labels[j]}_S{j}" for j in range(n_samples)],
        [f"F{j}" for j in range(n_features)],
    )
    metadata = case_metadata(
        test_case=test_case,
        n_samples=n_samples,
        n_features=n_features,
        n_clusters=n_taxa,
        noise=sigma2,
        generator="phylogenetic_brownian_continuous",
        source_family="phylogenetic_brownian",
        feature_representation="continuous",
        requires_precomputed_tbs_distance=True,
        precomputed_distance_condensed=distance_condensed,
        distance_metric="mahalanobis_time",
        extra={
            "feature_space": feature_space,
            "n_taxa": n_taxa,
            "samples_per_taxon": samples_per_taxon,
            "branch_length_profile": str(test_case.get("branch_length_profile", "balanced")),
            "sigma2": sigma2,
        },
    )
    return data_df, labels, matrix.astype(float, copy=False), metadata
