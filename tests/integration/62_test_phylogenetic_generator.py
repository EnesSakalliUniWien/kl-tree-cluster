import numpy as np

from benchmarks.shared.generators.generate_phylogenetic import generate_phylogenetic_data


def _mean_pairwise_hamming(sequences: list[np.ndarray]) -> float:
    if len(sequences) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            total += float(np.mean(sequences[i] != sequences[j]))
            count += 1
    return total / count if count else 0.0


def test_generate_phylogenetic_data_shapes_and_ranges():
    sample_dict, cluster_assignments, distributions, metadata = generate_phylogenetic_data(
        n_taxa=6,
        n_features=128,
        n_categories=4,
        samples_per_taxon=5,
        mutation_rate=0.2,
        random_seed=42,
    )

    assert len(sample_dict) == 30
    assert set(sample_dict.keys()) == set(cluster_assignments.keys())
    assert len(set(cluster_assignments.values())) == 6

    matrix = np.stack([sample_dict[k] for k in sorted(sample_dict.keys())], axis=0)
    assert matrix.shape == (30, 128)
    assert np.issubdtype(matrix.dtype, np.integer)
    assert matrix.min() >= 0
    assert matrix.max() < 4

    assert distributions.shape == (30, 128, 4)
    assert np.allclose(distributions.sum(axis=2), 1.0)

    leaf_distributions = metadata["leaf_distributions"]
    assert isinstance(leaf_distributions, dict)
    assert len(leaf_distributions) == 6
    for dist in leaf_distributions.values():
        assert dist.shape == (128, 4)
        assert np.allclose(dist.sum(axis=1), 1.0)


def test_mutation_rate_increases_leaf_sequence_divergence():
    _, _, _, low_meta = generate_phylogenetic_data(
        n_taxa=10,
        n_features=1500,
        n_categories=4,
        samples_per_taxon=2,
        mutation_rate=0.05,
        random_seed=123,
    )
    _, _, _, high_meta = generate_phylogenetic_data(
        n_taxa=10,
        n_features=1500,
        n_categories=4,
        samples_per_taxon=2,
        mutation_rate=0.6,
        random_seed=123,
    )

    low_leaf_sequences = [
        np.asarray(low_meta["leaf_sequences"][k]) for k in sorted(low_meta["leaf_sequences"].keys())
    ]
    high_leaf_sequences = [
        np.asarray(high_meta["leaf_sequences"][k])
        for k in sorted(high_meta["leaf_sequences"].keys())
    ]

    low_div = _mean_pairwise_hamming(low_leaf_sequences)
    high_div = _mean_pairwise_hamming(high_leaf_sequences)

    assert high_div > low_div + 0.03


def test_leaf_sequences_not_collapsed_at_moderate_mutation():
    _, _, _, metadata = generate_phylogenetic_data(
        n_taxa=12,
        n_features=300,
        n_categories=4,
        samples_per_taxon=3,
        mutation_rate=0.2,
        random_seed=7,
    )

    leaf_sequences = metadata["leaf_sequences"]
    unique = {
        np.asarray(leaf_sequences[k], dtype=np.int16).tobytes()
        for k in sorted(leaf_sequences.keys())
    }

    assert len(unique) >= 10
