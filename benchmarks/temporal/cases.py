"""Temporal evolution test cases for benchmarking.

Simulates sequence divergence along a single lineage over time.
Adjacent time points are more similar than distant ones.
"""

TEMPORAL_DNA_FAST_CASES = [
    # Fast evolution (high mutation rate) - easy to distinguish time points
    {
        "name": "temporal_dna_8tp_fast",
        "generator": "temporal_evolution",
        "n_time_points": 8,
        "n_features": 200,
        "n_categories": 4,
        "samples_per_time": 20,
        "mutation_rate": 0.35,
        "shift_strength": [0.2, 0.5],
        "seed": 12000,
    },
    {
        "name": "temporal_dna_10tp_fast",
        "generator": "temporal_evolution",
        "n_time_points": 10,
        "n_features": 300,
        "n_categories": 4,
        "samples_per_time": 15,
        "mutation_rate": 0.40,
        "shift_strength": [0.2, 0.5],
        "seed": 12001,
    },
    {
        "name": "temporal_dna_12tp_fast",
        "generator": "temporal_evolution",
        "n_time_points": 12,
        "n_features": 400,
        "n_categories": 4,
        "samples_per_time": 12,
        "mutation_rate": 0.35,
        "shift_strength": [0.2, 0.5],
        "seed": 12002,
    },
]

TEMPORAL_DNA_MODERATE_CASES = [
    # Moderate evolution - some overlap between adjacent time points
    {
        "name": "temporal_dna_8tp_mod",
        "generator": "temporal_evolution",
        "n_time_points": 8,
        "n_features": 200,
        "n_categories": 4,
        "samples_per_time": 20,
        "mutation_rate": 0.20,
        "shift_strength": [0.15, 0.4],
        "seed": 12100,
    },
    {
        "name": "temporal_dna_10tp_mod",
        "generator": "temporal_evolution",
        "n_time_points": 10,
        "n_features": 300,
        "n_categories": 4,
        "samples_per_time": 15,
        "mutation_rate": 0.22,
        "shift_strength": [0.15, 0.4],
        "seed": 12101,
    },
]

TEMPORAL_DNA_SLOW_CASES = [
    # Slow evolution (conserved) - hard to distinguish adjacent time points
    {
        "name": "temporal_dna_8tp_slow",
        "generator": "temporal_evolution",
        "n_time_points": 8,
        "n_features": 200,
        "n_categories": 4,
        "samples_per_time": 20,
        "mutation_rate": 0.10,
        "shift_strength": [0.1, 0.3],
        "seed": 12200,
    },
    {
        "name": "temporal_dna_6tp_slow",
        "generator": "temporal_evolution",
        "n_time_points": 6,
        "n_features": 300,
        "n_categories": 4,
        "samples_per_time": 25,
        "mutation_rate": 0.08,
        "shift_strength": [0.1, 0.3],
        "seed": 12201,
    },
]

TEMPORAL_PROTEIN_CASES = [
    # Protein-like evolution (20 amino acids)
    {
        "name": "temporal_protein_8tp",
        "generator": "temporal_evolution",
        "n_time_points": 8,
        "n_features": 100,
        "n_categories": 20,
        "samples_per_time": 20,
        "mutation_rate": 0.30,
        "shift_strength": [0.15, 0.4],
        "seed": 12300,
    },
    {
        "name": "temporal_protein_10tp",
        "generator": "temporal_evolution",
        "n_time_points": 10,
        "n_features": 150,
        "n_categories": 20,
        "samples_per_time": 15,
        "mutation_rate": 0.35,
        "shift_strength": [0.15, 0.4],
        "seed": 12301,
    },
]

TEMPORAL_LARGE_CASES = [
    # Large-scale temporal simulations
    {
        "name": "temporal_large_16tp",
        "generator": "temporal_evolution",
        "n_time_points": 16,
        "n_features": 500,
        "n_categories": 4,
        "samples_per_time": 15,
        "mutation_rate": 0.30,
        "shift_strength": [0.2, 0.5],
        "seed": 12400,
    },
    {
        "name": "temporal_large_20tp",
        "generator": "temporal_evolution",
        "n_time_points": 20,
        "n_features": 800,
        "n_categories": 4,
        "samples_per_time": 12,
        "mutation_rate": 0.32,
        "shift_strength": [0.2, 0.5],
        "seed": 12401,
    },
]

TEMPORAL_HIGHD_CASES = [
    # High-dimensional temporal evolution
    {
        "name": "temporal_highd_8tp_1k",
        "generator": "temporal_evolution",
        "n_time_points": 8,
        "n_features": 1000,
        "n_categories": 4,
        "samples_per_time": 20,
        "mutation_rate": 0.30,
        "shift_strength": [0.2, 0.5],
        "seed": 12500,
    },
    {
        "name": "temporal_highd_10tp_2k",
        "generator": "temporal_evolution",
        "n_time_points": 10,
        "n_features": 2000,
        "n_categories": 4,
        "samples_per_time": 15,
        "mutation_rate": 0.28,
        "shift_strength": [0.2, 0.5],
        "seed": 12501,
    },
]


def get_temporal_cases():
    """Get all temporal evolution test cases as a flat list."""
    return (
        [c.copy() for c in TEMPORAL_DNA_FAST_CASES]
        + [c.copy() for c in TEMPORAL_DNA_MODERATE_CASES]
        + [c.copy() for c in TEMPORAL_DNA_SLOW_CASES]
        + [c.copy() for c in TEMPORAL_PROTEIN_CASES]
        + [c.copy() for c in TEMPORAL_LARGE_CASES]
        + [c.copy() for c in TEMPORAL_HIGHD_CASES]
    )


# Category mapping for grouped access
TEMPORAL_CASES = {
    "temporal_dna_fast": TEMPORAL_DNA_FAST_CASES,
    "temporal_dna_moderate": TEMPORAL_DNA_MODERATE_CASES,
    "temporal_dna_slow": TEMPORAL_DNA_SLOW_CASES,
    "temporal_protein": TEMPORAL_PROTEIN_CASES,
    "temporal_large": TEMPORAL_LARGE_CASES,
    "temporal_highd": TEMPORAL_HIGHD_CASES,
}
