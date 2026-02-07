"""Phylogenetic test cases for benchmarking.

DNA and protein sequence evolution simulations.
Tests the algorithm's ability to recover phylogenetic relationships.
"""

PHYLOGENETIC_DNA_CASES = [
    # DNA-like (4 categories: A, C, G, T)
    {
        "name": "phylo_dna_4taxa_low_mut",
        "generator": "phylogenetic",
        "n_taxa": 4,
        "n_features": 100,
        "n_categories": 4,
        "samples_per_taxon": 25,
        "mutation_rate": 0.2,
        "root_concentration": 1.0,
        "seed": 11000,
    },
    {
        "name": "phylo_dna_8taxa_low_mut",
        "generator": "phylogenetic",
        "n_taxa": 8,
        "n_features": 200,
        "n_categories": 4,
        "samples_per_taxon": 20,
        "mutation_rate": 0.2,
        "root_concentration": 1.0,
        "seed": 11001,
    },
    {
        "name": "phylo_dna_8taxa_med_mut",
        "generator": "phylogenetic",
        "n_taxa": 8,
        "n_features": 200,
        "n_categories": 4,
        "samples_per_taxon": 20,
        "mutation_rate": 0.4,
        "root_concentration": 1.0,
        "seed": 11002,
    },
    {
        "name": "phylo_dna_16taxa_low_mut",
        "generator": "phylogenetic",
        "n_taxa": 16,
        "n_features": 500,
        "n_categories": 4,
        "samples_per_taxon": 15,
        "mutation_rate": 0.25,
        "root_concentration": 1.0,
        "seed": 11003,
    },
]

PHYLOGENETIC_PROTEIN_CASES = [
    # Protein-like (20 categories: amino acids)
    {
        "name": "phylo_protein_4taxa",
        "generator": "phylogenetic",
        "n_taxa": 4,
        "n_features": 50,
        "n_categories": 20,
        "samples_per_taxon": 30,
        "mutation_rate": 0.3,
        "root_concentration": 0.5,
        "seed": 11100,
    },
    {
        "name": "phylo_protein_8taxa",
        "generator": "phylogenetic",
        "n_taxa": 8,
        "n_features": 100,
        "n_categories": 20,
        "samples_per_taxon": 20,
        "mutation_rate": 0.35,
        "root_concentration": 0.5,
        "seed": 11101,
    },
    {
        "name": "phylo_protein_12taxa",
        "generator": "phylogenetic",
        "n_taxa": 12,
        "n_features": 150,
        "n_categories": 20,
        "samples_per_taxon": 15,
        "mutation_rate": 0.4,
        "root_concentration": 0.5,
        "seed": 11102,
    },
]

PHYLOGENETIC_DIVERGENT_CASES = [
    # High mutation rate - taxa are very divergent
    {
        "name": "phylo_divergent_4taxa",
        "generator": "phylogenetic",
        "n_taxa": 4,
        "n_features": 100,
        "n_categories": 4,
        "samples_per_taxon": 25,
        "mutation_rate": 0.7,
        "root_concentration": 1.0,
        "seed": 11200,
    },
    {
        "name": "phylo_divergent_8taxa",
        "generator": "phylogenetic",
        "n_taxa": 8,
        "n_features": 200,
        "n_categories": 4,
        "samples_per_taxon": 20,
        "mutation_rate": 0.8,
        "root_concentration": 1.0,
        "seed": 11201,
    },
]

PHYLOGENETIC_CONSERVED_CASES = [
    # Low mutation rate - taxa are very similar (hard to distinguish)
    {
        "name": "phylo_conserved_4taxa",
        "generator": "phylogenetic",
        "n_taxa": 4,
        "n_features": 100,
        "n_categories": 4,
        "samples_per_taxon": 25,
        "mutation_rate": 0.05,
        "root_concentration": 1.0,
        "seed": 11300,
    },
    {
        "name": "phylo_conserved_8taxa",
        "generator": "phylogenetic",
        "n_taxa": 8,
        "n_features": 200,
        "n_categories": 4,
        "samples_per_taxon": 20,
        "mutation_rate": 0.08,
        "root_concentration": 1.0,
        "seed": 11301,
    },
]

PHYLOGENETIC_LARGE_CASES = [
    # Large-scale phylogenetic simulations
    {
        "name": "phylo_large_32taxa",
        "generator": "phylogenetic",
        "n_taxa": 32,
        "n_features": 500,
        "n_categories": 4,
        "samples_per_taxon": 10,
        "mutation_rate": 0.3,
        "root_concentration": 1.0,
        "seed": 11400,
    },
    {
        "name": "phylo_large_64taxa",
        "generator": "phylogenetic",
        "n_taxa": 64,
        "n_features": 1000,
        "n_categories": 4,
        "samples_per_taxon": 8,
        "mutation_rate": 0.35,
        "root_concentration": 1.0,
        "seed": 11401,
    },
]


def get_phylogenetic_cases():
    """Get all phylogenetic test cases as a flat list."""
    return (
        [c.copy() for c in PHYLOGENETIC_DNA_CASES]
        + [c.copy() for c in PHYLOGENETIC_PROTEIN_CASES]
        + [c.copy() for c in PHYLOGENETIC_DIVERGENT_CASES]
        + [c.copy() for c in PHYLOGENETIC_CONSERVED_CASES]
        + [c.copy() for c in PHYLOGENETIC_LARGE_CASES]
    )


# Category mapping for grouped access
PHYLOGENETIC_CASES = {
    "phylogenetic_dna": PHYLOGENETIC_DNA_CASES,
    "phylogenetic_protein": PHYLOGENETIC_PROTEIN_CASES,
    "phylogenetic_divergent": PHYLOGENETIC_DIVERGENT_CASES,
    "phylogenetic_conserved": PHYLOGENETIC_CONSERVED_CASES,
    "phylogenetic_large": PHYLOGENETIC_LARGE_CASES,
}
