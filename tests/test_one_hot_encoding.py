#!/usr/bin/env python3
"""Verify that categorical/phylogenetic/temporal generators now produce
one-hot encoded binary data instead of raw category indices."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.shared.generators.generate_case_data import generate_case_data
import numpy as np

# Test categorical
tc_cat = {
    "name": "test_cat",
    "generator": "categorical",
    "n_rows": 20,
    "n_cols": 10,
    "n_categories": 4,
    "n_clusters": 3,
    "entropy_param": 0.1,
    "seed": 42,
}
data, y, X_orig, meta = generate_case_data(tc_cat)
print("=== Categorical ===")
print(f"data_df shape: {data.shape}  (was 20x10, now 20x{10 * 4}=20x40)")
print(f"values in [0,1]: {set(np.unique(data.values))}")
print(f"n_features in meta: {meta['n_features']}")
print(f"n_features_original: {meta['n_features_original']}")
print(f"X_orig shape (raw categories preserved): {X_orig.shape}")
print()

# Test phylogenetic
tc_phylo = {
    "name": "test_phylo",
    "generator": "phylogenetic",
    "n_taxa": 4,
    "n_features": 10,
    "n_categories": 4,
    "samples_per_taxon": 5,
    "mutation_rate": 0.3,
    "seed": 42,
}
data, y, X_orig, meta = generate_case_data(tc_phylo)
print("=== Phylogenetic ===")
print(f"data_df shape: {data.shape}  (was 20x10, now 20x{10 * 4}=20x40)")
print(f"values in [0,1]: {set(np.unique(data.values))}")
print(f"n_features in meta: {meta['n_features']}")
print()

# Test temporal evolution
tc_temp = {
    "name": "test_temporal",
    "generator": "temporal_evolution",
    "n_time_points": 3,
    "n_features": 8,
    "n_categories": 3,
    "samples_per_time": 5,
    "mutation_rate": 0.2,
    "seed": 42,
}
data, y, X_orig, meta = generate_case_data(tc_temp)
print("=== Temporal Evolution ===")
print(f"data_df shape: {data.shape}  (was 15x8, now 15x{8 * 3}=15x24)")
print(f"values in [0,1]: {set(np.unique(data.values))}")
print(f"n_features in meta: {meta['n_features']}")
