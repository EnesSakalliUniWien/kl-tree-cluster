#!/usr/bin/env python3
"""
Numerical Walkthrough: Child-Parent Edge Test (Gate 2)

This script demonstrates the complete child-parent divergence test with
concrete numerical examples, showing inputs and outputs at each stage.

The child-parent test (Gate 2) answers:
  "Does this child cluster significantly diverge from its parent?"

Uses: Projected Wald chi-square test with spectral dimension estimation
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import networkx as nx

from benchmarks.shared.generators import generate_random_feature_matrix
from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.decomposition.backends.eigen_backend import (
    eigendecompose_correlation_backend,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence import (
    annotate_child_parent_divergence,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence.child_parent_projected_wald import (
    run_child_parent_projected_wald_test,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.projection_dimension_estimation.projection_dimension_estimators import (
    marchenko_pastur_signal_count,
)
from kl_clustering_analysis.tree.distributions import populate_distributions
from kl_clustering_analysis.tree.poset_tree import PosetTree

np.set_printoptions(precision=6, suppress=True, linewidth=120)

# =============================================================================
# GENERATE EXAMPLE DATA
# =============================================================================

print("=" * 80)
print("CHILD-PARENT EDGE TEST (GATE 2) - NUMERICAL WALKTHROUGH")
print("=" * 80)

# Generate a simple dataset with clear cluster structure
leaf_matrix_dict, cluster_assignments = generate_random_feature_matrix(
    n_rows=40,
    n_cols=50,
    entropy_param=0.05,
    n_clusters=2,  # Simple 2-cluster case
    random_seed=2001,
    balanced_clusters=True,
    feature_sparsity=0.05,
)

leaf_names = sorted(leaf_matrix_dict.keys())
X = np.array([leaf_matrix_dict[n] for n in leaf_names], dtype=float)
true_labels = np.array([cluster_assignments[n] for n in leaf_names])
data = pd.DataFrame(
    X,
    index=leaf_names,
    columns=[f"F{j}" for j in range(X.shape[1])],
)

print("\n1. INPUT DATA")
print(f"   Shape: {X.shape} (n={X.shape[0]} samples, d={X.shape[1]} features)")
print(f"   True clusters: {len(np.unique(true_labels))}")
print(f"   Cluster sizes: {np.bincount(true_labels)}")
print("   Data type: Binary (Bernoulli)")
print(f"   Mean feature activation: {X.mean():.3f}")

# =============================================================================
# BUILD TREE
# =============================================================================

print("\n2. TREE CONSTRUCTION")
Z = linkage(pdist(X, metric=config.TREE_DISTANCE_METRIC), method=config.TREE_LINKAGE_METHOD)
tree = PosetTree.from_linkage(Z, leaf_names=data.index.tolist())
print(f"   Tree nodes: {tree.number_of_nodes()}")
print(f"   Internal nodes: {sum(1 for n in tree.nodes() if tree.out_degree(n) > 0)}")
print(f"   Leaf nodes: {sum(1 for n in tree.nodes() if tree.out_degree(n) == 0)}")

# Populate distributions
populate_distributions(tree, data)

# =============================================================================
# SELECT A SPECIFIC EDGE FOR DETAILED ANALYSIS
# =============================================================================

print("\n3. SELECTING EDGE FOR DETAILED ANALYSIS")
print("-" * 80)

# Get all edges
edges = list(tree.edges())
# Pick an internal edge (not at leaves)
# Find a node with reasonable subtree size
target_child = None
target_parent = None

for parent, child in edges:
    if tree.out_degree(child) > 0:  # Child is internal node (not leaf)
        # Count descendants using NetworkX
        descendants = list(nx.descendants(tree, child))
        leaf_descendants = [d for d in descendants if tree.out_degree(d) == 0]
        if 5 <= len(leaf_descendants) <= 15:  # Medium-sized subtree
            target_child = child
            target_parent = parent
            break

if target_child is None:
    # Fallback: use first edge
    target_parent, target_child = edges[0]

print(f"   Selected edge: {target_parent} → {target_child}")

# Get distributions
child_dist = tree.nodes[target_child]["distribution"].copy()
parent_dist = tree.nodes[target_parent]["distribution"].copy()


# Count leaves
def count_leaves(tree, node):
    if tree.out_degree(node) == 0:
        return 1
    return sum(count_leaves(tree, child) for child in tree.successors(node))


n_child = count_leaves(tree, target_child)
n_parent = count_leaves(tree, target_parent)

print(f"\n   Child node ({target_child}):")
print(f"      Distribution shape: {child_dist.shape}")
print(f"      Leaf count (n_child): {n_child}")
print(f"      Distribution mean: {child_dist.mean():.4f}")
print(f"      Distribution std: {child_dist.std():.4f}")

print(f"\n   Parent node ({target_parent}):")
print(f"      Distribution shape: {parent_dist.shape}")
print(f"      Leaf count (n_parent): {n_parent}")
print(f"      Distribution mean: {parent_dist.mean():.4f}")
print(f"      Distribution std: {parent_dist.std():.4f}")

# =============================================================================
# STEP 1: COMPUTE STANDARDIZED Z-SCORES
# =============================================================================

print("\n4. STEP 1: COMPUTE STANDARDIZED Z-SCORES")
print("-" * 80)

# The z-score formula:
# z = (child_dist - parent_dist) / sqrt(variance)
# where variance = parent_dist * (1 - parent_dist) * (1/n_child - 1/n_parent)

nested_factor = 1.0 / n_child - 1.0 / n_parent
print("\n   Nested factor: (1/n_child - 1/n_parent)")
print(f"   = (1/{n_child} - 1/{n_parent})")
print(f"   = {1.0/n_child:.6f} - {1.0/n_parent:.6f}")
print(f"   = {nested_factor:.6f}")

# Compute variance (Bernoulli variance scaled by nested factor)
variance = parent_dist * (1 - parent_dist) * nested_factor
print("\n   Variance per feature: parent_dist * (1 - parent_dist) * nested_factor")
print(f"   Variance mean: {variance.mean():.6f}")
print(f"   Variance std: {variance.std():.6f}")
print(f"   Variance range: [{variance.min():.6f}, {variance.max():.6f}]")

# Standardize
z_scores = (child_dist - parent_dist) / np.sqrt(np.maximum(variance, 1e-10))

print("\n   Z-scores (standardized differences):")
print(f"   Shape: {z_scores.shape}")
print(f"   Mean: {z_scores.mean():.4f}")
print(f"   Std: {z_scores.std():.4f}")
print(f"   Range: [{z_scores.min():.4f}, {z_scores.max():.4f}]")
print(f"\n   First 10 z-scores: {z_scores[:10]}")

# =============================================================================
# STEP 2: SPECTRAL DIMENSION ESTIMATION
# =============================================================================

print("\n5. STEP 2: SPECTRAL DIMENSION ESTIMATION")
print("-" * 80)


# Collect data for the parent node (all leaves under parent)
def get_leaf_indices(tree, node, leaf_names):
    """Get indices of all leaves under a node."""
    if tree.out_degree(node) == 0:
        return [leaf_names.index(node)]
    indices = []
    for child in tree.successors(node):
        indices.extend(get_leaf_indices(tree, child, leaf_names))
    return indices


parent_leaf_indices = get_leaf_indices(tree, target_parent, leaf_names)
parent_leaf_data = X[parent_leaf_indices, :]

print("\n   Data matrix for parent subtree:")
print(
    f"   Shape: {parent_leaf_data.shape} (n={parent_leaf_data.shape[0]}, d={parent_leaf_data.shape[1]})"
)

# Run eigendecomposition
eig_result = eigendecompose_correlation_backend(
    parent_leaf_data,
    compute_eigenvectors=False,
)

eigenvalues = eig_result.eigenvalues
d_active = eig_result.active_feature_count
n_rows = parent_leaf_data.shape[0]  # Use data shape directly

print("\n   Eigendecomposition results:")
print(f"   Mode: {'DUAL (n×n Gram)' if eig_result.use_dual else 'PRIMAL (d×d corr)'}")
print(f"   Active features: {d_active}")
print(f"   Number of eigenvalues: {len(eigenvalues)}")

# Count positive eigenvalues
positive_eigenvalues = eigenvalues[eigenvalues > 0]
print(f"   Positive eigenvalues: {len(positive_eigenvalues)}")

# Estimate signal dimension using Marchenko-Pastur
k_signal = marchenko_pastur_signal_count(eigenvalues, n_rows, d_active)

# Compute Marchenko-Pastur threshold
sigma2 = float(np.median(positive_eigenvalues)) if len(positive_eigenvalues) > 0 else 1.0
gamma = d_active / n_rows
lambda_plus = sigma2 * (1.0 + np.sqrt(gamma)) ** 2

print("\n   Marchenko-Pastur analysis:")
print(f"   sigma² (median eigenvalue): {sigma2:.6f}")
print(f"   gamma (d/n): {gamma:.4f}")
print(f"   λ+ (MP threshold): {lambda_plus:.6f}")
print(f"   k (signal dimensions): {k_signal}")

# Show top eigenvalues
print("\n   Top 10 eigenvalues:")
for i in range(min(10, len(eigenvalues))):
    ev = eigenvalues[i]
    tag = " << SIGNAL" if ev > lambda_plus else ""
    print(f"      λ_{i+1:3d} = {ev:.6f}{tag}")

# =============================================================================
# STEP 3: BUILD PROJECTION MATRIX
# =============================================================================

print("\n6. STEP 3: BUILD PROJECTION MATRIX")
print("-" * 80)

from kl_clustering_analysis.hierarchy_analysis.statistics.projection.projected_wald.projected_wald_projection_basis import (
    build_projection_basis_with_padding,
)

# Use estimated k as projection dimension
spectral_k = max(k_signal, config.SPECTRAL_MINIMUM_DIMENSION)
print(f"\n   Projection dimension (k): {spectral_k}")

# Build projection matrix (random projection since we don't have pre-computed PCA)
projection_matrix, whitening_eigenvalues = build_projection_basis_with_padding(
    n_features=len(z_scores),
    k=spectral_k,
    pca_projection=None,  # No pre-computed PCA
    pca_eigenvalues=None,
    random_state=42,
)

print(f"   Projection matrix shape: {projection_matrix.shape}")
print("   (Rows: projection vectors, Cols: features)")
print(f"   Whitening eigenvalues: {whitening_eigenvalues}")

# =============================================================================
# STEP 4: PROJECT Z-SCORES
# =============================================================================

print("\n7. STEP 4: PROJECT Z-SCORES ONTO PROJECTION BASIS")
print("-" * 80)

# Project: projected = R @ z
projected_z = projection_matrix @ z_scores

print("\n   Projected z-scores:")
print(f"   Shape: {projected_z.shape}")
print(f"   Values: {projected_z}")
print(f"   Mean: {projected_z.mean():.4f}")
print(f"   Std: {projected_z.std():.4f}")

# =============================================================================
# STEP 5: COMPUTE WALD STATISTIC AND P-VALUE
# =============================================================================

print("\n8. STEP 5: COMPUTE WALD STATISTIC AND P-VALUE")
print("-" * 80)

from kl_clustering_analysis.hierarchy_analysis.statistics.projection.projected_wald.projected_wald_reference_distribution import (
    compute_projected_pvalue,
)

# Compute test statistic: T = Σ (projected_z)²
# This is chi-squared distributed under null hypothesis

test_statistic_raw = float(np.sum(projected_z**2))
print("\n   Raw test statistic (T = Σ z²):")
print(f"   T = {test_statistic_raw:.4f}")

# Compute p-value from chi-squared distribution
from scipy.stats import chi2 as chi2_dist

p_value_raw = float(chi2_dist.sf(test_statistic_raw, df=spectral_k))
print("\n   P-value (chi-squared test):")
print(f"   df = {spectral_k}")
print(f"   p = P(χ²({spectral_k}) > {test_statistic_raw:.4f})")
print(f"   p = {p_value_raw:.6f}")

# =============================================================================
# STEP 6: FULL PROJECTED WALD TEST (WITH WHITENING)
# =============================================================================

print("\n9. STEP 6: FULL PROJECTED WALD TEST (WITH WHITENING)")
print("-" * 80)

# When we have eigenvalues, we can whiten the test statistic
# T_whitened = Σ (projected_z_i)² / λ_i

# For this example, use identity eigenvalues (no whitening)
# In production, eigenvalues come from parent node's PCA

test_statistic, effective_df, p_value = compute_projected_pvalue(
    projected_z,
    degrees_of_freedom=spectral_k,
    eigenvalues=whitening_eigenvalues,  # None in this example
)

print("\n   Test results:")
print(f"   Test statistic (T): {test_statistic:.4f}")
print(f"   Effective df: {effective_df}")
print(f"   P-value: {p_value:.6f}")

# =============================================================================
# STEP 7: COMPLETE EDGE TEST VIA HELPER FUNCTION
# =============================================================================

print("\n10. STEP 7: COMPLETE EDGE TEST (HELPER FUNCTION)")
print("-" * 80)

# Use the production helper function
test_stat, df, p_val, is_invalid = run_child_parent_projected_wald_test(
    child_dist=child_dist,
    parent_dist=parent_dist,
    n_child=n_child,
    n_parent=n_parent,
    branch_length=None,
    mean_branch_length=None,
    spectral_k=spectral_k,
    pca_projection=None,
    pca_eigenvalues=None,
)

print("\n   Production function results:")
print(f"   Test statistic: {test_stat:.4f}")
print(f"   Degrees of freedom: {df}")
print(f"   P-value: {p_val:.6f}")
print(f"   Invalid test: {is_invalid}")

# Decision
alpha = config.EDGE_ALPHA  # Default: 0.001
rejects_null = p_val < alpha
print(f"\n   Decision (α = {alpha}):")
print(
    f"   p {'<' if rejects_null else '>='} α → {'REJECT H₀ (significant)' if rejects_null else 'FAIL TO REJECT H₀ (not significant)'}"
)

# =============================================================================
# FULL ANNOTATION FOR ALL EDGES
# =============================================================================

print("\n11. FULL ANNOTATION: ALL EDGES IN TREE")
print("-" * 80)

# Prepare annotations DataFrame with leaf counts
leaf_counts = {}
for node in tree.nodes():
    leaf_counts[node] = count_leaves(tree, node)

annotations_df = pd.DataFrame({"leaf_count": leaf_counts})

# Run full annotation
annotated_df = annotate_child_parent_divergence(
    tree=tree,
    annotations_df=annotations_df,
    significance_level_alpha=config.EDGE_ALPHA,
    leaf_data=data,
)

# Show results for selected columns
print("\n   Annotated edges (sample):")
print(f"   Total edges: {len(annotated_df)}")

# Show edge test results
edge_columns = [
    "Child_Parent_Divergence_P_Value",
    "Child_Parent_Divergence_P_Value_BH",
    "Child_Parent_Divergence_df",
    "Child_Parent_Divergence_Significant",
]

# Filter to internal nodes only
internal_nodes = [n for n in tree.nodes() if tree.out_degree(n) > 0]
print("\n   Internal node edge tests:")
display_df = annotated_df.loc[internal_nodes, edge_columns].head(10)
print(display_df.to_string())

# =============================================================================
# SUMMARY TABLE
# =============================================================================

print("\n" + "=" * 80)
print("SUMMARY: CHILD-PARENT EDGE TEST INPUT/OUTPUT")
print("=" * 80)

print(
    """
INPUTS:
-------
1. child_dist (np.ndarray, shape=(d,))
   - Bernoulli distribution for child node
   - Values in [0, 1] representing feature activation probabilities
   
2. parent_dist (np.ndarray, shape=(d,))
   - Bernoulli distribution for parent node
   - Values in [0, 1]
   
3. n_child (int)
   - Number of leaves in child subtree
   
4. n_parent (int)
   - Number of leaves in parent subtree
   - Must be > n_child
   
5. spectral_k (int, optional)
   - Projection dimension for spectral test
   - Estimated via Marchenko-Pastur signal counting
   - Default: max(k_signal, SPECTRAL_MINIMUM_DIMENSION)
   
6. branch_length (float, optional)
   - Edge branch length for variance scaling
   - Used only if FELSENSTEIN_SCALING=True (default: False)

OUTPUTS:
--------
1. test_statistic (float)
   - Wald chi-square statistic: T = Σ (projected_z)²
   - Under H₀: T ~ χ²(k) where k = projection dimension
   
2. degrees_of_freedom (float)
   - Effective degrees of freedom (projection dimension)
   - Typically: k = spectral_k
   
3. p_value (float)
   - P(χ²(df) > test_statistic)
   - Small p-value → child significantly diverges from parent
   
4. is_invalid (bool)
   - True if test could not be computed (e.g., k=0)

DECISION:
---------
If p_value < EDGE_ALPHA (default: 0.001):
  → REJECT null hypothesis
  → Child diverges significantly from parent
  → Gate 2 PASSES (edge is "significant")
  
If p_value >= EDGE_ALPHA:
  → FAIL TO REJECT null hypothesis
  → Child does NOT significantly diverge
  → Gate 2 FAILS (edge is "not significant")
"""
)

# =============================================================================
# NUMERICAL EXAMPLE SUMMARY
# =============================================================================

print("=" * 80)
print(f"NUMERICAL EXAMPLE SUMMARY (Edge: {target_parent} → {target_child})")
print("=" * 80)

print(
    f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│ INPUT DATA                                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ child_dist.shape     = {child_dist.shape} (d={len(child_dist)} features)
│ parent_dist.shape    = {parent_dist.shape}
│ n_child              = {n_child}
│ n_parent             = {n_parent}
│ spectral_k           = {spectral_k}
├─────────────────────────────────────────────────────────────────────────────┤
│ INTERMEDIATE RESULTS                                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ nested_factor        = {nested_factor:.6f}
│ variance.mean()      = {variance.mean():.6f}
│ z_scores.mean()      = {z_scores.mean():.4f}
│ z_scores.std()       = {z_scores.std():.4f}
│ projection_matrix    = {projection_matrix.shape}
├─────────────────────────────────────────────────────────────────────────────┤
│ TEST RESULTS                                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│ test_statistic (T)   = {test_stat:.4f}
│ degrees_of_freedom   = {df}
│ p_value              = {p_val:.6f}
│ alpha (EDGE_ALPHA)   = {config.EDGE_ALPHA}
├─────────────────────────────────────────────────────────────────────────────┤
│ DECISION                                                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ p {'<=' if p_val < config.EDGE_ALPHA else '>'} {config.EDGE_ALPHA} → {('REJECT H₀' if p_val < config.EDGE_ALPHA else 'FAIL TO REJECT H₀'):20s} → Gate 2 {'PASSES' if p_val < config.EDGE_ALPHA else 'FAILS':7s}
└─────────────────────────────────────────────────────────────────────────────┘
"""
)

print("\nFor the full tree:")
significant_edges = annotated_df["Child_Parent_Divergence_Significant"].sum()
total_edges = len(internal_nodes)
print(
    f"  Significant edges: {significant_edges} / {total_edges} ({100*significant_edges/total_edges:.1f}%)"
)
print(f"  Mean p-value: {annotated_df['Child_Parent_Divergence_P_Value'].mean():.4f}")
print(f"  Median p-value: {annotated_df['Child_Parent_Divergence_P_Value'].median():.4f}")
print(f"  Median p-value: {annotated_df['Child_Parent_Divergence_P_Value'].median():.4f}")
