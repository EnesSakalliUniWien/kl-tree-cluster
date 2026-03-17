# KL-Divergence Hierarchical Clustering Toolkit

## Architecture Overview

This toolkit decomposes hierarchical clustering trees into statistically stable partitions using information-theoretic tests. The main flow:

```
Binary matrix → linkage → PosetTree → populate_node_divergences() → decompose() → cluster assignments
```

### Core Components

- **`PosetTree`** ([tree/poset_tree.py](kl_clustering_analysis/tree/poset_tree.py)): NetworkX DiGraph subclass; central data structure holding hierarchy, distributions, and statistical annotations via `stats_df` property. Constructed via `from_linkage()` or `from_agglomerative()`.

- **`TreeDecomposition`** ([hierarchy_analysis/tree_decomposition.py](kl_clustering_analysis/hierarchy_analysis/tree_decomposition.py)): Orchestrator class (~757 lines) that delegates gate logic to `GateEvaluator`. Handles pre-caching node metadata, annotation orchestration, post-hoc merge dispatch, and cluster assignment extraction. `decompose_tree()` and `decompose_tree_v2()` use `iterate_worklist` + `process_node` / `process_node_v2` from `gates.py`.

- **`GateEvaluator`** ([hierarchy_analysis/gates.py](kl_clustering_analysis/hierarchy_analysis/gates.py)): Extracted gate logic (~343 lines) applying three statistical gates to determine cluster boundaries:
  1. **Binary structure gate**: Parent must have exactly two children
  2. **Child-parent divergence gate**: Projected Wald chi-square test on KL(child || parent)
  3. **Sibling divergence gate**: Configurable via `config.SIBLING_TEST_METHOD` — one of `"wald"`, `"cousin_ftest"`, `"cousin_adjusted_wald"`, `"cousin_tree_guided"`, or `"cousin_weighted_wald"` (see Sibling Test Methods below)

  Constructor accepts injected `children_map`, `descendant_leaf_sets`, and `root` to avoid coupling to `PosetTree` internals. Also provides `should_split_v2()` with **signal localization** (`use_signal_localization=True`), which drills into WHERE divergence originates and enables cross-boundary partial merges. Includes a **power guard**: when localization finds zero significant difference pairs after BH correction, returns `(True, None)` — trusting the aggregate Gate 3 SPLIT but discarding the powerless localization result.

  Free functions `iterate_worklist`, `process_node`, `process_node_v2`, and dataclass `V2TraversalState` handle the top-down traversal loop.

- **`SignalLocalization`** ([hierarchy_analysis/signal_localization.py](kl_clustering_analysis/hierarchy_analysis/signal_localization.py)): Recursive cross-boundary testing to identify which subtrees are truly different vs. which could be merged, producing `SimilarityEdge` / `LocalizationResult` objects.

- **`config.py`** ([kl_clustering_analysis/config.py](kl_clustering_analysis/config.py)): Central configuration for thresholds (alpha levels, projection params, linkage method). Key constants:
  - `SIGNIFICANCE_ALPHA = 0.05` — edge-test (Gate 2) significance level
  - `SIBLING_ALPHA = 0.05` — sibling-test (Gate 3) significance level
  - `ALPHA_LOCAL = 0.05` — local significance level passed to decomposer
  - `SIBLING_TEST_METHOD = "cousin_weighted_wald"` — Gate 3 implementation toggle
  - `TREE_DISTANCE_METRIC = "hamming"` — distance metric for linkage
  - `TREE_LINKAGE_METHOD = "average"` — linkage method
  - `POSTHOC_MERGE = True` — enable/disable bottom-up merge pass
  - `POSTHOC_MERGE_ALPHA = None` — override alpha for post-hoc merge (defaults to `SIBLING_ALPHA`)
  - `USE_SIGNAL_LOCALIZATION = False` — enable v2 decomposition with signal localization
  - `PROJECTION_EPS = 0.3` — JL-lemma epsilon for projection dimension
  - `PROJECTION_MIN_K = "auto"` — minimum projection dimension; `"auto"` estimates from the data's effective rank (Shannon entropy of eigenvalue spectrum), clamped to `[4, 20]` (floor raised from 1→4 to prevent low-power χ²(1) tests at small clusters); set to an int for a fixed floor
  - `PROJECTION_RANDOM_SEED = 42` — seed for random projection matrix
  - `SPECTRAL_METHOD = "marchenko_pastur"` — per-node projection dimension via eigendecomposition (Gate 2 only)
  - `EPSILON = 1e-9` — numerical stability constant

## Key Patterns

### Statistical Test Return Convention
All statistical tests return `(test_statistic, degrees_of_freedom, p_value)` tuples:
```python
stat, df, pval = sibling_divergence_test(left_dist, right_dist, n_left, n_right)
```

### Data Preparation
Input must be a binary DataFrame. Continuous data is typically binarized via median threshold:
```python
X_binary = (X_continuous > np.median(X_continuous, axis=0)).astype(int)
data = pd.DataFrame(X_binary, index=[f"S{i}" for i in range(n)], columns=[f"F{j}" for j in range(p)])
```

### Minimal Pipeline Usage
```python
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis import config

Z = linkage(pdist(data.values, metric=config.TREE_DISTANCE_METRIC), method=config.TREE_LINKAGE_METHOD)
tree = PosetTree.from_linkage(Z, leaf_names=data.index.tolist())
results = tree.decompose(leaf_data=data, alpha_local=0.05, sibling_alpha=0.05)
```

### Decompose Result Accessors
`tree.decompose()` returns a **dict** with these keys:
```python
results["cluster_assignments"]   # dict[int, {"root_node", "leaves", "size"}]
results["num_clusters"]          # int — number of clusters found
results["posthoc_merge_audit"]   # list[dict] — merge audit trail
results["independence_analysis"] # dict — alpha, decision_mode, posthoc_merge flag
```
The annotated **stats DataFrame** (with all gate columns) is NOT in the returned dict — it is cached on the tree:
```python
stats_df = tree.stats_df  # pd.DataFrame, index = node IDs
```
Key gate columns in `stats_df`:
| Column | Type | Gate | Meaning |
|--------|------|------|---------|
| `Child_Parent_Divergence_Significant` | bool | 2 | Child diverges from parent |
| `Child_Parent_Divergence_P_Value_BH` | float | 2 | BH-corrected edge p-value |
| `Child_Parent_Divergence_df` | float | 2 | Edge test degrees of freedom |
| `Sibling_BH_Different` | bool | 3 | Siblings significantly different |
| `Sibling_Divergence_P_Value` | float | 3 | Raw sibling p-value |
| `Sibling_Divergence_P_Value_Corrected` | float | 3 | BH-corrected sibling p-value |
| `Sibling_Test_Statistic` | float | 3 | Raw Wald T (or deflated T_adj) |
| `Sibling_Degrees_of_Freedom` | float | 3 | Sibling test df |
| `Sibling_Divergence_Skipped` | bool | 3 | Test skipped (non-binary, leaf, or too few samples) |
| `Sibling_Test_Method` | str | 3 | Calibration method used |
| `Sibling_BH_Same` | bool | 3 | Siblings not significantly different |

Calibration audit (weighted Wald):
```python
audit = tree.stats_df.attrs.get("sibling_divergence_audit", {})
# Keys: calibration_method, calibration_n, global_c_hat, max_observed_ratio, diagnostics
# diagnostics keys: R2, beta, ...
```

### Annotation Flow
Statistical metrics are annotated incrementally to `results_df` (a DataFrame):
```python
results_df = annotate_child_parent_divergence(tree, results_df, significance_level_alpha=0.05)
results_df = annotate_sibling_divergence(tree, results_df, significance_level_alpha=0.05)
```

### Felsenstein Branch-Length Adjustment
Variance in both child-parent and sibling tests is scaled by normalized branch length
following Felsenstein's (1985) Phylogenetic Independent Contrasts:
- **Edge test** (`edge_significance.py`): `Var *= 1 + BL / mean_BL` (per-edge)
- **Sibling test** (`pooled_variance.py`): `Var *= 1 + BL_sum / (2 * mean_BL)` (pair)

Longer branches → more expected divergence → harder to declare significance.

**No fallbacks**: The codebase enforces strict error handling over defensive defaults:
- If `branch_length_sum` is provided but `mean_branch_length` is missing or ≤ 0, the code raises `ValueError`. Callers must supply valid branch lengths — there is no silent fallback to an arbitrary constant.
- In `pipeline.py`, SBM cases require the pre-computed modularity `distance_condensed`. If it is `None`, a `ValueError` is raised — there is no silent fallback to `pdist()` on raw adjacency data.
- Never use `.get(key, default)` to silently paper over missing data. If a value is required, its absence is a bug that must surface as an error.

### Post-hoc Merge with Audit Trail
`apply_posthoc_merge()` returns `(merged_roots, audit_trail)` where `audit_trail` is a
list of dicts recording every candidate merge, its p-value, and outcome.

## Developer Workflows

### Running Tests
```bash
pytest                           # all tests, quiet mode
pytest tests/test_poset_tree.py  # specific test file
pytest --progress                # show test count progress
```

The test suite spans 30+ test files. Key test files:
- `test_signal_localization.py` (23 tests) — signal localization correctness
- `test_categorical_distributions.py` (18 tests) — categorical distribution handling
- `test_cluster_validation_integration.py` (2 tests) — end-to-end K recovery and ARI validation

### Benchmarking
Benchmarks live in `benchmarks/` with shared utilities in `benchmarks/shared/`. Run via:
```python
from benchmarks.shared.pipeline import benchmark_cluster_algorithm
df_results, fig = benchmark_cluster_algorithm(test_cases=cases, methods=["kl"])
```

### Debug Scripts
`debug_scripts/` contains analysis scripts for specific test cases (e.g., `analyze_case_17.py`). These are standalone investigations, not part of the test suite.

## Module Structure

| Directory | Purpose |
|-----------|---------|
| `tree/` | `PosetTree` (poset_tree.py), distributions, tree utilities |
| `hierarchy_analysis/` | Tree decomposition (orchestrator), gate logic (`gates.py`), post-hoc merge, signal localization |
| `hierarchy_analysis/statistics/` | Statistical tests (KL chi-square, sibling divergence, multiple testing, pooled variance, branch_length_utils, power analysis, spectral_dimension) |
| `hierarchy_analysis/statistics/sibling_divergence/` | Sibling divergence test implementations: `sibling_divergence_test.py` (Wald), `cousin_calibrated_test.py` (F-test), `cousin_adjusted_wald.py` (adjusted Wald), `cousin_tree_guided.py` (tree-guided), `cousin_weighted_wald.py` (weighted Wald — current default) |
| `hierarchy_analysis/statistics/kl_tests/` | Edge significance testing (`edge_significance.py`) |
| `hierarchy_analysis/statistics/multiple_testing/` | BH correction, tree-aware BH |
| `information_metrics/kl_divergence/` | KL divergence calculations for Bernoulli/Categorical/Poisson |
| `core_utils/` | Data utilities, pipeline helpers, tree utilities |
| `plot/` | Cluster tree visualization, color mapping |
| `benchmarks/` | Benchmark runners, shared cases/generators, metrics (ARI, NMI, Purity), reports |

## Benchmark Infrastructure

Benchmark infrastructure is centralized in `benchmarks/shared/`:

| System | Path | Import Base | Used By |
|--------|------|-------------|---------|
| **Shared benchmark stack** | `benchmarks/shared/` | `benchmarks.shared.*` | All `benchmarks/*/run.py`, tests, and benchmark analysis scripts |

Current shared-stack features include:
- `matrix_audit` parameter and audit export infrastructure
- `debug_trace.py` for failure diagnosis
- `evolution.py` for temporal evolution tracking
- `audit_utils.py` for TensorBoard-style matrix exports
- `linkage_matrix` captured in KL runner `extra` dict

### Running Benchmarks
```bash
# Full benchmark (96 test cases, 9 methods: kl, kl_rogerstanimoto, leiden, louvain, kmeans, spectral, dbscan, optics, hdbscan)
python benchmarks/full/run.py

# Specialized overlapping/gaussian cases
python benchmarks/run_final.py

# Branch-length sensitivity
python benchmarks/branch_length/run.py
python benchmarks/branch_length_3d/run.py

# Multi-split cases
python benchmarks/multi_split/run.py
```

### Broken / Removed Benchmark Scripts (Known Issues)
- **`benchmarks/temporal/`**: Entirely removed. The module had multiple bugs: `run.py` imported non-existent functions (`run_incremental_temporal_benchmark`, `run_temporal_benchmark_suite`), and the `cases.py` referenced a `temporal_evolution` generator that was never wired into the shared pipeline. The `TEMPORAL_CASES` import has been removed from `benchmarks/shared/cases/__init__.py`.
- **`benchmarks/phylogenetic/run.py`**: Previously imported from non-existent `benchmarks.phylogenetic.generator`; fixed to use `benchmarks.shared.generators.generate_phylogenetic`.

### Benchmark Output
Results are saved to timestamped directories under `benchmarks/results/`:
```
benchmarks/results/run_YYYYMMDD_HHMMSSZ/
  ├── full_benchmark_comparison.csv
  ├── full_benchmark_report.pdf
  ├── failure_report.md
  ├── audit/
  └── plots/
      └── case_N.pdf
```

### Latest Benchmark Results (2026-02-14, 95 cases × 2 methods, post-extrapolation-guard)

| Metric | `kl` (hamming) | `kl_rogerstanimoto` |
|--------|---------------|---------------------|
| Mean ARI | 0.757 | 0.759 |
| Median ARI | 1.000 | 1.000 |
| Exact K | 59/95 | 61/95 |
| K=1 cases | 10 | 11 |

**Before the fix** (pre-extrapolation-guard): `kl` had Mean ARI 0.646, Exact K 49/95, K=1 in 30 cases.

**Known weaknesses**:
- **Phylogenetic cases**: Severe over-splitting (e.g. `phylo_divergent_8taxa` finds 72 clusters instead of 8). Tree construction from categorical phylogenetic data produces many leaves with near-zero branch lengths, causing the statistical gates to trigger everywhere.
- **Under-splitting (K=1)**: Largely resolved by the extrapolation guard (2026-02-14). Remaining K=1 cases are SBM and phylogenetic cases with structural data-type issues.
- **`kl` outperforms `kl_rogerstanimoto`** across all metrics — hamming distance produces better trees for binarized data.

## Statistical Gates — Detailed Flow

### Gate 1: Binary Structure
Parent must have exactly 2 children. Non-binary nodes are merged (no split).

### Gate 2: Child-Parent Divergence (Signal Detection)
Tests whether children are significantly different from parent using projected Wald chi-square:
- Nested variance: `Var = θ(1-θ) × (1/n_child - 1/n_parent)` accounts for child being subset of parent
- Felsenstein scaling: `Var *= 1 + BL/mean_BL`
- Projection dimension: determined by `config.SPECTRAL_METHOD`:
  - `"marchenko_pastur"` (default): Per-node eigendecomposition of the local **correlation** matrix of descendant data (leaves + internal node distributions). Projection dimension k = count of eigenvalues above the Marchenko-Pastur upper bound. Uses **PCA-based whitened projection** `T = Σ (vᵢᵀz)² / λᵢ ~ χ²(k)` (exact under H₀).
  - `None`: Legacy JL-based dimension `k ≈ 8·ln(n)/ε²`, with information cap (`k ≤ n` when `d ≥ 4n`), random orthonormal projection.
- FDR: `tree_bh` (default), `flat`, or `level_wise` correction
- If **neither** child diverges → MERGE (noise, no signal to split on)

**Note**: Spectral dimensions ARE passed to the sibling test (Gate 3) via the **min-child strategy**: for each binary parent, the sibling test uses `k = min(k_left, k_right)` where k_left and k_right are the Marchenko-Pastur spectral dimensions from Gate 2. This is conservative — the smaller child's signal dimension bounds what can be reliably detected. When neither child has a positive spectral k, the sibling test falls back to JL-based dimension.

### Gate 3: Sibling Divergence (Cluster Separation)
Tests whether siblings differ from each other. The test method is controlled by `config.SIBLING_TEST_METHOD`:
- If siblings are **same** → MERGE
- If siblings are **different** → SPLIT (or LOCALIZE in v2)
- If test was **skipped** (too few samples, no signal) → MERGE (conservative)

### Feature Usage
Sibling and edge divergence tests use all available features (no MI pre-filtering)
to avoid post-selection inference bias.

## Post-Selection Bias & Sibling Test Methods

### Root Cause: Post-Selection Inflation
The linkage tree is built from the **same data** used for statistical testing. This creates post-selection bias: the Wald χ² statistics are systematically inflated because the tree was optimized to separate the data. Empirical calibration under the null shows:
- **Edge test (Wald)**: 15.9% Type I error (expected: 5%)
- **Sibling test (Wald)**: 38.8% Type I error (expected: 5%)

This is not fixable by adjusting variance formulas, BH correction, or branch-length scaling — the inflation is structural.

### Sibling Test Methods (`config.SIBLING_TEST_METHOD`)

Five sibling test implementations are available, toggled via `config.SIBLING_TEST_METHOD`:

#### 1. `"wald"` — Original Wald χ² (anti-conservative)
- **File**: `sibling_divergence_test.py` → `annotate_sibling_divergence()`
- **Statistic**: `T = ||R·z||² ~ χ²(k)` where z is the standardized proportion difference
- **Pooled variance**: `Var = p(1-p) × (1/n₁ + 1/n₂)`, Felsenstein-scaled
- **Calibration**: ~38.8% Type I under null (inflated by post-selection)
- **Power**: Excellent — finds true clusters reliably, but over-splits noise
- **Use when**: Benchmarking, legacy comparison, or when over-splitting is acceptable

#### 2. `"cousin_ftest"` — Cousin-Calibrated F-test (conservative on K>2)
- **File**: `cousin_calibrated_test.py` → `annotate_sibling_divergence_cousin()`
- **Statistic**: `F = (T_LR/k_LR) / (T_UL_UR/k_UU) ~ F(k_LR, k_UU)`
- **Architecture**: At parent P with uncle U, computes sibling stat at P and cousin reference stat at U (same tree depth). Post-selection inflation c cancels in the ratio.
- **Calibration**: 5.0% Type I, KS p=0.98 (perfectly calibrated under null)
- **Power**: Good for K=2–3; severely under-splits for K>3 because cousins are also genuinely different when multiple clusters exist, making F≈1 even for real splits
- **Fallback**: When uncle is unavailable (root's children, leaf uncle), falls back to raw Wald. Fallback tracked via `Sibling_Test_Method` column.
- **Use when**: K is expected to be small (2–3), or calibration is more important than power

#### 3. `"cousin_adjusted_wald"` — Cousin-Adjusted Wald
- **File**: `cousin_adjusted_wald.py` → `annotate_sibling_divergence_adjusted()`
- **Architecture**: Two-pass approach:
  1. Compute raw Wald T for ALL sibling pairs
  2. Identify *null-like* pairs (neither child edge-significant) → these estimate pure inflation
  3. Fit log-linear regression: `log(T/k) = β₀ + β₁·log(BL_sum) + β₂·log(n_parent)`
  4. Predict ĉ for focal pairs, deflate: `T_adj = T / ĉ`, compare to χ²(k)
- **Calibration features**: Uses branch-length sum and n_descendants (not tree depth)
- **Fallback tiers**: ≥5 null-like pairs → regression; 3–4 → global median ĉ; <3 → raw Wald (ĉ=1)
- **Extrapolation guard**: `_predict_c()` clamps regression predictions at `max_observed_ratio` — the maximum T/k ratio from null-like calibration pairs. This prevents β₂·log(n_parent) from extrapolating ĉ beyond the calibration data range at the root node (where n_parent is largest).
- **Audit**: `df.attrs["sibling_divergence_audit"]` contains `calibration_method`, `global_c_hat`, `calibration_n`, `diagnostics` (including regression β, R², and `max_observed_ratio`)
- **Use when**: Trees with many internal nodes provide enough null-like calibration pairs

#### 4. `"cousin_tree_guided"` — Tree-Guided Cousin
- **File**: `cousin_tree_guided.py` → `annotate_sibling_divergence_tree_guided()`
- **Architecture**: Walks up the tree from each focal pair to find topologically nearest null-like relatives (neither child edge-significant), uses their median T/k as local ĉ. No global regression — adapts to local tree structure.
- **Fallback**: When no local null-like pairs found, falls back to global median ĉ.
- **Use when**: Trees with heterogeneous local structure where a global regression model is a poor fit.

#### 5. `"cousin_weighted_wald"` — Cousin-Weighted Wald (current default)
- **File**: `cousin_weighted_wald.py` → `annotate_sibling_divergence_weighted()`
- **Architecture**: Uses ALL sibling pairs with continuous weights `w_i = min(p_edge_L, p_edge_R)`. Fits an **intercept-only** Gamma GLM: `E[T_i/k_i] = exp(β₀)` with weights w_i. This estimates a single global inflation factor `ĉ = exp(β₀)` = weighted mean of T/k. Covariates `log(BL_sum)` and `log(n_parent)` were removed because they confound signal strength with post-selection inflation (under H₀, c is constant and independent of n and BL).
- **Advantage**: Uses all calibration data (every pair contributes) → stable estimate. Continuous weighting avoids the hard binary null/non-null split. No extrapolation risk since ĉ is a single constant (no covariates).
- **Prediction**: `predict_weighted_inflation_factor()` returns `exp(β₀)`, clamped to `[1.0, max_observed_ratio]`. `bl_sum` and `n_parent` parameters are retained for API compatibility but are not used.
- **Use when**: Default for all cases. Better calibration stability than binary-split methods.

### Diagnostic Script
`debug_scripts/diagnose_pipeline.py` traces all three gates for all three sibling methods on any test case, showing:
- Gate 1 (binary), Gate 2 (edge L/R), Gate 3 (sibling diff/skip) → final SPLIT/FAIL
- Calibration audit (regression β, R², ĉ, null-like pair count)
- Side-by-side K and ARI for wald, cousin_adjusted_wald, cousin_ftest

### Known Trade-offs (2026-02-14, post-extrapolation-guard)

| Case | True K | Wald | Adj. Wald | Cousin F |
|------|--------|------|-----------|----------|
| `trivial_2c` | 2 | K=2 ✓ | K=2 ✓ | K=2 ✓ |
| `block_4c` | 4 | K=4 ✓ | K=4 ✓ | K=4 ✓ |
| `gauss_clear_small` | 3 | K=3 ✓ | K=3 ✓ | K=3 ✓ |
| `sparse_72x72` | 4 | K=4 ✓ | K=4 ✓ | K=4 ✓ |
| `binary_perfect_4c` | 4 | K=4 ✓ | K=4 ✓ | K=2 ✗ |

The previous K=1 failures for `gauss_clear_small` (and ~30 other cases) were caused by regression extrapolation beyond calibration range. Fixed by clamping ĉ at `max_observed_ratio`. **Cousin F** still fails on K>3 because real multi-cluster signal inflates both numerator and denominator equally.

## Design Invariants

- `Sibling_BH_Same` is only set to `True` for **tested** parent nodes — untested nodes (leaves, skipped) keep the default `False`
- `Sibling_Divergence_Skipped` is `True` for nodes where the sibling test was skipped (non-binary parent, leaf node, insufficient samples, no child-parent signal) — these nodes are treated as MERGE decisions. All five sibling annotation methods consistently mark non-binary and leaf nodes as skipped.
- `GateEvaluator.should_split` (v1) and `GateEvaluator.should_split_v2` return `False` (not raise) when encountering skipped nodes. Both apply identical gate logic; v2 additionally runs signal localization when Gate 3 passes. (Prior to step 3.4, these lived as `_should_split` / `_should_split_v2` inline methods on `TreeDecomposition`.)
- **FIXED — `should_split_v2` localization power guard** (`gates.py`): When localization finds zero significant difference pairs after BH correction, `should_split_v2` now returns `(True, None)` — trusting the aggregate Gate 3 SPLIT decision but discarding the misleading localization result (which has only similarity edges). This triggers a hard v1-style split. Previously returned `(True, loc_result)` with empty `difference_pairs`, causing false cross-boundary merges via similarity edges.
- **FIXED — `extract_constrained_clusters` leaf overlap**: Ancestor merge_points are pruned before cluster extraction, and a leaf deduplication pass (smallest-cluster-wins) handles residual overlaps from ancestor-descendant graph nodes at different localization levels.
- `apply_posthoc_merge()` always returns `Tuple[Set[str], List[Dict]]` — destructure as `merged_roots, audit_trail = ...`
- `decompose()` dispatches to `decompose_tree_v2()` when `use_signal_localization=True` is passed

### Branch Length Attribute Handling
All branch-length collection code must only consider edges that **actually have** the `"branch_length"` attribute. Never use `.get("branch_length", 0)` — a missing attribute is not the same as zero-length. The correct pattern:
```python
# CORRECT — only edges with the attribute
branch_lengths = [
    tree.edges[p, c]["branch_length"]
    for p, c in tree.edges()
    if "branch_length" in tree.edges[p, c]
]
mean_bl = float(np.mean(branch_lengths)) if branch_lengths else None

# WRONG — conflates "absent" with "zero", produces mean_bl=0 → None
branch_lengths = [tree.edges[p, c].get("branch_length", 0) for p, c in tree.edges()]
```

When `mean_bl` is `None` (no branch lengths in tree), Felsenstein adjustment is skipped entirely — `nx.shortest_path_length(weight="branch_length")` must also be guarded behind `mean_bl is not None` since NetworkX defaults missing weights to 1 (hop count), which would create an inconsistency.

## Bugs Fixed (2026-02-14 Audit)

The following logic errors were identified and corrected:

1. **Cousin-Adjusted Wald Regression Extrapolation** (`cousin_adjusted_wald.py`): The log-linear regression `log(T/k) = β₀ + β₁·log(BL_sum) + β₂·log(n_parent)` extrapolated ĉ far beyond observed data at root nodes, predicting ĉ=3–17× while calibration data showed max ratio ~1.4×. Fixed by clamping `_predict_c()` output at `max_observed_ratio` — the max T/k from null-like pairs. Eliminated K=1 collapses in ~30 benchmark cases.

2. **Node IDs vs Labels in Signal Localization** (`signal_localization.py`): `_get_all_leaves()` returned raw NetworkX node IDs (e.g. `N5`) instead of leaf labels (e.g. `S42`). Fixed to return `tree.nodes[node].get("label", node)`.

3. **Post-hoc Merge Descendant Removal** (`posthoc_merge.py`): When merging at an LCA, all descendant cluster roots must be removed to maintain the antichain invariant. The merge correctly uses `cluster_roots -= nx.descendants(tree, lca)`.

4. **Signal Localization FDR Double-Counting** (`signal_localization.py`): `_recurse()` was appending parent-level results AND recursing deeper, causing the same divergence to be counted at multiple levels. Fixed to `continue` after recursive call so parent results are not duplicated.

5. **Felsenstein Fallback Constant** (`pooled_variance.py`): `standardize_proportion_difference()` silently fell back to `bl_normalized = 2.0` when `mean_branch_length` was missing. Replaced with `ValueError` — callers must supply valid branch lengths.

6. **Redundant `is_different` Lookup** (`tree_decomposition.py`): `_should_split_v2()` performed a redundant dictionary lookup for `is_different` that could silently mask bugs. Removed the unnecessary second lookup.

7. **`.get("branch_length", 0)` Antipattern** (3 files): Mean branch length computation used `.get("branch_length", 0)` which conflates "missing attribute" with "zero-length edge". Fixed in `tree_decomposition.py`, `sibling_divergence_test.py`, and `edge_significance.py` to only collect edges that actually carry the attribute. Shared logic now centralized in `branch_length_utils.py` → `compute_mean_branch_length(tree)`.

8. **`nx.shortest_path_length` Without Guard** (`tree_decomposition.py`): `_test_node_pair_divergence()` and `_test_cluster_pair_divergence()` called `nx.shortest_path_length(weight="branch_length")` even when the tree had no branch lengths. NetworkX defaults missing weights to 1 (hop count), creating `branch_length_sum > 0` with `mean_branch_length = None`, triggering the Felsenstein `ValueError`. Fixed to skip branch length computation when `_mean_branch_length is None`.

9. **`merge_similarity_graphs` p-value direction** (`signal_localization.py`): When duplicate edges existed across localization levels, `merge_similarity_graphs` kept the **lower** p-value. For similarity edges, higher p-value = stronger evidence of similarity (fail to reject H₀). Fixed to keep the **higher** p-value. Previously caused false cross-boundary merges by under-reporting similarity strength.

## Notes

- **Multiple testing**: Uses Benjamini-Hochberg FDR correction across all edges
- **Projection dimension (Gate 2 — edge test)**: When `SPECTRAL_METHOD = "marchenko_pastur"` (default), projection dimension is the per-node count of eigenvalues above the Marchenko-Pastur upper bound, with PCA-based eigenvector projection and eigenvalue whitening for exact χ²(k). The correlation matrix is built from both leaf rows AND internal descendant node distribution vectors to enrich the covariance estimate. Uses **dual-form eigendecomposition** when `n_desc < d_active`: computes `n×n` Gram matrix instead of `d×d` correlation matrix for O(n²d + n³) vs O(d³) — critical for high-d cases (e.g. n=10, d=2000: 10×10 eigh instead of 2000×2000).
- **Projection dimension (Gate 3 — sibling test)**: Uses **min-child spectral k** from Gate 2 output: `k = min(k_left, k_right)` where k_left/k_right are per-node Marchenko-Pastur spectral dimensions. Falls back to JL-based dimension `k ≈ 8·ln(n)/ε²` with **information cap** (when `d ≥ 4n`, k is capped at n_samples) when spectral dims are unavailable. Controlled by `config.PROJECTION_EPS` (default 0.3), `config.PROJECTION_MIN_K` (floor, default 10), and `config.PROJECTION_RANDOM_SEED` (seed, default 42)
- **Branch lengths**: Felsenstein (1985) adjustment scales variance by normalized branch length to account for phylogenetic distance. `mean_branch_length` is computed once from tree edges and threaded through all test functions.
- **Post-hoc merge**: Optional bottom-up merge pass (`posthoc_merge=True`) to reduce over-splitting; returns audit trail. Blocked at LCA boundaries where any pair shows significant difference.
- **Signal localization**: Optional v2 decomposition (`use_signal_localization=True`, default OFF) that drills down to find WHERE divergence originates, enabling cross-boundary partial merges via constrained greedy merge (respects Cannot-Link edges from significant pairs). **Benchmark (2026-02-17, 74 cases)**: v2 Mean ARI 0.431 vs v1's 0.757 — localization sub-tests lack power (small sub-samples + BH penalty → false similarity edges → incorrect merges). v2 does improve phylogenetic cases (phylo_divergent_8taxa: v2 ARI=1.0, K=8 vs v1 K=72 over-split). v2 also has computational cost issues (combinatorial cross-boundary tests make large K cases very slow). Kept as opt-in experimental feature.
- **Node naming**: Leaves are `L{idx}`, internal nodes are `N{idx}` in PosetTree
- **Distance metric**: Default is `"hamming"` (`config.TREE_DISTANCE_METRIC`); `"rogerstanimoto"` available via method registry as `"kl_rogerstanimoto"`
- **Sibling test toggle**: `config.SIBLING_TEST_METHOD` controls which Gate 3 implementation is used. Default: `"cousin_weighted_wald"`. Options: `"wald"`, `"cousin_ftest"`, `"cousin_adjusted_wald"`, `"cousin_tree_guided"`, `"cousin_weighted_wald"`. `PosetTree.decompose()` dispatches to the corresponding annotation function at runtime.
- **Branch length utilities**: `branch_length_utils.py` centralizes `compute_mean_branch_length(tree)` and `sanitize_positive_branch_length(value)`. Imported by `tree_decomposition.py` and `edge_significance.py` to avoid duplicated branch-length logic.
- **Projected p-value helper**: `projection/satterthwaite.py` provides `compute_projected_pvalue(projected, df, eigenvalues)` — the shared implementation of whitened and Satterthwaite χ² approximation. Used by both `edge_significance.py` (Gate 2) and `sibling_divergence_test.py` (Gate 3) to eliminate code duplication.

## Benchmark Rules (Known Structural Issues)

**FIXED — SBM distance bypass**: The KL runner now uses the pre-computed modularity `distance_condensed` for SBM cases. If `distance_condensed` is `None` for an SBM case, a `ValueError` is raised — there is no silent fallback to `pdist()` on raw adjacency data.

**BUG — Categorical/phylogenetic type mismatch**: Categorical generators return integer category indices (0 to K-1). The Bernoulli pipeline interprets these as probabilities, producing invalid KL divergences (values >1 break `p·log(p/q) + (1-p)·log((1-p)/(1-q))`). This is the root cause of phylogenetic over-splitting (72 clusters instead of 8). Fix requires one-hot encoding before `decompose()` or a dedicated categorical KL path.

**DESIGN — Gradient templates**: `_create_gradient_templates()` assigns `prob_ones = cluster_id / (n_clusters - 1)`. For k>4 clusters adjacent templates differ by ≈1/k fraction of features — below noise floor after bit-flip. Prefer block-diagonal / feature-ownership templates (like `_create_sparse_templates`) for k>4.

**DESIGN — Median binarization power**: `(X > median(X, axis=0)).astype(int)` forces every column to θ≈0.5, maximizing the Wald variance denominator θ(1-θ)=0.25 and minimizing test power. Signal still exists conditionally (per-cluster θ deviates from 0.5) but is attenuated.

**DESIGN — Minimum sample size**: Gaussian cases with n/K < 20 (e.g. n=30, K=3 → 10/cluster) are below the practical power threshold for the projected Wald χ² test.

**CODE — Global RNG**: `generate_random_feature_matrix` calls `np.random.seed(seed)` which sets global state. Use `rng = np.random.default_rng(seed)` and thread `rng` through helpers to avoid cross-call leakage.
