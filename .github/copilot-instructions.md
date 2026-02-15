# KL-Divergence Hierarchical Clustering Toolkit

## Architecture Overview

This toolkit decomposes hierarchical clustering trees into statistically stable partitions using information-theoretic tests. The main flow:

```
Binary matrix → linkage → PosetTree → populate_node_divergences() → decompose() → cluster assignments
```

### Core Components

- **`PosetTree`** ([tree/poset_tree.py](kl_clustering_analysis/tree/poset_tree.py)): NetworkX DiGraph subclass; central data structure holding hierarchy, distributions, and statistical annotations via `stats_df` property. Constructed via `from_linkage()` or `from_agglomerative()`.

- **`TreeDecomposition`** ([hierarchy_analysis/tree_decomposition.py](kl_clustering_analysis/hierarchy_analysis/tree_decomposition.py)): Top-down tree walker applying three statistical gates to determine cluster boundaries:
  1. **Binary structure gate**: Parent must have exactly two children
  2. **Child-parent divergence gate**: Projected Wald chi-square test on KL(child || parent)
  3. **Sibling divergence gate**: Configurable via `config.SIBLING_TEST_METHOD` — one of `"wald"`, `"cousin_ftest"`, or `"cousin_adjusted_wald"` (see Sibling Test Methods below)

  Also provides `decompose_tree_v2()` with **signal localization** (`use_signal_localization=True`), which drills into WHERE divergence originates and enables cross-boundary partial merges.

- **`SignalLocalization`** ([hierarchy_analysis/signal_localization.py](kl_clustering_analysis/hierarchy_analysis/signal_localization.py)): Recursive cross-boundary testing to identify which subtrees are truly different vs. which could be merged, producing `SimilarityEdge` / `LocalizationResult` objects.

- **`config.py`** ([kl_clustering_analysis/config.py](kl_clustering_analysis/config.py)): Central configuration for thresholds (alpha levels, projection params, linkage method). Key constants:
  - `SIGNIFICANCE_ALPHA = 0.05` — edge-test (Gate 2) significance level
  - `SIBLING_ALPHA = 0.05` — sibling-test (Gate 3) significance level
  - `ALPHA_LOCAL = 0.05` — local significance level passed to decomposer
  - `SIBLING_TEST_METHOD = "cousin_adjusted_wald"` — Gate 3 implementation toggle
  - `TREE_DISTANCE_METRIC = "hamming"` — distance metric for linkage
  - `TREE_LINKAGE_METHOD = "average"` — linkage method
  - `POSTHOC_MERGE = True` — enable/disable bottom-up merge pass
  - `POSTHOC_MERGE_ALPHA = None` — override alpha for post-hoc merge (defaults to `SIBLING_ALPHA`)
  - `USE_SIGNAL_LOCALIZATION = False` — enable v2 decomposition with signal localization
  - `PROJECTION_EPS = 0.3` — JL-lemma epsilon for projection dimension
  - `PROJECTION_MIN_K = 10` — minimum projection dimension
  - `PROJECTION_RANDOM_SEED = 42` — seed for random projection matrix
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

The test suite has **138 tests** across 26 test files. Key test files:
- `test_clt_validity.py` (40 tests) — CLT validity checks for statistical tests
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
| `hierarchy_analysis/` | Tree decomposition, post-hoc merge, signal localization |
| `hierarchy_analysis/statistics/` | Statistical tests (KL chi-square, sibling divergence, multiple testing, pooled variance, branch_length_utils, CLT validity, power analysis) |
| `hierarchy_analysis/statistics/sibling_divergence/` | Sibling divergence test implementations: `sibling_divergence_test.py` (Wald), `cousin_calibrated_test.py` (F-test), `cousin_adjusted_wald.py` (adjusted Wald) |
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
# Full benchmark (95 test cases, 2 methods: kl + kl_rogerstanimoto)
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
- Projection: `T = ||R·z||² ~ χ²(k)`, k from JL lemma
- FDR: `tree_bh` (default), `flat`, or `level_wise` correction
- If **neither** child diverges → MERGE (noise, no signal to split on)

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

Three sibling test implementations are available, toggled via `config.SIBLING_TEST_METHOD`:

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

#### 3. `"cousin_adjusted_wald"` — Cousin-Adjusted Wald (current default)
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
- `Sibling_Divergence_Skipped` is `True` for nodes where the sibling test was skipped (insufficient samples, no child-parent signal) — these nodes are treated as MERGE decisions
- `_should_split` (v1) and `_should_split_v2` return `False` (not raise) when encountering skipped nodes. Both apply identical gate logic; v2 additionally runs signal localization when Gate 3 passes.
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

## Notes

- **Multiple testing**: Uses Benjamini-Hochberg FDR correction across all edges
- **Projection**: Random projection is integral to the Wald chi-square test and always applied; dimension k is computed adaptively via the JL lemma (`compute_projection_dimension`). Controlled by `config.PROJECTION_EPS` (JL epsilon, default 0.3), `config.PROJECTION_MIN_K` (floor, default 10), and `config.PROJECTION_RANDOM_SEED` (seed, default 42)
- **Branch lengths**: Felsenstein (1985) adjustment scales variance by normalized branch length to account for phylogenetic distance. `mean_branch_length` is computed once from tree edges and threaded through all test functions.
- **Post-hoc merge**: Optional bottom-up merge pass (`posthoc_merge=True`) to reduce over-splitting; returns audit trail. Blocked at LCA boundaries where any pair shows significant difference.
- **Signal localization**: Optional v2 decomposition (`use_signal_localization=True`) that drills down to find WHERE divergence originates, enabling cross-boundary partial merges via constrained greedy merge (respects Cannot-Link edges from significant pairs)
- **Node naming**: Leaves are `L{idx}`, internal nodes are `N{idx}` in PosetTree
- **Distance metric**: Default is `"hamming"` (`config.TREE_DISTANCE_METRIC`); `"rogerstanimoto"` available via method registry as `"kl_rogerstanimoto"`
- **Sibling test toggle**: `config.SIBLING_TEST_METHOD` controls which Gate 3 implementation is used. Default: `"cousin_adjusted_wald"`. Options: `"wald"`, `"cousin_ftest"`, `"cousin_adjusted_wald"`. `PosetTree.decompose()` dispatches to the corresponding annotation function at runtime.
- **Branch length utilities**: `branch_length_utils.py` centralizes `compute_mean_branch_length(tree)` and `sanitize_positive_branch_length(value)`. Imported by `tree_decomposition.py` and `edge_significance.py` to avoid duplicated branch-length logic.

## Benchmark Rules (Known Structural Issues)

**FIXED — SBM distance bypass**: The KL runner now uses the pre-computed modularity `distance_condensed` for SBM cases. If `distance_condensed` is `None` for an SBM case, a `ValueError` is raised — there is no silent fallback to `pdist()` on raw adjacency data.

**BUG — Categorical/phylogenetic type mismatch**: Categorical generators return integer category indices (0 to K-1). The Bernoulli pipeline interprets these as probabilities, producing invalid KL divergences (values >1 break `p·log(p/q) + (1-p)·log((1-p)/(1-q))`). This is the root cause of phylogenetic over-splitting (72 clusters instead of 8). Fix requires one-hot encoding before `decompose()` or a dedicated categorical KL path.

**DESIGN — Gradient templates**: `_create_gradient_templates()` assigns `prob_ones = cluster_id / (n_clusters - 1)`. For k>4 clusters adjacent templates differ by ≈1/k fraction of features — below noise floor after bit-flip. Prefer block-diagonal / feature-ownership templates (like `_create_sparse_templates`) for k>4.

**DESIGN — Median binarization power**: `(X > median(X, axis=0)).astype(int)` forces every column to θ≈0.5, maximizing the Wald variance denominator θ(1-θ)=0.25 and minimizing test power. Signal still exists conditionally (per-cluster θ deviates from 0.5) but is attenuated.

**DESIGN — Minimum sample size**: Gaussian cases with n/K < 20 (e.g. n=30, K=3 → 10/cluster) are below the practical power threshold for the projected Wald χ² test.

**CODE — Global RNG**: `generate_random_feature_matrix` calls `np.random.seed(seed)` which sets global state. Use `rng = np.random.default_rng(seed)` and thread `rng` through helpers to avoid cross-call leakage.
