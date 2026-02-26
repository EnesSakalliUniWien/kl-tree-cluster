# KL-TE Clustering Benchmarks

This directory contains the benchmark infrastructure for KL-Divergence Hierarchical Clustering.

## Quick Start

```bash
# Run full benchmark (96 cases, all methods)
python benchmarks/full/run.py

# Run specific benchmark suites
python benchmarks/branch_length/run.py
python benchmarks/branch_length_3d/run.py
python benchmarks/multi_split/run.py

# Quick subset for fast iteration (~15 cases)
python benchmarks/run_subset.py

# Compare sibling test methods head-to-head
python benchmarks/compare_sibling_methods.py

# Real-world datasets (MNIST, Penguins, Digits)
python benchmarks/mnist/run.py
python benchmarks/umap_datasets/run.py

# Run statistical calibration suite (null/Type-I + TreeBH)
python benchmarks/calibration/run.py
```

## Architecture: Single Shared System

All benchmark code should import from `benchmarks.shared.*`.

### Shared Features

The `benchmarks/shared/` system provides:
- `matrix_audit` parameter for TensorBoard-style matrix exports
- `debug_trace.py` for failure diagnosis  
- `evolution.py` for temporal evolution tracking
- `audit_utils.py` for matrix export infrastructure
- `linkage_matrix` captured in KL runner's `extra` dict

## Key Files

| File                       | Purpose                                                         |
| -------------------------- | --------------------------------------------------------------- |
| `shared/cases/__init__.py` | All test case definitions (Gaussian, binary, SBM, phylogenetic) |
| `shared/runner.py`         | `run_benchmark()` - main entry point for all suites             |
| `shared/kl_runner.py`      | KL-specific runner with SBM `distance_condensed` handling       |
| `shared/metrics.py`        | ARI, NMI, Purity calculations                                   |
| `shared/generators.py`     | Data generators (phylogenetic, Gaussian, etc.)                  |

## Running Benchmarks

### Critical: Run from Project Root

All benchmark scripts **must** be run from the project root:

```bash
# Correct
python benchmarks/full/run.py

# Wrong - don't cd into benchmarks/
cd benchmarks/full && python run.py  # Will fail on imports
```

### Output Structure

Results saved to timestamped directories under each benchmark's `results/` folder:
```
benchmarks/full/results/run_YYYYMMDD_HHMMSSZ/
├── full_benchmark_comparison.csv    # Main results
├── failure_report.md              # Failed cases analysis
├── audit/                         # Matrix exports (if enabled)
└── plots/
    └── case_N.pdf                 # Per-case visualizations
```

## Benchmark Types

### 1. Full Suite ([full/](full/))

**Purpose**: Canonical end-to-end benchmark — runs the complete test case suite and compares all clustering methods.

**Data generation**: 96 test cases across 7 categories (see [Test Case Categories](#test-case-categories) below). Each case specifies a generator, sample count, feature count, cluster count, and noise level. The dispatcher (`generate_case_data`) routes to the appropriate generator, binarizes or one-hot-encodes as needed, and feeds the resulting binary matrix to each clustering method.

**Experiment setup**:

- Methods: up to 13 (KL variants + Leiden, Louvain, K-Means, Spectral, DBSCAN, OPTICS, HDBSCAN). Configurable via `KL_TE_METHODS` env var.
- Each case runs in an isolated subprocess (optional), with configurable timeout (default 1800 s) and retry count (default 4).
- Per-case PDF plots (tree, UMAP embedding, manifold comparison) are generated and merged into `full_benchmark_report.pdf`.

**Evaluation**: ARI, NMI, Purity, Exact-K match. Results saved to `full_benchmark_comparison.csv` with a `failure_report.md` for cases that error or time out.

---

### 2. Branch Length ([branch_length/](branch_length/))

**Purpose**: Measures how clustering performance degrades as evolutionary divergence increases between two groups.

**Data generation**: Two-group Jukes–Cantor substitution model. A root sequence of length `n_features` (default 200) over `n_categories` states (default 4) is generated uniformly. Group 1 keeps the root; Group 2 evolves along a branch of variable length. Within-group variation is added via short terminal branches (default 0.05). Total samples: `2 × n_leaves/2` (default 200).

**Experiment setup**:

- Sweeps over 16 branch lengths from 0.01 (nearly identical) to 2.0 (saturated divergence).
- 5 replicates per branch length (different random seeds).
- KL method only (`hamming` + `average`).

**Evaluation**: ARI and NMI plotted as a function of branch length, with mean ± std error bands across replicates. Produces a performance curve showing the method's sensitivity range.

---

### 3. Branch Length 3D ([branch_length_3d/](branch_length_3d/))

**Purpose**: Extends the branch-length benchmark to a 2D parameter sweep — varies both evolutionary divergence and the number of features simultaneously to create a performance surface.

**Data generation**: Same Jukes–Cantor model as Branch Length, but with a grid of `(branch_length, n_features)` combinations.

**Experiment setup**:

- Branch lengths: 9 values from 1 to 30.
- Feature counts: 6 values from 50 to 500.
- 3 replicates per grid cell.
- Fixed `n_leaves=200`, `n_categories=4`.

**Evaluation**: ARI surface plotted as a 3D heatmap (branch length × features × ARI). Shows the joint effect of signal strength and dimensionality on clustering accuracy.

---

### 4. Multi-Split ([multi_split/](multi_split/))

**Purpose**: Tests the method's ability to recover the correct number of clusters (K) in a balanced star phylogeny.

**Data generation**: Star-topology tree where all K groups diverge from a common ancestor. A root sequence (`n_features=200`, `n_categories=4`) is evolved along `between_group_branch=0.3` to create K group ancestors, then each ancestor generates `n_total/K` samples via short terminal branches (`within_group_branch=0.05`).

**Experiment setup**:

- Sweeps K over {2, 4, 6, 7, 8, 10, 12} groups.
- Fixed `n_total_samples=200` (divided equally among groups).
- 10 replicates per K.
- KL method only.

**Evaluation**: K-recovery accuracy (found K vs. true K), ARI, NMI. Plotted as a function of true K. Tests whether the statistical gates correctly identify all split points in a balanced tree.

---

### 5. Quick Subset ([run_subset.py](run_subset.py))

**Purpose**: Fast iteration benchmark — runs ~15 representative cases from the full suite for quick validation during development.

**Data generation**: Draws from the same 96-case pool used by the full suite. Hand-picked subset covers Gaussian (easy/moderate), Binary (perfect/noisy/sparse), Categorical, SBM, Overlapping, and Real Data categories.

**Experiment setup**:

- ~15 cases, KL method only.
- Plots enabled (UMAP comparison pages).
- Uses `benchmark_cluster_algorithm()` from `shared/pipeline.py`.

**Evaluation**: Same metrics as full suite (ARI, NMI, Exact K). Printed as a summary table to stdout.

---

### 6. Compare Sibling Methods ([compare_sibling_methods.py](compare_sibling_methods.py))

**Purpose**: Head-to-head comparison of Gate 3 (sibling divergence) calibration methods — `cousin_adjusted_wald` vs. `cousin_weighted_wald`.

**Data generation**: Uses the full 96-case suite. Cases with >700 samples are skipped for runtime.

**Experiment setup**:

- Runs each case in a subprocess for crash isolation (120 s timeout per case).
- For each case, toggles `config.SIBLING_TEST_METHOD` between the two methods and runs the full KL pipeline from scratch (distance → linkage → tree → decompose).
- All other config parameters held constant.

**Evaluation**: Per-case K-recovered and ARI for each method. Summary statistics (mean ARI, exact K, win/loss count) printed as a comparison table. Identifies cases where the two methods disagree.

---

### 7. MNIST ([mnist/](mnist/))

**Purpose**: Real-world benchmark on handwritten digit images (10 classes).

**Data generation**: Downloads a random subset of MNIST via `sklearn.datasets.fetch_openml` (default 1000 samples, 784 features). Pixel intensities are normalized to [0, 1], then binarized with threshold 0.1 (dark pixels → 0, light pixels → 1). Optional PCA pre-reduction to 50 components.

**Experiment setup**:

- Distance metric: `rogerstanimoto` (double-weights mismatches, better for sparse binary images).
- Linkage: `average`.
- α = 0.05 for both edge and sibling tests.

**Evaluation**: ARI, NMI, K-found vs. K-true (10 digits). Results saved to CSV and printed as a summary. Tests the method on high-dimensional, real-world data where cluster boundaries are not perfectly separable.

---

### 8. UMAP Datasets ([umap_datasets/](umap_datasets/))

**Purpose**: Benchmarks on the standard datasets featured in the UMAP documentation — Palmer Penguins and Sklearn Digits.

**Data generation**:

- **Palmer Penguins**: 333 samples × 4 numeric features (bill length/depth, flipper length, body mass), 3 species. Continuous features are discretized into 5 ordinal bins via `KBinsDiscretizer`, then treated as binary.
- **Sklearn Digits**: 1797 samples × 64 features (8×8 pixel images), 10 digit classes. Pixel values are binarized at threshold.

**Experiment setup**:

- KL method with `hamming` + `average`.
- Generates interactive Bokeh HTML plots for per-dataset embedding visualization.
- Results saved to timestamped output directory.

**Evaluation**: ARI, NMI, K-found vs. K-true. Per-dataset PDF and interactive HTML plots.

---

### 9. Calibration ([calibration/](calibration/))

**Purpose**: Empirical statistical calibration — verifies that the edge and sibling tests maintain correct Type I error rates under the null hypothesis, and that TreeBH FDR control is valid.

**Data generation**:

- **Null scenarios**: Pure-noise binary matrices with `p_one=0.5` (no cluster structure). Three sizes: 64×32, 128×64, 192×96.
- **TreeBH scenarios**: Tree-structured synthetic p-values with known alt/null partition. Varying alt fractions (0–50%) and signal strengths (beta distribution shape parameter 0.2–0.5).

**Experiment setup**:

- Null calibration: 30 replicates per scenario (configurable via `KL_TE_CAL_NULL_REPS`). Full pipeline runs (linkage → tree → decompose) on each replicate.
- TreeBH calibration: 200 replicates per scenario (configurable via `KL_TE_CAL_TREEBH_REPS`).
- Also includes crossfit permutation diagnostics.

**Evaluation**: Per-scenario Type I error rates with 95% binomial CIs. Edge and sibling rejection rates compared to nominal α = 0.05. TreeBH FDR and power summaries. Results written to CSV + `calibration_plots.pdf` + `calibration_report.md`.

---

## Test Case Categories

The full suite's 96 cases are organized into 7 categories:

### Gaussian (18 cases)

Generated via `sklearn.make_blobs` with configurable `cluster_std`, then **median-binarized** per feature: `(X > median(X, axis=0)).astype(int)`. Tests the algorithm on data where cluster structure originates from continuous Gaussian blobs but is observed through a lossy binary lens.

| Subcategory              | Cases | n_samples | n_features | K    | cluster_std |
| ------------------------ | ----- | --------- | ---------- | ---- | ----------- |
| `gaussian_clear`         | 3     | 30–50     | 30–50      | 3–5  | 0.5–1.0     |
| `gaussian_mixed`         | 5     | 25–65     | 25–65      | 2–6  | 0.7–2.0     |
| `gaussian_extreme_noise` | 3     | 30–300    | 30–20000   | 3–30 | 2.0–7.5     |
| `improved_gaussian`      | 7     | 30–120    | 20–80      | 3–8  | 0.5–2.5     |

### Binary (21 cases)

Generated directly as binary {0,1} matrices via `generate_random_feature_matrix`. Each cluster owns a distinctive subset of features with controlled bit-flip probabilities. `entropy_param` controls noise (0 = perfect separation, 0.5 = random). This is the most natural input format for the Bernoulli KL pipeline.

| Subcategory                  | Cases | n_rows  | n_cols  | K    | entropy   |
| ---------------------------- | ----- | ------- | ------- | ---- | --------- |
| `binary_balanced_low_noise`  | 2     | 72      | 72–120  | 4    | 0.25      |
| `binary_sparse_features`     | 2     | 72–100  | 72–500  | 4    | 0.10      |
| `improved_binary_perfect`    | 3     | 40–160  | 50–200  | 2–8  | 0.00      |
| `improved_binary_low_noise`  | 4     | 40–240  | 50–300  | 2–12 | 0.05–0.10 |
| `improved_binary_moderate`   | 3     | 80–200  | 100–250 | 4–8  | 0.12–0.15 |
| `improved_binary_hard`       | 2     | 100–280 | 200–400 | 4–8  | 0.15–0.20 |
| `improved_binary_unbalanced` | 2     | 100–150 | 150–200 | 4–6  | 0.10–0.12 |
| `improved_binary_edge_cases` | 3     | 50–300  | 60–2000 | 2–15 | 0.10      |

### SBM (3 cases)

Generated via Stochastic Block Model (`generate_sbm`). The adjacency matrix is converted to a **modularity-based distance** (`1 - B_norm` where `B = A - ddᵀ/2m`). This pre-computed distance is passed directly to the KL runner — there is no fallback to `pdist()` on raw adjacency data.

| Cases | Sizes (nodes)      | p_intra   | p_inter    | K   |
| ----- | ------------------ | --------- | ---------- | --- |
| 3     | [30,30]–[50,40,30] | 0.05–0.12 | 0.005–0.04 | 2–3 |

### Categorical (11 cases)

Generated via `generate_categorical_feature_matrix` as integer category indices (0 to K−1), then **one-hot encoded** into `(n_rows × n_cols × n_categories)` binary indicators before the KL pipeline. Tests the algorithm's handling of multi-valued features.

| Subcategory                    | Cases | n_rows  | n_cols   | K   | n_categories | entropy   |
| ------------------------------ | ----- | ------- | -------- | --- | ------------ | --------- |
| `categorical_clear`            | 3     | 100–150 | 50–80    | 4–6 | 3–5          | 0.05–0.08 |
| `categorical_moderate`         | 2     | 120–180 | 60–100   | 4–6 | 3–4          | 0.15–0.18 |
| `categorical_high_cardinality` | 2     | 200     | 40–50    | 4   | 10–20        | 0.10–0.12 |
| `categorical_unbalanced`       | 1     | 150     | 60       | 4   | 3            | 0.12      |
| `categorical_overlapping`      | 1     | 400     | 100      | 4   | 3            | 0.35      |
| `categorical_high_dimensional` | 2     | 200–300 | 500–1000 | 4–6 | 3–4          | 0.12–0.15 |

### Phylogenetic (13 cases)

Generated via `generate_phylogenetic_data` — simulates trait evolution along a random phylogenetic tree using a Jukes–Cantor-like substitution model. Each taxon is a cluster; `samples_per_taxon` samples are drawn from the evolved distribution at each leaf. Category-index matrix is **one-hot encoded** before the KL pipeline.

| Subcategory              | Cases | n_taxa (=K) | n_features | n_categories | samples/taxon | mutation_rate |
| ------------------------ | ----- | ----------- | ---------- | ------------ | ------------- | ------------- |
| `phylogenetic_dna`       | 4     | 4–16        | 100–500    | 4            | 15–25         | 0.2–0.4       |
| `phylogenetic_protein`   | 3     | 4–12        | 50–150     | 20           | 15–30         | 0.3–0.4       |
| `phylogenetic_divergent` | 2     | 4–8         | 100–200    | 4            | 20–25         | 0.7–0.8       |
| `phylogenetic_conserved` | 2     | 4–8         | 100–200    | 4            | 20–25         | 0.05–0.08     |
| `phylogenetic_large`     | 2     | 32–64       | 500–1000   | 4            | 8–10          | 0.3–0.35      |

**Known issue**: Severe over-splitting on many phylogenetic cases due to near-zero branch lengths in the constructed clustering tree (see [Known Issues](#known-issues--structural-problems)).

### Overlapping (29 cases)

Binary subcategories use the `binary` generator with high `entropy_param` (0.22–0.48) to create clusters whose feature profiles significantly overlap. Gaussian subcategory uses `blobs` with high `cluster_std` (3.0–6.0). Tests the algorithm's ability to correctly **merge** overlapping groups rather than over-split.

| Subcategory                     | Cases | n_samples | n_features | K    | Noise param       |
| ------------------------------- | ----- | --------- | ---------- | ---- | ----------------- |
| `overlapping_binary_heavy`      | 5     | 500–600   | 50–1000    | 4–6  | entropy 0.40–0.48 |
| `overlapping_binary_moderate`   | 4     | 400–1000  | 80–800     | 4–10 | entropy 0.28–0.32 |
| `overlapping_binary_partial`    | 4     | 400–1000  | 60–600     | 4–10 | entropy 0.22–0.26 |
| `overlapping_binary_highd`      | 4     | 500–1000  | 1000–5000  | 4–10 | entropy 0.28–0.35 |
| `overlapping_binary_unbalanced` | 4     | 400–1000  | 100–1000   | 4–10 | entropy 0.28–0.35 |
| `overlapping_gaussian`          | 8     | 300–1000  | 30–300     | 3–10 | std 3.0–6.0       |

### Real Data (1 case)

Loads `feature_matrix.tsv` from the repo root — a pre-existing binary GO-term feature matrix. No ground-truth labels; used for qualitative evaluation only.

---

## Evaluation Methods

All benchmarks (except calibration) evaluate clustering quality using:

| Metric                                  | Range   | Interpretation                                               |
| --------------------------------------- | ------- | ------------------------------------------------------------ |
| **ARI** (Adjusted Rand Index)           | [−1, 1] | 1 = perfect, 0 = random, negative = worse than random        |
| **NMI** (Normalized Mutual Information) | [0, 1]  | 1 = perfect correspondence between predicted and true labels |
| **Purity**                              | [0, 1]  | Fraction of samples in the dominant true class per cluster   |
| **Exact K**                             | count   | Number of cases where found K equals true K                  |

Note: K-Means and Spectral Clustering are given the **true K** as input, making them oracle baselines rather than fully unsupervised competitors.

## Clustering Methods (13 methods)

| Key                 | Name                 | Distance          | Linkage  | Notes                               |
| ------------------- | -------------------- | ----------------- | -------- | ----------------------------------- |
| `kl`                | KL Divergence        | hamming           | average  | Default — binary-native             |
| `kl_complete`       | KL (Complete)        | hamming           | complete | Complete-linkage variant            |
| `kl_single`         | KL (Single)          | hamming           | single   | Single-linkage variant              |
| `kl_ward`           | KL (Ward)            | euclidean         | ward     | Ward requires Euclidean distance    |
| `kl_rogerstanimoto` | KL (Rogers-Tanimoto) | rogerstanimoto    | average  | Double-weights mismatches           |
| `kl_v2`             | KL v2                | hamming           | average  | Experimental signal localization    |
| `leiden`            | Leiden               | KNN graph         | —        | Community detection, resolution=1.0 |
| `louvain`           | Louvain              | KNN graph         | —        | Community detection, resolution=1.0 |
| `kmeans`            | K-Means              | —                 | —        | **Oracle**: uses true K             |
| `spectral`          | Spectral             | nearest_neighbors | —        | **Oracle**: uses true K             |
| `dbscan`            | DBSCAN               | —                 | —        | Density-based, auto eps             |
| `optics`            | OPTICS               | —                 | —        | Density-based, xi=0.05              |
| `hdbscan`           | HDBSCAN              | —                 | —        | Density-based, min_cluster=5        |

## Adding New Benchmark Suites

### 1. Create Directory Structure

```bash
mkdir benchmarks/my_suite
mkdir benchmarks/my_suite/results
```

### 2. Create cases.py

```python
from benchmarks.shared.cases import BASE_CASES

MY_CASES = [
    {
        "name": "my_custom_case",
        "generator": "gaussian",
        "n_samples": 200,
        "n_features": 30,
        "n_clusters": 3,
        "cluster_std": 1.0,
        "random_state": 42,
    },
]

ALL_CASES = BASE_CASES + MY_CASES
```

### 3. Create run.py

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmarks.shared.runner import run_benchmark
from benchmarks.my_suite.cases import ALL_CASES

if __name__ == "__main__":
    results_df, fig = run_benchmark(
        test_cases=ALL_CASES,
        methods=["kl", "kl_rogerstanimoto"],
        output_dir=Path(__file__).parent / "results"
    )
```

### SBM-Specific: Pre-computed Distance Required

SBM cases **require** pre-computed modularity distance:
```python
# In kl_runner.py - distance_condensed is REQUIRED for SBM
if case["generator"] == "sbm":
    if distance_condensed is None:
        raise ValueError("SBM cases require pre-computed distance_condensed")
```

There is **no fallback** to `pdist()` on raw adjacency data.

## Known Issues & Structural Problems

### Broken/Removed Scripts

- **`benchmarks/temporal/`**: Entirely removed. Had import errors (`run_incremental_temporal_benchmark` doesn't exist) and `temporal_evolution` generator never wired into pipeline.

### Critical Issues

1. **Categorical/Phylogenetic Type Mismatch**  
   Categorical generators return integer indices (0 to K-1). Bernoulli pipeline interprets these as probabilities → invalid KL divergences. **Root cause** of phylogenetic over-splitting (72 clusters instead of 8). Fix requires one-hot encoding or dedicated categorical KL path.

2. **Gradient Templates Design Flaw**  
   `_create_gradient_templates()` assigns `prob_ones = cluster_id / (n_clusters - 1)`. For k>4, adjacent templates differ by ≈1/k fraction — below noise floor after bit-flip. Prefer `_create_sparse_templates` for k>4.

3. **Median Binarization Power Loss**  
   `(X > median(X, axis=0)).astype(int)` forces θ≈0.5, maximizing Wald variance denominator θ(1-θ)=0.25 and minimizing test power.

4. **Small Sample Size Failures**  
   Gaussian cases with n/K < 20 (e.g., n=30, K=3 → 10/cluster) are below practical power threshold for projected Wald χ² test. Explains K=1 under-splitting (`gauss_clear_medium`, `binary_perfect_2c`).

## Debugging Failed Cases

### Enable Debug Trace

```python
# In your run.py
results_df, fig = run_benchmark(
    test_cases=CASES,
    methods=["kl"],
    debug_trace=True,  # Detailed per-case logging
)
```

### Common Failure Patterns

| Symptom                      | Likely Cause                              | Investigation                              |
| ---------------------------- | ----------------------------------------- | ------------------------------------------ |
| K=1 (under-splitting)        | n/K < 20, or signal below noise floor     | Check `n_samples / n_clusters` ratio       |
| K>>expected (over-splitting) | Categorical data without one-hot encoding | Verify data type going into `decompose()`  |
| All cases K=1                | Branch length handling bug                | Check `mean_branch_length` computation     |
| SBM failures                 | Missing `distance_condensed`              | Verify case data has pre-computed distance |

### Analyze Specific Case

See `debug_scripts/analyze_case_*.py` for patterns:
```python
from benchmarks.shared.cases import CASES
from benchmarks.shared.generators import generate_gaussian
from kl_clustering_analysis import config

# Adjust thresholds for sensitivity analysis
config.SIGNIFICANCE_ALPHA = 0.05
```

## Latest Results (2026-02-14, 95 cases × 2 methods)

| Metric     | `kl` (hamming) | `kl_rogerstanimoto` |
| ---------- | -------------- | ------------------- |
| Mean ARI   | 0.757          | 0.759               |
| Median ARI | 1.000          | 1.000               |
| Exact K    | 59/95          | 61/95               |
| K=1 cases  | 10             | 11                  |

**Key Finding**: `kl` (hamming) outperforms `kl_rogerstanimoto` across all metrics for binarized data.

## Integration with Main Library

The benchmark system uses the core library:
```python
from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis import config

# Adjust config for benchmark runs
config.SIGNIFICANCE_ALPHA = 0.05
config.SIBLING_ALPHA = 0.05
config.TREE_DISTANCE_METRIC = "hamming"  # or "rogerstanimoto"
```

## Calibration Integration in Full Benchmark PDF

`benchmarks/full/run.py` can append calibration pages to `full_benchmark_report.pdf`.

Enable with environment flags:

```bash
KL_TE_RUN_CALIBRATION=1 \
KL_TE_CAL_NULL_REPS=30 \
KL_TE_CAL_TREEBH_REPS=200 \
python benchmarks/full/run.py
```

Outputs are written under the current run directory:
`benchmarks/results/run_<timestamp>/calibration/`
