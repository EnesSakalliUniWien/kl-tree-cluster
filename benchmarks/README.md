# KL-TE Clustering Benchmarks

This directory contains the benchmark infrastructure for KL-Divergence Hierarchical Clustering.

## Quick Start

```bash
# Run full benchmark (74-108 cases, ~2 methods)
python benchmarks/full/run.py

# Run specific benchmark suites
python benchmarks/branch_length/run.py
python benchmarks/branch_length_3d/run.py
python benchmarks/multi_split/run.py
python benchmarks/phylogenetic/run.py

# Run final curated suite
python benchmarks/run_final.py
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

| File | Purpose |
|------|---------|
| `shared/cases/__init__.py` | All test case definitions (Gaussian, binary, SBM, phylogenetic) |
| `shared/runner.py` | `run_benchmark()` - main entry point for all suites |
| `shared/kl_runner.py` | KL-specific runner with SBM `distance_condensed` handling |
| `shared/metrics.py` | ARI, NMI, Purity calculations |
| `shared/generators.py` | Data generators (phylogenetic, Gaussian, etc.) |

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

### 1. Branch Length ([branch_length/](branch_length/))
Measures clustering performance (ARI/NMI) as a function of evolutionary divergence between two groups (fixed leaves).

### 2. Branch Length 3D ([branch_length_3d/](branch_length_3d/))
Varies both branch length and number of features to create a performance surface.

### 3. Multi-Split ([multi_split/](multi_split/))
Tests ability to recover correct number of clusters (K) in star phylogeny.

### 4. Phylogenetic ([phylogenetic/](phylogenetic/))
Tests on categorical phylogenetic data (currently has over-splitting issues - see Known Issues).

### 5. UMAP Datasets ([umap_datasets/](umap_datasets/))
Benchmarks on standard datasets (Palmer Penguins, Sklearn Digits) from UMAP documentation.

### 6. MNIST ([mnist/](mnist/))
Evaluates on subset of MNIST handwritten digits.

### 7. Full Suite ([full/](full/))
Runs complete test case suite (74-108 cases across multiple data types).

## Test Case Patterns

### Defining a Case

```python
# In shared/cases/__init__.py or suite-specific cases.py
{
    "name": "gaussian_clear_large",
    "generator": "gaussian",           # or "binary", "sbm", "phylogenetic"
    "n_samples": 500,
    "n_features": 50,
    "n_clusters": 5,
    "cluster_std": 0.5,
    "random_state": 42,
}
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

| Symptom | Likely Cause | Investigation |
|---------|--------------|---------------|
| K=1 (under-splitting) | n/K < 20, or signal below noise floor | Check `n_samples / n_clusters` ratio |
| K>>expected (over-splitting) | Categorical data without one-hot encoding | Verify data type going into `decompose()` |
| All cases K=1 | Branch length handling bug | Check `mean_branch_length` computation |
| SBM failures | Missing `distance_condensed` | Verify case data has pre-computed distance |

### Analyze Specific Case

See `debug_scripts/analyze_case_*.py` for patterns:
```python
from benchmarks.shared.cases import CASES
from benchmarks.shared.generators import generate_gaussian
from kl_clustering_analysis import config

# Adjust thresholds for sensitivity analysis
config.SIGNIFICANCE_ALPHA = 0.05
config.USE_MI_FEATURE_FILTER = True
```

## Metrics Reference

| Metric | Range | Interpretation |
|--------|-------|----------------|
| ARI | [-1, 1] | 1 = perfect match, 0 = random, negative = worse than random |
| NMI | [0, 1] | Normalized mutual information, 1 = perfect |
| Purity | [0, 1] | % samples in dominant true class per cluster |
| Exact K | count | Cases finding correct cluster count |

## Latest Results (2026-02-07, 74 cases × 2 methods)

| Metric | `kl` (hamming) | `kl_rogerstanimoto` |
|--------|---------------|---------------------|
| Mean ARI | 0.579 | 0.567 |
| Median ARI | 0.755 | 0.746 |
| Mean NMI | 0.696 | 0.613 |
| Mean Purity | 0.785 | 0.647 |
| Exact K | 24/74 | 24/74 |

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
