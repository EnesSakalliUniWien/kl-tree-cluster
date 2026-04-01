# KL-TE Clustering Profiler Documentation

## Overview

The KL-TE Clustering Profiler (`debug_scripts/kl_clustering_profiler.py`) is a comprehensive performance analysis tool for the KL-TE clustering framework. It identifies and analyzes performance bottlenecks across the entire clustering pipeline, providing actionable optimization recommendations.

## Features

- **Component-level timing**: Measures execution time for each major pipeline component
- **Detailed breakdown**: Optional separate timing for Gate 2 and Gate 3 annotations
- **CSV export**: Saves detailed timing data for further analysis
- **Optimization recommendations**: Automatically generates prioritized recommendations based on profiling results
- **Configurable data sizes**: Profile with different sample sizes, feature counts, and cluster configurations
- **Test case support**: Can profile using predefined benchmark test cases

## Installation

No additional dependencies required beyond the existing KL-TE clustering framework requirements.

## Usage

### Basic Usage

```bash
# Run with default settings (n=80, d=100, K=4)
python debug_scripts/kl_clustering_profiler.py

# Save results to CSV
python debug_scripts/kl_clustering_profiler.py --output results.csv
```

### Command-Line Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--n-samples` | int | 80 | Number of samples (data points) |
| `--n-features` | int | 100 | Number of features (dimensions) |
| `--n-clusters` | int | 4 | Number of true clusters in synthetic data |
| `--case` | str | None | Use predefined test case (overrides size params) |
| `--seed` | int | 2001 | Random seed for reproducibility |
| `--output` | str | None | Output CSV file path |
| `--profile-level` | str | "component" | "component" or "function" |
| `--separate-gates` | flag | False | Profile Gate 2 and Gate 3 separately |
| `--profile-specific` | str | None | Profile specific component only |

### Examples

```bash
# Profile with custom data size
python debug_scripts/kl_clustering_profiler.py --n-samples 200 --n-features 150

# Profile a specific benchmark test case
python debug_scripts/kl_clustering_profiler.py --case binary_perfect_4c

# Get detailed Gate 2 vs Gate 3 breakdown
python debug_scripts/kl_clustering_profiler.py --separate-gates --output detailed.csv

# Profile only the annotation pipeline
python debug_scripts/kl_clustering_profiler.py --profile-specific full

# Function-level profiling with cProfile
python debug_scripts/kl_clustering_profiler.py --profile-level function
```

## Output Format

### Console Output

The profiler prints a formatted table showing:

1. **Configuration**: Data size and pipeline settings
2. **Timing Breakdown**: Execution time for each component
3. **Optimization Recommendations**: Prioritized suggestions based on bottlenecks

Example:
```
================================================================================
PROFILING RESULTS
================================================================================

Timing Breakdown:
--------------------------------------------------------------------------------
Component                    Time (ms)    % Total Description
--------------------------------------------------------------------------------
tree_decomposition              162.48      49.7% Top-down traversal with statistical
full_annotation_pipeline        161.59      49.4% Combined Gate 2 + Gate 3 annotation
data_generation                   1.78       0.5% Generate synthetic feature matrix
tree_construction                 0.75       0.2% Compute pairwise distances and linkage
distribution_population           0.56       0.2% Compute node distributions
--------------------------------------------------------------------------------
TOTAL                           327.17     100.0%

================================================================================
OPTIMIZATION RECOMMENDATIONS
================================================================================

1. [HIGH] FULL_ANNOTATION_PIPELINE
   Issue: Gate 2 annotation is the bottleneck
   Recommendation: Spectral decomposition (eigendecomposition) is O(min(n,d)³).
   Consider: (1) Using dual formulation when n < d, (2) Truncated eigendecomposition,
   (3) Randomized SVD for approximate eigenvalues, (4) Caching eigen decompositions.
```

### CSV Output

The CSV file contains detailed timing data with the following columns:

| Column | Description |
|--------|-------------|
| `Component` | Internal component name |
| `Description` | Human-readable description |
| `Time (ms)` | Execution time in milliseconds |
| `Time (s)` | Execution time in seconds |
| `Percentage (%)` | Percentage of total time |
| `Call Count` | Number of times component was called |
| `Time per Call (ms)` | Average time per call |

CSV files include metadata comments at the top:
```csv
# KL-TE Clustering Profiler Results
# Generated: 2026-03-28T13:53:46.047384
# n_samples: 80, n_features: 100, n_clusters: 4
#
Component,Description,Time (ms),Time (s),Percentage (%),...
```

## Profiled Components

The profiler measures the following components:

### 1. Data Generation (`data_generation`)
- Synthetic feature matrix generation
- Typically < 1% of total time
- Uses `generate_random_feature_matrix()`

### 2. Tree Construction (`tree_construction`)
- Pairwise distance computation (pdist)
- Hierarchical clustering linkage
- Typically < 1% of total time
- Uses scipy's `linkage()` with Hamming distance

### 3. Distribution Population (`distribution_population`)
- Computes Bernoulli parameters for all nodes
- Bottom-up aggregation from leaves
- Typically < 1% of total time

### 4. Gate 2 Annotation (`gate2_annotation`)
- Child-parent divergence testing
- Spectral decomposition (eigendecomposition)
- Projected Wald chi-square tests
- Tree-BH FDR correction
- **Major bottleneck**: ~25-50% of total time

### 5. Gate 3 Annotation (`gate3_annotation`)
- Sibling divergence testing
- Benjamini-Hochberg FDR correction
- Cousin-adjusted Wald statistics
- **Major bottleneck**: ~25-50% of total time

### 6. Full Annotation Pipeline (`full_annotation_pipeline`)
- Combined Gate 2 + Gate 3 (when using orchestrator)
- **Major bottleneck**: ~50-99% of total time

### 7. Tree Decomposition (`tree_decomposition`)
- Top-down traversal with gate evaluation
- Cluster formation
- Typically ~50% of total time (includes annotation time)

## Interpretation Guide

### Understanding the Results

**Normal Profile (n < 100, d < 150):**
- Annotation pipeline: 40-60%
- Tree decomposition: 40-60%
- Other components: < 5%

**Large Dataset (n > 200):**
- Annotation pipeline dominates (> 80%)
- Spectral decomposition becomes the bottleneck
- Consider optimization recommendations

**High Dimensionality (d > 500):**
- Tree construction time increases
- Distribution population may slow down
- Consider dimensionality reduction

### Priority Levels

Recommendations are prioritized as:

- **HIGH**: Critical bottleneck (> 40% of time), significant optimization potential
- **MEDIUM**: Notable impact (20-40% of time), moderate optimization potential
- **LOW**: Minor impact (< 20% of time), incremental improvements

### Common Bottlenecks

#### 1. Spectral Decomposition (Most Common)

**Symptoms:**
- `gate2_annotation` or `full_annotation_pipeline` > 40%
- Time scales as O(min(n,d)³)

**Solutions:**
1. Use dual formulation when n < d (already implemented)
2. Truncated eigendecomposition: `scipy.sparse.linalg.eigsh(k=...)`
3. Randomized SVD: `sklearn.utils.extmath.randomized_svd()`
4. Cache eigen decompositions for overlapping node sets

#### 2. Tree Decomposition Traversal

**Symptoms:**
- `tree_decomposition` > 30% (excluding annotation time)
- Many internal nodes

**Solutions:**
1. Pre-compute `has_descendant_split` flags (already implemented)
2. Early stopping for homogeneous subtrees
3. Parallel traversal for independent subtrees

#### 3. Large Dataset Scaling

**Symptoms:**
- n > 500 samples
- Total time > 10 seconds

**Solutions:**
1. Subsampling for initial tree construction
2. Approximate nearest neighbors for distances
3. Incremental/online clustering methods

## Programmatic Usage

The profiler can also be used programmatically:

```python
from debug_scripts.kl_clustering_profiler import KLProfiler

# Create profiler instance
profiler = KLProfiler(
    n_samples=100,
    n_features=150,
    n_clusters=4,
    random_seed=2001,
)

# Run profiling
results = profiler.profile(use_orchestrator=True)

# Generate report DataFrame
report_df = profiler.generate_report()
print(report_df)

# Save to CSV
profiler.save_csv("my_results.csv")

# Function-level profiling
from kl_clustering_analysis.hierarchy_analysis.decomposition.backends.eigen_backend import (
    eigendecompose_correlation_backend,
)
profiler.run_function_profile(
    eigendecompose_correlation_backend,
    profiler.data.values,
    compute_eigenvectors=False,
)
```

## Performance Benchmarks

Typical runtimes on modern hardware (Apple M1 / Intel i7):

| n | d | Total Time | Annotation % | Decomposition % |
|---|---|------------|--------------|-----------------|
| 80 | 100 | ~330 ms | 50% | 50% |
| 200 | 150 | ~1900 ms | 50% | 50% |
| 500 | 200 | ~15000 ms | 60% | 40% |

Note: Runtimes scale superlinearly due to O(n³) eigendecomposition.

## Troubleshooting

### Common Issues

**1. "Missing leaf_count column"**
- Ensure tree has been populated with distributions before profiling
- The profiler handles this automatically

**2. "Missing 'distribution' attribute"**
- Call `populate_distributions()` before annotation
- The profiler handles this automatically

**3. Slow performance on large datasets**
- Expected behavior due to O(n³) spectral decomposition
- Use `--profile-specific` to isolate bottlenecks
- Consider optimization recommendations

### Getting Help

For issues or questions:
1. Check the error message for specific component failures
2. Run with `--profile-specific` to isolate the issue
3. Review the CSV output for detailed timing information

## Contributing

To extend the profiler:

1. Add new timing points using the `_time()` context manager:
   ```python
   with self._time("new_component", "Description"):
       # code to profile
   ```

2. Add component-specific recommendations in `_print_recommendations()`

3. Update the documentation with new options

## License

Same license as the KL-TE clustering framework.

## Version History

- **v1.0** (2026-03-28): Initial release
  - Component-level profiling
  - CSV export
  - Optimization recommendations
  - Test case support
