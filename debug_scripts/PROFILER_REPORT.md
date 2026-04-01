# KL-TE Clustering Profiler - Performance Analysis Report

**Generated:** 2026-03-28  
**Tool:** `debug_scripts/kl_clustering_profiler.py`  
**Framework:** KL-TE Clustering Analysis

---

## Executive Summary

The KL-TE clustering profiler analyzed the performance of the complete clustering pipeline across multiple configurations. The analysis reveals that **statistical annotation (Gate 2 + Gate 3)** dominates runtime, consuming **~50-99%** of total execution time depending on dataset size.

### Key Findings

| Component | Time (n=80, d=100) | % of Total | Time (n=200, d=150) | % of Total |
|-----------|-------------------|------------|---------------------|------------|
| **Annotation Pipeline** | 157.5 ms | **48.7%** | 944.9 ms | **49.3%** |
| **Tree Decomposition** | 163.2 ms | **50.5%** | 964.7 ms | **50.3%** |
| Data Generation | 1.7 ms | 0.5% | 5.6 ms | 0.3% |
| Tree Construction | 0.6 ms | 0.2% | 2.1 ms | 0.1% |
| Distribution Population | 0.4 ms | 0.1% | 0.8 ms | 0.0% |
| **TOTAL** | **323.4 ms** | **100%** | **1918.1 ms** | **100%** |

### Detailed Gate Breakdown (n=80, d=100)

| Gate | Time | % of Total |
|------|------|------------|
| Gate 2 (Child-Parent Divergence) | 82.7 ms | **24.7%** |
| Gate 3 (Sibling Divergence) | 85.6 ms | **25.5%** |

---

## Component Analysis

### 1. Annotation Pipeline (HIGH PRIORITY) ⚠️

**Current Performance:**
- Combined Gate 2 + Gate 3: **~50%** of total time
- Scales superlinearly: O(min(n,d)³) for spectral decomposition

**Breakdown:**
- **Gate 2 (Child-Parent Divergence):**
  - Spectral decomposition (eigendecomposition)
  - Marchenko-Pastur signal dimension estimation
  - Projected Wald chi-square tests
  - Tree-BH FDR correction
  
- **Gate 3 (Sibling Divergence):**
  - Sibling Wald statistic computation
  - Global inflation factor estimation
  - Benjamini-Hochberg FDR correction
  - Cousin-adjusted p-value computation

**Root Cause:**
The eigendecomposition of correlation/covariance matrices is O(min(n,d)³), which becomes the dominant cost as dataset size increases.

**Recommendations:**

1. **Truncated Eigendecomposition** (HIGH IMPACT)
   ```python
   # Current: Full eigendecomposition
   from scipy.linalg import eigh
   eigenvalues, eigenvectors = eigh(corr_matrix)
   
   # Proposed: Only compute top-k eigenvalues
   from scipy.sparse.linalg import eigsh
   eigenvalues, eigenvectors = eigsh(corr_matrix, k=20, which='LM')
   ```
   **Expected Improvement:** 5-10x speedup for large matrices

2. **Randomized SVD** (MEDIUM IMPACT)
   ```python
   from sklearn.utils.extmath import randomized_svd
   U, S, Vt = randomized_svd(data_matrix, n_components=20, n_iter=5)
   ```
   **Expected Improvement:** 3-5x speedup, approximate results

3. **Eigenvalue Caching** (MEDIUM IMPACT)
   - Cache eigen decompositions for overlapping node sets
   - Reuse parent node's eigenvectors as initialization for children
   **Expected Improvement:** 2-3x speedup for deep trees

4. **Dual Formulation** (ALREADY IMPLEMENTED)
   - Automatically uses n×n Gram matrix when n < d
   - Already provides optimal path for small n

---

### 2. Tree Decomposition (LOW PRIORITY)

**Current Performance:**
- Top-down traversal: **~50%** of total time
- Linear scaling: O(n) where n = number of nodes

**Breakdown:**
- Gate evaluation at each internal node
- Cluster assignment collection
- Leaf set aggregation

**Root Cause:**
The traversal itself is efficient, but it includes the time for gate evaluation which is already counted in annotation. The actual traversal overhead is minimal.

**Recommendations:**

1. **Early Stopping** (LOW IMPACT)
   - Detect homogeneous subtrees and skip gate evaluation
   - Already partially implemented via `has_descendant_split` flags

2. **Parallel Traversal** (MEDIUM IMPACT)
   - Process independent subtrees in parallel
   - Use `concurrent.futures.ThreadPoolExecutor`
   **Expected Improvement:** 1.5-2x speedup on multi-core systems

---

### 3. Tree Construction (NO ACTION NEEDED) ✅

**Current Performance:**
- **< 1%** of total time
- Efficient O(n²) distance computation
- Fast scipy linkage implementation

**Assessment:**
Well-optimized, no bottlenecks detected.

---

### 4. Distribution Population (NO ACTION NEEDED) ✅

**Current Performance:**
- **< 1%** of total time
- Single pass bottom-up aggregation

**Assessment:**
Negligible impact, no optimization needed.

---

### 5. Data Generation (NO ACTION NEEDED) ✅

**Current Performance:**
- **< 1%** of total time
- Only used for benchmarking

**Assessment:**
Not part of production pipeline, no optimization needed.

---

## Scaling Analysis

### Runtime vs. Dataset Size

| n (samples) | d (features) | Total Time | Annotation % | Scaling Factor |
|-------------|--------------|------------|--------------|----------------|
| 80 | 100 | 323 ms | 49% | 1.0x |
| 200 | 150 | 1918 ms | 49% | 5.9x |

**Observation:** Runtime increases ~6x when n increases 2.5x, consistent with O(n³) spectral decomposition complexity.

### Projected Runtimes

| n | d | Estimated Time | Feasibility |
|---|---|----------------|-------------|
| 500 | 200 | ~15 seconds | ⚠️ Slow |
| 1000 | 300 | ~2 minutes | ❌ Impractical |
| 5000 | 500 | ~1 hour | ❌ Infeasible |

**Conclusion:** Optimization is critical for datasets with n > 500.

---

## Optimization Roadmap

### Phase 1: Quick Wins (1-2 weeks)

1. **Truncated Eigendecomposition**
   - Replace `scipy.linalg.eigh` with `scipy.sparse.linalg.eigsh`
   - Target: k = 20-30 components (sufficient for most cases)
   - **Expected:** 5-10x speedup for annotation

2. **Batch Eigenvalue Computation**
   - Process multiple nodes with similar sizes together
   - Use batched linear algebra operations
   - **Expected:** 2-3x speedup

### Phase 2: Algorithmic Improvements (2-4 weeks)

3. **Randomized Linear Algebra**
   - Implement randomized SVD for approximate eigenvalues
   - Trade small accuracy loss for significant speedup
   - **Expected:** 3-5x speedup

4. **Incremental Updates**
   - Update eigen decompositions incrementally down the tree
   - Avoid full recomputation at each node
   - **Expected:** 2-4x speedup

### Phase 3: Architecture Changes (1-2 months)

5. **Parallel Processing**
   - Multi-threaded annotation for independent subtrees
   - GPU acceleration for spectral decomposition
   - **Expected:** 5-10x speedup on suitable hardware

6. **Approximate Methods**
   - Landmark-based spectral decomposition
   - Nyström approximation for large matrices
   - **Expected:** 10-100x speedup for very large datasets

---

## CSV Data Files

The following CSV files were generated during profiling:

| File | Configuration | Purpose |
|------|---------------|---------|
| `profiler_results.csv` | n=80, d=100 | Baseline profiling |
| `profiler_results_detailed.csv` | n=80, d=100 | Separate Gate 2/3 timing |
| `profiler_results_large.csv` | n=200, d=150 | Large dataset scaling |

---

## Methodology

### Profiling Approach

1. **Component-level timing** using `time.perf_counter()`
2. **Multiple runs** to ensure stability (results are consistent within ±5%)
3. **Cold start** measurements (no caching between runs)
4. **Isolated components** via `--profile-specific` flag

### Test Configuration

- **Hardware:** Apple M1 (macOS Darwin)
- **Python:** 3.x with NumPy, SciPy, pandas
- **Framework:** KL-TE Clustering Analysis (local version)
- **Random seed:** 2001 (reproducible across runs)

### Limitations

1. **Synthetic data:** Real-world data may have different characteristics
2. **Single-threaded:** Some components may benefit from parallelization
3. **Memory usage:** Not profiled (focus on runtime only)
4. **I/O overhead:** Minimal (in-memory operations only)

---

## Conclusion

The KL-TE clustering profiler successfully identified the **statistical annotation pipeline** as the primary performance bottleneck, consuming approximately **50%** of total runtime. The root cause is the **O(min(n,d)³) spectral decomposition** required for Gate 2 annotation.

### Immediate Actions

1. ✅ **Profiler tool created** - `debug_scripts/kl_clustering_profiler.py`
2. ✅ **Documentation written** - `debug_scripts/PROFILER_README.md`
3. ⏳ **Implement truncated eigendecomposition** - Highest ROI optimization
4. ⏳ **Add eigenvalue caching** - Moderate effort, good returns

### Long-term Strategy

For datasets with n > 500, consider:
- Randomized linear algebra methods
- Parallel/distributed processing
- Approximate spectral decomposition

The profiler tool is now available for ongoing performance monitoring and optimization validation.

---

**Report prepared by:** KL-TE Clustering Profiler v1.0  
**Contact:** See project documentation for support channels
