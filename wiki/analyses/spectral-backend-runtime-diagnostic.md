---
title: Spectral Backend Runtime Diagnostic
type: analysis
status: reviewed
updated: 2026-06-01
sources:
  - benchmarks/diagnostics/spectral/mp/profile_spectral_backends.py
  - benchmarks/results/diagnostics/spectral_backend_profile_20260526.csv
  - benchmarks/results/diagnostics/spectral_backend_profile_broader_20260526.csv
  - benchmarks/results/diagnostics/spectral_backend_profile_after_diagonal_vectorization_20260526.csv
  - benchmarks/results/diagnostics/spectral_backend_profile_broader_after_diagonal_vectorization_20260526.csv
  - benchmarks/results/diagnostics/spectral_backend_profile_after_grouped_categorical_20260526.csv
  - tree_break_selection/hierarchy_analysis/statistics/contrast_covariance.py
  - tree_break_selection/hierarchy_analysis/statistics/projection/spectral/marchenko_pastur.py
  - tree_break_selection/hierarchy_analysis/statistics/projection/spectral/tree_estimator.py
  - tree_break_selection/hierarchy_analysis/decomposition/backends/eigen/decomposition.py
tags:
  - method
  - spectral
  - performance
---

# Spectral Backend Runtime Diagnostic

## Summary

The main TBS spectral runtime bottleneck was not SciPy eigendecomposition. On
representative slow cases, eigensolves on already materialized node-local
matrices took milliseconds. Runtime was dominated by repeated construction of
null-whitened tangent matrices.

The 2026-05-26 profile first tested a diagonal shortcut for pure Bernoulli and
one-dimensional continuous blocks. That shortcut is no longer the active
continuous contract: generated continuous benchmark inputs now use one full
empirical-Gaussian block spanning the raw continuous columns. The retained
production shortcut is the exact vectorized Bernoulli whitening map.

A broader 22-case profile confirmed that this is not isolated to the initial
four cases. Across Gaussian, binary, continuous Gaussian, outlier, SBM,
categorical, phylogenetic, overlap, and quantile one-hot examples, matrix
materialization remained larger than exact eigendecomposition on already
materialized matrices. The slowest non-diagonal cases were high-dimensional
categorical and phylogenetic one-hot datasets, where the next exact speed path
is grouped multinomial/simplex whitening rather than diagonal whitening.

The first production repair implemented exact vectorized Bernoulli validation
and whitening in `contrast_covariance.py`. This does not change the
mathematical map for Bernoulli blocks. The historical post-change profile
reduced `binary_many_features` from 3.25 s to 0.11 s in the spectral worker.
The continuous diagonal speedup recorded on 2026-05-26 is now historical
evidence for an abandoned covariance contract, not current production behavior.

The second production repair grouped pure categorical and phylogenetic
one-hot blocks by category count and applied the same multinomial
drop-last whitening map with batched Cholesky/solve operations. This also does
not change the mathematical map: it computes
\[
  \Sigma_j(p)^{-1/2}(x_{j,1:K-1}-p_{j,1:K-1}),
  \qquad
  \Sigma_j(p)=\operatorname{diag}(p_{j,1:K-1})-p_{j,1:K-1}p_{j,1:K-1}^{\top}
  +\epsilon I .
\]
A production A/B comparison against the generic block loop produced identical
TBS result rows on selected categorical and phylogenetic cases. The post-change
profile reduced `cat_highd_3cat_500feat` from about 14.86 s to 0.31 s,
`phylo_dna_16taxa_low_mut` from about 21.83 s to 0.46 s, and
`phylo_large_32taxa` from about 27.32 s to 0.70 s in the spectral worker.

## Details

`profile_spectral_backends.py` extracts the same per-node matrices used by the
production Marchenko--Pastur spectral context: the active tree, feature-space
contract, leaf-only spectral row set, null distribution, and null-whitened
tangent coordinates all match the production path. It then profiles:

- `current_process_node`: the production per-node worker, including matrix
  construction, eigendecomposition, MP dimension estimation, and projection
  recovery.
- `full_eigh_projection`: full exact eigendecomposition and projection
  recovery on already materialized matrices.
- `eigvalsh_dimension_only`: eigenvalues only for MP dimension estimation.
- `eigvalsh_then_subset_projection`: exact two-stage SciPy path, with all
  eigenvalues first and top-\(k\) eigenvectors second.

The representative 2026-05-26 run showed:

| case | current worker | materialize once | full eigh on materialized matrices | eigvalsh only |
| --- | ---: | ---: | ---: | ---: |
| `binary_many_features` | 3.25 s | 3.21 s | 0.034 s | 0.022 s |
| `cat_highcard_20cat_4c` | 1.29 s | 1.21 s | 0.067 s | 0.039 s |
| `dim_consolidated_4c_72f_continuous` | 1.39 s | 1.35 s | 0.017 s | 0.012 s |
| `gauss_extreme_noise_highd_continuous` | 91.10 s | 91.95 s | 0.056 s | 0.035 s |

This means a SciPy eigenvalue-only pass can reduce the eigensolver part by
about \(1.4\times\) to \(1.7\times\), but that is the wrong first
optimization target for the current workloads. The exact two-stage top-\(k\)
projection path was slightly slower than full `eigh` in these small local
matrices, because the tree nodes have few rows even when the raw feature
dimension is large.

The exact Bernoulli vectorization diagnostic is the highest-value production
repair already made for binary blocks. For Bernoulli blocks, the
null-whitened tangent map is
\[
  (x-p)/\sqrt{p(1-p)+\epsilon},
\]
which can be applied to all one-dimensional Bernoulli blocks in one NumPy
operation. Before the repair, the generic block path looped over blocks and
invoked Cholesky factorization and triangular solves even when each block was
\(1\times 1\).

Categorical high-cardinality data did not enter the diagonal vectorization
arm, because each categorical feature is a multi-category simplex block. It
still showed the same qualitative bottleneck: materialization took 1.21 s
while full eigensolves on materialized matrices took 0.067 s. The analogous
optimization there is batched or grouped multinomial block whitening, not a
GPU eigensolver.

The broader profile added representative cases from the rest of the benchmark
surface. The largest materialization-heavy cases were:

| case | generator | current worker | materialization | full eigensolve |
| --- | --- | ---: | ---: | ---: |
| `phylo_large_32taxa` | phylogenetic | 32.74 s | 30.91 s | 0.316 s |
| `phylo_dna_16taxa_low_mut` | phylogenetic | 21.30 s | 21.24 s | 0.176 s |
| `cat_highd_3cat_500feat` | categorical | 14.97 s | 19.42 s | 0.083 s |
| `gauss_extreme_noise_many` | binary Gaussian | 9.53 s | 9.21 s | 0.188 s |
| `overlap_hd_4c_1k` | binary | 8.73 s | 8.79 s | 0.941 s |
| `gauss_overlap_8c_highd` | binary Gaussian | 4.59 s | 3.73 s | 0.710 s |

The categorical and phylogenetic rows are not covered by the diagonal
diagnostic because each feature block has a multinomial covariance in
drop-last simplex coordinates. They still have the same runtime shape:
materialization dominates exact eigendecomposition. Therefore the appropriate
production design is a family-specific exact whitening backend:

- vectorized whitening for pure one-dimensional Bernoulli blocks;
- grouped multinomial Cholesky/solve by category count for categorical and
  phylogenetic one-hot blocks;
- generic block whitening for continuous empirical-Gaussian blocks and
  genuinely mixed or irregular feature-space contracts.

After diagonal vectorization, the diagonal-family cases are no longer dominated
by block-loop materialization:

| case | current worker before | current worker after | materialization after |
| --- | ---: | ---: | ---: |
| `gauss_clear_large` | 0.144 s | 0.012 s | 0.005 s |
| `binary_many_features` | 3.25 s | 0.108 s | 0.063 s |
| `gauss_extreme_noise_many` | 9.53 s | 0.391 s | 0.180 s |
| `dim_diffuse_6c_136f_continuous` | 2.96 s | 0.109 s | 0.034 s |
| `overlap_hd_4c_1k` | 8.73 s | 0.864 s | 0.190 s |
| `gauss_extreme_noise_highd_continuous` | 91.10 s | 1.05 s | 0.973 s |

The remaining slow rows are exactly the non-diagonal one-hot families:
`cat_highd_3cat_500feat` stayed about 14.86 s, `phylo_dna_16taxa_low_mut`
about 21.83 s, and `phylo_large_32taxa` about 27.32 s. This makes grouped
categorical/phylogenetic whitening the next performance target.

After grouped categorical whitening, those rows are no longer the dominant
runtime class:

| case | current worker before | current worker after | materialization after |
| --- | ---: | ---: | ---: |
| `cat_highcard_20cat_4c` | 1.23 s | 0.18 s | 0.10 s |
| `cat_highd_3cat_500feat` | 14.86 s | 0.31 s | 0.24 s |
| `phylo_dna_16taxa_low_mut` | 21.83 s | 0.46 s | 0.37 s |
| `phylo_large_32taxa` | 27.32 s | 0.70 s | 0.50 s |
| `gauss_overlap_3c_small_q5` | 1.42 s | 0.11 s | 0.06 s |

The remaining performance question is no longer whether simplex whitening
should be grouped; that is now part of the active exact contract. The residual
work is finer-grained instrumentation inside the full benchmark pipeline and,
if needed, a compiled or cached path for genuinely mixed or irregular
feature-space contracts.

## Evidence

- `profile_spectral_backends.py` is the diagnostic entrypoint and records the
  measured backend and materialization timings.
- `spectral_backend_profile_20260526.csv` is the saved representative run.
- `spectral_backend_profile_broader_20260526.csv` is the broader profile across
  18 additional representative cases.
- `spectral_backend_profile_after_diagonal_vectorization_20260526.csv` records
  the post-repair representative profile.
- `spectral_backend_profile_broader_after_diagonal_vectorization_20260526.csv`
  records the post-repair broader family profile.
- `spectral_backend_profile_after_grouped_categorical_20260526.csv` records
  the post-repair grouped categorical and phylogenetic profile.
- `contrast_covariance.py` now contains an exact vectorized Bernoulli
  whitening path for pure one-dimensional Bernoulli feature spaces, plus exact
  grouped multinomial whitening for pure categorical one-hot feature spaces;
  continuous empirical-Gaussian blocks use the generic full-block covariance
  path.
- `marchenko_pastur.py` calls `build_null_whitened_tangent_matrix()` for each
  node before eigendecomposition.
- `tree_estimator.py` builds the per-node spectral tasks from the production
  tree and feature-space contract.
- `decomposition.py` shows that production eigendecomposition already chooses
  the smaller primal or dual covariance representation.

## Links

- [[local-marchenko-pastur-rule]]
- [[benchmark-pipeline-contract]]
- [[open-mathematical-questions]]

## Open Questions

- Should full benchmark result rows include stage timings for tree population,
  tangent materialization, eigendecomposition, edge gate, sibling gate, and traversal?
- Should partial eigensolvers remain diagnostic-only until a workload appears
  where eigendecomposition, not tangent materialization, dominates?
- Do genuinely mixed or irregular feature-space contracts need a compiled
  whitening backend, or is the generic block path sufficient because those
  cases are not current benchmark bottlenecks?
