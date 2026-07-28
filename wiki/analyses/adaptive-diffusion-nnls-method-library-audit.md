---
title: Adaptive Diffusion and NNLS Method/Library Audit
type: analysis
status: reviewed
updated: 2026-07-28
sources:
  - reports/adaptive_diffusion_nnls_method_library_audit_20260728.md
  - reports/benchmark_execution_audit_20260728.md
  - tree_break_selection/space_separation/diffusion.py
  - tree_break_selection/tree/feature_space.py
  - tree_break_selection/tree/optimized_branch_lengths.py
  - benchmarks/shared/runners/tbs_diffusion_runner.py
  - benchmarks/shared/runners/tbs_runner.py
  - benchmarks/shared/runners/method_registry.py
  - benchmarks/shared/util/time.py
  - tests/core/test_adaptive_diffusion_runner.py
  - tests/tree/test_optimized_branch_lengths.py
tags:
  - diffusion
  - nnls
  - branch-time
  - performance
  - dependencies
---

# Adaptive Diffusion and NNLS Method/Library Audit

## Summary

The canonical adaptive-diffusion NNLS route is a two-stage estimator:
pydiffmap Hamming diffusion plus average linkage selects topology, then SciPy
bounded least squares fits non-negative lengths to squared standardized
Euclidean distances in the original encoded columns. NNLS does not alter the
diffusion map or topology. Its benchmark improvement is a branch-length and
downstream-gating effect.

On the 14-case representative panel, NNLS improved six cases, left seven
unchanged, worsened none, raised exact-K recovery from `4/13` to `10/13`
successful rows, and added about 36% wall time. This does not yet validate the
fitted lengths as stochastic time.

The pydiffmap backend has a correctness blocker. Its hidden local-bandwidth KDE
needs seven retained positive distances per row. Clean data failed for public
K values 2 through 8 even though the repository resolver accepted them.
`binary_perfect_4c`, with 80 samples but seven distinct rows, failed at smaller
K from insufficient nonzero support and at larger K from zero bandwidth and
non-finite weights. Sample-count or distinct-count K bounds alone cannot repair
the method.

## Details

### Method boundary

The route is:

```text
encoded features
  -> pydiffmap Hamming adaptive kernel
  -> dense full eigendecomposition
  -> diffusion-coordinate Euclidean distance
  -> average-linkage rooted topology
  -> optional fixed-topology NNLS on raw standardized-feature distances
  -> normalized branch-length variance
  -> edge/sibling gates and traversal
```

Diffusion time is spectral-coordinate time, not branch time. The NNLS result is
an additive projection of a second geometry onto a topology selected from the
first geometry.

### Library boundary

pydiffmap 0.2.0.1 is the direct adaptive-kernel dependency and uses
scikit-learn nearest-neighbor sparse distance graphs. Sparse zero distances are
omitted, violating pydiffmap's assumption that each row retains enough
positive support. The repository additionally reads `scaled_dists` and mutates
`epsilon_fitted`, coupling the method to pydiffmap internal state.

SciPy owns dense eigendecomposition, pairwise distance, linkage, sparse
incidence matrices, and bounded least squares. Its solver reports multiple
termination conditions and numeric status, but the repository reduces every
`success=True` result to `ok`. Several representative large cases stopped on
relative cost change with reported optimality from `1.944` to `5.511`, so
`ok` does not mean tight first-order optimality.

NetworkX owns tree paths for NNLS. NumPy and pandas own the matrix/index
contract. Graphtools and scikit-bio are installed alternative diffusion and
neighbor-joining libraries but are not used in this exact route.

### Geometry and family mismatch

The NNLS target standardizes every encoded column independently. For binary
data this inverse-variance weights rare features. For one-hot categorical data
it ignores categorical blocks and is not invariant to encoding redundancy or
category count. For SBM adjacency data it treats coupled graph columns as
independent continuous coordinates.

Spearman agreement between diffusion distance and the NNLS target ranged from
`0.028` on `sbm_clear_small` to `0.791` on `binary_2clusters`. This is evidence
that the stages can encode materially different local geometry.

### Stability and performance

When all leaf pairs are used, the NNLS random seed has no effect. When the
50,000-pair cap is active, changing only pair seed changed fitted edge vectors
substantially in two tested large cases: edge correlations were `0.563–0.742`
and relative L2 changes were `0.454–0.547`. Final labels happened to remain
unchanged in this check, but the branch-time object is not stable enough to
infer that outcome generally.

The adaptive kernel begins sparse, but the repository converts it to dense and
computes a full `eigh`. Thus `n_components=30` does not limit eigenpair work.
The route carries `O(n^2)` memory and `O(n^3)` eigensolver pressure before the
50,000-pair NNLS cap becomes the main bound. Neighbor execution also
hard-codes `n_jobs=-1` instead of using the repository worker policy.

### Better implementations

Two existing alternatives were executed with the same NNLS policy on the
representative panel. Graphtools adaptive K completed all `14/14` cases with
mean ARI `0.837` and exact K `8/14`, but took `12.98` seconds, emitted
duplicate warnings, remains GPL-only, and is still densified by the repository
adapter.

The repository-owned fixed-Hamming kNN diffusion plus NNLS completed all
`14/14`, recovered exact K in `11/14`, achieved mean ARI `0.863` and median ARI
`1.0`, and took `11.60` seconds. It recovered `binary_perfect_4c` exactly.
This is the best immediate permissive-license implementation candidate,
although it is fixed-bandwidth rather than a mathematical replacement for
variable-bandwidth pydiffmap.

The best long-term adaptive implementation is a small repository-owned sparse
kernel using scikit-learn neighbor search, an explicit duplicate-state and
positive-distance bandwidth rule, sparse normalization, and
`scipy.sparse.linalg.eigsh` for only the requested eigenpairs. SciPy
`lsq_linear` should remain the NNLS solver; its surrounding target,
cross-validation, convergence, stability, and observability contract needs
improvement more than the numerical solver needs replacement.

### Installed comparison backends

The project now declares PyGSP `>=0.6.1` in the `diffusion` extra and deeptime
`>=0.4.5` in the `trajectory` extra. PHATE `>=2.0.0` is installed through
`experimental-gpl` beside graphtools because PHATE is GPLv2. SciPy and
scikit-learn remain core dependencies and already supply `eigsh`,
`NearestNeighbors`, and `SpectralEmbedding`.

Functional smokes passed for a duplicate-aware sparse neighbor operator,
partial `eigsh`, precomputed-affinity `SpectralEmbedding`, PHATE embedding,
PyGSP heat filtering, and deeptime TICA. PHATE reported the expected
disconnected-graph warning on a deliberately separated toy dataset.

Datafold `2.0.2` is not added to the project environment. The resolver would
downgrade NumPy from `2.3.4` to `1.26.4`, SciPy from `1.16.3` to `1.11.4`, and
scikit-learn from `1.7.2` to `1.2.2`; the latter contradicts the project's
declared `scikit-learn>=1.3.0` interface. It therefore requires a separately
validated environment or a future compatible release rather than an
uncontrolled downgrade of the canonical numerical stack.

### Three feature-family diffusion routes

`FeatureSpace` defines `bernoulli`, `categorical`, and `continuous` families.
They should select distinct geometry builders before sharing sparse diffusion
coordinates and tree construction:

| Family | Preferred geometry | Library direction |
| --- | --- | --- |
| Bernoulli | duplicate-explicit Hamming, Jaccard, Rogers–Tanimoto, or model-based Bernoulli neighbor affinity | repository-owned scikit-learn neighbors plus SciPy `eigsh` |
| Categorical | per-feature block mismatch, Hellinger, or multinomial/simplex distance | repository-owned block affinity, then direct SciPy or datafold precomputed kernel |
| Continuous | standardized/whitened Gaussian or adaptive Gaussian manifold kernel | datafold for generic point clouds; Scanpy for AnnData/scRNA |

Graph adjacency is a structured Bernoulli application, not an independent
Bernoulli feature table. It should use direct random-walk, normalized
Laplacian, heat-kernel, or PageRank diffusion; PyGSP is the strongest focused
library candidate.

Datafold is the most complete generic classical-diffusion-map candidate:
sparse kernels, selected eigenpairs, explicit time exponent, density
normalization, symmetric conjugation, and Nyström extension. Scanpy provides a
maintained adaptive Gaussian diffusion map over sparse neighbor graphs and is
appropriate for the repository's scRNA applications. Palantir and PHATE are
trajectory-oriented comparators instead of one global clustering backend.

### Decision boundary

Fixed-topology NNLS should remain an experimental branch-length policy until
the following are present:

- a duplicate-safe adaptive kernel or an explicit unsupported-input contract;
- family-aware NNLS targets routed through `FeatureSpace`;
- held-out pair fit and pair-seed edge-length stability;
- numeric solver status, active-bound, and conditioning diagnostics;
- downstream fail-closed handling for nonconverged branch fits;
- sparse partial eigendecomposition and exported NNLS timing.

The detailed executed evidence and implementation order are in
`reports/adaptive_diffusion_nnls_method_library_audit_20260728.md`.

The post-audit construction correction made the dual geometry explicit rather
than silently replacing it. All NNLS calls now require a branch-geometry
matrix. The adaptive production candidate deliberately supplies the original
distributional features and records
`explicit_original_distributional_features`; diffusion geometry is returned
separately as coordinates, condensed distances, and backend evidence. A
same-diffusion-coordinate NNLS experiment was rejected because mean ARI fell
to `0.3132`, exact-K to `2/13`, and runtime increased to about `29.6` seconds
on the same representative panel. This execution shows that geometry
coherence alone is not sufficient evidence for a useful branch-time model.

## Evidence

- `tree_break_selection/space_separation/diffusion.py` bounds K by sample count,
  reads pydiffmap internal state, densifies the kernel, and calls full
  `scipy.linalg.eigh`.
- `tree_break_selection/tree/feature_space.py` establishes Bernoulli,
  categorical-simplex, and continuous empirical-Gaussian as separate data
  contracts.
- `tree_break_selection/tree/optimized_branch_lengths.py` builds the
  path-incidence system, defines the per-column standardized target, samples
  pairs, calls `scipy.optimize.lsq_linear`, and mutates tree lengths.
- `benchmarks/shared/runners/method_registry.py` combines pydiffmap topology
  parameters, NNLS parameters, and normalized branch-length variance in the
  canonical method.
- The executed boundary test found K 2 through 8 invalid on
  `gauss_clear_small`; the duplicate-heavy perfect-binary case remained invalid
  through K 79.
- The pair-seed check found stable labels but unstable sampled NNLS edge
  vectors on the two large overlap cases.
- Alternative execution found the repository fixed-Hamming plus NNLS route
  completed all cases with exact K `11/14`, while graphtools adaptive K plus
  NNLS completed all cases with exact K `8/14`.
- Construction now rejects non-binary rooted results, reports exact tie
  burden, resolves input-order tie breaking deterministically, requires
  explicit NNLS branch geometry, and rejects explicit feature-family/metric
  mismatches.
- `reports/adaptive_diffusion_nnls_method_library_audit_20260728.md` records
  versions, formulas, correlations, solver diagnostics, complexity, ambiguous
  assumptions, and recommended tests.

## Links

- [[tree-construction-method-map]]
- [[edge-gate-distance-time-contract-20260623]]
- [[repository-execution-and-benchmark-map]]
- [[scrna-branch-length-effect-audit-20260624]]
- [[method-application-and-plot-seams-20260727]]

## Open Questions

- Should exact duplicate samples be represented as unique states with
  multiplicities, or should this pydiffmap route reject them?
- Which `FeatureSpace`-aware distance is justified as branch time for binary,
  categorical, continuous, and graph families?
- What solver optimality and pair-seed stability thresholds are required
  before fitted lengths may enter production gates?
- Should the old pydiffmap dependency be replaced by a repository-owned sparse
  kernel implementation with a direct tested contract?
