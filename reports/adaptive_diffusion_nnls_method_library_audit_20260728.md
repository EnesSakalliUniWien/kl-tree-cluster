# Adaptive Diffusion and NNLS Method/Library Audit — 2026-07-28

## Scope

This audit examines the canonical
`tbs_diffusion_adaptive_nnls` execution path, its raw-linkage control
`tbs_diffusion_adaptive`, and the libraries that implement their geometry,
topology, branch-length fitting, and traversal inputs. It is a method-contract
and implementation audit, not evidence that one 14-case benchmark is
scientifically representative.

The audited revision is `dev` at `4a502615`. The executable environment is the
repository `uv` environment.

## Executive conclusion

The NNLS addition is useful but currently over-claimed by its method name and
benchmark status:

1. **It does not improve the diffusion map or the tree topology.** It keeps the
   adaptive-pydiffmap distance, average-linkage topology, and linkage root
   fixed, then changes edge lengths and therefore the downstream
   branch-variance gates.
2. **It is a dual-geometry estimator.** Topology is selected using Hamming
   adaptive-diffusion distance, while branch lengths are fitted to squared
   standardized Euclidean distances in the original encoded feature matrix.
   This can be a deliberate two-stage design, but it is not one coherent
   diffusion-tree metric.
3. **The current pydiffmap backend is not safe for its declared input domain.**
   Ordinary distinct-row data fails for public `k` values 2 through 8 because
   pydiffmap has a hidden seven-nonzero-distance bandwidth requirement.
   Duplicate-heavy binary data can remain invalid at every larger `k` because
   zero distances disappear from sparse support and can produce zero local
   bandwidths.
4. **The small benchmark gain is real but insufficient branch-time evidence.**
   NNLS improved six cases, left seven unchanged, worsened none, and increased
   wall time about 36%. On sampled large cases, however, changing only the pair
   seed materially changed fitted edge lengths while leaving the tested final
   labels unchanged.
5. **Library success is being interpreted too broadly.** SciPy may return
   `success=True` after a relative cost-change condition even when the reported
   first-order optimality is not small. The repository records this as `ok`,
   omits the numeric solver status, mutates the tree before checking
   convergence, and proceeds downstream even on `solver_not_converged`.

The safest methodological position is therefore:

- retain fixed-topology NNLS as an experimental branch-time policy;
- do not treat pydiffmap adaptive diffusion as a duplicate-safe canonical
  backend;
- do not interpret diffusion time as stochastic branch time;
- require family-aware targets, solver diagnostics, held-out pair error, and
  branch-length stability before promoting the NNLS lengths as calibrated
  scientific time.

## Executed method map

```text
encoded sample-by-feature matrix
  |
  | pydiffmap Kernel
  |   metric = hamming
  |   k = 10
  |   variable bandwidth = -1/(d+2)
  |   global epsilon = median positive scaled squared distance
  v
sparse adaptive kernel
  |
  | convert to dense
  | symmetric full eigendecomposition
  | diffusion time = 3, retain <= 30 nontrivial coordinates
  v
Euclidean diffusion distances
  |
  | SciPy average linkage
  v
fixed rooted topology
  |
  | optional SciPy bounded least squares
  | target = squared standardized Euclidean distance in raw encoded features
  | <= 50,000 sampled leaf pairs, non-negative edge lengths
  v
linkage-ultrametric or NNLS branch lengths
  |
  | normalized branch-length variance policy
  | edge and sibling gates
  | top-down traversal
  v
clusters
```

The raw adaptive method stops at linkage-derived edge lengths. The NNLS
variant changes only the branch-length stage and the branch-length variance
policy.

## Library inventory and responsibility

The installed versions were read from the executable repository environment.

| Library | Installed | Responsibility in this route | Audit assessment |
| --- | ---: | --- | --- |
| NumPy | 2.3.4 | dense arrays, standardization, pair sampling, diagnostics | Appropriate; dense conversions amplify the algorithmic scaling problem |
| pandas | 2.3.3 | indexed feature-matrix contract | Appropriate; not a material bottleneck in this stage |
| pydiffmap | 0.2.0.1 | nearest-neighbor kernel, local bandwidth density, adaptive scaling | Highest-risk dependency: PyPI labels the release Alpha; version 0.2.0.1 was released in February 2019 and advertises Python classifiers only through 3.7 |
| scikit-learn | 1.7.2 | `NearestNeighbors` used inside pydiffmap | Sparse distance graphs omit zero-distance entries, which is incompatible with pydiffmap's assumption that every row retains enough nonzero distances |
| SciPy | 1.16.3 | `pdist`, `eigh`, hierarchical linkage, sparse matrices, `lsq_linear` | Core numerical backend; solver outputs are richer than the repository's current `ok`/`not converged` reduction |
| NetworkX | 3.5 | rooted tree representation and root-to-leaf edge paths for NNLS | Correct responsibility, but Python graph/path traversal adds overhead when building large incidence systems |
| joblib | 1.5.2 | transitive parallel execution through scikit-learn | The route hard-codes `n_jobs=-1`, bypassing the repository's `TBS_N_JOBS` policy |
| matplotlib | 3.10.7 | downstream report plotting only | Not part of diffusion, topology, or NNLS inference |
| graphtools | 2.1.0 | alternative optional diffusion backend | Not used by this exact pydiffmap method; its benchmark resolver already has duplicate-aware K policy absent from the pydiffmap route |
| scikit-bio | 0.7.2 | optional neighbor-joining topology | Not used by the canonical average-linkage pydiffmap route |

The dependency declaration uses lower bounds without an upper compatibility
bound for pydiffmap, SciPy, scikit-learn, and NetworkX. The lockfile makes the
current environment reproducible, but pydiffmap itself is old and the
repository couples to its non-public state:

- `Kernel.scaled_dists` is read directly;
- `Kernel.epsilon_fitted` is overwritten directly;
- pydiffmap's private nearest-neighbor KDE behavior is part of the effective
  method contract.

Official package references:

- <https://pypi.org/project/pydiffmap/>
- <https://github.com/DiffusionMapsAcademics/pyDiffMap>

## Adaptive pydiffmap analysis

### Hidden neighbor-support contract

`resolve_neighbor_search_k()` currently checks only:

```text
2 <= k <= n_samples - 1
```

pydiffmap 0.2.0.1 separately constructs an `NNKDE` with a hard-coded local
`k=8`. Its fit requests a distance graph with one fewer neighbor and then
selects seven nonzero distances per row. Because sparse distance graphs do not
retain zero values, the actual condition is not a sample-count condition. It
is a per-row positive-distance-support condition.

An executed boundary check on `gauss_clear_small` found:

| Requested K | Result |
| --- | --- |
| 2 through 8 | `kth(=6) out of bounds` |
| 9 through 12 | finite diffusion distances |

This is an undocumented lower bound in the current backend. The public
resolver accepts configurations that the backend cannot execute.

### Duplicate-row failure

`binary_perfect_4c` has:

- 80 samples;
- 100 binary features;
- four balanced truth classes;
- only seven distinct encoded rows.

At requested `k` values from 8 through 26, pydiffmap failed because at least
one row had fewer than seven retained positive distances. At `k` values from
27 through 79, the initial index-selection failure disappeared but pydiffmap
produced zero bandwidths, divide-by-zero warnings, and non-finite diffusion
weights.

The failure is therefore not fixed by validating `k` against distinct-row
count or by silently increasing `k`. A robust implementation must define what
duplicate samples mean:

- collapse to unique states, retain multiplicities, construct the kernel over
  states, then lift coordinates back to samples; or
- implement a duplicate-safe local scale with explicit zero-distance policy;
  or
- reject exact duplicates before entering this backend with a precise
  unsupported-input error.

The first two options change the method and require mathematical tests. The
third is only fail-fast hygiene.

### Hamming geometry versus bandwidth theory

The route uses Hamming distance but asks pydiffmap to estimate an intrinsic
dimension and applies bandwidth exponent `-1/(d+2)`. The implementation comes
from variable-bandwidth diffusion-map machinery normally motivated for smooth
metric/manifold settings. In this repository it is applied to discrete binary
and one-hot spaces without a documented derivation showing that pydiffmap's
dimension estimate or exponent retains the intended interpretation.

This does not prove the geometry is unusable. It means the current exponent is
a heuristic for Hamming data and should be named and validated as such.

### Internal-state coupling

The repository creates a pydiffmap `Kernel` with temporary epsilon 1.0, fits
the kernel, reads `scaled_dists`, computes its own median positive scaled
squared distance, overwrites `epsilon_fitted`, and calls `compute()`.

That procedure gives the repository control over global epsilon, but it relies
on attributes that are not a stable public adapter contract. A future library
change could alter units, sparsity, or the fit lifecycle without a clear API
error.

### Complexity and bottlenecks

Let `n` be samples, `p` features, `k` neighbors, and `c` retained diffusion
coordinates.

| Stage | Current scaling pressure |
| --- | --- |
| Hamming neighbor search | Can approach `O(n^2 p)` with brute-force/high-dimensional discrete data |
| Sparse kernel | Approximately `O(nk)` storage before repository conversion |
| Dense conversion | `O(n^2)` memory |
| Full symmetric `eigh` | `O(n^3)` time and `O(n^2)` memory |
| Diffusion-coordinate `pdist` | `O(n^2 c)` time and `O(n^2)` condensed output |
| Hierarchical linkage | `O(n^2)` memory/time class for the condensed-distance route |

`n_components=30` limits output coordinates but does not make the
eigendecomposition partial: the code computes the full dense spectrum and
discards all but the requested components. This is the dominant scaling
contradiction in the diffusion implementation.

The hard-coded `neighbor_params={"n_jobs": -1}` can also oversubscribe CPUs
when benchmarks already constrain BLAS or repository worker counts.

## Fixed-topology NNLS analysis

### Objective

For a fixed rooted tree with non-negative edge lengths \(\tau_e\), the solver
fits:

\[
D_{ij} \approx \sum_{e \in path(i,j)} \tau_e,\qquad \tau_e \ge 0.
\]

The design matrix has one row per selected leaf pair and one column per tree
edge. A row contains ones on the symmetric difference of the two root paths.
SciPy `lsq_linear(method="trf", lsmr_tol="auto")` solves the bounded sparse
least-squares problem.

This is a valid fixed-topology additive-distance projection. It is not a
topology estimator and does not guarantee that the selected topology is
appropriate for the target distances.

### Dual geometry

The topology target and branch-length target differ:

| Stage | Geometry |
| --- | --- |
| Topology selection | Euclidean distance between Hamming adaptive-diffusion coordinates |
| Branch-length fitting | Squared standardized Euclidean distance over original encoded columns |

Across the 13 executable representative cases, rank correlation between these
two pairwise geometries varied widely:

| Case | Spearman | Pearson |
| --- | ---: | ---: |
| `sbm_clear_small` | 0.028 | 0.012 |
| `binary_moderate_6c` | 0.283 | 0.716 |
| `overlap_heavy_4c_small_feat` | 0.304 | 0.360 |
| `overlap_mod_4c_small` | 0.479 | 0.516 |
| `gauss_clear_large` | 0.490 | 0.892 |
| `gauss_clear_small` | 0.571 | 0.861 |
| `cat_clear_3cat_4c` | 0.695 | 0.849 |
| `gauss_overlap_3c_small` | 0.733 | 0.806 |
| `binary_2clusters` | 0.791 | 0.890 |

The remaining successful cases fell inside these ranges. High Pearson with
lower Spearman indicates that broad separation can agree while local ordering
differs. The near-zero SBM result is especially important: the NNLS target is
not graph-specific geometry.

### Feature-family assumptions

The standardized target treats every encoded column as an independent
continuous coordinate:

1. center each column;
2. divide by sample standard deviation, replacing zero/bad scales with 1;
3. compute squared Euclidean distance;
4. divide by number of columns.

Consequences:

- **Binary data:** rare Bernoulli features receive stronger inverse-variance
  weighting than common features, unlike uniform Hamming topology.
- **One-hot categorical data:** each indicator is standardized independently.
  The target is not invariant to category encoding, block size, or redundant
  one-hot columns, and it does not use the repository `FeatureSpace` simplex or
  multinomial covariance contract.
- **SBM adjacency data:** adjacency columns are treated as independent
  continuous coordinates even though graph rows and columns are coupled.
- **Continuous data:** per-column standardization can be reasonable, but it
  discards covariance structure and still needs justification as stochastic
  branch time.

The strong categorical benchmark improvement is therefore evidence for a
useful gate reweighting on that encoded matrix, not evidence that the fitted
lengths are categorical evolutionary time.

### Solver and convergence ambiguity

The 13 successful representative fits used:

- 13 to 63 iterations, median 20;
- about 0.011 to 1.027 seconds in the NNLS stage, mean 0.239 seconds;
- relative training RMSE from 0.018 to 0.156, mean 0.099;
- SciPy `success=True` for every fit.

Several large cases stopped because relative cost change fell below tolerance
while reporting comparatively large first-order optimality:

| Case | Reported optimality | Iterations |
| --- | ---: | ---: |
| `gauss_overlap_3c_small` | 1.944 | 63 |
| `overlap_heavy_4c_small_feat` | 2.741 | 28 |
| `overlap_mod_4c_small` | 5.511 | 60 |

SciPy exposes numeric `status`, `success`, `message`, `optimality`, active
bounds, and iteration count. The repository stores most summaries but reduces
all `success=True` outcomes to `status="ok"` and omits numeric solver status.
It also writes the fitted edge lengths before interpreting convergence and
does not stop the downstream gates when status is `solver_not_converged`.

The current `ok` therefore means "SciPy accepted a termination condition," not
"the branch lengths satisfy a tight optimality criterion."

### Pair sampling and stability

All pairs are used when \(n(n-1)/2 \le 50,000\); otherwise the implementation
draws exactly 50,000 unique pairs using a Python set and a fixed random seed.
The fit error is measured on those same fitting pairs. There is no held-out
pair error, topology resampling, or edge-length uncertainty.

Changing only the pair-sampling seed produced:

| Case | Pair regime | Final result across seeds 0/1/2 | Edge-length correlation vs seed 0 | Relative edge-length L2 change |
| --- | --- | --- | ---: | ---: |
| `gauss_overlap_3c_small` | all 44,850 pairs | identical K=6, ARI 0.674 | 1.000 | 0.000 |
| `overlap_mod_4c_small` | sampled 50,000/79,800 | identical K=4, ARI 0.852 | 0.739–0.742 | 0.454–0.455 |
| `overlap_heavy_4c_small_feat` | sampled 50,000/124,750 | identical K=1, ARI 0.000 | 0.563–0.604 | 0.518–0.547 |

This limited check is reassuring for these final labels but not for the fitted
branch-time object. Other cases can lie near gate thresholds, where these
edge-length changes may alter traversal.

The Python-set sampler also becomes inefficient as requested sample size
approaches the full pair population because duplicates are repeatedly drawn
and discarded. Direct sampling of condensed-distance indices without
replacement would have a clearer complexity contract.

### NNLS complexity

For a binary rooted tree there are approximately `2n-2` edges. With `m`
selected leaf pairs and tree depth `h`:

| Stage | Current scaling pressure |
| --- | --- |
| Root-path extraction | `O(nh)` Python/NetworkX traversal |
| Pair sampling | expected overhead above `O(m)` due to set deduplication |
| Sparse path incidence | `O(mh)` nonzeros and Python list construction |
| Target construction | `O(mp)` temporary differences |
| TRF + iterative LSMR | depends on iterations and incidence-matrix conditioning |

The 50,000-pair cap bounds one part of cost, but dense diffusion and full
eigendecomposition will generally become limiting earlier.

## Benchmark interpretation

On the same 14-case representative subset:

| Outcome | Adaptive diffusion | Adaptive diffusion + NNLS |
| --- | ---: | ---: |
| Wall time | 8.22 s | 11.19 s |
| Successful rows | 13 | 13 |
| Explicit skips | 1 | 1 |
| Exact K among successful rows | 4/13 | 10/13 |
| Mean ARI | 0.5445 | 0.8722 |
| Median ARI | 0.7054 | 1.0000 |

NNLS improved six ARIs, left seven unchanged, and worsened none. The 36%
wall-time increase is larger than the mean isolated NNLS time because the
small comparison includes runner and gate effects and was not a controlled
microbenchmark.

The result supports this narrow claim:

> On this deterministic representative subset, replacing linkage-derived
> branch lengths with a fixed-topology non-negative fit to standardized raw
> feature distances produced better downstream gate/traversal outcomes.

It does not establish:

- duplicate-safe adaptive diffusion;
- superiority across the full benchmark suite;
- calibrated branch-time semantics;
- feature-family invariance;
- stable edge lengths under pair sampling;
- out-of-sample additive-tree fit;
- a performance bound for larger `n`.

The runner records `branch_length_optimization_sec` internally, but the
canonical benchmark timing schema does not export it. Method-level CSVs cannot
currently separate diffusion, topology, NNLS, and gating overhead completely.

## Better implementation alternatives

Two executable alternatives were run on the same 14-case panel with the same
fixed-topology NNLS policy.

| Diffusion implementation | Successful | Exact K | Mean ARI | Median ARI | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| pydiffmap adaptive + NNLS | 13/14 | 10/13 successful | 0.872 | 1.000 | 11.19 s |
| graphtools fragmentation-guard adaptive K + NNLS | 14/14 | 8/14 | 0.837 | 0.984 | 12.98 s |
| repository fixed-Hamming kNN + NNLS | 14/14 | 11/14 | 0.863 | 1.000 | 11.60 s |

The repository fixed-Hamming implementation is the best immediate production
starting point:

- it uses the existing permissive NumPy/SciPy/scikit-learn stack;
- it completed `binary_perfect_4c` with exact K and ARI `1.0`;
- it achieved the highest all-requested-case exact-K count;
- it avoids pydiffmap private state and hidden KDE support;
- it is already supported by the `tbs_diffusion` runner when NNLS parameters
  are supplied.

It is not a drop-in mathematical replacement for variable-bandwidth
pydiffmap. Its kernel uses symmetric fixed-k Hamming-neighbor similarities, so
the method should have its own explicit registry name and benchmark contract.
It also still converts its sparse graph to dense and uses full `eigh`.

The graphtools route is the strongest ready-made adaptive alternative. Its
fragmentation-guard resolver is duplicate-aware and it completed all 14 cases.
However:

- graphtools emitted duplicate-distance warnings on this panel;
- the adapter still densifies the graph and uses the same full eigensolver;
- graphtools is GPLv2 and is intentionally isolated behind the
  `experimental-gpl` extra;
- its panel scores were not better than the simpler repository-owned route.

The best long-term implementation is therefore not another monolithic
diffusion-map dependency. It is a small repository-owned sparse kernel:

1. use scikit-learn only for explicit neighbor indices and distances;
2. define duplicate states and multiplicities before bandwidth estimation;
3. choose local scale from the r-th **positive** distance with a documented
   fallback;
4. construct and symmetrize a sparse affinity directly;
5. normalize it into a symmetric diffusion operator;
6. use `scipy.sparse.linalg.eigsh` for only the requested eigenpairs;
7. apply eigenvalue powers explicitly for diffusion time;
8. retain finite-support, connected-component, bandwidth, and duplicate
   diagnostics in method metadata.

scikit-learn's precomputed-affinity `SpectralEmbedding` is a maintained
engineering alternative, but it does not by itself preserve this method's
explicit diffusion-eigenvalue-time contract. A direct SciPy `eigsh` adapter is
clearer. Deeptime targets time-series dynamical models rather than this static
Hamming feature-graph problem and is not a closer replacement.

For the branch fit, SciPy `lsq_linear` remains the appropriate general sparse
bounded solver. Replacing it with dense `scipy.optimize.nnls` or a generic
convex-programming package would not solve the current methodological
problems. The better NNLS implementation is to retain `lsq_linear` while
adding:

- family-aware target builders;
- direct condensed-index pair sampling;
- a held-out pair partition;
- numeric termination and active-bound diagnostics;
- optional row weights or regularization;
- a sparse `LinearOperator` path if explicit incidence storage becomes a
  bottleneck.

### Installed comparison backends

The project now declares PyGSP `>=0.6.1` in the `diffusion` extra, deeptime
`>=0.4.5` in the `trajectory` extra, and PHATE `>=2.0.0` beside graphtools in
`experimental-gpl`. SciPy and scikit-learn remain core dependencies.
Functional smokes passed for explicit duplicate-aware neighbors, partial
`eigsh`, `SpectralEmbedding`, PHATE, PyGSP heat filtering, and deeptime TICA.

Datafold `2.0.2` cannot share the current locked environment: its resolver plan
would downgrade NumPy `2.3.4` to `1.26.4`, SciPy `1.16.3` to `1.11.4`, and
scikit-learn `1.7.2` to `1.2.2`. The scikit-learn downgrade violates the
project's declared `scikit-learn>=1.3.0` interface, so datafold was not added
or installed in the canonical environment.

## Diffusion methods by the three data forms

The repository's `FeatureSpace` contract defines exactly three feature
families: `bernoulli`, `categorical`, and `continuous`. Diffusion geometry
should dispatch on that contract before topology construction. A single
flattened Hamming or Euclidean route should not represent all three.

| Feature family | Appropriate diffusion construction | Strong library route | Main caution |
| --- | --- | --- | --- |
| Bernoulli | kNN graph from Hamming, Jaccard, Rogers–Tanimoto, or a model-based Bernoulli divergence; explicit duplicate multiplicities | repository-owned scikit-learn neighbor graph + SciPy sparse eigensolver | local scale must use positive support; rare-feature weighting must be explicit |
| Categorical | block-mismatch, block Hellinger, or multinomial/simplex distance per categorical feature, followed by a precomputed sparse affinity | repository-owned block distance feeding datafold or direct SciPy diffusion | never infer geometry by independently standardizing or Hamming-weighting flattened one-hot columns |
| Continuous | Gaussian/adaptive Gaussian diffusion map over standardized, whitened, or domain-selected coordinates | datafold for generic point clouds; Scanpy diffusion map for AnnData/scRNA | bandwidth, covariance, and density normalization must match the scientific model |

### Bernoulli and binary matrices

The strongest practical methods are:

1. **Fixed Hamming kNN diffusion.** This is already implemented locally and
   was the strongest robust route in the executed comparison.
2. **Duplicate-aware self-tuning diffusion.** Use an r-th positive Hamming
   distance per unique state, retain sample multiplicities, and build
   \(\exp[-d(x_i,x_j)^2/(\sigma_i\sigma_j)]\) only on explicit neighbors.
3. **Jaccard diffusion for sparse-presence data.** This is preferable when
   shared zeros should not count as similarity.
4. **Rogers–Tanimoto or Sokal–Michener diffusion.** These can be benchmarked
   when both matches and mismatches matter but Hamming's linear weighting is
   too simple.

Scanpy's neighbor builder accepts Hamming, Jaccard, Rogers–Tanimoto, and other
binary metrics and can build adaptive Gaussian connectivity. It is a
maintained implementation, but adopting the AnnData/Scanpy execution model for
generic benchmark matrices is heavier than owning the small sparse kernel
directly.

### Categorical blocks

Categorical diffusion should operate on feature blocks, not one-hot columns.
For samples \(x_i,x_j\), viable base distances include:

- mean per-feature mismatch;
- weighted mismatch with one weight per categorical feature;
- Hellinger distance between smoothed categorical distributions;
- multinomial/simplex Mahalanobis distance in the repository's drop-last
  chart.

The resulting precomputed distance or affinity can feed:

- a direct SciPy normalized diffusion operator;
- datafold through a custom/precomputed point-cloud kernel;
- scikit-learn `SpectralEmbedding(affinity="precomputed")` when only a
  Laplacian embedding is required.

Datafold documents sparse kernel matrices and explicitly notes that duplicate
zero distances must be stored in sparse kernels. That is a better contract
than pydiffmap's silent zero removal, but a custom categorical kernel is still
required.

### Continuous matrices

For generic continuous data, **datafold `DiffusionMaps`** is the strongest
library candidate. Version 2.0.1 documents:

- an explicit `time_exponent`;
- a selected `n_eigenpairs`;
- density-renormalization parameter `alpha`;
- symmetric conjugation for stable sparse Hermitian eigensolvers;
- sparse kernel matrices;
- Nyström out-of-sample transformation.

For scRNA applications already represented as AnnData, **Scanpy
`pp.neighbors(method="gauss")` plus `tl.diffmap`** is the better
application-level route. Scanpy exposes an adaptive Gaussian connectivity
kernel, sparse neighbor graphs, configurable neighbor backends, transition
eigenvalues, and diffusion components. It is not a reason to impose AnnData on
the core binary/categorical package.

**Palantir** provides a maintained MIT adaptive anisotropic diffusion map and
can accept a precomputed CSR kernel. It is useful for scRNA trajectory and
multiscale diffusion-space analyses, not as the general three-family
clustering backend.

### Structured graph adjacency

An adjacency matrix is numerically binary but is not an ordinary Bernoulli
feature matrix: columns are coupled by one graph. SBM and network cases should
diffuse on the supplied graph itself using:

- random-walk or symmetric normalized adjacency powers;
- normalized-Laplacian heat kernels;
- personalized PageRank/resolvent diffusion;
- multiscale graph wavelet or heat-kernel distances.

PyGSP is the best focused library candidate here. It supports partial graph
Fourier bases, normalized/combinatorial Laplacians, heat diffusion, and
Chebyshev approximations whose filtering cost is linear in graph edges. It
should not replace sample-manifold diffusion for independent feature rows.

### Related methods that are not direct replacements

| Method/library | What it computes | Appropriate use here |
| --- | --- | --- |
| PHATE/graphtools | diffusion-potential distance followed by MDS, with optional landmark approximation | trajectory visualization and geometry comparator; not the canonical tree metric without a separate validation |
| scikit-learn SpectralEmbedding | normalized-Laplacian eigenmap from built or precomputed affinity | maintained embedding engine when diffusion-time eigenvalue powers are not required |
| Palantir | adaptive anisotropic and multiscale diffusion for trajectories | scRNA trajectory applications |
| PyGSP Heat | graph heat-kernel filtering and graph spectral bases | direct network/adjacency inputs |
| deeptime | transfer-operator and dynamical models for time-series processes | temporal trajectories, not static binary/categorical sample matrices |

The resulting architecture should expose three geometry builders and one
structured-graph specialization:

```text
FeatureSpace.bernoulli   -> binary metric kernel
FeatureSpace.categorical -> block/simplex kernel
FeatureSpace.continuous  -> Gaussian/manifold kernel
graph adjacency app      -> direct graph diffusion
                         -> shared sparse diffusion coordinates
                         -> topology builder
                         -> family-aware NNLS target
```

## Ambiguous names and assumptions

The following code or result surfaces can lead a reader to the wrong
assumption:

| Surface | Likely wrong assumption | Actual contract |
| --- | --- | --- |
| `adaptive` | K and bandwidth automatically adapt safely to data degeneracy | global epsilon adapts; public K is fixed and duplicate handling is unsafe |
| `diffusion_time=3` | tree edges represent three units of stochastic time | diffusion time scales spectral coordinates only |
| `fixed_topology_nnls` | topology and branch lengths are jointly optimized | topology is frozen before NNLS |
| `squared_standardized_euclidean` | family-aware normalized distance | independent per-column standardization of the encoded matrix |
| NNLS `status="ok"` | tight optimal solution | SciPy accepted one of several termination conditions |
| `n_components=30` | only 30 eigenpairs are computed | full dense eigendecomposition is performed |
| `k` resolver | valid for the backend | valid only by sample count |
| canonical benchmark method | production-safe backend | registry prominence does not remove pydiffmap's unsupported duplicate behavior |
| ARI improvement | branch-time model validated | only final labels improved on one small labeled panel |

## Implemented construction-contract correction

The post-audit correction implemented the assumptions that can be enforced
without promoting an unsupported topology selector:

- diffusion methods now return one result containing coordinates, condensed
  distances, and backend evidence;
- diffusion linkage must be named explicitly at the runner and registry seam;
- `build_tree(...)` validates distance shape and values, unique leaves, one
  root, full binary shape, exact leaf coverage, and finite non-negative branch
  lengths;
- MAD endpoint roots are represented by a new binary root with a zero-length
  edge rather than by a degree-three root;
- natural label ordering, including numeric-aware numbered strings, makes
  exact-tie resolution invariant to input row order and reports exact
  distance/merge-height tie burden;
- NNLS refuses an omitted branch geometry;
- the retained adaptive NNLS route explicitly supplies the original
  distributional feature matrix and records its dual-geometry role;
- explicit continuous/Hamming, discrete/Euclidean, and mixed-family adaptive
  geometry mismatches fail before diffusion.

An alternative that fitted NNLS to the diffusion coordinates was executed
before rejection. On the same 14-case representative panel it completed
`13/14`, but exact-K fell from the retained method's `10/13` to `2/13`, mean
ARI fell from `0.8722` to `0.3132`, median ARI fell from `1.0` to `0.0`, and
runtime increased from about `11.6` to `29.6` seconds. Geometry coherence does
not by itself validate branch time. The retained dual geometry is now an
explicit hypothesis, not an implicit fallback.

## Recommended implementation order

### Priority 0 — correctness and contract

1. Remove the pydiffmap adaptive route from canonical/default status until its
   input contract is explicit.
2. Add a backend preflight that reports duplicate counts, per-row positive
   neighbor support, the effective pydiffmap minimum, and zero local
   bandwidths. Do not silently reinterpret samples or raise K.
3. Decide the duplicate-state method: unique-state kernel with multiplicities,
   a repository-owned duplicate-safe bandwidth, or explicit unsupported-input
   failure.
4. Preserve SciPy numeric termination status, active-bound counts, and
   convergence reason. Do not run branch-time gates with a nonconverged
   solution unless the caller explicitly opts into diagnostic replay.

### Priority 1 — methodological validity

1. Define family-aware NNLS targets using `FeatureSpace`: Bernoulli,
   categorical-block, continuous-covariance, and graph-specific targets should
   not share the current per-column fallback without justification.
2. Report held-out pair RMSE/MAE and repeat the fit across pair seeds when
   sampling is active.
3. Record edge-length stability, fraction at the lower bound, matrix rank or
   conditioning diagnostics, and gate-label stability.
4. Quantify same-data optimism through full-pipeline selected-hierarchy null
   reconstruction that repeats topology selection, branch fitting, and gate testing.

### Priority 2 — performance and observability

1. Keep the adaptive kernel sparse and use a partial symmetric eigensolver for
   the requested components.
2. Route neighbor parallelism through the repository worker policy rather than
   `n_jobs=-1`.
3. Sample condensed pair indices directly without replacement.
4. Export `branch_length_optimization_sec` and retain timings on failed rows.
5. Add scaling benchmarks over `n`, `p`, unique-row ratio, K, and pair cap.

### Required direct tests

- public K values below the pydiffmap hidden minimum;
- exact duplicate rows and all-identical rows;
- seven-state/80-sample `binary_perfect_4c` regression;
- finite local bandwidth and finite kernel checks;
- categorical encoding invariance and redundant one-hot columns;
- graph-family target behavior;
- solver cost-convergence versus optimality-convergence;
- deliberate nonconvergence and downstream fail-closed behavior;
- pair-seed branch-length and cluster-label stability;
- held-out pair residual;
- proof that partial eigensolver output matches the current dense route within
  tolerance.

## Files establishing the contract

- `tree_break_selection/space_separation/diffusion.py`
- `tree_break_selection/tree/feature_space.py`
- `tree_break_selection/tree/optimized_branch_lengths.py`
- `tree_break_selection/tree/construction/hierarchical.py`
- `benchmarks/shared/runners/tbs_diffusion_runner.py`
- `benchmarks/shared/runners/tbs_runner.py`
- `benchmarks/shared/runners/dispatch.py`
- `benchmarks/shared/runners/method_registry.py`
- `benchmarks/shared/util/time.py`
- `tests/core/test_adaptive_diffusion_runner.py`
- `tests/tree/test_optimized_branch_lengths.py`
- `tests/pipeline/51_test_dispatch_contract.py`
- `reports/benchmark_execution_audit_20260728.md`
