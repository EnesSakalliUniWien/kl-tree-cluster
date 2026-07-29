---
title: Repository Execution and Benchmark Map
type: analysis
status: reviewed
updated: 2026-07-28
sources:
  - README.md
  - applications/README.md
  - tree_break_selection/tree/README.md
  - tree_break_selection/space_separation/README.md
  - tree_break_selection/plot/README.md
  - benchmarks/README.md
  - benchmarks/shared/README.md
  - benchmarks/smoke/run_subset.py
  - benchmarks/regression/run_gate.py
  - benchmarks/full/run.py
  - reports/benchmark_execution_audit_20260728.md
tags:
  - architecture
  - benchmarks
  - applications
  - plotting
  - performance
---

# Repository Execution and Benchmark Map

## Summary

The current repository has one reusable method package, three dataset
application families, and purpose-specific benchmark orchestration. Production
method code lives under `tree_break_selection/`; applications adapt domain
data and compose reports; benchmark runners consume the same method interfaces
through `benchmarks/shared/`.

The fresh benchmark audit confirms executable health and records a complete
canonical full-suite execution, but it also exposes method-availability and
comparison-contract gaps. The 121-case, 10-method full run produced all 1,210
case-method rows, with 1,139 successful rows, 71 explicit skips, and no
error-status rows. K-means and spectral receive the true cluster count while
TBS-family methods infer it, so the aggregate ranking is descriptive rather
than an apples-to-apples model-selection result.

## Details

### Repository responsibility map

| Module or directory | Interface and responsibility | May depend on |
| --- | --- | --- |
| `tree_break_selection/tree/` | Tree feature contracts, distances, topology construction, rooting, branch lengths, and `PosetTree` | Core utilities; the current `PosetTree` convenience facade also reaches into hierarchy analysis |
| `tree_break_selection/space_separation/` | In-memory adaptive-cosine, invariant/equivariant, and diffusion geometry | Scientific Python dependencies, not applications or benchmarks |
| `tree_break_selection/hierarchy_analysis/` | Distribution annotation, edge and sibling gates, traversal, and cluster extraction | Tree and core-utility modules |
| `tree_break_selection/plot/` | Reusable colors, tree rendering, image panels, UMAP overlays, and file-safe backend selection | Reusable tree/result objects, not application paths |
| `applications/endotypes/` | GO/endotype matrix adaptation, subspace orchestration, annotation, and reports | Public method/plot interfaces and selected public benchmark runner contracts |
| `applications/scrna/` | Adult/fetal-pancreas adaptation, analyses, and dataset-specific plots | Public method interfaces and selected public benchmark contracts |
| `applications/mnist/` | Reconstruction and presentation of retained MNIST experiment results | MNIST experiment result contracts plus tree rendering |
| `benchmarks/shared/` | Case generation, method registry/dispatch, result records, metrics, and benchmark plots | Public method package |
| `benchmarks/smoke/`, `regression/`, `full/` | Fast representative execution, descriptive sensitive-case execution, and full report orchestration | `benchmarks/shared/` |
| `benchmarks/experiments/` | Narrow scientific experiments outside comparable full-suite claims | Shared contracts and public method interfaces |
| `benchmarks/validation/` | Evidence contracts, statistical validation, sweeps, and tree comparisons | Shared contracts and public method interfaces |
| `benchmarks/diagnostics/` | Failure localization and exploratory calibration/spectral analyses | Shared contracts, validation helpers, and public method internals under investigation |
| `benchmarks/cloud/` | Remote wrappers around maintained validation or diagnostic entry points | Validation, diagnostics, and shared contracts |

The primary dependency direction is:

```text
tree + space separation
        |
        v
hierarchy analysis + reusable plots
        |
        +------------------+
        |                  |
        v                  v
benchmarks/shared      applications
        |
        v
smoke / regression / full / experiments / validation / diagnostics
```

Applications are allowed to use stable public contracts from
`benchmarks/shared/` when the application itself is a benchmark workflow.
Reusable method modules do not import applications or benchmarks.

### Tree-construction map

The registered topology builders are explicitly ordered as `linkage`,
`neighbor_joining`, and `iqtree3`. Direct and diffusion-derived condensed
distances feed linkage; direct and graphtools distances can feed neighbor
joining; IQ-TREE consumes encoded feature-state alignments. Rooting and
fixed-topology branch-length fitting remain separate seams.

The complete geometry-to-topology matrix, including endotype, scRNA, and MNIST
routes, is maintained in [[tree-construction-method-map]].

### Plotting-engine map

| Engine | Public interface | Ownership |
| --- | --- | --- |
| Backend selection | `configure_matplotlib_backend()` | Selects `Agg` before lazy plotting imports unless `MPLBACKEND` is explicit |
| Cluster colors | `ClusterColorSpec`, `build_cluster_color_spec()` | Stable discrete cluster/color mapping |
| Tree rendering | `plot_tree_with_clusters()` | Rectangular/radial tree plots and gate annotations |
| Image panels | `draw_image_panel()` | Reusable report-panel composition |
| Selected-family multiscale UMAP | `benchmarks/diagnostics/calibration/selected/family/multiscale_umap.py` | Diagnostic existing-coordinate region overlays |
| Benchmark plots | `benchmarks/shared/plots/` | Case/result-record covers, embeddings, runtime pages, and PDF export |
| Application plots | `applications/*/plots/` or application pipelines | Dataset labels, layouts, and report composition |

Plotting reconstructs or consumes a selected tree for display. It is not a
fourth topology builder.

### Benchmark execution map and claim levels

| Surface | Current scope | Valid claim |
| --- | --- | --- |
| `benchmarks/smoke/run_subset.py` | 14 representative cases, canonical `tbs`, plots enabled | Development and plotting health only |
| `benchmarks/regression/run_gate.py` | 17 historically sensitive cases, configurable methods | Descriptive regression comparison; no enforced threshold |
| `benchmarks/full/run.py` | Selected mathematical suite, method grids, resumable output, reports | Comparable benchmark evidence when the run is locked and complete |
| `benchmarks/experiments/` | One scientific question per experiment | Question-specific evidence only |
| `benchmarks/validation/` | Statistical, sweep, tree, and manifest contracts | Evidence only within the explicit validation target |
| `benchmarks/diagnostics/` | Failure attribution and candidate-law exploration | Diagnostic evidence, not production promotion |

The shared execution path is:

```text
case recipe
  -> generate_case_data
  -> PreparedCaseInputs
  -> method registry and parameter grid
  -> method dispatch
  -> MethodRunResult
  -> metrics + canonical result row
  -> optional plots and reports
```

There are 121 full-suite cases across seven suite selectors and 32 registered
method IDs. The canonical default comparison contains three TBS-family methods
and seven external clustering baselines.

### Fresh benchmark check

At revision `9331319a`, the canonical full runner completed 121 cases for 10
methods in approximately 13 minutes 56 seconds, including plots and report
assembly. It produced 1,139 `ok` rows and 71 explicit `skip` rows with no
error-status rows. Mean ARI on available rows was `0.8639` for K-means,
`0.8338` for spectral, `0.8163` for Leiden, `0.8125` for Louvain, `0.8088` for
adaptive diffusion plus NNLS TBS, `0.7751` for canonical TBS, and `0.4971` for
fixed-neighbor diffusion TBS. On the 72 cases completed by every method,
adaptive diffusion plus NNLS TBS reached `0.8525`, canonical TBS `0.8305`, and
fixed-neighbor diffusion TBS `0.4523`.

Across the 92 cases jointly completed by the two diffusion variants, adaptive
diffusion plus NNLS improved 57, tied 22, and worsened 13, with mean paired ARI
change `+0.3594`. Against canonical TBS on 74 jointly completed cases, it
improved 29, tied 37, and worsened 8, with mean paired change `+0.0214`.
Canonical TBS had 28 missing empirical-null calibration skips; fixed-neighbor
diffusion had 13 continuous-geometry and 3 calibration skips; adaptive
diffusion plus NNLS had 13 continuous-preset, 11 pydiffmap duplicate/support,
and 3 calibration skips.

At revision `4a502615`:

- smoke: 11 `ok`, 3 fail-closed `skip`, mean ARI `0.898`, median ARI
  `0.970`, exact K `8/14` requested or `8/11` successful;
- adaptive pydiffmap diffusion: 13 `ok`, 1 duplicate-row `skip`, mean ARI
  `0.5445`, median ARI `0.7054`, exact K `4/13` successful;
- adaptive pydiffmap diffusion plus fixed-topology NNLS: 13 `ok`, 1
  duplicate-row `skip`, mean ARI `0.8722`, median ARI `1.0000`, exact K
  `10/13` successful; six case ARIs improved, seven were unchanged, and none
  worsened;
- descriptive regression: 11 `ok`, 6 fail-closed `skip`, mean ARI `0.5957`,
  median ARI `0.6803`, exact K `4/17` requested or `4/11` successful;
- focused core/integration/pipeline/tree/visualization tests: `288 passed`.

The nine canonical-TBS smoke/regression skips lacked admissible strict-null or
stopped-edge positive-weight records for the sibling-inflation model. The
implementation correctly refused to treat selected non-null records as
calibration support. The separate adaptive-diffusion skip occurs before NNLS:
`binary_perfect_4c` has 80 samples but only seven distinct rows, and
pydiffmap's adaptive bandwidth estimator receives insufficient nonzero
neighbor support for the configured `k`.

### Ambiguity and performance findings

`regression/run_gate.py` is a descriptive runner despite its gate name. It
does not assert score floors, status ceilings, or case-specific expectations.
Its summary also prints exact K over every requested row, including skips.
Both facts can lead a caller to the wrong release conclusion.

Skipped benchmark rows contain no stage timings, so result files lose the
runtime spent before a fail-closed exception. Successful rows show edge and
sibling gates dominate tree construction on this small suite.

The canonical full comparison also combines oracle-K and inferred-K methods:
K-means and spectral are passed the true cluster count, while TBS, Leiden,
Louvain, and density methods infer a partition size. Availability differs too,
so ranking available-row means rewards methods that skip their incompatible or
fail-closed cases. Reports must show coverage and common-case contrasts beside
aggregate means.

The adaptive diffusion + NNLS default fixes Hamming geometry for the full
suite. Its 13 native-continuous skips therefore describe the preset's scope,
not adaptive diffusion's general continuous-data capability. The full suite
also expands the pydiffmap positive-neighbor failure from one representative
case to 11 cases.

Adaptive diffusion and its NNLS variant share geometry, average-linkage
topology, and rooting. NNLS only refits edge lengths and activates normalized
branch-time variance; its large representative-subset improvement is therefore
a branch-time/gating effect, not a different-topology effect. It added about
36% wall time, but `branch_length_optimization_sec` is not exported through the
canonical result-row schema.

Profiling also shows a repeated-contract bottleneck: when callers omit an
explicit `FeatureSpace`, each Wald contrast reconstructs and validates a full
Bernoulli feature space. One regression profile created 6,920 feature spaces
and invoked about 3.3 million feature-block validations. The feature-space
contract should be resolved once per run or cached as an immutable fallback.

The adaptive neighbor resolver bounds `k` by sample count only. Duplicate-heavy
inputs can have much smaller nonzero-neighbor support, causing pydiffmap's
bandwidth estimator to fail before tree construction. The smoke entrypoint also
executes on import because `run_subset.py` has no `main()` guard; it is a script,
not a safe constants module.

The remaining reverse dependency is the `PosetTree` convenience interface:
`tree/poset_tree.py` imports hierarchy-analysis defaults and lazily imports
`TreeDecomposition` and cluster-assignment helpers. Tree construction itself
is independent, but this facade prevents the whole tree module from being a
strict lower layer.

## Evidence

- Static import analysis found dependencies from applications and benchmarks
  into the reusable package and no imports in the reverse direction.
- `tree_break_selection/plot/__init__.py` configures the backend before lazily
  loading public plotting engines.
- `benchmarks/shared/README.md` and live imports agree on the case,
  generation, preparation, dispatch, execution, result, and plot sequence.
- `reports/benchmark_execution_audit_20260728.md` records commands, outcomes,
  full-suite results, skip reasons, timing coverage, and profile attribution.
- The focused 288-test run covered the reusable core, benchmark integration,
  pipeline contracts, tree construction, and visualization surfaces.

## Links

- [[tree-construction-method-map]]
- [[method-application-and-plot-seams-20260727]]
- [[benchmark-pipeline-contract]]
- [[repository-hygiene-and-completion-audit-20260727]]
- [[spectral-backend-runtime-diagnostic]]

## Open Questions

- Should the descriptive regression runner be renamed, or should locked
  case-specific thresholds be justified and enforced?
- Should the remaining `PosetTree.build_sample_cluster_assignments()` facade
  move fully to hierarchy analysis now that decomposition itself has one
  explicit owner?
- Should run-scoped feature-space resolution be mandatory at the benchmark
  dispatch seam?
- How should adaptive diffusion bound or aggregate duplicate-heavy inputs
  before pydiffmap bandwidth fitting?
- Should NNLS optimization time become a canonical result-record field?
- How should partial stage timings survive fail-closed method exceptions?
