# Benchmark Execution Audit — 2026-07-28

## Scope

The initial audit was run from clean `dev` revision `4a502615` after the
repository was reduced to the `main` and `dev` branches. It checks executable
health, benchmark claim boundaries, fail-closed behavior, and the dominant
runtime paths. A subsequent section records the canonical full-suite run at
revision `9331319a`; neither run replaces a statistical validation study.

The repository virtual environment was used directly because the managed
execution environment could not read the global `uv` cache.

## Executed checks

### Representative smoke benchmark

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/tbs-matplotlib \
TBS_N_JOBS=1 .venv/bin/python -m benchmarks.smoke.run_subset
```

The runner selected 14 of the 121 registered cases.

| Outcome | Result |
| --- | ---: |
| Successful rows | 11 |
| Explicit fail-closed skips | 3 |
| Exact K, as printed by the runner | 8/14 |
| Exact K among successful rows | 8/11 |
| Mean ARI among rows with labels | 0.898 |
| Median ARI among rows with labels | 0.970 |

The three skipped cases were `binary_perfect_4c`,
`binary_many_features`, and `overlap_heavy_4c_small_feat`. Each stopped
because the selected sibling-inflation model had no strict-null or stopped-edge
positive-weight calibration records. Selected non-null records were correctly
rejected as empirical-null support.

This is a development smoke result. Its aggregate score is not full-suite
method evidence.

### Descriptive regression benchmark

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/tbs-matplotlib \
TBS_N_JOBS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
.venv/bin/python -m benchmarks.regression.run_gate \
  --output-dir /private/tmp/tbs-regression-gate-20260728 \
  --methods tbs
```

The 17-case run completed in 13.29 seconds.

| Outcome | Result |
| --- | ---: |
| Successful rows | 11 |
| Explicit fail-closed skips | 6 |
| Exact K, as printed by the runner | 4/17 |
| Exact K among successful rows | 4/11 |
| Mean ARI among rows with labels | 0.5957 |
| Median ARI among rows with labels | 0.6803 |

All six skips had the same missing empirical-null support condition as the
smoke skips. The runner is currently descriptive: it defines no acceptance
threshold, performs no assertion over scores or status counts, and exits
successfully after writing results. Its `run_gate.py` name must therefore not
be interpreted as an enforced regression gate.

### Adaptive diffusion with and without NNLS branch time

The same 14 representative cases were run without plots through
`tbs_diffusion_adaptive` and `tbs_diffusion_adaptive_nnls`.

| Outcome | Adaptive diffusion | Adaptive diffusion + NNLS |
| --- | ---: | ---: |
| Wall time | 8.22 s | 11.19 s |
| Successful rows | 13 | 13 |
| Explicit skips | 1 | 1 |
| Exact K among successful rows | 4/13 | 10/13 |
| Mean ARI | 0.5445 | 0.8722 |
| Median ARI | 0.7054 | 1.0000 |

NNLS improved ARI on six cases, left seven unchanged, and worsened none in this
small deterministic comparison. The largest changes were three cases that
moved from one cluster and ARI `0.0` to exact recovery and ARI `1.0`.

Both variants use the same adaptive pydiffmap distance, average-linkage
topology, and linkage root. NNLS does not discover a different topology. It
fits non-negative edge lengths on that fixed topology and activates normalized
branch-length variance in the gates. The observed clustering improvement is
therefore a branch-time/gating effect.

NNLS increased wall time by about 36% in this run. Although the runner records
`branch_length_optimization_sec` internally, the canonical benchmark result-row
schema does not export that column. The CSV cannot directly attribute the NNLS
overhead.

Both variants failed on `binary_perfect_4c` with
`kth(=6) out of bounds (1)`. The case has 80 samples but only seven distinct
feature rows. `resolve_neighbor_search_k()` bounds pydiffmap's `k` against
sample count, not effective distinct-neighbor support. Pydiffmap's adaptive
bandwidth estimator then requests more nonzero neighbor distances than a
collapsed row provides. This is a degenerate-input handling bug in the
adaptive diffusion route, not an NNLS failure.

### Focused contract tests

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/tbs-matplotlib \
TBS_N_JOBS=1 .venv/bin/python -m pytest -q \
  tests/core tests/integration tests/pipeline tests/tree tests/visualization
```

All 288 tests passed. Warnings were limited to logical-core fallback,
duplicate-point warnings in a graphtools fixture, a disconnected spectral
fixture, and an expected empty-slice warning in a consensus test.

## Performance profile

The regression suite was also run through `cProfile`. Profiling increased the
wall time to 27.69 seconds, so those absolute times are not compared with the
unprofiled run. The cumulative call graph is still useful for attribution:

| Call path | Calls | Cumulative profiled time |
| --- | ---: | ---: |
| `run_gate_annotation_pipeline` | 17 | 25.38 s |
| `compute_whitened_wald_contrast` | 12,758 | 12.61 s |
| child-parent divergence annotation | 17 | 10.59 s |
| fixed-subspace sibling annotation | 11 | 8.73 s |
| child-parent tree testing | 17 | 8.48 s |
| sibling pair-record collection | 17 | 4.44 s |
| spectral decomposition | 17 | 1.94 s |

The avoidable hotspot is fallback feature-space resolution. When no explicit
`FeatureSpace` is supplied, `_resolve_distribution_feature_space()` constructs
a complete Bernoulli feature space for each contrast. The profile recorded:

- 6,920 calls to `bernoulli_feature_space_from_columns()`;
- about 3.3 million `FeatureBlock` construction/validation calls;
- repeated full-vector and full-matrix validation inside the same tree run.

For the 11 successful rows, the non-overlapping parent stage totals show that
tree construction is not the small-suite bottleneck:

| Recorded stage | Sum over successful rows |
| --- | ---: |
| Edge gate | 1.251 s |
| Sibling gate | 1.136 s |
| Tree construction | 0.034 s |
| Distribution population | 0.039 s |
| Traversal | 0.0004 s |

Substep columns are nested inside their parent stages and must not be added to
the parent totals.

## Benchmark contract findings

1. **The executable paths are healthy.** Smoke execution, plotting through the
   noninteractive backend, regression execution, and 288 focused tests all
   complete.
2. **Calibration coverage is incomplete.** Three smoke cases and six
   regression cases fail closed because the production sibling-inflation path
   lacks admissible empirical-null support.
3. **`regression/run_gate.py` is not an enforced gate.** It needs scientifically
   justified thresholds or a non-gate name before CI can use it as a release
   decision.
4. **Printed exact-K denominators include skipped rows.** The output should
   report both all requested cases and successful rows to avoid treating
   unsupported cases as ordinary clustering misses.
5. **Failed rows lose stage timings.** Every timing field is null on the six
   skipped regression rows, so current result files cannot localize time spent
   before a fail-closed exception.
6. **Repeated feature-space construction is the main demonstrated avoidable
   runtime cost.** A run-scoped explicit feature-space contract or cached
   immutable fallback would remove millions of validation operations.
7. **The adaptive diffusion route does not guard duplicate-row support.**
   Neighbor `k` is valid for the sample count but can be invalid for
   pydiffmap's effective nonzero-neighbor graph.
8. **The smoke runner is not import-safe.** Importing
   `benchmarks.smoke.run_subset` starts the full benchmark and plotting
   immediately because the module has no `main()` guard.

## Conclusion

The benchmark infrastructure is executable and structurally well tested.
Adaptive diffusion with fixed-topology NNLS is materially stronger than raw
linkage branch time on the representative subset, but the comparison also
exposes a duplicate-row failure and missing NNLS timing export. The current
canonical TBS path is not benchmark-complete: unsupported calibration contexts
remain visible, and the descriptive regression runner does not enforce a
performance or quality threshold. Those are method and benchmark contract
issues, not directory or plotting-backend failures.

## Canonical full-suite run

The canonical full runner was executed from clean `dev` revision `9331319a`
with the noninteractive plotting backend and single-threaded numerical-library
settings:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=<temporary-directory> TBS_N_JOBS=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
uv run python -m benchmarks.full.run
```

The run completed in approximately 13 minutes 56 seconds, including
relationship analysis and PDF assembly. It generated all 121 cases for the 10
canonical methods, producing 1,210 unique case-method rows, 1,139 `ok` rows,
71 explicit `skip` rows, and no error-status rows. The generated result
directory is
`benchmarks/results/run_20260728_163153Z_full/`; it contains the canonical CSV,
performance grids, relationship tables and report, failure diagnosis, 292
audit files, 132 PDFs, and the consolidated report.

| Method | OK / 121 | Mean ARI on OK rows | Median ARI | Exact-K rate on OK rows |
| --- | ---: | ---: | ---: | ---: |
| K-means | 121 | 0.8639 | 1.0000 | 1.0000 |
| Spectral | 121 | 0.8338 | 1.0000 | 1.0000 |
| Leiden | 121 | 0.8163 | 1.0000 | 0.7603 |
| Louvain | 121 | 0.8125 | 1.0000 | 0.7769 |
| Adaptive diffusion + NNLS TBS | 94 | 0.8088 | 1.0000 | 0.7447 |
| Canonical TBS | 93 | 0.7751 | 0.9916 | 0.6452 |
| HDBSCAN | 121 | 0.6005 | 0.8204 | 0.4545 |
| DBSCAN | 121 | 0.5729 | 0.7806 | 0.5785 |
| OPTICS | 121 | 0.5242 | 0.6839 | 0.5372 |
| Fixed-neighbor diffusion TBS | 105 | 0.4971 | 0.5826 | 0.2571 |

These aggregate ranks are not an apples-to-apples model-selection result.
K-means and spectral are configured with `n_clusters=true`, so their exact-K
rate is guaranteed by the benchmark input. TBS-family methods must infer the
cluster count, and their available-row means exclude unsupported or
fail-closed cases. On the 72 cases where all methods returned labels, mean ARI
was 0.8955 for K-means, 0.8939 for Leiden, 0.8897 for Louvain, 0.8632 for
spectral, 0.8525 for adaptive diffusion + NNLS TBS, 0.8305 for canonical TBS,
and 0.4523 for fixed-neighbor diffusion TBS.

Across the 92 cases jointly completed by both diffusion TBS variants, adaptive
diffusion plus NNLS improved ARI in 57 cases, tied in 22, and worsened in 13,
for a mean paired change of `+0.3594`. This is a much broader and less uniformly
favorable result than the earlier 14-case smoke comparison. Against canonical
TBS on 74 jointly completed cases, adaptive diffusion plus NNLS improved 29,
tied 37, and worsened 8, for a mean paired change of `+0.0214`.

The 71 skips expose three distinct contracts:

- canonical TBS skipped 28 cases because the sibling-inflation model had no
  admissible strict-null or stopped-edge empirical-null support;
- fixed-neighbor diffusion TBS skipped 13 native-continuous cases because its
  Hamming geometry is unsupported, plus 3 calibration-support cases;
- adaptive diffusion + NNLS skipped the same 13 native-continuous cases because
  its canonical preset fixes `metric=hamming`, 11 duplicate/support-degenerate
  cases inside pydiffmap, and 3 calibration-support cases.

The pydiffmap failure is therefore broader than `binary_perfect_4c`: the
full-suite run found 11 affected duplicate-heavy or low-support cases. The
adaptive preset's continuous skips are a benchmark-configuration scope issue,
not evidence that adaptive Euclidean diffusion cannot process continuous data.

CSV integrity checks found no duplicate case-method cells, no non-finite ARI
among `ok` rows, no finite ARI among skipped rows, no label-count mismatch, and
no ARI outside its valid range. Recorded top-level TBS stages exclude diffusion
construction and NNLS fitting because those timings are not exported. On the
92 jointly successful diffusion cases, those recorded stages averaged 0.5037
seconds for fixed-neighbor diffusion and 0.5206 seconds for adaptive diffusion
plus NNLS; this small difference must not be reported as end-to-end NNLS
overhead.
