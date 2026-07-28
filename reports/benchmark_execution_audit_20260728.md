# Benchmark Execution Audit — 2026-07-28

## Scope

This audit was run from clean `dev` revision `4a502615` after the repository
was reduced to the `main` and `dev` branches. It checks executable health,
benchmark claim boundaries, fail-closed behavior, and the dominant runtime
paths. It does not replace a locked full benchmark or statistical validation
study.

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
