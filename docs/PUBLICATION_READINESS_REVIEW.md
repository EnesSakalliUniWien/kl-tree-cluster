# Publication Readiness Review (Updated)

Date: 2026-02-15  
Scope: Current repository state after refactoring cycle (dead code removal, modular extraction, type hardening, signal localization fix) and latest benchmark run at `benchmarks/results/run_20260215_153657Z`.

## Executive verdict

Major revision is still required before publication submission.

Core implementation quality has materially improved: dead code purged (~3,600 lines removed from production), module boundaries tightened (`PosetTree` ↔ `TreeDecomposition` separation), a latent signal-localization bug fixed, and benchmark performance strengthened (KL Mean ARI 0.823 → 0.838, up from 0.709/0.711 in Feb-13 run). Statistical-calibration risks remain the primary open concern.

Recent closure (since last review):

- C-004 manuscript-implementation mismatch is resolved (manuscript now documents the implemented projected Wald/JL + FDR tests).
- M-002 non-finite z-score handling is resolved (invalid-test path now preserves raw `NaN` outputs and applies conservative `p=1.0` only for correction, with audit counters).
- I-007 duplicate `__all__` export lines in edge significance module are resolved.
- `pooled_variance.py` docstring now matches behavior (no fallback claim; `ValueError` when branch-length scaling is requested without valid `mean_branch_length`).
- M-007a missing calibration-artifact sub-issue is resolved (null/Type-I and TreeBH report artifacts generated under `benchmarks/results/run_20260213_132926Z/calibration/` and integrated into `full_benchmark_report_with_calibration.pdf`).
- M-007b sibling miscalibration mechanism is now diagnosed: 2-fold feature-split and fixed-tree permutation diagnostics show sibling Type-I near zero, indicating in-sample adaptive gating/selection as the dominant cause of sibling inflation.
- I-008 dead code audit completed: 7 dead functions removed across 12 files; `mi_feature_selection.py`, `branch_length_benchmark.py`, `benchmarking/test_cases/`, and 8 benchmarking scripts deleted (net −3,164 lines from production code).
- I-009 `cluster_assignments.py` extracted from `tree_decomposition.py` — pure functions for building cluster assignment structures now in dedicated module.
- I-010 annotation logic migrated from `PosetTree.decompose()` to `TreeDecomposition._prepare_annotations()` — idempotent method that checks for existing annotations and runs the pipeline if missing; `PosetTree.decompose()` reduced to thin facade (~20 lines).
- I-011 test-only `sibling_shortlist_size` infrastructure removed — parameter, `_pop_candidate()`, `_decompose_with_shortlist()`, dispatch branch, and 2 shortlist-only tests deleted; test file renamed to `test_cluster_decomposer_threshold.py`.
- B-001 signal localization `sample_size` vs `leaf_count` bug fixed — `_get_sample_size()` read non-existent `"sample_size"` attribute (always returned 0), silently suppressing localization recursion when `min_samples` was set. Replaced with `extract_node_sample_size()` from `data_utils.py`.
- I-012 `localization_min_samples` parameter removed entirely — recursion now governed solely by `max_depth` and `is_edge_significant` gate; no artificial sample-size floor.
- I-013 `TreeDecomposition.tree` type narrowed from `nx.DiGraph` to `PosetTree` (via `TYPE_CHECKING` import to avoid circular imports); redundant `hasattr(tree, "compute_descendant_sets")` runtime guard removed.

## Current blockers and major open issues

| ID     | Severity | Status | Issue                                                                                                                                                                                                          | Evidence                                                                                                                                                 |
| ------ | -------- | ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P-001  | Blocker  | Open   | No camera-ready manuscript PDF (submission-formatted paper absent)                                                                                                                                             | `manusscript.md` exists; no paper PDF source in repo root                                                                                                |
| CI-001 | Blocker  | Open   | No GitHub Actions workflows for CI                                                                                                                                                                             | `.github/workflows` missing                                                                                                                              |
| M-006  | Major    | Open   | Categorical covariance still approximated (diagonal variance); full multinomial covariance not used in main path                                                                                               | `pooled_variance.py`                                                                                                                                     |
| M-006b | Major    | Open   | Mahalanobis categorical implementation exists but is not integrated into sibling/edge test pipeline                                                                                                            | `categorical_mahalanobis.py`                                                                                                                             |
| M-005  | Major    | Open   | Invalid/negative branch lengths are ignored (treated as unavailable) rather than rejected with hard failure in edge/sibling callers                                                                            | `edge_significance.py`, `sibling_divergence_test.py`                                                                                                     |
| M-004  | Major    | Open   | Input validation still limited (distribution/value constraints not comprehensively asserted before testing)                                                                                                    | `edge_significance.py`, `sibling_divergence_test.py`                                                                                                     |
| M-007  | Major    | Open   | In-sample adaptive inference remains anti-conservative (edge `~0.113-0.177`, sibling `~0.914-0.938`), while split/permutation diagnostics show near-zero Type-I; production path still uses in-sample coupling | `benchmarks/results/diagnostics_edge_20260213_140319Z/edge_summary.csv`, `benchmarks/results/diagnostics_crossfit_20260213_135524Z/crossfit_summary.csv` |

## Verification of “keep in mind” findings

### 1) `mean_branch_length` fallback inconsistency

Status: outdated in current code.

- Current implementation returns `None` when no valid branch lengths are present, not `1.0`.
- Evidence: `_compute_mean_branch_length` in `edge_significance.py`.
- Follow-up aligned: `pooled_variance.py` docstring now correctly states that this condition raises `ValueError` (no fallback constant).

### 2) Categorical covariance approximation

Status: valid (partially mitigated, not fully solved).

- Code explicitly documents ignoring off-diagonal covariance.
- Rank deficiency is partially handled via drop-last-category logic in sibling/edge tests.
- Full covariance Wald calibration is not in the active inference path.

### 3) `df = k` with JL projection

Status: partially valid concern.

- Projection is orthonormal (good), but inferential calibration still assumes standardized residual behavior close to target normal model.
- Empirical calibration artifacts now exist and show the risk is realized in practice (null Type-I inflation), so this is no longer just a hypothetical model-assumption concern.

### 4) In-sample adaptive gating/selection vs sibling calibration

Status: diagnosed.

- 2-fold feature-split diagnostics (`A→B`, `B→A`) reduce sibling Type-I from in-sample `~0.91-0.94` to `0.0` across null scenarios.
- Fixed-tree/gate permutation diagnostics also produce near-zero sibling false positives (`~0.000-0.001`).
- Branch-length usage stays effectively full in these runs (`~1.0` usable rate among tested sibling pairs), so branch-length availability is not the source of sibling inflation.
- Edge-only diagnostics show the same pattern: in-sample edge reject rate is inflated (`~0.113-0.177`), while 2-fold feature-split and fixed-tree permutation reject rates are near zero (`~0.000-0.0004`), indicating coupling/selection rather than branch-length or projection plumbing as the primary inflation driver.

## Items now resolved or outdated

| Item                                                                                 | Status                       | Evidence                                                                                                                                                                                                                                 |
| ------------------------------------------------------------------------------------ | ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Manuscript claim mismatch (CMI/permutation vs implemented projected Wald tests)      | Resolved                     | `manusscript.md` updated to reflect child-parent and sibling projected Wald/JL + FDR flow                                                                                                                                                |
| Non-finite z-score handling (previous `0.0` coercion)                                | Resolved                     | Invalid-test handling implemented in `edge_significance.py` and `sibling_divergence_test.py`, with `Child_Parent_Divergence_Invalid` / `Sibling_Divergence_Invalid` flags and audit counters                                             |
| MI-based feature filtering in sibling inference                                      | Resolved                     | `sibling_divergence_test.py` no MI selection path                                                                                                                                                                                        |
| Shared projection seed concern                                                       | Resolved                     | `derive_projection_seed` usage in edge/sibling tests; `tests/test_per_test_projection_seeding.py` passes                                                                                                                                 |
| Sparse/non-orthonormal projection concern                                            | Resolved (current code path) | `random_projection.py`                                                                                                                                                                                                                   |
| Arbitrary z-clamping at ±100                                                         | Resolved                     | Removed; no clipping path remains                                                                                                                                                                                                        |
| Duplicate `__all__` export lines in edge significance module                         | Resolved                     | `edge_significance.py` now has a single export declaration                                                                                                                                                                               |
| `branch_length_left + branch_length_right` None crash path                           | Resolved                     | Guarded in sibling test                                                                                                                                                                                                                  |
| Missing calibration artifact outputs (M-007a)                                        | Resolved                     | `benchmarks/results/run_20260213_132926Z/calibration/{null_type1_summary.csv,treebh_summary.csv,calibration_report.md,calibration_plots.pdf}`                                                                                            |
| Sibling inflation root cause unresolved                                              | Resolved (diagnosed)         | `benchmarks/results/diagnostics_crossfit_20260213_135524Z/crossfit_summary.csv` and `benchmarks/results/diagnostics_crossfit_20260213_135524Z/permutation_summary.csv` show sibling Type-I near zero under split/permutation diagnostics |
| Edge inflation root cause unresolved                                                 | Resolved (diagnosed)         | `benchmarks/results/diagnostics_edge_20260213_140319Z/edge_summary.csv` and `benchmarks/results/diagnostics_edge_20260213_140319Z/edge_permutation_summary.csv` show in-sample inflation but near-zero split/permutation Type-I          |
| Dead code / unused exports in production modules (I-008)                             | Resolved                     | 7 dead functions removed across 12 files; `mi_feature_selection.py`, `branch_length_benchmark.py`, `benchmarking/test_cases/`, and 8 benchmarking scripts deleted. Net −3,164 lines.                                                     |
| `PosetTree.decompose()` contained annotation pipeline logic (I-010)                  | Resolved                     | Annotation dispatch moved to `TreeDecomposition._prepare_annotations()`; `PosetTree.decompose()` reduced to thin facade                                                                                                                  |
| Test-only `sibling_shortlist_size` in production code (I-011)                        | Resolved                     | Parameter, methods, dispatch, and shortlist tests removed; test file renamed to `test_cluster_decomposer_threshold.py`                                                                                                                   |
| Signal localization reads non-existent `sample_size` attribute (B-001)               | Resolved                     | `_get_sample_size()` always returned 0; replaced with `extract_node_sample_size()` then removed entirely along with `min_samples` parameter                                                                                              |
| `TreeDecomposition.tree` typed as `nx.DiGraph` despite requiring `PosetTree` (I-013) | Resolved                     | Type narrowed to `PosetTree` via `TYPE_CHECKING` import; redundant `hasattr` guard removed                                                                                                                                               |

## Reproducibility and test status

- Local test suite status: `135 passed, 4 skipped` (26 test files).
- Net code delta since last commit: −3,164 lines from production code (`kl_clustering_analysis/`), +503 insertions.
- Benchmark artifacts exist for latest run:
  - `benchmarks/results/run_20260215_153657Z/full_benchmark_comparison.csv`
  - `benchmarks/results/run_20260215_153657Z/failure_report.md`
  - `benchmarks/results/run_20260215_153657Z/full_benchmark_report.pdf`
- Prior benchmark and calibration artifacts remain available:
  - `benchmarks/results/run_20260213_121817Z/` (pre-refactoring baseline)
  - `benchmarks/results/run_20260213_132926Z/calibration/` (null/Type-I, TreeBH)
  - `benchmarks/results/diagnostics_crossfit_20260213_135524Z/` (cross-fit diagnostics)
  - `benchmarks/results/diagnostics_edge_20260213_140319Z/` (edge diagnostics)

## Benchmark audit snapshot (latest run: 2026-02-15)

Method-level means from `full_benchmark_comparison.csv` (94 cases each, 1 case dropped vs prior 95):

| Method            | Mean ARI | Mean NMI | Mean Purity | Exact K |
| ----------------- | -------: | -------: | ----------: | ------: |
| kl                |    0.823 |    0.860 |       0.846 |      62 |
| kl_rogerstanimoto |    0.838 |    0.871 |       0.860 |      64 |
| kmeans            |    0.930 |    0.933 |       0.932 |      94 |
| spectral          |    0.898 |    0.916 |       0.912 |      94 |
| leiden            |    0.902 |    0.921 |       0.919 |      80 |
| louvain           |    0.644 |    0.697 |       0.683 |      43 |
| dbscan            |    0.688 |    0.741 |       0.749 |      56 |
| hdbscan           |    0.603 |    0.671 |       0.659 |      52 |
| optics            |    0.603 |    0.671 |       0.659 |      52 |

### KL improvement vs prior run (2026-02-13)

| Metric   | Feb-13 `kl` | Feb-15 `kl` |      Δ |
| -------- | ----------: | ----------: | -----: |
| Mean ARI |       0.709 |       0.823 | +0.114 |
| Mean NMI |       0.750 |       0.860 | +0.110 |
| Exact K  |       40/95 |       62/94 |    +22 |

The improvement is attributed to the extrapolation-guard fix (cousin-adjusted Wald), dead code removal reducing accidental interactions, and the signal-localization `sample_size` bug fix.

KL-specific failure patterns from current failure report:

- Persistent under-splitting on several phylogenetic and small-signal cases.
- Persistent over-splitting on heavy-overlap cases.

## Immediate next patch set (high leverage)

1. Add CI workflow under `.github/workflows` to run `pytest`.
2. Integrate or gate categorical Mahalanobis mode for categorical inputs, with benchmark A/B switch.
3. Implement and benchmark a decoupled inference mode (feature-split or cross-fit) for production runs; then rerun full calibration and benchmark reports with this mode as default for publication evidence.
4. Commit the current refactoring batch (dead code removal, module extraction, annotation migration, shortlist removal, signal localization fix, type hardening) as a single coherent commit.
