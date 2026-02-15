# Method Assessment: KL-Tree Statistical Testing

Date: 2026-02-12  
Scope: child-parent divergence test, sibling divergence test, random projection, and multiple-testing control in `kl_clustering_analysis/hierarchy_analysis/statistics/`.

## Verdict

Major revision required before publication-grade inferential claims.

- Statistical soundness: 2.5/5
- Implementation robustness: 3.0/5
- Reproducibility confidence: 2.5/5
- Method publication readiness: Not ready

## Findings

### M-001

- Severity: Blocker
- Status: Verified
- Problem: Shared random projection seed across tests.
- Evidence:
  - `kl_clustering_analysis/config.py:69`
  - `kl_clustering_analysis/hierarchy_analysis/statistics/kl_tests/edge_significance.py:258`
  - `kl_clustering_analysis/hierarchy_analysis/statistics/sibling_divergence/sibling_divergence_test.py:168`
- Why this matters: Shared projection randomness couples hypothesis tests and can distort uncertainty calibration.
- Exact fix: Derive deterministic per-test seeds from `(base_seed, test_id)` and pass unique seed per edge/sibling test.
- Estimated effort: M
- Re-check criterion: Log and verify seed uniqueness per hypothesis; rerun null calibration.

### M-002
- Severity: Major
- Status: Verified
- Problem: Non-finite z-scores are replaced with `0.0`.
- Evidence:
  - `kl_clustering_analysis/hierarchy_analysis/statistics/kl_tests/edge_significance.py:186`
  - `kl_clustering_analysis/hierarchy_analysis/statistics/sibling_divergence/sibling_divergence_test.py:152`
- Why this matters: Silent repair biases tests toward null and hides instability.
- Exact fix: Mark test as uncomputable (`NaN`) and store explicit failure reason; do not coerce to zero.
- Estimated effort: M
- Re-check criterion: No silent non-finite coercion paths remain; failures are explicit and counted.

### M-003
- Severity: Major
- Status: Addressed
- Problem: Historical issue where MI-based feature selection and hypothesis testing used the same data.
- Evidence:
  - MI filtering path removed from sibling divergence inference.
- Why this matters: Post-selection inference bias can make p-values anti-conservative.
- Exact fix: Disable MI filtering in inferential mode.
- Estimated effort: L
- Re-check criterion: No MI-based preselection remains in inferential pipeline.

### M-004
- Severity: Major
- Status: Verified
- Problem: Missing strict validation for shape/probability constraints on distributions.
- Evidence:
  - `kl_clustering_analysis/hierarchy_analysis/statistics/kl_tests/edge_significance.py:126`
  - `kl_clustering_analysis/hierarchy_analysis/statistics/sibling_divergence/sibling_divergence_test.py:135`
- Why this matters: Invalid/broadcasted inputs can produce plausible but wrong p-values.
- Exact fix: Enforce same shape, finite values, `[0,1]` bounds, and categorical row sums approx 1.
- Estimated effort: M
- Re-check criterion: Add unit tests for malformed inputs; each fails fast with clear error.

### M-005
- Severity: Major
- Status: Verified
- Problem: Edge branch-length adjustment does not reject negative branch lengths.
- Evidence:
  - `kl_clustering_analysis/hierarchy_analysis/statistics/kl_tests/edge_significance.py:132`
- Why this matters: Negative lengths can shrink variance and inflate significance.
- Exact fix: Validate `branch_length >= 0` before adjustment; raise on violation.
- Estimated effort: S
- Re-check criterion: Unit test with negative branch length raises `ValueError`.

### M-006
- Severity: Major
- Status: Unverified claim
- Problem: TreeBH implementation is assumed correct but not validated against calibration benchmarks.
- Evidence:
  - `kl_clustering_analysis/hierarchy_analysis/statistics/multiple_testing/tree_bh_correction.py:130`
- Why this matters: Without calibration evidence, FDR-control claims are not publication-safe.
- Exact fix: Add simulation study validating FDR control under tree-structured null/alternative settings.
- Estimated effort: L
- Re-check criterion: Empirical FDR remains at/below configured alpha across scenarios.

### M-007
- Severity: Major
- Status: Missing evidence
- Problem: No formal calibration/power study for the end-to-end statistical method.
- Evidence:
  - Current tests focus on sanity (identical vs strongly different), e.g. `tests/test_categorical_distributions.py:285`.
- Why this matters: Publication requires demonstrated operating characteristics, not only example correctness.
- Exact fix: Add method benchmark suite with null calibration, power curves, and sensitivity to projection/branch-length settings.
- Estimated effort: L
- Re-check criterion: Report includes calibration and power figures/tables tied to method claims.

### M-008
- Severity: Major
- Status: Verified
- Problem: Statistical test suite drift (stale random-projection tests).
- Evidence:
  - `pytest -q tests/test_random_projection.py` fails on missing `_PROJECTOR_CACHE` import.
- Why this matters: Reduces confidence in ongoing reproducibility of statistical core.
- Exact fix: Update tests to current API and remove stale references.
- Estimated effort: M
- Re-check criterion: Statistical test subset passes in CI.

### M-009
- Severity: Minor
- Status: Verified
- Problem: Duplicate `__all__` exports in edge significance module.
- Evidence:
  - `kl_clustering_analysis/hierarchy_analysis/statistics/kl_tests/edge_significance.py:366`
- Why this matters: Signals cleanup debt and review friction.
- Exact fix: Keep a single `__all__` line.
- Estimated effort: S
- Re-check criterion: Duplicate export lines removed.

## Supporting Runtime Snapshot

Recent benchmark CSV (`benchmarks/results/run_20260212_191854Z/full_benchmark_comparison.csv`) shows:

- 210 rows, 7 methods, 30 cases
- All methods `Status=ok`
- No NaNs in ARI/NMI/Purity

This confirms operational stability for that run, but does not replace inferential calibration evidence.

## Priority Fix Order

1. M-001 per-test projection seeds
2. M-002 remove silent z-score coercion
3. M-004 input/probability validation
4. M-005 negative branch-length guard
5. M-008 repair stale projection tests
6. M-003 and M-006/M-007 calibration-grade inference validation
