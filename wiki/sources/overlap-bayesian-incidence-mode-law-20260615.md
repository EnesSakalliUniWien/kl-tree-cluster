---
title: Overlap Bayesian Incidence Mode Law 2026-06-15
type: source
status: reviewed
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/overlap/overlap_bayesian_incidence_mode_law.py
  - tests/validation/calibration/overlap/128_test_overlap_bayesian_incidence_mode_law.py
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/bayesian_incidence_mode_law/overlap_bayesian_incidence_mode_rows.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/bayesian_incidence_mode_law/overlap_bayesian_incidence_mode_summary.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/bayesian_incidence_mode_law/manifest.json
tags:
  - source
  - diagnostics
  - overlap
  - traversal
  - bayesian
---

# Overlap Bayesian Incidence Mode Law 2026-06-15

## Summary

`overlap_bayesian_incidence_mode_law.py` expresses the branch-incidence result
as a diagnostic Bayesian law with two latent modes: continuation mode and
emergent local-outcome mode. The continuation score uses the richer
metric-family alignment score when available, not only top-coordinate Jaccard.
It remains diagnostic-only and fail-closed for context-negative emergent rows
when current conditioning variables do not separate truth recovery from
negatives.

## Key Points

- The runner writes `overlap_bayesian_incidence_mode_rows.csv`,
  `overlap_bayesian_incidence_mode_summary.csv`, and `manifest.json`.
- In the focused overlap panel there are `33` transfer-gap rows:
  `5` truth-recovery and `28` negative rows.
- The continuation mode has `0` candidates because metric-family
  branch-incidence alignment is absent.
- The positive-context local-outcome mode selects `4` rows, all
  truth-recovery, with `0` negative candidates.
- The context-negative emergent mode is not identified: it contains `26`
  ambiguous rows, including `1` truth-recovery row and `25` negative rows.
- A follow-up edge-conditioning scan checked the existing child-parent edge
  tests inside those `26` rows and found no zero-negative separator; edge
  strength overlaps negatives and edge rejection flags are saturated.
- A selected-neighborhood topology scan then found a single-positive focused
  candidate: the incoming/outgoing balance product separates the one truth row
  from `25` negatives, but this remains diagnostic-only pending transfer
  validation.
- The missed row `overlap_unbal_4c_small` replicate `1` node `N797` remains
  `context_negative_emergent_mode_ambiguous` and
  `fail_closed_requires_additional_conditioning`; its branch alignment score
  is `0.112750`.
- The summary status is `context_negative_emergent_mode_not_identified`.

## Method Implication

The current admissible diagnostic law is not a branch-continuation law. It is a
local-outcome law with a hard nonnegative context condition:

\[
\text{local outcome candidate}
\Longleftrightarrow
\text{selected Bayes evidence}
\land
\text{soft outgoing structure}
\land
\text{nonnegative local context}.
\]

This recovers `4/5` truth-recovery rows with `0` negatives in the focused
panel. The remaining context-negative truth row cannot be promoted under the
available conditions because the same context-negative emergent mode also
contains many selected-null, diffuse/wrong, and fragment-like rows. The next
statistical task is not to relax context, but to find an additional
conditioning variable that separates the `1` true context-negative emergent
row from the `25` negative context-negative emergent rows.

## Evidence

- `tests/validation/calibration/overlap/128_test_overlap_bayesian_incidence_mode_law.py` verifies
  that positive-context local-outcome rows are candidates, context-negative
  emergent rows remain ambiguous when truth and negatives share that evidence
  pattern, and outputs are written.
- Verification passed:
  `pytest tests/validation/calibration/overlap/128_test_overlap_bayesian_incidence_mode_law.py -q`
  and
  `ruff check benchmarks/diagnostics/calibration/overlap/overlap_bayesian_incidence_mode_law.py tests/validation/calibration/overlap/128_test_overlap_bayesian_incidence_mode_law.py`.

## Links

- [[overlap-branch-incidence-junction-panel-20260615]]
- [[overlap-context-negative-edge-conditioning-20260615]]
- [[overlap-context-negative-topology-conditioning-20260615]]
- [[overlap-income-outcome-junction-law-20260615]]
- [[open-mathematical-questions]]
