---
title: Overlap Income-Outcome Junction Law 2026-06-15
type: source
status: reviewed
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/overlap/overlap_income_outcome_junction_law.py
  - tests/validation/calibration/overlap/126_test_overlap_income_outcome_junction_law.py
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/income_outcome_junction_law/overlap_income_outcome_junction_rows.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/income_outcome_junction_law/overlap_income_outcome_junction_summary.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/income_outcome_junction_law/manifest.json
tags:
  - source
  - diagnostics
  - overlap
  - traversal
  - bayesian
---

# Overlap Income-Outcome Junction Law 2026-06-15

## Summary

`overlap_income_outcome_junction_law.py` turns the degree-aware traversal idea
into an income/outcome-aware diagnostic. A node is represented by the role of
its incident relations: the incoming selected parent relation and the outgoing
child-sibling decomposition relation. The current implementation is a proxy:
incoming evidence is the selected context of the parent junction, and outgoing
evidence is the node's own internal-node likelihood row.

## Key Points

- The runner writes `overlap_income_outcome_junction_rows.csv`,
  `overlap_income_outcome_junction_summary.csv`, and `manifest.json`.
- In the focused overlap panel there are `33` residual rows, `32` with an
  observed incoming parent relation and `1` root-or-unobserved incoming case.
- No row has nonnegative incoming parent context: `incoming_context_pass_count`
  is `0`.
- Four truth-recovery rows are still recovered because their outgoing
  internal-node evidence is strong enough under the current local rule.
- The remaining missed truth row,
  `overlap_unbal_4c_small` replicate `1` node `N797`, has incoming parent zone
  `nonaccepted_or_leaf`, incoming context margin `-0.014967`, outgoing context
  margin `-0.005752`, and passes outgoing soft-structure checks. It is
  classified as
  `truth_recovery_income_outcome_context_transition_required`.
- All `28` negative rows remain blocked by outgoing context or root/unobserved
  income status; `negative_default_candidate_count` is `0`.
- The summary status is `income_outcome_transition_law_required`.

## Method Implication

The traversal law should not be merely degree-aware. It should be
income/outcome-aware: root, internal, and leaf roles matter, but the statistic
must also separate incoming selection evidence from outgoing decomposition
evidence and model the transition between them. In the focused overlap case,
parent/incoming context is negative even for recovered truth rows. The current
local rule succeeds when outgoing evidence becomes positive and soft-structure
supported; the remaining missed row is exactly the case where outgoing soft
structure is present but local outgoing context stays negative. The next
Bayesian object is therefore a transition law over the junction:

\[
P(H_u \mid I_u, O_u, I_u \to O_u, \mathcal S_u),
\]

where \(I_u\) is incoming selected-parent evidence, \(O_u\) is outgoing
child-sibling evidence, \(I_u \to O_u\) is the transition relation, and
\(\mathcal S_u\) is the selected traversal event.

## Evidence

- `tests/validation/calibration/overlap/126_test_overlap_income_outcome_junction_law.py` verifies
  that incoming and outgoing roles are kept separate, that a context-negative
  soft-supported truth row is classified as transition-law required, and that
  outputs are written.
- Verification passed:
  `pytest tests/validation/calibration/overlap/126_test_overlap_income_outcome_junction_law.py -q`
  and
  `ruff check benchmarks/diagnostics/calibration/overlap/overlap_income_outcome_junction_law.py tests/validation/calibration/overlap/126_test_overlap_income_outcome_junction_law.py`.

## Links

- [[overlap-internal-node-transfer-gap-audit-20260615]]
- [[overlap-internal-node-likelihood-transfer-20260615]]
- [[open-mathematical-questions]]
