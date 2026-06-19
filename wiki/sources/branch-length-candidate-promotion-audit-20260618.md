---
title: Branch Length Candidate Promotion Audit 2026-06-18
type: source
status: reviewed
updated: 2026-06-18
sources:
  - raw/assets/benchmark-results/branch_length_candidate_promotion_audit_20260618/manifest.json
  - raw/assets/benchmark-results/branch_length_candidate_promotion_audit_20260618/promotion_audit_report.md
  - raw/assets/benchmark-results/branch_length_candidate_promotion_audit_20260618/method_summary.csv
  - raw/assets/benchmark-results/branch_length_candidate_promotion_audit_20260618/case_pairwise_audit.csv
  - raw/assets/benchmark-results/branch_length_candidate_promotion_audit_20260618/branch_vs_current_relation_summary.csv
  - raw/assets/benchmark-results/branch_length_candidate_promotion_audit_20260618/branch_vs_current_by_category.csv
  - raw/assets/benchmark-results/branch_length_candidate_promotion_audit_20260618/branch_current_wins.csv
  - raw/assets/benchmark-results/branch_length_candidate_promotion_audit_20260618/branch_current_losses.csv
  - raw/assets/benchmark-results/branch_length_candidate_promotion_audit_20260618/branch_only_ok_vs_current.csv
  - raw/assets/benchmark-results/branch_length_candidate_promotion_audit_20260618/current_only_ok_vs_branch.csv
  - raw/assets/benchmark-results/branch_length_candidate_promotion_audit_20260618/skip_reason_summary.csv
  - raw/assets/benchmark-results/branch_length_candidate_promotion_audit_20260618/hard_overlap_rows.csv
tags:
  - source
  - benchmarks
  - guarded
  - branch-length
  - decision
---

# Branch Length Candidate Promotion Audit 2026-06-18

## Summary

This audit converts the full `121`-case branch-length benchmark into an
explicit promotion decision. The decision is not to promote
`kl_internal_filter_branch_length_v1` as the global production default. It
should remain the next candidate to test and a guarded comparator.

The reason is not that branch-length lacks signal. It improves important rows
and preserves fail-closed severe-overlap behavior. The blocker is that current
`kl` still has better full-suite completed-row mean ARI, more exact-K rows, and
fewer skips. A global replacement would regress known solved continuous and
phylogenetic rows.

## Key Points

- Current `kl` records `93` OK rows, `28` skips, `66/121` exact-K rows, mean
  ARI `0.819354`, and median ARI `1.0`.
- `kl_internal_filter_branch_length_v1` records `91` OK rows, `30` skips,
  `62/121` exact-K rows, mean ARI `0.801364`, and median ARI `1.0`.
- Legacy `kl_legacy_c2ef9a69` records `121` OK rows, no skips, `78/121`
  exact-K rows, mean ARI `0.726685`, and median ARI `0.994656`.
- Against current KL, branch-length has higher ARI on `6` cases, lower ARI on
  `7` cases, tied ARI on `66` cases, branch-only OK status on `12` cases,
  current-only OK status on `14` cases, and both-skip or non-OK status on
  `16` cases.
- The strongest branch-length gains versus current are
  `phylo_dna_8taxa_low_mut`, `phylo_protein_8taxa`, `cat_mod_4cat_6c`,
  `binary_2clusters`, `gauss_outlier_cluster_4c`, and `gauss_noisy_many`.
- The largest branch-length losses versus current are
  `gauss_clear_medium_continuous`, `dim_consolidated_4c_24f_continuous`,
  `phylo_dna_4taxa_low_mut`, `phylo_protein_4taxa`,
  `phylo_dna_8taxa_med_mut`, `gauss_single_outlier_4c_continuous`, and
  `binary_unbalanced_med`.

## Interpretation

The next defensible branch-length direction is a fixed-candidate traversal and
support audit, not an adaptive routing policy. The immediate target is to record
the live traversal, the edge-reachable traversal that walks until edge tests
close, the visited tuples, and branch lengths at those tuples. Case-family names
from the synthetic benchmark are useful for diagnosis but should not become
production routing rules.

Branch-length remains important because it keeps severe unsupported overlap
fail-closed, unlike legacy. However, the branch-only OK rows include both real
rescues and low-information completions such as high-cardinality or heavy
overlap one-cluster outputs. Those completions need stricter interpretation
before they can support promotion.

## Evidence

- `promotion_audit_report.md` records the decision, method summary, matched
  current comparison, strongest gains, largest losses, blockers, positive
  evidence, and next evaluation target.
- `case_pairwise_audit.csv` is the row-level matched comparison across current,
  branch-length, and legacy methods.
- `branch_only_ok_vs_current.csv` and `current_only_ok_vs_branch.csv` expose the
  status asymmetry that blocks a simple global replacement.
- `hard_overlap_rows.csv` records the safety boundary: current KL and
  branch-length skip `overlap_extreme_4c`, while legacy completes with near-zero
  ARI.

## Links

- [[branch-length-candidate-full-big-20260618]]
- [[branch-length-candidate-run-gate-20260618]]
- [[benchmark-runner-guarded-contract-fix-20260618]]
