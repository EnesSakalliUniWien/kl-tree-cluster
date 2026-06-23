---
title: Path Conditioned Traversal Audit 2026-06-18
type: source
status: reviewed
updated: 2026-06-18
sources:
  - raw/assets/benchmark-results/path_conditioned_traversal_audit_20260618/manifest.json
  - raw/assets/benchmark-results/path_conditioned_traversal_audit_20260618/path_conditioned_traversal_tuples.csv
  - raw/assets/benchmark-results/path_conditioned_traversal_audit_20260618/path_conditioned_traversal_case_summary.csv
  - raw/assets/benchmark-results/path_conditioned_traversal_audit_20260618/path_conditioned_pass_through_summary.csv
  - raw/assets/benchmark-results/path_conditioned_traversal_audit_20260618/path_conditioned_boundary_summary.csv
  - raw/assets/benchmark-results/path_conditioned_traversal_audit_20260618/method_rows.csv
  - raw/assets/benchmark-results/path_conditioned_traversal_audit_20260618/run.log
  - raw/assets/benchmark-results/path_conditioned_traversal_audit_20260618/verification.log
tags:
  - source
  - benchmarks
  - traversal
  - branch-length
  - guarded
---

# Path Conditioned Traversal Audit 2026-06-18

## Summary

This audit extends the fixed-candidate traversal tuple evidence with
path-conditioned covariates. Each tuple keeps its existing traversal state and
adds incoming-parent state, ancestor-chain counts, downstream edge-reachable
support counts, and benchmark truth-label diagnostics. The output is
diagnostic-only: it does not change clustering decisions, promote
branch-length internal filtering, or introduce adaptive method routing.

## Key Points

- The run covers the same `16` selected cases and `2` methods as
  [[branch-length-traversal-audit-20260618]], producing `32` method rows and
  `3032` path-conditioned tuple rows.
- Ancestor reconstruction is complete for all tuple rows:
  `ancestor_chain_complete=True` for every row and the total missing-ancestor
  count is `0`.
- The audit records `34` live pass-through rows, split evenly between current
  `tbs` and `tbs_internal_filter_branch_length_v1`.
- All live non-root pass-through rows have a closed sibling gate and an open
  incoming child-parent edge, matching the intended pass-through diagnostic
  condition rather than a changed traversal rule.
- There are `14` stacked pass-through rows. They concentrate in
  `gauss_single_outlier_4c`, `phylo_protein_4taxa`, and one
  `phylo_dna_8taxa_low_mut` chain.
- Truth-label diagnostics classify `386` tuple rows as pure boundaries, `1`
  as a mixed boundary, `89` as truth-coherent split paths, and `2556` as
  unresolved tuples.
- No pass-through row in this panel reaches a truth-coherent descendant live
  split under the recorded truth-label diagnostic.

## Evidence

- `path_conditioned_traversal_tuples.csv` stores one row per edge-reachable
  tuple with incoming edge, ancestor-chain, descendant-support, branch-length,
  and truth-label columns.
- `path_conditioned_traversal_case_summary.csv` summarizes status, traversal
  counters, pass-through counts, stacked pass-through counts, and truth-label
  row classes per case and method.
- `path_conditioned_pass_through_summary.csv` isolates live pass-through rows
  by method and case, including stacked pass-through counts, incoming branch
  length medians, and descendant truth-coherent split counts.
- `path_conditioned_boundary_summary.csv` summarizes live boundary rows,
  pure/mixed boundary labels, edge-open sibling-closed boundaries, and their
  ancestor/descendant context.
- `run.log` records command context and per-case method progress.
- `verification.log` records targeted tests, CSV sanity checks,
  `git diff --check`, and the current `make wiki-lint` result.

## Links

- [[branch-length-traversal-audit-20260618]]
- [[branch-length-candidate-promotion-audit-20260618]]
- [[top-down-traversal]]
