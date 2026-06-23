---
title: Path Conditioned Hypothesis Audit 2026-06-18
type: source
status: reviewed
updated: 2026-06-18
sources:
  - raw/assets/benchmark-results/path_conditioned_hypothesis_audit_20260618/manifest.json
  - raw/assets/benchmark-results/path_conditioned_hypothesis_audit_20260618/branch_current_path_hypothesis_table.csv
  - raw/assets/benchmark-results/path_conditioned_hypothesis_audit_20260618/method_case_path_hypothesis_table.csv
  - raw/assets/benchmark-results/path_conditioned_hypothesis_audit_20260618/path_burden_outcome_summary.csv
  - raw/assets/benchmark-results/path_conditioned_hypothesis_audit_20260618/recent_method_connection_summary.csv
  - raw/assets/benchmark-results/path_conditioned_hypothesis_audit_20260618/hypothesis_solution_matrix.csv
  - raw/assets/benchmark-results/path_conditioned_hypothesis_audit_20260618/path_conditioned_hypothesis_report.md
  - raw/assets/benchmark-results/path_conditioned_hypothesis_audit_20260618/run.log
  - raw/assets/benchmark-results/path_conditioned_hypothesis_audit_20260618/verification.log
tags:
  - source
  - benchmarks
  - traversal
  - branch-length
  - guarded
---

# Path Conditioned Hypothesis Audit 2026-06-18

## Summary

This audit turns the path-conditioned traversal hypothesis into a falsification
table. It joins the path-conditioned traversal audit to the branch-length
promotion audit, the full big benchmark summary, and the manual guarded
six-method smoke. The output is diagnostic-only and does not define a
production rule, promote branch-length internal filtering, or route between
methods after observing outcomes.

The first version of this audit joined to a traversal run that used
`0.05/0.05` diagnostic alpha defaults. Use
[[path-conditioned-alpha-contract-recheck-20260618]] for the benchmark-aligned
interpretation under `edge_alpha=0.001` and `sibling_alpha=0.01`.

## Key Points

- The audit covers the same `16` selected traversal cases, producing `16`
  branch-current case rows, `32` method-case rows, and `9` recent method
  connection rows.
- Six selected cases have status mismatches between the full promotion audit
  and the path-conditioned traversal audit. These rows are marked
  `outcome_path_status_mismatch` and should not be used to explain full-suite
  branch gains or losses from this traversal run.
- Among status-consistent rows, branch-length gains without branch
  pass-through burden appear on `gauss_outlier_cluster_4c` and
  `phylo_dna_8taxa_low_mut`.
- The strongest stacked-pass-through failure marker is
  `phylo_protein_4taxa`, where branch-length has `9` pass-through rows,
  `6` stacked pass-through rows, and lower ARI than current in both the
  promotion audit and the traversal audit.
- Isolated branch pass-through losses occur on `phylo_dna_4taxa_low_mut` and
  `phylo_dna_8taxa_med_mut`; these weaken a pure stacked-pass-through-only
  explanation and require a separate isolated-pass-through check.
- `traversal_deep_signal_under_same_parent` remains a branch-only OK row with
  one mixed boundary and no branch pass-through burden, supporting a separate
  root/deep support-stop failure mode rather than a pass-through-chain rule.
- The method-connection table keeps current TBS as the reference, legacy as a
  power comparator, internal filtering as support-threshold evidence,
  branch-length as the fixed candidate, bandwidth context as selected-
  neighborhood support evidence, and rescued legacy as a negative control from
  the guarded smoke.

## Evidence

- `branch_current_path_hypothesis_table.csv` contains one row per selected
  case with promotion-audit status, path-audit status, delta-source connection,
  path-burden columns, and separate promotion/path hypothesis classes.
- `method_case_path_hypothesis_table.csv` contains one row per method and case
  joined to tuple-burden aggregates.
- `path_burden_outcome_summary.csv` summarizes case counts and path-burden
  totals by guarded hypothesis class.
- `recent_method_connection_summary.csv` connects `tbs`, branch-length, legacy,
  internal-filter, bandwidth-context, and rescued-legacy methods to the
  hypothesis.
- `hypothesis_solution_matrix.csv` records admissible solutions for isolated
  pass-through, stacked pass-through, branch-length ambiguity, mixed boundary,
  and legacy/rescued-method evidence.
- `path_conditioned_hypothesis_report.md` is the reader-facing report for the
  audit.

## Links

- [[path-conditioned-traversal-audit-20260618]]
- [[path-conditioned-alpha-contract-recheck-20260618]]
- [[branch-length-traversal-audit-20260618]]
- [[branch-length-candidate-promotion-audit-20260618]]
