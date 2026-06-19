---
title: Path Conditioned Alpha Contract Recheck 2026-06-18
type: source
status: reviewed
updated: 2026-06-18
sources:
  - raw/assets/benchmark-results/path_conditioned_traversal_audit_alpha_contract_20260618/manifest.json
  - raw/assets/benchmark-results/path_conditioned_traversal_audit_alpha_contract_20260618/path_conditioned_traversal_tuples.csv
  - raw/assets/benchmark-results/path_conditioned_traversal_audit_alpha_contract_20260618/path_conditioned_traversal_case_summary.csv
  - raw/assets/benchmark-results/path_conditioned_traversal_audit_alpha_contract_20260618/path_conditioned_pass_through_summary.csv
  - raw/assets/benchmark-results/path_conditioned_traversal_audit_alpha_contract_20260618/path_conditioned_boundary_summary.csv
  - raw/assets/benchmark-results/path_conditioned_traversal_audit_alpha_contract_20260618/method_rows.csv
  - raw/assets/benchmark-results/path_conditioned_traversal_audit_alpha_contract_20260618/run.log
  - raw/assets/benchmark-results/path_conditioned_traversal_audit_alpha_contract_20260618/verification.log
  - raw/assets/benchmark-results/path_conditioned_hypothesis_audit_alpha_contract_20260618/manifest.json
  - raw/assets/benchmark-results/path_conditioned_hypothesis_audit_alpha_contract_20260618/branch_current_path_hypothesis_table.csv
  - raw/assets/benchmark-results/path_conditioned_hypothesis_audit_alpha_contract_20260618/path_burden_outcome_summary.csv
  - raw/assets/benchmark-results/path_conditioned_hypothesis_audit_alpha_contract_20260618/path_conditioned_hypothesis_report.md
  - raw/assets/benchmark-results/path_conditioned_hypothesis_audit_alpha_contract_20260618/verification.log
tags:
  - source
  - benchmarks
  - traversal
  - branch-length
  - guarded
---

# Path Conditioned Alpha Contract Recheck 2026-06-18

## Summary

This recheck reruns the path-conditioned traversal audit under the canonical
benchmark alpha contract: `edge_alpha=0.001` and `sibling_alpha=0.01`. The
earlier path-conditioned traversal audit used `0.05/0.05`, which explained the
six outcome/path status mismatches in the first hypothesis join. The corrected
alpha-contract run has zero status mismatches against the full branch-length
promotion audit on the selected 16-case panel.

## Key Points

- The corrected traversal audit records `significance_level=0.01` and
  `edge_alpha=0.001` in its manifest.
- The alpha-contract traversal audit writes `3819` path-conditioned tuple rows,
  `32` method rows, `27` boundary summary rows, and `8` pass-through summary
  rows.
- The corrected hypothesis join has `0/16` outcome/path status mismatch rows.
- Among comparable OK/OK rows, `12` branch-minus-current ARI deltas exactly
  match the full benchmark promotion audit.
- The corrected path-burden classification changes the interpretation:
  `phylo_dna_4taxa_low_mut` is now the only branch loss with stacked
  pass-through burden, while `phylo_protein_8taxa` is a branch gain despite
  stacked pass-through burden.
- Therefore stacked pass-through is not a sufficient badness rule. It remains
  a diagnostic interaction term that must be read with case family, boundary
  purity, and support status.
- The audit defaults in the traversal diagnostic runners were corrected to use
  the canonical alpha constants so future reruns inherit the benchmark
  contract unless explicitly overridden.

## Evidence

- `path_conditioned_traversal_case_summary.csv` gives the corrected
  method-case statuses and traversal counts under the benchmark alpha contract.
- `branch_current_path_hypothesis_table.csv` shows all selected cases have
  `outcome_path_status_connection=status_consistent`.
- `path_burden_outcome_summary.csv` summarizes the corrected outcome classes:
  five branch gains without pass-through burden, one branch gain despite
  stacked pass-through, two branch losses with isolated pass-through, one
  branch loss with stacked pass-through, and one branch loss without path
  burden.
- `path_conditioned_hypothesis_report.md` records the reader-facing corrected
  interpretation.
- `verification.log` records targeted tests, CSV sanity checks, `git diff
  --check`, and the current wiki-lint state.

## Links

- [[path-conditioned-hypothesis-audit-20260618]]
- [[path-conditioned-traversal-audit-20260618]]
- [[branch-length-candidate-promotion-audit-20260618]]
