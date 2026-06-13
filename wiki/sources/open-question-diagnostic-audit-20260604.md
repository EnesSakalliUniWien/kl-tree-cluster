---
title: Open Question Diagnostic Audit 2026-06-04
type: source
status: reviewed
updated: 2026-06-04
sources:
  - raw/assets/benchmark-results/open_question_diagnostic_audit_20260604/open_question_diagnostic_audit.csv
  - wiki/questions/open-mathematical-questions.md
  - raw/assets/benchmark-results/selected_tail_law_q5_validation_20260604/q5_selected_tail_law_summary.csv
  - raw/assets/benchmark-results/sibling_null_weight_rule_validation_20260604/sibling_null_weight_rule_summary.csv
  - raw/assets/benchmark-results/sibling_projection_dimension_rule_grid_20260604/sibling_projection_dimension_rule_grid.csv
  - raw/assets/benchmark-results/mp_kmin_q14_q15_smoke_20260604/mp_kmin_contract_smoke.csv
  - raw/assets/benchmark-results/alpha_grid_full_20260604/alpha_grid_summary.csv
  - raw/assets/benchmark-results/traversal_sibling_fdr_smoke_20260604/synthetic/traversal_sibling_fdr_summary.csv
  - raw/assets/benchmark-results/traversal_sibling_fdr_smoke_20260604/binary/traversal_sibling_fdr_summary.csv
tags:
  - source
  - diagnostics
  - questions
  - calibration
---

# Open Question Diagnostic Audit 2026-06-04

## Summary

This source records the all-open-question diagnostic audit requested on
2026-06-04. The audit does not claim that every mathematical question is
solved. It assigns each of the 44 open questions a current diagnostic status,
the strongest available result, and the next diagnostic needed before the
question can be closed or promoted into a production method rule.

## Key Points

- The focused diagnostic gate passed locally: 146 validation/statistics tests
  covering oracle traces, selected hierarchy calibration, MP diagnostics,
  root margins, alpha grids, traversal FDR, Q5, Q10, and Q17.
- Q5 is partially answered: `q5_barycentric_edge_spectral` has median
  residual-tail absolute error `0.002386`, improving on the prior
  edge/spectral model value `0.007755`, but the full all-variable barycentric
  law still fails transfer.
- Q10 remains diagnostic-only because the selected-geometry input has no true
  support labels. The current product-BH rule has effective sample size about
  `72.63`; min-BH and geometric-mean rules increase that to about `174.70`,
  but this is not a calibration validation.
- Q17 remains diagnostic-only: the current edge-derived rule uses `k=1` on
  about `75.0%` of rows and `k=2` on about `25.0%`; raw MP differs from the
  current rule on about `58.7%` of rows and would set `k=0` on about `47.1%`.
- The alpha grid supports `edge_alpha=0.001`, `sibling_alpha=0.01` for best
  mean ARI in the current benchmark grid, while lower edge alpha gives better
  cluster-count error. This is benchmark evidence, not Type-I proof.
- The traversal FDR smoke separates four failure layers: algorithmic repeated
  BH behavior, fixed-tree Wald calibration, selected-tree Wald distortion, and
  inflation support failure.

## Evidence

- `open_question_diagnostic_audit.csv` is the 44-row current-status ledger.
- The Q5, Q10, Q17, Q14/Q15, alpha-grid, and traversal-FDR CSVs provide the
  directly rerun or already locked diagnostic summaries used by the ledger.
- `wiki/questions/open-mathematical-questions.md` now points to this audit as
  the compact checkpoint for all open points.

## Links

- [[open-mathematical-questions]]
- [[barycentric-method-literature-request-20260604]]
- [[recursive-method-followups-20260604]]
- [[selected-tail-law-q5-validation-20260604]]
