---
title: Open Question Full Diagnostic Contract 2026-06-04
type: source
status: reviewed
updated: 2026-06-04
sources:
  - benchmarks/diagnostics/open_questions/full_diagnostic_contract.py
  - tests/validation/71_test_open_question_full_diagnostic_contract.py
  - raw/assets/benchmark-results/open_question_full_diagnostic_contract_20260604/open_question_full_diagnostic_contract.csv
  - raw/assets/benchmark-results/open_question_full_diagnostic_contract_20260604/open_question_full_diagnostic_summary.csv
  - raw/assets/benchmark-results/open_question_full_diagnostic_contract_20260604/manifest.json
  - wiki/questions/open-mathematical-questions.md
tags:
  - source
  - diagnostics
  - questions
  - validation
---

# Open Question Full Diagnostic Contract 2026-06-04

## Summary

This source records the conversion of every open mathematical question into a
fully specified diagnostic contract. The contract is not a method-claim
promotion. It states, for each `Q1`--`Q44`, the diagnostic family, scale,
entrypoint or missing prototype, required inputs, required outputs, acceptance
criteria, and blocker before any result can become a method claim.

## Key Points

- `open_question_full_diagnostic_contract.csv` contains exactly `44` rows,
  one for each question from `Q1` through `Q44`.
- Every row has `diagnostic_contract_status=fully_specified`.
- The summary groups the required diagnostics by scale: `20` cloud-panel
  diagnostics, `8` full-suite diagnostics, `4` theory-plus-simulation
  diagnostics, `3` local-and-cloud diagnostics, `3` manual-manifest
  diagnostics, `3` prototype-then-cloud diagnostics, `2` theory-plus-cloud
  diagnostics, and `1` manual-selection diagnostic.
- `tests/validation/71_test_open_question_full_diagnostic_contract.py`
  enforces coverage, non-empty decision fields, acceptance criteria, and
  artifact writing.
- The contract preserves the distinction between "fully specified diagnostic"
  and "validated method claim"; no open question is marked solved merely
  because it now has a complete diagnostic work unit.

## Evidence

- `benchmarks/diagnostics/open_questions/full_diagnostic_contract.py` defines
  the contract table, completeness validator, summary generation, and CLI.
- `open_question_full_diagnostic_contract.csv` is the durable row-level
  diagnostic contract.
- `open_question_full_diagnostic_summary.csv` records coverage counts by
  evidence status, diagnostic scale, family, and contract status.
- `manifest.json` records the generated artifact paths and study role.

## Links

- [[open-mathematical-questions]]
- [[open-question-diagnostic-audit-20260604]]
- [[recursive-method-followups-20260604]]
