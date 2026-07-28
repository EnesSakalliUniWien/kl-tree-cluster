---
title: Overlap Selected-Family Law Requirements 2026-06-14
type: source
status: reviewed
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/overlap/overlap_selected_family_law_requirements.py
  - tests/validation/calibration/overlap/119_test_overlap_selected_family_law_requirements.py
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/selected_family_law_requirements/overlap_selected_family_conditioning_envelope.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/selected_family_law_requirements/overlap_selected_family_law_requirements.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/selected_family_law_requirements/manifest.json
tags:
  - source
  - diagnostics
  - overlap
  - traversal
  - selected-family
---

# Overlap Selected-Family Law Requirements 2026-06-14

## Summary

`overlap_selected_family_law_requirements.py` converts the residual
selected-family diagnostics and the threshold-stability contract into a
requirements envelope for the missing overlap traversal law. It is
diagnostic-only: it makes the conditioning variables explicit but does not
promote any production threshold.

## Key Points

- The runner writes `overlap_selected_family_conditioning_envelope.csv`,
  `overlap_selected_family_law_requirements.csv`, and `manifest.json`.
- The conditioning envelope groups residual families by truth role and
  summarizes family size, selected-family p-value evidence, homogeneity gain,
  continuous context margin, subspace consensus, depth, parent size,
  barycentric balance, fragment risk, balanced recovery proxy, size balance,
  and edge-norm balance.
- The focused residual family set has `5` truth-recovery families, `3`
  non-recovery signal families, and `8` selected-null-like families.
- The selected-family null-evidence requirement is
  `law_required_not_transferable_cutpoint`: p-value evidence separates
  recovery from selected null in the focused panel, but transfer leaks selected
  null and non-recovery families.
- The structural-recovery requirement is
  `law_required_structural_target_missing`: non-recovery selected signal can
  be more p-value-extreme than recovery, so the law must condition on
  structural recovery, not only extremeness.
- The transfer-validation requirement is
  `fail_closed_transfer_not_satisfied`: residual focused cutpoints are not
  accepted unless a predeclared rule passes held-out case and replicate
  validation.
- The reporting requirement is
  `stable_reporting_only_no_production_promotion`: residual weak families stay
  in unstable multi-scale output until the selected-family law is validated.
- Method implication: the next mathematical object is a continuous
  context-conditioned selected-family structural recovery law. Root, shallow,
  large-parent, imbalanced, weak-homogeneity, and subspace-incoherent families
  must be penalized through conditioning variables, not through a single global
  homogeneity cutoff.

## Evidence

- `tests/validation/calibration/overlap/119_test_overlap_selected_family_law_requirements.py`
  verifies conditioning-envelope summaries, the four core requirement rows, and
  output writing.
- Verification passed:
  `pytest tests/validation/calibration/overlap/119_test_overlap_selected_family_law_requirements.py -q`
  and
  `ruff check benchmarks/diagnostics/calibration/overlap/overlap_selected_family_law_requirements.py tests/validation/calibration/overlap/119_test_overlap_selected_family_law_requirements.py`.

## Links

- [[overlap-residual-family-recovery-20260614]]
- [[overlap-residual-recovery-eligibility-20260614]]
- [[overlap-residual-threshold-transfer-20260614]]
- [[overlap-threshold-stability-contract-20260614]]
- [[open-mathematical-questions]]
