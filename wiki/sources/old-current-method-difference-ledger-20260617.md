---
title: Old Current Method Difference Ledger 2026-06-17
type: source
status: reviewed
updated: 2026-06-17
sources:
  - benchmarks/diagnostics/calibration/old_current_method_difference_ledger.py
  - tests/validation/183_test_old_current_method_difference_ledger.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/old_current_method_difference_ledger_20260617
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_kernel_spectral_tail_law_20260617
tags:
  - source
  - diagnostics
  - legacy
  - calibration
---

# Old Current Method Difference Ledger 2026-06-17

## Summary

`old_current_method_difference_ledger.py` turns the scattered old-versus-current
evidence into one normalized component table. It does not rerun clustering.
Instead, it reads the existing method-stack contract, full legacy comparison,
root-tail legacy comparison, internal spectral comparison, spectral versus
bandwidth tradeoff, and kernel-spectral candidate summaries.

The ledger is the current durable overview of what was removed, replaced,
added, retained only diagnostically, and still unresolved between the old
`c2ef9a69` method and the current method.

## Key Points

- The generated ledger has `18` component rows and `18` unique components.
- `5` rows are replaced-or-removed components, `4` rows are added current
  guards, and `4` rows are diagnostic-retained or candidate-diagnostic
  components.
- The old method has positive power evidence in `2` rows, but also selected-null
  safety regression evidence in `2` rows.
- On compact legacy cases, full legacy gains signal power
  (`signal_mean_delta_ari_legacy_minus_current = 0.098485`) but adds one
  selected-null false split.
- On seven root-tail overlap cases, full legacy still gains modest signal
  power (`0.01671`) but adds two selected-null false splits.
- Internal barycenter spectra strongly perturb MP counts
  (`191.0` selected-null and `190.5` signal raw MP-count deltas) without
  changing completed compact partitions, so they remain a spectral-filter
  diagnostic rather than production calibration.
- Default old bandwidth interpolation suppresses selected-null positives but
  catches zero direct signal positives. Widening `tau_s` alone reopens
  selected-null positives at fraction `0.808659` while recovering signal at
  fraction `0.450639`.
- Strict spectral transport removes `114` selected-null false splits in the
  selected-family panel but has `4` signal regressions, so it remains a guard
  candidate rather than a standalone method.
- The full-Julia fragmentation row records the descriptive gap:
  old prior TBS stack `670` clusters versus current conditional-topology
  diagnostic `410` clusters.
- The kernel-spectral candidate row records `kernel_available_count = 2` and
  `strict_fail_closed_kernel_available_count = 1`, but
  `kernel_nonzero_support_target_count = 0`, so its decision is
  `not_promotable_positive_tail_support_missing`.

## Evidence

- The rows table records component, old behavior, current behavior, difference
  status, measurement family, selected-null effect, signal effect,
  fragmentation effect, support-calibration effect, metric ids, metric values,
  decision, and next tracking step.
- `183_test_old_current_method_difference_ledger.py` verifies that compact
  legacy, root-tail legacy, bandwidth, spectral-transport, internal-spectral,
  kernel-spectral candidate, and full-Julia fragmentation rows are emitted and
  summarized.
- The summary reports `summary_status = ledger_ready_diagnostic_only`.

## Links

- [[old-vs-current-method-stack-comparison-20260615]]
- [[legacy-c2ef9a69-root-tail-overlap-comparison-20260617]]
- [[root-conditional-kernel-spectral-law]]
- [[root-selected-kernel-spectral-tail-law-20260617]]
- [[spectral-vs-bandwidth-tradeoff-panel-20260616]]
