---
title: Spectral Versus Bandwidth Tradeoff Panel 2026-06-16
type: source
status: reviewed
updated: 2026-06-16
sources:
  - benchmarks/diagnostics/calibration/spectral_vs_bandwidth_tradeoff_panel.py
  - tests/validation/150_test_spectral_vs_bandwidth_tradeoff_panel.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/spectral_vs_bandwidth_tradeoff_panel/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/spectral_vs_bandwidth_tradeoff_panel/spectral_vs_bandwidth_tradeoff_rows.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/spectral_vs_bandwidth_tradeoff_panel/spectral_vs_bandwidth_tradeoff_summary.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_family_traversal_spectral_transport_promoted_replicates/selected_family_traversal_rows.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_pvalue_interpolation_comparison_overlap_expanded_candidates/selected_neighborhood_pvalue_interpolation_summary.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_pvalue_interpolation_comparison_overlap_expanded_candidates/selected_neighborhood_pvalue_interpolation_tau_s_sensitivity.csv
  - raw/assets/benchmark-results/old_vs_current_method_stack_20260615/stack_contract_comparison/method_stack_behavior_summary.csv
  - raw/assets/benchmark-results/old_vs_current_method_stack_20260615/stack_contract_comparison/method_stack_pairwise_overlap.csv
tags:
  - source
  - diagnostics
  - spectral
  - bandwidth
  - traversal
---

# Spectral Versus Bandwidth Tradeoff Panel 2026-06-16

## Summary

`spectral_vs_bandwidth_tradeoff_panel.py` compares the current strict
MP-supported spectral transport pass-through rule against the older bandwidth
interpolation idea on the evidence surfaces that overlap. The comparison is
not a runnable old-versus-new clustering benchmark, because the old bandwidth
logic currently exists as a p-like interpolation diagnostic, not as a
production traversal profile. The panel verdict is
`hybrid_needed_diagnostic_only`.

## Key Points

- Strict MP spectral transport sharply reduces selected-null oversplitting in
  the 50-replicate selected-family panel: baseline false splits drop from
  `117/150` to `3/150`, a reduction of `114`.
- The same strict spectral rule is not default-promotable because it regresses
  `4/150` paired signal rows. The worst signal delta ARI is `-0.822005`, and
  the promotion gate remains `diagnostic_only_not_promoted`, blocked by
  `selected_family_signal_retention`.
- Default old bandwidth interpolation is conservative on expanded overlap
  candidate p-values: `716` directly significant selected-null rows become
  nonsignificant and zero selected-null rows are reopened.
- That same default bandwidth setting misses all directly significant signal
  rows in the candidate diagnostic: `861/861` direct signal positives become
  nonsignificant, with zero signal catches.
- Widening `tau_s` alone gives the wrong tradeoff. At `tau_s = 20`, the
  best-case calculation reopens `579/716` selected-null direct positives
  (`0.808659`) while recovering only `388/861` signal direct positives
  (`0.450639`).
- On full Julia, the prior old KL stack remains more fragmented than the
  current conditional-topology diagnostic summary: `670` clusters with `648`
  singleton clusters versus `410` clusters with `312` singleton clusters.
  Their pairwise ARI is `0.062803` and NMI is `0.923045`.

## Evidence

- `spectral_vs_bandwidth_tradeoff_panel.py` reads the spectral promotion gate,
  selected-family traversal rows, bandwidth interpolation summaries, and
  old/current full-Julia stack summaries, then writes a common rows table and
  one-row summary.
- `150_test_spectral_vs_bandwidth_tradeoff_panel.py` verifies the split between
  spectral signal-regression blocking and bandwidth tau-sensitivity blocking.
- The output directory contains `28` metric rows and one comparison summary.
  Its recommended next step is to use bandwidth interpolation as
  support-gated locality evidence and spectral transport as a fail-closed
  bottleneck guard, without promoting either as a standalone rescue rule.

## Links

- [[selected-neighborhood-measurability-law]]
- [[selected-neighborhood-pvalue-interpolation-comparison-20260616]]
- [[spectral-transport-promoted-replicate-panel-20260616]]
- [[old-vs-current-method-stack-comparison-20260615]]
