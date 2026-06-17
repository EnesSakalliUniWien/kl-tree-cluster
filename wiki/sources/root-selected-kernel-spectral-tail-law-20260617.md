---
title: Root Selected Kernel Spectral Tail Law 2026-06-17
type: source
status: reviewed
updated: 2026-06-17
sources:
  - benchmarks/diagnostics/calibration/root_selected_kernel_spectral_tail_law_panel.py
  - tests/validation/184_test_root_selected_kernel_spectral_tail_law_panel.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_kernel_spectral_tail_law_20260617/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_kernel_spectral_tail_law_20260617/root_selected_kernel_spectral_tail_law_rows.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_kernel_spectral_tail_law_20260617/root_selected_kernel_spectral_tail_law_summary.csv
tags:
  - source
  - diagnostics
  - root
  - kernel
  - spectral
---

# Root Selected Kernel Spectral Tail Law 2026-06-17

## Summary

`root_selected_kernel_spectral_tail_law_panel.py` is the first executable
candidate bridge between the old neighborhood smoother and the current
fail-closed selected-root spectral-tail law. It does not change clustering. It
uses old-style kernel locality only as admissible support weights for
\(S_{H_u}\), after filtering to selected-null or external-null support rows.
It now reports both the original scalar selected-geometry kernel and a stricter
topology-conditioned kernel that first requires exact or coarsened root
bifurcation signatures.

The seven-root run shows that kernel support can be measured, but it is not yet
a rescue law. Scalar kernel support is available for two roots, but topology
conditioning removes all promotable support: six roots have no matching
topology support and one root has only one effective topology support row.

## Key Points

- The panel writes `7` target rows and one summary row under
  `root_selected_kernel_spectral_tail_law_20260617`.
- Current strict tail support is available for `3/7` roots and fail-closed for
  `4/7`.
- Kernel weighting reports available diagnostic support for `2/7` roots, but
  only one of those is newly available relative to strict support.
- `kernel_nonzero_support_target_count = 0`: none of the seven targets has a
  usable nearby support row with positive deformed \(S_{H_u}\).
- The topology-conditioned channel reports
  `topology_kernel_available_count = 0`,
  `topology_support_missing_count = 6`, and
  `topology_degenerate_support_count = 1`.
- Observed-target root topology is enriched from the root selected-region
  replay before topology matching; missing topology remains an explicit
  fail-closed reason.
- `overlap_heavy_4c_small_feat` moves from strict fail-closed to kernel support
  available, but its target \(S_{H_u}=0\), so this is not a signal rescue.
- Positive-tail roots such as `overlap_mod_6c_med`,
  `overlap_part_4c_small`, and `overlap_unbal_4c_small` remain fail-closed
  with `fail_closed_kernel_nonzero_s_h_u_support_missing`.
- The nearest effective support row is a zero-tail external-null row for every
  target after importance and kernel weighting. This localizes the next
  bottleneck: generate or reweight admissible nonzero \(S_{H_u}\) support in
  the kernel neighborhood.
- The candidate remains diagnostic-only and does not change clustering.

## Evidence

- The row table records target \(S_{H_u}\), strict support status, admissible
  kernel support count, positive support count, exceedance count, effective
  sample size, max weight share, weighted exceedance fraction, conservative
  p-value, candidate decision, topology signatures, topology support counts,
  topology-weighted tail fields, nearest support row, and legacy overlay
  fields.
- `184_test_root_selected_kernel_spectral_tail_law_panel.py` verifies that
  diagnostic proposal rows are excluded, kernel support can be added for a
  strict fail-closed target, and positive targets with only zero-tail support
  suppress the p-value and fail closed. It also verifies observed-root topology
  enrichment from the selected-region summary.
- The summary reports
  `summary_status = scalar_kernel_support_but_topology_fail_closed`.

## Links

- [[root-conditional-kernel-spectral-law]]
- [[old-current-method-difference-ledger-20260617]]
- [[root-selected-same-geometry-external-support-attempt-20260617]]
- [[legacy-c2ef9a69-root-tail-overlap-comparison-20260617]]
