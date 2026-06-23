---
title: Ad Hoc Sibling Gate Selected Null Diffusion Comparison 2026-06-17
type: source
status: reviewed
updated: 2026-06-17
sources:
  - raw/assets/benchmark-results/ad_hoc_sibling_gate_selected_null_legacy_comparison_20260617/rows.csv
  - raw/assets/benchmark-results/ad_hoc_sibling_gate_selected_null_legacy_comparison_20260617/summary_by_method_role.csv
  - raw/assets/benchmark-results/ad_hoc_sibling_gate_selected_null_legacy_comparison_20260617/summary_by_case_method_role.csv
  - raw/assets/benchmark-results/ad_hoc_sibling_gate_selected_null_legacy_comparison_20260617/manifest.json
  - raw/assets/benchmark-results/ad_hoc_kl_diffusion_tree_comparison_20260617/summary.csv
  - raw/assets/benchmark-results/ad_hoc_kl_diffusion_tree_comparison_20260617/aggregate.csv
  - raw/assets/benchmark-results/ad_hoc_kl_diffusion_selected_null_legacy_comparison_20260617/rows.csv
  - raw/assets/benchmark-results/ad_hoc_kl_diffusion_selected_null_legacy_comparison_20260617/summary_by_method_role.csv
  - raw/assets/benchmark-results/ad_hoc_kl_diffusion_selected_null_legacy_comparison_20260617/combined_method_tradeoff.csv
tags:
  - source
  - diagnostics
  - sibling-gate
  - legacy
  - diffusion
  - bandwidth
---

# Ad Hoc Sibling Gate Selected Null Diffusion Comparison 2026-06-17

## Summary

This ad hoc smoke compares sibling-gate and tree-construction variants on small
signal and selected-null cases. The first user-run panel covers
`binary_2clusters` and `cat_clear_3cat_4c`, null plus signal, two replicates,
Hamming/average tree, `edge_alpha = 0.001`, and `sibling_alpha = 0.01`. The
follow-up diffusion runs evaluate registered `tbs_diffusion` and
`tbs_diffusion_adaptive` on the same selected-null/signal shape, plus a
four-case signal-only smoke.

The main result is that legacy `c2ef9a69` is the cleanest method on this tiny
selected-null smoke: zero null false splits and perfect signal ARI. Coordinate
fixed gates address the adaptive-parent-PCA null-law problem but are not a
standalone replacement. Diffusion tree construction improves signal behavior,
especially adaptive diffusion, but still leaks selected-null rows.

## Key Points

- In the selected-null/signal smoke, legacy `c2ef9a69` has `0/4` selected-null
  false splits, `4/4` signal exact-K rows, and signal mean ARI `1.0`.
- Adaptive parent PCA has no selected-null false split among completed null
  runs, but only `1/4` null rows complete. The other null rows fail closed
  because no valid empirical-null calibration support is available.
- Unguarded fixed coordinate/block BH gates have selected-null false split rate
  `0.75` and signal mean ARI `0.686073`, mostly from fragmentation.
- `fixed_coordinate_selective_root_v1` reduces selected-null false split rate
  to `0.25` and raises signal mean ARI to `0.879946`, but signal exact-K remains
  `0.0`.
- On the same selected-null/signal shape, plain `tbs_diffusion` has selected-null
  false split rate `0.25`, signal mean ARI `0.897059`, and signal exact-K rate
  `0.75`.
- `tbs_diffusion_adaptive` has strong signal behavior, with signal mean ARI
  `0.986815` and signal exact-K rate `1.0`, but selected-null false split rate
  `0.50` over all null rows and only `2/4` null rows completing.
- On the four-case signal-only smoke, `tbs_diffusion` has mean ARI `0.965497`
  and exact-K rate `1.0`; `tbs_diffusion_adaptive` has mean ARI `0.922392` and
  exact-K rate `0.75`.

## Evidence

- `summary_by_method_role.csv` in
  `ad_hoc_sibling_gate_selected_null_legacy_comparison_20260617` records the
  legacy and sibling-gate selected-null/signal summary.
- `combined_method_tradeoff.csv` in
  `ad_hoc_kl_diffusion_selected_null_legacy_comparison_20260617` joins that
  selected-null/signal summary with the diffusion follow-up.
- `aggregate.csv` in `ad_hoc_kl_diffusion_tree_comparison_20260617` records the
  signal-only diffusion smoke: plain diffusion ranks above adaptive diffusion,
  adaptive parent PCA, and the fixed gates by mean ARI.

## Links

- [[legacy-c2ef9a69-root-tail-overlap-comparison-20260617]]
- [[legacy-c2ef9a69-edge-alpha-comparison-20260617]]
- [[root-conditional-kernel-spectral-law]]
- [[selected-neighborhood-measurability-law]]
