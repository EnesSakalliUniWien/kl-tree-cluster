---
title: Spectral Transport Promoted Replicate Panel 2026-06-16
type: source
status: reviewed
updated: 2026-06-16
sources:
  - benchmarks/diagnostics/calibration/selected_family_traversal_panel.py
  - benchmarks/diagnostics/calibration/spectral_transport_promotion_gate.py
  - tree_break_selection/hierarchy_analysis/decomposition/gates/orchestrator.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_family_traversal_spectral_transport_promoted_replicates/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_family_traversal_spectral_transport_promoted_replicates/selected_family_traversal_rows.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_family_traversal_spectral_transport_promoted_replicates/multiscale_node_decisions.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_family_traversal_spectral_transport_promoted_replicates/production_admissibility_summary.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/spectral_transport_promotion_gate_promoted_replicates/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/spectral_transport_promotion_gate_promoted_replicates/spectral_transport_promotion_components.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/spectral_transport_promotion_gate_promoted_replicates/spectral_transport_promotion_summary.csv
tags:
  - source
  - diagnostics
  - traversal
  - spectral
  - validation
---

# Spectral Transport Promoted Replicate Panel 2026-06-16

## Summary

The 50-replicate selected-family panel tests whether the promoted spectral
transport pass-through profile should become the default traversal rule. It
does not support default promotion. The strict MP-support guard sharply reduces
selected-null false splits, but it also blocks four signal replicates and fails
the zero-regression promotion gate.

## Key Points

- The run covers `overlap_part_4c_small`, `overlap_mod_4c_small`, and
  `overlap_heavy_4c_small_feat` under selected-null and signal roles, with
  `50` replicates for both `fixed_coordinate_global_passthrough_refined_v1`
  and `fixed_coordinate_spectral_transport_passthrough_v1`.
- Selected-null behavior improves strongly: baseline false-split count is
  `117/150`, candidate false-split count is `3/150`, and `114` false splits
  are reduced with no introduced selected-null false splits.
- Signal behavior is not clean enough for default traversal. There are `4`
  signal regressions out of `150` paired signal rows, with minimum signal
  delta ARI `-0.822005` and mean signal delta ARI `-0.006734`.
- The four signal regressions are root-level spectral pass-through blocks:
  three in `overlap_heavy_4c_small_feat` and one in `overlap_mod_4c_small`.
  Each candidate row collapses to one cluster after the strict MP-support
  guard blocks pass-through.
- The replicate-aware promotion gate returns
  `diagnostic_only_not_promoted`, blocked by
  `selected_family_signal_retention`.
- `fixed_coordinate_spectral_transport_passthrough_v1` therefore remains an
  opt-in candidate, not the default traversal rule.

## Evidence

- `selected_family_traversal_panel.py` wrote `600` selected-family traversal
  rows plus node, region, sample, guard, and production-admissibility outputs.
- `spectral_transport_promotion_gate.py` evaluated the 50-replicate selected
  family rows together with the promoted standard-dispatch panel and recorded
  `2` required promotion components passing and `1` failing.
- `orchestrator.py` marks
  `fixed_coordinate_spectral_transport_passthrough_v1` as
  `opt_in_candidate_not_default` after this run.

## Links

- [[spectral-transport-passthrough-guard-20260616]]
- [[spectral-transport-promotion-gate-20260616]]
- [[spectral-transport-threshold-calibration-panel-20260616]]
- [[selected-neighborhood-measurability-law]]
