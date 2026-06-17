---
title: Spectral Transport Overlap Dispatch Panel 2026-06-16
type: source
status: reviewed
updated: 2026-06-16
sources:
  - benchmarks/diagnostics/calibration/spectral_transport_overlap_dispatch_panel.py
  - tests/validation/147_test_spectral_transport_overlap_dispatch_panel.py
  - benchmarks/shared/runners/method_registry.py
  - benchmarks/shared/util/method_sets.py
  - kl_clustering_analysis/hierarchy_analysis/decomposition/gates/spectral_transport.py
  - benchmarks/diagnostics/calibration/spectral_transport_threshold_calibration_panel.py
  - benchmarks/diagnostics/calibration/spectral_transport_promotion_gate.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/spectral_transport_overlap_dispatch_panel/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/spectral_transport_overlap_dispatch_panel/spectral_transport_overlap_dispatch_rows.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/spectral_transport_overlap_dispatch_panel/spectral_transport_overlap_dispatch_pairwise.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/spectral_transport_overlap_dispatch_panel/spectral_transport_overlap_dispatch_summary.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_family_traversal_spectral_transport_overlap_three_case_current/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_family_traversal_spectral_transport_overlap_three_case_current/selected_family_traversal_rows.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_family_traversal_spectral_transport_overlap_three_case_current/multiscale_node_decisions.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/spectral_transport_threshold_calibration_panel/spectral_transport_threshold_summary.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/spectral_transport_promotion_gate/spectral_transport_promotion_summary.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/spectral_transport_overlap_dispatch_panel_promoted/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/spectral_transport_overlap_dispatch_panel_promoted/spectral_transport_overlap_dispatch_pairwise.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/spectral_transport_overlap_dispatch_panel_promoted/spectral_transport_overlap_dispatch_summary.csv
tags:
  - source
  - diagnostics
  - traversal
  - spectral
  - overlap
---

# Spectral Transport Overlap Dispatch Panel 2026-06-16

## Summary

`spectral_transport_overlap_dispatch_panel.py` compares the promoted
`kl_spectral_transport_passthrough` method against the matched
`kl_global_passthrough_refined_diagnostic` baseline on the same overlap cases
through the standard benchmark dispatch path. The promoted method uses strict
measured MP support when `require_mp_blocks=True`: unmeasured no-MP paths are
bottlenecks rather than support, while the optional non-required mode keeps
floor-only paths neutral for diagnostic comparison.

## Key Points

- The refined global pass-through baseline is now exposed as
  `kl_global_passthrough_refined_diagnostic`, so the spectral method can be
  compared against the same profile without the spectral guard.
- The promoted spectral method is exposed as `kl_spectral_transport_passthrough`
  and resolves to `fixed_coordinate_spectral_transport_passthrough_v1`. The
  old diagnostic id remains available as an alias for diagnostic runs.
- The default spectral transport max-cost threshold is `1.2`. Under strict
  measured MP support, the threshold panel marks `0.75`, `1.0`, `1.2`, and
  `1.5` as threshold candidates on the three selected-family overlap cases.
- On the promoted standard-dispatch three-case run, the spectral method is neutral:
  all three pairwise rows have identical partitions to the refined baseline,
  mean ARI is `0.598648` for both methods, mean found clusters is `5.666667`
  for both methods, and `spectral_transport_passthrough_blocked_count` is `0`.
- The current-code selected-family rerun preserves signal rows and fixes the
  `overlap_mod_4c_small` selected-null oversplit: the baseline returns `7`
  clusters and ARI `0.0`, while the spectral profile blocks one pass-through
  node and returns `1` cluster and ARI `1.0`.
- The promotion gate over these outputs returns `promotion_admissible`: all
  three targeted components pass.
- This supersedes the earlier neutral matched-mode-only conclusion. The
  evidence is now strong enough for the targeted pass-through traversal guard
  on these overlap panels, but broader production admissibility still requires
  larger confidence and transfer evidence.

## Evidence

- `spectral_transport_overlap_dispatch_panel.py` writes per-method rows,
  pairwise candidate-minus-baseline rows, summaries, and a manifest using the
  shared benchmark registry and dispatcher.
- `147_test_spectral_transport_overlap_dispatch_panel.py` checks the pairwise
  guard-effect classification and output-writing contract.
- The standard-dispatch panel writes `6` method rows, `3` pairwise rows, and
  `3` summary rows under
  `raw/assets/benchmark-results/specific_small_method_benchmark_20260615/spectral_transport_overlap_dispatch_panel/`.
- The current selected-family rerun writes fresh node and traversal rows under
  `raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_family_traversal_spectral_transport_overlap_three_case_current/`.
- The strict threshold panel and promotion gate record the corresponding
  traversal-promotion evidence.

## Links

- [[spectral-transport-passthrough-guard-20260616]]
- [[spectral-transport-threshold-calibration-panel-20260616]]
- [[spectral-transport-promotion-gate-20260616]]
- [[selected-neighborhood-spectral-flow-diagnostic-20260616]]
- [[selected-neighborhood-bottleneck-law]]
