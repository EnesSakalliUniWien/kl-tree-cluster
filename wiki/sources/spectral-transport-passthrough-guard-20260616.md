---
title: Spectral Transport Passthrough Guard 2026-06-16
type: source
status: reviewed
updated: 2026-06-16
sources:
  - tree_break_selection/hierarchy_analysis/decomposition/gates/spectral_transport.py
  - tree_break_selection/hierarchy_analysis/decomposition/gates/gate_evaluator.py
  - tree_break_selection/hierarchy_analysis/decomposition/gates/orchestrator.py
  - tree_break_selection/hierarchy_analysis/tree_decomposition.py
  - benchmarks/shared/runners/method_registry.py
  - benchmarks/shared/runners/dispatch.py
  - benchmarks/shared/runners/tbs_runner.py
  - benchmarks/shared/util/method_sets.py
  - benchmarks/diagnostics/calibration/spectral_transport_overlap_dispatch_panel.py
  - benchmarks/diagnostics/calibration/spectral_transport_threshold_calibration_panel.py
  - benchmarks/diagnostics/calibration/selected_family_traversal_panel.py
  - benchmarks/diagnostics/calibration/spectral_transport_promotion_gate.py
  - tests/localization/35_test_gates_traversal.py
  - tests/pipeline/51_test_dispatch_contract.py
  - tests/integration/63_test_local_structural_kernel_regression.py
  - tests/validation/147_test_spectral_transport_overlap_dispatch_panel.py
  - tests/validation/102_test_selected_family_traversal_panel.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_family_traversal_spectral_transport_overlap_three_case/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_family_traversal_spectral_transport_overlap_three_case/selected_family_traversal_rows.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_family_traversal_spectral_transport_overlap_three_case/multiscale_node_decisions.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/spectral_transport_overlap_dispatch_panel/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/spectral_transport_overlap_dispatch_panel/spectral_transport_overlap_dispatch_summary.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_family_traversal_spectral_transport_overlap_three_case_current/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_family_traversal_spectral_transport_overlap_three_case_current/selected_family_traversal_rows.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/spectral_transport_threshold_calibration_panel/spectral_transport_threshold_summary.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/spectral_transport_promotion_gate/spectral_transport_promotion_summary.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/spectral_transport_overlap_dispatch_panel_promoted/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_family_traversal_spectral_transport_overlap_three_case_promoted/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/spectral_transport_promotion_gate_promoted/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_family_traversal_spectral_transport_promoted_replicates/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/spectral_transport_promotion_gate_promoted_replicates/spectral_transport_promotion_summary.csv
tags:
  - source
  - traversal
  - spectral
  - passthrough
  - diagnostics
---

# Spectral Transport Passthrough Guard 2026-06-16

## Summary

`fixed_coordinate_spectral_transport_passthrough_v1` introduces the MP
mode-transport idea into traversal as an opt-in fail-closed pass-through
support guard.
The guard annotates each selected-tree node with multiplicity-aware MP block
transport support and lets `GateEvaluator` continue pass-through only when a
supported MP-mode path reaches a descendant split. Under the promoted
`require_mp_blocks=True` profile, unmeasured no-MP paths are bottlenecks rather
than support; the optional non-required mode keeps floor-only paths neutral for
diagnostic comparison. The guard does not open sibling splits and does not
create calibrated p-values. The opt-in profile is available through the
standard benchmark registry as `tbs_spectral_transport_passthrough`; the older
`fixed_coordinate_spectral_transport_passthrough_diagnostic_v1` and
`tbs_spectral_transport_passthrough_diagnostic` names remain diagnostic aliases.

## Key Points

- The new library module builds MP mode blocks from raw MP signal counts,
  eigenvectors, eigenvalues, multiplicities, projectors, and normalized
  characteristic polynomials.
- For each parent-child edge, it computes an optimal mode-transport cost. With
  `require_mp_blocks=True`, a pass-through path needs measured MP-mode support
  whose cost is at most the configured threshold. With
  `require_mp_blocks=False`, unmeasured edges remain neutral and only finite
  measured costs can veto a path.
- `GateEvaluator` now accepts an optional `passthrough_supported` map. When the
  map is absent, traversal behavior is unchanged. When present, `PASS_THROUGH`
  is returned only if the selected node has descendant split evidence and
  spectral transport support.
- The diagnostic profile
  `fixed_coordinate_spectral_transport_passthrough_diagnostic_v1` reuses the
  refined global selected-family pass-through guard and adds the spectral
  transport pass-through guard.
- The opt-in profile `fixed_coordinate_spectral_transport_passthrough_v1` has
  status `opt_in_candidate_not_default` and the same strict guard parameters.
- The standard benchmark method registry exposes
  `tbs_spectral_transport_passthrough`, which points to the TBS runner with the
  promoted spectral transport sibling-gate profile. The diagnostic method id
  remains available for backward compatibility.
- The refined baseline profile is also exposed as
  `tbs_global_passthrough_refined_diagnostic`, which lets the standard benchmark
  path isolate the spectral guard effect.
- `run_clustering_result` forwards explicit spectral transport parameters into
  the TBS runner, and `_run_tbs_method` records the resolved spectral transport
  configuration in `result.extra`.
- A direct dispatch smoke with the registry method id returned `status='ok'`,
  the expected spectral sibling-gate profile, and
  `spectral_transport_passthrough_guard=True`.
- The current standard-dispatch overlap run is neutral versus the refined
  baseline on the three signal rows: all paired rows have identical
  partitions, zero spectral pass-through blocks, and equal mean ARI
  `0.598648`.
- The strict selected-family rerun preserves the three signal rows and fixes
  the `overlap_mod_4c_small` selected-null oversplit: the refined baseline
  returns `7` clusters and ARI `0.0`, while the spectral profile blocks one
  pass-through node and returns `1` cluster and ARI `1.0`.
- The threshold panel marks every tested default threshold as
  `threshold_candidate`.
- The one-replicate targeted promotion gate returns `promotion_admissible` for
  the method/profile pair, but the 50-replicate panel returns
  `diagnostic_only_not_promoted` because selected-family signal retention
  fails. The profile therefore remains opt-in rather than default traversal.

## Evidence

- `spectral_transport.py` implements MP block construction, mode transport
  matching, path support, bottleneck labels, and pass-through support
  annotation.
- `gate_evaluator.py` consumes the optional support map and blocks
  pass-through without altering split decisions.
- `orchestrator.py` registers
  `fixed_coordinate_spectral_transport_passthrough_diagnostic_v1` and adds the
  spectral transport annotation step to the gate pipeline when that profile is
  selected.
- `tree_decomposition.py` passes spectral transport support into traversal and
  records support/bottleneck fields in the traversal trace.
- `method_registry.py`, `method_sets.py`, `dispatch.py`, and `tbs_runner.py`
  expose the diagnostic profile as a standard TBS benchmark method and carry its
  resolved runtime configuration through result metadata.
- `spectral_transport_overlap_dispatch_panel.py` compares the registered
  spectral method to the refined baseline through the standard benchmark
  dispatch path.
- `spectral_transport_threshold_calibration_panel.py` records the strict
  threshold operating points.
- `spectral_transport_promotion_gate.py` now defaults to the 50-replicate
  selected-family evidence and records selected-family signal retention as the
  blocking component.
- `51_test_dispatch_contract.py` covers registry exposure, dispatch as a TBS
  runner method, and forwarding of spectral transport parameters.
- `63_test_local_structural_kernel_regression.py` checks that TBS runner result
  metadata includes default spectral transport values.
- The current paired standard-dispatch signal rows remain neutral, while the
  selected-family and promotion-gate outputs record targeted traversal
  promotion evidence for the strict pass-through guard.

## Links

- [[selected-neighborhood-measurability-law]]
- [[selected-neighborhood-bottleneck-law]]
- [[selected-neighborhood-spectral-flow-diagnostic-20260616]]
- [[spectral-transport-overlap-dispatch-panel-20260616]]
- [[spectral-transport-threshold-calibration-panel-20260616]]
- [[spectral-transport-promoted-replicate-panel-20260616]]
- [[spectral-transport-promotion-gate-20260616]]
- [[overlap-method-clustering-comparison-20260616]]
