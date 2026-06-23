---
title: Overlap Conditional Topology Law Panel 2026-06-15
type: source
status: reviewed
updated: 2026-06-15
sources:
  - benchmarks/diagnostics/calibration/overlap_conditional_topology_law_panel.py
  - tests/validation/134_test_overlap_conditional_topology_law_panel.py
  - raw/inbox/c2ef-cosine-subspace-method-notes-20260615.md
  - tree_break_selection/hierarchy_analysis/decomposition/gates/orchestrator.py
  - benchmarks/diagnostics/calibration/selected_family_traversal_panel.py
  - benchmarks/shared/runners/method_registry.py
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/conditional_topology_law/overlap_conditional_topology_law_rows.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/conditional_topology_law/overlap_conditional_topology_law_component_summary.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/conditional_topology_law/overlap_conditional_topology_law_summary.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/conditional_topology_law/overlap_conditional_topology_law_benchmark_summary.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/conditional_topology_law/overlap_conditional_topology_law_analytical_cases.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/conditional_topology_law/manifest.json
  - raw/assets/benchmark-results/conditional_topology_law_20260615/regression_gate/regression_gate_comparison.csv
  - raw/assets/benchmark-results/conditional_topology_law_20260615/regression_gate/regression_gate_metadata.json
  - raw/assets/benchmark-results/conditional_topology_law_20260615/binary_selected_family_smoke/selected_family_traversal_rows.csv
  - raw/assets/benchmark-results/conditional_topology_law_20260615/binary_selected_family_smoke/multiscale_node_decisions.csv
  - raw/assets/benchmark-results/conditional_topology_law_20260615/binary_selected_family_smoke/production_admissibility_summary.csv
  - raw/assets/benchmark-results/conditional_topology_law_20260615/julia_selected_family/manifest.json
  - raw/assets/benchmark-results/conditional_topology_law_20260615/julia_selected_family/umap_overlay/multiscale_umap_overlay.png
  - raw/assets/benchmark-results/old_vs_current_method_stack_20260615/topology_neighborhood_conditional_law_smoke/manifest.json
  - raw/assets/benchmark-results/old_vs_current_method_stack_20260615/topology_neighborhood_conditional_law_smoke/overlap_conditional_topology_law_rows.csv
  - raw/assets/benchmark-results/old_vs_current_method_stack_20260615/topology_neighborhood_conditional_law_smoke/overlap_conditional_topology_law_summary.csv
  - raw/assets/benchmark-results/old_vs_current_method_stack_20260615/topology_neighborhood_conditional_law_smoke/overlap_conditional_topology_law_component_summary.csv
tags:
  - source
  - diagnostics
  - overlap
  - traversal
  - topology
  - bayesian
---

# Overlap Conditional Topology Law Panel 2026-06-15

## Summary

`overlap_conditional_topology_law_panel.py` makes the selected-neighborhood
topology object directed and incidence-aware. It scores root, internal,
pass-through, and leaf rows using local income/outcome topology features only:
no cross-fit split, no selected-family permutation replay, and no learned
threshold.

## Key Points

- The law uses the vector
  \(Z_u=(r_u,d_u,B^{in}_u,B^{out}_u,E^{out}_u,F^{out}_u,S_u,C_u)\), where the
  incidence term distinguishes root, internal, pass-through, and leaf rows.
- Leaves are marked `leaf_no_outgoing_test_fail_closed`; roots have no fake
  incoming component; missing topology features fail closed.
- The row score is
  `logit(prior) + topology_core + selected_family_weight * log1p(selected_family)
  + min(context, 0) * context_penalty + neighborhood_scale_component
  + topology_neighborhood_component - root_penalty - passthrough_penalty` where
  the penalties apply only to the matching incidence indicators.
- The old neighborhood/local-scale idea is now explicit as
  `neighborhood_scale_log_component`, with row-level
  `neighborhood_scale_support_*` counts and fail-closed status when scale
  support is too thin. Missing scale is neutral only when the input table has no
  scale evidence at all.
- The old topology-aware sibling-null bandwidth idea is now also represented as
  `topology_neighborhood_log_component`. The panel builds cached all-pairs tree
  distances from `node_id` and `parent_id`, reports `tau_b`, `tau_t`, `tau_s`,
  and `h_k`, and excludes rows marked `selected_nonnull` from empirical support.
  It only activates when explicit `topology_support_role` or
  `topology_signal_role` evidence is present; otherwise the component is
  `topology_neighborhood_unavailable_neutral`.
- The runner reconstructs analytical cases for context-positive recovery,
  context-negative emergence, fragment false positives, closed-root
  pass-through false positives, closed-root pass-through many-cluster signal,
  and weak-incoming coherent/incoherent outcome transitions.
- The panel now exposes guarded internal recovery fields. The diagnostic rule
  is `recover_internal_split = root/null guards pass AND support sufficient AND
  balance_product/outgoing_edge evidence high`. It is a row-level recovery
  contract, not a production calibration rule.
- On the focused `26`-row context-negative overlap slice, the single
  truth-recovery row remains rank `1`. Its conditional log odds are
  `31.262811`; the strongest negative is `26.568325`, giving margin
  `4.694486`.
- The production status remains
  `diagnostic_only_support_insufficient_fail_closed`: the relevant internal
  stratum still has only one truth-recovery row, so the result is a supported
  conditioning object but not a promotable traversal rule.
- In the focused multi-positive benchmark, the guarded recovery rule recovers
  `3/3` truth rows and blocks `4/4` hard negatives. In the real
  context-negative overlap slice, it recovers `0` rows because the only truth
  row remains support-insufficient. This is the desired fail-closed boundary:
  coherent topology evidence is necessary but not enough without support.
- The diagnostic profile
  `fixed_coordinate_conditional_topology_diagnostic_v1` is now registered for
  benchmark selection. It does not apply the topology law during traversal.
  Multi-scale node decisions expose directed incidence fields and a
  fail-closed conditional-topology status placeholder.
- In the 17-case regression gate, the benchmark-facing profile runs without
  skips, has mean ARI `0.460293`, median ARI `0.480000`, and exact-K count
  `4/17`. This is a runnable diagnostic profile, not a production improvement
  over all existing cases.
- In the tiny binary selected-family smoke on `binary_2clusters`, the profile
  closes the null row with one cluster and keeps the signal row with ARI
  `0.847628`; the production summary remains `fail_closed_undefined`.
- On the Julia binary matrix (`703` samples, `14766` features), the profile
  returns `410` final clusters and `412` stable regions, with `703` leaf
  fragments and two unstable pass-through zones in the node-decision table.
  The UMAP overlay is generated for inspection, but this run is diagnostic and
  highlights fragmentation rather than production readiness.
- In the regenerated old/current smoke run, the topology-conditioning producer
  emits explicit `parent_id`, `topology_support_role`, and
  `topology_signal_role` fields. The source table has `17` strict-null support
  rows, `8` selected-nonnull exclusions, and one explicit topology signal row.
  The cached topology-neighborhood layer is active on `25/26` rows; the
  remaining sparse group is
  `topology_neighborhood_support_insufficient_fail_closed`.
- The activated topology-neighborhood component is not a standalone separator
  on this slice: the truth row's component is `-0.606531`, the negative median
  is `-0.471195`, and `13` negatives are above the truth on that component
  alone. The Bayesian topology core still keeps the truth row ranked first.
  This shows the common and breaking point: the old bandwidth geometry is useful
  structural evidence plumbing, but it must remain subordinate to the
  support-aware conditional topology law.

## Evidence

- `tests/validation/134_test_overlap_conditional_topology_law_panel.py`
  verifies directed incidence, root/leaf handling, topology evidence
  monotonicity, selected-family/context-only non-promotion, missing-feature
  fail-closed behavior, explicit neighborhood-scale support accounting,
  analytical cases, and output writing.
- Focused verification passed:
  `pytest tests/validation/134_test_overlap_conditional_topology_law_panel.py -q`.
- The topology-neighborhood contract is covered by tests for all-pairs tree
  distance caching, auxiliary parent-node caching, missing-parent-edge status,
  selected-nonnull support exclusion, and cached context ranking.
- Regression, binary smoke, and Julia matrix runs wrote outputs under
  `raw/assets/benchmark-results/conditional_topology_law_20260615/`.
- The panel output is stored under
  `raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/conditional_topology_law/`.
- The old/current topology-neighborhood smoke is stored under
  `raw/assets/benchmark-results/old_vs_current_method_stack_20260615/topology_neighborhood_conditional_law_smoke/`.

## Links

- [[overlap-context-negative-bayesian-topology-law-20260615]]
- [[overlap-context-negative-bayesian-topology-sensitivity-20260615]]
- [[overlap-context-negative-topology-transfer-20260615]]
- [[topology-vector-benchmark-20260615]]
- [[open-mathematical-questions]]
