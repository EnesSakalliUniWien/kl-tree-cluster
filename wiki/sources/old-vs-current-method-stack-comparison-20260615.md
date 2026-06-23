---
title: Old Versus Current Method Stack Comparison 2026-06-15
type: source
status: reviewed
updated: 2026-06-15
sources:
  - raw/inbox/c2ef-cosine-subspace-method-notes-20260615.md
  - raw/assets/benchmark-results/old_vs_current_method_stack_20260615/stack_contract_comparison/manifest.json
  - raw/assets/benchmark-results/old_vs_current_method_stack_20260615/stack_contract_comparison/method_stack_contract_comparison.csv
  - raw/assets/benchmark-results/old_vs_current_method_stack_20260615/stack_contract_comparison/method_stack_behavior_summary.csv
  - raw/assets/benchmark-results/old_vs_current_method_stack_20260615/stack_contract_comparison/method_stack_pairwise_overlap.csv
  - raw/assets/benchmark-results/julia_tree_estimators_20260614/summary.csv
  - raw/assets/benchmark-results/julia_tree_estimators_20260614/kl_assignments.csv
  - raw/assets/benchmark-results/conditional_topology_law_20260615/julia_selected_family/cluster_assignments.csv
  - tree_break_selection/hierarchy_analysis/decomposition/gates/orchestrator.py
  - tree_break_selection/hierarchy_analysis/statistics/sibling_divergence/inflation_correction/empirical_null_inflation_estimation.py
  - tree_break_selection/hierarchy_analysis/statistics/sibling_divergence/fixed_subspace_annotation.py
tags:
  - source
  - diagnostics
  - calibration
  - traversal
  - julia
---

# Old Versus Current Method Stack Comparison 2026-06-15

## Summary

The corrected "old versus current" comparison is a method-stack comparison,
not a classical clustering comparison. The relevant old objects are bandwidths,
calibration strategies, heuristic checks, and traversal laws around the sibling
gate.

## Key Points

- The old c2ef stack used adaptive parent-PCA sibling projected-Wald statistics
  and a permissive calibration layer based on sibling null priors from edge
  p-values.
- The old topology-aware bandwidths were `tau_b`, `tau_t`, `tau_s`, and `h_k`,
  derived from stopping-edge distance, nearest stable/signal tree distance, and
  log projection-scale spread.
- The old bandwidth equations acted on blocked child-level sibling-null priors:
  ancestor stopping-edge support, stable-neighbor p-value smoothing, log-scale
  matching, and nearby-signal suppression were combined before the sibling gate
  was evaluated. They were not an explicit selected-family traversal law.
- A direct recheck against commit `c2ef9a69e0888168950bdee4a41ae8ab9996e32f`
  confirms the code path:
  `compute_adaptive_kernel_bandwidths` estimated `tau_b`, `tau_t`, `tau_s`,
  and `h_k`; `kernel_interpolation` applied ancestor, stable-neighbor, and
  signal-neighbor weights; `interpolate_sibling_null_priors` wrote
  `sibling_null_prior_from_edge_pvalue`, `smoothed_sibling_null_prior`,
  `ancestor_support`, and `neighborhood_reliance` back onto sibling records.
  This was a selected blocked-sibling prior update, not a conditional law for
  the selected traversal event.
- The current stack removes the old sibling-null-prior interpolation module.
  Its default empirical-null inflation model uses strict-null or edge-blocked
  calibration support with context bandwidths over log sibling projection
  dimension and log parent size.
- The current conditional-topology diagnostics already replay the useful part
  of the old bandwidth idea as a cached, support-gated
  `topology_neighborhood_log_component`. On the focused overlap slice this
  component is active but non-separating: the single truth row has component
  `-0.606531`, the null-like median is `-0.471195`, and `13` negatives exceed
  the truth on this component alone.
- The current fixed-coordinate diagnostic profiles add non-cross-fit sibling
  gates, selected-topology alpha penalties, root-stability guards, and optional
  selected-family/permutation guards.
- The base traversal law is not the main difference. Old and current both split
  on binary plus edge plus sibling evidence and pass through when binary plus
  edge evidence exists below a closed sibling gate with descendant signal. The
  main difference is how `Sibling_BH_Different` is produced and calibrated.
- A fresh old c2ef full-Julia rerun was interrupted after more than five
  minutes inside `adaptive_kernel_bandwidths`, specifically the nearest
  signal-neighborhood tree-distance loop. This confirms the old structural
  bandwidth layer is both method-relevant and computationally problematic on
  the full Julia tree unless tree distances are cached or vectorized.
- Existing full-Julia outputs show the prior full TBS stack returning `670`
  clusters with `648` singleton clusters, while the current conditional
  topology diagnostic profile returns `410` clusters with `312` singleton
  clusters. Their overlap has ARI `0.062803` and NMI `0.923045`, indicating
  closely related fine-fragment structure with different stopping behavior.
- The targeted real-search run shows why the old bandwidth logic cannot be
  restored as a traversal rule: low completed-balance selected-event rows can
  be signal-side rows yet truth-classify as false fragments rather than branch
  recoveries.

## Evidence

- `raw/assets/benchmark-results/old_vs_current_method_stack_20260615/stack_contract_comparison/method_stack_contract_comparison.csv`
  records the old/current method-stack comparison.
- `raw/assets/benchmark-results/old_vs_current_method_stack_20260615/stack_contract_comparison/method_stack_behavior_summary.csv`
  records the existing full-Julia behavior summaries.
- `raw/assets/benchmark-results/old_vs_current_method_stack_20260615/stack_contract_comparison/method_stack_pairwise_overlap.csv`
  records overlap between the prior full TBS stack and current conditional
  topology output.

## Links

- [[cosine-band-coherence-comparator-20260615]]
- [[overlap-conditional-topology-law-panel-20260615]]
- [[open-mathematical-questions]]
