---
title: Selected Tail Topology Refinement 2026-06-03
type: source
status: reviewed
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/selected/tail/selected_tail_topology_refinement.py
  - tests/validation/calibration/selected/tail/58_test_selected_tail_topology_refinement.py
  - raw/assets/benchmark-results/selected_hierarchy_topology_refinement_input_20260603_300/manifest.json
  - raw/assets/benchmark-results/selected_hierarchy_topology_refinement_input_20260603_300/candidate_equation_holdout.csv
  - raw/assets/benchmark-results/selected_hierarchy_topology_refinement_input_20260603_300/selected_ratio_tail_law.csv
  - raw/assets/benchmark-results/selected_tail_topology_refinement_20260603_300/manifest.json
  - raw/assets/benchmark-results/selected_tail_topology_refinement_20260603_300/refinement_summary.csv
  - raw/assets/benchmark-results/selected_tail_topology_refinement_20260603_300/base_context_refinement_comparison.csv
tags:
  - source
  - calibration
  - selection
  - topology
---

# Selected Tail Topology Refinement 2026-06-03

## Summary

This diagnostic tests whether adding topology, balance, merge-persistence,
edge-path, and spectral-alignment bins restores held-out selected-tail
homogeneity for root and medium/large parent contexts. It uses a new
300-replicate row-level selected-hierarchy geometry run over `gauss_null_large`,
`gauss_clear_medium`, `cat_highcard_20cat_4c`, and `cat_highd_3cat_500feat`.
The result is negative for production calibration, but the reason is more
specific than the first summary implied. The base contexts are predeclared;
the refined contexts are data-adaptive bins learned inside this diagnostic
panel. Therefore refined contexts can pass a diagnostic support and precision
check, but they cannot be production-admissible calibration contexts. In this
panel, exact topology-aware stratification does not provide a production-valid
medium/large selected-tail law. It usually fragments support; where it lowers
a standard error, the subcontext is still exploratory rather than an
admissible calibration rule.

## Key Points

- The row-level input run generated selected records for Gaussian and
  categorical source families with topology fields including subtree height,
  local Colless imbalance, Sackin mean depth, and branch-length condition
  summaries.
- The base context family has `48` contexts, `19` descriptive contexts, and
  `2` production-admissible contexts. The admissible contexts are the already
  known small-parent Gaussian high-edge contexts.
- Balance refinement has the largest diagnostic support-contract pass count,
  `5`, but all refined passes are data-adaptive and therefore
  non-production. They occur in small-parent Gaussian contexts, not
  medium/large contexts.
- Topology, merge-persistence, spectral-alignment, and compact combined
  refinements create many more contexts. The combined refinement creates
  `1,587` contexts with `0` diagnostic support-contract passes and `0`
  production-admissible contexts.
- In high-edge medium/large/root contexts, exact refinements mostly report
  `no_tail_precision_improvement` or
  `refinement_fragmented_no_valid_tail_law`.
- Some refined subcontexts have lower held-out standard error or pass the
  diagnostic support check. Those rows are evidence of heterogeneity, not
  admissible calibration, because the refinement bins are data-adaptive.
- The same row-level input run rechecks selected-ratio candidate equations.
  Under leave-one-source-family transfer, edge-sampling and edge-spectral
  equations retain tail AUCs near `0.995`, while the selected-energy candidate
  falls to about `0.152` and the full descriptive equation to about `0.050`.
  This supports using the equation diagnostics as variable screening, not as a
  fitted selected-tail law.
- The next direction is not exact multiway topology matching. The result points
  toward a lower-dimensional modeled selected-tail law or a predeclared
  coarser topology coordinate that preserves independent support.

## Evidence

- `benchmarks/diagnostics/calibration/selected/tail/selected_tail_topology_refinement.py`
  implements the refinement diagnostic and writes base/refined comparison
  tables.
- `tests/validation/calibration/selected/tail/58_test_selected_tail_topology_refinement.py` verifies bin
  construction, input contracts, support summaries, and output writing.
- `raw/assets/benchmark-results/selected_hierarchy_topology_refinement_input_20260603_300/manifest.json`
  records the row-level input run over the four target cases.
- `raw/assets/benchmark-results/selected_hierarchy_topology_refinement_input_20260603_300/candidate_equation_holdout.csv`
  records the candidate-equation transfer check used to separate stable
  variable screens from panel-specific descriptive fits.
- `raw/assets/benchmark-results/selected_tail_topology_refinement_20260603_300/refinement_summary.csv`
  records context-family support and precision summaries.
- `raw/assets/benchmark-results/selected_tail_topology_refinement_20260603_300/base_context_refinement_comparison.csv`
  records base-versus-refined comparisons for each context.

## Links

- [[phylogenetic-ml-topological-selected-tail-literature-20260603]]
- [[selected-hierarchy-null-support-contract]]
- [[selected-hierarchy-geometric-law-map]]
- [[selected-tail-admissibility-domain-20260603]]
- [[open-mathematical-questions]]
