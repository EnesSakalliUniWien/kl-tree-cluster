---
title: Retained Pass-Through Topology Likelihood Panel 2026-06-15
type: source
status: reviewed
updated: 2026-06-15
sources:
  - benchmarks/diagnostics/calibration/retained_pass_through_topology_likelihood_panel.py
  - tests/validation/138_test_retained_pass_through_topology_likelihood_panel.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/retained_pass_through_topology_likelihood/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_distribution/selected_neighborhood_distribution_rows.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/retained_pass_through_topology_likelihood/retained_pass_through_topology_likelihood_rows.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/retained_pass_through_topology_likelihood/retained_pass_through_topology_match_rows.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/retained_pass_through_topology_likelihood/retained_pass_through_topology_likelihood_summary.csv
tags:
  - source
  - diagnostics
  - traversal
  - topology
  - likelihood
---

# Retained Pass-Through Topology Likelihood Panel 2026-06-15

## Summary

`retained_pass_through_topology_likelihood_panel.py` checks whether the
signal-side topology likelihood for retained pass-through walks is identifiable
from the current selected-neighborhood candidate rows. It does not fit a
Bayesian rule and does not promote production behavior.

The target law is:

\[
\log LR_{\mathrm{topo}}(u)
=
\log p(T_u \mid Y_u=1,S_u,M_u)
-
\log p(T_u \mid Y_u=0,S_u,M_u),
\]

where \(S_u\) is the selected retained-pass-through event, \(Y_u=1\) denotes a
signal retained pass-through candidate, \(Y_u=0\) denotes a matched
selected-null pass-through control, \(T_u\) is the topology feature vector, and
\(M_u\) is the matching context based on depth and descendant mass.

The directed traversal-neighborhood refinement is:

\[
M(u,v)
=
\frac{|d_u-d_v|}{h_d}
+
\frac{|\log(1+n_u)-\log(1+n_v)|}{h_n}
+
\frac{|a_u-a_v|}{h_a}
+
\frac{|s_u-s_v|}{h_s},
\]

where \(d_u\) is depth, \(n_u\) is descendant-leaf mass, \(a_u\) is distance to
the nearest pass-through context including \(u\) itself, and \(s_u\) is distance
to the first downstream accepted split. Exact tree-network distance is only
defined inside the same selected tree. For signal rows matched to selected-null
controls from different selected trees, the panel reports
`cross_case_network_distance_unavailable` rather than inventing a graph
distance.

## Key Points

- The selected event is
  `left_pass_through_downstream_split_right_stops`: the conditional topology
  profile passes through a node and reaches downstream accepted splits, while
  the refined profile stops or blocks the same walk.
- The panel extracts signal retained-pass-through candidates and selected-null
  pass-through controls from
  `selected_neighborhood_candidate_method_contrast_rows.csv`.
- It matches signal rows to selected-null controls by absolute depth
  difference, absolute log descendant-leaf difference, pass-through-context
  distance, and downstream accepted-split distance when the traversal context
  is finite on both sides.
- On the compact overlap run, the panel finds `7` signal rows and `5`
  selected-null control rows.
- Of the `7` signal rows, `5` have matched selected-null controls under the
  configured distance tolerances and `2` remain unmatched.
- The directed traversal context is observed for all retained pass-through
  rows: `7/7` signal rows and `5/5` selected-null controls have finite
  pass-through-context distance and downstream accepted-split distance.
- All `5` matched rows have matched traversal-network context, but all `5`
  are cross-case selected-null matches, so exact tree-network distance is
  reported as unavailable.
- The signal side has `0` finite `balance_product` values and `0` finite
  outgoing edge-norm balance values.
- The selected-null control side has `1` finite `balance_product` and `1`
  finite outgoing edge-norm balance value overall, but `0` matched controls
  with finite topology features for the matched signal rows.
- The summary status is
  `signal_topology_likelihood_not_identifiable`, with production action
  `fail_closed_until_likelihood_identifiable`.
- This confirms the current method-readiness conclusion: the conditional
  topology profile may be exposing deep signal pass-through walks, but the
  signal-side topology likelihood cannot yet be estimated from the available
  topology evidence.

## Evidence

- `tests/validation/138_test_retained_pass_through_topology_likelihood_panel.py`
  verifies the retained-pass-through signal/control fixture, matching, and
  fail-closed identifiability status when signal topology features are absent.
- The compact run used
  `raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_distribution/selected_neighborhood_candidate_method_contrast_rows.csv`
  and
  `raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_distribution/selected_neighborhood_distribution_rows.csv`
  as inputs.
- The outputs are stored under
  `raw/assets/benchmark-results/specific_small_method_benchmark_20260615/retained_pass_through_topology_likelihood/`.

## Links

- [[selected-neighborhood-distribution-panel-20260615]]
- [[traversal-neighborhood-method-comparison]]
- [[open-mathematical-questions]]
