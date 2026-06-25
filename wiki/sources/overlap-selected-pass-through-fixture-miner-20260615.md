---
title: Overlap Selected Pass-Through Fixture Miner 2026-06-15
type: source
status: reviewed
updated: 2026-06-15
sources:
  - benchmarks/diagnostics/calibration/overlap_selected_pass_through_fixture_miner.py
  - tests/validation/139_test_overlap_selected_pass_through_fixture_miner.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/overlap_selected_pass_through_fixture_miner
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/overlap_selected_pass_through_expanded_traversal
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/overlap_selected_pass_through_expanded_distribution
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/overlap_selected_pass_through_expanded_fixture_miner
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/overlap_selected_pass_through_expanded_fixture_miner_truth_labeled
tags:
  - source
  - diagnostics
  - overlap
  - traversal
  - pass-through
---

# Overlap Selected Pass-Through Fixture Miner 2026-06-15

## Summary

`overlap_selected_pass_through_fixture_miner.py` turns paired
selected-neighborhood candidate rows into explicit retained pass-through
fixture rows. It conditions on the selected event
`left_pass_through_downstream_split_right_stops`: the conditional topology
profile passes through a candidate and reaches downstream accepted splits, while
the refined global pass-through profile stops or blocks the same candidate.

The diagnostic does not fit a traversal law. It asks whether this selected
event stratum has enough signal-side and selected-null topology support to make
a conditional topology likelihood identifiable.

## Key Points

- The compact run mines `12` selected pass-through event rows.
- The selected event rows split into `7` signal candidates and `5`
  selected-null controls.
- The signal candidates all come from `overlap_unbal_4c_small`, replicate `0`.
- The selected-null controls come from `overlap_extreme_4c` and
  `overlap_mod_4c_small`.
- Directed traversal context is finite on all mined rows:
  `7/7` signal candidates and `5/5` selected-null controls have finite
  downstream accepted-split distance.
- The direct topology rows remain sparse: the signal side has `0` finite direct
  topology rows and the selected-null side has `1`.
- The miner now also computes structural topology fallback features from the
  selected tree itself: incoming branch balance, outgoing child balance, and
  their balance product.
- Structural topology is observed for `7/7` signal candidates and `5/5`
  selected-null controls.
- Completed topology support, using direct topology first and structural
  fallback second, is observed for `7/7` signal candidates and `3/5`
  selected-null controls.
- Completed balance product separates the finite compact selected-event rows in
  the low direction: threshold `0.02594` retains `7/7` signal candidates and
  `0` finite selected-null controls.
- The structural fallback validation status is
  `structural_fallback_separates_selected_event_diagnostic_only`.
- An expanded overlap run across seven binary overlap cases and five
  replicates mines `50` selected pass-through event rows: `19` signal
  candidates and `31` selected-null controls.
- In the expanded run, completed topology support is observed for `17/19`
  signal candidates and `20/31` selected-null controls.
- The low-direction compact threshold no longer separates: the expanded run
  retains `19/19` signal candidates but also `10` finite selected-null
  controls below the signal maximum threshold `0.133971`.
- The expanded structural fallback validation status is
  `structural_fallback_overlaps_selected_null_controls`.
- The truth-labeled expanded rerun attaches synthetic benchmark truth from
  traversal `data_seed` and `multiscale_gene_assignments.csv` path membership.
- In that truth-labeled run, the `19` signal rows contain `0` full branch
  recoveries, `0` partial branch recoveries, `1` barycentric mixture candidate,
  `14` false fragments, and `4` unresolved signal candidates.
- The barycentric mixture row is `overlap_unbal_4c_small`, replicate `4`,
  node `N790`: the downstream split has ARI `0.364506` and mean child purity
  `0.734586`, but both children have the same majority truth label.
- The expanded truth-labeled next required step is
  `derive_barycentric_mixture_vs_branch_recovery_conditioning`.
- The direct fixture support status remains `signal_topology_support_missing`.
- The completed fixture support status is
  `selected_pass_through_completed_topology_support_observed_diagnostic_only`.
- Production remains `fail_closed_until_fixture_support_observed`.

## Mathematical Interpretation

Let \(S_u\) denote the selected retained pass-through event at node \(u\). The
needed likelihood is:

\[
\log LR(u)
=
\log
\frac{
p(T_u \mid Y_u=1,S_u,M_u)
}{
p(T_u \mid Y_u=0,S_u,M_u)
},
\]

where \(T_u\) is the topology feature vector and \(M_u\) is the directed
matching context containing depth, descendant mass, pass-through-context
distance, and downstream accepted-split distance. The miner shows that \(M_u\)
is observed in the compact fixture, but direct \(T_u\) is missing on the
signal side. The structural fallback defines

\[
\tilde B^{in}_u =
\frac{\min(n_u,n_{\mathrm{sib}(u)})}{n_{\mathrm{parent}(u)}},
\quad
\tilde B^{out}_u =
\frac{\min(n_{\ell(u)},n_{r(u)})}{n_u},
\quad
\tilde P_u = \tilde B^{in}_u\tilde B^{out}_u.
\]

This makes retained signal pass-through walks measurable, but only as a
diagnostic structural fallback. The expanded run shows that the fallback alone
does not define a valid retention rule, because it overlaps selected-null
controls. The next blocker is a stronger conditional law with truth labels and
additional conditioning variables for selected pass-through neighborhoods, not
production promotion.

The truth-labeled extension adds an oracle benchmark diagnostic for synthetic
cases. For a retained pass-through node \(u\), let \(v(u)\) be the nearest
accepted descendant split, let \(A_u\) be the adjusted Rand index between the
two children of \(v(u)\) and the truth labels inside that split, let
\(\pi_u\) be the mean child truth purity, and let \(D_u\) indicate distinct
child majority truth labels. The diagnostic labels:

- full branch recovery when \(D_u=1\), \(A_u\ge 0.5\), and \(\pi_u\ge 0.65\);
- partial branch recovery when \(D_u=1\), \(0.25\le A_u<0.5\), and
  \(\pi_u\ge 0.65\);
- barycentric mixture when \(D_u=0\), \(A_u\ge 0.25\), and \(\pi_u\ge 0.65\);
- false fragment when the node is truth-homogeneous or the downstream split is
  near-zero ARI with high child purity.

The observed selected pass-through signal rows are therefore not clean branch
recoveries. They are mostly fragments, with one barycentric mixture where the
children move mixture proportions without producing different truth-majority
branches. This is why the next law must distinguish branch recovery from
barycentric mixture before any retained pass-through split can be promoted.

## Evidence

- `tests/validation/139_test_overlap_selected_pass_through_fixture_miner.py`
  verifies signal/control classification, directed traversal context, the
  structural balance formula, fail-closed support summary, and output writing.
- The compact run used
  `raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_distribution/selected_neighborhood_candidate_method_contrast_rows.csv`
  and
  `raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_distribution/selected_neighborhood_distribution_rows.csv`.
- Outputs are stored under
  `raw/assets/benchmark-results/specific_small_method_benchmark_20260615/overlap_selected_pass_through_fixture_miner/`.
- The expanded run writes traversal, selected-neighborhood, and miner outputs
  under
  `raw/assets/benchmark-results/specific_small_method_benchmark_20260615/overlap_selected_pass_through_expanded_traversal/`,
  `raw/assets/benchmark-results/specific_small_method_benchmark_20260615/overlap_selected_pass_through_expanded_distribution/`,
  and
  `raw/assets/benchmark-results/specific_small_method_benchmark_20260615/overlap_selected_pass_through_expanded_fixture_miner/`.
- The truth-labeled expanded rerun writes outputs under
  `raw/assets/benchmark-results/specific_small_method_benchmark_20260615/overlap_selected_pass_through_expanded_fixture_miner_truth_labeled/`.

## Links

- [[selected-neighborhood-distribution-panel-20260615]]
- [[retained-pass-through-topology-likelihood-panel-20260615]]
- [[traversal-neighborhood-method-comparison]]
- [[open-mathematical-questions]]
