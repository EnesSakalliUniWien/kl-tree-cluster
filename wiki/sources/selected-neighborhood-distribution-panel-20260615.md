---
title: Selected Neighborhood Distribution Panel 2026-06-15
type: source
status: reviewed
updated: 2026-06-15
sources:
  - benchmarks/diagnostics/calibration/selected_neighborhood_distribution_panel.py
  - tests/validation/137_test_selected_neighborhood_distribution_panel.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_distribution
tags:
  - source
  - diagnostics
  - traversal
  - neighborhood
  - distribution
---

# Selected Neighborhood Distribution Panel 2026-06-15

## Summary

`selected_neighborhood_distribution_panel.py` turns the corrected old/current
method comparison into an executable diagnostic. It joins multi-scale traversal
node decisions with optional context-negative topology rows and conditional-law
rows, then summarizes neighborhood distributions separately for split,
boundary, and pass-through traversal states.

The panel is diagnostic-only. It does not change traversal, does not promote
recovery splits, and does not use neighborhood variables as direct production
cutoffs.

## Key Points

- The panel classifies each node-decision row into `split`, `boundary`, or
  `pass_through`, then records a traversal stop reason:
  `accepted_split`, `edge_closed`, `sibling_closed_no_descendant_split`,
  `sibling_closed_descendant_split_available`,
  `explicit_guard_blocked`, or `non_binary_or_leaf_boundary`.
- It preserves the corrected method understanding: neighborhood evidence
  changes the sibling/calibration context, while the traversal evaluator still
  acts through binary structure, child-parent edge evidence, sibling evidence,
  and pass-through availability.
- It reports which evidence family is available per row:
  `traversal_only`, `current_directed_topology`,
  `old_topology_neighborhood`, or `old_and_current`.
- It writes a coverage summary by method, data role, traversal state, and stop
  reason. This separates actual neighborhood-supported decisions from rows
  where only the traversal state is available.
- It also writes a case-level coverage summary so the overlap failures can be
  read by case rather than only as pooled split/boundary/pass-through totals.
- It writes a paired method-contrast summary keyed by
  `case_id,data_role,replicate,node_id`, so two traversal profiles can be
  compared on the same selected node rather than through unrelated aggregate
  counts.
- It writes a candidate-only paired contrast using the same paired-node keys
  but conditioning on nodes where either method split, passed through,
  explicit-guard-blocked, or had old-and-current neighborhood evidence.
- It writes node-level candidate contrast rows. These rows expose the exact
  `case_id,data_role,replicate,node_id` candidates where the two profiles
  agree or diverge and record the candidate reason.
- It summarizes node-level candidates into ambiguity buckets:
  `candidate_agreement`, `conservative_selected_null_suppression`,
  `possible_signal_over_suppression`, `refined_profile_more_open`, and
  `other_candidate_divergence`.
- It summarizes local traversal features for each ambiguity bucket and profile
  side: edge/sibling open counts, pass-through candidate counts, guard counts,
  sibling p-values, depth, descendant size, and available topology evidence.
- It converts ambiguity buckets into law targets and production actions. These
  targets are diagnostic obligations, not promotion rules.
- It writes a stop-rule comparison summary that groups candidate rows by
  ambiguity bucket and compares edge-open, sibling-open, pass-through, explicit
  guard, and descendant-outcome counts between the conditional-topology profile
  and the refined global pass-through profile.
- It writes a retention-evidence summary that conditions on the stop-rule
  pattern itself and reports the local evidence still available for deciding
  whether a pass-through walk should be retained.
- It writes a retention-gap summary that converts the retained-pass-through
  evidence into explicit observed evidence, missing evidence, method
  implication, next law requirement, and production action rows.
- It writes a method-contract summary that separates the shared traversal
  skeleton from the two selected-pass-through components where the methods
  differ.
- It writes a profile-config contract summary derived from
  `SIBLING_GATE_PROFILES`, so the behavioral comparison can be checked against
  the actual profile parameters.
- It writes a method-readiness summary with per-method verdicts for the shared
  traversal skeleton, selected-null pass-through control, and signal
  pass-through retention.
- The summarized variables include traversal fields, sibling p-values and
  projection dimension, current directed topology variables
  (`incoming_branch_balance`, `outgoing_balance`, `balance_product`,
  outgoing edge norm, fragment risk), and old-style topology-neighborhood
  variables (`tau_b`, `tau_t`, `tau_s`, `h_k`, nearest stable/signal
  distances).
- On the compact small-overlap run, the panel wrote `22,376` row records:
  `43` split rows, `22,308` boundary rows, and `25` pass-through rows.
- Only `38/22,376` node rows had joined old-and-current topology-neighborhood
  evidence. The remaining rows are `traversal_only`. This is now an explicit
  coverage problem rather than an implicit missing-data issue.
- Coverage is concentrated around accepted splits and a few selected-null
  stopped rows. In the compact run, signal accepted splits have old/current
  neighborhood coverage for `5/15` rows under
  `fixed_coordinate_conditional_topology_diagnostic_v1` and `5/13` rows under
  `fixed_coordinate_global_passthrough_refined_v1`. Signal pass-through rows
  have `0/10` and `0/4` coverage respectively, so the previous neighborhood
  machinery did not supply a pass-through traversal law on this slice.
- Selected-null accepted splits have higher but still partial coverage:
  `5/9` and `4/6` rows for the two methods. Selected-null sibling-closed
  boundaries have only `6/2378` and `7/2381` coverage. Therefore the old
  tree-neighborhood bandwidth terms are diagnostic context for a narrow set of
  internal candidates, not a complete stop/proceed law over all traversal
  states.
- Case-level coverage sharpens the same conclusion. `overlap_extreme_4c`
  signal rows have `0/2398` old-and-current neighborhood coverage under both
  tested profiles, including `0/3` accepted splits and `0/2` pass-through rows.
  `overlap_mod_4c_small` signal rows have `1/1598` total coverage, and
  `overlap_unbal_4c_small` signal rows have `4/1598`. The previous method's
  neighborhood layer therefore cannot be interpreted as the missing traversal
  law for these signal failures.
- The only places with complete coverage are tiny selected candidate sets,
  such as `overlap_unbal_4c_small` selected-null accepted splits (`3/3`) and
  refined-profile signal accepted splits (`4/4`). These are too narrow to
  justify a production cutoff and support only diagnostic internal-node
  interpretation.
- The paired method-contrast summary compares
  `fixed_coordinate_conditional_topology_diagnostic_v1` against
  `fixed_coordinate_global_passthrough_refined_v1` on `11,188` shared selected
  nodes. They agree on `11,151/11,188` traversal decisions (`0.996693`), but
  that pooled agreement is dominated by leaves and non-candidate boundaries.
- Candidate-only contrast reduces the denominator from `11,188` paired nodes
  to `50` candidate nodes. Agreement drops from `0.996693` to `32/50`
  (`0.64`), showing that the two profiles are materially different exactly at
  traversal-relevant nodes.
- The conditional-topology diagnostic profile has more candidate movement:
  `24` split rows and `18` pass-through rows, versus `19` split rows and `7`
  pass-through rows for the refined global pass-through profile. The refined
  profile has more explicit guard blocks (`12` versus `7`) and more boundaries
  (`11,162` versus `11,146`).
- The disagreements are localized. Selected-null `overlap_extreme_4c` and
  `overlap_mod_4c_small` are where the refined profile suppresses conditional
  splits/pass-throughs. In `overlap_unbal_4c_small` signal, the conditional
  profile reports `6` splits and `8` pass-throughs, while the refined profile
  reports `4` splits and `2` pass-throughs; candidate-only agreement there is
  only `5/14` (`0.357143`). Thus the refined profile is more conservative, but
  it also removes signal candidate movement in the unbalanced overlap case.
- The node-level candidate rows split into `32` agreeing candidates and `18`
  divergent candidates. The most common divergent pattern is conditional
  pass-through versus refined `not_visited`/`boundary`: `10` candidate rows
  have reason `left_pass_through`. In `overlap_unbal_4c_small` signal, most
  divergent pass-through rows are `traversal_only`, so the current old/current
  neighborhood variables do not yet explain why those conditional pass-throughs
  should be retained.
- Divergent selected-null rows include conditional splits that the refined
  profile guard-blocks, which is the desired conservative behavior. Divergent
  signal rows include the same suppression pattern, which is the remaining
  ambiguity: the method needs a selected-neighborhood law that distinguishes
  false selected-null pass-through/split candidates from true unbalanced signal
  candidates.
- The ambiguity summary makes the tradeoff symmetric and explicit. There are
  `8` conservative selected-null suppressions and `8` possible signal
  over-suppressions. The selected-null suppression bucket has `2/8`
  old-and-current coverage and `6/8` traversal-only pairs. The signal
  over-suppression bucket has `0/8` old-and-current coverage and `8/8`
  traversal-only pairs. Therefore the current available neighborhood variables
  explain neither side well enough to promote a rule.
- Agreements are split between evidence-backed and traversal-only candidates:
  `32` agreeing candidate rows contain `16` old-and-current pairs and `16`
  traversal-only pairs. Agreement alone is not proof of validity; it still
  mixes structurally evidenced candidates with pure traversal outcomes.
- The local feature summary identifies the unresolved signal pattern. In
  `overlap_unbal_4c_small` signal, the possible over-suppression bucket has
  `8` rows. On the conditional profile side, all `8/8` have the incoming
  child-parent edge open, `7/8` are pass-through candidates, only `1/8` has
  sibling open, median sibling p-value is `0.003135`, median depth is `5.5`,
  and median descendant leaf count is `86.5`. The same rows have no finite
  `balance_product` coverage, so the current structural topology evidence
  cannot yet decide whether these deep pass-through candidates should be kept.
- The descendant-outcome extension shows why the old traversal stop rule is
  not enough. In `overlap_unbal_4c_small` signal, the possible
  over-suppression bucket has `7` conditional-profile descendant accepted
  splits and `0` refined-profile descendant accepted splits. However, selected
  null conservative suppressions also have downstream conditional splits:
  `3` in `overlap_extreme_4c` and `2` in `overlap_mod_4c_small`. Therefore a
  downstream split is evidence that the previous pass-through mechanism was
  active, not proof that the split should be retained.
- The stop-rule comparison summary gives this pattern a stable name:
  `left_pass_through_downstream_split_right_stops`. It appears in both
  selected-null conservative suppression buckets and in the
  `overlap_unbal_4c_small` possible signal over-suppression bucket. The two
  methods are therefore not disagreeing about the algebraic traversal skeleton;
  they disagree about whether selected pass-through neighborhoods should be
  retained after the same edge-open/sibling-closed/downstream-split event.
- Conditioning directly on that pattern sharpens the evidence gap. The
  selected-null version has `5` rows, `5` conditional-profile pass-throughs,
  `5` downstream conditional accepted splits, `4/5` traversal-only pairs, and
  `1` finite `balance_product` value. The signal version has `7` rows, `7`
  conditional-profile pass-throughs, `7` downstream conditional accepted
  splits, `7/7` traversal-only pairs, and `0` finite `balance_product` values;
  its median depth is `5.0` and median descendant leaf count is `90`. This
  makes the next missing object more precise: the signal failure lacks
  structural topology coverage at the retained pass-through neighborhood.
- The retention-gap summary translates this into two law requirements. The
  selected-null side carries
  `selected_null_rows_share_pass_through_downstream_split_pattern` and keeps
  production action `retain_fail_closed_refined_guard`; its next requirement
  is a null-side selected-pass-through false-positive law. The signal side
  carries
  `missing_structural_topology_coverage_at_signal_pass_through_rows` and
  production action `fail_closed_until_topology_law_validated`; its next
  requirement is a signal-side topology likelihood for retained pass-through
  walks.
- The method-contract summary makes the comparison final for this diagnostic
  layer. `base_traversal_skeleton` is shared:
  `shared_binary_edge_sibling_passthrough`. The first non-shared component is
  `selected_null_pass_through_control`, where the refined profile is still
  required and production action is `retain_fail_closed_refined_guard`. The
  second is `signal_pass_through_retention`, where conditional recovery remains
  unvalidated and production action is
  `fail_closed_until_topology_law_validated`.
- The profile-config contract confirms the same conclusion from runtime
  profile definitions. Both profiles share `fixed_coordinate_bh`,
  sibling-gate alpha penalty `50.0`, root-stability threshold `0.24`, `12`
  root-stability subsample replicates, root-stability feature fraction `0.8`,
  and root-stability seed `0`. The concrete profile difference is the refined
  selected-family guard: the conditional-topology profile has
  `root_selective_permutation_guard_replicates=0`, while the refined profile
  has `99`, guard alpha `0.01`, and scope
  `global_sibling_min_passthrough_descendant_refined`.
- The method-readiness summary is the current decision boundary. Both profiles
  are `diagnostic_ready_shared_skeleton` for the base traversal skeleton, but
  this does not imply production promotion. The refined profile is
  `ready_as_fail_closed_guard_candidate` for
  `selected_null_pass_through_control` and keeps
  `retain_fail_closed_refined_guard`. The conditional topology profile is
  `not_ready_missing_topology_likelihood` for
  `signal_pass_through_retention` and remains
  `fail_closed_until_topology_law_validated`.
- The selected-null conservative suppression buckets also have low sibling
  p-values: median `2.728782e-05` in `overlap_extreme_4c` and `0.004946513`
  in `overlap_mod_4c_small`. The refined profile suppresses these despite the
  local p-values, which is why a law based only on edge/sibling p-values would
  leak false positives.
- The law-target summary names the unresolved work. `8` rows require
  `derive_traversal_only_pass_through_retention_law`, now with evidence gap
  `missing_topology_evidence_for_downstream_splits`, and carry production
  action `fail_closed_until_law_validated`. Another traversal-only divergence
  also fails closed under `inspect_other_traversal_only_divergence`. The `8`
  selected-null suppressions target
  `validate_false_positive_suppression_law` and keep
  `retain_fail_closed_refined_guard`. The remaining `32` candidate agreements
  are `agreement_context_not_calibration`.
- The compact run has `recover_internal_split_count = 0`. No recovery split is
  promoted.
- For `balance_product`, the finite evidence again shows the sparse-support
  pattern. The truth-recovery split row has singleton value `0.232891`; signal
  diffuse/wrong split rows range from `0.197500` to `0.222500`; selected-null
  split rows remain lower. This supports continued study of the variable but
  still leaves the real truth support count at one.
- The joint summary reports observed, singleton, constant-input, or no-finite
  statuses for selected metric pairs instead of silently treating undefined
  correlations as evidence.

## Evidence

- `tests/validation/137_test_selected_neighborhood_distribution_panel.py`
  verifies traversal-state classification, stop-reason classification,
  old/current evidence joins, descendant accepted-split counts, distribution
  summaries, joint summaries, stop-rule comparison summaries, and output
  writing. It also verifies the retention-evidence status for a synthetic
  pass-through/downstream-split candidate with no topology evidence and the
  corresponding retention-gap production action, method-contract row, and
  profile-config difference. It also verifies the method-readiness verdicts.
- The compact run used:
  `raw/assets/benchmark-results/specific_small_method_benchmark_20260615/small_overlap_selected_family/multiscale_node_decisions.csv`,
  `raw/assets/benchmark-results/specific_small_method_benchmark_20260615/context_negative_topology_conditioning/overlap_context_negative_topology_conditioning_rows.csv`,
  and
  `raw/assets/benchmark-results/specific_small_method_benchmark_20260615/conditional_law_weight1_min2/overlap_conditional_topology_law_rows.csv`.
- The outputs are stored under
  `raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_distribution/`.

## Links

- [[traversal-neighborhood-method-comparison]]
- [[specific-small-method-benchmark-20260615]]
- [[overlap-conditional-topology-law-panel-20260615]]
- [[open-mathematical-questions]]
