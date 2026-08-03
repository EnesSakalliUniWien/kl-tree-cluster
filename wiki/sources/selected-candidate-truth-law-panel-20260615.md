---
title: Selected Candidate Truth Law Panel 2026-06-15
type: source
status: reviewed
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/selected/family/selected_candidate_truth_law_panel.py
  - tests/validation/calibration/selected/family/141_test_selected_candidate_truth_law_panel.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_candidate_truth_law_stress
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_candidate_truth_law_generated_support
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_candidate_truth_law_overlap_expanded
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_candidate_truth_law_real_search
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_candidate_truth_law_binary_suite
tags:
  - source
  - diagnostics
  - traversal
  - selected-candidates
  - truth-geometry
---

# Selected Candidate Truth Law Panel 2026-06-15

## Summary

`selected_candidate_truth_law_panel.py` is a diagnostic-only oracle panel for
selected traversal candidates. It reads node-level candidate method contrasts,
multi-scale path assignments, and traversal seeds, then regenerates benchmark
truth labels and classifies the candidate node's immediate child split.

This differs from the retained pass-through branch-conditioning panel. The new
panel audits all selected candidates, including accepted split rows and
selected-null controls, so it can tell whether branch recovery exists anywhere
in the selected tree before asking whether pass-through should be promoted.

## Key Points

- The panel writes `selected_candidate_truth_law_rows.csv`,
  `selected_candidate_truth_law_summary.csv`,
  `selected_candidate_truth_law_state_summary.csv`,
  `selected_candidate_feature_metric_summary.csv`,
  `selected_candidate_feature_metric_state_summary.csv`,
  `selected_candidate_context_metric_state_summary.csv`,
  `selected_candidate_family_likelihood_rows.csv`,
  `selected_candidate_family_likelihood_summary.csv`, and `manifest.json`.
  It also writes `selected_candidate_collision_law_components.csv`, which
  converts the family evidence into explicit accepted-split, pass-through,
  selected-null, and guard-stop components.
  `selected_candidate_accepted_split_filter_rows.csv` and
  `selected_candidate_accepted_split_filter_summary.csv` evaluate non-oracle
  family metrics for accepted split preservation. The pairwise extension writes
  `selected_candidate_accepted_split_pair_filter_summary.csv`, which searches
  axis-aligned two-metric rules for zero-negative accepted split filters.
  `selected_candidate_accepted_split_frontier_summary.csv` checks whether
  fragment families dominate clean families under predeclared monotone
  multi-metric frontiers. `selected_candidate_frontier_law_rows.csv` and
  `selected_candidate_frontier_law_summary.csv` add the non-permutation
  selected-family frontier law diagnostic: each clean family gets a continuous
  margin against fragment controls, and each frontier gets a Beta posterior
  over clean-family non-domination.
  `selected_candidate_frontier_ablation_summary.csv` removes one frontier
  coordinate at a time to test whether the candidate law is robust or held by a
  brittle coordinate. `selected_candidate_frontier_witness_rows.csv` records
  the nearest fragment-family witness for each clean family and the active
  coordinate that gives its frontier margin.
  `selected_candidate_sibling_rescue_audit_rows.csv` and
  `selected_candidate_sibling_rescue_audit_summary.csv` test whether
  sibling-p frontier rescues have any positive structural-coordinate support.
  `selected_candidate_sibling_rescue_guard_summary.csv` reports the
  conservative result of blocking unsupported sibling-p rescues.
- Each row records the selected candidate state, the immediate child split
  under the candidate node, child sample counts, own-split ARI, mean child
  truth purity, whether the child majority truth labels differ, and a truth
  class.
- Each row now also records non-oracle feature geometry for the same immediate
  child split: pairwise binary Jaccard homogeneity, top-coordinate subspace
  consensus, child contrast norm, branch geometry score, and barycentric
  geometry score.
- Truth classes are `own_split_branch_recovery`,
  `own_split_partial_branch_recovery`, `own_split_false_fragment`,
  `own_split_unresolved`, `own_split_unavailable`, and
  `selected_null_control`.
- The panel keeps selected-null candidate rows as explicit controls. This
  improves on the inline own-split audit, which only reported the `114`
  signal rows in the stress run.
- On `traversal_deep_branch_recovery_stress`, the reusable panel reports
  `116` selected candidate rows: `114` signal rows and `2` selected-null
  controls.
- The stress run has `13` full own-split branch recoveries, `3` partial branch
  recoveries, `84` false fragments, and `14` unresolved signal rows.
- All `13` full branch recoveries are accepted split rows with split/split
  traversal state. None are retained pass-through rows.
- Pass-through candidate rows have maximum own-split ARI `0.140288`, while
  accepted split rows reach ARI `1.0`.
- Feature geometry is observed for all `116` rows in the stress run. Two
  non-oracle metrics separate full branch recoveries from false fragments plus
  selected-null controls with zero negative leakage:
  `feature_homogeneity_gain_min` has branch minimum `0.029938` versus negative
  maximum `0.003107`, and `feature_branch_geometry_score` has branch minimum
  `0.003761` versus negative maximum `0.000321`.
- `feature_subspace_consensus_jaccard_topk` has strong but non-separating AUC
  `0.979875`; selected-null controls can have high subspace consensus without
  homogeneity gain, so consensus alone is not enough.
- The generated-support mode converts the generated retained-pass-through
  support fixture into immediate split-candidate rows at
  `truth_downstream_split_node_id`. This prevents the candidate law from
  testing the one-child pass-through parent as if it were a split.
- The generated-support candidate run contains `9` rows: `3` full branch
  recoveries, `2` false fragments, `3` selected-null controls, and `1`
  unresolved overlap partial row under the current child-purity floor.
- On generated support, `feature_homogeneity_gain_min`,
  `feature_child_contrast_norm`, and `feature_branch_geometry_score` all
  separate the `3` full branch recoveries from false fragments plus
  selected-null controls with zero negative leakage. For
  `feature_branch_geometry_score`, branch minimum is `0.147971` and negative
  maximum is `0.0`.
- The panel also writes
  `selected_candidate_context_metric_state_summary.csv`, summarizing traversal
  context metrics such as depth, descendant leaves, sibling p-values,
  descendant accepted split counts, pass-through counts, and stable-boundary
  counts by traversal state.
- On real expanded overlap candidates, the panel reports `302` rows:
  `45` full branch recoveries, `19` partial branch recoveries, `16` false
  fragments, `133` selected-null controls, and `89` unresolved rows. Unlike
  the stress run, it finds `2` pass-through branch recoveries.
- Real expanded overlap feature geometry is observed for all `302` rows, but
  no feature metric has zero-negative separation. For all candidates,
  `feature_homogeneity_gain_min` has AUC `0.960179` but branch minimum
  `0.002040` versus negative maximum `0.015827`; `feature_branch_geometry_score`
  has AUC `0.932886` but branch minimum `0.000409` versus negative maximum
  `0.007635`.
- The state-scoped summary shows the problem directly. In expanded overlap,
  `pass_through_any` has only `2` branch positives, and
  `feature_branch_geometry_score` has AUC `0.733333` with no zero-negative
  separation. Therefore the pass-through-positive rows are real but
  weak-feature rows.
- The targeted real-search slice reports `82` rows with `34` full branch
  recoveries, including the same `2` pass-through branch recoveries. It also
  fails zero-negative separation for the pass-through scope.
- The full binary-suite candidate audit reports `1000` rows: `268` full
  branch recoveries, `50` partial branch recoveries, `364` false fragments,
  `98` selected-null controls, and `220` unresolved rows. It finds only `1`
  pass-through branch recovery. Pooled feature metrics have high AUC
  (`0.994766` for homogeneity gain and `0.984970` for branch geometry), but
  still no zero-negative separation.
- Context metrics do not solve pass-through retention either. In expanded
  overlap `pass_through_any`, the two pass-through branch recoveries have
  sibling p-values `0.094013` and `0.001126`, one descendant accepted split,
  and weak feature scores. No context metric has zero-negative separation. In
  the binary-suite `pass_through_any` scope, `min_sibling_p_value` separates
  the single branch row from negatives, but this does not transfer to expanded
  overlap where one branch row has p-value `0.001126`.
- The family-likelihood extension groups selected candidates by
  `(case_id, data_role, replicate, ambiguity_bucket, stop_rule_pattern)` and
  marks whether signal families collide with matched selected-null families
  under the same selected stop-rule pattern. It is an oracle diagnostic and
  does not fit or promote a production likelihood.
- Generated support remains a clean diagnostic: it has `3` clean branch
  families and no colliding branch families. Real pass-through positives are
  different. Expanded overlap has `2` pass-through branch-positive families
  under `left_pass_through_downstream_split_right_stops`; both collide with
  selected-null controls and one also contains `5` false fragments. The
  targeted real-search run has the same two colliding pass-through branch
  families. The full binary-suite run has `1` pass-through branch-positive
  family, and it also collides with a matched selected-null control.
- Accepted split/split families are separable as a different regime: expanded
  overlap has `22` clean accepted branch families, targeted real search has
  `14`, and the binary suite has `65`. This supports preserving accepted
  branch splits separately from pass-through retention.
- The collision-law component summary makes the contract explicit. On expanded
  overlap, accepted split preservation has `32` signal families, `43` branch
  recoveries, `22` clean branch families, and `1` colliding branch family, so
  its status is `accepted_split_preservation_needs_fragment_filter`. The
  pass-through retention component has `8` signal families, `2` branch
  recoveries, `2` pass-through branch families, and `2` colliding branch
  families, so its status is `pass_through_branch_families_all_collide` with
  production action `fail_closed_until_collision_law_validated`. The targeted
  real-search and binary-suite component outputs repeat the same pass-through
  all-collide verdict. Selected-null suppression remains
  `selected_null_suppression_required`, and guard-stop fragment suppression
  remains `guard_stop_fragment_or_null_suppression_required`.
- The accepted-split filter summary closes the next shortcut. Generated
  support has zero-negative separation for homogeneity gain, branch geometry,
  and child contrast. In real transfer, every accepted-split metric overlaps
  fragment-mixed families. Expanded overlap has `22` clean accepted families
  versus `1` fragment-mixed branch family and no metric separates them. The
  targeted real-search run has `14` clean versus `2` fragment-mixed families
  with no separation. The full binary suite has `65` clean versus `40`
  fragment-mixed families, also with no zero-negative separation. The best
  binary-suite AUCs are still overlap-prone, e.g. max child contrast has AUC
  `0.886923` but clean range `1.991432`-`7.937726` overlaps fragment-mixed
  range `5.927023`-`25.588903`.
- The pairwise accepted-split filter summary tests whether two-metric
  conjunctions rescue the fragment filter. Generated support has many full
  zero-negative pairwise filters. Real transfer has only partial filters:
  expanded overlap can retain at most `20/22` clean accepted families at zero
  negatives, targeted real-search at most `10/14`, and the full binary suite at
  most `44/65`. The binary-suite best pair uses low max child contrast and low
  max barycentric score, but it remains a partial diagnostic subset with
  production action `fail_closed_until_fragment_filter_validated`.
- The frontier summary tests whether clean families are negative-dominated
  under monotone multi-metric orderings. In the binary suite, the
  `feature_strength_high` frontier leaves only `9/65` clean families
  undominated and the `feature_strength_with_fragment_penalty` frontier leaves
  `16/65`; the `binary_transfer_best_pair_context` frontier leaves `57/65`.
  The `full_family_context_frontier` has no dominated clean families in the
  real transfer runs, but this is only a diagnostic opening: it does not give a
  zero-negative threshold and still requires a selected-family law over the
  frontier surface.
- The selected-family frontier-law extension makes that law explicit without
  permutation replay. For a clean family vector \(x\), with all
  frontier coordinates oriented so larger is better, it computes
  \(m(x)=\min_{y\in\mathcal N}\max_j(x_j-y_j)\) against fragment controls
  \(\mathcal N\). The indicator \(1\{m(x)>0\}\) feeds a
  \(\operatorname{Beta}(1,1)\) diagnostic posterior for clean-family
  non-domination. This still cannot promote production behavior by itself.
- The panel now reports both raw-p and log-p sibling frontier variants.
  `median_min_sibling_p_value` exposes whether a frontier depends on raw
  p-value ordering, while `median_negative_log10_min_sibling_p_value` tests
  whether the margin problem is only numerical p-scale compression.
- On the full binary suite, the `full_family_context_frontier` is the only
  current accepted-split frontier law with zero clean-family domination: it has
  `65/65` clean families non-dominated against `40` finite fragment controls,
  posterior mean `0.985075`, lower `90%` approximation `0.960888`, and status
  `selected_family_frontier_law_margin_thin_diagnostic`. The other
  binary-suite frontiers are leaky: `feature_strength_high` has `56/65`
  dominated clean families, `feature_strength_with_fragment_penalty` has
  `49/65`, and `binary_transfer_best_pair_context` has `8/65`.
- The log-sibling version changes the binary-suite conclusion about margin
  thickness but not about production readiness. On the binary suite,
  `full_family_log_sibling_context_frontier` has `65/65` clean families
  non-dominated, minimum clean margin `0.011136`, status
  `selected_family_frontier_law_candidate_diagnostic`, and its conservative
  sibling-rescue guard blocks `0/65`. However, expanded overlap remains
  support-limited and has `4/22` unsupported log-sibling rescues, while
  real-search remains support-limited despite retaining `14/14` under the
  guard. Therefore the raw-p margin problem was partly a scale artifact, but
  the transfer and sibling-rescue law are still open.
- The ablation summary shows why the raw-p context frontier still cannot move
  to production. With all full-context coordinates present, the binary-suite
  minimum clean margin is only `1.245512e-11` and the p10 margin is
  `2.519312e-09`, so the zero-leakage frontier is numerically thin. Removing
  `median_min_sibling_p_value` makes `11/65` clean families dominated;
  removing `max_feature_child_contrast_norm` makes `10/65` dominated; removing
  `median_feature_branch_geometry_score` makes `1/65` dominated. Removing
  homogeneity gain or barycentric score does not break zero leakage on this
  binary-suite run. Therefore the current candidate is driven mainly by
  sibling p-value and child-contrast context, not by a robust pure homogeneity
  law.
- The witness rows identify the nearest fragment-family comparator and the
  active coordinate for every clean family. On the binary-suite
  `full_family_context_frontier`, `10/65` clean families have
  `frontier_margin_thin`; all ten are saved only by
  `median_min_sibling_p_value`. Across all `65` clean families, the active
  margin coordinate is `max_feature_child_contrast_norm` for `43`, sibling
  p-value for `11`, homogeneity gain for `8`, and branch geometry for `3`.
  The low-margin witnesses are mainly `binary_many_clusters`,
  `overlap_hd_4c_1k`, and `binary_low_noise_12c` clean families compared
  against `binary_unbalanced_med`, `binary_low_noise_8c`, or
  `binary_low_noise_4c` fragment-mixed families.
- The sibling-rescue audit makes this fail-closed. On the binary-suite
  `full_family_context_frontier`, `11/65` clean families use sibling p-value
  as the active coordinate, including the `10` margin-thin rows; all `11` have
  no positive structural gap against their nearest fragment witness. The
  summary status is `sibling_only_thin_rescue_requires_law` with production
  action `fail_closed_until_sibling_rescue_law_validated`. Expanded overlap
  also has `4/22` full-context sibling-only thin rescues. The real-search
  full-context rows do not use sibling p-value as the active coordinate, but
  that run has only two fragment controls and remains support-limited.
- The conservative sibling-rescue guard quantifies the cost of respecting that
  fail-closed result. It blocks unsupported sibling-p rescues and retains
  `54/65` clean families on the binary-suite `full_family_context_frontier`,
  `18/22` on expanded overlap, and `14/14` on real-search. This is a useful
  diagnostic guard, but it is not production promotion because it discards
  clean families and still requires a structural sibling-rescue law to recover
  them safely.
- The generated-support, expanded-overlap, real-search, and stress frontier-law
  outputs stay fail-closed for support reasons or leakage. Generated support
  has `3` clean families but only `2` fragment controls. Expanded overlap and
  real-search have no dominated clean families on the full context frontier,
  but only `1` and `2` finite fragment-control families respectively, so their
  status remains `selected_family_frontier_law_support_insufficient`.
- The summary status is
  `pass_through_branch_recovery_observed_diagnostic_only` on the real overlap,
  real-search, and binary-suite audits, with production action
  `diagnostic_only_no_promotion`.
- The method implication is sharper: immediate-split feature homogeneity and
  branch geometry are useful support terms for accepted split candidates, but
  they are not a standalone pass-through retention law. Local traversal context
  metrics are also insufficient alone. Real pass-through branch positives are
  selected-family collision cases, so they require a higher-order
  selected-neighborhood law that conditions on stop-rule pattern, matched
  selected-null risk, and fragment mixture.

## Evidence

- `benchmarks/diagnostics/calibration/selected/family/selected_candidate_truth_law_panel.py`
  implements the reusable diagnostic runner and CLI.
- `tests/validation/calibration/selected/family/141_test_selected_candidate_truth_law_panel.py` verifies
  branch recovery, homogeneous pass-through fragments, selected-null controls,
  directed state summaries, and output writing.
- `raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_candidate_truth_law_stress/selected_candidate_truth_law_summary.csv`
  contains the stress-run summary.
- `raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_candidate_truth_law_stress/selected_candidate_truth_law_state_summary.csv`
  shows that full branch recovery occurs only in split/split rows.
- `raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_candidate_truth_law_stress/selected_candidate_feature_metric_summary.csv`
  shows which non-oracle feature metrics separate full branch recovery from
  false fragments and selected-null controls.
- `raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_candidate_truth_law_generated_support/`
  contains the generated-support candidate conversion and validation run.
- `raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_candidate_truth_law_overlap_expanded/`
  contains the expanded real overlap transfer run.
- `raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_candidate_truth_law_real_search/`
  contains the targeted real-search transfer run.
- `raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_candidate_truth_law_binary_suite/`
  contains the full binary-suite transfer run.
- `selected_candidate_family_likelihood_summary.csv` in each output directory
  records clean branch families, colliding branch families, selected-null
  suppression families, and fail-closed unresolved families.
- `selected_candidate_collision_law_components.csv` records the resulting
  selected-family law components and their fail-closed or diagnostic-only
  production actions.
- `selected_candidate_accepted_split_filter_summary.csv` records accepted
  split clean-family versus fragment-mixed-family metric overlap.
- `selected_candidate_accepted_split_pair_filter_summary.csv` records
  zero-negative pairwise filter recall and confirms that real accepted split
  preservation still lacks a full fragment filter.
- `selected_candidate_accepted_split_frontier_summary.csv` records monotone
  dominance counts for clean accepted families versus fragment-mixed families.
- `selected_candidate_frontier_law_rows.csv` and
  `selected_candidate_frontier_law_summary.csv` record the continuous
  selected-family frontier margins and the Beta diagnostic posterior over
  clean-family non-domination.
- `selected_candidate_frontier_ablation_summary.csv` records leave-one-feature
  frontier ablations and flags leakage or thin-margin zero-leakage behavior.
- `selected_candidate_frontier_witness_rows.csv` records the nearest fragment
  witness and active frontier coordinate for each clean family.
- `selected_candidate_sibling_rescue_audit_rows.csv` and
  `selected_candidate_sibling_rescue_audit_summary.csv` record whether
  sibling-p active-coordinate rescues have structural support or require a
  separate fail-closed law.
- `selected_candidate_sibling_rescue_guard_summary.csv` records how many clean
  families remain after blocking unsupported sibling-p rescues.

## Links

- [[selected-pass-through-branch-recovery-conditioning-20260615]]
- [[selected-neighborhood-distribution-panel-20260615]]
- [[open-mathematical-questions]]
