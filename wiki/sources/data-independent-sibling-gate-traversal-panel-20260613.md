---
title: Data-Independent Sibling Gate Traversal Panel 2026-06-13
type: source
status: reviewed
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/sibling/gates/data_independent_sibling_gate_traversal_panel.py
  - benchmarks/diagnostics/calibration/sibling/gates/fixed_sibling_gate_profile_validation.py
  - tests/validation/calibration/sibling/gates/100_test_data_independent_sibling_gate_traversal_panel.py
  - tree_break_selection/hierarchy_analysis/decomposition/gates/orchestrator.py
  - tree_break_selection/hierarchy_analysis/tree_decomposition.py
  - tests/statistics/31_test_registry_config_wiring.py
  - tests/core/test_gate_annotation_reuse.py
  - raw/inbox/data-independent-sibling-gate-traversal-20260613.md
tags:
  - source
  - diagnostics
  - calibration
  - traversal
---

# Data-Independent Sibling Gate Traversal Panel 2026-06-13

## Summary

`data_independent_sibling_gate_traversal_panel.py` evaluates the fixed
data-independent sibling gates through the actual top-down decomposition
traversal. It keeps the existing edge gate, replaces same-sample adaptive
sibling PCA with fixed coordinate/block p-values, applies traversal-aligned BH,
and reports cluster-level outcomes.

## Key Points

- The panel is diagnostic-only and does not change production Tree-Break Selection behavior.
- It caches each replicate's tree, edge annotations, and fixed sibling p-values,
  then evaluates multiple methods and selected-topology penalties without
  recomputing the spectral edge gate.
- It now reports split-geometry diagnostics, including root split rate, false
  root split rate, first split child sizes, first split imbalance, and minimum
  observed split p-value. It also writes a traversal transfer summary across
  cases for each feature family, method, and selected-topology penalty.
- It includes an optional selected-root permutation diagnostic. The diagnostic
  preserves Bernoulli/categorical feature-block margins, reruns selected tree
  construction on each permuted null sample, and reports a Monte-Carlo
  selected-root p-value for the fixed sibling gate.
- It includes an optional selected-root feature-subsample stability diagnostic
  and guard. The diagnostic rebuilds selected trees on random feature-block
  subsets and compares the resulting root bipartitions with the original root
  split by ARI. The guard can block an opened root split when this stability is
  below a predeclared threshold.
- It now emits production-admissibility component and summary files for
  traversal transfer evidence. These are deliberately conservative: successful
  transfer rows remain `diagnostic_only`, while inflated transfer contexts
  fail closed.
- Traversal signal retention is much stronger than all-parent row-level signal
  retention. In the representative smoke, `coordinate_bh` gives mean signal ARI
  around `0.879`--`0.921` for binary cases and `0.810`--`0.831` for direct
  categorical cases across penalties `10`--`50`.
- Selected-null traversal still false-splits too often. At penalty `50`,
  `coordinate_bh` has max false-split rate `0.125` across the binary cases and
  `0.250` across the direct categorical cases in the eight-replicate smoke.
- A stricter mixed smoke with edge alpha `0.0001`, penalties `500` and `1000`,
  and `16` replicates shows binary transfer candidates for both penalties:
  max null false-split rate is `0.0000` and minimum signal mean ARI is
  `0.987067`--`0.989778`. Direct categorical transfer still fails because
  `cat_highcard_10cat_4c` has a root false-split rate of `0.0625` and
  `cat_mod_3cat_4c` falls just below the signal threshold.
- A larger `cat_highcard_10cat_4c` geometry check at penalty `1000` estimates
  null false splitting at `7/128 = 0.0546875`; every false split is a selected
  root split with first child size `9`--`44` out of `200`. The matching signal
  run has mean ARI `0.755123` and first root min-child size at least `48`, but
  other valid signal cases can have first root child sizes near `20`--`30`.
- A four-replicate selected-root permutation smoke with `9` permutation draws
  moved the known high-cardinality false root from raw p-value `1.707e-6` to
  selected-root p-value `0.1`. However, true high-cardinality signal roots also
  had selected-root p-values `0.1`--`0.2`, so the permutation diagnostic is
  useful evidence for selected-null geometry but too conservative as a
  production stopping rule.
- A traversal transfer contract smoke over `binary_2clusters` and
  `cat_highcard_10cat_4c` at penalty `1000` produced the intended production
  decisions: binary fixed-coordinate transfer was `diagnostic_only`, while the
  categorical high-cardinality transfer row was `fail_closed_undefined`.
- With root feature-subsample stability enabled at threshold `0.08`, the
  16-replicate `cat_highcard_10cat_4c` smoke blocked the one null false root,
  reduced null false splitting from `0.0625` to `0.0`, and blocked no signal
  roots; signal mean ARI stayed `0.824357`.
- In the six-case mixed smoke at penalty `1000`, the same guard controlled all
  binary and categorical nulls. Binary transfer remained a candidate with
  minimum signal mean ARI `0.985565`; categorical transfer remained
  signal-weak because `cat_mod_3cat_4c` mean ARI was `0.698390`.
- Categorical penalty and edge-alpha sweeps with the guard controlled all
  tested categorical nulls, but the best minimum categorical signal mean ARI
  was `0.747419`, just below the current `0.75` threshold.
- A follow-up threshold sweep with stability threshold `0.15`, edge alpha
  `0.001`, and `12` feature-subsample replicates moved the categorical result
  over the signal threshold: penalties `20` and `50` both had max categorical
  null false-split rate `0.0` and minimum categorical signal mean ARI
  `0.771764` in the 16-replicate categorical smoke.
- In the 16-replicate six-case mixed smoke, penalty `50` transferred for both
  binary and direct categorical cases. Binary max null false-split rate was
  `0.0` with minimum signal mean ARI `0.865005`; categorical max null
  false-split rate was `0.0` with minimum signal mean ARI `0.771764`.
- The panel now reports confidence-bound transfer diagnostics: Wilson upper
  confidence bounds for null false-split rates and one-sided t lower confidence
  bounds for signal mean ARI. Production-admissibility rows include both
  point-transfer and confidence-bound components.
- In the 16-replicate candidate confidence smoke, both binary and categorical
  point estimates transferred, but both confidence components failed closed.
  With zero observed false splits in 16 replicates, the null false-split upper
  confidence bound was `0.193608`, above the `0.05` target.
- The transfer summary now reports validation-support sizing. Under the current
  95% Wilson bound and `0.05` false-split target, zero false splits require
  `73` null replicates per case. The 16-replicate candidate smoke therefore
  needs `57` additional zero-false-split null replicates per case before the
  null confidence component can pass.
- A 73-replicate support-target validation with root-stability threshold `0.15`
  did not pass confidence: both binary and categorical families had max null
  false-split rate `0.027397` and null upper confidence `0.094501`.
- The panel now writes `root_stability_threshold_sensitivity.csv`, a post-run
  point/confidence sensitivity summary for stricter root-stability thresholds.
  On the pre-alignment 73-replicate evidence, threshold `0.24` was the first
  tested threshold that passed both binary and direct categorical point and
  confidence checks: both families had max null false split `0.0`, null upper
  confidence `0.049992`, and signal lower confidence above `0.75`.
- The traversal panel's selected-tree replay is now explicitly aligned with
  the shared TBS runner: selected trees, selected-tree oracle cuts, root-
  selective p-values, and root feature-subsample stability replays use Hamming
  distance and average linkage. Row outputs and the manifest record this
  metric/linkage contract. Earlier traversal numbers from before this
  correction are pre-alignment diagnostics unless rerun with the new fields.
- A corrected Hamming/average 16-replicate six-case mixed smoke with
  `coordinate_bh`, penalty `50`, edge alpha `0.001`, and root-stability
  threshold `0.24` was a point-transfer candidate for both binary and direct
  categorical families, but confidence remained null-uncertain because the
  Wilson upper bound with 16 null replicates is `0.193608`.
- The corrected Hamming/average 73-replicate support-target validation did not
  production-clear confidence. Binary and direct categorical families both
  remained point-transfer candidates with signal lower confidence above
  `0.75`, but each had one null false root (`1/73 = 0.013699`), giving Wilson
  upper confidence `0.073597`. With one observed false split, the panel now
  reports that `110` total null replicates, or `37` additional zero-false null
  replicates, are needed for the upper bound to fall below `0.05`.
- A corrected Hamming/average 16-replicate penalty-grid probe showed that
  penalty `500` is not a safe default replacement despite closing the rare
  binary false root: direct categorical signal becomes weak at penalties `200`
  and `500`, while penalties `50` and `100` retain categorical point transfer.
- The panel now writes `root_selective_guard_sensitivity.csv` when selected-root
  permutation draws are available. This post-hoc diagnostic blocks an opened
  root unless its selected-root permutation p-value is at or below
  `sibling_alpha`. In targeted 99-draw checks on the two corrected-Hamming false
  roots, the null roots were blocked (`0.02` for `binary_2clusters`, `0.09` for
  `cat_clear_3cat_4c`) while matched strong signal roots were retained at the
  Monte Carlo floor (`0.01`). A nine-draw smoke was too coarse and blocked
  signal, so useful evaluation needs enough permutation resolution.
- Selected-root permutation is computed lazily in the traversal panel: rows
  always record the cheap observed root p-value, but permutation draws are run
  only for method/penalty rows where the selected root opened or was blocked by
  the root-stability guard. Closed-root rows keep the selected-root p-value
  missing. This keeps the diagnostic broad-validation path computationally
  feasible.
- Broader stress probes show that this remains a limited candidate rather than
  a solved method. The full supported traversal surface contains 42
  binary-template and 11 direct categorical-multinomial cases, but all-surface
  inline runs were too slow without checkpointing. Three-replicate targeted
  probes with the same constants found zero observed null false splits for
  both families. The targeted binary probe retained all seven tested signal
  cases with minimum mean ARI `0.805955`. The direct categorical probe failed
  signal transfer: `cat_highcard_20cat_4c` had mean ARI `0.080649`, and
  `cat_overlap_3cat_4c` had mean ARI `0.703263`.
- The traversal panel now writes checkpoint row files after each completed
  case-role-replicate unit, so broad validation runs preserve partial evidence
  if interrupted. It also reports `selected_tree_oracle_ari`, the ARI obtained
  by cutting the selected average-linkage tree at the true number of clusters.
- A targeted categorical method/penalty probe shows that low-penalty `block_bh`
  is the best current fixed gate for `cat_highcard_20cat_4c`, reaching mean ARI
  `0.659781` with zero observed null false splits. This equals the selected
  tree oracle cut mean ARI for that case, so the high-cardinality categorical
  weakness is a tree-recoverability ceiling under the current average-linkage
  construction, not a remaining adaptive-projection null-law failure. Raising
  edge alpha to `0.01` did not improve the failing categorical cases.
- The fixed-subspace sibling gates are now available through the normal
  production-facing annotation/decomposition path by passing
  `sibling_gate_method="fixed_global_chi_square"`,
  `sibling_gate_method="fixed_coordinate_bh"`, or
  `sibling_gate_method="fixed_block_bh"`. The default remains
  `projected_wald_inflation`. The global option is the direct fixed-subspace
  Wald reference `chi2.sf(z.T @ z, df=len(z))`; the coordinate and block
  options are sparse/block BH aggregations. All fixed options avoid parent PCA
  projections and edge-derived sibling projection dimensions; they are method
  plumbing, not automatic production promotion. `sibling_gate_alpha_penalty`
  exposes the selected-topology penalty used in diagnostics by dividing the
  sibling FDR alpha before annotation.
- Diagnostic BH p-value paths now delegate to the production
  `fixed_subspace_sibling_p_value` implementation, so validation evidence and
  the production-facing opt-in path share the same fixed-subspace p-value code.
  Regression coverage verifies that the fixed gate does not resolve parent PCA
  sibling inputs and that penalty changes invalidate cached gate annotations.
- The selected-root feature-subsample stability guard is also available through
  the production-facing annotation/decomposition path as an explicit opt-in.
  `root_stability_guard_threshold`, `root_stability_subsample_replicates`,
  `root_stability_feature_fraction`, and `root_stability_seed` are captured in
  gate config metadata. When configured, an unstable open root is closed by
  setting `Sibling_BH_Different=False` and `Sibling_BH_Same=True` on the root
  row, and `Root_Stability_*` diagnostic columns record the guard evidence.
  Defaults keep the guard off.
- The production-facing guard now also records the root replay metric and
  linkage in config metadata. For the shared TBS runner fixed-profile path, the
  guard uses Hamming distance and the runner's tree linkage method instead of
  default Euclidean linkage replay, aligning the root-stability diagnostic with
  the selected tree being guarded.
- The matching traversal diagnostic panel now uses the same Hamming/average
  selected-tree replay contract, so future traversal evidence and
  production-facing fixed-profile runs refer to the same root geometry.
- The traversal diagnostic now also records and accepts `root_stability_seed`,
  defaulting to `0`, so its root-stability subsampling contract matches the
  named production-facing profiles. This fixes an earlier diagnostic/runtime
  mismatch where traversal rows used a data-seed-derived subsample seed while
  `fixed_coordinate_guarded_v1` used the profile constant seed `0`.
- A tiny `global_chi_square` traversal smoke over `binary_2clusters` wrote all
  expected artifacts and behaved as a point-transfer candidate, but production
  admissibility stayed fail-closed because the confidence component was still
  null-uncertain with only two null replicates.
- Named profiles `fixed_coordinate_guarded_v1` and `fixed_global_guarded_v1`
  now package the non-cross-fit method knobs as auditable diagnostic
  candidates. They set the fixed sibling method, selected-topology penalty
  `50`, root-stability threshold `0.24`, `12` stability subsamples, feature
  fraction `0.8`, and deterministic seed `0`; the profile id is stored in
  gate config metadata and participates in cache reuse checks.
- The shared TBS benchmark runner now forwards profile/fixed-gate settings to
  both gate annotation and decomposition, and records them in
  `MethodRunResult.extra`. The method-constants manifest tracks
  `sibling_gate_profile`, `fixed_sibling_gate_alpha_penalty`,
  `root_stability_guard_threshold`, `root_stability_subsample_replicates`, and
  `root_stability_feature_fraction` as validation targets. Their manifest
  evidence status remains `missing` until broader profile-validation artifacts
  are attached.
- The result separates the method layers: replacing adaptive sibling PCA with a
  fixed gate repairs much of the signal behavior. The current strongest method
  candidate is fixed coordinate BH plus selected-topology penalty plus
  selected-root feature-subsample stability. It remains diagnostic-only in the
  production contract because the passing smoke-suite statuses are candidate
  statuses, not production-ready statuses. The immediate open method problem is
  categorical power in high-cardinality and overlap settings without reopening
  selected-root null inflation. The latest oracle check further separates that
  open point: some categorical failures require changing or qualifying the
  selected tree construction rather than changing the sibling statistic.
- The root/null recheck at data seed `20309045` now has aligned traversal and
  profile evidence. With profile seed `0`, `binary_2clusters` null is blocked
  by root stability (`mean ARI = 0.184170 < 0.24`) while the matched signal is
  retained. `cat_clear_3cat_4c` null still opens under root stability
  (`mean ARI = 0.283508 > 0.24`), but a 99-draw selected-root permutation
  diagnostic gives selected-root p-value `0.17`, which would block that false
  root at `sibling_alpha = 0.01`; the matched categorical signal remains at
  the Monte Carlo floor `0.01`. This is targeted evidence, so production
  summaries still fail closed.
- The selected-root permutation layer is now also exposed as an explicit
  production-facing, default-off runtime guard for fixed-subspace sibling
  methods. It preserves Bernoulli/categorical feature-block margins, reruns
  Hamming/average selected-tree construction under permutation, writes
  `Root_Selective_Permutation_*` audit columns, and closes only an open root
  whose selected-root p-value exceeds the guard alpha.
- The runtime guard remains a validation candidate, not a production
  promotion. It fixes the known `cat_clear_3cat_4c` false root in the targeted
  fixed-profile replay while retaining matched signal, but broad confidence,
  guard replicate count, guard alpha, and supported feature-family/tree-domain
  evidence remain open.
- The packaged profile for future direct runtime validation is
  `fixed_coordinate_selective_root_v1`: fixed coordinate BH, selected-topology
  penalty `50`, root-stability threshold `0.24`, `12` stability subsamples,
  feature fraction `0.8`, and a `99`-draw selected-root permutation guard at
  alpha `0.01`.

## Evidence

- `tests/validation/calibration/sibling/gates/100_test_data_independent_sibling_gate_traversal_panel.py`
  verifies summary statuses and output writing.
- `raw/inbox/data-independent-sibling-gate-traversal-20260613.md` records the
  six-case binary/categorical traversal smoke, stricter transfer sweep, and
  high-cardinality categorical split-geometry, selected-root permutation,
  traversal production-contract, root-stability guard follow-ups, and the
  broader binary/categorical stress probes, checkpointing update, and
  selected-tree oracle check, plus the production-facing fixed-gate option,
  aligned root-stability seed recheck, and runtime selected-root guard update.

## Links

- [[data-independent-sibling-gate-panel-20260613]]
- [[projected-wald-statistic]]
- [[top-down-traversal]]
- [[open-mathematical-questions]]
