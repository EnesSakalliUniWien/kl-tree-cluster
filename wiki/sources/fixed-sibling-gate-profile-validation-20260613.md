---
title: Fixed Sibling Gate Profile Validation 2026-06-13
type: source
status: reviewed
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/sibling/gates/fixed_sibling_gate_profile_validation.py
  - benchmarks/shared/runners/tbs_runner.py
  - tree_break_selection/hierarchy_analysis/decomposition/gates/orchestrator.py
  - tree_break_selection/hierarchy_analysis/tree_decomposition.py
  - tests/validation/calibration/sibling/gates/101_test_fixed_sibling_gate_profile_validation.py
  - benchmarks/diagnostics/calibration/traversal/production_admissibility_contract.py
  - raw/inbox/fixed-sibling-gate-profile-validation-20260613.md
tags:
  - source
  - diagnostics
  - calibration
  - validation
---

# Fixed Sibling Gate Profile Validation 2026-06-13

## Summary

`fixed_sibling_gate_profile_validation.py` validates the named fixed
sibling-gate profiles through the shared TBS benchmark runner. It is a
diagnostic artifact for routing and evidence fields, not a production
calibration rule.

## Key Points

- The panel runs selected binary or direct categorical null/signal cases
  through `_run_tbs_method` with a named profile such as
  `fixed_coordinate_guarded_v1` or `fixed_global_guarded_v1`.
- It records gate config metadata, observed `Sibling_Test_Method` values,
  whether any `projected_wald_inflation` sibling rows were used, traversal
  cluster counts, ARI, null false-split flags, root-stability guard blocking,
  root-level sibling p-values/open decisions, root-stability mean/median/q10,
  and the effective profile sibling alpha.
- The root-stability guard now records and uses the selected-tree replay
  contract for fixed-profile TBS runs: Hamming distance and average linkage.
  This removes an earlier diagnostic mismatch where feature-subsample root
  splits were recomputed with SciPy's default Euclidean metric while the TBS
  selected tree was Hamming/average.
- Profile rows require `adaptive_projection_avoided=True`: the emitted profile
  id and sibling method must match the profile, no adaptive projected-Wald
  sibling rows may appear, and at least one fixed-method sibling row must be
  annotated.
- The summary reports null false-split Wilson upper bounds and signal mean-ARI
  lower bounds. Transfer summaries combine included cases by feature family and
  profile.
- The production-admissibility rows remain conservative. Fixed-profile routing
  and transfer candidates are diagnostic-only, while missing coverage,
  adaptive projection fallback, null inflation, weak signal, or uncertain
  confidence fails closed.
- The panel writes `method_constant_evidence_fields.json` for the method
  constants `sibling_gate_profile`, `fixed_sibling_gate_alpha_penalty`,
  `root_stability_guard_threshold`, `root_stability_subsample_replicates`,
  `root_stability_feature_fraction`,
  `root_selective_permutation_guard_replicates`, and
  `root_selective_permutation_guard_alpha`.
- TooManyCells is not used as a direct comparator. It is only relational
  context for tree-first divisive stopping, while this artifact validates
  Tree-Break Selection's fixed sibling-gate profile routing.
- A targeted profile replay at data seed `20309045` shows the current profile
  boundary: `fixed_coordinate_guarded_v1` blocks the known binary null root via
  root stability with seed `0`, but still opens the known
  `cat_clear_3cat_4c` null root. This confirms that fixed sibling gates plus
  root stability are not yet a full categorical root/null fix.
- The selected-root permutation guard is now executable in the production-facing
  gate annotation path as a default-off opt-in. The fixed-profile validation
  CLI forwards guard replicates, seed, and alpha into the shared TBS runner,
  records `Root_Selective_Permutation_*` audit columns, and summarizes
  selected-root p-values and guard block rates.
- The selected-root method layer is now packaged as
  `fixed_coordinate_selective_root_v1`. This profile sets fixed coordinate BH,
  selected-topology penalty `50`, root-stability threshold `0.24`, `12`
  stability subsamples, feature fraction `0.8`, selected-root permutation
  replicates `99`, selected-root guard seed `0`, and selected-root guard alpha
  `0.01`.
- The older `fixed_coordinate_guarded_v1` and `fixed_global_guarded_v1`
  profiles keep selected-root permutation disabled by default, so existing
  profile-validation smoke runs remain compatible. They can still accept
  explicit selected-root guard settings for targeted replay.
- The guard is deliberately limited to fixed-subspace sibling methods. It
  raises with `projected_wald_inflation` because the same-sample adaptive PCA
  statistic is the layer being avoided, not a valid input to repair.
- The shared TBS runner now records resolved profile settings in
  `MethodRunResult.extra`, so benchmark artifacts report the actual fixed
  sibling method, selected-topology penalty, stability constants, and
  selected-root guard constants rather than only raw explicit kwargs.
- The method-constant evidence builder also uses resolved profile settings.
  A profile-validation run that asks only for `fixed_coordinate_selective_root_v1`
  now reports selected-root permutation grids `[99]` and `[0.01]` in evidence
  JSON, rather than incorrectly treating the guard as disabled because the CLI
  override knobs were left at defaults.
- A targeted profile replay at data seed `20309045`, guard seed `20318458`,
  and `99` permutation draws fixes the known `cat_clear_3cat_4c` null root in
  the runtime path: the null selected-root p-value is `0.17`, the guard closes
  the root, and the run returns one cluster. The matched categorical signal is
  retained with selected-root p-value `0.01` and ARI `0.881909`.
- The same replay keeps the known binary null fixed: root stability already
  closes the root, selected-root permutation would also block it with p-value
  `0.02`, and the matched signal is retained at ARI `0.923113`.
- The remaining null-sibling problem is therefore narrower: after removing
  adaptive sibling PCA, aligning root-stability replay geometry, and adding an
  executable selected-root permutation guard, production promotion still needs
  enough null confidence, guard replicate/alpha validation, and broader
  root/topology coverage, especially for categorical high-cardinality and
  overlap cases.
- The root-only selected permutation profile does not by itself solve
  pass-through null leakage. In the six-case two-replicate mixed smoke,
  `fixed_coordinate_selective_root_v1` leaves one categorical null false split:
  `cat_clear_3cat_4c`, null replicate `1`, reaches a descendant split and
  returns three clusters after the root sibling gate is already closed.
- The broad selected-subtree scope,
  `fixed_coordinate_selective_traversal_v1`, closes that leak but is too
  conservative for signal. In the same two-replicate mixed smoke it has zero
  observed null false splits, but binary signal mean ARI is `0.651330` and
  categorical signal mean ARI is `0.521072`.
- The narrowed scope is packaged as
  `fixed_coordinate_selective_passthrough_v1`. It runs selected-subtree
  permutation only for open descendant splits reachable through an ordinary
  closed sibling ancestor, and skips descendant compounding below roots already
  closed by explicit root-stability or selected-permutation guards.
- In the exact `cat_clear_3cat_4c` two-replicate replay,
  `fixed_coordinate_selective_passthrough_v1` closes both null replicates and
  retains signal with mean ARI `0.902190`. In the six-case two-replicate mixed
  smoke it has zero observed null false splits, binary signal mean ARI
  `0.941101`, and categorical signal mean ARI `0.848463`. The production
  summary still fails closed because the null Wilson upper bound is `0.65762`
  with only two null replicates per case. The refreshed transfer summary now
  reports the required support explicitly: `73` zero-false-split null
  replicates per case, or `71` additional zero-false null replicates from the
  current smoke.
- Checkpoint resume now reads checkpoint CSVs with `keep_default_na=False`.
  Without this, the literal data role `null` is parsed as NaN and resumed
  summaries lose their null rows, which breaks transfer confidence and support
  sizing.
- A six-case ten-replicate recheck of
  `fixed_coordinate_selective_passthrough_v1` finds the remaining
  rooting/null-sibling failure. Categorical nulls stay closed, but
  `binary_many_clusters`, null replicate `7` at seed `20316108`, returns five
  clusters through a pass-through descendant under an already closed root.
  Binary transfer is therefore `fixed_profile_null_inflated`: max null
  false-split rate is `0.1`, Wilson upper confidence is `0.404150`, and one
  observed false split raises the required support target to `110` total null
  replicates per case.
- The false row is not fixed by a simple root-stability barrier. Treating every
  closed unstable root as a hard pass-through stop removes the null split, but
  also collapses two strong `binary_many_clusters` signal rows from ARI `1.0`
  to ARI `0.0`. Increasing local selected-subtree permutation resolution also
  does not solve it: with `999` draws, the false descendant remains selected at
  p-value `0.002`, while the strong signal roots are at p-value `0.001`.
- The next viable non-cross-fit fix is a global selected-family null for
  pass-through descendants. On the observed false row, a full-tree
  pass-through replay gives global p-value about `0.06` with `49` draws, and a
  conservative global-min sibling-family replay gives p-value about `0.05`
  with `99` draws. These diagnostics account for the global search over
  descendants below closed roots, but they are not yet production behavior
  because signal sensitivity and runtime still need optimized validation.
- The global selected-family correction is now executable as the diagnostic
  profile `fixed_coordinate_global_passthrough_v1`. Its
  `global_sibling_min_passthrough_descendant` scope keeps the existing
  selected-root permutation law for roots, but evaluates pass-through
  descendants against the minimum fixed-subspace sibling p-value over every
  binary parent in each fully reselected feature-block permutation null tree.
- A targeted `_rows_for_replicate` replay closes the known
  `binary_many_clusters` null replicate `7`: clusters change from five to one,
  ARI becomes `1.0`, and the global selected-family p-value is `0.05`. The
  matched strong `binary_many_clusters` signal replicate `0` stays at 15
  clusters with ARI `1.0` and selected p-value `0.01`.
- The global selected-family path now uses exact vectorized discrete
  whitening for pure Bernoulli and pure categorical `fixed_coordinate_bh`
  statistics. It computes the same coordinate-wise BH p-value as the canonical
  contrast-covariance path, while mixed feature spaces still use the canonical
  covariance fallback. The known false-row CLI replay now runs in about `22`
  seconds and keeps selected p-value `0.05`.
- The one-row CLI artifact
  `/tmp/klte_global_passthrough_false_row_fast_20260614` confirms the same
  false null seed through the normal fixed-profile validation runner. It
  returns one cluster with `false_split = false` and one global selected-family
  guard block; the transfer summary remains
  `fixed_profile_insufficient_coverage` because this artifact has one null row
  and no matched signal coverage.
- The ten-replicate transfer smokes are favorable but still diagnostic. In
  `/tmp/klte_global_passthrough_binary_10rep_20260614`, the three binary cases
  have zero null false splits across `30` null rows, signal mean ARI
  `0.945084`, minimum signal ARI `0.775207`, and a null Wilson upper bound
  `0.277533`. In
  `/tmp/klte_global_passthrough_categorical_10rep_20260614`, the three direct
  categorical cases have zero null false splits across `30` null rows, signal
  mean ARI `0.832022`, minimum signal ARI `0.661189`, and a null Wilson upper
  bound `0.277533`.
- `/tmp/klte_global_passthrough_categorical_10rep_fast_20260614` reruns the
  same categorical smoke after exact vectorized categorical whitening. It
  preserves zero false splits across `30` null rows and signal mean ARI
  `0.832022`; wall time improves from about `309` seconds to about `147`
  seconds. The remaining runtime cost is the selected-family permutation loop.
- `/tmp/klte_global_passthrough_binary_73rep_support_20260614` extends the
  same binary transfer set to `142` null/signal replicates per case using
  checkpoint resume. `binary_2clusters` and `binary_many_clusters` have zero
  false splits through `142` null rows each. `binary_unbalanced_low` has three
  false splits, so its point false-split rate is `0.021127` but the Wilson
  upper bound is still `0.060270`; production confidence remains fail-closed.
  Signal remains retained, with minimum case-level signal mean ARI `0.854491`
  and lower confidence bound `0.831573`.
- The three support-run false rows are all pass-through descendants below a
  closed root and all land on the `99`-draw selected-family p-value floor
  `0.01`. A `999`-draw replay gives p-values `0.017`, `0.005`, and `0.029`.
  This motivated `fixed_coordinate_global_passthrough_refined_v1`, which keeps
  the same statistic and selected-family null but reruns floor cases at `999`
  draws before accepting a pass-through split.
- Replaying the three false rows through the refined profile closes the two
  rows with refined p-values above `0.01` and keeps the genuinely stronger row
  at p-value `0.005`. The refined ten-replicate binary smoke
  `/tmp/klte_global_passthrough_refined_binary_10rep_20260614` has zero false
  splits across `30` null rows and signal mean ARI `0.945084`, but remains
  confidence-limited and slower because floor cases trigger refinement.
- `/tmp/klte_global_passthrough_refined_binary_142rep_support_20260614`
  validates the refined profile at `142` null/signal replicates per case on
  the same binary transfer set. It has one false split across `426` null rows:
  `binary_unbalanced_low` replicate `96`, with selected-family p-value
  `0.005`. `binary_unbalanced_low` has point false-split rate `0.007042` and
  Wilson upper bound `0.038809`; the binary confidence row becomes
  `fixed_profile_confidence_candidate`. Signal remains retained, with minimum
  case-level signal mean ARI `0.855292` and lower confidence bound `0.832434`.
- `fixed_coordinate_global_passthrough_refined_v1` is therefore the current
  strongest diagnostic candidate for the pass-through null fix. It is not
  production promoted: the binary production summary is `diagnostic_only`, and
  direct-categorical support evidence is still missing.

## Evidence

- `tests/validation/calibration/sibling/gates/101_test_fixed_sibling_gate_profile_validation.py` checks
  profile validation, adaptive-projection detection, fail-closed production
  components for missing coverage, and a tiny `binary_2clusters` output run.
- `raw/inbox/fixed-sibling-gate-profile-validation-20260613.md` records the
  diagnostic purpose, command shape, outputs, targeted root replay, runtime
  selected-root guard replay, narrowed pass-through descendant replay, and
  interpretation.

## Links

- [[data-independent-sibling-gate-traversal-panel-20260613]]
- [[toomanycells-method-20260613]]
- [[open-mathematical-questions]]
