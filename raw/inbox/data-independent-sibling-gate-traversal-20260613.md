# Data-Independent Sibling Gate Traversal 2026-06-13

Command:

```bash
python -m benchmarks.diagnostics.calibration.data_independent_sibling_gate_traversal_panel \
  --output-dir /tmp/kl_te_data_independent_gate_traversal \
  --suite full \
  --case-names binary_2clusters,binary_many_clusters,binary_unbalanced_low,cat_clear_3cat_4c,cat_mod_3cat_4c,cat_highcard_10cat_4c \
  --data-roles null,signal \
  --candidate-methods coordinate_bh,block_bh \
  --sibling-alpha 0.01 \
  --edge-alpha 0.001 \
  --selected-topology-penalties 10,20,50 \
  --replicates 8 \
  --base-seed 20260613
```

Key traversal summary rows:

| source_family | method | penalty | role | mean_ari | max_false_split | min_case_ari |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| binary_template | coordinate_bh | 10 | null | 0.875000 | 0.250 | 0.750000 |
| binary_template | coordinate_bh | 10 | signal | 0.878677 | 0.000 | 0.737441 |
| binary_template | coordinate_bh | 20 | null | 0.958333 | 0.125 | 0.875000 |
| binary_template | coordinate_bh | 20 | signal | 0.896781 | 0.000 | 0.759242 |
| binary_template | coordinate_bh | 50 | null | 0.958333 | 0.125 | 0.875000 |
| binary_template | coordinate_bh | 50 | signal | 0.920961 | 0.000 | 0.813734 |
| categorical_multinomial | coordinate_bh | 10 | null | 0.750000 | 0.375 | 0.625000 |
| categorical_multinomial | coordinate_bh | 10 | signal | 0.830896 | 0.000 | 0.802733 |
| categorical_multinomial | coordinate_bh | 20 | null | 0.833333 | 0.250 | 0.750000 |
| categorical_multinomial | coordinate_bh | 20 | signal | 0.815213 | 0.000 | 0.747419 |
| categorical_multinomial | coordinate_bh | 50 | null | 0.875000 | 0.250 | 0.750000 |
| categorical_multinomial | coordinate_bh | 50 | signal | 0.809927 | 0.000 | 0.747419 |
| categorical_multinomial | block_bh | 10 | null | 0.583333 | 1.000 | 0.000000 |
| categorical_multinomial | block_bh | 10 | signal | 0.836058 | 0.000 | 0.746205 |
| categorical_multinomial | block_bh | 50 | null | 0.791667 | 0.500 | 0.500000 |
| categorical_multinomial | block_bh | 50 | signal | 0.815621 | 0.000 | 0.747419 |

Interpretation:

Traversal-level signal retention is much stronger than all-parent row-level
signal retention because traversal only tests the edge-reachable frontier.
`coordinate_bh` with fixed selected-topology penalties gives useful binary and
categorical signal ARI. However, selected-null traversal still false-splits too
often for production, especially direct categorical high-cardinality nulls and
the small binary two-cluster null surrogate. This means the sibling projection
adaptation failure has a viable fixed-gate replacement direction, but production
requires a remaining selected-topology/edge-null control layer or a stricter,
formally justified traversal penalty.

Follow-up mixed transfer smoke after adding split-geometry and transfer-summary
outputs:

```bash
python -m benchmarks.diagnostics.calibration.data_independent_sibling_gate_traversal_panel \
  --output-dir /tmp/klte_traversal_transfer_16 \
  --suite default \
  --case-names binary_2clusters,binary_many_clusters,binary_unbalanced_low,cat_clear_3cat_4c,cat_mod_3cat_4c,cat_highcard_10cat_4c \
  --candidate-methods coordinate_bh \
  --data-roles null,signal \
  --sibling-alpha 0.01 \
  --edge-alpha 0.0001 \
  --selected-topology-penalties 500,1000 \
  --replicates 16 \
  --base-seed 20260613
```

Transfer summary:

| source_family | penalty | max_null_false_split | max_null_false_root_split | min_signal_mean_ari | status |
| --- | ---: | ---: | ---: | ---: | --- |
| binary_template | 500 | 0.0000 | 0.0000 | 0.987067 | transfer_candidate |
| binary_template | 1000 | 0.0000 | 0.0000 | 0.989778 | transfer_candidate |
| categorical_multinomial | 500 | 0.0625 | 0.0625 | 0.748698 | null_inflated |
| categorical_multinomial | 1000 | 0.0625 | 0.0625 | 0.717796 | null_inflated |

High-cardinality categorical geometry follow-up:

- In `cat_highcard_10cat_4c` null with `128` replicates, penalty `1000`, and
  edge alpha `0.0001`, false-split rate was `7/128 = 0.0546875`.
- Every false split was a root split. The first selected child sizes were
  `9`, `16`, `22`, `31`, `41`, `42`, and `44` out of `200`; root sibling
  p-values ranged from `3.385e-11` to `3.651e-6`.
- In the matching signal run, mean ARI was `0.755123`, root split rate was
  `0.96875`, and the first root min-child size was never below `48`.

Interpretation of the follow-up:

The binary fixed-coordinate traversal candidate transfers in smoke-scale runs.
The direct categorical case remains at the production boundary: stronger
selected-topology penalties suppress the selected null but reduce moderate
categorical signal, while a simple universal min-child guard is not defensible
because other valid signal cases have first root child sizes around `20`--`30`.
The evidence points to a selective/adaptive traversal null law conditioned on
selected topology geometry, not cross-fit projection and not a hard
case-specific balance threshold.

Selective-root permutation diagnostic added to the traversal panel:

```bash
python -m benchmarks.diagnostics.calibration.data_independent_sibling_gate_traversal_panel \
  --output-dir /tmp/klte_selective_root_smoke \
  --suite categorical \
  --case-names cat_highcard_10cat_4c \
  --candidate-methods coordinate_bh \
  --data-roles null,signal \
  --sibling-alpha 0.01 \
  --edge-alpha 0.0001 \
  --selected-topology-penalties 1000 \
  --replicates 4 \
  --base-seed 20260613 \
  --root-selective-bootstrap-replicates 9
```

The diagnostic preserves Bernoulli/categorical feature-block margins, reruns
selected tree construction on each permuted null sample, and reports a
Monte-Carlo p-value for the selected root sibling p-value. In the four-replicate
high-cardinality smoke, the selected-root permutation p-values were not below
`0.01` for either null or signal rows. The known false null root at replicate
`1` moved from raw root p-value `1.707e-6` to selective p-value `0.1`, but
signal roots also had selective p-values `0.1`--`0.2`.

Interpretation:

The permutation diagnostic is a useful executable selected-null object: it
shows that the tiny raw root p-values are not surprising after selected tree
construction under preserved margins. It is not yet the production stopping
rule because it is too conservative for true high-cardinality signal. The next
method step is to search for a selective statistic or conditioning set that
keeps this null control while retaining signal power.

Traversal transfer production-contract smoke:

```bash
python -m benchmarks.diagnostics.calibration.data_independent_sibling_gate_traversal_panel \
  --output-dir /tmp/klte_traversal_contract_smoke \
  --suite default \
  --case-names binary_2clusters,cat_highcard_10cat_4c \
  --candidate-methods coordinate_bh \
  --data-roles null,signal \
  --sibling-alpha 0.01 \
  --edge-alpha 0.0001 \
  --selected-topology-penalties 1000 \
  --replicates 4 \
  --base-seed 20260613
```

Production-admissibility summary:

| source family | method | penalty | transfer status | production decision |
| --- | --- | ---: | --- | --- |
| binary_template | coordinate_bh | 1000 | traversal_transfer_candidate | diagnostic_only |
| categorical_multinomial | coordinate_bh | 1000 | traversal_transfer_null_inflated | fail_closed_undefined |

Interpretation:

The fixed-coordinate traversal repair now has an explicit conservative
admissibility surface. Binary transfer evidence can be carried forward as a
diagnostic-only method candidate, while categorical high-cardinality selected
traversal remains fail-closed. This prevents the binary repair from being
blurred together with the unresolved categorical selective-null problem.

Root feature-subsample stability diagnostic and guard:

The traversal panel now optionally measures selected-root stability by
subsampling feature blocks, rebuilding the selected tree, and computing the ARI
between the original root bipartition and the subsampled root bipartition. A
guard threshold can block a root split when the fixed sibling gate opens but the
selected root is unstable.

High-cardinality categorical smoke:

```bash
python -m benchmarks.diagnostics.calibration.data_independent_sibling_gate_traversal_panel \
  --output-dir /tmp/klte_root_stability_guard_highcard \
  --suite categorical \
  --case-names cat_highcard_10cat_4c \
  --candidate-methods coordinate_bh \
  --data-roles null,signal \
  --sibling-alpha 0.01 \
  --edge-alpha 0.0001 \
  --selected-topology-penalties 1000 \
  --replicates 16 \
  --base-seed 20260613 \
  --root-stability-subsample-replicates 16 \
  --root-stability-feature-fraction 0.8 \
  --root-stability-guard-threshold 0.08
```

Result: the guard blocked the one null false root (`root_stability_subsample_mean_ari = 0.041836`),
reduced null false-split rate from `0.0625` to `0.0`, and blocked no signal
roots. Signal mean ARI stayed `0.824357`.

Mixed six-case smoke:

```bash
python -m benchmarks.diagnostics.calibration.data_independent_sibling_gate_traversal_panel \
  --output-dir /tmp/klte_root_stability_guard_mixed \
  --suite default \
  --case-names binary_2clusters,binary_many_clusters,binary_unbalanced_low,cat_clear_3cat_4c,cat_mod_3cat_4c,cat_highcard_10cat_4c \
  --candidate-methods coordinate_bh \
  --data-roles null,signal \
  --sibling-alpha 0.01 \
  --edge-alpha 0.0001 \
  --selected-topology-penalties 1000 \
  --replicates 8 \
  --base-seed 20260613 \
  --root-stability-subsample-replicates 12 \
  --root-stability-feature-fraction 0.8 \
  --root-stability-guard-threshold 0.08
```

Transfer summary: binary transfer remained a candidate with max null
false-split rate `0.0` and minimum signal mean ARI `0.985565`. Direct
categorical nulls were controlled, but categorical transfer was still
signal-weak because `cat_mod_3cat_4c` had mean ARI `0.698390` at penalty
`1000`.

Categorical penalty/edge follow-up:

- With edge alpha `0.0001`, penalties `100`, `200`, `500`, and `1000`, the
  stability guard controlled all tested categorical nulls. The best minimum
  signal mean ARI was `0.747419` at penalty `100`, still just below the current
  `0.75` threshold because of `cat_mod_3cat_4c`.
- With edge alpha `0.001`, penalties `50`, `100`, and `200`, the stability
  guard again controlled all tested categorical nulls. The best minimum signal
  mean ARI was still `0.747419`.

Interpretation:

Root feature-subsample stability is the first same-data, non-cross-fit
mechanism that directly attacks the selected categorical root artifact while
preserving high-cardinality signal. It is not yet production-ready because the
moderate categorical case remains marginally signal-weak under the current
transfer threshold. The next method question is whether the categorical signal
criterion should be traversal/ARI based with uncertainty intervals, or whether
the stability statistic needs a smoother threshold or feature-family-specific
calibration.

Follow-up threshold sweep:

With edge alpha `0.001`, stability threshold `0.15`, feature-block subsampling
fraction `0.8`, and `12` subsample replicates, categorical penalties `20` and
`50` transferred in a 16-replicate categorical smoke:

- penalty `20`: max categorical null false-split rate `0.0`, minimum signal
  mean ARI `0.771764`
- penalty `50`: max categorical null false-split rate `0.0`, minimum signal
  mean ARI `0.771764`

Mixed binary/categorical candidate smoke:

```bash
python -m benchmarks.diagnostics.calibration.data_independent_sibling_gate_traversal_panel \
  --output-dir /tmp/klte_root_stability_guard_mixed_candidate \
  --suite default \
  --case-names binary_2clusters,binary_many_clusters,binary_unbalanced_low,cat_clear_3cat_4c,cat_mod_3cat_4c,cat_highcard_10cat_4c \
  --candidate-methods coordinate_bh \
  --data-roles null,signal \
  --sibling-alpha 0.01 \
  --edge-alpha 0.001 \
  --selected-topology-penalties 20,50 \
  --replicates 16 \
  --base-seed 20260613 \
  --root-stability-subsample-replicates 12 \
  --root-stability-feature-fraction 0.8 \
  --root-stability-guard-threshold 0.15
```

Transfer summary:

| source family | penalty | max null false split | min signal mean ARI | transfer status | production decision |
| --- | ---: | ---: | ---: | --- | --- |
| binary_template | 20 | 0.0625 | 0.831289 | null_inflated | fail_closed_undefined |
| binary_template | 50 | 0.0000 | 0.865005 | transfer_candidate | diagnostic_only |
| categorical_multinomial | 20 | 0.0000 | 0.771764 | transfer_candidate | diagnostic_only |
| categorical_multinomial | 50 | 0.0000 | 0.771764 | transfer_candidate | diagnostic_only |

Interpretation:

The current strongest method candidate is:

1. remove same-sample adaptive sibling PCA and sibling projection-dimension
   selection;
2. use fixed coordinate-wise BH p-values for sibling contrasts;
3. apply a selected-topology penalty, with penalty `50` transferring across
   both binary and direct categorical smoke cases here;
4. add selected-root feature-subsample stability, using threshold `0.15` in the
   current smoke, to block unstable selected-root artifacts.

This candidate directly addresses the original failure mechanism without
cross-fitting. It remains diagnostic-only because the stability threshold and
selected-topology penalty are smoke-calibrated rather than production-proved.

Confidence-bound follow-up:

The traversal transfer summary now reports Wilson upper confidence bounds for
null false-split rates and one-sided t lower confidence bounds for signal mean
ARI. The production-admissibility components include both point-transfer
evidence and confidence-bound evidence.

Candidate confidence smoke:

```bash
python -m benchmarks.diagnostics.calibration.data_independent_sibling_gate_traversal_panel \
  --output-dir /tmp/klte_candidate_confidence_smoke \
  --suite default \
  --case-names binary_2clusters,binary_many_clusters,binary_unbalanced_low,cat_clear_3cat_4c,cat_mod_3cat_4c,cat_highcard_10cat_4c \
  --candidate-methods coordinate_bh \
  --data-roles null,signal \
  --sibling-alpha 0.01 \
  --edge-alpha 0.001 \
  --selected-topology-penalties 50 \
  --replicates 16 \
  --base-seed 20260613 \
  --root-stability-subsample-replicates 12 \
  --root-stability-feature-fraction 0.8 \
  --root-stability-guard-threshold 0.15
```

| source family | point max null false split | null upper confidence | point min signal ARI | signal lower confidence | confidence status |
| --- | ---: | ---: | ---: | ---: | --- |
| binary_template | 0.0000 | 0.193608 | 0.865005 | 0.822223 | confidence_null_uncertain |
| categorical_multinomial | 0.0000 | 0.193608 | 0.771764 | 0.714284 | confidence_null_uncertain |

Interpretation:

The method candidate still transfers by point estimates, but the 16-replicate
smoke is not strong enough to prove production null control. The Wilson upper
bound for zero false splits is still `0.193608`, far above the `0.05` target.
The production contract therefore fails closed on the confidence component.
The next production validation step is a larger replicate run or an analytic
bound that makes the null upper confidence bound compatible with the false
split target while preserving the signal lower bound.

Validation support sizing:

The transfer summary now reports the per-case null replicate support required
for zero observed false splits to satisfy the Wilson confidence target. For a
false-split target of `0.05` and the current 95% Wilson upper bound, zero false
splits require `73` null replicates per case. The 16-replicate candidate smoke
therefore needs `57` additional zero-false-split null replicates per case
before the null confidence component could pass, assuming no new false splits.
Signal support still needs separate lower-bound validation because the
categorical signal lower confidence bound was `0.714284`, below the `0.75`
threshold.

73-replicate support-target validation:

```bash
python -m benchmarks.diagnostics.calibration.data_independent_sibling_gate_traversal_panel \
  --output-dir /tmp/klte_candidate_73rep_validation \
  --suite default \
  --case-names binary_2clusters,binary_many_clusters,binary_unbalanced_low,cat_clear_3cat_4c,cat_mod_3cat_4c,cat_highcard_10cat_4c \
  --candidate-methods coordinate_bh \
  --data-roles null,signal \
  --sibling-alpha 0.01 \
  --edge-alpha 0.001 \
  --selected-topology-penalties 50 \
  --replicates 73 \
  --base-seed 20260613 \
  --root-stability-subsample-replicates 12 \
  --root-stability-feature-fraction 0.8 \
  --root-stability-guard-threshold 0.15
```

At threshold `0.15`, point-transfer still passed, but confidence failed because
false splits appeared:

| source family | max null false split | null upper confidence | min signal mean ARI | signal lower confidence | confidence status |
| --- | ---: | ---: | ---: | ---: | --- |
| binary_template | 0.027397 | 0.094501 | 0.885719 | 0.870023 | confidence_null_uncertain |
| categorical_multinomial | 0.027397 | 0.094501 | 0.782621 | 0.758985 | confidence_null_uncertain |

False split inspection showed all remaining errors were root splits with
stability just above `0.15`: binary false roots had stability `0.217079` and
`0.230716`; categorical false roots had stability `0.152171`, `0.167870`, and
`0.186931`.

Post-run root-stability threshold sensitivity:

The panel now writes `root_stability_threshold_sensitivity.csv`, which rescales
point outcomes under stricter root-stability thresholds without rerunning tree
construction. Using the pre-alignment 73-replicate evidence, threshold `0.24`
is the first tested threshold that passes both point and confidence checks for
binary and direct categorical cases:

| source family | threshold | max null false split | null upper confidence | min signal mean ARI | signal lower confidence | confidence status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| binary_template | 0.24 | 0.000000 | 0.049992 | 0.835616 | 0.762835 | confidence_candidate |
| categorical_multinomial | 0.24 | 0.000000 | 0.049992 | 0.782621 | 0.758985 | confidence_candidate |

Threshold `0.25` still passed binary confidence but failed categorical signal
confidence (`0.743601` lower bound), and thresholds `0.28`--`0.30` failed
signal confidence for both families. Thus the candidate threshold is narrow and
must be validated prospectively rather than treated as production-calibrated.

Pre-alignment prospective 73-replicate validation at threshold `0.24`:

```bash
python -m benchmarks.diagnostics.calibration.data_independent_sibling_gate_traversal_panel \
  --output-dir /tmp/klte_candidate_73rep_threshold024 \
  --suite default \
  --case-names binary_2clusters,binary_many_clusters,binary_unbalanced_low,cat_clear_3cat_4c,cat_mod_3cat_4c,cat_highcard_10cat_4c \
  --candidate-methods coordinate_bh \
  --data-roles null,signal \
  --sibling-alpha 0.01 \
  --edge-alpha 0.001 \
  --selected-topology-penalties 50 \
  --replicates 73 \
  --base-seed 20260613 \
  --root-stability-subsample-replicates 12 \
  --root-stability-feature-fraction 0.8 \
  --root-stability-guard-threshold 0.24
```

Transfer summary before Hamming/average replay alignment:

| source family | max null false split | null upper confidence | min signal mean ARI | signal lower confidence | confidence status | production decision |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| binary_template | 0.000000 | 0.049992 | 0.835616 | 0.762835 | confidence_candidate | diagnostic_only |
| categorical_multinomial | 0.000000 | 0.049992 | 0.782621 | 0.758985 | confidence_candidate | diagnostic_only |

Case-level summary:

- `binary_2clusters` null false-split rate `0.0`, signal mean ARI `0.885719`
  with lower confidence `0.870023`.
- `binary_many_clusters` null false-split rate `0.0`, signal mean ARI
  `0.835616` with lower confidence `0.762835`.
- `binary_unbalanced_low` null false-split rate `0.0`, signal mean ARI
  `0.897470` with lower confidence `0.859809`.
- `cat_clear_3cat_4c` null false-split rate `0.0`, signal mean ARI `0.821587`
  with lower confidence `0.796457`.
- `cat_highcard_10cat_4c` null false-split rate `0.0`, signal mean ARI
  `0.861350` with lower confidence `0.827391`.
- `cat_mod_3cat_4c` null false-split rate `0.0`, signal mean ARI `0.782621`
  with lower confidence `0.758985`.

Interpretation:

The threshold `0.24` candidate passed the predeclared 73-replicate support
target on the six-case binary/direct-categorical smoke suite. The production
contract still reports `diagnostic_only`, because transfer and confidence
candidate statuses are not production-ready statuses. The remaining work is
external/generalization validation and deciding whether the guarded fixed-gate
candidate should become the production sibling/traversal method.

Broader supported-surface stress probes:

The full supported traversal surface contains 42 binary-template cases and 11
direct categorical-multinomial cases. A 20-replicate all-supported run was
started with the same candidate constants, but it was interrupted after roughly
11 minutes because the diagnostic panel currently writes only at the end and
large selected-tree builds dominate runtime. A 10-replicate targeted stress run
over large binary/categorical cases was also interrupted for the same reason.
This exposed an implementation need: the panel should support case-level
checkpointing before broad validation is attempted inline.

Fast three-replicate probes were then run with the same candidate constants:
fixed `coordinate_bh`, edge alpha `0.001`, selected-topology penalty `50`,
root-stability feature fraction `0.8`, `12` subsample replicates, and
root-stability threshold `0.24`.

Binary stress probe:

```bash
python -m benchmarks.diagnostics.calibration.data_independent_sibling_gate_traversal_panel \
  --output-dir /tmp/klte_candidate_binary_stress_3rep_threshold024 \
  --suite binary \
  --case-names binary_2clusters,binary_many_clusters,binary_many_features,binary_hard_8c,binary_unbalanced_med,binary_noise_feat_80i_400n,binary_noise_feat_30i_500n \
  --candidate-methods coordinate_bh \
  --data-roles null,signal \
  --sibling-alpha 0.01 \
  --edge-alpha 0.001 \
  --selected-topology-penalties 50 \
  --replicates 3 \
  --base-seed 20260613 \
  --root-stability-subsample-replicates 12 \
  --root-stability-feature-fraction 0.8 \
  --root-stability-guard-threshold 0.24
```

Binary result: zero observed null false splits across the seven targeted cases;
all seven signal cases retained. Minimum signal mean ARI was `0.805955`
(`binary_2clusters`), with lower confidence `0.784937`. The family transfer
status was `data_independent_traversal_transfer_candidate`, while confidence
remained `data_independent_traversal_transfer_confidence_null_uncertain`
because three replicates cannot prove the null bound.

Categorical stress probe:

```bash
python -m benchmarks.diagnostics.calibration.data_independent_sibling_gate_traversal_panel \
  --output-dir /tmp/klte_candidate_categorical_3rep_threshold024 \
  --suite categorical \
  --case-names cat_clear_3cat_4c,cat_clear_4cat_4c,cat_clear_5cat_6c,cat_mod_3cat_4c,cat_mod_4cat_6c,cat_highcard_10cat_4c,cat_highcard_20cat_4c,cat_unbal_3cat_4c,cat_overlap_3cat_4c,cat_highd_3cat_500feat,cat_highd_4cat_1000feat \
  --candidate-methods coordinate_bh \
  --data-roles null,signal \
  --sibling-alpha 0.01 \
  --edge-alpha 0.001 \
  --selected-topology-penalties 50 \
  --replicates 3 \
  --base-seed 20260613 \
  --root-stability-subsample-replicates 12 \
  --root-stability-feature-fraction 0.8 \
  --root-stability-guard-threshold 0.24
```

Categorical result: zero observed null false splits across all eleven direct
categorical cases, but signal transfer failed. The minimum signal mean ARI was
`0.080649` for `cat_highcard_20cat_4c`; `cat_overlap_3cat_4c` also remained
weak at mean ARI `0.703263`. The family transfer status was
`data_independent_traversal_transfer_signal_weak`, and the production contract
failed closed.

Interpretation:

The current non-cross-fit candidate is not broadly solved. It appears to
preserve the targeted binary regimes in the small stress probe while keeping
null false splits at zero, but the direct categorical high-cardinality and
overlap regimes expose a power problem under the current coordinate gate and
root-stability guard. The next method work should improve categorical power
without reopening selected-root null inflation, and the diagnostic runner
should add checkpointed case-level output before larger sweeps.

Checkpoint and selected-tree oracle follow-up:

The traversal panel now writes recoverable checkpoint rows after every completed
case-role-replicate unit under `checkpoint_rows/`, while preserving the final
CSV output contract. This directly addresses the broad-run bottleneck observed
above: interrupted all-surface runs now keep completed partial evidence instead
of losing every completed case.

The panel also reports `selected_tree_oracle_ari`, the ARI of the selected
average-linkage tree cut at the true number of clusters. This distinguishes
statistical stopping/gating failures from selected-tree recoverability limits.

Categorical penalty/method probe:

```bash
python -m benchmarks.diagnostics.calibration.data_independent_sibling_gate_traversal_panel \
  --output-dir /tmp/klte_categorical_penalty_method_probe_5rep \
  --suite categorical \
  --case-names cat_clear_3cat_4c,cat_highcard_20cat_4c,cat_overlap_3cat_4c \
  --candidate-methods coordinate_bh,block_bh \
  --data-roles null,signal \
  --sibling-alpha 0.01 \
  --edge-alpha 0.001 \
  --selected-topology-penalties 1,2,5,10,20,50 \
  --replicates 5 \
  --base-seed 20260613 \
  --root-stability-subsample-replicates 12 \
  --root-stability-feature-fraction 0.8 \
  --root-stability-guard-threshold 0.24
```

Result: all tested method/penalty combinations had zero observed null false
splits, but no combination reached categorical transfer. `block_bh` was
strictly better than `coordinate_bh` on the high-cardinality case: the best
`cat_highcard_20cat_4c` mean ARI was `0.659781` at penalty `2`, compared with
`0.402845` for the best coordinate-BH penalty. `cat_overlap_3cat_4c` topped out
near `0.729530`.

Edge-alpha probe:

```bash
python -m benchmarks.diagnostics.calibration.data_independent_sibling_gate_traversal_panel \
  --output-dir /tmp/klte_categorical_edge_probe_5rep \
  --suite categorical \
  --case-names cat_highcard_20cat_4c,cat_overlap_3cat_4c \
  --candidate-methods block_bh,coordinate_bh \
  --data-roles null,signal \
  --sibling-alpha 0.01 \
  --edge-alpha 0.01 \
  --selected-topology-penalties 1,2,5 \
  --replicates 5 \
  --base-seed 20260613 \
  --root-stability-subsample-replicates 12 \
  --root-stability-feature-fraction 0.8 \
  --root-stability-guard-threshold 0.24
```

Result: increasing edge alpha from `0.001` to `0.01` did not improve the
failing categorical cases. Edge-open counts were already saturated for the
signal rows, so the bottleneck is not edge alpha.

Selected-tree oracle check:

For the same five signal replicates, the selected average-linkage tree cut at
the true cluster count had mean oracle ARI:

| case | oracle cut mean ARI | best traversal mean ARI |
| --- | ---: | ---: |
| `cat_highcard_20cat_4c` | `0.659781` | `0.659781` |
| `cat_overlap_3cat_4c` | `0.697741` | `0.729530` |
| `cat_clear_3cat_4c` | `0.757266` | `0.842969` |

Interpretation:

The high-cardinality categorical failure is no longer primarily a sibling
gate/projection failure. Low-penalty `block_bh` reaches the selected-tree
oracle on `cat_highcard_20cat_4c`; the tree itself is the limiting object under
the current average-linkage construction. `cat_overlap_3cat_4c` is similarly
near the selected-tree ceiling. The projection-law fix should therefore remain
focused on fixed-subspace sibling gates, while broader categorical production
support needs either a better categorical tree construction or a benchmark
acceptance criterion that accounts for selected-tree oracle recoverability.

Production-facing fixed-subspace gate:

The fixed-subspace method is now available through the normal gate annotation
and decomposition paths as an explicit opt-in sibling gate:

- `sibling_gate_method="fixed_global_chi_square"`
- `sibling_gate_method="fixed_coordinate_bh"`
- `sibling_gate_method="fixed_block_bh"`

The default remains `sibling_gate_method="projected_wald_inflation"`. The new
fixed gates compute covariance-whitened sibling contrasts from the existing
feature-space covariance layer. `fixed_global_chi_square` applies the full
predeclared fixed-subspace chi-square statistic with `df = len(z)`;
`fixed_coordinate_bh` and `fixed_block_bh` apply predeclared coordinate-wise or
feature-block BH aggregation. They do not use parent PCA projections or
edge-derived sibling projection dimensions, so they remove the identified
same-null-sample adaptive projection failure from the sibling statistic.
The selected-topology penalty is exposed as `sibling_gate_alpha_penalty`, which
divides `sibling_alpha` before sibling FDR while preserving the public
`sibling_alpha` in metadata.

Runtime smoke:

```python
tree.decompose(
    annotations_df=tree.annotations_df.copy(),
    leaf_data=data,
    sibling_gate_method="fixed_coordinate_bh",
    sibling_gate_alpha_penalty=50.0,
    edge_alpha=0.001,
    sibling_alpha=0.01,
)
```

On `binary_2clusters`, the penalized fixed-coordinate smoke wrote `49`
fixed-method sibling rows and `5` open sibling rows in the end-to-end
decomposition smoke. This is production-facing plumbing only; production
promotion still requires the validation contract to pass for the intended
domain and constants.

Implementation alignment:

The diagnostic `coordinate_bh` and `block_bh` p-value paths now delegate to the
production `fixed_subspace_sibling_p_value` implementation. Bonferroni variants
remain diagnostic-only local helpers. A regression test also monkeypatches the
parent-PCA sibling input resolver and verifies that
`sibling_gate_method="fixed_coordinate_bh"` does not call it. This locks the
core repair invariant into tests: fixed gates must not resolve same-sample
parent PCA projections or edge-derived sibling dimensions for the sibling
statistic.

Penalty wiring:

`sibling_gate_alpha_penalty` is now part of gate annotation config metadata and
cache reuse. Tests verify that penalty `50` passes effective sibling alpha
`0.0002` when `sibling_alpha=0.01`, rejects invalid nonpositive penalties, and
forces annotation recomputation when a cached bundle was built with a different
penalty.

Production-facing root stability guard:

The selected-root feature-subsample stability guard is now also exposed through
the production-facing annotation/decomposition path as an explicit opt-in:

- `root_stability_guard_threshold`
- `root_stability_subsample_replicates`
- `root_stability_feature_fraction`
- `root_stability_seed`

When configured, the pipeline compares the supplied tree's root split with
average-linkage root splits rebuilt on deterministic feature-block subsamples.
If the root sibling gate is open and the mean ARI falls below the predeclared
threshold, the root sibling gate is closed by setting
`Sibling_BH_Different=False` and `Sibling_BH_Same=True` on the root row. The
guard writes `Root_Stability_*` diagnostic columns and is included in gate
annotation config metadata, so cached gate annotations are invalidated when the
guard configuration changes. Defaults preserve existing behavior: the guard is
off unless a threshold is supplied.

Regression coverage verifies direct guard blocking, metadata capture, invalid
inert guard configuration, and cache invalidation across guard changes. The
method signature now matches the strongest diagnostic candidate:
fixed-subspace sibling gate plus selected-topology alpha penalty plus
selected-root stability fail-closed guard. This remains an opt-in candidate and
does not change the production-admissibility rule.

Full fixed-subspace chi-square baseline:

The production fixed-subspace gate now also exposes
`fixed_global_chi_square`, the direct fixed-subspace Wald reference
`chi2.sf(z.T @ z, df=len(z))`. The data-independent diagnostic panels expose
the same option as `candidate_method="global_chi_square"`. A two-replicate
`binary_2clusters` traversal smoke with penalty `50`, edge alpha `0.001`,
sibling alpha `0.01`, and root-stability guard threshold `0.24` wrote all
expected artifacts. It produced a point transfer candidate
(`max_null_false_split_rate=0.0`, `min_signal_mean_ari=0.832822`) but remained
fail-closed in production admissibility because the confidence component was
`data_independent_traversal_transfer_confidence_null_uncertain`, as expected
for only two null replicates.

Named non-cross-fit method profiles:

To avoid manually reassembling the repaired method knobs, the gate pipeline now
defines named sibling-gate profiles:

- `fixed_coordinate_guarded_v1`
- `fixed_global_guarded_v1`

Both profiles are marked `diagnostic_candidate`. They expand to
`sibling_gate_alpha_penalty=50.0`,
`root_stability_guard_threshold=0.24`,
`root_stability_subsample_replicates=12`,
`root_stability_feature_fraction=0.8`, and `root_stability_seed=0`; they differ
only in the fixed-subspace sibling statistic (`fixed_coordinate_bh` versus
`fixed_global_chi_square`). The profile id is stored in gate annotation config
metadata, and cache reuse checks include it. Tests verify that
`fixed_coordinate_guarded_v1` does not resolve parent PCA sibling inputs and
that stale adaptive-projection bundles are recomputed when the profile is
requested.

Example:

```python
tree.decompose(
    annotations_df=tree.annotations_df.copy(),
    leaf_data=data,
    sibling_gate_profile="fixed_coordinate_guarded_v1",
    edge_alpha=0.001,
    sibling_alpha=0.01,
)
```

Benchmark-runner and constants-manifest wiring:

The shared KL benchmark runner now accepts the same profile and fixed-gate
parameters:

- `sibling_gate_profile`
- `sibling_gate_method`
- `sibling_gate_alpha_penalty`
- `root_stability_guard_threshold`
- `root_stability_subsample_replicates`
- `root_stability_feature_fraction`
- `root_stability_seed`

It forwards them to both `run_gate_annotation_pipeline` and
`TreeDecomposition`, and records the chosen values in `MethodRunResult.extra`.
A regression test runs a small binary fixture through `_run_kl_method` with
`sibling_gate_profile="fixed_global_guarded_v1"` and checks that the emitted
gate-bundle metadata records the fixed global method and profile constants.

The method-constants manifest now also tracks the fixed-profile constants as
validation targets: `sibling_gate_profile`,
`fixed_sibling_gate_alpha_penalty`, `root_stability_guard_threshold`,
`root_stability_subsample_replicates`, and
`root_stability_feature_fraction`. Their evidence status remains `missing` in
the manifest skeleton until a validation run supplies all required fields.

Hamming selected-tree replay alignment:

The traversal diagnostic panel now builds selected trees, selected-tree oracle
cuts, root-selective p-values, and root feature-subsample stability replays
with explicit Hamming distance and average linkage. Row outputs include
`tree_distance_metric`, `tree_linkage_method`,
`root_stability_tree_distance_metric`, and
`root_stability_tree_linkage_method`, and the manifest records the selected
tree replay contract. This aligns future traversal evidence with the shared KL
runner's supported fixed-profile path. Earlier traversal panel numbers that
were produced before this correction should be treated as pre-alignment
diagnostics unless rerun with the new metric fields present.

Corrected Hamming/average 16-replicate mixed smoke:

```bash
python -m benchmarks.diagnostics.calibration.data_independent_sibling_gate_traversal_panel \
  --output-dir /tmp/klte_traversal_hamming_16 \
  --suite full \
  --case-names binary_2clusters,binary_many_clusters,binary_unbalanced_low,cat_clear_3cat_4c,cat_mod_3cat_4c,cat_highcard_10cat_4c \
  --candidate-methods coordinate_bh \
  --data-roles null,signal \
  --sibling-alpha 0.01 \
  --edge-alpha 0.001 \
  --selected-topology-penalties 50 \
  --replicates 16 \
  --base-seed 20260613 \
  --root-stability-subsample-replicates 12 \
  --root-stability-feature-fraction 0.8 \
  --root-stability-guard-threshold 0.24
```

Result: both binary and direct categorical families were point-transfer
candidates with zero observed null false splits. The run remained
confidence-uncertain because the Wilson upper bound with `16` null replicates
is `0.193608`; signal lower confidence also remained below `0.75` for the
weakest cases.

Corrected Hamming/average 73-replicate support-target validation:

The same command with `--replicates 73` wrote
`/tmp/klte_traversal_hamming_73`. The manifest records
`selected_tree_distance_metric="hamming"` and
`selected_tree_linkage_method="average"`.

| family | max null false split | null upper confidence | min signal mean ARI | signal lower confidence | status |
| --- | ---: | ---: | ---: | ---: | --- |
| binary_template | 0.013699 | 0.073597 | 0.835616 | 0.762835 | point candidate, confidence null-uncertain |
| categorical_multinomial | 0.013699 | 0.073597 | 0.790348 | 0.766477 | point candidate, confidence null-uncertain |

The two false roots occurred at replicate `48` with data seed `20309045`:

- `binary_2clusters` null: root split, root-stability mean `0.422081`,
  q10 `0.179766`, first split `20/50`, sibling p-value `2.070262e-05`.
- `cat_clear_3cat_4c` null: root split, root-stability mean `0.247199`,
  q10 `-0.020921`, first split `24/100`, sibling p-value `1.674478e-04`.

Signal confidence passed for both families, so the remaining blocker is null
confidence. With one observed false split in a 73-replicate null case, Wilson
support requires `110` total null replicates for the upper bound to fall below
`0.05`, assuming no additional false split. That means `37` additional
zero-false null replicates for each affected context. The traversal summary now
reports this observed-count-aware support target.

Corrected Hamming/average penalty-grid probe:

A 16-replicate grid over penalties `50`, `100`, `200`, and `500` found that
binary point transfer remained stable through penalty `500`. Direct
categorical point transfer remained viable at penalties `50` and `100`, but
became signal-weak at `200` and `500`; `cat_highcard_10cat_4c` and
`cat_mod_3cat_4c` were the limiting signal cases. Therefore the evidence does
not justify switching the fixed profile to penalty `500`: it may close the
rare binary false root, but it suppresses categorical signal.

Selected-root permutation guard sensitivity:

The traversal panel now writes `root_selective_guard_sensitivity.csv` when
`root_selective_bootstrap_replicates > 0`. This post-hoc diagnostic asks what
would happen if an opened selected root were blocked unless its selected-root
permutation p-value is at or below `sibling_alpha`. It is diagnostic-only and
does not change traversal defaults.

The selected-root permutation computation is now lazy: every row records the
cheap observed root p-value, but expensive permutation draws are run only when
the root actually opened or was blocked by the root-stability guard for that
method/penalty row. Closed-root rows keep `root_selective_p_value` as missing.
This makes broad validation of the selected-root guard feasible without paying
permutation cost for irrelevant closed-root rows.

Targeted 99-draw checks on the two corrected Hamming false-root contexts:

| case | role | raw root p | selected-root p | interpretation |
| --- | --- | ---: | ---: | --- |
| binary_2clusters | null | 2.070262e-05 | 0.02 | blocked at alpha 0.01 |
| binary_2clusters | signal | 7.749600e-10 | 0.01 | retained at alpha 0.01 |
| cat_clear_3cat_4c | null | 1.674478e-04 | 0.09 | blocked at alpha 0.01 |
| cat_clear_3cat_4c | signal | 1.027797e-11 | 0.01 | retained at alpha 0.01 |

A nine-draw smoke is too coarse: both binary null and signal roots land at
selected-root p-value `0.1`, which would block signal. With `99` draws, the
Monte Carlo floor is `0.01`, enough to retain the matched strong signal root
while blocking the false null root. This suggests a viable next method layer:
fixed-subspace sibling gate plus selected-topology penalty plus root-stability
guard plus selected-root permutation guard. It still needs broad validation
because the current evidence is targeted to the observed false roots.

## Root-Stability Seed Alignment Recheck

The traversal diagnostic now records and accepts an explicit
`root_stability_seed`, defaulting to `0`, matching the named
`fixed_coordinate_guarded_v1` profile. This fixes a diagnostic/runtime mismatch:
older traversal runs used `data_seed + 1701` for root-stability subsampling,
while the production-facing profile used the profile constant
`root_stability_seed = 0`.

Replayed profile check:

```bash
python -m benchmarks.diagnostics.calibration.fixed_sibling_gate_profile_validation \
  --output-dir /tmp/klte_profile_known_false_root_recheck_20260613_v2 \
  --suite full \
  --case-names binary_2clusters,cat_clear_3cat_4c \
  --profiles fixed_coordinate_guarded_v1 \
  --data-roles null,signal \
  --sibling-alpha 0.01 \
  --edge-alpha 0.001 \
  --replicates 1 \
  --base-seed 20309045
```

The profile row audit now records root-level values. With
`root_stability_seed = 0`, `binary_2clusters` null has raw root p-value
`2.070262e-05`, but the root-stability mean ARI is `0.184170`, below the
`0.24` threshold, so the profile blocks the root and returns one cluster. The
matched binary signal root has root-stability mean ARI `1.0` and is retained.
For `cat_clear_3cat_4c`, the null root has root-stability mean ARI `0.283508`,
above the `0.24` threshold, so the profile still falsely opens the root. The
matched categorical signal root has root-stability mean ARI `0.993333` and is
retained.

Aligned traversal replay:

```bash
python -m benchmarks.diagnostics.calibration.data_independent_sibling_gate_traversal_panel \
  --output-dir /tmp/klte_root_selective_recheck_20260613_v2 \
  --suite full \
  --case-names binary_2clusters,cat_clear_3cat_4c \
  --data-roles null,signal \
  --candidate-methods coordinate_bh \
  --selected-topology-penalties 50 \
  --sibling-alpha 0.01 \
  --edge-alpha 0.001 \
  --replicates 1 \
  --base-seed 20309045 \
  --root-selective-bootstrap-replicates 99 \
  --root-stability-subsample-replicates 12 \
  --root-stability-feature-fraction 0.8 \
  --root-stability-seed 0 \
  --root-stability-guard-threshold 0.24
```

Aligned traversal rows reproduce the profile/root-stability behavior:

| case | role | root stability mean | root split | stability blocked | selected-root p | ARI |
| --- | --- | ---: | --- | --- | ---: | ---: |
| binary_2clusters | null | 0.184170 | false | true | 0.02 | 1.000000 |
| binary_2clusters | signal | 1.000000 | true | false | 0.01 | 0.923113 |
| cat_clear_3cat_4c | null | 0.283508 | true | false | 0.17 | 0.000000 |
| cat_clear_3cat_4c | signal | 0.993333 | true | false | 0.01 | 0.881909 |

The previous selected-root categorical null p-value `0.09` is superseded by
the current aligned replay value `0.17` for this working tree; both imply the
same decision at `sibling_alpha = 0.01`: the selected-root guard would block
the categorical false root. The targeted selected-root guard sensitivity
therefore becomes:

- binary: root stability already fixes the known null root under the profile
  seed, but selected-root permutation also blocks it;
- direct categorical: root stability alone does not fix the known null root,
  while selected-root permutation blocks the null root and retains the matched
  signal root in this targeted replay;
- production: both transfer summaries still fail closed because this is
  one-replicate targeted evidence, not broad confidence evidence.

Runtime guard update:

The selected-root permutation idea has now moved from traversal sensitivity
analysis into the production-facing gate annotation path as a default-off
runtime guard. The guard preserves Bernoulli/categorical feature-block margins,
reruns Hamming/average selected-tree construction under block permutations, and
compares the selected-root fixed-subspace sibling p-value with the permutation
null distribution. It writes `Root_Selective_Permutation_*` audit columns and
only closes an open root when the selected-root p-value is above the guard
alpha. It is deliberately restricted to fixed-subspace sibling methods; using
it with `projected_wald_inflation` raises, because the adaptive PCA statistic
is the invalid layer being avoided.

The runtime target remains fail-closed for production. The targeted
`cat_clear_3cat_4c` replay fixes the known null root while retaining matched
signal, but broad validation must still establish the guard replicate count,
guard alpha, null false-split confidence, signal ARI confidence, and supported
feature-family/tree-construction domain.

The packaged profile name for this current candidate is
`fixed_coordinate_selective_root_v1`. It keeps the fixed coordinate sibling
gate and selected-topology penalty from `fixed_coordinate_guarded_v1`, retains
the root-stability guard, and adds a predeclared 99-draw selected-root
permutation guard at alpha `0.01`. This name should be used for future direct
profile-validation runs of the selected-root method layer.
