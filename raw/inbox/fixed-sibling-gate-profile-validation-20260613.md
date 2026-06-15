# Fixed Sibling Gate Profile Validation 2026-06-13

Purpose:

Validate the non-cross-fit fixed sibling-gate profiles through the shared KL
runner. The artifact is designed to prove routing and evidence fields:
production-facing profiles must avoid adaptive sibling projection rows, emit the
fixed sibling method in gate metadata, and produce traversal point/confidence
summaries for null false splits and signal ARI.

Root-stability replay note:

The production-facing root-stability guard now records and uses the same
selected-tree replay contract as the KL benchmark runner: Hamming distance and
average linkage for the supported fixed-profile KL path. This closes an earlier
diagnostic mismatch where subsampled root splits were recomputed with SciPy's
default Euclidean distance even though the selected tree was built with
Hamming distance.

Command shape:

```bash
python -m benchmarks.diagnostics.calibration.fixed_sibling_gate_profile_validation \
  --output-dir /tmp/klte_fixed_profile_validation \
  --suite binary \
  --case-names binary_2clusters \
  --profiles fixed_global_guarded_v1 \
  --data-roles null,signal \
  --sibling-alpha 0.01 \
  --edge-alpha 0.001 \
  --replicates 1 \
  --base-seed 20260613
```

Outputs:

- `fixed_sibling_gate_profile_validation_rows.csv`
- `fixed_sibling_gate_profile_validation_summary.csv`
- `fixed_sibling_gate_profile_validation_transfer_summary.csv`
- `method_constant_evidence_fields.json`
- `production_admissibility_components.csv`
- `production_admissibility_summary.csv`
- `manifest.json`

Interpretation:

This is a routing and smoke-evidence panel. It checks that named profiles such
as `fixed_coordinate_guarded_v1` and `fixed_global_guarded_v1` use fixed
subspace sibling gates in the normal KL runner and do not silently fall back to
`projected_wald_inflation`. Candidate statuses remain diagnostic-only under the
production-admissibility contract unless broader null and signal confidence
bounds pass. TooManyCells is not treated as a direct comparator here; it remains
relational background for tree-first divisive methods and stopping logic.

Root audit update:

The profile validation rows now include root-level sibling p-values,
`root_sibling_open`, `root_stability_guard_blocked`, and root-stability
mean/median/q10 values. This makes root/null failures visible without manually
reading the annotation table.

Targeted replay:

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

With profile seed `0`, the binary null root had sibling p-value
`2.070262e-05` but was closed by root stability (`mean ARI = 0.184170` below
threshold `0.24`). The matched binary signal was retained. The direct
categorical null root had sibling p-value `1.674478e-04` and root-stability
mean ARI `0.283508`, so it remained open and produced a false split. The
matched categorical signal was retained. Therefore the current profile fixes
the known binary null root under aligned seed settings but not the known
direct-categorical null root.

Runtime selected-root guard replay:

The selected-root permutation guard is now available in the production-facing
gate annotation path as an explicit opt-in for fixed-subspace sibling gates.
It is default-off (`root_selective_permutation_guard_replicates = 0`) and is
invalid with `projected_wald_inflation`, because the guard is designed to test
the fixed-subspace sibling p-value after selected-tree construction rather than
repair the same-sample adaptive PCA statistic.

Targeted profile-validation command:

```bash
python -m benchmarks.diagnostics.calibration.fixed_sibling_gate_profile_validation \
  --output-dir /tmp/klte_profile_root_selective_guard_20260614 \
  --suite full \
  --case-names binary_2clusters,cat_clear_3cat_4c \
  --profiles fixed_coordinate_guarded_v1 \
  --data-roles null,signal \
  --sibling-alpha 0.01 \
  --edge-alpha 0.001 \
  --replicates 1 \
  --base-seed 20309045 \
  --root-selective-permutation-guard-replicates 99 \
  --root-selective-permutation-guard-seed 20318458 \
  --root-selective-permutation-guard-alpha 0.01
```

Targeted replay rows:

| case | role | found clusters | ARI | root stability blocked | selected-root p | selected-root guard blocked |
| --- | --- | ---: | ---: | --- | ---: | --- |
| binary_2clusters | null | 1 | 1.000000 | true | 0.02 | false |
| binary_2clusters | signal | 4 | 0.923113 | false | 0.01 | false |
| cat_clear_3cat_4c | null | 1 | 1.000000 | false | 0.17 | true |
| cat_clear_3cat_4c | signal | 6 | 0.881909 | false | 0.01 | false |

Interpretation:

The new guard fixes the known direct-categorical false root in the runtime
profile path while retaining the matched categorical signal in this targeted
replay. The binary null root is already closed by root stability under the
aligned profile seed; selected-root permutation would also block it, but does
not need to rewrite the row because the root is already closed. Production
summaries still fail closed because this is one-replicate targeted evidence,
not broad null and signal confidence evidence.

Packaged selective-root profile:

The same method layer is now packaged as
`sibling_gate_profile="fixed_coordinate_selective_root_v1"`. This profile
sets fixed coordinate BH sibling p-values, selected-topology penalty `50`,
root-stability threshold `0.24`, `12` stability subsamples, feature fraction
`0.8`, selected-root permutation replicates `99`, selected-root guard seed
`0`, and selected-root guard alpha `0.01`. The previous
`fixed_coordinate_guarded_v1` profile still leaves selected-root permutation
off by default and can accept explicit guard settings for targeted replay.

The shared KL runner now records resolved profile settings in `MethodRunResult`
extras rather than raw kwargs. This matters for validation artifacts: a run
that uses a named profile now reports the actual fixed sibling method,
selected-topology penalty, root-stability constants, and selected-root
permutation constants that reached the gate annotation bundle.

Resolved evidence-grid update:

The method-constant evidence builder now derives profile grids from the same
profile resolver used by runtime annotation. A validation run that specifies
only `--profiles fixed_coordinate_selective_root_v1` therefore reports
`permutation_replicate_grid = [99]`, `alpha_grid = [0.01]`, and a profile-level
selected-root guard status of `enabled`, even when the CLI guard override
arguments are left at their defaults. This prevents profile-owned guard
settings from being misreported as disabled in evidence JSON.

Pass-through descendant guard update:

The root-only selected permutation profile fixes selected-root artifacts but
does not catch a pass-through descendant false split when the root sibling gate
is already closed. In the six-case two-replicate smoke at base seed
`20309045`, `fixed_coordinate_selective_root_v1` still has one categorical
null false split: `cat_clear_3cat_4c`, null replicate `1`, returns three
clusters through a descendant split.

The broad selected-subtree profile,
`fixed_coordinate_selective_traversal_v1`, closes that null leak but is too
conservative. In the same six-case two-replicate smoke it has zero observed
null false splits, but binary signal mean ARI is `0.651330` and categorical
signal mean ARI is `0.521072`.

The narrower profile,
`fixed_coordinate_selective_passthrough_v1`, uses selected-subtree permutation
only for open descendant splits that are reachable through an ordinary closed
sibling ancestor. It does not compound below roots already closed by explicit
root-stability or selected-permutation guards. In the exact `cat_clear_3cat_4c`
two-replicate replay it closes both null replicates and retains signal with
mean ARI `0.902190`. In the six-case two-replicate mixed smoke, it has zero
observed null false splits, binary signal mean ARI `0.941101`, and categorical
signal mean ARI `0.848463`. The minimum categorical signal row remains
`0.681800`, so broad signal confidence is still not certified.

Production interpretation:

`fixed_coordinate_selective_passthrough_v1` is the strongest current
diagnostic candidate for the rooting/null-sibling problem. It fixes the known
root artifact and the known pass-through descendant null leak without the broad
signal collapse of `open_internal`. It is still not production-ready: with
only two null replicates per case, the Wilson false-split upper bound remains
`0.65762`, so the production-admissibility contract remains fail-closed on
confidence. The refreshed transfer summary now reports the explicit support
target: `73` zero-false-split null replicates per case are required for a
`0.05` Wilson upper-bound target, so the two-replicate smoke needs `71`
additional zero-false null replicates per case.

Checkpoint resume note:

Checkpoint reads now use `keep_default_na=False` so the literal data role
`null` is preserved instead of being converted to pandas NaN. This matters for
resumed profile-validation summaries, because losing the `null` role removes
the null rows from transfer confidence and support-sizing calculations.

10-replicate pass-through recheck:

```bash
python -m benchmarks.diagnostics.calibration.fixed_sibling_gate_profile_validation \
  --output-dir /tmp/klte_selective_passthrough_profile_mixed_10rep_20260614 \
  --suite full \
  --case-names binary_2clusters,binary_many_clusters,binary_unbalanced_low,cat_clear_3cat_4c,cat_mod_3cat_4c,cat_highcard_10cat_4c \
  --profiles fixed_coordinate_selective_passthrough_v1 \
  --data-roles null,signal \
  --sibling-alpha 0.01 \
  --edge-alpha 0.001 \
  --replicates 10 \
  --base-seed 20309045
```

The six-case, ten-replicate recheck keeps categorical nulls closed but exposes
a binary pass-through null failure. Binary null rows have one false split in
30 rows (`binary_many_clusters`, null replicate `7`, seed `20316108`), so the
binary transfer status becomes `fixed_profile_null_inflated`. Categorical null
rows have zero false splits. The transfer summary reports:

- binary: max null false-split rate `0.1`, Wilson upper bound `0.404150`,
  min signal mean ARI `0.876793`, and one observed null false split. With one
  observed false split, `110` total null replicates per case are required for
  the current `0.05` Wilson target, or `100` additional zero-false null
  replicates from this ten-replicate support level.
- categorical: max null false-split rate `0.0`, Wilson upper bound `0.277533`,
  min signal mean ARI `0.777650`, and `73` required zero-false null replicates
  per case, or `63` additional zero-false null replicates from this support
  level.

The false row is not a selected root split. The root sibling gate is already
closed (`root_sibling_open = false`, root p-value `0.017416`) and root
stability is very low (`mean ARI = 0.016858`), but the root is not explicitly
guard-blocked because the root sibling gate was closed before the stability
guard could rewrite it. A descendant node `N585` with `53` leaves and child
sizes `25/28` opens with raw sibling p-value `1.819745e-05`; the local
selected-subtree permutation guard gives p-value `0.01` with `99` draws and
does not block at alpha `0.01`.

Two attempted simple fixes are not acceptable as production changes:

- A hard pass-through barrier below closed unstable roots fixes the false null
  row, but it also collapses two strong `binary_many_clusters` signal rows
  from ARI `1.0` to ARI `0.0`. The binary signal mean ARI would fall from
  `0.945084` to `0.878418`, with minimum signal ARI `0.0`.
- Increasing local selected-subtree permutation resolution does not separate
  null from signal. At `999` draws, the false null descendant has selected
  p-value `0.002`, while the two strong signal root contexts have selected
  p-value `0.001`.

The promising non-cross-fit repair is a global selected-family null for
pass-through descendants. A diagnostic replay on the false row that rebuilds
the whole selected tree under feature-block permutations and compares the
observed pass-through minimum sibling p-value against the null minimum over
the selected pass-through family gives global p-value about `0.06` with
`49` full-tree draws. A conservative all-binary-parent global-min sibling
family check gives p-value about `0.05` with `99` draws. These diagnostics
block the observed null failure, unlike the local selected-subtree law, because
they account for the global search over closed-root descendants. They are not
yet production behavior: signal sensitivity and runtime need a dedicated,
optimized validation path before a new guard can be promoted.

Implemented global pass-through profile:

The global selected-family correction is now executable as
`fixed_coordinate_global_passthrough_v1`. The profile keeps fixed coordinate
BH, selected-topology penalty `50`, root-stability threshold `0.24`, `12`
stability subsamples, feature fraction `0.8`, selected-root permutation draws
`99`, alpha `0.01`, and uses guard scope
`global_sibling_min_passthrough_descendant`.

The new scope still uses the existing selected-root permutation law for roots.
For pass-through descendant candidates it computes the observed minimum
sibling p-value over current pass-through candidates, then compares it with
the minimum fixed-subspace sibling p-value over every binary parent in each
feature-block-permuted, fully reselected null tree. This is conservative for
the pass-through family and directly corrects the missed global descendant
search.

Targeted replay through `_rows_for_replicate`:

| case | role | replicate | seed | clusters | ARI | min selected p | blocked |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| binary_many_clusters | null | 7 | 20316108 | 1 | 1.000000 | 0.05 | true |
| binary_many_clusters | signal | 0 | 20309045 | 15 | 1.000000 | 0.01 | false |

This closes the known binary pass-through null failure while retaining the
matched strong many-cluster signal. It is still diagnostic, not production:
the first implementation took several minutes for two rows, and broad
support-target null/signal transfer has not been run for the global
selected-family profile.

Runtime optimization:

The global selected-family path now uses the exact vectorized discrete
whitening path for pure Bernoulli and pure categorical `fixed_coordinate_bh`
statistics. For Bernoulli feature spaces it computes the same coordinate-wise
BH p-value as the canonical contrast-covariance path, using
\[
z_j =
\frac{p_{L,j}-p_{R,j}}
{\sqrt{(n_L^{-1}+n_R^{-1})\hat p_j(1-\hat p_j)+10^{-12}}}
\]
before applying the existing coordinate-wise BH aggregation. For pure
categorical blocks it calls the same grouped multinomial drop-last whitening
used by `compute_whitened_wald_contrast`, then applies the existing
coordinate-wise BH aggregation. Mixed feature spaces still fall back to the
canonical covariance object. The known false-row CLI replay now runs in about
`22` seconds while preserving the same selected-family p-value `0.05`; the
earlier direct `_rows_for_replicate` replay remains about `16` seconds.

One-row CLI artifact:

`/tmp/klte_global_passthrough_false_row_20260614` reruns the known null seed
through the normal fixed-profile validation CLI. The row returns one cluster,
ARI `1.0`, `false_split = false`, one global selected-family guard block, and
scope `global_sibling_min_passthrough_descendant`. The transfer summary is
still `fixed_profile_insufficient_coverage`, as expected for a one-row null
artifact with no matched signal coverage.

Small transfer smokes:

- `/tmp/klte_global_passthrough_binary_10rep_20260614`: three binary cases,
  null/signal roles, ten replicates each. Null rows have zero false splits
  (`0/30`), signal mean ARI is `0.945084`, minimum signal ARI is `0.775207`,
  and the transfer row is `fixed_profile_transfer_candidate` with confidence
  still `fixed_profile_confidence_null_uncertain` (`0.277533` Wilson upper
  bound; `63` additional zero-false null replicates per case needed).
- `/tmp/klte_global_passthrough_categorical_10rep_20260614`: three direct
  categorical cases, null/signal roles, ten replicates each. Null rows have
  zero false splits (`0/30`), signal mean ARI is `0.832022`, minimum signal ARI
  is `0.661189`, and the transfer row is `fixed_profile_transfer_candidate`
  with confidence still `fixed_profile_confidence_null_uncertain` (`0.277533`
  Wilson upper bound; `63` additional zero-false null replicates per case
  needed).
- `/tmp/klte_global_passthrough_categorical_10rep_fast_20260614` reruns the
  same three direct categorical cases after routing fixed-coordinate
  categorical p-values through exact vectorized grouped whitening. It preserves
  the same aggregate behavior: zero false splits across `30` null rows, signal
  mean ARI `0.832022`, minimum signal ARI `0.661189`, and the same `0.277533`
  null Wilson upper bound. Wall time drops from about `309` seconds for the
  earlier categorical smoke to about `147` seconds. The remaining cost is the
  repeated selected-tree permutation loop, not a different categorical
  statistic.

Binary support run:

`/tmp/klte_global_passthrough_binary_73rep_support_20260614` extends the same
three binary cases to support-level sample sizes with checkpoint resume. At
`73` replicates per case, `binary_unbalanced_low` has one null false split,
moving the observed-count Wilson support target to `110`. At `110`, it has two
false splits, moving the target to `142`. At `142`, it has three false splits:
the point false-split rate is `0.021127`, but the Wilson upper bound remains
`0.060270`, so the production confidence component stays fail-closed.
`binary_2clusters` and `binary_many_clusters` have zero false splits through
`142` null replicates each. Signal remains retained: the minimum signal mean
ARI across the three cases is `0.854491`, with lower confidence bound
`0.831573`.

All three false rows are pass-through descendants below a closed root and land
on the `99`-draw selected-family Monte Carlo floor p-value `0.01`. A
high-resolution `999`-draw replay gives selected-family p-values `0.017`,
`0.005`, and `0.029`. This motivates a separate refined diagnostic profile,
`fixed_coordinate_global_passthrough_refined_v1`, which keeps the base
`99`-draw global pass-through guard but reruns floor cases at `999` draws
before accepting a pass-through split. Replaying the three false rows through
the normal runner closes the two rows with refined p-values above `0.01` and
keeps the row with p-value `0.005`.

Refined binary smoke:

`/tmp/klte_global_passthrough_refined_binary_10rep_20260614` reruns the same
three binary cases for ten null/signal replicates with
`fixed_coordinate_global_passthrough_refined_v1`. It preserves the prior smoke
behavior: zero false splits across `30` null rows, signal mean ARI `0.945084`,
and minimum signal ARI `0.775207`. Wall time is about `106` seconds because
floor pass-through cases trigger the `999`-draw refinement.

Refined binary support:

`/tmp/klte_global_passthrough_refined_binary_142rep_support_20260614` extends
the refined profile to `142` null/signal replicates per case on the same three
binary cases. It has one false split across `426` null rows: the known
`binary_unbalanced_low` replicate `96`, with refined selected-family p-value
`0.005`. `binary_2clusters` and `binary_many_clusters` have zero false splits
through `142` null rows each. `binary_unbalanced_low` has point false-split
rate `0.007042` and Wilson upper bound `0.038809`, so the refined binary
confidence row becomes `fixed_profile_confidence_candidate`. Signal remains
retained: minimum case-level signal mean ARI is `0.855292`, with lower
confidence bound `0.832434`. The production summary is therefore
`diagnostic_only`, not `fail_closed_undefined`, because all required
components remain diagnostic rather than production-ready.

Interpretation update:

`fixed_coordinate_global_passthrough_v1` is no longer the endpoint for the
non-cross-fit pass-through null fix. It repairs the original
`binary_many_clusters` failure and has acceptable point behavior on the binary
support run, but its `99`-draw selected-family floor leaves boundary false
splits in `binary_unbalanced_low` and fails the current Wilson confidence
target. The next diagnostic candidate is
`fixed_coordinate_global_passthrough_refined_v1`: it keeps the same
fixed-subspace statistic and selected-family null, but adds high-resolution
replay for Monte Carlo floor pass-through families. The refined profile clears
the binary support confidence blocker on the current transfer set, but it is
not production-promoted: it still needs direct-categorical support evidence
and a production contract that can promote diagnostic candidate components
only after the broader method evidence is accepted.
