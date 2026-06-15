---
title: Projected-Wald Statistic
type: concept
status: reviewed
updated: 2026-06-14
sources:
  - manuscript/guides/full_method_logic_map.md
  - manuscript/sections/method/edge_test.tex
  - manuscript/sections/method/sibling_test.tex
  - benchmarks/diagnostics/calibration/statistic_distribution_shape_panel.py
  - benchmarks/diagnostics/calibration/covariance_laplacian_panel.py
  - benchmarks/diagnostics/calibration/differential_statistic_validity_panel.py
  - benchmarks/diagnostics/calibration/regularized_wald_statistic_panel.py
  - benchmarks/diagnostics/calibration/null_law_decomposition_panel.py
  - benchmarks/diagnostics/calibration/data_independent_sibling_gate_panel.py
  - benchmarks/diagnostics/calibration/data_independent_sibling_gate_traversal_panel.py
  - benchmarks/diagnostics/calibration/fixed_sibling_gate_profile_validation.py
  - kl_clustering_analysis/hierarchy_analysis/decomposition/gates/orchestrator.py
tags:
  - method
  - statistics
---

# Projected-Wald Statistic

## Summary

The projected-Wald statistic is the projected quadratic form used to combine a
standardized high-dimensional contrast into a lower-dimensional test statistic
with a chi-square reference under the fixed-subspace approximation.

## Details

For edge tests, the raw child-parent contrast is standardized coordinatewise
with the nested variance model. For sibling tests, the left-right contrast is
standardized with the two-sample Bernoulli variance model. In both cases, the
standardized vector is projected into parent-local spectral directions before
summing squared projected coordinates.

The manuscript terminology distinguishes the implemented orthonormal
projected-Wald reference from alternative whitening statistics. Eigenvalues
select the local subspace, while the implemented reference compares the raw
orthonormal projected quadratic against a chi-square distribution with the
selected projection dimension.

Two covariance objects must remain separate. The sibling contrast covariance
standardizes the left-right test statistic. The parent spectral covariance
summarizes descendant null-whitened tangent geometry and selects the projection
basis and dimension. A diagonal sibling contrast covariance can therefore
coexist with a dense parent spectral covariance.

Covariance-inferred Satterthwaite degrees of freedom and scale are diagnostic
comparators for distribution-shape analysis. Current diagnostics show that
effective df can be close to the implemented projection df while the inferred
scale remains large, so df count alone is not enough to explain the observed
tail behavior.

The differential statistic-validity panel adds a local validity decomposition:
Fisher boundary geometry, whitening sensitivity, projection sensitivity, and
tree-selection geometry are checked before a selected sibling statistic is
treated as a fixed-subspace chi-square candidate. Boundary or selection
instability remains fail-closed for production.

The regularized Wald statistic panel tests Jeffreys, Dirichlet, and root-shrink
smoothing as candidate repairs for Fisher/Wald boundary instability. The first
fixed-tree smoke shows that smoothing removes boundary instability but leaves
the chi-square tail law severely misaligned, so the implemented statistic still
cannot be promoted by smoothing alone.

The null-law decomposition panel identifies the sharper mechanism. Under fixed
topology, the same-sample adaptive parent projection yields inflated sibling
tails even though the projection rows are orthonormal and the induced
quadratic-operator weights are `1`. Independent tree-sample and random fixed
orthonormal projections are near nominal, so the broken assumption is not the
quadratic formula itself but using a projection and dimension learned from the
same null sample as the tested sibling contrast.

The data-independent sibling-gate panel tests the corresponding non-cross-fit
repair direction: keep the selected topology but remove adaptive PCA and
adaptive projection dimension from the tested statistic. Fixed coordinate-wise
Bonferroni or BH aggregation with a predeclared selected-topology penalty is a
diagnostic candidate because it no longer learns the sibling test subspace from
the sibling contrast itself. It is not yet production calibration; the current
contract records successful smoke rows as diagnostic-only until the topology
penalty and generalization are validated.

The first transfer evidence favors BH over Bonferroni for binary cases. Across
`binary_2clusters`, `binary_many_clusters`, and `binary_unbalanced_low`,
`coordinate_bh` with selected-topology penalty `10` is the only tested grid
candidate that is null-controlled and signal-retaining under the current
all-parent thresholds. The same candidate is not yet adequate for direct
categorical cases because categorical signal is concentrated in larger parents
and falls below the all-parent signal-retention threshold.

Feature-block fixed gates provide the natural categorical comparator: each
categorical feature block contributes a chi-square statistic with degrees of
freedom equal to its simplex chart dimension. The categorical transfer smoke
shows that this block aggregation remains null-conservative but does not fix
all-parent categorical power, so the remaining categorical issue lies outside
adaptive projection and simple block aggregation.

The traversal panel shows why all-parent row retention was too pessimistic as
the final method criterion. When fixed coordinate/block p-values are evaluated
only on the edge-reachable traversal frontier, binary and direct categorical
signal ARI are strong. The remaining production blocker is selected-null
traversal false splitting, not same-sample adaptive sibling projection.

The stricter traversal follow-up narrows that blocker. Binary cases transfer in
smoke-scale runs with fixed `coordinate_bh`, edge alpha `0.0001`, and
selected-topology penalties `500`--`1000`. Direct categorical cases remain
borderline: high-cardinality null false splits are selected root splits with
small first children, but a universal min-child guard would block other valid
signal regimes. The next non-cross-fit mathematical object is therefore a
selective/adaptive traversal null law conditioned on selected topology geometry,
not a return to sample splitting and not a hard balance threshold.

The traversal panel now carries this boundary into the production-admissibility
contract. Binary fixed-coordinate transfer can be represented as a
diagnostic-only candidate, while categorical high-cardinality transfer fails
closed. This is the current concrete repair to the original adaptive projection
failure: remove the learned sibling projection and dimension from the tested
statistic, use a fixed coordinate BH gate with a selected-topology penalty, and
restrict any promotion to domains where traversal transfer is independently
validated.

The current categorical extension adds selected-root feature-subsample
stability. This does not reintroduce adaptive projection into the statistic;
instead, it asks whether the selected root topology is stable under
feature-block perturbations. In smoke runs, this stability guard blocks
high-cardinality categorical null root artifacts while preserving
high-cardinality signal, but moderate categorical transfer remains just under
the current signal threshold.

The follow-up threshold sweep identifies the current strongest diagnostic
method: fixed coordinate BH, selected-topology penalty `50`, edge alpha
`0.001`, root feature-subsample fraction `0.8`, `12` stability subsamples, and
root-stability threshold `0.15`. In a 16-replicate six-case mixed smoke this
candidate controlled binary and categorical selected nulls and retained signal
above the current ARI threshold. The production question is no longer the
original adaptive-projection failure; it is validation of the penalty and
stability threshold beyond smoke-scale synthetic cases.

The confidence-bound transfer layer prevents overclaiming this smoke. The same
16-replicate candidate has zero observed null false splits, but the Wilson upper
confidence bound is `0.193608`, so production remains fail-closed on the
confidence component. The method candidate has fixed the identified statistic
failure in diagnostic form; production promotion requires enough null support or
an analytic selected-null bound to make the false-split upper bound small.
For the current 95% Wilson bound and a `0.05` false-split target, that means
`73` zero-false-split null replicates per case.

The first 73-replicate validation at root-stability threshold `0.15` found that
the threshold was too low: observed null false splits kept the null confidence
upper bound at `0.094501`. A post-run threshold-sensitivity summary identifies
`0.24` as the first tested threshold that would pass both binary and categorical
confidence checks on that evidence. Because this threshold was chosen after
looking at the validation rows, it is a candidate for prospective validation,
not a production constant.

The corrected Hamming/average prospective validation at threshold `0.24` did
not production-clear: binary and direct categorical families each had one null
false root in the 73-replicate support run, leaving Wilson upper confidence at
`0.073597`. After aligning the traversal diagnostic with the profile
`root_stability_seed = 0`, the known binary null root is blocked by root
stability, but the known `cat_clear_3cat_4c` null root still opens. A targeted
99-draw selected-root permutation diagnostic blocks that categorical false root
while retaining the matched signal. The concrete replacement candidate is
therefore fixed coordinate BH sibling p-values plus selected-topology penalty
plus root-stability guarding, with selected-root permutation as the next
non-cross-fit layer to validate. Production remains fail-closed until broad
confidence evidence or a derivation supports that selected-root layer.

The selected-root permutation layer is now implemented as a default-off runtime
guard for fixed-subspace sibling gates. It does not change the fixed
coordinate/block/global sibling statistic. Instead, it conditions the root
decision on a Monte-Carlo selected-tree null that preserves
Bernoulli/categorical feature-block margins and reruns Hamming/average tree
selection. A targeted profile replay closes the known `cat_clear_3cat_4c`
null root and retains the matched signal, so the remaining projected-Wald
issue is production validation of this selected-root null layer and supported
tree domains, not the original same-sample adaptive PCA statistic.

The packaged non-cross-fit candidate is
`fixed_coordinate_selective_root_v1`: fixed coordinate BH replaces the
same-sample adaptive sibling projection, selected-topology penalty `50`
controls edge-selected traversal multiplicity, root-stability threshold `0.24`
guards unstable selected roots, and the `99`-draw selected-root permutation
guard conditions the remaining root decision on selected-tree null behavior.
This is a named diagnostic profile, not a production default.

The newest rooting/null-sibling refinement is
`fixed_coordinate_selective_passthrough_v1`. It keeps the fixed coordinate BH
statistic and selected-root guard, but adds selected-subtree permutation only
for descendant splits reached through an ordinary closed sibling ancestor. This
targets the pass-through leak that root-only guarding misses without applying
the broad `open_internal` guard to every internal signal split. In the
six-case two-replicate mixed smoke, the narrowed scope has zero observed null
false splits, binary signal mean ARI `0.941101`, and categorical signal mean
ARI `0.848463`; production remains fail-closed because the null confidence
bound is still smoke-scale. The fixed-profile validation summary now reports
the support requirement directly: with zero observed false splits, production
confidence at a `0.05` Wilson upper-bound target requires `73` null replicates
per case, so the current smoke needs `71` additional zero-false null replicates
per case.

The ten-replicate pass-through recheck shows that the local selected-subtree
law is still too narrow. `binary_many_clusters`, null replicate `7`, selects a
descendant split below an already closed unstable root; the local
selected-subtree p-value remains small even when permutation resolution is
increased to `999` draws. A hard barrier below every closed unstable root is
also invalid, because it collapses strong many-cluster signal rows whose root
binary split is unstable but whose descendant structure is real. The next
non-cross-fit statistic is therefore a global selected-family null for
pass-through descendants: rebuild the whole selected tree under
feature-block permutations and compare the observed pass-through minimum
sibling p-value against the null minimum over the selected family. Diagnostic
replays move the observed null failure to global p-values around `0.05` to
`0.06`, but this layer is not yet production behavior because signal
sensitivity and runtime need optimized validation.

That global correction is now executable as
`fixed_coordinate_global_passthrough_v1`. Its guard scope,
`global_sibling_min_passthrough_descendant`, compares pass-through descendant
candidates with the minimum fixed-subspace sibling p-value over every binary
parent in each fully reselected feature-block permutation null tree. In a
targeted replay it closes the known `binary_many_clusters` null replicate `7`
and retains the matched strong signal replicate at ARI `1.0`. The method
now uses exact vectorized discrete whitening for pure Bernoulli and pure
categorical fixed coordinate BH. In a three-case, ten-replicate binary smoke
it has zero null false splits across `30` null rows and signal mean ARI
`0.945084`; in a three-case, ten-replicate direct-categorical smoke it has zero
null false splits across `30` null rows and signal mean ARI `0.832022`. The
direct-categorical rerun preserves those statistics while improving wall time
from about `309` seconds to about `147` seconds. The method remains
diagnostic-only because support-level binary validation finds boundary
pass-through false splits in `binary_unbalanced_low`.

The refined successor,
`fixed_coordinate_global_passthrough_refined_v1`, leaves the statistic and
selected-family null unchanged but reruns `99`-draw Monte Carlo floor
pass-through families at `999` draws. In the binary support run, the unrefined
profile has three false splits in `binary_unbalanced_low` at `142` null
replicates, with Wilson upper bound `0.060270`; the refined replay closes two
of those three rows and leaves one genuinely strong selected-family event at
p-value `0.005`. This makes the refined profile the current strongest
diagnostic candidate. Its own `142`-replicate binary support run has one false
split across `426` null rows, `binary_unbalanced_low` Wilson upper bound
`0.038809`, and minimum signal mean ARI lower confidence `0.832434`, moving the
binary production summary from fail-closed to diagnostic-only. Direct
categorical support evidence remains open.

## Evidence

- `manuscript/sections/method/edge_test.tex` defines the nested edge contrast,
  local spectral summary, and projected edge p-value.
- `manuscript/sections/method/sibling_test.tex` defines the sibling contrast,
  sibling projection dimension, and orthonormal reference law.
- `manuscript/guides/full_method_logic_map.md` gives the canonical
  terminology and assumptions.
- `benchmarks/diagnostics/calibration/statistic_distribution_shape_panel.py`
  compares current chi-square df against covariance-inferred df/scale.
- `benchmarks/diagnostics/calibration/covariance_laplacian_panel.py` separates
  sibling contrast covariance connectivity from parent spectral covariance
  connectivity.
- `benchmarks/diagnostics/calibration/differential_statistic_validity_panel.py`
  tests local Fisher, projection, and nonsmooth selection conditions for the
  projected-Wald sibling statistic.
- `benchmarks/diagnostics/calibration/regularized_wald_statistic_panel.py`
  tests smoothed projected-Wald variants under fixed-tree-first null
  validation.
- `benchmarks/diagnostics/calibration/null_law_decomposition_panel.py`
  decomposes fixed-topology sibling null behavior into same-sample adaptive,
  independent tree-sample, and random fixed orthonormal projection sources.
- `benchmarks/diagnostics/calibration/data_independent_sibling_gate_panel.py`
  evaluates fixed-coordinate sibling gates as a non-cross-fit candidate repair
  for same-data selected topology.
- `benchmarks/diagnostics/calibration/data_independent_sibling_gate_traversal_panel.py`
  evaluates the fixed gates through the actual top-down decomposition
  traversal.
- `benchmarks/diagnostics/calibration/fixed_sibling_gate_profile_validation.py`
  validates the runtime fixed-profile path and the opt-in selected-root
  permutation guard evidence fields.

## Links

- [[kl-te-method]]
- [[top-down-traversal]]
- [[tree-decomposition]]
- [[differential-statistic-validity-panel-20260613]]
- [[regularized-wald-statistic-panel-20260613]]
- [[null-law-decomposition-panel-20260613]]
- [[data-independent-sibling-gate-panel-20260613]]
- [[data-independent-sibling-gate-traversal-panel-20260613]]

## Open Questions

- Does the final manuscript present a proof under fixed projection, or clearly
  label the data-dependent projection as a working approximation?
- Which sensitivity analysis justifies the minimum spectral dimension of `2`?
