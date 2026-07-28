---
title: Selected Neighborhood Bottleneck Law
type: analysis
status: draft
updated: 2026-07-28
sources:
  - wiki/analyses/traversal-neighborhood-method-comparison.md
  - wiki/sources/selected-neighborhood-signal-flow-literature-20260617.md
  - wiki/sources/graph-neural-geometry-spectral-artifact-literature-20260617.md
  - wiki/sources/selected-neighborhood-conditional-support-panel-20260617.md
  - wiki/sources/selected-neighborhood-internal-spectral-flow-panel-20260617.md
  - wiki/sources/old-vs-current-method-stack-comparison-20260615.md
  - wiki/sources/sibling-null-prior-interpolation-audit-20260604.md
  - wiki/sources/specific-small-method-benchmark-20260615.md
  - wiki/sources/selected-neighborhood-distribution-panel-20260615.md
  - wiki/sources/overlap-conditional-topology-law-panel-20260615.md
  - wiki/sources/retained-pass-through-topology-likelihood-panel-20260615.md
  - wiki/sources/selected-neighborhood-spectral-flow-diagnostic-20260616.md
  - wiki/sources/spectral-transport-passthrough-guard-20260616.md
  - tree_break_selection/hierarchy_analysis/decomposition/gates/gate_evaluator.py
  - tree_break_selection/hierarchy_analysis/decomposition/gates/orchestrator.py
  - tree_break_selection/hierarchy_analysis/statistics/sibling_divergence/inflation_correction/empirical_null_inflation_estimation.py
  - benchmarks/diagnostics/calibration/overlap/overlap_conditional_topology_law_panel.py
  - benchmarks/diagnostics/calibration/selected/neighborhood/selected_neighborhood_distribution_panel.py
tags:
  - analysis
  - traversal
  - neighborhood
  - bandwidth
  - calibration
---

# Selected Neighborhood Bottleneck Law

## Summary

The merged method idea is a selected-neighborhood bottleneck law: keep the
current selected-family guards that suppress null pass-through over-splitting,
recover internal signal only when directed incoming/outgoing topology is
coherent, and use the old bandwidth variables to localize why the inference is
unsupported rather than to decide the split by themselves.

This makes bandwidth a bottleneck diagnostic and support regularizer, not a
standalone traversal threshold. Fragmentation is an audit outcome and failure
mode label, not a production penalty term.

## Details

The shared traversal skeleton remains unchanged:

```text
split = binary(parent) AND edge_open(parent) AND sibling_open(parent)
pass-through = binary(parent) AND edge_open(parent) AND sibling_closed(parent)
               AND descendant_can_split(parent)
```

The merged law acts before promotion of a sibling-closed or guard-blocked
internal candidate. It separates three questions that were previously mixed:

1. Is this selected family allowed to keep walking without reopening selected
   null false positives?
2. Is the candidate an incoming/outgoing coherent internal recovery under the
   selected topology?
3. Which bandwidth coordinate makes the inference unsupported or nonlocal?

The proposed score is a diagnostic posterior/logit, not a production p-value:

```text
recovery_logit =
  directed_topology_score
  + bandwidth_support_score
  - selected_family_false_positive_risk
```

There should not be a direct fragmentation penalty such as a cost on the final
cluster count, singleton count, or effective cluster count. Those quantities
are downstream diagnostics for evaluating a run. They are not measurable local
evidence for whether a selected sibling candidate is valid.

The directed topology score should be dominated by incoming branch balance,
outgoing balance, outgoing edge-norm balance, and their income-conditioned
product. This follows the small-method benchmark finding that the old
topology-neighborhood bandwidth term is not a standalone separator, while
outgoing topology and `balance_product` carry the useful internal recovery
signal.

The selected-family false-positive risk remains the refined guard's job. The
selected-neighborhood distribution panel shows the same
`left_pass_through_downstream_split_right_stops` pattern on selected-null rows
and possible signal over-suppression rows. Therefore downstream accepted splits
are not enough to retain a pass-through walk.

The bandwidth support score should reuse the old neighborhood coordinates:

- `tau_b`: stopping-edge distance scale.
- `tau_t`: stable/support neighborhood distance scale.
- `tau_s`: signal-neighborhood distance scale.
- `h_k`: log-scale spread for projection/neighborhood matching.
- nearest stable distance and nearest signal distance.
- local support count, signal count, and selected-nonnull exclusion count.

The current conditional-topology diagnostic already computes these values with
cached tree distances and excludes selected non-null rows from empirical-null
support. The merged law should keep that rule. A selected non-null row may
explain why a candidate is near signal, but it must not become empirical-null
calibration support.

The bandwidth object should be region-specific and set-valued. For a selected
topology region \(R\), use

\[
\theta_R=(\tau_{b,R},\tau_{t,R},\tau_{s,R},h_{k,R})
\]

or an interval set

\[
\Theta_R=
[\tau^-_{b,R},\tau^+_{b,R}]
\times[\tau^-_{t,R},\tau^+_{t,R}]
\times[\tau^-_{s,R},\tau^+_{s,R}]
\times[h^-_{k,R},h^+_{k,R}].
\]

For the signal-neighborhood coordinate, an optimistic admissible interval can
be estimated by comparing direct signal recovery against selected-null
reopening in the same benchmark region. If \(q_r^{\mathrm{sig}}(R)\) is the
\(r\)-quantile of required signal \(\tau_s\) values and
\(q_\ell^{0}(R)\) is the \(\ell\)-quantile of selected-null required
\(\tau_s\) values, then

\[
I_s(R;r,\ell)=
\left[q_r^{\mathrm{sig}}(R),\ q_\ell^{0}(R)\right].
\]

The interval is diagnostic-only and usable only when the lower endpoint is no
larger than the upper endpoint. If \(q_r^{\mathrm{sig}}(R)>q_\ell^{0}(R)\),
the selected null reopens before the desired signal fraction is recovered, so
the region remains fail-closed.

The distance inside these kernels is now measurable both as topology-hop
distance and as the branch-length tree metric

\[
d_\ell(u,v)=\sum_{e\in\operatorname{path}(u,v)} \ell_e,
\]

instead of only the hop count \(d_T(u,v)\). The branch-length rerun closes the
operational distance-cache gap: all `69,860` comparison rows report
`cached_all_pairs_branch_length_tree_distances`. It changes the scale of
\(\tau_s\) and nearest-neighborhood localization, but it does not change the
admissibility conclusion. At the method level, the branch-length interval
\[
I_s(0.25,0.05)=[0.048578,0.003706]
\]
is empty, so the selected-null side still opens before the requested signal
fraction is recovered. The same failure appears in hop units,
\([15.480732,10.617175]\), but branch lengths make the mechanism more local and
more visible.

The external literature gives the correct interpretation of this neighborhood
term. Diffusion maps make a kernel neighborhood into a local diffusion
operator, so \(\tau_t\) and \(\tau_s\) should measure where coherent signal can
flow through the selected tree geometry. Tree wavelets and treelets make
coarse/fine tree scale explicit: coherent branch signal should persist under
coarsening, while fragment artifacts should appear as local high-frequency
detail. Therefore bandwidth is evidence about locality, smoothness, and
support. It is not itself a calibrated p-value.

The corresponding statistical target is not

\[
p_{\mathrm{interp}}(u) < \alpha .
\]

It is admissible conditional support for a selected row:

\[
\Pr_0\{S_u \ge s_u \mid
\mathcal E_u,\ G_u,\ H_u,\ \mathcal N_\tau(u)\},
\]

where \(\mathcal E_u\) contains the selected traversal/root event and
\(\mathcal N_\tau(u)\) is restricted to admissible null or external-support
records. If the support neighborhood is empty, topology-incompatible, or
dominated by selected non-null rows, the row is unsupported and remains
fail-closed.

The bandwidth bottleneck status should explicitly name the failing inference
coordinate. At minimum, each ambiguous candidate should be assigned one or more
of these support statuses:

- `support_bottleneck`: not enough strict-null, edge-blocked, or labeled
  recovery support in the local selected-neighborhood stratum.
- `distance_bottleneck`: nearest stable or signal neighborhood is too far
  relative to `tau_t` or `tau_s`.
- `scale_bottleneck`: local log projection scale is outside the supported
  `h_k` band.
- `selection_bottleneck`: useful nearby evidence exists only through selected
  non-null rows excluded from support.
- `coverage_bottleneck`: the row is traversal-only, with no joined old/current
  topology-neighborhood evidence.
- `guard_bottleneck`: the selected-family guard blocks the candidate and no
  null-side pass-through law has been validated.
- `spectral_bottleneck`: MP-supported eigenvectors rotate, eigenvalues drift,
  or the shared MP-certified dimension disappears across the local
  parent-child neighborhood.

This gives the method two outputs for every ambiguous node: a conservative
traversal action and a localized reason why the bandwidth inference could not
support recovery. High fragmentation is then controlled by selected-family
false-positive control and support admissibility, not by penalizing
fragmentation after the fact.

The promotion rule should be asymmetric:

```text
selected-null side:
  keep refined fail-closed guard until a null-side selected-pass-through
  false-positive law exists.

signal side:
  allow diagnostic recovery only when directed topology is coherent and the
  bandwidth bottleneck status is not support/coverage/selection blocked.
```

That asymmetry addresses both observed failure modes. It prevents the old
neighborhood layer from reopening selected-null over-splits, while giving the
current fail-closed method a precise path for recovering real internal
pass-through signal once the missing topology likelihood is identifiable.

## Evidence

- [[traversal-neighborhood-method-comparison]] records that old and current
  stacks share the same traversal skeleton; the real difference is the
  sibling-gate and calibration layer.
- [[selected-neighborhood-signal-flow-literature-20260617]] records the
  related selective-inference, diffusion-map, tree-wavelet, and treelet
  literature. It supports neighborhood smoothing as a multiscale support
  operator, not as an unconditional split rescue.
- [[graph-neural-geometry-spectral-artifact-literature-20260617]] records the
  graph signal processing, graph neural network, and graph geometry analogy.
  It identifies internal barycenters as graph low-pass filters, explains why
  smoothing can help and oversmooth simultaneously, and points to
  connection-Laplacian angle transport plus curvature/topology conditioning as
  the right refinement.
- [[selected-neighborhood-pvalue-interpolation-comparison-20260616]] records
  the new region-level bandwidth and `tau_s` interval audit. The original
  expanded distribution had no joined old bandwidth rows; the joined
  topology-bandwidth rerun produced only `48/139,860` old-and-current rows.
  On the joined comparison, `116/140` role-regions still lack finite `tau_b`,
  `24/140` have sparse finite `tau_b`, and the method-level `tau_s` interval
  remains empty because selected-null rows reopen first. This supports
  estimating a region-specific bandwidth set while keeping it diagnostic and
  fail-closed.
- The same p-value interpolation source now records the branch-length distance
  rerun. Branch lengths compress median nearest-support distance from `1.0` hop
  to `0.03` branch-length units and median best-case required `tau_s` from
  `150.826030` to `2.838978`, but the admissible interval remains empty:
  signal needs `0.048578` while selected-null can reopen at `0.003706` under the
  `25%` signal, `5%` null-leak target. On `overlap_extreme_4c`, branch-length
  interpolation catches `426` signal positives but also opens `939`
  selected-null positives.
- [[selected-neighborhood-conditional-support-panel-20260617]] records the
  first root-conditioned support join over the expanded overlap candidates.
  With strict root validity, topology, local support, and spectral-flow checks,
  the run reports `0` conditional neighborhood support passes, `0`
  selected-null neighborhood leaks, and `0` hard-negative leaks. This keeps
  `overlap_extreme_4c` blocked and localizes the current gap to topology/root
  context rather than raw bandwidth availability.
- [[selected-neighborhood-internal-spectral-flow-panel-20260617]] records that
  internal barycenters do recover many MP-supported spectral-flow edges, but
  the gain is mirrored on selected-null rows: `710` selected-null support
  creations versus `702` signal support creations in the three-case overlap
  run. The angular/radial node-pairwise layer shows mostly internal-only spike
  creation and stable shared objects, with no whole-object rotations observed
  and only rare single-mode rotations. This supports internal distributions as
  a tree-filter diagnostic, not as an unconditional rescue rule.
- The same panel now includes graph Dirichlet-style neighborhood energy over
  selected parent-child edges. Strict shared MP transport improves in only one
  of three overlap cases and degrades in two, while internal-only support
  remains large and mirrored on selected-null rows. This makes angle/radius
  energy a bottleneck localizer and coherence diagnostic, not a split rescue.
- [[selected-neighborhood-internal-spectral-flow-conditional-energy-20260617]]
  records the larger seven-case overlap run and the root-conditioned
  postprocess. Internal MP support reacts strongly, but `0/7` signal rows are
  admissible rescue candidates after root validity, root-tail support, strict
  shared energy improvement, hard-negative blocking, and paired selected-null
  checks. This does not contradict the empirical fact that older bandwidth and
  internal-node mechanisms helped some clustering rows; it says those
  mechanisms are not yet valid unconditional rescue evidence.
- [[old-vs-current-method-stack-comparison-20260615]] records the old
  `tau_b`, `tau_t`, `tau_s`, and `h_k` bandwidth layer and why it cannot be
  restored as a permissive prior update.
- [[sibling-null-prior-interpolation-audit-20260604]] records that the old
  interpolated sibling-null priors borrowed selected non-null rows and were
  descriptive, not strict production calibration.
- [[specific-small-method-benchmark-20260615]] records that the bandwidth
  component is useful local context but not a standalone separator; coherent
  outgoing topology and `balance_product` carry the recovery signal.
- [[selected-neighborhood-distribution-panel-20260615]] records that signal
  pass-through rows lack joined old/current neighborhood evidence, turning the
  issue into an explicit coverage and selected-neighborhood law problem.
- [[overlap-conditional-topology-law-panel-20260615]] records the cached,
  support-gated reintroduction of topology-neighborhood bandwidth fields.
- [[retained-pass-through-topology-likelihood-panel-20260615]] records that the
  retained pass-through likelihood is not identifiable on the compact run
  because finite topology evidence is missing on the matched signal/control
  rows.
- [[selected-neighborhood-spectral-flow-diagnostic-20260616]] records that
  MP-supported eigenspace flow has weak signal-vs-selected-null separation and
  many floor-only edges, supporting spectral flow as a bottleneck localizer
  rather than a standalone split rule.
- [[spectral-transport-passthrough-guard-20260616]] records the first
  traversal integration of that bottleneck: the spectral term is fail-closed
  and applies only to pass-through support, not sibling split creation.

## Links

- [[traversal-neighborhood-method-comparison]]
- [[selected-neighborhood-signal-flow-literature-20260617]]
- [[graph-neural-geometry-spectral-artifact-literature-20260617]]
- [[selected-neighborhood-conditional-support-panel-20260617]]
- [[selected-neighborhood-internal-spectral-flow-panel-20260617]]
- [[old-vs-current-method-stack-comparison-20260615]]
- [[specific-small-method-benchmark-20260615]]
- [[selected-neighborhood-distribution-panel-20260615]]
- [[overlap-conditional-topology-law-panel-20260615]]
- [[retained-pass-through-topology-likelihood-panel-20260615]]
- [[selected-neighborhood-spectral-flow-diagnostic-20260616]]
- [[spectral-transport-passthrough-guard-20260616]]

## Open Questions

- What minimum labeled support makes a selected-neighborhood recovery stratum
  identifiable without borrowing from selected non-null rows?
- Which bottleneck status should dominate when coverage exists but selected
  non-null exclusion removes nearly all local support?
- Can the bandwidth bottleneck statuses be validated on full Julia without
  making the diagnostic as expensive as the old uncached tree-distance loop?
- Can MP-supported spectral flow define a stable enough stratum to condition
  bandwidth interpolation, despite most overlap edges being floor-only?
- Can internal-barycenter spectral flow be conditioned by topology/root
  geometry strongly enough to keep its signal support gain while removing the
  mirrored selected-null support gain?
- Can the selected-neighborhood distance cache be upgraded from unit
  parent-link distance to branch-length distance without breaking the
  selected-row joins or making the benchmark prohibitively expensive?
  Answered operationally: yes. The open question is now whether branch-length
  neighborhoods can be conditioned by root validity, selected action geometry,
  and spectral-tail support strongly enough to reverse the observed
  selected-null-first ordering.
- What generator or analytic law can produce topology-coherent, strict
  spectral-flow-supported non-direct rows without reopening selected-null
  neighborhoods, given that the first root-conditioned support panel reports
  `0` conditional support passes?
