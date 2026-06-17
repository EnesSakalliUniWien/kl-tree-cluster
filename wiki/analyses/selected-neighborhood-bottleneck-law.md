---
title: Selected Neighborhood Bottleneck Law
type: analysis
status: draft
updated: 2026-06-16
sources:
  - wiki/analyses/traversal-neighborhood-method-comparison.md
  - wiki/sources/old-vs-current-method-stack-comparison-20260615.md
  - wiki/sources/sibling-null-prior-interpolation-audit-20260604.md
  - wiki/sources/specific-small-method-benchmark-20260615.md
  - wiki/sources/selected-neighborhood-distribution-panel-20260615.md
  - wiki/sources/overlap-conditional-topology-law-panel-20260615.md
  - wiki/sources/retained-pass-through-topology-likelihood-panel-20260615.md
  - wiki/sources/selected-neighborhood-spectral-flow-diagnostic-20260616.md
  - wiki/sources/spectral-transport-passthrough-guard-20260616.md
  - kl_clustering_analysis/hierarchy_analysis/decomposition/gates/gate_evaluator.py
  - kl_clustering_analysis/hierarchy_analysis/decomposition/gates/orchestrator.py
  - kl_clustering_analysis/hierarchy_analysis/statistics/sibling_divergence/inflation_correction/empirical_null_inflation_estimation.py
  - benchmarks/diagnostics/calibration/overlap_conditional_topology_law_panel.py
  - benchmarks/diagnostics/calibration/selected_neighborhood_distribution_panel.py
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
