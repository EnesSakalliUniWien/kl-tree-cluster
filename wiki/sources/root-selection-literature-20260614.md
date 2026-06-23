---
title: Root Selection Literature 2026-06-14
type: source
status: reviewed
updated: 2026-06-14
sources:
  - raw/inbox/root-selection-literature-20260614.md
tags:
  - source
  - literature
  - root
  - selection
  - clustering
---

# Root Selection Literature 2026-06-14

## Summary

The literature supports a strict separation between root orientation and root
split inference. In ordinary hierarchical clustering, the root is the
all-sample cluster and the hard question is whether the first selected split is
valid under a selected null. Phylogenetic and trajectory roots are different
objects: they require external direction, time, ancestor, outgroup, clock, or
lineage information. For Tree-Break Selection, the current production blocker is therefore
not "find a better root" as an abstract high-dimensional point; it is validate
or replace the selected root split/stopping law.

## Key Points

- Hierarchical clustering conventionally roots the dendrogram at the unique
  cluster containing all samples. Tree-Break Selection's root is therefore a whole-sample
  parent whose children are the selected first split.
- Selective inference for hierarchical clustering directly supports the Tree-Break Selection
  diagnosis: testing selected clusters with a classical mean-difference/Wald
  law is anti-conservative unless the selection event is accounted for.
- Pvclust and Shimodaira-style multiscale bootstrap give cluster uncertainty
  and selected-region geometry tools, including sample-size behavior and
  curvature ideas. They are useful diagnostics but not an automatic production
  calibration rule for Tree-Break Selection.
- SHC/SigClust is the nearest root-down testing family: it tests nodes starting
  at the root under a fitted single-Gaussian null and supplies an FWER
  procedure. Its inferential object differs from Tree-Break Selection's Bernoulli/categorical
  selected-topology null.
- Gap statistic, silhouette-like criteria, PDDP, bisecting k-means, and graph
  modularity objectives are tree-construction or stopping tools. They can
  improve the top-level split but do not remove post-selection bias from a
  subsequently tested root.
- TooManyCells is relationally important: it starts at the all-cell root and
  stops recursive spectral splits using Newman-Girvan modularity rather than a
  selected projected-Wald p-value. It is not a direct Tree-Break Selection comparator.
- Phylogenetic rooting and pseudotime root selection are orientation problems.
  They are relevant only if Tree-Break Selection adds external temporal, ancestor, outgroup,
  or lineage assumptions.
- High-dimensional root splitting is hard because distances inflate,
  eigen-directions can be unstable, and many weak feature effects can dominate
  selected topology. This supports root-stability and selected-family replay as
  diagnostics rather than blind root optimization.

## Evidence

- `raw/inbox/root-selection-literature-20260614.md` records the checked
  external sources and project-specific synthesis.
- The local root-selected-region diagnostic already found that root child
  construction margins are not the dominant coordinate for root selected ratios;
  edge-opening/path action is more informative in the small continuous panel.
- The selected-null profile validation found that null false splits are not
  caused by guarded roots opening under null, but by pass-through descendants
  below closed roots in binary overlap templates.

## Links

- [[root-selected-region-margins-20260603]]
- [[toomanycells-method-20260613]]
- [[selected-hierarchy-selection-geometry]]
- [[open-mathematical-questions]]
