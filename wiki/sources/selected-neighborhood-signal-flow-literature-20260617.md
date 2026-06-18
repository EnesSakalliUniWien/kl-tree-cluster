---
title: Selected Neighborhood Signal Flow Literature 2026-06-17
type: source
status: reviewed
updated: 2026-06-17
sources:
  - raw/inbox/selected-neighborhood-signal-flow-literature-20260617.md
tags:
  - source
  - literature
  - neighborhood
  - selection
  - spectral
---

# Selected Neighborhood Signal Flow Literature 2026-06-17

## Summary

This literature capture connects KL-TE's selected-neighborhood development to
two nearby but distinct mathematical lines. Selective inference and multiscale
bootstrap support the fail-closed rule for algorithm-selected roots and
clusters. Diffusion maps, tree wavelets, and treelets support the use of local
kernel or tree-scale neighborhoods as multiscale support evidence.

The combined implication is narrow: neighborhood smoothing can localize
conditional support and coherent signal flow, but it should not be promoted to
an unconditional p-value rescue.

## Key Points

- Gao, Bien, and Witten show that hypotheses formed after hierarchical
  clustering need selective inference. This supports conditioning on the
  selected root or selected topology before interpreting a root or cluster
  tail.
- The Suzuki-Shimodaira pvclust and Terada-Shimodaira selected-region line
  treats selected hypotheses through multiscale geometry. Its scale coordinate
  is bootstrap sample size and region curvature; KL-TE's scale coordinates are
  selected tree topology, spectral state, and traversal neighborhood.
- Diffusion maps interpret a kernel graph through a Markov/diffusion operator.
  For KL-TE, this makes \(\tau\) a locality/diffusion scale and makes slow
  eigenvectors plausible coherent support coordinates, not automatic split
  evidence.
- Multiscale wavelets on trees and treelets show why tree coarsening can keep
  coherent low-frequency signal while attenuating high-frequency local
  artifacts. For KL-TE, this supports treating internal-node or barycentric
  distributions as tree filters.
- Internal-node information is dependent on the leaves. It can improve signal
  localization, but it must not be counted as extra independent rows in a
  Marchenko--Pastur or selected-tail calibration.
- The safe KL-TE object is therefore a support-gated conditional tail:
  \[
  \Pr_0\{S_{H_u}\ge s_u \mid
  R_{\mathrm{root}},G_u,T_u,A_u,E_u,B_u,H_u,\mathcal N_\tau(u)\}.
  \]
  If the support neighborhood has insufficient admissible null or external
  records, the decision remains fail-closed.

## Evidence

- `raw/inbox/selected-neighborhood-signal-flow-literature-20260617.md`
  records the source list and KL-TE relationship.
- [Gao, Bien, and Witten, 2024](https://arxiv.org/abs/2012.02936) motivate
  selective inference after hierarchical clustering.
- [Suzuki and Shimodaira, 2006](https://academic.oup.com/bioinformatics/article/22/12/1540/207339)
  provide the pvclust multiscale bootstrap uncertainty route for hierarchical
  clustering.
- [Terada and Shimodaira, 2019](https://www.frontiersin.org/journals/ecology-and-evolution/articles/10.3389/fevo.2019.00174/full)
  connect tree/edge hypotheses to selective inference.
- [Nadler, Lafon, Coifman, and Kevrekidis, 2005](https://arxiv.org/abs/math/0506090)
  interpret spectral clustering and diffusion maps through kernel-induced
  diffusion and Fokker-Planck eigenfunctions.
- [Gavish, Nadler, and Coifman, 2010](https://icml.cc/Conferences/2010/papers/137.pdf)
  construct multiscale wavelets on trees and graphs.
- [Lee, Nadler, and Wasserman, 2008](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-2/issue-2/TreeletsAn-adaptive-multi-scale-basis-for-sparse-unordered-data/10.1214/07-AOAS137.full)
  present treelets as an adaptive multiscale basis for unordered sparse data.

## Links

- [[root-conditional-kernel-spectral-law]]
- [[selected-neighborhood-bottleneck-law]]
- [[selected-neighborhood-measurability-law]]
- [[selected-hierarchy-geometric-law-map]]
