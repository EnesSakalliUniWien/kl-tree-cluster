---
title: Graph Neural Geometry Spectral Artifact Literature 2026-06-17
type: source
status: reviewed
updated: 2026-06-17
sources:
  - raw/inbox/graph-neural-geometry-spectral-artifact-literature-20260617.md
tags:
  - source
  - literature
  - graph
  - spectral
  - geometry
  - neural-networks
---

# Graph Neural Geometry Spectral Artifact Literature 2026-06-17

## Summary

This literature capture connects Tree-Break Selection's internal-barycenter and
selected-neighborhood spectral problem to graph signal processing, graph neural
network smoothing theory, and graph geometry. The central message is
consistent across the papers: graph smoothing and low-pass filtering can
increase coherent-looking spectral support, but this is not sufficient for a
valid decision because smoothing can also create selected-null artifacts,
oversmooth discriminative structure, or distort flow through bottlenecks.

For Tree-Break Selection, internal distributions should therefore be treated as a graph
low-pass/tree-filter diagnostic. They need angle/radius persistence and
selected-topology conditioning before they can support a decision.

## Key Points

- Graph signal processing defines graph Fourier modes, graph filters, and
  localized multiscale transforms. This is the right mathematical frame for
  Tree-Break Selection's internal-barycenter operator: it is a graph/tree low-pass filter, not
  an independent-sample generator.
- Spectral graph wavelets, diffusion wavelets, and graph scattering support
  multiscale persistence diagnostics. A real coarse mode should persist across
  graph scales; a local fragment should be high-frequency or unstable.
- GCN theory repeatedly identifies graph convolution as Laplacian smoothing or
  low-pass filtering. This explains why internal barycenters can help:
  smoothing denoises and raises coarse signal power.
- The oversmoothing literature gives the warning. Repeated smoothing can drive
  embeddings toward indistinguishable low-energy states, so smoothness alone
  is not discriminative evidence.
- The Tree-Break Selection diagnostic result mirrors that warning: internal barycenters create
  many new MP-supported edges on both selected-null and signal rows.
- Oversquashing/curvature papers identify graph bottlenecks as places where
  information flow is distorted. For Tree-Break Selection, root validity, selected tie-rank,
  edge action, child balance, and topology replay are the corresponding
  conditioning coordinates.
- Vector diffusion maps and the connection Laplacian are the closest analogy
  for the angle-specific transport panel. They use local vector-frame
  transport and connection consistency; Tree-Break Selection's version compares principal
  angles/projectors and log-eigenvalue radii across selected-tree spectral
  objects.

## Evidence

- `raw/inbox/graph-neural-geometry-spectral-artifact-literature-20260617.md`
  records the source list and Tree-Break Selection relationship.
- [Shuman et al., 2013](https://arxiv.org/abs/1211.0053) frame graph signals,
  graph spectral domains, graph filtering, and localized multiscale graph
  transforms.
- [Hammond, Vandergheynst, and Gribonval, 2011](https://arxiv.org/abs/0912.3848)
  construct spectral graph wavelets from kernels of the graph Laplacian.
- [Coifman and Maggioni, 2006/2008](https://perception.inrialpes.fr/~Horaud/Courses/www-documents/diffusion-wavelets.pdf)
  use diffusion as a smoothing and scaling tool for graph/manifold
  multiresolution analysis.
- [Zou and Lerman, 2020](https://arxiv.org/abs/1804.00099) connect graph
  scattering to permutation-invariant and graph-stable features.
- [Li, Han, and Wu, 2018](https://arxiv.org/abs/1801.07606) interpret GCN graph
  convolution as Laplacian smoothing and warn about oversmoothing.
- [NT and Maehara, 2019](https://arxiv.org/abs/1905.09550) analyze GNNs as
  low-pass graph filters.
- [Oono and Suzuki, 2020](https://openreview.net/forum?id=S1ldO2EFPr) relate
  deep GNN asymptotics to graph spectra and exponential loss of expressive
  power.
- [Cai and Wang, 2020](https://arxiv.org/abs/2006.13318) use Dirichlet energy
  to describe oversmoothing and loss of discriminative power.
- [Zhao and Akoglu, 2020](https://openreview.net/forum?id=rkecl1rtwB) propose
  PairNorm to counteract embeddings becoming too similar under graph
  convolution.
- [Topping et al., 2022](https://arxiv.org/abs/2111.14522) connect
  oversquashing to graph bottlenecks and edge-based curvature.
- [Chamberlain et al., 2021](https://arxiv.org/abs/2110.09443) formulate GNNs
  as neural diffusion and Beltrami flow on graphs.
- [Singer and Wu, 2012](https://arxiv.org/abs/1102.0075) develop vector
  diffusion maps and the connection Laplacian for transporting local vector
  data.
- [Ollivier, 2009](https://arxiv.org/abs/math/0701886) defines Ricci curvature
  for Markov chains as a contraction property of random-walk measures.

## Links

- [[selected-neighborhood-bottleneck-law]]
- [[selected-neighborhood-internal-spectral-flow-panel-20260617]]
- [[selected-neighborhood-spectral-flow-diagnostic-20260616]]
- [[root-conditional-kernel-spectral-law]]
