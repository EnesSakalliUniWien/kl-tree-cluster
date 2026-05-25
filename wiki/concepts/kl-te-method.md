---
title: KL-TE Method
type: concept
status: reviewed
updated: 2026-05-24
sources:
  - README.md
  - manuscript/guides/full_method_logic_map.md
  - manuscript/sections/method/overview.tex
tags:
  - method
  - kl-te
---

# KL-TE Method

## Summary

KL-TE evaluates a candidate binary hierarchy by estimating subtree feature
distributions, testing child-parent and sibling contrasts with projected-Wald
statistics, correcting p-values, and traversing the tree to decide the final
cluster partition.

## Details

The hierarchy is treated as a list of proposed groups and splits, not as proof
that each branch is a valid cluster. Each internal node supplies a parent
subtree and two child subtrees. The method estimates feature rates for those
subtrees, evaluates local evidence, and then uses [[top-down-traversal]] to
return terminal clusters.

The child-parent edge test asks whether a child differs from the parent
background. The sibling test asks whether two children under the same parent
are distinct enough to keep as separate clusters. Both stages use the
[[projected-wald-statistic]] in a local spectral subspace.

## Evidence

- `README.md` describes the package workflow and statistical gates.
- `manuscript/guides/full_method_logic_map.md` records the estimator and
  statistic chain.
- `manuscript/sections/method/overview.tex` states the inferential objective.

## Links

- [[project-overview]]
- [[projected-wald-statistic]]
- [[top-down-traversal]]
- [[poset-tree]]
- [[tree-decomposition]]

## Open Questions

- How should the manuscript state the assumptions behind data-dependent local
  projection?
- Which simulation outputs should support the current edge and sibling
  calibration claims?
