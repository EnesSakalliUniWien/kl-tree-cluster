---
title: Top-Down Traversal
type: concept
status: reviewed
updated: 2026-05-24
sources:
  - manuscript/sections/method/decomposition.tex
  - kl_clustering_analysis/hierarchy_analysis/tree_decomposition.py
tags:
  - method
  - traversal
---

# Top-Down Traversal

## Summary

Top-down traversal converts precomputed edge and sibling annotations into the
final cluster partition. It does not estimate a new model or compute new
p-values.

## Details

Traversal begins at the root. If a node has an eligible binary split, enough
edge evidence, and a significant sibling decision, traversal continues into
both children. Otherwise the node is ordinarily terminal. The manuscript and
implementation also describe a pass-through behavior that can descend through a
shallow sibling failure when deeper significant sibling structure exists.

The implementation is iterative rather than recursive. It uses a stack,
consults gate decisions, and records terminal `ClusterBoundary` values before
building per-sample cluster assignments.

## Evidence

- `manuscript/sections/method/decomposition.tex` states the decision-extraction
  role and pass-through rule.
- `kl_clustering_analysis/hierarchy_analysis/tree_decomposition.py` implements
  the traversal loop in `TreeDecomposition.decompose_tree`.

## Links

- [[kl-te-method]]
- [[tree-decomposition]]
- [[poset-tree]]

## Open Questions

- Which validation page should record the empirical effect of pass-through
  traversal on shallow sibling-test failures?
