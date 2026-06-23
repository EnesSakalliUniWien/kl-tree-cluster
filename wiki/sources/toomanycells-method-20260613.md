---
title: TooManyCells Method 2026-06-13
type: source
status: reviewed
updated: 2026-06-13
sources:
  - raw/inbox/toomanycells-method-notes-20260613.md
tags:
  - source
  - clustering
  - diagnostics
  - relational
---

# TooManyCells Method 2026-06-13

## Summary

TooManyCells is a relational reference point for Tree-Break Selection, not a direct
comparator. It is a single-cell clade analysis method centered on divisive
hierarchical spectral clustering: it recursively partitions cells into a binary
tree and stops recursion using a Newman-Girvan modularity criterion rather than
a post-selection projected-Wald test.

## Key Points

- The public documentation describes `make-tree` as starting with all cells in
  one node, applying spectral clustering to split a node into two groups, and
  continuing recursively only when Newman-Girvan modularity is positive.
- The method is explicitly tree-first and multiresolution: the output tree is a
  primary object for clade visualization, diversity, paths, differential
  analysis, and downstream summaries.
- The documented preprocessing path is abundance-matrix based, with optional
  cell/feature filtering, TF-IDF normalization, and optional LSA.
- The paper/project description emphasizes matrix-free hierarchical spectral
  clustering, avoiding explicit full cell-cell similarity and normalized
  Laplacian eigendecompositions at each candidate split.
- For Tree-Break Selection, the useful role is relational. TooManyCells helps position Tree-Break Selection
  among tree-first single-cell clustering methods and shows a different
  recursive spectral-tree stopping philosophy.
- TooManyCells is not a direct benchmark comparator, not a calibration proof,
  and not a replacement proposal for the current Tree-Break Selection sibling gate. It does
  not validate the current Tree-Break Selection Wald p-values.

## Evidence

- `raw/inbox/toomanycells-method-notes-20260613.md` records the external
  public documentation and publication links consulted on 2026-06-13.
- The relational check follows the null-law decomposition result: the Tree-Break Selection
  fixed chi-square reference is valid only for a fixed projection, while
  same-sample projection adaptation breaks the sibling null law.

## Links

- [[projected-wald-statistic]]
- [[null-law-decomposition-panel-20260613]]
- [[open-mathematical-questions]]
