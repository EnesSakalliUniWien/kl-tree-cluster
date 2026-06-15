---
title: TooManyCells Method Notes 2026-06-13
captured: 2026-06-13
source_urls:
  - https://gregoryschwartz.github.io/too-many-cells/
  - https://github.com/GregorySchwartz/too-many-cells
  - https://www.nature.com/articles/s41592-020-0748-5
---

# TooManyCells Method Notes 2026-06-13

TooManyCells is a single-cell clade exploration toolkit built around a binary
hierarchical tree. The public documentation describes the method as recursively
dividing cells into clusters and relating clusters instead of placing each cell
in one low-dimensional layout.

Method points captured from the public documentation:

- `make-tree` builds a binary tree by hierarchical spectral clustering.
- The algorithm starts with all cells in one node, applies spectral clustering
  to partition the current node into two groups, and evaluates the candidate
  split with Newman-Girvan modularity.
- If modularity is positive, recursion continues; if modularity is non-positive,
  the node is treated as a leaf/final cluster.
- The default preprocessing outline includes matrix reading, optional cell and
  feature filtering, TF-IDF normalization, and optional LSA.
- The documented input model is an abundance matrix; the project was designed
  for single-cell RNA-seq but documents use with other abundance data.
- The paper and public project description emphasize a matrix-free divisive
  hierarchical spectral clustering implementation to avoid explicit full
  cell-cell similarity and Laplacian eigendecompositions at every node.

Relational notes for KL-TE:

- TooManyCells uses a graph/community objective as the split/stopping rule,
  while KL-TE currently uses a selected hierarchy plus statistical edge and
  sibling gates.
- TooManyCells does not appear to solve the KL-TE post-selection Wald null-law
  problem directly; it avoids that specific test by using modularity as the
  recursive stopping criterion.
- The relevant use for KL-TE is relational, not comparative: TooManyCells
  helps locate KL-TE among tree-first single-cell clustering methods and shows
  one way a recursive spectral tree can be stopped without a projected-Wald
  null law. It is not a direct comparator, benchmark target, or replacement
  proposal.
