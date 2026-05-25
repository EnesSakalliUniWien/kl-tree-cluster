---
title: TreeDecomposition
type: entity
status: reviewed
updated: 2026-05-24
sources:
  - README.md
  - kl_clustering_analysis/hierarchy_analysis/tree_decomposition.py
  - manuscript/sections/method/decomposition.tex
tags:
  - entity
  - decomposition
---

# TreeDecomposition

## Summary

`TreeDecomposition` prepares gate annotations and walks a `PosetTree` to return
cluster assignments from child-parent and sibling evidence.

## Details

The class accepts a tree, annotations or a reusable gate annotation bundle,
edge and sibling alpha values, optional leaf data and feature-space metadata,
and a pass-through flag. It validates or builds the annotation table, extracts
required boolean decision columns, constructs a gate evaluator, and performs
the iterative traversal.

The returned result includes cluster assignments, the number of clusters, and
basic independence-analysis metadata. Conceptually, it is the implementation
form of [[top-down-traversal]].

## Evidence

- `kl_clustering_analysis/hierarchy_analysis/tree_decomposition.py` contains
  initialization, annotation preparation, bundle reuse checks, and traversal.
- `manuscript/sections/method/decomposition.tex` documents the extraction of
  final clusters from test annotations.
- `README.md` describes decomposition as the step that turns statistical gates
  into stable cluster assignments.

## Links

- [[poset-tree]]
- [[top-down-traversal]]
- [[kl-te-method]]
- [[projected-wald-statistic]]

## Open Questions

- Should gate annotation bundle metadata get its own entity page if cache reuse
  becomes a frequent maintenance topic?
