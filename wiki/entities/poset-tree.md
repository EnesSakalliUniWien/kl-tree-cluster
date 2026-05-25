---
title: PosetTree
type: entity
status: reviewed
updated: 2026-05-24
sources:
  - kl_clustering_analysis/tree/README.md
  - kl_clustering_analysis/tree/poset_tree.py
tags:
  - entity
  - tree
---

# PosetTree

## Summary

`PosetTree` is the central directed tree structure for the KL-TE pipeline. It
stores parent-child topology, node metadata, descendant sets, distributions,
and helpers for decomposition and sample cluster assignments.

## Details

The tree package README describes `PosetTree` as a NetworkX `DiGraph`
subclass. It can be built from SciPy linkage output, sklearn agglomerative
fits, or oriented undirected edges. Its distribution population path computes
leaf and internal node distributions, records leaf counts, and exposes data to
the decomposition stage.

`PosetTree` is the implementation entity that carries candidate hierarchy
structure into [[tree-decomposition]].

## Evidence

- `kl_clustering_analysis/tree/README.md` maps the public methods and related
  topology and distribution helpers.
- `kl_clustering_analysis/tree/poset_tree.py` contains the class
  implementation.

## Links

- [[kl-te-method]]
- [[tree-decomposition]]
- [[top-down-traversal]]

## Open Questions

- Should wiki coverage split distribution population and topology helpers into
  separate entity pages as the wiki grows?
