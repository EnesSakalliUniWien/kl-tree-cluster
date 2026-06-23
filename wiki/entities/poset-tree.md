---
title: PosetTree
type: entity
status: reviewed
updated: 2026-06-23
sources:
  - tree_break_selection/tree/README.md
  - tree_break_selection/tree/poset_tree.py
tags:
  - entity
  - tree
---

# PosetTree

## Summary

`PosetTree` is the central directed tree structure for the Tree-Break Selection pipeline. It
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

- `tree_break_selection/tree/README.md` maps the public methods and related
  topology and distribution helpers.
- `tree_break_selection/tree/poset_tree.py` contains the class
  implementation.

## Links

- [[tree-break-selection]]
- [[tree-decomposition]]
- [[top-down-traversal]]

## Open Questions

- Should wiki coverage split distribution population and topology helpers into
  separate entity pages as the wiki grows?
