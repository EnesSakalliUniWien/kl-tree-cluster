---
title: PosetTree
type: entity
status: reviewed
updated: 2026-07-28
sources:
  - tree_break_selection/tree/README.md
  - tree_break_selection/tree/poset_tree.py
  - tree_break_selection/tree/construction/build.py
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
subclass. Construction is separate: the explicit construction interface builds
linkage, neighbor-joining, or IQ-TREE topologies and returns a `PosetTree` plus
construction evidence. `PosetTree` itself owns rooted hierarchy operations,
distribution population, leaf counts, and decomposition access; it no longer
contains pass-through constructor aliases.

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
