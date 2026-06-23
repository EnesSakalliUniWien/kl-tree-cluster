---
title: TreeDecomposition
type: entity
status: reviewed
updated: 2026-06-23
sources:
  - README.md
  - tree_break_selection/hierarchy_analysis/tree_decomposition.py
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

The returned result includes cluster assignments, traversal traces, the number
of clusters, and basic independence-analysis metadata. Cluster assignments keep
the boundary `root_node` for tree-local provenance and a sorted
`leaf_signature` tuple for stable set-based cluster identity. Traversal rows
expose the tested edge tuples `(parent, child)` and sibling tuple
`(parent, left_child, right_child)`, plus raw/corrected edge and sibling
p-value diagnostics. This lets readers distinguish child-keyed edge evidence,
parent-keyed sibling evidence, and sibling rows that were not tested because
traversal prerequisites failed. Conceptually, it is the implementation form of
[[top-down-traversal]].

Sibling annotations now separate the active gate p-value from auxiliary
interpretation channels. The active gate reports its calibration role, while
fixed-subspace sparse evidence (`fixed_coordinate_bh` or `fixed_block_bh`) and
fixed-subspace dense evidence (`fixed_global_chi_square`) are carried as
separate columns. This keeps child-parent edge p-values as traversal and
reachability evidence and prevents selected projected edge or sibling energy
from being silently reused as sibling-scale production evidence.

Pass-through traversal is audited as its own decision layer. Trace rows expose
whether pass-through was enabled, whether split prerequisites were open, whether
the sibling gate already opened a normal split, whether a descendant split was
available, whether the node was a pass-through candidate, and whether support
guarding blocked the candidate. These fields keep pass-through from being
inferred only from the final traversal action. For pass-through-scoped
selected-family profiles, a selected-family permutation guard that would block
the candidate now closes pass-through support even when an earlier root or
sibling guard has already closed the normal sibling split flag.

## Evidence

- `tree_break_selection/hierarchy_analysis/tree_decomposition.py` contains
  initialization, annotation preparation, bundle reuse checks, and traversal.
- `tree_break_selection/hierarchy_analysis/cluster_assignments.py` carries
  sorted leaf signatures from boundary metadata into sample assignment tables.
- `manuscript/sections/method/decomposition.tex` documents the extraction of
  final clusters from test annotations.
- `README.md` describes decomposition as the step that turns statistical gates
  into stable cluster assignments.

## Links

- [[poset-tree]]
- [[top-down-traversal]]
- [[tree-break-selection]]
- [[projected-wald-statistic]]

## Open Questions

- Should gate annotation bundle metadata get its own entity page if cache reuse
  becomes a frequent maintenance topic?
