---
title: Phylogenetic Tree Builders 2026-06-14
type: source
status: reviewed
updated: 2026-06-14
sources:
  - tree_break_selection/tree/phylogenetic.py
  - benchmarks/shared/runners/tbs_runner.py
  - benchmarks/shared/runners/method_registry.py
  - tests/core/test_phylogenetic_tree_builders.py
tags:
  - source
  - tree
  - phylogenetic
  - rooting
---

# Phylogenetic Tree Builders 2026-06-14

## Summary

The TBS runner now has opt-in tree estimators for neighbor joining and IQ-TREE 3.
Both produce unrooted metric trees first and then orient them with minimum
ancestor deviation before constructing a `PosetTree`. This changes tree
estimation and root orientation, but it does not solve the selected-root or
selected-family null-law problem.

## Key Points

- `tbs_neighbor_joining` builds a neighbor-joining tree from the configured TBS
  tree distance vector and roots the resulting unrooted metric tree with MAD.
- `tbs_iqtree3` writes the feature matrix as a binary/morphological-state
  alignment, calls an external `iqtree3` executable with the default `JC2`
  binary model, parses
  the `.treefile` Newick output, and then applies the same MAD rooting layer.
- The MAD objective minimizes average relative pairwise ancestor deviation,
  using terms of the form
  `((d(x, rho) - d(y, rho)) / d(x, y)) ** 2` over leaf pairs.
- The default benchmark method set is unchanged. IQ-TREE remains optional
  because it requires an external binary and because likelihood-tree feature
  encoding is a modeling choice, not a default Tree-Break Selection contract.
- MAD rooting supplies a deterministic orientation for phylogenetic candidate
  trees. It is not a p-value calibration rule and does not condition on the
  same-data selection event that creates the root and descendant traversal
  hypotheses.

## Evidence

- `tree_break_selection/tree/phylogenetic.py` implements
  `minimum_ancestor_deviation_root`, neighbor-joining construction, IQ-TREE 3
  execution/parsing, and promotion to `PosetTree`.
- `benchmarks/shared/runners/method_registry.py` registers
  `tbs_neighbor_joining` and `tbs_iqtree3` as explicit opt-in methods.
- `benchmarks/shared/runners/tbs_runner.py` keeps current linkage behavior for
  `tbs` while routing the new methods through the shared `PosetTree` and gate
  pipeline.
- `tests/core/test_phylogenetic_tree_builders.py` checks exact MAD midpoint
  rooting, NJ conversion, and IQ-TREE command/Newick integration with a
  monkeypatched process call.

## Links

- [[root-selection-literature-20260614]]
- [[selected-root-selected-family-traversal-literature-20260614]]
- [[benchmark-pipeline-contract]]
- [[open-mathematical-questions]]
