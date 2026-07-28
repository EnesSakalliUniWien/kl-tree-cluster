---
title: Selected Root Selected Family Traversal Literature 2026-06-14
type: source
status: reviewed
updated: 2026-07-28
sources:
  - raw/inbox/selected-root-selected-family-traversal-literature-20260614.md
tags:
  - source
  - literature
  - selection
  - traversal
  - calibration
---

# Selected Root Selected Family Traversal Literature 2026-06-14

## Summary

There is related literature, but no exact off-the-shelf selected-root plus
selected-family pass-through null law for Tree-Break Selection. The closest pieces are
selective inference after hierarchical clustering, selective inference on
selected families of hypotheses, hierarchical FDR/FWER testing, TreeScan or
scan-statistic max tests over many overlapping tree regions, and
cluster-based permutation tests. Together they support the current direction:
the pass-through correction should calibrate the whole searched descendant
family, not the one descendant node after it has already been selected.

## Key Points

- Selective inference for hierarchical clustering validates the core warning:
  a classical Wald or mean-difference test is invalid when the tested clusters
  are selected by clustering. The valid object conditions on the clustering
  event.
- Selective inference on multiple families of hypotheses matches the
  "selected-family" part: selecting promising families and then testing inside
  them needs a different error-control target than testing each family
  separately or pooling all hypotheses globally.
- Hierarchical FDR and hierarchical FWER methods provide useful traversal
  discipline for tree-structured hypotheses, especially root-down procedures.
  Their usual contract is weaker for Tree-Break Selection because the tree of hypotheses is
  often assumed predeclared, while Tree-Break Selection rebuilds selected topology from the
  same data.
- TreeScan and spatial/tree scan statistics are the closest operational shape
  to the Tree-Break Selection pass-through law: scan many overlapping branches or cells,
  compute a maximum statistic, and use Monte Carlo calibration to account for
  the search over the whole family.
- Cluster-based permutation testing is an adjacent max-statistic method. It
  creates connected suprathreshold clusters, scores them, and uses the maximum
  cluster statistic under permutations. It supports whole-frontier calibration
  and warns against overinterpreting localization after a global family test.
- For Tree-Break Selection, the selected-root law and selected-family pass-through law should
  be kept separate. Root-selected inference asks whether the first split opens;
  selected-family traversal inference asks whether any split reachable below a
  closed root is stronger than expected under the same whole-tree search.

## Evidence

- `raw/inbox/selected-root-selected-family-traversal-literature-20260614.md`
  records the checked sources and project mapping.
- `benchmarks/diagnostics/calibration/sibling/gates/fixed_sibling_gate_profile_validation.py`
  implements the current selected-root and selected-family permutation
  diagnostics.
- `tree_break_selection/hierarchy_analysis/decomposition/gates/orchestrator.py`
  exposes `global_sibling_min_passthrough_descendant_refined`, the current
  TreeScan/maxT-like diagnostic candidate for pass-through descendants.

## Links

- [[root-selection-literature-20260614]]
- [[fixed-sibling-gate-profile-validation-20260613]]
- [[selected-hierarchy-selection-geometry]]
- [[open-mathematical-questions]]
