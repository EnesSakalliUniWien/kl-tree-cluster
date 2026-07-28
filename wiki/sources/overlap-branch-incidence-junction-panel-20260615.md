---
title: Overlap Branch-Incidence Junction Panel 2026-06-15
type: source
status: reviewed
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/overlap/overlap_branch_incidence_junction_panel.py
  - tests/validation/calibration/overlap/127_test_overlap_branch_incidence_junction_panel.py
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/branch_incidence_junction_panel/overlap_branch_incidence_junction_rows.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/branch_incidence_junction_panel/overlap_branch_incidence_junction_summary.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/branch_incidence_junction_panel/manifest.json
tags:
  - source
  - diagnostics
  - overlap
  - traversal
  - bayesian
---

# Overlap Branch-Incidence Junction Panel 2026-06-15

## Summary

`overlap_branch_incidence_junction_panel.py` replaces the parent-context proxy
with actual branch-incidence vectors. For each non-root binary internal node it
computes the incoming edge, incoming selected-family contrast, and outgoing
child-sibling contrast from descendant means, then measures compatibility with
several lenses: top-coordinate overlap, raw cosine, centered cosine, diagonal
Fisher-weighted cosine, and Fisher-weighted top-coordinate overlap.

## Key Points

- The runner writes `overlap_branch_incidence_junction_rows.csv`,
  `overlap_branch_incidence_junction_summary.csv`, and `manifest.json`.
- The row schema now also carries existing child-parent edge-test evidence:
  raw/BH p-values, tested/rejected flags, and outgoing edge summaries for the
  incoming node, incoming sibling, and both outgoing children.
- In the focused overlap rerun, the panel emits `148` branch-incidence rows and
  annotates `32` transfer-gap rows. One earlier transfer-gap row is
  root-or-unobserved-income and is therefore outside the non-root branch
  incidence panel.
- Among the annotated rows there are `5` truth-recovery rows and `27`
  negative rows.
- The panel finds `0/5` truth-recovery branch-transition candidates and
  `5/5` truth-recovery coordinate branch-incidence mismatches.
- The median truth-recovery incoming-family/outgoing top-k Jaccard is `0.0`;
  the negative median is also `0.0`.
- The median truth-recovery metric-family alignment score is `0.029662`;
  the negative median is `0.043478`.
- The previously missed truth row, `overlap_unbal_4c_small` replicate `1`
  node `N797`, has incoming-family/outgoing top-k Jaccard `0.0` and
  incoming-family/outgoing absolute cosine `0.112750`.
- The richer metric-family score does not rescue this row: its best
  compatibility score is still `0.112750`, classified as
  `metric_family_branch_mismatch`.
- The recovered truth rows are also metric-family branch mismatches under
  these actual incoming/outgoing vector metrics.
- The summary status is `coordinate_branch_incidence_truth_mismatch`.

## Method Implication

The missed row is not a clean weak-income to coherent-outcome transition under
the actual branch-incidence vectors. This conclusion is not based only on
Jaccard: raw, centered, and diagonal Fisher-weighted compatibility remain weak.
More importantly, the four rows recovered by the local internal-node
likelihood are also not explained by incoming branch compatibility. The overlap
recovery signal appears to be a local outgoing subspace phenomenon rather than
a continuation of the selected incoming branch.

Therefore the next Bayesian law should not require incoming/outgoing branch
alignment as a hard recovery condition. The correct object must separate at
least two modes:

- **continuation mode**: outgoing structure is compatible with incoming branch
  evidence;
- **emergent local-outcome mode**: outgoing structure is coherent but lives in
  a different subspace than the incoming branch.

The current focused overlap failures live in the second mode.

## Evidence

- `tests/validation/calibration/overlap/127_test_overlap_branch_incidence_junction_panel.py`
  verifies the branch-incidence geometry classifier and output writing.
- Verification passed:
  `pytest tests/validation/calibration/overlap/127_test_overlap_branch_incidence_junction_panel.py -q`
  and
  `ruff check benchmarks/diagnostics/calibration/overlap/overlap_branch_incidence_junction_panel.py tests/validation/calibration/overlap/127_test_overlap_branch_incidence_junction_panel.py`.

## Links

- [[overlap-income-outcome-junction-law-20260615]]
- [[overlap-internal-node-transfer-gap-audit-20260615]]
- [[overlap-context-negative-edge-conditioning-20260615]]
- [[overlap-context-negative-topology-conditioning-20260615]]
- [[open-mathematical-questions]]
