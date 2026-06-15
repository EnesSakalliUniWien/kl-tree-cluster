---
title: Overlap Structural Sibling Panel 2026-06-14
type: source
status: reviewed
updated: 2026-06-14
sources:
  - benchmarks/diagnostics/calibration/overlap_structural_sibling_panel.py
  - tests/validation/103_test_overlap_structural_sibling_panel.py
  - raw/assets/benchmark-results/overlap_structural_sibling_20260614/overlap_structural_sibling_rows.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_20260614/overlap_structural_sibling_summary.csv
tags:
  - source
  - diagnostics
  - overlap
  - traversal
  - barycentric
---

# Overlap Structural Sibling Panel 2026-06-14

## Summary

`overlap_structural_sibling_panel.py` reconstructs selected binary overlap
cases as individual analytical cases and diagnoses whether each relevant
traversal sibling decision is supported by the same structural coordinates as
within-node homogeneity gain. It is diagnostic-only and does not change
production traversal or calibration.

## Key Points

- The panel focuses on the refined-profile overlap failures identified by
  [[selected-family-traversal-panel-20260614]]:
  `overlap_extreme_4c`, `overlap_mod_4c_small`, `overlap_mod_8c_large`, and
  `overlap_unbal_4c_small`.
- For each visited binary internal node, the panel compares top-coordinate
  support for the sibling contrast, the parent-child edge directions, and the
  variance-drop homogeneity focus.
- The panel now reports explicit structure-versus-homogeneity support overlap:
  `left_edge_homogeneity_jaccard_topk`,
  `right_edge_homogeneity_jaccard_topk`,
  `max_edge_homogeneity_jaccard_topk`, and
  `subspace_consensus_jaccard_topk`, the last being the minimum agreement
  across sibling-homogeneity, sibling-edge, and edge-homogeneity support.
- The panel also separates heterogeneous structural change from unrelated
  subspace signal. It reports heterogeneity-focused support fields
  (`delta_heterogeneity_jaccard_topk`,
  `max_edge_heterogeneity_jaccard_topk`, and
  `heterogeneity_subspace_consensus_jaccard_topk`), pairwise heterogeneity
  gains, and `structural_change_mode`.
- It also reports pairwise binary Jaccard homogeneity before and after a split,
  barycentric balance/leverage, truth alignment for signal rows, and a
  diagnostic `structural_sibling_status`.
- In the one-replicate null/signal overlap run, all `38` selected-null
  analytical rows are classified as `weak_homogeneity_gain`. The accepted null
  splits have strong sibling p-values, but the minimum within-child Jaccard
  gain over the parent is near zero or negative.
- The sibling contrast and child-parent edge support have
  `max_delta_edge_jaccard_topk = 1.0` for accepted split rows. In these binary
  cases, the explicit edge-homogeneity overlap equals the
  sibling-homogeneity overlap because the barycentric edge and sibling
  supports coincide.
- The observed failure is therefore not a different edge subspace in these
  overlap null examples. It is selected barycentric contrast without a
  corresponding structural homogeneity gain.
- Signal rows separate into two behaviors: some internal splits are
  `structural_same_subspace_supported` and truth-aligned, while several weak
  overlap splits remain `weak_homogeneity_gain` and truth-misaligned.

## Evidence

- `tests/validation/103_test_overlap_structural_sibling_panel.py` verifies the
  top-coordinate, Jaccard, cosine, pairwise homogeneity, status-classification,
  and tiny output contracts.
- The focused run wrote `68` analytical rows and summary rows under
  `raw/assets/benchmark-results/overlap_structural_sibling_20260614/`.
- The regenerated run keeps all `38` selected-null rows in
  `weak_homogeneity_gain`; no row in this run moves to
  `structural_homogeneity_subspace_mismatch`, but the status is now part of
  the diagnostic contract for future cases.
- A second regeneration added heterogeneous-change modes. The same four-case
  run has no rows with `heterogeneity_gain_max >= 0.02`; selected-null rows
  remain `weak_or_mixed_structural_change`, while only five signal rows are
  `homogeneous_same_subspace`. This means the tested overlap failures are not
  strong same-subspace heterogeneity events either.
- Verification passed:
  `pytest tests/validation/103_test_overlap_structural_sibling_panel.py -q`
  and
  `ruff check benchmarks/diagnostics/calibration/overlap_structural_sibling_panel.py tests/validation/103_test_overlap_structural_sibling_panel.py`.

## Links

- [[selected-family-traversal-panel-20260614]]
- [[path-conditioned-barycentric-action-diagnostics-20260606]]
- [[open-mathematical-questions]]
