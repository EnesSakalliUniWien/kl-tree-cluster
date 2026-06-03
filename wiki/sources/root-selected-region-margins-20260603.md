---
title: Root Selected Region Margins 20260603
type: source
status: reviewed
updated: 2026-06-03
sources:
  - benchmarks/diagnostics/calibration/root_selected_region_margins.py
  - tests/validation/55_test_root_selected_region_margins.py
  - raw/assets/benchmark-results/root_selected_region_margins_20260603/manifest.json
  - raw/assets/benchmark-results/root_selected_region_margins_20260603/root_selected_region_summary.csv
  - raw/assets/benchmark-results/root_selected_region_margins_20260603/root_selected_region_merge_margins.csv
  - raw/assets/benchmark-results/root_selected_region_margins_20260603/root_selected_region_relationships.csv
  - raw/assets/benchmark-results/root_selected_region_margins_20260603_continuous/manifest.json
  - raw/assets/benchmark-results/root_selected_region_margins_20260603_continuous/root_selected_region_summary.csv
  - raw/assets/benchmark-results/root_selected_region_margins_20260603_continuous/root_selected_region_relationships.csv
tags:
  - source
  - selection
  - geometry
  - calibration
---

# Root Selected Region Margins 20260603

## Summary

`benchmarks/diagnostics/calibration/root_selected_region_margins.py` replays
the average-linkage hierarchy used by KL-TE and records the merge-selection
inequality margins that construct the two root child clusters. It joins those
margins to observed root edge, sibling, and spectral quantities. The diagnostic
is descriptive selected-region geometry, not an external calibration model.
Schema `v3` also records smooth first-order signed-distance geometry when the
tree metric is Euclidean and explicitly marks discrete or nonsmooth tie-cell
geometry otherwise. For continuous Euclidean cases, it additionally computes
the first-order null-whitened distance under the root empirical-Gaussian
feature covariance.

## Key Points

- The replay contract verifies that each selected merge is a minimum active
  average-linkage pair. It fails if the linkage matrix and condensed distances
  disagree instead of repairing or guessing.
- The final root merge has no competitor because only two active clusters
  remain. The selected-region information is therefore in the merge
  inequalities that construct the root child clusters, not in a fictitious
  final-root margin.
- Five representative cases were run:
  `binary_low_noise_2c`, `cat_mod_4cat_6c`, `dim_diffuse_6c_136f`,
  `dim_diffuse_6c_136f_continuous`, and `cat_highcard_20cat_4c`.
- Hamming/discretized and categorical root selections are dominated by exact
  or numerical-zero merge margins. Near-active construction counts are `18`,
  `89`, `89`, and `133` in the four non-continuous representatives.
- In schema `v3`, those four non-continuous representatives are classified as
  `discrete_tie_cell_geometry_required`, with `18`, `89`, `89`, and `133`
  selected discrete tie-cell construction constraints.
- The continuous Euclidean diffuse representative has smooth first-order
  signed-distance geometry for all `178` root-child construction constraints.
  Its minimum merge margin is about `4.05e-4`; after dividing by the actual
  merge-inequality gradient norm, the minimum first-order signed distance is
  about `2.24e-4`; after root empirical-Gaussian null scaling, the minimum
  null-whitened first-order distance is about `1.73e-4`.
- In the eight supported moderate continuous cases, every root context has
  null-whitened first-order merge distance. The high-dimensional continuous
  case is outside the dense covariance contract and is intentionally not part
  of this relationship check.
- The continuous relationship table shows that merge-margin geometry is not
  the dominant coordinate for root selected ratio in this small panel:
  Spearman correlation with log root sibling ratio is about `-0.024` for raw
  merge margin, `0.095` for ambient first-order distance, and `-0.190` for
  null-whitened first-order distance.
- Root edge-action proxy is the dominant coordinate in the same panel, with
  Spearman correlation `1.0` against log root sibling ratio. This points the
  next selected-region development toward the edge-opening boundary, not more
  merge-margin tuning.
- Root selected sibling ratios remain large in all five representatives:
  about `755`, `748`, `379`, `32.7`, and `665`, respectively.
- Root edge-action proxies are enormous for Hamming/discretized and categorical
  representatives, but lower for the continuous Euclidean representative
  (`14.2`).
- The diagnostic supports a sharper selected-region question: tie-heavy
  discrete hierarchy cells and positive-margin continuous hierarchy cells
  likely need separate geometric descriptions before any external selected
  tail law can be production-valid.
- Curvature is still not computed. The continuous output marks curvature as
  `not_materialized_high_dimensional_hessian_operator`; this is an explicit
  remaining proof/diagnostic gap, not a fallback.

## Evidence

- `tests/validation/55_test_root_selected_region_margins.py` verifies average
  linkage replay, rejects a nonminimal linkage matrix, and marks the
  two-leaf root case as having root children that are leaves.
- `raw/assets/benchmark-results/root_selected_region_margins_20260603/manifest.json`
  records schema `root_selected_region_margins/v3`, five root rows, `775`
  merge-margin rows, five relationship rows, and the diagnostic role
  `descriptive_root_selected_region_geometry_not_calibration`.
- `raw/assets/benchmark-results/root_selected_region_margins_20260603_continuous/manifest.json`
  records schema `root_selected_region_margins/v3`, eight supported continuous
  root rows, `1124` merge-margin rows, and five relationship rows.
- `raw/assets/benchmark-results/root_selected_region_margins_20260603/root_selected_region_summary.csv`
  records one row per representative case with root edge, sibling, spectral,
  and margin summaries.
- `raw/assets/benchmark-results/root_selected_region_margins_20260603/root_selected_region_merge_margins.csv`
  records step-level merge margins and root-child construction roles.

## Links

- [[root-selected-region-model]]
- [[method-proof-web]]
- [[selected-hierarchy-geometric-law-map]]
- [[open-mathematical-questions]]
