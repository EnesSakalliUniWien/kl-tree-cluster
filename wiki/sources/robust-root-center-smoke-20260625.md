---
title: Robust Root Center Smoke 2026-06-25
type: source
status: draft
updated: 2026-06-25
sources:
  - benchmarks/diagnostics/calibration/robust_root_center_smoke.py
  - raw/assets/benchmark-results/robust_root_center_smoke_20260625/manifest.json
  - raw/assets/benchmark-results/robust_root_center_smoke_20260625/robust_root_center_rows.csv
  - raw/assets/benchmark-results/robust_root_center_smoke_20260625/robust_root_center_summary.csv
tags:
  - source
  - diagnostics
  - root
  - robust-statistics
  - tukey-depth
---

# Robust Root Center Smoke 2026-06-25

## Summary

`robust_root_center_smoke.py` tests whether robust center estimators can act as
high-dimensional root anchors for a branched point cloud with latent root at
the origin. The diagnostic compares the centroid, coordinate median,
geometric median, Euclidean medoid, approximate deepest observed point, and an
approximate random-direction halfspace-depth center over a continuous
candidate pool.

The result supports a narrow use: robust continuous centers help locate a
latent root anchor under outliers. It does not support replacing selected-root
topology validation with a center estimate.

## Key Points

- The smoke uses dimensions `5`, `20`, and `100`, four branch clusters, balanced
  and imbalanced branch sizes, and clean versus `10%` outlier scenarios.
- In balanced clean data, the centroid is best or tied-best, as expected for a
  symmetric Gaussian root anchor.
- Under `10%` outliers, the geometric median and continuous approximate
  halfspace-depth center dominate the centroid across dimensions.
- In dimension `100` with balanced `10%` outliers, mean root error is `0.933`
  for the continuous approximate halfspace-depth center and geometric median,
  versus `1.208` for the centroid.
- In dimension `20` with imbalanced `10%` outliers, the continuous approximate
  halfspace-depth center has mean root error `0.978`, geometric median `0.980`,
  centroid `1.329`.
- Observed-point center proxies are poor high-dimensional root anchors:
  Euclidean medoid and approximate deepest observed point have mean root error
  around `8`--`9` in dimension `100` because observed points remain on noisy
  branch leaves rather than at the latent root.
- The method-facing interpretation is diagnostic-only: center estimates may
  provide a root anchor or prior, but the selected root \(G_{\hat r}\) still
  requires topology replay, selected-root validity, and tail-support evidence.

## Evidence

- The script writes row-level records to
  `raw/assets/benchmark-results/robust_root_center_smoke_20260625/robust_root_center_rows.csv`.
- The summary CSV records method ranks by dimension and scenario.
- The manifest records the run configuration, including `40` replicates,
  `1024` random halfspace-depth directions, and a continuous candidate pool
  size of `400`.
- `py_compile` and Ruff passed for
  `benchmarks/diagnostics/calibration/robust_root_center_smoke.py`.

## Links

- [[root-selected-validity-replay-panel-20260617]]
- [[root-tree-geometry-hard-negative-replay-20260617]]
- [[root-selected-spectral-tail-law-with-legacy-overlay-20260617]]
