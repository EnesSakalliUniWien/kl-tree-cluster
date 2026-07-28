---
title: Robust Center Rooted Tree Smoke 2026-06-25
type: source
status: draft
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/root/center/robust_center_rooted_tree_smoke.py
  - raw/assets/benchmark-results/robust_center_rooted_tree_smoke_20260625/manifest.json
  - raw/assets/benchmark-results/robust_center_rooted_tree_smoke_20260625/robust_center_rooted_tree_rows.csv
  - raw/assets/benchmark-results/robust_center_rooted_tree_smoke_20260625/robust_center_rooted_tree_summary.csv
  - raw/assets/benchmark-results/robust_center_rooted_tree_smoke_20260625_min15/manifest.json
  - raw/assets/benchmark-results/robust_center_rooted_tree_smoke_20260625_min15/robust_center_rooted_tree_rows.csv
  - raw/assets/benchmark-results/robust_center_rooted_tree_smoke_20260625_min15/robust_center_rooted_tree_summary.csv
tags:
  - source
  - diagnostics
  - root
  - robust-statistics
  - topology
---

# Robust Center Rooted Tree Smoke 2026-06-25

## Summary

`robust_center_rooted_tree_smoke.py` tests centroid-rooted,
geometric-median-rooted, and approximate Tukey-depth-rooted versions of the
same unrooted average-linkage tree. For each simulated branched point cloud,
the script estimates a center, chooses the nontrivial tree edge closest to that
center, and scores the resulting root cut by distance to the latent root,
branch integrity, and outlier isolation.

The result is mixed. Robust centers are better root anchors under outliers, but
rooting an already-built average-linkage tree by nearest center does not
automatically improve the selected root edge. A side-size/root-validity guard
is necessary; otherwise the nearest root edge can simply isolate the outlier
block for every center method.

## Key Points

- The loose `5%` side-size guard lets the chosen root edge isolate the `10%`
  outlier block in outlier scenarios. In those rows, centroid, geometric
  median, and approximate Tukey-depth rooting often select the same edge, so
  robust centers improve center error but not tree rooting.
- Under the stricter `15%` side-size guard, outlier isolation is reduced and
  robust rooting gives modest branch-integrity gains in some outlier settings.
- In dimension `100`, imbalanced `10%` outliers, with the `15%` guard:
  centroid-rooted branch integrity is `0.871`, while geometric-median-rooted
  and Tukey-depth-rooted branch integrity are `0.894`.
- In dimension `5`, balanced `10%` outliers, with the `15%` guard:
  geometric-median and Tukey-depth rooting have root-edge true distance
  `0.253`, versus centroid-rooted `0.271`.
- The centroid still wins or ties several clean and imbalanced-clean rows,
  consistent with the earlier center-only smoke.
- Approximate Tukey-depth rooting is usually nearly identical to
  geometric-median rooting because the continuous candidate pool frequently
  selects the geometric median or a close candidate.
- The method-facing lesson is not "replace the root by a robust center." The
  stronger statement is: robust centers can be useful root priors only when
  paired with nontrivial-side, outlier, topology-stability, and selected-tail
  validity guards.

## Evidence

- The primary `5%` guard run writes rows and summary under
  `raw/assets/benchmark-results/robust_center_rooted_tree_smoke_20260625/`.
- The `15%` side-size sensitivity writes rows and summary under
  `raw/assets/benchmark-results/robust_center_rooted_tree_smoke_20260625_min15/`.
- `py_compile` and Ruff passed for
  `benchmarks/diagnostics/calibration/root/center/robust_center_rooted_tree_smoke.py`.
- `make wiki-lint` passed after adding this page and index coverage.

## Links

- [[robust-root-center-smoke-20260625]]
- [[root-selected-validity-replay-panel-20260617]]
- [[root-tree-geometry-hard-negative-replay-20260617]]
