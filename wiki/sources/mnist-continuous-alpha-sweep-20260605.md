---
title: MNIST Continuous Alpha Sweep 2026-06-05
type: source
status: reviewed
updated: 2026-06-05
sources:
  - benchmarks/results/experiments/mnist/alpha_sweep_continuous_pca50_20260605/alpha_sweep_summary.csv
  - benchmarks/results/experiments/mnist/alpha_sweep_continuous_pca50_20260605/alpha_heatmaps.png
  - benchmarks/results/experiments/mnist/alpha_sweep_continuous_pca50_20260605/best_pca2_panels.png
tags:
  - source
  - benchmarks
  - mnist
  - continuous
---

# MNIST Continuous Alpha Sweep 2026-06-05

## Summary

A focused alpha sweep ran on 2026-06-05 for continuous MNIST PCA50. The sweep
used the same `2000` sampled normalized MNIST images and PCA50 representation
as [[mnist-continuous-pca50-run-20260605]], then evaluated `complete`,
`weighted`, and `ward` Euclidean trees over a `5 x 5` grid of `edge_alpha` and
`sibling_alpha` values: `0.0001`, `0.001`, `0.01`, `0.05`, and `0.1`.

## Key Points

- All `75` alpha/linkage configurations completed without calibration-support
  or dense-covariance failures.
- The best setting was `ward` with `edge_alpha=0.0001` and
  `sibling_alpha=0.0001`: `11` clusters, ARI `0.514854`, NMI `0.646689`.
- Ward was highly alpha-sensitive. Increasing either alpha made the traversal
  split more aggressively, rising from `11` clusters at the strictest pair to
  `166` clusters at `edge_alpha=0.1`, `sibling_alpha=0.1`, while ARI fell to
  `0.087`.
- Complete linkage remained weakly alpha-sensitive: its best setting was
  `edge_alpha=0.001`, `sibling_alpha=0.05` with `21` clusters, ARI `0.274665`,
  and NMI `0.485078`.
- Weighted linkage was driven mainly by `sibling_alpha`, ranging from one
  cluster at `sibling_alpha=0.0001` to six clusters at `0.05`, with best ARI
  `0.247486`.
- Visual diagnostics show that strict Ward separates several digit-shaped
  regions but still mixes hard handwriting confusions such as `3/5/8`, `4/9`,
  and part of `7/9/4`.

## Evidence

- `benchmarks/results/experiments/mnist/alpha_sweep_continuous_pca50_20260605/alpha_sweep_summary.csv`
  contains the 75-row metric table.
- `benchmarks/results/experiments/mnist/alpha_sweep_continuous_pca50_20260605/alpha_sweep_assignments.csv`
  contains sample-level predicted labels for each alpha pair.
- `benchmarks/results/experiments/mnist/alpha_sweep_continuous_pca50_20260605/alpha_heatmaps.png`
  visualizes ARI and predicted cluster counts over the alpha grid.
- `benchmarks/results/experiments/mnist/alpha_sweep_continuous_pca50_20260605/best_pca2_panels.png`
  compares the best setting for each linkage against true digit labels in a
  two-component PCA projection.

## Links

- [[mnist-continuous-pca50-run-20260605]]
- [[mnist-benchmark-run-20260605]]
- [[alpha-grid-full-20260604]]
