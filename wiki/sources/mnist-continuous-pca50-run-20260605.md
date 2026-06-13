---
title: MNIST Continuous PCA50 Run 2026-06-05
type: source
status: reviewed
updated: 2026-06-05
sources:
  - benchmarks/results/experiments/mnist/mnist_continuous_pca50_summary.csv
tags:
  - source
  - benchmarks
  - mnist
  - continuous
---

# MNIST Continuous PCA50 Run 2026-06-05

## Summary

A continuous MNIST probe ran on 2026-06-05 using `2000` sampled images,
normalized pixel intensities, PCA to `50` continuous components, Euclidean tree
distances, and five linkage methods.

## Key Points

- PCA50 retained `83.13%` of variance.
- Best result was `euclidean + complete`: `18` clusters, ARI `0.258244`, NMI
  `0.466613`.
- `euclidean + weighted` found `6` clusters with ARI `0.247486` and NMI
  `0.407067`.
- `euclidean + ward` over-split to `96` clusters with ARI `0.135648` and NMI
  `0.560762`.
- `euclidean + single` and `euclidean + average` returned one cluster, with
  ARI and NMI both `0.0`.

## Evidence

- `benchmarks/results/experiments/mnist/mnist_continuous_pca50_summary.csv`
  contains the result table.

## Links

- [[mnist-benchmark-run-20260605]]
- [[full-benchmark-run-20260605]]
- [[selected-hierarchy-null-support-contract]]
