---
title: Gaussian Inner Node Branch-Time Debug 2026-06-23
type: source
status: reviewed
updated: 2026-06-24
sources:
  - raw/assets/benchmark-results/gaussian_inner_node_debug_20260623/manifest.json
  - raw/assets/benchmark-results/gaussian_inner_node_debug_20260623/summary.csv
  - raw/assets/benchmark-results/gaussian_inner_node_debug_20260623/gauss_clear_medium_continuous_standardized_active_vs_no_time_inner_nodes.csv
  - raw/assets/benchmark-results/gaussian_inner_node_debug_20260623/gauss_moderate_3c_continuous_standardized_active_vs_no_time_inner_nodes.csv
  - raw/assets/benchmark-results/gaussian_inner_node_debug_20260623/gaussian_eigen_branch_alignment_20260623.csv
tags:
  - source
  - benchmarks
  - gaussian
  - branch-length
  - validation
---

# Gaussian Inner Node Branch-Time Debug 2026-06-23

## Summary

The inner-node debug compares the active standardized continuous Gaussian TBS
tree against the same tree with only the branch-time covariance multiplier
disabled. It covers `gauss_clear_medium_continuous` and
`gauss_moderate_3c_continuous`. Both cases under-split to one cluster under
the active gate but recover the true Gaussian partition when the time multiplier
is removed from the Wald covariance path.

## Key Points

- `gauss_clear_medium_continuous` uses `standardized_euclidean` tree geometry
  with mean branch length `0.084749`. Active default TBS finds `K=1`, ARI `0.0`;
  the no-time run finds `K=4`, ARI `1.0`.
- `gauss_moderate_3c_continuous` uses `standardized_euclidean` tree geometry
  with mean branch length `0.119578`. Active default TBS finds `K=1`, ARI `0.0`;
  the no-time run finds `K=3`, ARI `1.0`.
- In `gauss_clear_medium_continuous`, the root `N118` is a pure `1:15` branch
  against a `0:15;2:15;3:15` branch. Its long pure child edge has normalized
  branch length `0.845880`, giving a child-edge variance multiplier `10.98x`;
  the active child-parent p-value is `0.123403`, while the no-time p-value is
  `1.05e-10`.
- The same case has two deeper true Gaussian split nodes, `N117` and `N116`.
  `N117` closes actively with sibling p-value `0.157299` but opens with
  no-time p-value `0.000118`; `N116` has both pure child edges closed actively
  at p-values around `0.24` but open at `5.08e-07` without the multiplier.
- In `gauss_moderate_3c_continuous`, root `N88` and child node `N87` follow
  the same pattern. The root's pure child edge multiplier is `7.49x`, active
  child-parent p-value is `0.053340`, and no-time p-value is `2.90e-10`.
  Node `N87` has pure child edges with roughly `6.5x` multipliers, active
  p-values around `0.108`, and no-time p-values `5.17e-07`.
- The fixed-coordinate side diagnostics are already very small at these nodes,
  but switching only the sibling gate to fixed-coordinate is insufficient for
  full recovery because the traversal child-parent prerequisite still uses the
  branch-time multiplier.
- Eigen diagnostics show that the tested Gaussian contrast is already embedded
  in the parent covariance. At the audited nodes, the mass-weighted between
  term accounts for `0.34` to `0.95` of the parent covariance trace; the top one
  or two raw covariance eigenvectors capture `0.87` to essentially `1.0` of
  the child-contrast energy.
- After null-whitening by that same parent covariance, the descendant covariance
  eigenvalues flatten: the whitened eigenvalue max/min ratio is `1.0` and the
  raw MP signal count is `0` at the audited nodes. The continuous spectral
  selector therefore falls back to the minimum projection dimension instead of
  detecting the Gaussian mean-shift axes as signal.
- The full no-time Wald norm saturates near the parent node size (`58.98` for
  root size `60`, `43.98` for root size `45`, and `29.0` for size-`30`
  binary splits). This matches the algebra for a parent covariance of the form
  within covariance plus \(m_L m_R \Delta\Delta^\top\): as separation grows,
  \(\Delta^\top\widehat\Sigma_u^{-1}\Delta\) saturates instead of continuing to
  grow.

## Evidence

- `raw/assets/benchmark-results/gaussian_inner_node_debug_20260623/manifest.json`
  records static provenance for the CSV bundle with
  `provenance_timestamp = 2026-06-24T20:27:01+02:00`; the original generation
  timestamp was not recorded in the CSV artifacts.
- `raw/assets/benchmark-results/gaussian_inner_node_debug_20260623/summary.csv`
  records active versus no-time cluster counts, ARI, open sibling counts, and
  changed-node counts.
- `raw/assets/benchmark-results/gaussian_inner_node_debug_20260623/gauss_clear_medium_continuous_standardized_active_vs_no_time_inner_nodes.csv`
  records all internal node values for the clear four-cluster continuous
  Gaussian case.
- `raw/assets/benchmark-results/gaussian_inner_node_debug_20260623/gauss_moderate_3c_continuous_standardized_active_vs_no_time_inner_nodes.csv`
  records all internal node values for the moderate three-cluster continuous
  Gaussian case.
- `raw/assets/benchmark-results/gaussian_inner_node_debug_20260623/gaussian_eigen_branch_alignment_20260623.csv`
  records raw covariance eigen alignment, whitened spectral eigenvalues,
  no-time/full Wald norms, projected Wald shares, and branch-time multipliers
  for the audited Gaussian internal nodes.

## Links

- [[full-benchmark-run-20260623]]
- [[continuous-tree-geometry-rethink-20260623]]
- [[projected-wald-statistic]]
