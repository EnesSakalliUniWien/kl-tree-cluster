---
title: Family Metric NNLS Grid 2026-07-09
type: source
status: reviewed
updated: 2026-07-09
sources:
  - benchmarks/validation/family_metric_nnls_grid.py
  - tests/validation/197_test_family_metric_nnls_grid.py
  - reports/family_metric_nnls_grid_20260709/family_metric_nnls_report.md
  - reports/family_metric_nnls_grid_20260709/family_metric_nnls_cells.csv
  - reports/family_metric_nnls_grid_20260709/family_metric_nnls_selection.csv
  - reports/family_metric_nnls_grid_20260709/family_metric_nnls_branch_time_pairs.csv
  - reports/family_metric_nnls_grid_20260709/family_metric_nnls_pairwise_agreement.csv
  - reports/family_metric_nnls_grid_20260709/family_metric_nnls_manifest.json
  - benchmarks/shared/runners/tbs_diffusion_runner.py
  - benchmarks/shared/runners/tbs_runner.py
  - tree_break_selection/tree/optimized_branch_lengths.py
tags:
  - source
  - benchmarks
  - geometry
  - graphtools
  - nnls
  - topology
  - branch-lengths
---

# Family Metric NNLS Grid 2026-07-09

## Summary

A focused evidence-only grid replaced the universal raw-coordinate Hamming
geometry with one label-free family geometry shared by adaptive-K graphtools
and fixed-topology NNLS. The original observations remained the input to TBS
distributional tests. Six representative cases were run through all eight
topology methods with branch-time disabled and enabled, producing `96` cells,
`86` successful results, and `10` fail-closed results.

The result is not a production promotion. Family geometry fixes topology for
clear continuous Gaussian, balanced binary, and categorical examples, and the
label-free selector finds strong categorical and moderate-overlap partitions.
It does not fix high-dimensional continuous or SBM topology. Clear Gaussian
trees are already externally perfect at the known four-cluster cut, but the
selected sibling law blocks every TBS split. This separates topology failure
from inferential failure and rules out a global alpha increase as the common
repair.

## Key Points

- The tested profiles were standardized Euclidean for ordinary continuous
  data, Ledoit--Wolf shrinkage Mahalanobis for `p >= n` continuous data,
  Hamming represented as squared Euclidean for balanced binary data, cosine
  represented on the unit sphere for overlap binary data, regularized
  Laplacian spectral coordinates for SBM, and equally weighted categorical
  block mismatch represented as a Hellinger-style one-hot embedding.
- The graph kernel and NNLS fit receive the same selected embedding. The new
  NNLS target `squared_euclidean` does not standardize that prepared embedding
  a second time. Existing callers retain the prior squared standardized
  Euclidean default.
- `binary_low_noise_4c` is recovered exactly by all eight topology methods:
  ARI, NMI, macro F1, known-K branch cut, exact-clade recovery, and weighted
  clade purity are all `1.0`. Mean normalized NNLS residual is `0.0592`.
- `cat_clear_3cat_4c` has mean ARI `0.8540`; average, complete, single, and
  median recover the four labels exactly, and the label-free selector chooses
  complete with ARI/NMI/macro F1 `1.0`. Across methods, exact true-cluster
  clade recovery is `0.96875` and weighted clade purity is `0.984375`.
- The categorical known-K longest-branch cut has ARI near zero despite strong
  clade recovery. The topology contains the clusters, but globally fitted
  NNLS edge lengths do not rank cluster-boundary edges above within-cluster
  edges. Branch lengths therefore cannot be interpreted as a cluster-cut
  score.
- `overlap_mod_4c_small` has mean ARI `0.6632`; the label-free selector chooses
  average with ARI `0.9148`, NMI `0.8883`, and macro F1 `0.9676`. The known-K
  branch cut and exact-clade audit are poor, so useful traversal decisions do
  not imply globally monophyletic ground-truth groups in this overlap case.
- Every clear-Gaussian topology has known-K ARI/NMI/macro F1 `1.0`, exact
  true-cluster clades, weighted clade purity `1.0`, and normalized NNLS
  residual near `0.033`. TBS nevertheless returns one cluster. Without
  branch-time, linkage roots have BH edge p-values as small as `1.8e-13` but
  corrected sibling p-values from `0.067` to `0.494`; branch-time closes the
  root edge path. This is an inferential-law failure after topology recovery.
- `cont_lowrank_pggn_shrinkage` has no exact true-cluster clades, weighted
  clade purity `0.2675`, and known-K ARI near zero. Five linkage methods per
  branch mode fail because the sibling inflation model has only selected
  non-null records and no strict-null or stopped-edge calibration support.
  Global total-covariance whitening also risks suppressing mean-separation
  directions, so a covariance/spectral alternative is still required.
- `sbm_moderate` has mean normalized NNLS residual `0.6883`, no exact true
  cluster clades, weighted clade purity `0.3551`, and known-K mean ARI
  `0.0193`. The tested regularized-Laplacian/eigengap profile is not adequate;
  adjacency spectral geometry and an SBM-specific fit gate remain open.
- Across `43` paired-success topology cells, branch-time on versus off changes
  no final partition and has mean delta zero for ARI, NMI, macro F1,
  silhouette, and largest-cluster fraction. It does change edge-open counts,
  so the null final delta must not be read as branch-time irrelevance.
- NNLS residual alone is not an admissibility certificate. The
  high-dimensional profile has a low residual on a poor topology, while the
  categorical profile has strong clades but misleading longest-edge cuts.
  A production branch-time gate needs aligned geometry, normalized residual,
  path-distance correlation, resampling stability, and topology stability.
- Bernoulli deviance was not duplicated as a raw-leaf metric because deviance
  between unsmoothed `0/1` observations is undefined or infinite; it requires
  an explicit local probability estimator and smoothing contract. For pure
  equally weighted categorical one-hot blocks, mismatch, categorical Gower,
  and squared Hellinger differ only by fixed scaling, so duplicate topology
  cells add no information. Jaccard and adjacency-spectral alternatives remain
  separate candidates for the next focused panel.

## Evidence

- `reports/family_metric_nnls_grid_20260709/family_metric_nnls_report.md`
  records the run scope, family summaries, label-free selections, branch-time
  paired effect, and fail-closed cells.
- `reports/family_metric_nnls_grid_20260709/family_metric_nnls_cells.csv`
  records external metrics, label-free internal metrics, fragmentation,
  selected-node p-value summaries, NNLS residuals, root balance, known-K
  branch cuts, and true-cluster clade audits for each cell.
- `reports/family_metric_nnls_grid_20260709/family_metric_nnls_manifest.json`
  records completeness, label integrity, label-free selector invariance, and
  explicit `no_promotion` status.
- `tests/validation/197_test_family_metric_nnls_grid.py` verifies exact
  Hamming, cosine, and categorical block-distance identities, finite
  continuous/SBM embeddings, and artifact generation.

## Links

- [[full-graphtools-tree-consensus-gate-20260706]]
- [[graphtools-adaptive-k-tree-consensus-focus-benchmark-20260630]]
- [[scrna-branch-length-effect-audit-20260624]]
