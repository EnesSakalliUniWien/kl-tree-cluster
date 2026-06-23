---
title: Continuous Tree Geometry Rethink
type: analysis
status: reviewed
updated: 2026-06-23
sources:
  - tree_break_selection/tree/continuous_distance.py
  - benchmarks/shared/generators/case_data_contracts.py
  - benchmarks/shared/runners/dispatch.py
  - tree_break_selection/tree/branch_lengths.py
  - manuscript/sections/method/representation.tex
  - raw/assets/benchmark-results/continuous_tree_geometry_rethink_20260623/compact_family_results.txt
  - raw/assets/benchmark-results/continuous_tree_geometry_rethink_20260623/compact_family_results_after_patch.txt
  - raw/assets/benchmark-results/continuous_tree_geometry_rethink_20260623/compact_family_method_results_after_patch.csv
  - raw/assets/benchmark-results/continuous_tree_geometry_rethink_20260623/compact_family_oracle_results_after_patch.csv
  - raw/assets/benchmark-results/continuous_tree_geometry_rethink_20260623/continuous_geometry_spectral_size_eigen_diagnostics.csv
tags:
  - analysis
  - geometry
  - continuous
  - tree-break-selection
---

# Continuous Tree Geometry Rethink

## Summary

The continuous tree-distance contract now separates Gaussian location-style
continuous cases from covariance/Brownian diagnostic cases. Gaussian blobs,
dimensional Gaussian, and Gaussian outlier continuous cases precompute
`standardized_euclidean` tree distances. Low-rank, spiked covariance, and
phylogenetic Brownian continuous diagnostics keep `mahalanobis_time`.

The reason for the split is quantitative. Before this patch, the consolidated
continuous mean-shift case used pooled full-covariance Mahalanobis tree
geometry. That made exact-\(K\) oracle ARI `0.000186` and TBS ARI `0.000`.
After switching that family to standardized Euclidean geometry, the same case
has exact-\(K\) oracle ARI `1.000` and active TBS ARI `1.000`.

The rethought method boundary is therefore: tree-building geometry and
projected-Wald contrast geometry must be treated as separate objects. The tree
distance proposes candidate subtrees; the local feature-space covariance
standardizes tests after those subtrees are selected.

## Details

The implemented Mahalanobis-time diagnostic distance is

\[
d_T(a,b)
=
\frac{1}{p}
(x_a-x_b)^\top \widehat\Sigma^{-1}(x_a-x_b),
\]

where \(\widehat\Sigma\) is estimated from the full generated continuous
matrix. For a mixture with within-cluster covariance \(\Sigma_W\) and
between-cluster covariance \(\Sigma_B\),

\[
\widehat\Sigma_{\mathrm{total}}
\approx
\Sigma_W+\Sigma_B.
\]

Whitening by this total covariance divides along the same directions that carry
mean-shift signal. In eigen-coordinates,

\[
d_T(a,b)
=
\frac{1}{p}\sum_j
\frac{\{u_j^\top(x_a-x_b)\}^2}{\lambda_j}.
\]

When a signal direction has a large mixture eigenvalue \(\lambda_j\), the
distance shrinks that signal direction. This is self-normalization by the
alternative, not a neutral Brownian-time correction.

The standardized Euclidean replacement for Gaussian location-style tree
construction is

\[
d_Z(a,b)
=
\lVert D^{-1/2}(x_a-x_b)\rVert_2,
\qquad
D=\operatorname{diag}(\widehat\sigma_1^2,\ldots,\widehat\sigma_p^2).
\]

This still handles coordinate scale, but it does not rotate into the mixture
eigenvectors and divide away high-variance signal modes.

### Eigenvalue And Eigenvector Relation

For `dim_consolidated_4c_24f_continuous`, the failure is strongly tied to the
eigensystem of the pooled total covariance:

```text
trace_between_share                              0.567886
top_kminus1_total_eigen_signal_mass             0.998799
top_kminus1_eigenvector_center_subspace_cos2    0.998783
largest_to_median_eigenvalue                   11.795953
mahalanobis_to_raw_between_energy               0.086563
standardized_to_raw_between_energy              0.273968
```

Nearly all cluster-mean signal lies in the top total-covariance eigenvectors,
and those eigenvectors are almost the same subspace as the cluster-center
subspace. Pooled Mahalanobis geometry therefore divides the main signal by the
largest mixture eigenvalues. The observed pairwise distance gap rises from
`0.590465` pooled-SD units under `mahalanobis_time` to `2.040918` under
`standardized_euclidean`. The root split changes from a singleton-like
`1/159` split to a `40/120` split, and exact-\(K\) oracle ARI rises from
`0.000186` to `1.000`.

For `gauss_clear_medium_continuous` and `gauss_single_outlier_4c_continuous`,
Mahalanobis-time still has exact-\(K\) oracle ARI `1.000`, but it produces
much weaker distance gaps and, for the outlier case, a singleton root split:

```text
gauss_clear_medium:       gap 2.322381 -> 26.716167
gauss_single_outlier_4c:  gap 1.490055 ->  9.481466
```

The same eigenvalue mechanism is visible there: between-cluster signal is
almost entirely in top total-covariance eigenvectors and
`mahalanobis_to_raw_between_energy` is about `0.003`.

For `cont_lowrank_pggn_shrinkage`, the failure is different:

```text
trace_between_share                              0.040126
top_kminus1_total_eigen_signal_mass             0.076295
top_kminus1_eigenvector_center_subspace_cos2    0.075177
mahalanobis_to_raw_between_energy               0.037223
standardized_to_raw_between_energy              0.998107
standardized_euclidean distance_gap_pooled_sd   0.088069
standardized_euclidean exact-K oracle ARI       0.002069
```

The low-rank case does not put truth signal into a clean top-eigenvector
location subspace, and standardized Euclidean barely separates within and
between pairs. This is why the fix should not be generalized to "all
continuous data are solved by z-scoring"; the low-rank case needs a
factor-aware tree geometry.

### Family Failure Split

The compact family diagnostic separates failure modes:

- `dim_consolidated_4c_24f_continuous` was a tree-geometry failure under pooled
  Mahalanobis distance and is now solved under active standardized Euclidean
  average linkage.
- `cont_lowrank_pggn_shrinkage` remains unrecoverable under active and
  standardized Euclidean geometries; it needs a separate factor/subspace model
  rather than a simple distance replacement.
- `binary_low_noise_4c` has an exactly recoverable active Hamming tree but the
  default traversal over-splits; this is a selected-gate/traversal problem.
- `cat_clear_3cat_4c` is partially recoverable under active Hamming average
  linkage and benefits from fixed sibling gates, but still belongs to the
  selected-root/traversal validation problem.
- `overlap_unbal_4c_small` and `sbm_moderate` have weak oracle recoverability
  under the active tree geometry, so they should not be interpreted as pure
  gate failures.

Branch length should remain a normalized variance relaxation, not a substitute
tree model. The current linkage branch length is

\[
\ell(u,c)=\frac{h(u)-h(c)}{h(\mathrm{root})},
\]

so each linkage-derived root-to-leaf path sums to one when the root height is
positive. Edge and sibling tests then use normalized branch time as
dimensionless variance multipliers. This can relax overconfident tests, but it
cannot recover a candidate tree built from the wrong distance.

The practical method revision is:

1. For Gaussian location-style continuous benchmarks, use standardized
   Euclidean tree geometry.
2. Reserve pooled/full Mahalanobis tree distance for regimes with an external,
   known, held-out, or otherwise validated null covariance; do not estimate the
   tree distance covariance from the same selected mixture by default.
3. Keep local projected-Wald covariance as the test standardization object, but
   state that continuous selected-hierarchy calibration remains unvalidated
   outside supported regimes.
4. Treat low-rank continuous data as outside the simple Euclidean mean-shift
   contract until a factor-aware candidate tree is validated.

## Evidence

- `tree_break_selection/tree/continuous_distance.py` defines
  `mahalanobis_time` as a full empirical-covariance Mahalanobis time distance
  and `standardized_euclidean` as a diagonal-scale Euclidean tree distance.
- `benchmarks/shared/generators/case_data_contracts.py` precomputes the
  requested continuous tree distance for benchmark metadata.
- `benchmarks/shared/runners/dispatch.py` supports both continuous tree
  metrics and forwards provided precomputed distances to TBS.
- `tree_break_selection/tree/branch_lengths.py` normalizes linkage-derived
  branch lengths by the root merge height.
- `manuscript/sections/method/representation.tex` states the split between
  Gaussian location-style standardized Euclidean tree distances and diagnostic
  continuous metric-specific distances.
- `raw/assets/benchmark-results/continuous_tree_geometry_rethink_20260623/compact_family_results.txt`
  records the pre-patch compact current-vs-standardized-Euclidean and oracle
  checks.
- `raw/assets/benchmark-results/continuous_tree_geometry_rethink_20260623/compact_family_results_after_patch.txt`
  records the post-patch active-family check.
- `raw/assets/benchmark-results/continuous_tree_geometry_rethink_20260623/continuous_geometry_spectral_size_eigen_diagnostics.csv`
  records the eigenvalue, eigenvector-alignment, cluster-size, root-size, and
  distance-separation diagnostics.

## Links

- [[tree-break-selection]]
- [[projected-wald-statistic]]
- [[dimensional-gaussian-representation-diagnostic]]
- [[selected-hierarchy-selection-geometry]]
- [[oracle-gate-path-diagnostic]]

## Open Questions

- What factor-aware tree geometry should represent low-rank continuous cases?
- Which continuous regimes can validate a full Mahalanobis tree distance with
  an external or held-out null covariance?
- Should Gaussian location-style standardized Euclidean use average linkage as
  the family default in every benchmark profile, or should complete linkage
  remain an explicit robustness candidate?
