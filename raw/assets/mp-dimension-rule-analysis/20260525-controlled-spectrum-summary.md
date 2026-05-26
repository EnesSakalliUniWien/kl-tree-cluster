# Controlled Marchenko-Pastur Spectrum Probe, 2026-05-25

This raw evidence note records the local controlled simulations run during the
Marchenko-Pastur dimension-rule audit. The probe used the production
`eigendecompose_covariance` backend and the production
`estimate_marchenko_pastur_dimension` rule.

## Implementation Path Checked

- `kl_clustering_analysis/hierarchy_analysis/decomposition/backends/eigen/decomposition.py`
- `kl_clustering_analysis/hierarchy_analysis/decomposition/backends/eigen/operators.py`
- `kl_clustering_analysis/hierarchy_analysis/statistics/projection/projection_dimension_estimation/projection_dimension_estimators.py`
- `kl_clustering_analysis/hierarchy_analysis/statistics/projection/spectral/marchenko_pastur.py`

The backend centers rows, builds either `X.T @ X / n` or `X @ X.T / n`, and
returns descending nonnegative eigenvalues. The dual and primal paths therefore
share the same nonzero feature-covariance eigenvalue scale.

## Pure Gaussian Null

Rows are independent `N(0, I_d)`. `raw_fp_rate_n` is the fraction of runs with
at least one eigenvalue above `(1 + sqrt(d/n))^2`. `raw_fp_rate_nminus1` uses
`d/(n-1)` after centering. `final_k_floor2_mean` is the returned dimension when
the production floor is 2.

| n | d | raw_fp_rate_n | raw_fp_rate_nminus1 | final_k_floor2_mean | mp_edge_n |
|---:|---:|---:|---:|---:|---:|
| 20 | 100 | 0.0967 | 0.0400 | 2.0 | 10.4721 |
| 40 | 100 | 0.1300 | 0.0767 | 2.0 | 6.6623 |
| 100 | 50 | 0.1300 | 0.1133 | 2.0 | 2.9142 |
| 100 | 500 | 0.0867 | 0.0433 | 2.0 | 10.4721 |
| 30 | 20 | 0.0767 | 0.0467 | 2.0 | 3.2997 |
| 200 | 20 | 0.0967 | 0.0933 | 2.0 | 1.7325 |

Interpretation: the raw asymptotic edge gives finite-sample false spike counts
around 8--13% in these small local matrices. The production dimension floor
then returns two dimensions even when the raw MP count is zero.

## Centering Convention Recheck

The eigendecomposition backend centers rows and divides by `n`, not by `n-1`.
For a centered Gaussian matrix, the nonzero spectrum is approximately
\[
\frac{n-1}{n}
\left(1+\sqrt{\frac{d}{n-1}}\right)^2
\]
at the upper edge if one models the centered rank as \(n-1\). This
scale-matched centered edge is slightly lower than `(1 + sqrt(d/n))^2`, while
the unscaled `n-1` edge is slightly higher.

| n | d | current edge | centered scale-matched edge | unscaled n-1 edge |
|---:|---:|---:|---:|---:|
| 20 | 100 | 10.4721 | 10.3089 | 10.8515 |
| 40 | 100 | 6.6623 | 6.5975 | 6.7667 |
| 100 | 50 | 2.9142 | 2.8971 | 2.9264 |
| 100 | 500 | 10.4721 | 10.4397 | 10.5452 |
| 30 | 20 | 3.2997 | 3.2389 | 3.3506 |
| 200 | 20 | 1.7325 | 1.7259 | 1.7345 |

In 400-run Gaussian null probes with the production backend, false top-spike
rates were:

| n | d | current edge | centered scale-matched edge | unscaled n-1 edge |
|---:|---:|---:|---:|---:|
| 20 | 100 | 0.090 | 0.128 | 0.037 |
| 40 | 100 | 0.130 | 0.168 | 0.077 |
| 100 | 50 | 0.122 | 0.147 | 0.110 |
| 100 | 500 | 0.085 | 0.110 | 0.052 |
| 30 | 20 | 0.090 | 0.128 | 0.068 |
| 200 | 20 | 0.105 | 0.122 | 0.102 |

Interpretation: the current edge is not exactly the finite centered Gaussian
edge, but the exact centered scale correction would lower the edge and increase
finite-sample false spike rates. The unscaled `n-1` edge is more conservative
but is not scale-matched to the backend's `1/n` covariance. A real finite
sample calibration needs a null quantile or Tracy-Widom-style correction, not
only an `n` versus `n-1` substitution.

## Rank-One Spiked Gaussian Probe

Rows are `N(0, I_d) + strength * z * v.T` with one unit loading vector. This is
a basic power check, not a full hierarchy simulation.

| n | d | strength | signal variance | raw_detect_rate | top_eig_mean | mp_edge |
|---:|---:|---:|---:|---:|---:|---:|
| 40 | 100 | 0.5 | 0.25 | 0.145 | 6.2660 | 6.6623 |
| 40 | 100 | 1.0 | 1.00 | 0.165 | 6.3746 | 6.6623 |
| 40 | 100 | 1.5 | 2.25 | 0.600 | 6.7646 | 6.6623 |
| 40 | 100 | 2.0 | 4.00 | 0.920 | 7.9861 | 6.6623 |
| 40 | 100 | 3.0 | 9.00 | 1.000 | 12.6870 | 6.6623 |
| 40 | 100 | 4.0 | 16.00 | 1.000 | 19.6753 | 6.6623 |

Interpretation: the rule has the expected high-dimensional spike threshold
behavior. Weak spikes below the local upper edge are usually not counted.

## Deterministic Internal Rows Under Null

Rows are independent Gaussian leaves. Internal rows are deterministic means of
random groups of leaves, then stacked below the leaves before eigendecomposition.
The current production-style count treats the augmented row count as `n`.

| n_leaf | d | internal groups | mean_aug_rows | leaf_only_fp_rate | aug_with_aug_n_fp_rate | aug_with_leaf_n_fp_rate | aug_with_aug_n_mean_count |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 40 | 100 | 10 | 49.86 | 0.100 | 0.323 | 0.020 | 0.330 |
| 40 | 100 | 30 | 62.31 | 0.107 | 1.000 | 0.080 | 2.437 |
| 100 | 50 | 20 | 119.88 | 0.090 | 0.023 | 0.000 | 0.023 |
| 30 | 20 | 10 | 39.54 | 0.103 | 0.093 | 0.017 | 0.093 |

Interpretation: counting deterministic internal summaries as independent rows
can materially change the MP threshold. In the high-dimensional small-leaf
case, the augmented-row threshold produces frequent null spikes.

## Noise-Scale Misspecification Probe

Rows are `sigma * N(0, I_d)` for `n=40, d=100`. The current rule assumes the
null-whitened scale is exactly one. The median-scale rule estimates the local
bulk scale from the positive spectrum and then applies the MP upper-edge ratio.

| sigma | current_fp_rate | median_scale_fp_rate | current_mean_count | median_scale_mean_count |
|---:|---:|---:|---:|---:|
| 0.8 | 0.000 | 0.143 | 0.000 | 0.153 |
| 1.0 | 0.120 | 0.140 | 0.120 | 0.143 |
| 1.2 | 1.000 | 0.160 | 4.950 | 0.163 |
| 1.5 | 1.000 | 0.150 | 13.577 | 0.157 |
| 2.0 | 1.000 | 0.133 | 23.973 | 0.133 |

Interpretation: exact unit-scale whitening is a strong assumption. A modest
scale error makes the current fixed-scale edge count many null eigenvalues as
signals. A median-scale estimator repairs this isolated scale problem, but it
does not by itself solve dependent/internal-row spectra.
