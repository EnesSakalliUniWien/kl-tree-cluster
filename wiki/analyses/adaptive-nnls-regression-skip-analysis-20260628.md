---
title: Adaptive NNLS Regression Skip Analysis 2026-06-28
type: analysis
status: reviewed
updated: 2026-06-28
sources:
  - benchmarks/results/run_20260628_082116Z_full/full_benchmark_comparison.csv
  - benchmarks/results/run_20260628_082116Z_full/failure_report.md
  - reports/adaptive_nnls_regression_analysis_20260628/finite_negative_deltas_vs_hamming_diffusion.csv
  - reports/adaptive_nnls_regression_analysis_20260628/new_skips_while_hamming_diffusion_ok.csv
  - reports/adaptive_nnls_regression_analysis_20260628/adaptive_nnls_skip_reason_counts.csv
  - reports/adaptive_nnls_regression_analysis_20260628/targeted_variant_diagnostics.csv
  - reports/adaptive_nnls_regression_analysis_20260628/multi_metric_deltas_vs_hamming_diffusion_long.csv
  - reports/adaptive_nnls_regression_analysis_20260628/multi_metric_case_summary_vs_hamming_diffusion.csv
  - reports/adaptive_nnls_regression_analysis_20260628/multi_metric_aggregate_vs_hamming_diffusion.csv
  - reports/adaptive_nnls_regression_analysis_20260628/ari_negative_rows_nonari_quality_context.csv
  - reports/adaptive_nnls_regression_analysis_20260628/independent_metric_context_vs_hamming_diffusion.csv
  - reports/adaptive_nnls_regression_analysis_20260628/diffusion_backend_nnls_variant_panel.csv
  - reports/adaptive_nnls_regression_analysis_20260628/diffusion_backend_nnls_variant_metrics_compact.csv
  - reports/adaptive_nnls_regression_analysis_20260628/diffusion_backend_nnls_within_backend_deltas.csv
  - reports/adaptive_nnls_regression_analysis_20260628/diffusion_backend_case_best_summary.csv
  - benchmarks/shared/generators/generate_categorical_matrix.py
  - benchmarks/shared/generators/categorical_cases.py
  - benchmarks/shared/generators/dirichlet_multinomial.py
  - benchmarks/shared/generators/gaussian_cases.py
  - benchmarks/shared/generators/generate_dimensional_gaussian.py
  - benchmarks/shared/runners/tbs_diffusion_runner.py
  - benchmarks/shared/runners/tbs_runner.py
  - tree_break_selection/tree/optimized_branch_lengths.py
  - tree_break_selection/hierarchy_analysis/statistics/sibling_divergence/inflation_correction/empirical_null_inflation_estimation.py
tags:
  - analysis
  - benchmarks
  - diffusion
  - nnls
  - regression
---

# Adaptive NNLS Regression Skip Analysis 2026-06-28

## Summary

The full benchmark still favors `tbs_diffusion_adaptive_nnls` over the old
Hamming NN diffusion method overall across the broader metric set, not only by
ARI. On the `92` paired finite rows, adaptive NNLS improves most rows by NMI,
AMI, V-measure, macro F1, and cluster-count absolute error. The remaining
finite regressions and skips split into different causes. Most finite negative
ARI deltas against `tbs_diffusion` are already present with adaptive pydiffmap
topology and linkage branch lengths, before NNLS branch-length refitting or
normalized branch-time variance is applied. The clearest NNLS-specific finite
regression in the targeted panel is `dim_consolidated_4c_24f`.

The `kth(=6) out of bounds` skips occur before TBS traversal and before NNLS:
pydiffmap fails while estimating variable bandwidths on sparse neighbor rows.
The strict sibling-inflation skips are gate-calibration support failures: the
runner found only selected non-null positive-weight records, not strict-null or
stopped-edge empirical-null support.

## Details

### Full-run multi-metric shape

The broad metric comparison confirms that ARI is not the only evidence surface.
Across paired finite rows, adaptive NNLS improves:

- ARI on `57` rows, worsens `13`, ties `22`, mean delta `+0.359403`.
- NMI on `58` rows, worsens `12`, ties `22`, mean delta `+0.332140`.
- AMI on `58` rows, worsens `12`, ties `22`, mean delta `+0.329695`.
- V-measure on `58` rows, worsens `12`, ties `22`, mean delta `+0.332140`.
- Macro F1 on `53` rows, worsens `7`, ties `32`, mean delta `+0.356372`.
- Cluster-count absolute error on `53` rows, worsens `8`, ties `31`, mean
  delta `-0.663043` where negative is better.
- Under-splitting on `46` rows, worsens `1`, ties `45`, mean delta `-0.489130`
  where negative is better.

The unsupervised shape metrics are less uniformly available, but also do not
reverse the overall signal: silhouette improves on `28` of `51` finite paired
rows and worsens on `11`, while Davies-Bouldin improves on `24` of `51` rows
and worsens on `15`.

### Independent cluster-quality metrics

Metrics that do not use the true labels give a stricter read on which
regressions are intrinsic cluster-quality drops versus truth-alignment or
fragmentation issues.

`cat_overlap_3cat_4c` and `overlap_mod_4c_small` are broad independent-metric
regressions. They keep `4` clusters, but silhouette, Calinski-Harabasz,
effective cluster count, entropy, Davies-Bouldin, largest-cluster fraction, and
Gini all move in the wrong direction. The adaptive topology therefore changes
the boundary inside the same cluster-count regime and worsens compactness and
balance without needing truth labels to see the degradation.

`overlap_unbal_4c_small` is mostly the same pattern. It keeps `4` clusters and
silhouette, Calinski-Harabasz, effective cluster count, entropy,
largest-cluster fraction, and Gini worsen. Davies-Bouldin improves slightly, so
it is not a unanimous internal-metric drop, but the independent metrics still
support a real same-`K` boundary degradation.

`dim_consolidated_4c_24f` should not be called a broad independent-metric
regression. Against labels it worsens, and the targeted panel shows an
NNLS-specific gate effect, but independent metrics mostly improve: silhouette,
Calinski-Harabasz, effective cluster count, entropy, largest-cluster fraction,
and Gini improve, while Davies-Bouldin worsens. This is a semantic/truth
alignment failure: NNLS branch lengths reopen a `7`-cluster solution that looks
more balanced and internally structured, but less aligned with the simulated
four consolidated classes.

`cat_highd_3cat_500feat` is mixed over-fragmentation. Adaptive NNLS moves from
`14` to `68` clusters and creates `20` singleton clusters. Davies-Bouldin,
largest-cluster fraction, effective cluster count, entropy, and Gini improve
because the giant class fragments become smaller and more balanced, but
silhouette and Calinski-Harabasz worsen and the singleton rate becomes
nonzero. This is not simply "bad clusters"; it is excessive local resolution in
high-dimensional one-hot space.

`gauss_overlap_3c_small` is also mixed over-fragmentation. It moves from `4` to
`6` clusters. Effective cluster count, entropy, and largest-cluster fraction
improve, but silhouette, Calinski-Harabasz, Davies-Bouldin, and Gini worsen.
The adaptive topology creates more local coverage while reducing internal
compactness.

`gauss_overlap_8c_highd` is over-fragmentation with broad independent-metric
support for degradation. It moves from `8` to `14` clusters; silhouette,
Calinski-Harabasz, Davies-Bouldin, singleton metrics, and Gini all worsen,
while only effective cluster count and entropy improve as descriptive
fragmentation consequences.

### Diffusion backend plus NNLS panel

A focused backend panel compared Hamming NN diffusion, adaptive pydiffmap, and
graphtools diffusion on the seven regression cases. Each backend was run with
linkage branch lengths, fixed-topology NNLS branch lengths without edge
branch-time variance, and fixed-topology NNLS with normalized edge branch-time.
`graphtools` was installed through the project `experimental-gpl` extra for
this panel.

The main pattern is that branch-length refitting usually does not alter labels
when the diffusion topology is fixed. Hamming, pydiffmap, and graphtools rows
are unchanged by NNLS on `cat_overlap_3cat_4c`, `overlap_unbal_4c_small`,
`overlap_mod_4c_small`, and `gauss_overlap_8c_highd`. Pydiffmap is also
unchanged by NNLS on `cat_highd_3cat_500feat` and `gauss_overlap_3c_small`;
the over-fragmentation is already in its topology.

The NNLS-sensitive rows are concentrated in two regimes:

- `dim_consolidated_4c_24f`: NNLS increases split count across all tested
  backends. Hamming changes from `7` clusters to `8` without edge branch-time
  and `10` with edge branch-time. Pydiffmap changes from `4` to `7`. Graphtools
  changes from `5` to `9`. This confirms a branch-length-driven gate reopening
  effect on the dimensional consolidated case.
- `cat_highd_3cat_500feat`: Hamming NNLS changes `14` clusters to `22` without
  edge branch-time and `25` with edge branch-time, while pydiffmap remains
  fixed at `68` and graphtools remains fixed at `4`. Here branch lengths can
  increase fragmentation on the Hamming topology, but the larger production
  regression is still pydiffmap topology.

The graphtools backend is a useful contrast:

- It repairs `cat_highd_3cat_500feat` to `4` clusters with ARI/NMI `1.0` and
  the strongest independent compactness/balance profile in the panel.
- It matches Hamming on `gauss_overlap_8c_highd` with `8` clusters and perfect
  external metrics, while pydiffmap over-fragments to `14`.
- It is close to Hamming and better than pydiffmap on the binary overlap cases,
  but does not dominate Hamming.
- It over-fragments `gauss_overlap_3c_small` to `8` clusters and is the weakest
  backend on that row.
- On `cat_overlap_3cat_4c`, graphtools linkage under-splits to `3` clusters;
  NNLS reopens `4` clusters and improves external metrics, but Hamming linkage
  remains best overall.

This panel makes the next development targets more specific. Pydiffmap needs a
topology guard or backend selection rule for high-dimensional one-hot and
high-dimensional Gaussian overlap cases. NNLS needs a gate-flip audit on
`dim_consolidated_4c_24f` across backends. Graphtools is promising as a backend
candidate for high-dimensional categorical and high-dimensional Gaussian rows,
but not as a universal replacement.

### Finite regression shape

The full benchmark has `13` finite rows where `tbs_diffusion_adaptive_nnls`
underperforms old `tbs_diffusion` by ARI. Those rows do not all mean the same
thing under the expanded metrics.

Label-aware broad regressions, where most non-ARI external quality metrics also worsen,
include:

- `cat_overlap_3cat_4c`: NMI, AMI, V-measure, Fowlkes-Mallows, macro recall,
  macro F1, and worst-cluster recall all worsen.
- `overlap_unbal_4c_small`: the same seven non-ARI quality metrics worsen.
- `overlap_mod_4c_small`: the same seven non-ARI quality metrics worsen.
- `dim_consolidated_4c_24f`: NMI, AMI, V-measure, Fowlkes-Mallows, macro
  recall, and macro F1 worsen, but independent metrics mostly improve.

Fragmentation-sensitive or mixed regressions include:

- `cat_highd_3cat_500feat`: ARI `0.561805` to `0.098460`, with `14` clusters
  becoming `68`. NMI, AMI, V-measure, and Fowlkes-Mallows worsen, but macro F1
  is unchanged in the benchmark table, so this is mainly an over-fragmentation
  and assignment-pairing failure rather than a pure label-coverage collapse.
- `gauss_overlap_3c_small`: ARI `0.861194` to `0.674369`, with `4` clusters
  becoming `6`. NMI/AMI/V-measure/Fowlkes-Mallows worsen, but macro recall,
  macro F1, and worst-cluster recall improve slightly.
- `cat_simplex_face_rare_20cat`: ARI `0.142186` to `0.000000`, with `2`
  clusters becoming `1`. This is a broad under-split failure despite
  Fowlkes-Mallows increasing under the degenerate partition.
- `gauss_overlap_8c_highd`: ARI `1.000000` to `0.934300`, with `8` clusters
  becoming `14`; macro F1 is unchanged, while information-theoretic metrics
  worsen.

The targeted variant panel compared four variants on representative cases:
old Hamming diffusion, adaptive pydiffmap with linkage branch lengths, adaptive
pydiffmap with NNLS branch lengths but no edge branch-time variance, and the
default adaptive NNLS branch-time method. For most top regressions, the adaptive
linkage row exactly matches the adaptive NNLS rows, so NNLS is not the primary
cause. Examples:

- `cat_highd_3cat_500feat`: adaptive linkage, NNLS without edge branch-time,
  and NNLS with edge branch-time all return ARI `0.098460` and `68` clusters.
- `gauss_overlap_3c_small`: all three adaptive variants return ARI `0.674369`
  and `6` clusters.
- `cat_simplex_face_rare_20cat`: all three adaptive variants return ARI `0.0`
  and `1` cluster.
- `cat_overlap_3cat_4c`, `overlap_unbal_4c_small`, and
  `gauss_overlap_8c_highd` show the same pattern.

`dim_consolidated_4c_24f` differs. Adaptive linkage gives ARI `0.665329` and
`4` clusters, while both NNLS variants give ARI `0.529803` and `7` clusters.
Because NNLS with `edge_branch_length_variance_policy="none"` already matches
the branch-time row, the local cause is the branch-length refit entering
downstream gate calculations, not specifically the normalized edge branch-time
policy.

### How the regressing cases are simulated

`cat_highd_3cat_500feat` is generated by the categorical multinomial generator
with `200` samples, `500` categorical features, `4` balanced clusters, `3`
categories per feature, and `entropy_param=0.12`. The matrix is then one-hot
encoded to `1500` binary columns. Plain TBS reaches ARI `1.0` on this case,
which means the case is not intrinsically impossible; the regression is tied to
adaptive diffusion geometry.

`cat_overlap_3cat_4c` uses the same categorical generator with `400` samples,
`100` categorical features, `4` balanced clusters, `3` categories, and
`entropy_param=0.35`, then expands to `300` one-hot columns.

`cat_simplex_face_rare_20cat` is a method-proof categorical Dirichlet-multinomial
case with `240` samples, `60` categorical features, `4` clusters, `20`
categories, a rare-tail base distribution, and no overdispersion. It expands to
`1200` one-hot columns and is explicitly a categorical covariance-boundary
stress case.

`gauss_overlap_3c_small` and `gauss_overlap_8c_highd` are Gaussian blobs with
cluster standard deviation `3.5`; the runner median-binarizes continuous
coordinates before TBS sees them. `dim_consolidated_4c_24f` is a dimensional
Gaussian case with `12` informative dimensions, `12` noise dimensions,
separation `2.8`, and median binarization.

### Skip modes

Adaptive NNLS has `15` skips. The skip-count report groups them as:

- `kth(=6) out of bounds (1)`: `5` rows.
- `kth(=6) out of bounds (6)`: `3` rows.
- `kth(=6) out of bounds (3)`: `2` rows.
- `kth(=6) out of bounds (4)`: `1` row.
- Strict sibling-inflation support failures with `199` selected non-null
  positive-weight records: `2` rows.
- Strict sibling-inflation support failures with `39` selected non-null
  positive-weight records: `2` rows.

The pydiffmap traceback shows the `kth` failures in
`pydiffmap/kernel.py::_reduce_nn`, called from `_build_nn_kde()` during adaptive
bandwidth construction. This happens before the TBS tree is constructed and
before NNLS branch lengths are fitted. Representative skipped cases have
duplicate or tied binary rows after binarization, such as `binary_perfect_2c`
with only `5` unique rows out of `40`, and `gauss_clear_medium` with `41`
unique rows out of `60`.

The sibling-inflation skips come from
`fit_empirical_null_inflation_model()`, which deliberately refuses to fit when
there are no strict-null or stopped-edge empirical-null calibration records with
positive weight. The skipped rows are support-contract failures, not numeric
solver failures.

## Evidence

- `benchmarks/results/run_20260628_082116Z_full/full_benchmark_comparison.csv`
  records the full benchmark rows.
- `reports/adaptive_nnls_regression_analysis_20260628/finite_negative_deltas_vs_hamming_diffusion.csv`
  records the `13` finite negative deltas versus old Hamming diffusion.
- `reports/adaptive_nnls_regression_analysis_20260628/targeted_variant_diagnostics.csv`
  records the adaptive-linkage versus adaptive-NNLS isolation panel.
- `reports/adaptive_nnls_regression_analysis_20260628/multi_metric_aggregate_vs_hamming_diffusion.csv`
  records the paired aggregate win/loss/tie counts for every directional
  metric.
- `reports/adaptive_nnls_regression_analysis_20260628/ari_negative_rows_nonari_quality_context.csv`
  records the non-ARI context for every ARI-negative finite row.
- `reports/adaptive_nnls_regression_analysis_20260628/independent_metric_context_vs_hamming_diffusion.csv`
  records the independent internal and partition-shape metric directions for
  the main regression cases.
- `reports/adaptive_nnls_regression_analysis_20260628/diffusion_backend_nnls_variant_panel.csv`
  records the Hamming, pydiffmap, and graphtools backend comparison with
  linkage, NNLS without edge branch-time, and NNLS branch-time modes.
- `reports/adaptive_nnls_regression_analysis_20260628/diffusion_backend_case_best_summary.csv`
  records the best backend/mode by external and independent metric summaries.
- `reports/adaptive_nnls_regression_analysis_20260628/new_skips_while_hamming_diffusion_ok.csv`
  records the `13` rows where old Hamming diffusion was ok but adaptive NNLS
  skipped.
- `benchmarks/shared/runners/tbs_diffusion_runner.py` builds the adaptive
  pydiffmap distance before calling the TBS runner.
- `tree_break_selection/tree/optimized_branch_lengths.py` implements native
  fixed-topology NNLS branch-length fitting with `scipy.optimize.lsq_linear`.
- `tree_break_selection/hierarchy_analysis/statistics/sibling_divergence/inflation_correction/empirical_null_inflation_estimation.py`
  defines the strict sibling-inflation support checks.

## Links

- [[full-adaptive-nnls-benchmark-run-20260628]]
- [[full-adaptive-pydiffmap-benchmark-run-20260627]]
- [[scrna-branch-length-effect-audit-20260624]]
- [[hierarchy-gate-separation-20260603]]

## Open Questions

- Should adaptive pydiffmap use a preflight unique-row or sparse-neighbor support
  guard and fall back to old Hamming diffusion for tied discrete cases?
- Should the high-dimensional categorical rows use a categorical-aware adaptive
  bandwidth or a block-aware one-hot distance before topology construction?
- For `dim_consolidated_4c_24f`, which downstream gate consumes the NNLS branch
  lengths strongly enough to reopen the `4`-cluster adaptive-linkage solution
  into `7` clusters?
