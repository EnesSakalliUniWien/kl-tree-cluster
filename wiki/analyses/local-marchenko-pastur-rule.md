---
title: Local Marchenko-Pastur Rule
type: analysis
status: reviewed
updated: 2026-05-26
sources:
  - kl_clustering_analysis/hierarchy_analysis/decomposition/backends/eigen/decomposition.py
  - kl_clustering_analysis/hierarchy_analysis/decomposition/backends/eigen/operators.py
  - kl_clustering_analysis/hierarchy_analysis/statistics/projection/projection_dimension_estimation/projection_dimension_estimators.py
  - kl_clustering_analysis/hierarchy_analysis/statistics/projection/spectral/spectral_decomposition_result.py
  - kl_clustering_analysis/hierarchy_analysis/statistics/projection/spectral/marchenko_pastur.py
  - kl_clustering_analysis/hierarchy_analysis/statistics/projection/spectral/tree_estimator.py
  - kl_clustering_analysis/config.py
  - benchmarks/diagnostics/spectral/compare_mp_dimension_contracts.py
  - manuscript/sections/method/edge_test.tex
  - manuscript/sections/method/assumptions_validation.tex
  - benchmarks/validation/method_constants_manifest.py
  - raw/assets/mp-dimension-rule-analysis/20260525-controlled-spectrum-summary.md
  - raw/assets/mp-dimension-rule-analysis/20260525-dimension-contract-subset.csv
  - raw/assets/mp-dimension-rule-analysis/20260525-dimension-contract-finite-null-smoke.csv
  - raw/assets/mp-dimension-rule-analysis/20260526-threshold-policy-targeted-smoke.csv
tags:
  - method
  - spectral
  - validation
---

# Local Marchenko-Pastur Rule

## Summary

The implemented Marchenko--Pastur upper-edge calculation is scale-consistent
with the covariance eigenvalues returned by the backend. The primal path
diagonalizes \(\widetilde M_u^\top \widetilde M_u / m_u\), and the dual path
diagonalizes \(\widetilde M_u\widetilde M_u^\top / m_u\). These matrices share
the same nonzero feature-covariance eigenvalues, so the implemented edge
\[
b_u^{\mathrm{MP}} = \left(1+\sqrt{d_u/m_u}\right)^2
\]
is not transposed in the high-dimensional dual branch.

That does not make the rule a validated calibration-and-power guarantee. The
calculation is correct for an ideal independent, unit-scale, local null
spectrum. The code now separates raw MP signal count, projected-Wald test
dimension, effective independent row count, and the MP threshold row count.
Production records descendant leaf rows as the effective independent row count
even when deterministic internal distributions are included for PCA directions,
but it keeps the current augmented-row MP threshold as the active method
because promoting the leaf-count threshold regressed an existing
high-cardinality categorical integration test. The remaining open layers are
the minimum projection dimension of two, finite-sample upper-edge calibration,
null-whitening scale, and data-selected PCA directions.

## Details

### What The Code Calculates

For each internal node, the spectral worker maps descendant rows into the
null-whitened tangent coordinates used by the projected-Wald tests. It then
removes constant active features, centers the rows, and eigendecomposes either
the feature covariance or the smaller dual Gram matrix. Both paths divide by
the row count \(m_u\), so the positive dual eigenvalues remain on the same
scale as the feature covariance.

The dimension estimator counts eigenvalues above
\((1+\sqrt{d_u/m_u^{\mathrm{MP}}})^2\), then floors the returned dimension to
the configured minimum. Here \(m_u^{\mathrm{MP}}\) is the row count used by the
threshold; it is recorded separately from the effective independent row count
\(m_u^{\mathrm{eff}}\). In the active Gate 2 context, the minimum dimension is
two. The raw MP signal count and the returned test dimension are therefore
different mathematical objects:
\[
k_u^{\mathrm{raw}}
=
\#\{\lambda_{u,i}>b_u^{\mathrm{MP}}\},
\qquad
k_u^{\mathrm{test}}
=
\max\{2,k_u^{\mathrm{raw}}\}.
\]

For a centered Gaussian matrix, the backend divides by the original row count
rather than by the centered rank. A scale-matched centered Gaussian edge would
be \(\frac{m_u-1}{m_u}(1+\sqrt{d_u/(m_u-1)})^2\), which is slightly lower than
the current asymptotic edge and increases finite-sample false spike rates in
controlled null probes. The more conservative unscaled \(m_u-1\) edge is not
scale-matched to the backend covariance. This reinforces that local
finite-sample calibration needs a null quantile or asymptotic fluctuation
correction, not only an \(m\) versus \(m-1\) substitution.

The code exposes these objects through a typed spectral decomposition result:
`raw_mp_signal_counts_by_node`, `test_projection_dimensions_by_node`,
`effective_independent_rows_by_node`, and `mp_threshold_rows_by_node`.
Downstream edge and sibling tests consume the test dimension; diagnostics and
validation can inspect whether the MP rule actually saw signal or only the
regularization floor, and whether the threshold used leaf-only or augmented
rows.

### What The Controlled Probe Shows

Under pure independent Gaussian noise, the raw asymptotic upper edge produced
at least one false spike in roughly 8--13% of the small local matrices checked.
Using \(m_u-1\) after centering lowered some false-spike rates, but it did not
turn the asymptotic edge into an exact finite-sample test. This is expected:
the largest noise eigenvalue has finite-sample fluctuation, and the MP edge is
an asymptotic support boundary rather than a calibrated finite-\(m_u\)
quantile.

The rank-one spiked probe behaved qualitatively as expected: weak spikes below
the high-dimensional edge were often missed, while stronger spikes were counted
reliably. This supports the rule as a sensible signal-dimension heuristic, not
as a proof of optimal power for the tree-selected projected-Wald pipeline.

The most important failure mode is internal spectral rows. When deterministic
internal distribution summaries were stacked with leaf rows and counted as
additional observations, high-dimensional null spectra could cross the MP edge
far too often. In a controlled \(n_{\mathrm{leaf}}=40,d=100\) case with about
62 augmented rows, the augmented-row threshold produced at least one null spike
in every run and an average of about 2.4 counted spikes. Counting only the leaf
row count for the threshold was much more conservative in that probe.

The other important failure mode is null-whitening scale error. For
\(n=40,d=100\), scaling pure noise by \(1.2\) made the fixed unit-scale rule
count null spikes in every run, with a mean of about five counted components. A
bulk-median scale estimator corrected that isolated scale-misspecification
probe, but it did not solve the dependent/internal-row setting. Therefore a
median-scale rule is a candidate diagnostic or validation arm, not yet a
production replacement.

### Dimension-Contract Benchmark

The representative diagnostic compared:

- internal-row PCA directions with leaf-count MP threshold;
- leaf-only spectra;
- the old augmented-row threshold;
- projection floors \(0,1,2\);
- a finite-null upper-edge smoke variant on four cases.

On the 14-case representative subset, `internal_leaf_floor2` matched the old
`internal_augmented_floor2` mean ARI and exact-\(K\) count, but exposed a much
higher zero-raw-signal fraction. This means the augmented-row threshold often
converts deterministic internal rows into apparent raw MP signal, while the
explicit contract shows that downstream success frequently comes from the
two-dimensional test floor rather than detected spectral spikes. However, a
full pytest run showed that promoting the leaf-count threshold to production
regressed `cat_highcard_20cat_4c` from the expected two clusters to one.
Therefore leaf-count thresholding remains a diagnostic arm until the
high-cardinality categorical behavior is understood.

Floor \(0\) is not currently a valid full-method benchmark. The
projected-Wald kernel correctly rejects a zero-dimensional spectral context
when the contrast vector is nonzero. A floor-zero experiment therefore needs a
new explicit zero-dimensional edge-test semantics before it can be treated as a
clustering benchmark. Floor \(1\) was runnable, but it over-split at least one
clear Gaussian case that floor \(2\) recovered exactly in the representative
subset.

The finite-null smoke with 50 null replications on four cases matched the
floor-two clustering outcomes of the current leaf-count threshold, but it is
not a validation study. A real finite-null calibration would need enough
replications and case coverage to estimate local upper-edge quantiles
reliably.

A targeted 2026-05-26 threshold-policy smoke compared the current
`internal_augmented_floor2` rule with leaf-count and finite-null alternatives
on high-cardinality categorical, dimensional Gaussian, high-dimensional binary,
and heavy-overlap binary cases. The result does not justify replacing the
active augmented-row threshold. On `binary_many_features`, the augmented,
leaf-count, and finite-null floor-two variants all recovered four clusters
with ARI \(1.0\). On `cat_highcard_20cat_4c`, the current augmented rule found
two clusters with ARI about \(0.36\), while finite-null and leaf-count rules
collapsed to one cluster. On the dimensional Gaussian cases, finite-null and
leaf-count variants also collapsed to one cluster; the current augmented rule
failed explicitly because the strict sibling calibration-support contract
found only selected non-null calibration records. That failure is a separate
post-selection calibration issue, not evidence that finite-null MP
thresholding is the correct production replacement.

Selection-aware thresholding remains a different mathematical object from the
finite-null smoke. A finite-null edge samples fixed local null matrices. A
selection-aware edge would condition on hierarchy construction, edge-gate
selection, and the selected spectral context. The current evidence only
supports keeping selection-aware MP calibration as an open research question,
not adding it as a production fallback.

## Evidence

- `projection_dimension_estimators.py` implements the fixed unit-scale MP
  upper edge and returns an explicit dimension estimate with raw count, test
  dimension, effective independent row count, and MP threshold row count.
- `spectral_decomposition_result.py` defines the typed tree-level spectral
  output contract consumed by Gate 2 and Gate 3.
- `decomposition.py` and `operators.py` show that both primal and dual
  eigendecomposition paths divide by the same row count, preserving the
  feature-covariance eigenvalue scale.
- `marchenko_pastur.py` maps rows through the null-whitened tangent chart,
  optionally stacks internal distributions for PCA directions, records the
  descendant leaf count as the effective independent row count, and records the
  row count used for the current MP threshold separately.
- `tree_estimator.py` documents the current spectral orchestration and uses
  `config.INCLUDE_INTERNAL_IN_SPECTRAL` when the caller does not override the
  internal-row policy.
- `edge_test.tex` states the MP edge and the minimum-dimension floor in the
  manuscript method.
- `assumptions_validation.tex` marks the MP rule, minimum spectral dimension,
  and internal spectral rows as validation gaps.
- `method_constants_manifest.py` lists the MP upper-edge threshold as an
  explicit method constant requiring dimension-selection, null-calibration,
  and planted-signal validation outputs.
- `20260525-controlled-spectrum-summary.md` records the controlled null,
  spiked, internal-row, and noise-scale probes from this audit.
- `compare_mp_dimension_contracts.py` runs the MP contract variants without
  adding production config.
- `20260525-dimension-contract-subset.csv` records the 14-case representative
  contract benchmark.
- `20260525-dimension-contract-finite-null-smoke.csv` records the finite-null
  smoke on four representative cases.
- `20260526-threshold-policy-targeted-smoke.csv` records the targeted
  threshold-policy smoke showing that finite-null thresholding is not a
  drop-in production replacement for the current augmented-row threshold.

## Links

- [[open-mathematical-questions]]
- [[projected-wald-statistic]]
- [[oracle-gate-path-diagnostic]]

## Open Questions

1. Should internal distribution rows remain in PCA direction estimation, or
   should production use leaf-only spectra?
2. What finite-sample upper-edge calibration, if any, should replace the
   asymptotic edge if local Type-I behavior is the target? The targeted smoke
   does not support the tested finite-null edge as a drop-in replacement.
3. Is a local bulk-scale estimator valid for Bernoulli, categorical, and
   continuous null-whitened tangent spectra, or only as a diagnostic?
4. What explicit semantics would make \(k_{\min}=0\) a valid full-method
   experiment rather than a projected-Wald contract violation?
5. How much of the remaining benchmark weakness is caused by MP dimension
   selection versus hierarchy recoverability, sibling FDR, or empirical-null
   inflation support?
