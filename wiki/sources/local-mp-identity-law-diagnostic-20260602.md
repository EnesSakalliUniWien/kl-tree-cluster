---
title: Local MP Identity Law Diagnostic 2026-06-02
type: source
status: reviewed
updated: 2026-06-03
sources:
  - benchmarks/diagnostics/spectral/local_mp_identity_law_diagnostic.py
  - tests/validation/54_test_local_mp_identity_law_diagnostic.py
  - raw/assets/benchmark-results/local_mp_identity_law_20260602_representative/manifest.json
  - raw/assets/benchmark-results/local_mp_identity_law_20260602_representative/case_summary.csv
  - raw/assets/benchmark-results/local_mp_identity_law_20260602_representative/node_spectrum.csv
  - raw/assets/benchmark-results/local_mp_identity_law_20260603_selected_tree_spectral_law/manifest.json
  - raw/assets/benchmark-results/local_mp_identity_law_20260603_selected_tree_spectral_law/case_summary.csv
  - raw/assets/benchmark-results/local_mp_identity_law_20260603_selected_tree_spectral_law/node_spectrum.csv
  - raw/assets/benchmark-results/local_mp_identity_law_20260603_selected_tree_spectral_law/spectral_law_relationships.csv
  - raw/assets/benchmark-results/local_mp_identity_law_20260603_selected_tree_spectral_law/categorical_extreme_nodes.csv
tags:
  - source
  - spectral
  - validation
  - mp
---

# Local MP Identity Law Diagnostic 2026-06-02

## Summary

This diagnostic tests whether the production node-local null-whitened tangent
spectra look like the identity-population Marchenko--Pastur law used by the
current MP edge
\[
\lambda_+=(1+\sqrt{d/m})^2.
\]
It materializes the same production node matrices used by the spectral worker,
computes eigenvalues, compares positive eigenvalue quantiles with the
conditional identity-MP positive spectrum, and records descriptive departures.
It also reports a centered self-whitening reference eigenvalue
\[
(m-1)/m
\]
to test the continuous empirical-Gaussian path, where each node's rows are
whitened by that same node's empirical covariance before eigendecomposition.
The regenerated table also reports a finite-sample identity-null top-eigenvalue
quantile, using simulated centered Gaussian matrices with the same row count,
active feature count, and backend \(1/m\) covariance scale. This separates
ordinary finite-size top-edge fluctuation from selected-hierarchy spectral
inflation. It remains diagnostic-only and is not a production threshold.
The 2026-06-03 rerun adds explicit selected-tree spectral-law coordinates:
node-size fraction, aspect ratio, row count, active feature count, and
\(\log(\lambda_{\max}/q_{0.95}^{\mathrm{finite\ identity}})\). It also writes
categorical extreme nodes into a separate table so categorical failures are
not mixed with Bernoulli/discretized selected spectral inflation.

The result is not a production threshold change. It does not estimate a
deformed-MP edge, does not add bootstrap, and does not validate selected
sibling calibration. It is a screen for whether \(H=\delta_1\) is a plausible
local population-spectrum model after null whitening.

## Key Points

- The representative run evaluated seven cases:
  `gauss_null_large`, `binary_many_features`, `cat_highcard_20cat_4c`,
  `cat_highd_3cat_500feat`, `dim_diffuse_6c_136f`,
  `gauss_null_large_continuous`, and `dim_diffuse_6c_136f_continuous`.
- Bernoulli/discretized Gaussian selected spectra frequently sit at or above
  the identity MP edge. `gauss_null_large` has raw MP spikes in about `59%` of
  evaluated nodes, `binary_many_features` in about `75%`, and
  `dim_diffuse_6c_136f` in about `75%`.
- Those same cases also exceed the finite-sample identity-null 95% top-edge
  envelope far more often than ordinary finite fluctuation should:
  `gauss_null_large` in about `54.5%` of evaluated nodes,
  `binary_many_features` in about `72.1%`, and `dim_diffuse_6c_136f` in about
  `70.5%`. This supports selected-hierarchy spectral inflation rather than a
  simple finite-sample MP fluctuation explanation.
- The median top-eigenvalue-over-MP-edge ratios for those same three cases are
  close to but above one: about `1.017`, `1.018`, and `1.042`. This is
  consistent with selected-tree spectra living near the MP boundary rather
  than behaving like an unselected identity-null sample covariance.
- High-cardinality categorical spectra are mixed. `cat_highcard_20cat_4c` has
  raw spikes in only about `8.5%` of evaluated nodes and median top-edge ratio
  about `0.803`, but its q95 top-edge ratio is about `1.126`. The
  high-dimensional categorical case has median top-edge ratio about `0.975`
  and raw spikes in about `13.3%` of evaluated nodes, with a few extreme
  selected root/large-node spectra.
- Against the same finite identity-null 95% envelope, the categorical cases are
  much closer to ordinary finite fluctuation: about `7.0%` of evaluated nodes
  exceed it in `cat_highcard_20cat_4c`, and about `12.0%` in
  `cat_highd_3cat_500feat`. The categorical open problem is therefore more
  localized: a few selected extremes and covariance/projection/FDR interactions,
  not blanket spectral inflation across most evaluated nodes.
- The selected-tree covariate table shows a strong node-size/aspect-ratio
  relationship in `cat_highcard_20cat_4c`: for the log top-eigenvalue ratio
  over the finite identity-null 95% edge, \(R^2\) is about `0.79` for
  log node-size fraction and about `0.87` for log aspect ratio. The analogous
  `cat_highd_3cat_500feat` relationships are much weaker, with \(R^2\) about
  `0.17` and `0.21`.
- The categorical extreme-node table separates the selected-node geometry.
  `cat_highd_3cat_500feat` has root and half-tree extremes with ratios around
  `9.54`, `7.58`, and `3.81` over the finite identity-null 95% edge, followed
  by many two- or three-row extreme nodes with ratios around `1.0`--`1.4`.
  `cat_highcard_20cat_4c` has fewer and smaller extremes, led by the root at
  about `1.34` and selected large internal nodes around `1.28`, `1.17`, and
  `1.12`.
- Continuous empirical-covariance cases do not look identity-MP-like because
  they follow a different finite-rank self-whitening geometry. Both continuous
  cases have zero raw MP spikes and median top-edge ratios far below one:
  about `0.081` for `gauss_null_large_continuous` and `0.048` for
  `dim_diffuse_6c_136f_continuous`. The positive-spectrum KS distance against
  identity MP is `1.0`.
- The same continuous rows exactly match the centered self-whitening reference:
  the median top-eigenvalue ratio and the median positive-eigenvalue ratio
  against \((m-1)/m\) are both `1.0` in both continuous cases. This explains
  the apparent identity-MP failure without invoking a new clustering effect.
- This continuous result is a contract warning, not a benchmark fix. A node
  spectrum whitened by its own empirical covariance is a finite-rank hat-matrix
  object, not a fresh identity-MP sample covariance object. The next
  mathematical object is an effective/deformed local spectral law, not a
  bootstrap threshold fallback.

## Evidence

- `benchmarks/diagnostics/spectral/local_mp_identity_law_diagnostic.py`
  implements the screen and writes `case_summary.csv`, `node_spectrum.csv`, and
  `manifest.json`.
- `tests/validation/54_test_local_mp_identity_law_diagnostic.py` validates the
  closed-form MP support, numerical positive-spectrum quantiles, centered
  self-whitening scale, strong-spike detection, and insufficient-row status.
- `raw/assets/benchmark-results/local_mp_identity_law_20260602_representative/manifest.json`
  records the diagnostic role, case names, limitations, code commit, and dirty
  worktree status.
- `raw/assets/benchmark-results/local_mp_identity_law_20260602_representative/case_summary.csv`
  records case-level identity-MP, finite identity-null, and self-whitening
  departure summaries.
- `raw/assets/benchmark-results/local_mp_identity_law_20260602_representative/node_spectrum.csv`
  records node-level spectra, MP edges, finite identity-null top-edge quantile
  ratios, self-whitening ratios, and diagnostic statuses.
- `raw/assets/benchmark-results/local_mp_identity_law_20260603_selected_tree_spectral_law/spectral_law_relationships.csv`
  records selected-tree covariate relationships against the log finite-null
  top-edge exceedance target, including explicit insufficiency and
  constant-covariate statuses.
- `raw/assets/benchmark-results/local_mp_identity_law_20260603_selected_tree_spectral_law/categorical_extreme_nodes.csv`
  records the categorical nodes exceeding the finite identity-null top-edge
  quantile.

## Links

- [[local-marchenko-pastur-rule]]
- [[selected-geometry-mp-integral-literature-20260602]]
- [[selected-hierarchy-geometric-law-map]]
- [[open-mathematical-questions]]
