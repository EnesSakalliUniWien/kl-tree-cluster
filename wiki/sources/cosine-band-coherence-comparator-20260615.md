---
title: Cosine Band Coherence Comparator 2026-06-15
type: source
status: reviewed
updated: 2026-06-15
sources:
  - raw/inbox/c2ef-cosine-subspace-method-notes-20260615.md
  - benchmarks/diagnostics/spectral/cosine_band_coherence_comparator.py
  - tests/validation/135_test_cosine_band_coherence_comparator.py
  - benchmarks/diagnostics/spectral/adaptive_cosine_kak_benchmark_probe.py
  - raw/assets/benchmark-results/cosine_band_coherence_comparator_20260615/selected_root_pass_through_null/manifest.json
  - raw/assets/benchmark-results/cosine_band_coherence_comparator_20260615/selected_root_pass_through_null/cosine_band_comparator_rows.csv
  - raw/assets/benchmark-results/cosine_band_coherence_comparator_20260615/julia_binary_sharded/merged/manifest.json
  - raw/assets/benchmark-results/cosine_band_coherence_comparator_20260615/julia_binary_sharded/merged/shard_exit_codes.csv
  - raw/assets/benchmark-results/cosine_band_coherence_comparator_20260615/julia_binary_sharded/merged/cosine_band_comparator_rows.csv
  - raw/assets/benchmark-results/cosine_band_coherence_comparator_20260615/julia_binary_sharded/merged/cosine_band_comparator_cluster_coherence.csv
  - raw/assets/benchmark-results/cosine_band_coherence_comparator_20260615/julia_binary_sharded/merged/cosine_band_comparator_spectrum.csv
tags:
  - source
  - diagnostics
  - spectral
  - cosine
  - coherence
---

# Cosine Band Coherence Comparator 2026-06-15

## Summary

`cosine_band_coherence_comparator.py` ports the useful c2ef cosine-subspace
checks into the current diagnostic framework without restoring the old scripts
as production method paths. It keeps the fixed historical cosine eigen-bands,
builds a current KL decomposition tree for each band, and reports cluster
coherence evidence.

## Key Points

- The fixed band contract is historical and predeclared:
  `common_mode_01`, `variation_02_05`, `variation_06_15`,
  `variation_16_35`, `variation_36_80`, `broad_variation_02_35`,
  `broad_variation_02_80`, and `all_modes_01_80`, truncated by available rank
  but not renamed.
- The comparator reuses the current cosine eigendecomposition and current
  gate/decomposition stack instead of reintroducing the old standalone
  decomposition path.
- Cluster coherence uses old-style enrichment summaries: Fisher exact tests per
  feature, BH correction, top-term prevalence deltas, coherent-cluster counts,
  and mean within-cluster TF-IDF cosine.
- Output rows are marked
  `diagnostic_only_not_production_calibration`; they are comparator evidence
  for subspace behavior, not a traversal null law or production promotion rule.
- The companion test verifies the fixed band contract, enriched feature-block
  detection, and diagnostic-only row output.
- On the compact selected-root/pass-through null fixture, all binary and TF-IDF
  fixed bands return one cluster. This means the fixed cosine-band comparator
  does not reproduce the selected-root/pass-through false split on that null
  fixture.
- The Julia binary matrix was run as an AWS on-demand sharded sweep, one
  `c7i.xlarge` shard per fixed band. The merged manifest records
  `all_exit_codes_zero: true`, `row_count: 8`, `coherence_row_count: 2927`,
  `spectrum_row_count: 640`, and S3 provenance under
  `s3://phylomovies-iqtree-067744548702-us-east-1/topology-search/kl-te-cosine-julia-binary-sharded-20260615-140208Z`.
- The Julia run is fragmentation-heavy. The best coherent-cluster fraction is
  `variation_36_80` with `102/292 = 0.349315`; the least singleton-heavy broad
  band is `broad_variation_02_35` with `224` clusters, singleton fraction
  `0.486607`, and coherence fraction `0.258929`; `variation_06_15` and
  `variation_16_35` are especially fragmentary with singleton fractions
  `0.875899` and `0.846743`.

## Evidence

- `tests/validation/135_test_cosine_band_coherence_comparator.py` verifies the
  fixed c2ef band contract and coherence summaries.
- The implementation cites the old scripts as source evidence but runs through
  the current diagnostic framework.
- `raw/assets/benchmark-results/cosine_band_coherence_comparator_20260615/selected_root_pass_through_null/cosine_band_comparator_rows.csv`
  records the compact null-fixture check.
- `raw/assets/benchmark-results/cosine_band_coherence_comparator_20260615/julia_binary_sharded/merged/manifest.json`
  and `shard_exit_codes.csv` record the on-demand AWS sharded Julia run and
  completion status.

## Links

- [[adaptive-cosine-kak-benchmark-probe-20260605]]
- [[overlap-conditional-topology-law-panel-20260615]]
- [[open-mathematical-questions]]
