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

## Evidence

- `tests/validation/135_test_cosine_band_coherence_comparator.py` verifies the
  fixed c2ef band contract and coherence summaries.
- The implementation cites the old scripts as source evidence but runs through
  the current diagnostic framework.

## Links

- [[adaptive-cosine-kak-benchmark-probe-20260605]]
- [[overlap-conditional-topology-law-panel-20260615]]
- [[open-mathematical-questions]]
