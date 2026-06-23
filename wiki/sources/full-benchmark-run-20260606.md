---
title: Full Benchmark Run 2026-06-06
type: source
status: reviewed
updated: 2026-06-07
sources:
  - benchmarks/results/run_20260605_224254Z_full_big/full_benchmark_comparison.csv
  - benchmarks/results/run_20260605_224254Z_full_big/benchmark_relationship_report.md
  - benchmarks/results/run_20260605_224254Z_full_big/failure_report.md
  - benchmarks/results/run_20260605_224254Z_full_big/full_benchmark_report.pdf
  - benchmarks/results/diagnostics/clustering_debug_20260606_corrected_root_audit/clustering_debug_report.md
tags:
  - source
  - benchmarks
  - results
  - validation
---

# Full Benchmark Run 2026-06-06

## Summary

The canonical full benchmark was rerun on 2026-06-06 with the default `full`
suite, default methods, plots, relationship analysis, and failure diagnosis.
The run completed `120` cases and `9` methods, producing `1080` result rows in
`benchmarks/results/run_20260605_224254Z_full_big/`.

## Key Points

- Mean ARI over ok rows ranked methods as `kmeans` `0.862787`,
  `tbs_diffusion` `0.850871`, `spectral` `0.832380`, `tbs` `0.817390`,
  `leiden` `0.813803`, `louvain` `0.810504`, `hdbscan` `0.612964`, `dbscan`
  `0.582780`, and `optics` `0.538905`.
- Status counts match the previous full-run profile: `tbs` had `92` ok rows and
  `28` skips; `tbs_diffusion` had `104` ok rows and `16` skips; all other
  methods produced `120` ok rows.
- Section mean ARI over ok rows was highest for `binary` (`0.947522`) and
  `phylogenetic` (`0.904700`), and lowest for `sbm` (`0.191355`) and
  `method_proof` (`0.395514`).
- For `tbs`, section mean ARI over ok rows was `0.986022` on binary,
  `0.953169` on overlapping, `0.872292` on categorical, `0.860555` on
  phylogenetic, `0.728018` on Gaussian, `0.380551` on SBM, and `0.285714` on
  method-proof cases.
- Most `tbs` skips were strict calibration-support failures: selected non-null
  positive-weight records were present, but no strict-null or stopped-edge
  empirical-null positive-weight calibration records were available. Two dense
  continuous cases hit explicit covariance memory/dimension contracts.
- On 2026-06-07 the canonical derived reports in this run directory were
  regenerated after the root-audit analyzer fix. The root sibling decision is
  now read from the root audit row, matching
  [[clustering-root-audit-debug-20260606]].
- Under the corrected root-row audit, TBS accepted-root ok rows have much higher
  mean ARI (`0.901264`) than TBS rejected-root ok rows (`0.307692`), but four
  low-ARI TBS rows are accepted-root post-root sibling traversal stalls rather
  than root rejections.

## Evidence

- `benchmarks/results/run_20260605_224254Z_full_big/full_benchmark_comparison.csv`
  contains the full `1080`-row result table.
- `benchmarks/results/run_20260605_224254Z_full_big/benchmark_relationship_report.md`
  contains regenerated method, section, pairwise, correlation, regression, and
  corrected audit-factor summaries.
- `benchmarks/results/run_20260605_224254Z_full_big/failure_report.md`
  contains the regenerated TBS low-ARI failure diagnosis table.
- `benchmarks/results/run_20260605_224254Z_full_big/full_benchmark_report.pdf`
  is the regenerated merged report with cover, manifest, section pages, case
  plots, and corrected relationship plots.
- `benchmarks/results/diagnostics/clustering_debug_20260606_corrected_root_audit/clustering_debug_report.md`
  contains the corrected root-row audit interpretation and post-root-stall
  summary.

## Links

- [[full-benchmark-run-20260605]]
- [[benchmark-pipeline-contract]]
- [[clustering-root-audit-debug-20260606]]
- [[selected-hierarchy-null-support-contract]]
- [[barycentric-action-equation-diagnostic-20260606]]
