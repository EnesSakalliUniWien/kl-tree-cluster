---
title: Full Benchmark Run 2026-06-05
type: source
status: reviewed
updated: 2026-06-05
sources:
  - benchmarks/results/run_20260605_084136Z_full/full_benchmark_comparison.csv
  - benchmarks/results/run_20260605_084136Z_full/benchmark_relationship_report.md
  - benchmarks/results/run_20260605_084136Z_full/failure_report.md
tags:
  - source
  - benchmarks
  - results
  - validation
---

# Full Benchmark Run 2026-06-05

## Summary

The canonical full benchmark completed on 2026-06-05 with `120` cases and `9`
methods, producing `1080` result rows. The run includes the historical full
suite plus the new `method_proof` cases. Post-run relationship analysis,
failure diagnosis, section pages, and the merged PDF report were generated in
`benchmarks/results/run_20260605_084136Z_full/`.

## Key Points

- Mean ARI over ok rows ranked methods as `kmeans` `0.8628`,
  `kl_diffusion` `0.8509`, `spectral` `0.8324`, `kl` `0.8174`, `leiden`
  `0.8138`, `louvain` `0.8105`, `hdbscan` `0.6130`, `dbscan` `0.5828`, and
  `optics` `0.5389`.
- `kl` produced `92` ok rows and `28` skips; most skips came from the strict
  empirical-null support contract rejecting selected non-null positive-weight
  calibration records. `kl_diffusion` produced `104` ok rows and `16` skips.
- Section mean ARI over ok rows was highest for `binary` (`0.9475`) and
  `phylogenetic` (`0.9047`), and lowest for `sbm` (`0.1914`) and
  `method_proof` (`0.3955`).
- For `kl` specifically, section mean ARI over ok rows was `0.9860` on
  binary, `0.9532` on overlapping, `0.8723` on categorical, `0.8606` on
  phylogenetic, `0.7280` on Gaussian, `0.3806` on SBM, and `0.2857` on
  method-proof cases.
- The method-proof subset behaved as intended as a stress suite rather than an
  ARI leaderboard: rare categorical, low-rank `p >> n`, Brownian phylogenetic
  null, above-BBP spike, and traversal cases exposed under-splitting,
  unsupported covariance, or support-contract skips.
- Failure diagnosis for low-ARI `kl` ok rows identified root-split rejection
  as the dominant mode for dimensional Gaussian, hard SBM, and several
  method-proof cases.

## Evidence

- `benchmarks/results/run_20260605_084136Z_full/full_benchmark_comparison.csv`
  contains the complete 1080-row result table.
- `benchmarks/results/run_20260605_084136Z_full/benchmark_relationship_report.md`
  records method, section, pairwise, correlation, and regression summaries.
- `benchmarks/results/run_20260605_084136Z_full/failure_report.md` records the
  KL low-ARI failure diagnosis table.
- `benchmarks/results/run_20260605_084136Z_full/full_benchmark_report.pdf`
  is the merged report with cover, manifest, section pages, case plots, and
  relationship plots.

## Links

- [[traceable-mathematical-benchmark-suite]]
- [[benchmark-pipeline-contract]]
- [[selected-hierarchy-null-support-contract]]
- [[hierarchy-gate-separation-20260603]]
