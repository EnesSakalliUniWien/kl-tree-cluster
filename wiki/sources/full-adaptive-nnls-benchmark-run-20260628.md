---
title: Full Adaptive NNLS Benchmark Run 2026-06-28
type: source
status: reviewed
updated: 2026-07-28
sources:
  - benchmarks/results/run_20260628_082116Z_full/full_benchmark_comparison.csv
  - benchmarks/results/run_20260628_082116Z_full/failure_report.md
  - benchmarks/shared/util/method_sets.py
  - benchmarks/shared/runners/method_registry.py
  - benchmarks/shared/runners/dispatch.py
  - benchmarks/shared/runners/tbs_diffusion_runner.py
tags:
  - source
  - benchmarks
  - diffusion
  - nnls
---

# Full Adaptive NNLS Benchmark Run 2026-06-28

## Summary

The full `121`-case benchmark was rerun after promoting
`tbs_diffusion_adaptive_nnls` into the default shared benchmark method set. The
new method uses adaptive pydiffmap topology, fixed-topology NNLS branch-length
refitting, and `edge_branch_length_variance_policy="normalized_branch_length"`.
Plots and relationship analysis were disabled for this run.

## Key Points

- The run completed `1210` rows: `121` cases across `10` methods. Results were
  written under `benchmarks/results/run_20260628_082116Z_full/`.
- Mean ARI over ok rows ranked `kmeans` `0.863921`, `spectral` `0.837000`,
  `leiden` `0.816252`, `louvain` `0.812457`, `tbs` `0.775146`,
  `tbs_diffusion_adaptive_nnls` `0.736107`, `hdbscan` `0.600511`, `dbscan`
  `0.572938`, `optics` `0.524159`, and `tbs_diffusion` `0.497085`.
- `tbs_diffusion_adaptive_nnls` produced `106` ok rows and `15` skips, compared
  with `105` ok rows and `16` skips for `tbs_diffusion`, and `93` ok rows plus
  `28` skips for plain `tbs`.
- On the `92` rows where both diffusion methods had finite ARI, adaptive NNLS
  beat Hamming NN diffusion on `57`, tied on `22`, and lost on `13`; the mean
  paired ARI delta was `+0.359403` and the median paired delta was `+0.158260`.
- Adaptive NNLS fixed the quantized Gaussian q4/q5 gap: both
  `gauss_overlap_3c_small_q4` and `gauss_overlap_3c_small_q5` reached ARI
  `1.0000`, while `tbs_diffusion` stayed at ARI `0.5698`.
- The `phylo_large` rows improved versus Hamming NN diffusion: `phylo_large_32taxa`
  reached ARI `0.9702` versus `0.6849`, and `phylo_large_64taxa` reached ARI
  `0.7652` versus `0.4531`.
- Regressions remain real: high-dimensional categorical
  `cat_highd_3cat_500feat` fell from `0.5618` to `0.0985`, and
  `gauss_overlap_3c_small` fell from `0.8612` to `0.6744`.
- The `15` adaptive-NNLS skips split between strict sibling-inflation
  calibration-support failures and small-neighborhood `kth(=6) out of bounds`
  failures.

## Evidence

- `benchmarks/results/run_20260628_082116Z_full/full_benchmark_comparison.csv`
  records all result rows.
- `benchmarks/results/run_20260628_082116Z_full/failure_report.md` records the
  failure diagnosis.
- `benchmarks/shared/util/method_sets.py` includes `tbs_diffusion_adaptive_nnls` in
  `DEFAULT_METHODS`.
- `benchmarks/shared/runners/method_registry.py` defines the adaptive NNLS method
  parameters.
- `benchmarks/shared/runners/dispatch.py` and
  `benchmarks/shared/runners/tbs_diffusion_runner.py` forward the branch-length
  variance policy and NNLS optimizer settings into the TBS runner.

## Links

- [[full-adaptive-pydiffmap-benchmark-run-20260627]]
- [[alpha-structure-sweep-20260627]]
- [[edge-gate-distance-time-contract-20260623]]
