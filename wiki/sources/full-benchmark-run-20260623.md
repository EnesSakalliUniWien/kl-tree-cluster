---
title: Full Benchmark Run 2026-06-23
type: source
status: reviewed
updated: 2026-06-23
sources:
  - benchmarks/results/run_20260623_110027Z_full_big/full_benchmark_comparison.csv
  - benchmarks/results/run_20260623_110027Z_full_big/benchmark_relationship_report.md
  - benchmarks/results/run_20260623_110027Z_full_big/failure_report.md
  - benchmarks/results/run_20260623_110027Z_full_big/full_benchmark_report.pdf
tags:
  - source
  - benchmarks
  - results
  - validation
---

# Full Benchmark Run 2026-06-23

## Summary

The canonical full benchmark was run on 2026-06-23 after the continuous tree
geometry change that routes location-style continuous Gaussian cases through
`standardized_euclidean` tree distances. The command was
`PYTHONUNBUFFERED=1 TBS_CASE_SUITE=full TBS_RUN_DIR=benchmarks/results/run_20260623_110027Z_full_big uv run python -m benchmarks.full.run`.
The run completed `121` cases and `9` methods, producing `1089` result rows in
`benchmarks/results/run_20260623_110027Z_full_big/`.

## Key Points

- Mean ARI over ok rows ranked methods as `kmeans` `0.863921`, `spectral`
  `0.833765`, `leiden` `0.816252`, `louvain` `0.812457`, `tbs` `0.763189`,
  `hdbscan` `0.600511`, `dbscan` `0.572938`, `tbs_diffusion` `0.531015`, and
  `optics` `0.524159`.
- `tbs` produced `98` ok rows and `23` skips; `tbs_diffusion` produced `107`
  ok rows and `14` skips. All non-TBS baselines produced `121` ok rows.
- The relationship report analyzed `1052` ok rows. Overall section mean ARI was
  highest for `phylogenetic` (`0.897`) and `binary` (`0.873`), and lowest for
  `sbm` (`0.178`) and `method_proof` (`0.389`).
- The standardized continuous geometry did repair the compact
  `dim_consolidated_4c_24f_continuous` default TBS case in the full pipeline
  (`K=4`, ARI `1.0`), but it did not make the continuous default gate broadly
  reliable. `gauss_clear_medium_continuous`,
  `gauss_moderate_3c_continuous`, `dim_consolidated_4c_72f_continuous`, and
  `dim_diffuse_6c_136f_continuous` all under-split to one cluster.
- The hard continuous diagnostics remain distinct regimes:
  `cont_lowrank_pggn_shrinkage` still skips for missing empirical-null support,
  `mp_spike_below_bbp_continuous` is a null exact-K row (`true K=1`, ARI
  `1.0`), and `mp_spike_above_bbp_continuous` under-splits (`K=1` vs true
  `K=2`, ARI `0.0`).
- Compared with the 2026-06-06 full-big run on common rows, the default `tbs`
  ok-row ARI change was negative on average (`-0.036171`, median `0.0` over
  `92` common ok rows). The largest TBS regressions were the continuous
  Gaussian rows that now under-split; the largest improvements were several
  phylogenetic protein/DNA rows.
- Failure diagnosis again points to root and support boundaries: low-ARI TBS
  ok rows include continuous root rejections, dimensional Gaussian mixed
  failures, hard SBM, heavy binary overlap, rare categorical/simplex cases, and
  method-proof traversal/spike rows. Root split accepted rows had mean ARI
  `0.839` versus `0.237` for rejected rows.
- Follow-up edge debugging showed that the main continuous Gaussian
  under-splits are gate-edge closures, not absent tree topology. With the
  active branch-time variance multiplier, `gauss_clear_medium_continuous`
  changes from raw-Euclidean root-open/deeper-split behavior to a
  standardized root split of `15/45` whose root sibling p-value is `0.1457`,
  so traversal stops at the root. Disabling only the tree-time multiplier in a
  throwaway process reopens the same standardized tree (`root p=8.482e-05`,
  `3` open sibling nodes, `K=4`). `gauss_moderate_3c_continuous` behaves
  similarly: active standardized root p-value is `0.1291` and `K=1`, while
  disabling only the multiplier gives root p-value `0.000375` and `K=3`.
  The normalized branch-time sibling multiplier is large in these roots
  (`6.13x` and `4.74x`), while it is modest for the repaired
  `dim_consolidated_4c_24f_continuous` root (`1.12x`).

## Evidence

- `benchmarks/results/run_20260623_110027Z_full_big/full_benchmark_comparison.csv`
  contains the complete `1089`-row result table.
- `benchmarks/results/run_20260623_110027Z_full_big/benchmark_relationship_report.md`
  records method, section, pairwise, audit-factor, correlation, and regression
  summaries for `1052` ok rows.
- `benchmarks/results/run_20260623_110027Z_full_big/failure_report.md` records
  the low-ARI default TBS failure diagnosis table.
- `benchmarks/results/run_20260623_110027Z_full_big/full_benchmark_report.pdf`
  is the merged report with cover, manifest, section pages, case plots, and
  relationship plots.

## Links

- [[full-benchmark-run-20260606]]
- [[continuous-tree-geometry-rethink-20260623]]
- [[benchmark-pipeline-contract]]
- [[root-selected-spectral-tail-law-with-legacy-overlay-20260617]]
