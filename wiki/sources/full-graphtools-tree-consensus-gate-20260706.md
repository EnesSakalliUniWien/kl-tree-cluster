---
title: Full Graphtools Tree Consensus Gate 2026-07-06
type: source
status: reviewed
updated: 2026-07-06
sources:
  - benchmarks/results/run_20260706_170815Z_full_graphtools_tree_consensus/full_benchmark_comparison.csv
  - benchmarks/results/run_20260706_170815Z_full_graphtools_tree_consensus/benchmark_performance_grid_summary.csv
  - benchmarks/results/run_20260706_170815Z_full_graphtools_tree_consensus/tree_consensus_report.md
  - benchmarks/results/run_20260706_170815Z_full_graphtools_tree_consensus/tree_consensus_summary.csv
  - benchmarks/results/run_20260706_170815Z_full_graphtools_tree_consensus/tree_consensus_selection.csv
  - benchmarks/results/run_20260706_170815Z_full_graphtools_tree_consensus/tree_consensus_rankings.csv
  - benchmarks/results/run_20260706_170815Z_full_graphtools_tree_consensus/tree_consensus_pairwise_agreement.csv
  - benchmarks/results/run_20260706_170815Z_full_graphtools_tree_consensus/tree_consensus_stability.csv
  - benchmarks/results/run_20260706_170815Z_full_graphtools_tree_consensus/tree_consensus_label_assignments.csv
tags:
  - source
  - benchmarks
  - diffusion
  - graphtools
  - adaptive-k
  - tree-inference
  - consensus
  - nnls
---

# Full Graphtools Tree Consensus Gate 2026-07-06

## Summary

The frozen label-free topology selector from the adaptive-K graphtools NNLS
focus panel was run as a full benchmark gate over the `121`-case suite and the
eight existing topology cells for `tbs_diffusion_graphtools_adaptive_nnls`.
The run produced the expected `968` result rows, persisted sample labels during
the benchmark path, and wrote consensus artifacts that select one topology per
case before external labels are audited.

The hard gate status is `pass`: completeness, label integrity, label-free
invariance, paired-valid mean external metrics, median external metrics, and
category-level regression checks all pass. This supports the frozen selector as
the next scientifically validated adaptive-K graphtools NNLS topology layer,
but it is not by itself a production-default promotion for the broader method.

## Key Points

- The full run used only `tbs_diffusion_graphtools_adaptive_nnls` with the
  `graphtools_adaptive_k_tree_strategy` grid: average, complete, weighted,
  single, centroid, median, Ward, and neighbor joining.
- `full_benchmark_comparison.csv` contains `968` rows: `121` cases times `8`
  run cells. There are `121` unique cases, `8` unique run ids, and no duplicate
  `case_id`/`run_id` pairs.
- The benchmark status profile is `925` `ok` rows and `43` `skip` rows.
  Skipped topology cells remain represented in the result table and fail
  closed in the selector.
- Persisted label rows pass integrity exactly: `221,800` combined label
  assignments equal the sum of `labels_length` over `ok` rows.
- The selector produced `101` selected cases and `20`
  `skip_no_valid_topology` cases.
- Selected topology counts are weighted `45`, Ward `14`, neighbor joining
  `12`, single `7`, median `6`, centroid `6`, complete `6`, and average `5`.
- The selected rows have mean ARI `0.868327`, median ARI `1.0`, mean NMI
  `0.880812`, and mean macro F1 `0.921718`.
- Against average linkage on paired-valid rows, selected rows improve mean ARI
  from `0.838002` to `0.865667`, mean NMI from `0.861688` to `0.878405`, and
  mean macro F1 from `0.902504` to `0.920137`; medians remain tied at `1.0`.
- Against weighted linkage on paired-valid rows, selected rows improve mean
  ARI from `0.844943` to `0.868158`, mean NMI from `0.863204` to `0.880949`,
  and mean macro F1 from `0.904543` to `0.921373`; medians remain tied at
  `1.0`.
- No benchmark category with at least five paired-valid cases loses mean ARI
  against both average and weighted baselines by more than `0.02`.
- The label-free integrity check passes: blanking or permuting external
  columns does not change the selected topology output.
- The full-grid result strengthens the focus-panel conclusion that a single
  global tree topology is weaker than a frozen label-free selector over all
  accessible topology builders, followed by fixed-topology NNLS branch fitting.

## Evidence

- `benchmarks/results/run_20260706_170815Z_full_graphtools_tree_consensus/full_benchmark_comparison.csv`
  records the `968` benchmark result rows.
- `benchmarks/results/run_20260706_170815Z_full_graphtools_tree_consensus/benchmark_performance_grid_summary.csv`
  records the eight run-cell performance summaries.
- `benchmarks/results/run_20260706_170815Z_full_graphtools_tree_consensus/tree_consensus_summary.csv`
  records `gate_status = pass` and the hard-gate criteria.
- `benchmarks/results/run_20260706_170815Z_full_graphtools_tree_consensus/tree_consensus_selection.csv`
  records one selected topology per valid case and
  `skip_no_valid_topology` for cases with no valid candidate.
- `benchmarks/results/run_20260706_170815Z_full_graphtools_tree_consensus/tree_consensus_rankings.csv`
  records all candidate ranks, penalties, scores, selector status, and
  external audit metrics.
- `benchmarks/results/run_20260706_170815Z_full_graphtools_tree_consensus/tree_consensus_pairwise_agreement.csv`
  records label-free adjusted Rand agreement between topology partitions.
- `benchmarks/results/run_20260706_170815Z_full_graphtools_tree_consensus/tree_consensus_stability.csv`
  records per-case method stability summaries used as tie-breaks.
- `benchmarks/results/run_20260706_170815Z_full_graphtools_tree_consensus/tree_consensus_label_assignments.csv`
  records the persisted sample labels used for agreement analysis.
- `benchmarks/results/run_20260706_170815Z_full_graphtools_tree_consensus/tree_consensus_report.md`
  provides the human-readable pass/fail report.

## Links

- [[graphtools-adaptive-k-tree-consensus-focus-benchmark-20260630]]
- [[graphtools-adaptive-k-tree-inference-focus-benchmark-20260630]]
- [[full-graphtools-nnls-benchmark-run-20260630]]
