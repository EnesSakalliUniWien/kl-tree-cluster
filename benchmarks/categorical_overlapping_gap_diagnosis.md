# Categorical And Overlapping Gap Diagnosis

Date: 2026-03-20

Reference run:
- [run_20260320_133847Z](/Users/berksakalli/Projects/kl-te-cluster/benchmarks/results/run_20260320_133847Z)
- [full_benchmark_comparison.csv](/Users/berksakalli/Projects/kl-te-cluster/benchmarks/results/run_20260320_133847Z/full_benchmark_comparison.csv)
- [benchmark_relationship_method_section_summary.csv](/Users/berksakalli/Projects/kl-te-cluster/benchmarks/results/run_20260320_133847Z/benchmark_relationship_method_section_summary.csv)
- [failure_report.md](/Users/berksakalli/Projects/kl-te-cluster/benchmarks/results/run_20260320_133847Z/failure_report.md)

## Summary

The full benchmark gap between `kl` and `leiden`/`louvain` is driven mainly by two families:

- categorical:
  - `kl` mean ARI `0.6946`
  - `leiden` mean ARI `0.9655`
  - `louvain` mean ARI `0.9655`
- overlapping:
  - `kl` mean ARI `0.6814`
  - `leiden` mean ARI `0.8381`
  - `louvain` mean ARI `0.8340`

The failure mode is mostly `KL under-splitting`, not graph-method dominance across every case.

## Main finding

The two families fail for different reasons.

### 1. Categorical gap

This is mostly a conservative early-tree split problem.

- `kl` under-splits on `7/12` categorical cases.
- `kl` over-splits on `0/12` categorical cases.
- In all `7/12` under-split categorical cases, the root raw sibling p-value is `< 0.05`.
- In `6/7`, the root raw sibling p-value is `< 0.01`.

That means the issue is usually not "no root signal." The issue is that the hierarchical sibling decision stops too early.

There are two recurring patterns:

1. Root-level non-split after correction.
2. One missed downstream split after the root opens.

Representative cases:

| Case | KL result | Root raw p | Root corrected p | Interpretation |
| --- | --- | ---: | ---: | --- |
| `cat_overlap_3cat_4c` | `1/4`, ARI `0.0` | `0.0182` | `0.4871` | Root signal exists, but corrected sibling decision is too conservative. |
| `cat_highcard_20cat_4c` | `1/4`, ARI `0.0` | `3.73e-4` | `0.0350` | Strong root raw signal, but still above `SIBLING_ALPHA = 0.01`. |
| `cat_clear_3cat_4c` | `3/4`, ARI `0.7080` | split | split | Root opens, but one later branch remains merged. |
| `cat_mod_3cat_4c` | `3/4`, ARI `0.7090` | split | split | Root opens, but node `N235` stops at corrected p `0.0636`. |

Representative downstream miss:

- `cat_mod_3cat_4c`
  - root `N238` splits
  - child `N237` splits
  - later node `N235` has raw sibling p `0.0050` but corrected p `0.0636`
  - traversal merges there, so the tree ends one split short

### 2. Overlapping gap

This family is mixed. Some misses are also conservative sibling decisions, but the hardest cases fail because `kl` does not see one clean global binary split at the root.

- `kl` under-splits on `9/25` overlapping cases.
- `kl` over-splits on `0/25`.
- Only `3/9` under-split overlapping cases have root raw sibling p-value `< 0.05`.
- Only `2/9` have root corrected sibling p-value `< 0.05`.
- One root miss is a full sibling skip.

So overlapping is not one single problem.

There are three regimes:

1. Moderate-overlap cases where raw root signal exists, but the corrected sibling decision kills the split.
2. Cases where the root opens, but one later branch remains merged.
3. Hard overlap cases where Gate 2 or the raw sibling test already sees too little global separation.

Representative cases:

| Case | KL result | Root raw p | Root corrected p | Interpretation |
| --- | --- | ---: | ---: | --- |
| `overlap_mod_4c_small` | `1/4`, ARI `0.0` | `0.00126` | `0.3940` | Root signal exists, but corrected sibling decision kills the split. |
| `overlap_mod_6c_med` | `4/6`, ARI `0.6516` | `0.00238` | `0.0155` | Root almost opens, but correction still closes it. |
| `gauss_overlap_3c_small_q5` | `2/3`, ARI `0.5698` | `1.88e-11` | `3.70e-9` | Root opens; later tree stops one split short. |
| `overlap_unbal_4c_small` | `1/4`, ARI `0.0` | `0.5320` | `0.9302` | No strong global root split signal. |
| `overlap_hd_4c_1k` | `1/4`, ARI `0.0` | `NaN` | `NaN` | Root sibling test skipped because both root children fail Gate 2. |

Important `overlap_hd_4c_1k` detail:

- root node `N998`
- both children have child-parent p-value `0.6033`
- both children are tested, not ancestor-blocked
- both are non-significant
- sibling test is skipped at the root

That is not a multiple-testing artifact. It is a failure to detect a useful root partition at all.

## Why Leiden And Louvain Win Here

`kl` is a top-down binary split method:

- Gate 2 requires at least one child-parent divergence.
- Gate 3 requires sibling difference.
- if a node fails, traversal merges the whole subtree there
- pass-through only helps when a descendant is already known to split

Relevant code:

- [config.py](/Users/berksakalli/Projects/kl-te-cluster/kl_clustering_analysis/config.py)
- [gate_evaluator.py](/Users/berksakalli/Projects/kl-te-cluster/kl_clustering_analysis/hierarchy_analysis/decomposition/gates/gate_evaluator.py)
- [tree_decomposition.py](/Users/berksakalli/Projects/kl-te-cluster/kl_clustering_analysis/hierarchy_analysis/tree_decomposition.py)

In contrast, `leiden` and `louvain` work on a k-NN graph built from the benchmark distance matrix and can recover local communities without requiring one globally certifiable binary split near the root.

Relevant code:

- [core.py](/Users/berksakalli/Projects/kl-te-cluster/benchmarks/shared/util/core.py)
- [leiden_runner.py](/Users/berksakalli/Projects/kl-te-cluster/benchmarks/shared/runners/leiden_runner.py)
- [louvain_runner.py](/Users/berksakalli/Projects/kl-te-cluster/benchmarks/shared/runners/louvain_runner.py)

This matters most when:

- local community structure exists
- but the top-down tree does not expose it as one clean binary split

## Important nuance

The graph methods are not uniformly better in the hardest overlap cases.

Examples:

- `overlap_heavy_4c_med_feat`
  - `kl`: `1/4`, ARI `0.0`
  - `leiden`: `8/4`, ARI `0.183`
  - `louvain`: `9/4`, ARI `0.137`
- `overlap_heavy_8c_large_feat`
  - `kl`: `1/8`, ARI `0.0`
  - `leiden`: `10/8`, ARI `0.032`
  - `louvain`: `12/8`, ARI `0.027`

So some of the `leiden`/`louvain` advantage comes from legitimate local recovery, but some comes from tolerating over-splitting where `kl` refuses to split.

## Working interpretation

### Categorical

Primary issue:
- overly conservative sibling split control at the upper tree

Secondary issue:
- one missed downstream split after the root opens

### Overlapping

Primary issue:
- no single global binary split in the hardest cases

Secondary issue:
- conservative corrected sibling decision in moderate-overlap cases

This means one fix is unlikely to close both gaps.

## What to inspect next

1. Categorical:
   - inspect sibling BH correction and per-tree multiplicity at the upper tree
   - especially cases where root raw p is clearly significant but corrected p blocks the split

2. Moderate overlap:
   - inspect why root raw signal is being corrected away so aggressively
   - especially `overlap_mod_4c_small` and `overlap_mod_6c_med`

3. Hard overlap:
   - inspect Gate 2 and tree construction, not only sibling calibration
   - especially `overlap_hd_4c_1k` and the heavy-overlap family

## Representative audit files

- [case_56_kl_divergence_stats.csv](/Users/berksakalli/Projects/kl-te-cluster/benchmarks/results/run_20260320_133847Z/audit/case_56_kl_divergence_stats.csv)
- [case_58_kl_divergence_stats.csv](/Users/berksakalli/Projects/kl-te-cluster/benchmarks/results/run_20260320_133847Z/audit/case_58_kl_divergence_stats.csv)
- [case_78_kl_divergence_stats.csv](/Users/berksakalli/Projects/kl-te-cluster/benchmarks/results/run_20260320_133847Z/audit/case_78_kl_divergence_stats.csv)
- [case_79_kl_divergence_stats.csv](/Users/berksakalli/Projects/kl-te-cluster/benchmarks/results/run_20260320_133847Z/audit/case_79_kl_divergence_stats.csv)
- [case_84_kl_divergence_stats.csv](/Users/berksakalli/Projects/kl-te-cluster/benchmarks/results/run_20260320_133847Z/audit/case_84_kl_divergence_stats.csv)
- [case_98_kl_divergence_stats.csv](/Users/berksakalli/Projects/kl-te-cluster/benchmarks/results/run_20260320_133847Z/audit/case_98_kl_divergence_stats.csv)
