---
title: Graphtools Adaptive K Tree Consensus Focus Benchmark 2026-06-30
type: source
status: reviewed
updated: 2026-06-30
sources:
  - reports/graphtools_adaptive_k_tree_consensus_20260630/focus_tree_consensus_report.md
  - reports/graphtools_adaptive_k_tree_consensus_20260630/focus_tree_label_free_consensus_selection.csv
  - reports/graphtools_adaptive_k_tree_consensus_20260630/focus_tree_label_free_consensus_rankings.csv
  - reports/graphtools_adaptive_k_tree_consensus_20260630/focus_tree_method_pairwise_agreement.csv
  - reports/graphtools_adaptive_k_tree_consensus_20260630/focus_tree_method_stability.csv
  - reports/graphtools_adaptive_k_tree_consensus_20260630/focus_tree_method_run_status.csv
  - reports/graphtools_adaptive_k_tree_consensus_20260630/focus_tree_method_label_assignments.csv
  - reports/graphtools_adaptive_k_tree_inference_20260630/focus_tree_inference_metrics_compact.csv
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

# Graphtools Adaptive K Tree Consensus Focus Benchmark 2026-06-30

## Summary

The seven-case adaptive-K graphtools NNLS tree-inference panel was rerun to
capture sample-level labels for all eight accessible topology strategies:
average, complete, weighted, single, centroid, median, Ward, and MAD-rooted
neighbor joining. The resulting consensus analysis measures pairwise partition
agreement without ground-truth labels and defines a label-free topology
selector that chooses a topology before external ARI, NMI, macro F1, or purity
are inspected.

The final selector uses internal fit first, not ARI: mean within-case rank of
silhouette, Calinski-Harabasz, and Davies-Bouldin, plus a small cluster-count
parsimony term and a dominant-cluster penalty for largest-cluster fraction at
least `0.45`. Partition agreement is used as a tie-break, and equivalent
partitions prefer weighted linkage over slower or collapse-prone alternatives.

## Key Points

- All `56` method-case rows completed with status `ok`, generating `21,280`
  sample-label assignments and `196` pairwise method-agreement rows.
- A first broad Borda selector was rejected because it rewarded high effective
  cluster count and very small largest-cluster fraction, which can favor
  fragmentation. It selected average linkage on `dim_consolidated_4c_24f`
  despite weaker label-free internal separation than neighbor joining.
- The final sharp selector selected complete linkage for
  `cat_overlap_3cat_4c`, average linkage for `overlap_unbal_4c_small`, Ward
  linkage for `overlap_mod_4c_small`, neighbor joining for
  `dim_consolidated_4c_24f`, and weighted linkage for
  `cat_highd_3cat_500feat`, `gauss_overlap_3c_small`, and
  `gauss_overlap_8c_highd`.
- The selected rows achieved mean ARI `0.884535`, mean NMI `0.872214`, and
  mean macro F1 `0.953596`, improving over average linkage by mean ARI
  `+0.119951` and over the best single global fixed method, weighted linkage,
  by mean ARI `+0.052847`.
- The selected rows were within mean ARI `0.001710` of the external best
  per-case topology, but the selector did not use labels to make the choices.
- `gauss_overlap_3c_small` is the clearest average-linkage topology failure:
  average fragments into `8` clusters, while weighted, single, centroid, and
  neighbor joining recover the same `3`-cluster partition. The selector chooses
  weighted because that partition is equivalent and much faster than neighbor
  joining.
- `dim_consolidated_4c_24f` favors neighbor joining by label-free internal
  evidence: best silhouette and Calinski-Harabasz, acceptable balance, and no
  fragmentation reward.
- `cat_overlap_3cat_4c` shows why internal fit alone is insufficient: single
  and neighbor joining have strong silhouette and Calinski-Harabasz but put
  about half the samples in one cluster. The dominant-cluster guard moves the
  selection to complete linkage, near the external-best centroid row.
- A single global topology is not supported by the focus panel. The defensible
  development direction is a predeclared label-free topology-selection layer
  over candidate tree builders/linkages, followed by fixed-topology NNLS branch
  fitting on the selected topology.

## Evidence

- `reports/graphtools_adaptive_k_tree_consensus_20260630/focus_tree_consensus_report.md`
  records the selector definition, selected topologies, aggregate audit, and
  recursive-analysis notes.
- `reports/graphtools_adaptive_k_tree_consensus_20260630/focus_tree_label_free_consensus_selection.csv`
  records one final selected topology per focus case and the external audit
  metrics.
- `reports/graphtools_adaptive_k_tree_consensus_20260630/focus_tree_label_free_consensus_rankings.csv`
  records all candidate ranks, penalties, scores, and external audit metrics.
- `reports/graphtools_adaptive_k_tree_consensus_20260630/focus_tree_method_pairwise_agreement.csv`
  records pairwise adjusted Rand agreement between topology-method partitions,
  independent of the benchmark truth labels.
- `reports/graphtools_adaptive_k_tree_consensus_20260630/focus_tree_method_stability.csv`
  records per-method agreement summaries used for tie-breaking.
- `reports/graphtools_adaptive_k_tree_consensus_20260630/focus_tree_method_run_status.csv`
  records the `56` rerun statuses and timing metadata.
- `reports/graphtools_adaptive_k_tree_consensus_20260630/focus_tree_method_label_assignments.csv`
  records the sample-level labels used for partition-agreement analysis.
- `reports/graphtools_adaptive_k_tree_inference_20260630/focus_tree_inference_metrics_compact.csv`
  provides the internal and external metrics audited by the consensus report.

## Links

- [[graphtools-adaptive-k-tree-inference-focus-benchmark-20260630]]
- [[graphtools-adaptive-k-nnls-focus-benchmark-20260630]]
- [[full-graphtools-nnls-benchmark-run-20260630]]
