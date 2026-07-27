---
title: scRNA Branch-Length Effect Audit 2026-06-24
type: analysis
status: draft
updated: 2026-06-24
sources:
  - applications/scrna/analysis/audit_branch_length_effects.py
  - applications/scrna/pancreas_benchmark.py
  - applications/scrna/goncalves_benchmark.py
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/method_metrics.csv
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_tree_branch_length_summary.csv
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_branch_time_sensitivity.csv
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_umap_tree_highlighting_audit.csv
  - raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/method_metrics.csv
  - raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/tbs_tree_branch_length_summary.csv
  - raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/tbs_branch_time_sensitivity.csv
  - raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/tbs_umap_tree_highlighting_audit.csv
  - raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/goncalves_tbs_relation_umap_tree_pages_ggtree.pdf
  - raw/assets/benchmark-results/scrna_branch_length_effect_audit_20260624/manifest.json
  - raw/assets/benchmark-results/scrna_branch_length_effect_audit_20260624/scrna_branch_length_effect_audit.md
  - raw/assets/benchmark-results/scrna_branch_length_effect_audit_20260624/branch_length_method_effects.csv
  - raw/assets/benchmark-results/scrna_branch_length_effect_audit_20260624/branch_length_assignment_similarity.csv
  - raw/assets/benchmark-results/scrna_branch_length_effect_audit_20260624/branch_time_sensitivity_combined.csv
tags:
  - scrna
  - pancreas
  - branch-length
  - clustering
---

# scRNA Branch-Length Effect Audit 2026-06-24

## Summary

The rerun supports the adaptive-diffusion TBS tree as the current review
surface for the adult and Goncalves scRNA analyses. Branch lengths are not how
cells are placed on the tree. The benchmark first infers a tree topology from a
distance matrix, then stores or refits branch lengths on that fixed topology.

Branch lengths change clustering only when the edge gate uses the explicit
`normalized_branch_length` variance policy. Under that policy, standardized-PCA
linkage is unstable in the current scRNA runs, while adaptive-diffusion topology
is stable.

## Details

### How the tree is inferred

The benchmark builds a PCA representation from the current scRNA workflow. The
standardized-PCA tree computes pairwise Euclidean distances in PCA space and
passes them to average-linkage hierarchical clustering. The adaptive-diffusion
tree builds a variable-bandwidth diffusion distance from the PCA kNN graph
(`k=15`, diffusion time `3`) and passes that distance matrix to the same
average-linkage tree builder.

Linkage heights are diagnostic branch lengths. In NNLS branch-time rows, the
topology is held fixed and non-negative least squares refits edge lengths to
better approximate continuous pairwise distances. TBS then traverses the fixed
tree with child-parent edge gates and sibling gates. Branch lengths enter the
edge gate only when
`edge_branch_length_variance_policy="normalized_branch_length"` is enabled.

### Adult pancreas branch-length effect

For adult pancreas, standardized-PCA topology-only and NNLS branch-time both
return `45` TBS clusters with identical assignments. The raw-linkage
branch-time diagnostic collapses that standardized-PCA row to `9` clusters.

The adaptive-diffusion tree is stable: topology-only and NNLS branch-time both
return `43` clusters with identical assignments, while raw-linkage branch-time
returns `41` clusters and remains assignment-close to topology-only
(`ARI = 0.9970`).

### Goncalves branch-length effect

For Goncalves fetal pancreas, standardized-PCA topology-only returns `40` TBS
clusters. Both standardized-PCA branch-time rows collapse to `1` cluster, with
zero assignment agreement against the topology-only clustering at the cluster
partition level.

The adaptive-diffusion tree is stable for the Goncalves progenitor review:
topology-only, NNLS branch-time, and raw-linkage branch-time all return the same
`24` clusters with identical assignments. This is the tree used by the
progenitor UMAP/tree relation PDF.

### Placement and plotting

The UMAP/tree highlighting audits support correct placement of single-cell
assignments on the plotted trees. The adult UMAP/tree audit reports `226/226`
colored cluster clades as exact, and the Goncalves audit reports `114/114`.
This means the confusing interpretation came from mixed biological subtrees and
earlier plot layout/alias issues, not from assigning cluster colors to the wrong
tree leaves.

The current full Goncalves relation review artifact is
`goncalves_tbs_relation_umap_tree_pages_ggtree.pdf`, which cross-references TBS
clusters, fetal populations, progenitor fraction, and progenitor state against
the adaptive-diffusion topology tree.

## Evidence

- `applications/scrna/analysis/audit_branch_length_effects.py` joins the adult and Goncalves
  method metrics, branch-length summaries, assignment tables, and branch-time
  sensitivity scans into the branch-length audit output folder.
- `raw/assets/benchmark-results/scrna_branch_length_effect_audit_20260624/scrna_branch_length_effect_audit.md`
  records the rerun summary, tree-inference steps, branch-length effect table,
  and branch-time sensitivity table.
- `raw/assets/benchmark-results/scrna_branch_length_effect_audit_20260624/branch_length_method_effects.csv`
  records the adult standardized-PCA collapse from `45` to `9` clusters under
  raw-linkage branch-time, the Goncalves standardized-PCA collapse from `40` to
  `1` cluster under both branch-time rows, and the adaptive-diffusion stability
  rows.
- `raw/assets/benchmark-results/scrna_branch_length_effect_audit_20260624/branch_length_assignment_similarity.csv`
  records identical adaptive-diffusion topology/NNLS assignments for both
  datasets and identical Goncalves adaptive-diffusion assignments across all
  three branch-length variants.
- `raw/assets/benchmark-results/scrna_branch_length_effect_audit_20260624/branch_time_sensitivity_combined.csv`
  records the broader edge-gate variance scaling scan, including adaptive rows
  that keep nearly the same cluster pattern under moderate scaling and
  standardized-PCA rows that are more sensitive.
- `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_umap_tree_highlighting_audit.csv`
  reports `226/226` adult colored TBS cluster clades as exact.
- `raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/tbs_umap_tree_highlighting_audit.csv`
  reports `114/114` Goncalves colored TBS cluster clades as exact.
- Regenerating the adult and Goncalves plot manifests with
  `applications/scrna/plot_pipeline.py --dataset <dataset> --strict` completed
  successfully on 2026-06-24. The adult manifest records `66` present plot
  files plus `54` older orphaned files, and the Goncalves manifest records
  `66` present plot files plus `8` orphaned q50 action/split-action diagnostic
  files. Both manifest JSON files now carry a `generated_at` timestamp.

## Links

- [[pancreas-scrna-clustering-benchmark-20260623]]
- [[goncalves-pancreas-progenitor-benchmark-prep-20260624]]
- [[goncalves-tbs-progenitor-analysis-20260624]]
- [[scrna-plot-pipeline-audit-20260624]]

## Open Questions

- Should normalized branch-time variance remain a diagnostic mode for scRNA
  until a stochastic-time interpretation is justified?
- Should the standardized-PCA branch-time collapse be treated as an argument to
  remove that row from summary plots, or should it remain visible as a
  sensitivity warning?
