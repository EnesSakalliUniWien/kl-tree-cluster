---
title: Pancreas TBS Inner Node Progenitor Comparison 2026-06-24
type: source
status: reviewed
updated: 2026-06-24
sources:
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_adaptive_inner_node_lineage_summary.csv
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_adaptive_inner_node_progenitor_review.csv
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_adaptive_terminal_cluster_root_review.csv
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_adaptive_terminal_cluster_inner_node_summary.csv
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_adaptive_two_three_cluster_junction_review.csv
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_adaptive_two_three_cluster_junction_child_summary.csv
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_adaptive_two_three_cluster_junction_mixed_umap.png
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_adaptive_two_three_cluster_junction_mixed_umap.pdf
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_adaptive_monophyletic_subtree_meeting_review.csv
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_adaptive_monophyletic_subtree_meeting_child_summary.csv
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_adaptive_monophyletic_subtree_meeting_mixed_umap.png
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_adaptive_monophyletic_subtree_meeting_mixed_umap.pdf
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_adaptive_inner_node_marker_means.csv
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/pancreas_progenitor_signature_definitions.csv
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/pancreas_progenitor_signature_dataset_summary.csv
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/pancreas_full_neurog3_positive_cells.csv
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_adaptive_monophyletic_meeting_progenitor_signature_comparison.csv
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_adaptive_monophyletic_meeting_child_progenitor_signature_comparison.csv
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/pancreas_benchmark_celltype_progenitor_signature_scores.csv
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/pancreas_full_celltype_progenitor_signature_scores.csv
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_adaptive_cluster_progenitor_signature_scores.csv
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/pancreas_progenitor_signature_summary_rows.csv
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/pancreas_progenitor_signature_comparison.png
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/pancreas_progenitor_signature_comparison.pdf
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/pancreas_progenitor_signature_umap.png
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/pancreas_progenitor_signature_umap.pdf
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_adaptive_inner_node_progenitor_comparison.png
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_adaptive_inner_node_progenitor_comparison.pdf
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/method_assignments.csv
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_adaptive_diffusion_topology_projected_adaptive_k90_alpha0p01_edge0p001_tree_edges.csv
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/pancreas_classical_pipeline.h5ad
  - applications/scrna/analysis/analyze_pancreas_inner_nodes.py
  - applications/scrna/analysis/compare_pancreas_progenitor_signatures.py
tags:
  - source
  - benchmark
  - scrna
  - pancreas
  - progenitor
  - tree
---

# Pancreas TBS Inner Node Progenitor Comparison 2026-06-24

## Summary

This analysis compares internal nodes from the adaptive-diffusion TBS pancreas
tree with coarse pancreatic lineage/progenitor expectations. Internal nodes are
hierarchy nodes, not observed cells. The review therefore treats progenitor
language as a lineage-consistency comparison and checks descendant cell-type
composition plus marker means rather than claiming biological ancestry.

## Key Points

- The adaptive-diffusion TBS tree has `2,499` internal nodes. Among nodes with
  at least `50` descendant leaves, `61` are mature cell-type/subcluster nodes,
  `21` are local geometry nodes, `17` are broad mixed-pancreas nodes, and `4`
  are endocrine mixed descendant nodes.
- The main lineage-coherent shared ancestor is `N4963`, with `947` descendants
  across `5` final clusters. Its composition is `beta:463`, `alpha:418`,
  `delta:34`, `gamma:24`, plus small trace non-endocrine counts. This looks
  like an endocrine/islet super-node rather than a validated progenitor.
- `N4963` has `NEUROG3` mean `0.0`; its descendant marker means are dominated
  by mature hormone markers (`INS`, `GCG`, `SST`, `PPY`) rather than a
  progenitor program. The correct call is lineage-consistent hierarchy ancestor,
  not progenitor cell state.
- The other endocrine mixed nodes are within one final TBS cluster:
  `N4921` is beta-dominant with delta/gamma traces, while `N4881` and `N4789`
  are delta/gamma/epsilon endocrine-neighbor nodes. They also have `NEUROG3`
  mean `0.0`.
- A terminal-cluster-root pass checked every final adaptive TBS cluster root.
  All `43` final clusters have an internal terminal clade root. Root groupings
  are `33` local geometry nodes, `9` mature cell-type nodes, and `1` endocrine
  mixed descendant node. The only terminal root with an ancestor-like mixed
  endocrine composition is cluster `36`, node `N4881`, with `119` descendants
  (`delta:85`, `gamma:31`, `epsilon:3`) and `NEUROG3 = 0.0`; it remains a
  lineage-consistent endocrine-neighbor node, not a validated progenitor.
- Inside terminal cluster subtrees, the only mixed endocrine internal nodes that
  survive the review are `N4789` and `N4881` in cluster `36`, plus `N4921`
  inside beta-dominant cluster `42`. None carries progenitor marker support.
- A direct junction pass then restricted the review to internal nodes where
  exactly two or three final TBS clusters meet. There are `10` such direct
  junctions, and `5` are compositionally mixed. All five mixed junctions are
  small (`19` to `43` cells), have `NEUROG3 = 0.0`, and are better interpreted
  as local mature-state or boundary contacts than progenitor states:
  `N4957` (`C14,C15,C16`; `alpha:35`, `beta:5`, `gamma:2`, `ductal:1`),
  `N4964` (`C7,C8`; `beta:22`, `alpha:3`, `gamma:2`, `delta:1`), `N4922`
  (`C15,C16`; `alpha:12`, `beta:5`, `gamma:2`, `ductal:1`), `N4965`
  (`C30,C31`; `PSC:10`, `mesenchyme:4`, `ductal:4`, `beta:2`), and `N4857`
  (`C38,C39`; `gamma:15`, `alpha:3`, `beta:1`).
- The mixed junction UMAP panel shows these two/three-cluster contacts are
  localized neighborhoods rather than broad bridging structures across the
  embedding.
- A stricter monophyletic-subtree meeting pass compresses the final cluster
  tree and keeps only internal nodes whose immediate child branches are exact
  unions of complete final TBS clusters. It finds `42` such meetings, as
  expected for a binary tree over `43` terminal clusters.
- Most exact monophyletic-subtree meetings are ladder-like broad context joins,
  where one terminal clade or small subtree is split from the remaining trunk.
  The focused mixed, non-ladder set is `N4963`, `N4917`, `N4957`, `N4964`,
  `N4922`, `N4965`, and `N4857`.
- `N4963` is the only large balanced mixed monophyletic-subtree meeting:
  `C38,C39,C40,C41` versus `C42`, `947` cells total, `beta:463`,
  `alpha:418`, `delta:34`, and `gamma:24`. It is endocrine-lineage coherent
  (`99.3%` endocrine) but still has `NEUROG3 = 0.0`, so it remains a hierarchy
  ancestor over mature endocrine states, not a progenitor cell state.
- The other focused mixed monophyletic-subtree meetings are either the same
  small direct junctions from the two/three-cluster pass or `N4917`, a mostly
  alpha endocrine branch with a small gamma-rich child. They do not add
  marker-supported progenitor evidence.
- A progenitor signature pass compared the focused monophyletic-subtree
  meetings, child branches, final TBS clusters, benchmark cell types, and full
  AnnData cell types against strict endocrine-progenitor, endocrine-commitment,
  trunk/ductal, tip/acinar, and mature-state signatures.
- The full AnnData contains only `5/14,693` `NEUROG3+` cells. The `2,500`-cell
  benchmark subset used for the TBS tree contains `0/2,500` `NEUROG3+` cells.
  Therefore, no TBS internal node can be a sampled `NEUROG3+` endocrine
  progenitor population in this benchmark.
- Compared with the rare full-AnnData `NEUROG3+` reference cells, `N4963` has
  no core endocrine-progenitor signal: `NEUROG3` mean `0.0`, `0%`
  `NEUROG3+`, endocrine-progenitor-core score `-0.049`, and low endocrine
  commitment score `-0.314`.
- The small focused junctions `N4957`, `N4964`, and `N4922` score high for
  `PAX4` and endocrine-commitment factors, but also high for mature endocrine
  hormones and still have `NEUROG3 = 0.0`; they are endocrine-committed or
  mature-state neighborhoods, not supported progenitor cells.
- Benchmark ductal cells score high for trunk/ductal markers, while acinar
  cells score high for tip/acinar markers. These are expected adult-state
  overlaps with developmental marker panels and are not sufficient evidence for
  progenitor identity.
- No convincing ductal-acinar progenitor node remains after requiring meaningful
  fractions from both ductal and acinar descendants. Earlier trace ductal
  contamination inside acinar-rich nodes should not be interpreted as a
  progenitor signal.
- Root and near-root nodes are broad mixed-pancreas geometry with endocrine,
  exocrine, stromal, endothelial, and immune descendants. They are too broad to
  be progenitor-specific.
- The conclusion is that TBS internal nodes capture useful lineage hierarchy
  structure, especially endocrine grouping, but the current adult pancreas
  benchmark does not validate internal nodes as biological progenitors.

## Evidence

- `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_adaptive_inner_node_lineage_summary.csv`
  records every adaptive-diffusion TBS internal node, descendant counts, final
  cluster coverage, cell-type composition, lineage category, and interpretation.
- `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_adaptive_inner_node_progenitor_review.csv`
  records the concise review nodes and marker means.
- `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_adaptive_terminal_cluster_root_review.csv`
  records one row per final adaptive TBS cluster, its exact terminal clade root,
  internal-node status, cell-type composition, marker means, and progenitor call.
- `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_adaptive_terminal_cluster_inner_node_summary.csv`
  records reviewed internal nodes inside each final cluster subtree.
- `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_adaptive_two_three_cluster_junction_review.csv`
  records the direct two/three-final-cluster junction nodes, including
  composition, marker means, and progenitor call.
- `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_adaptive_two_three_cluster_junction_child_summary.csv`
  records the immediate child branches for each direct junction.
- `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_adaptive_two_three_cluster_junction_mixed_umap.png`
  highlights the mixed direct junction cells on the benchmark UMAP.
- `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_adaptive_monophyletic_subtree_meeting_review.csv`
  records all exact meetings where immediate child branches are unions of
  complete final TBS clusters, including ladder flags and focused mixed
  junction flags.
- `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_adaptive_monophyletic_subtree_meeting_child_summary.csv`
  records child-branch composition for those exact monophyletic-subtree
  meetings.
- `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_adaptive_monophyletic_subtree_meeting_mixed_umap.png`
  highlights the focused mixed exact monophyletic-subtree meetings on the
  benchmark UMAP.
- `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_adaptive_inner_node_progenitor_comparison.png`
  plots internal-node size versus effective cell-type count and a marker heatmap
  for selected internal nodes.
- `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/pancreas_progenitor_signature_dataset_summary.csv`
  records that the full AnnData has `5` `NEUROG3+` cells and the benchmark
  subset has none.
- `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/pancreas_full_neurog3_positive_cells.csv`
  records the individual full-AnnData `NEUROG3+` reference cells and their
  signature scores.
- `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_adaptive_monophyletic_meeting_progenitor_signature_comparison.csv`
  records progenitor and mature-state signature scores for the focused
  monophyletic-subtree meeting nodes.
- `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_adaptive_monophyletic_meeting_child_progenitor_signature_comparison.csv`
  records the same signature scores for each immediate child branch of those
  meetings.
- `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/pancreas_benchmark_celltype_progenitor_signature_scores.csv`
  and `tbs_adaptive_cluster_progenitor_signature_scores.csv` compare benchmark
  celltype states and final TBS clusters against the progenitor signatures.
- `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/pancreas_progenitor_signature_comparison.png`
  and `pancreas_progenitor_signature_umap.png` visualize the signature
  comparison.
- `applications/scrna/analysis/analyze_pancreas_inner_nodes.py` regenerates the lineage summary,
  progenitor review, and plot from the TBS edge table, assignments, and AnnData
  raw marker expression.
- `applications/scrna/analysis/compare_pancreas_progenitor_signatures.py` regenerates the
  progenitor-signature comparison.

## Links

- [[pancreas-scrna-clustering-benchmark-20260623]]
