---
title: scRNA Distributional Action Audit 2026-06-24
type: analysis
status: draft
updated: 2026-06-24
sources:
  - tree_break_selection/hierarchy_analysis/statistics/distributional_action.py
  - tests/statistics/47_test_distributional_action_contract.py
  - tests/pipeline/51_test_dispatch_contract.py
  - tests/pipeline/65_test_scrna_benchmark_branch_time_config.py
  - tests/pipeline/66_test_scrna_benchmark_distributional_action.py
  - benchmarks/shared/runners/dispatch.py
  - benchmarks/shared/runners/tbs_runner.py
  - scripts/audit_scrna_distributional_action.py
  - scripts/pancreas_scrna_cluster_benchmark.py
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/method_metrics.csv
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/benchmark_subset_pca.csv
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_topology_projected_adaptive_k90_alpha0p01_edge0p001_tree_edges.csv
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_adaptive_diffusion_topology_projected_adaptive_k90_alpha0p01_edge0p001_tree_edges.csv
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_branch_time_recomputed_nnls_projected_adaptive_k90_alpha0p01_edge0p001_tree_edges.csv
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_raw_linkage_branch_time_diagnostic_projected_adaptive_k90_alpha0p01_edge0p001_tree_edges.csv
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_adaptive_diffusion_branch_time_recomputed_nnls_projected_adaptive_k90_alpha0p01_edge0p001_tree_edges.csv
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_adaptive_diffusion_raw_linkage_branch_time_diagnostic_projected_adaptive_k90_alpha0p01_edge0p001_tree_edges.csv
  - raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/method_metrics.csv
  - raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/benchmark_subset_pca.csv
  - raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/tbs_topology_projected_adaptive_k90_alpha0p01_edge0p001_tree_edges.csv
  - raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/tbs_adaptive_diffusion_topology_projected_adaptive_k90_alpha0p01_edge0p001_tree_edges.csv
  - raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/tbs_branch_time_recomputed_nnls_projected_adaptive_k90_alpha0p01_edge0p001_tree_edges.csv
  - raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/tbs_raw_linkage_branch_time_diagnostic_projected_adaptive_k90_alpha0p01_edge0p001_tree_edges.csv
  - raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/tbs_adaptive_diffusion_branch_time_recomputed_nnls_projected_adaptive_k90_alpha0p01_edge0p001_tree_edges.csv
  - raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/tbs_adaptive_diffusion_raw_linkage_branch_time_diagnostic_projected_adaptive_k90_alpha0p01_edge0p001_tree_edges.csv
  - raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/goncalves_tbs_inner_node_progenitor_signature_scores.csv
  - raw/assets/benchmark-results/distributional_action_continuous_smoke_20260624/manifest.json
  - raw/assets/benchmark-results/distributional_action_continuous_smoke_20260624/continuous_benchmark_comparison.csv
  - raw/assets/benchmark-results/distributional_action_continuous_smoke_20260624/failure_report.md
  - raw/assets/benchmark-results/scrna_distributional_action_audit_20260624/manifest.json
  - raw/assets/benchmark-results/scrna_distributional_action_audit_20260624/scrna_distributional_action_audit.md
  - raw/assets/benchmark-results/scrna_distributional_action_audit_20260624/scrna_distributional_action_edges.csv
  - raw/assets/benchmark-results/scrna_distributional_action_audit_20260624/scrna_distributional_action_method_summary.csv
  - raw/assets/benchmark-results/scrna_distributional_action_audit_20260624/distributional_action_vs_branch_length.png
  - raw/assets/benchmark-results/scrna_distributional_action_audit_20260624/top_internal_distributional_action_edges.png
  - raw/assets/benchmark-results/scrna_distributional_action_audit_20260624/distributional_action_vs_edge_statistic.png
tags:
  - scrna
  - pancreas
  - branch-length
  - clustering
  - distribution
---

# scRNA Distributional Action Audit 2026-06-24

## Summary

The scRNA tree outputs already record descendant mass through leaf counts, but
leaf count alone is not enough to say how much an internal node changes the
distribution. The audit adds a direct diagnostic,
`subtree_distributional_action = child_leaf_count *
mean_standardized_PCA_delta^2`, where the delta is the child barycenter minus
the parent barycenter in the benchmark PCA space.

The adult pancreas and Goncalves adaptive-diffusion TBS rows confirm the issue:
top distributional-action edges do not coincide with the largest subtrees. In
both adaptive-diffusion topology runs, the top-10 action edges have zero overlap
with the top-10 descendant-mass edges.

The implementation mistake was treating this barycentric action as if it could
be inserted after gate annotation and used to close the existing
`Child_Parent_Divergence_Significant` flags. That bypasses the actual
clustering formula: projected-Wald child-parent tests, sibling FDR, covariance
normalization, projection dimension, branch-time policy, and passthrough
traversal. The current code therefore rejects non-`none` split-action filter
policies and keeps distributional action as annotations and exported audit
columns only.

The adult pancreas and Goncalves benchmarks were rerun after this correction.
Both current `method_metrics.csv` files contain `11` active rows and no method
label containing action or split filtering. Adult adaptive-diffusion topology
remains `43` clusters with `ARI = 0.412704` and `V = 0.662011`; Adult
adaptive-diffusion raw-linkage diagnostic is `41` clusters with
`ARI = 0.415078` and `V = 0.665311`. Goncalves adaptive-diffusion topology,
NNLS branch-time, and raw-linkage diagnostic all produce `24` clusters with
`ARI = 0.205837` and `V = 0.409991`; standardized-PCA branch-time remains
unstable on Goncalves, with the NNLS and raw-linkage branch-time rows
collapsing to one cluster.

## Details

`scripts/audit_scrna_distributional_action.py` reconstructs each tree node
barycenter from the saved `benchmark_subset_pca.csv` leaf order and the TBS
edge tables. It standardizes the PCA coordinates over the benchmark subset, then
computes per-edge displacement, child leaf count, child leaf fraction, branch
length, edge-test metadata, and the mass-weighted distributional-action score.
The core mass-weighted squared-displacement formula now lives in
`tree_break_selection/hierarchy_analysis/statistics/distributional_action.py`
so tests can protect it independently from the scRNA audit script. The helper
also exposes mass-bearing summaries: edge summaries carry `parent_mass`,
`child_mass`, and `child_parent_mass_fraction`; binary split summaries carry
`left_mass`, `right_mass`, `parent_mass`, and child-parent mass fractions.
The scRNA benchmark exporter now writes the same edge mass/action diagnostics
into each current canonical `*_tree_edges.csv` file, so benchmark artifacts
carry both leaf-count mass and distributional movement without rerunning the
separate audit script.

`benchmarks/shared/runners/dispatch.py` and
`benchmarks/shared/runners/tbs_runner.py` still have a parameter surface for
`distributional_action_split_filter_policy`, but
`tree_break_selection/hierarchy_analysis/statistics/distributional_action.py`
now raises for any policy except `"none"`. This is deliberate: a usable action
gate must first be calibrated against the projected-Wald edge/sibling tests and
the traversal law. The scRNA benchmark registry no longer registers activated
`split-action q50` rows, and the audit script no longer reads stale q50 output
files from earlier runs.

The refreshed audit reads only the six canonical TBS rows per dataset:
topology, recomputed-NNLS branch-time, and raw-linkage diagnostic for both
standardized-PCA and adaptive-diffusion geometries. It records `47,556` edge
diagnostics, with `split_filter_policy = none` and
`split_filter_filtered_count = 0` for every method.

For adaptive-diffusion topology, adult pancreas has weak action-vs-leaf-count
rank association (`Spearman = 0.0760`) and only moderate action-vs-branch-length
association (`Spearman = 0.3672`). Goncalves has essentially no
action-vs-leaf-count association (`Spearman = -0.0334`) and weak
action-vs-branch-length association (`Spearman = 0.2097`). Therefore mass and
branch length are useful but not substitutes for distributional action.

The Goncalves top-action internal edges include both non-progenitor separations
and progenitor-relevant structure. `N2925->N2910` is a mixed fetal
progenitor-state grouping, `N2920->N2898` is a broad progenitor-rich
neighborhood, and `N2922->N2851` is a tip-progenitor-enriched grouping. These
edges are more informative than a leaf-count-only ranking because they combine
how many cells move with how far the child distribution moves away from its
parent.

The superseded q50 action filters remain useful only as negative controls. The
edge-local version was wrong because a child edge is only one contribution to
the parent split action, while the split is tied through the sibling contrast
and the parent barycenter. The parent split-action q50 version used the right
barycentric object, but it was still an uncalibrated global threshold and
removed small split nodes that the traversal needed. Neither filter is an
active clustering method after the correction.

NNLS branch lengths still fit additive path distances on a fixed topology. They
do not currently optimize or display this mass-weighted action quantity. The
edge projected-Wald statistic is closer to this concern because it includes
sample-size and covariance normalization, but it is a test statistic rather
than a direct visual contribution score for internal-node interpretation.

The continuous benchmark smoke was also rerun for the standard TBS and the
guarded within-child covariance candidate. The continuous-guarded method
completed `9/9` rows with mean `ARI = 0.769202`, while the standard TBS row had
`8` ok rows plus `1` skip and mean `ARI = 0.368780` over ok rows. This confirms
the distributional-action rewrite did not replace the continuous covariance
gate path.

## Evidence

- `raw/assets/benchmark-results/scrna_distributional_action_audit_20260624/scrna_distributional_action_audit.md`
  records the audit method, summary tables, and top internal edges.
- `raw/assets/benchmark-results/scrna_distributional_action_audit_20260624/scrna_distributional_action_method_summary.csv`
  records the rank correlations, top-edge overlaps, open-edge counts, and
  zero split-filter counts for the canonical adult/Goncalves TBS rows.
- `raw/assets/benchmark-results/scrna_distributional_action_audit_20260624/scrna_distributional_action_edges.csv`
  records `47,556` per-edge diagnostic rows; all rows have
  `split_filter_policy = none`.
- `raw/assets/benchmark-results/scrna_distributional_action_audit_20260624/distributional_action_vs_branch_length.png`
  shows that branch length and action are related but not interchangeable.
- `raw/assets/benchmark-results/scrna_distributional_action_audit_20260624/top_internal_distributional_action_edges.png`
  ranks the strongest internal-node distributional moves.
- `raw/assets/benchmark-results/scrna_distributional_action_audit_20260624/distributional_action_vs_edge_statistic.png`
  compares action against projected-Wald edge statistics.
- `tests/statistics/47_test_distributional_action_contract.py` verifies the
  binary split variance identity, that edge action is independent of branch
  length, that action changes when child-parent distributional movement changes
  under a fixed branch length, and that edge/split action summaries expose the
  distribution masses used to compute the action.
- `tests/pipeline/66_test_scrna_benchmark_distributional_action.py` verifies
  that the scRNA benchmark tree-edge exporter writes distributional-action
  mass fields, the exact standardized action field, and filter fields to
  generated benchmark edge CSVs.
- The adult pancreas and Goncalves benchmarks were rerun on 2026-06-24 with
  `--max-cells 2500 --n-pcs 30 --seed 0`; the Goncalves run used
  `--skip-download --n-hvgs 2000`. Current `method_metrics.csv` files in both
  benchmark output folders have `11` rows and no active action/split filter
  labels.
- `raw/assets/benchmark-results/distributional_action_continuous_smoke_20260624/continuous_benchmark_comparison.csv`
  records the continuous smoke rerun: guarded within-child covariance is
  `9/9` ok with mean `ARI = 0.769202`; standard TBS is `8` ok and `1` skip
  with mean `ARI = 0.368780` across ok rows.
- `raw/assets/benchmark-results/distributional_action_continuous_smoke_20260624/failure_report.md`
  records the refreshed continuous smoke failure diagnosis with
  `Generated at: 2026-06-24T20:14:29+02:00`.
- Verification on 2026-06-24: `ruff check`, `py_compile`, and focused pytest
  over the distributional-action, dispatch, scRNA config, export, and
  benchmark-smoke tests passed; the continuous smoke, Adult benchmark,
  Goncalves benchmark, and audit script reruns completed successfully.

## Links

- [[scrna-branch-length-effect-audit-20260624]]
- [[scrna-plot-pipeline-audit-20260624]]
- [[goncalves-tbs-progenitor-analysis-20260624]]
- [[pancreas-scrna-clustering-benchmark-20260623]]
- [[goncalves-pancreas-progenitor-benchmark-prep-20260624]]

## Open Questions

- Should `subtree_distributional_action` be promoted into the standard tree edge
  summaries and plot labels?
- Should report figures rank internal nodes by action rather than leaf count or
  branch length when the goal is biological interpretation?
- Should a covariance-normalized action score be added alongside the PCA
  standardized score, closer to the projected-Wald geometry?
- What calibrated statistic would let split action participate in the
  projected-Wald edge/sibling gate system instead of acting as an uncalibrated
  post-hoc threshold?
