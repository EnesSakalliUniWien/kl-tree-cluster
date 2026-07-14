# Tree Literature Alpha Replay

## Scope

This diagnostic replays tree-testing literature policy families on the
existing fail-closed adaptive-K graphtools NNLS traversal traces. It is
validation evidence only and does not change production alpha defaults,
tree traversal, topology selection, or fallback behavior.

Inputs:

- Trace CSV: `reports/tree_consensus_fail_closed_pvalues_20260709/fail_closed_traversal_trace.csv`
- Loss taxonomy CSV: `reports/tree_consensus_fail_closed_pvalues_20260709/fail_closed_loss_taxonomy.csv`

## Literature Policy Mapping

| policy_id                             | literature_family                                           | status                |   alpha |   cap | note                                                                                                                     |
|:--------------------------------------|:------------------------------------------------------------|:----------------------|--------:|------:|:-------------------------------------------------------------------------------------------------------------------------|
| current_traversal_gate                | current TBS traversal baseline                              | native_current        |    0.01 |  0.01 | Frozen edge-open plus corrected active sibling p-value gate.                                                             |
| yekutieli_hierarchical_fdr_bh         | hierarchical FDR                                            | executable_analogue   |    0.01 |  0.01 | Parent-gated BH over active sibling p-values within each reached depth.                                                  |
| treebh_multiresolution_bh             | TreeBH / multiresolution tree testing                       | executable_analogue   |    0.01 |  0.01 | Depth-wise BH with parent rejection-fraction alpha scaling.                                                              |
| lynch_guo_dependence_robust_by        | hierarchical FDR under dependence                           | conservative_proxy    |    0.01 |  0.01 | Parent-gated BY-style harmonic correction as a dependence-robust stress test.                                            |
| bretz_graphical_gatekeeping_recycle   | graphical gatekeeping / alpha recycling                     | executable_analogue   |    0.01 |  0.01 | Sequential gatekeeping that conserves inherited alpha along accepted branches.                                           |
| gao_selective_inference_required      | selective inference for hierarchical clustering             | validation_constraint |    0.01 |  0.01 | Fails closed because exact selected-clustering p-values are not in this trace.                                           |
| wu_randomized_alpha_spending_required | randomized dendrogram p-values plus adaptive alpha spending | validation_constraint |    0.01 |  0.01 | Fails closed because randomized node p-values are not in this trace.                                                     |
| trace_adaptive_alpha_spending_proxy   | adaptive alpha spending                                     | exploratory_proxy     |    0.01 |  0.2  | Trace-only adaptive cap using edge and diagnostic context; includes a child-balance guard and is not a formal FDR claim. |

Interpretation of status:

- `native_current`: already implemented in the current TBS traversal.
- `executable_analogue`: executable replay of the literature idea on the
  available TBS trace, not a claim of paper-exact implementation.
- `conservative_proxy`: deliberately stricter dependence stress test.
- `validation_constraint`: the paper's required p-value object is absent,
  so the policy fails closed rather than inventing a substitute.
- `exploratory_proxy`: trace-only diagnostic for mechanism discovery, not
  formal FDR or production evidence.

## Method Summary

| policy_id                             |   cases_with_candidate_split |   edge_gate_cases_with_split |   sibling_gate_cases_with_split |   total_split_cells |   total_opened_nodes |   total_pass_through_nodes |   max_cluster_count |   min_largest_cluster_fraction |   max_local_alpha |
|:--------------------------------------|-----------------------------:|-----------------------------:|--------------------------------:|--------------------:|---------------------:|---------------------------:|--------------------:|-------------------------------:|------------------:|
| current_traversal_gate                |                            0 |                            0 |                               0 |                   0 |                    0 |                          0 |                   1 |                       1        |              0.01 |
| yekutieli_hierarchical_fdr_bh         |                            0 |                            0 |                               0 |                   0 |                    0 |                          0 |                   1 |                       1        |              0.01 |
| treebh_multiresolution_bh             |                            0 |                            0 |                               0 |                   0 |                    0 |                          0 |                   1 |                       1        |              0.01 |
| lynch_guo_dependence_robust_by        |                            0 |                            0 |                               0 |                   0 |                    0 |                          0 |                   1 |                       1        |              0.01 |
| bretz_graphical_gatekeeping_recycle   |                            0 |                            0 |                               0 |                   0 |                    0 |                          0 |                   1 |                       1        |              0.01 |
| gao_selective_inference_required      |                            0 |                            0 |                               0 |                   0 |                    0 |                          0 |                   1 |                       1        |              0.01 |
| wu_randomized_alpha_spending_required |                            0 |                            0 |                               0 |                   0 |                    0 |                          0 |                   1 |                       1        |              0.01 |
| trace_adaptive_alpha_spending_proxy   |                            2 |                            0 |                               2 |                   4 |                    5 |                          1 |                   3 |                       0.833333 |              0.2  |

## Requirement Audit

| requested_family                                    | paper_exact_status                  |   cases_with_candidate_split |   max_local_alpha | production_interpretation         |
|:----------------------------------------------------|:------------------------------------|-----------------------------:|------------------:|:----------------------------------|
| Hierarchical FDR                                    | executable_analogue                 |                            0 |              0.01 | no_promotion                      |
| TreeBH / multiresolution tree testing               | executable_analogue                 |                            0 |              0.01 | no_promotion                      |
| Hierarchical procedures under dependence            | conservative_proxy                  |                            0 |              0.01 | no_promotion                      |
| Graphical gatekeeping / alpha recycling             | executable_analogue                 |                            0 |              0.01 | no_promotion                      |
| Selective inference for clustering                  | required_p_values_absent            |                            0 |              0.01 | blocked_pending_selected_p_values |
| Adaptive alpha spending for hierarchical clustering | required_p_values_absent_with_proxy |                            2 |              0.2  | diagnostic_only_no_promotion      |

## Companion FDR Smoke

This companion run tests the algorithmic and selected-tree layers on
`binary_2clusters` null replicates. It is a smoke diagnostic, not a
full calibration proof.

| layer                  |   mean_fdp |   false_rejection_rate |   n_ok |   n_support_failures | outcome                        |
|:-----------------------|-----------:|-----------------------:|-------:|---------------------:|:-------------------------------|
| synthetic_valid_p      |       0    |                   0    |     20 |                    0 | algorithmic_fdr_control        |
| fixed_tree_wald        |       0.25 |                   0.25 |     20 |                    0 | fixed_tree_calibration_failure |
| selected_tree_wald     |       0.95 |                   0.95 |     20 |                    0 | edge_selected_family_failure   |
| selected_tree_inflated |       0    |                   0    |      2 |                   18 | inflation_support_failure      |


## Candidate Split Cases

| policy_id                           | case_id                     | loss_bucket                         |   split_cells |   opened_nodes_total |   pass_through_nodes_total |   max_cluster_count |   min_largest_cluster_fraction |   max_local_alpha |   min_rejected_p_value |
|:------------------------------------|:----------------------------|:------------------------------------|--------------:|---------------------:|---------------------------:|--------------------:|-------------------------------:|------------------:|-----------------------:|
| trace_adaptive_alpha_spending_proxy | overlap_heavy_8c_large_feat | sibling_gate_closed_after_edge_open |             1 |                    1 |                          1 |                   3 |                       0.95125  |               0.2 |              0.154825  |
| trace_adaptive_alpha_spending_proxy | sbm_moderate                | sibling_gate_closed_after_edge_open |             3 |                    4 |                          0 |                   3 |                       0.833333 |               0.2 |              0.0221597 |

## Conclusions

- Parent-gated hierarchical FDR, TreeBH-style depth scaling, dependence-
  robust BY replay, and graphical alpha recycling preserve the effective
  `0.01` sibling budget on these traces. They therefore do not rescue the
  overlap fail-closed cases by themselves.
- Exact selective-inference and randomized-dendrogram alpha-spending
  variants cannot be truthfully replayed from the current trace because
  the required selected or randomized node p-values are absent. The gate
  fails closed for those families.
- Any candidate splits opened by the trace-only adaptive proxy are
  mechanism-discovery evidence only. They are not production-valid
  without selected-hierarchy/null calibration and stronger structural
  guards.

## Literature Sources

- Yekutieli 2008 hierarchical FDR:
  https://www.math.tau.ac.il/~yekutiel/papers/JASA%20FDR%20trees.pdf
- Bogomolov, Peterson, Benjamini, Sabatti TreeBH:
  https://arxiv.org/abs/1705.07529
- Lynch and Guo hierarchical FDR under dependence:
  https://arxiv.org/abs/1612.04467
- Bretz, Maurer, Brannath, Posch graphical gatekeeping:
  https://doi.org/10.1002/sim.3495
- Gao, Bien, Witten selective inference for hierarchical clustering:
  https://arxiv.org/abs/2012.02936
- Wu, Bien, Panigrahi randomized hierarchical clustering with confidence:
  https://arxiv.org/abs/2512.06522

## Generated Artifacts

- `reports/tree_consensus_literature_policy_replay_20260709/literature_policy_replay_decisions.csv`
- `reports/tree_consensus_literature_policy_replay_20260709/literature_policy_replay_cell_summary.csv`
- `reports/tree_consensus_literature_policy_replay_20260709/literature_policy_replay_case_summary.csv`
- `reports/tree_consensus_literature_policy_replay_20260709/literature_policy_replay_method_summary.csv`
- `reports/tree_consensus_literature_policy_replay_20260709/literature_policy_replay_requirement_audit.csv`
- `reports/tree_consensus_literature_policy_replay_20260709/literature_policy_replay_manifest.json`
