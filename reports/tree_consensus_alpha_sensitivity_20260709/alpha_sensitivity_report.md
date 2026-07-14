# Alpha Sensitivity For Fail-Closed P-Value Cases

## Scope

This report checks whether adapting `edge_alpha` or `sibling_alpha` can rescue
the real multi-cluster fail-closed cases from the full adaptive-K graphtools
NNLS tree-consensus gate.

Inputs:

- `reports/tree_consensus_fail_closed_pvalues_20260709/fail_closed_loss_taxonomy.csv`
- `reports/tree_consensus_fail_closed_pvalues_20260709/fail_closed_traversal_trace.csv`
- `reports/tree_consensus_alpha_sensitivity_20260709/sibling_alpha_sweep_cells.csv`
- `reports/tree_consensus_alpha_sensitivity_20260709/sibling_alpha_sweep_summary.csv`
- `reports/tree_consensus_alpha_sensitivity_20260709/alpha_threshold_requirements.csv`
- `reports/tree_consensus_alpha_sensitivity_20260709/sibling_alpha_sweep_best_by_case.csv`

Baseline gate values were `edge_alpha = 0.001` and
`sibling_alpha = 0.01`.

## Threshold Findings

The p-values imply two different alpha requirements:

| case | gate needing relaxation | minimal observed alpha to open |
|---|---:|---:|
| `dim_diffuse_6c_136f_continuous` | edge | 0.011940 |
| `phylo_brownian_null_16taxa` | edge | 0.033455 |
| `dim_consolidated_4c_24f_continuous` | edge | 0.080709 |
| `dim_consolidated_4c_72f_continuous` | edge | 0.173662 |
| `gauss_clear_medium_continuous` | edge | 0.421158 |
| `gauss_single_outlier_4c_continuous` | edge | 0.455507 |
| `mp_spike_above_bbp_continuous` | edge | 0.567858 |
| `gauss_moderate_3c_continuous` | edge | 0.621264 |
| `gauss_outlier_cluster_4c_continuous` | edge | 0.637371 |
| `sbm_hard` | sibling | 0.012573 |
| `sbm_moderate` | sibling | 0.022160 |
| `cont_lowrank_pggn_shrinkage` | sibling | 0.081928 |
| `overlap_heavy_8c_large_feat` | sibling | 0.099473 |
| `overlap_extreme_4c` | sibling | 0.194125 |

Opening the gate is only a necessary condition. It does not guarantee that the
resulting partition is scientifically useful.

## Actual Sibling-Alpha Sweep

I reran the five edge-supported fail-closed cases on the topology cells that
already had edge-open traversal. The sweep used `sibling_alpha` values
`0.015`, `0.025`, `0.05`, `0.10`, and `0.20`, while keeping
`edge_alpha = 0.001`.

Best observed result per case:

| case | best sibling_alpha | best ARI | best NMI | best macro F1 | best found clusters | minimum largest-cluster fraction |
|---|---:|---:|---:|---:|---:|---:|
| `sbm_moderate` | 0.025 | 0.076937 | 0.140099 | 0.455969 | 3 | 0.833333 |
| `sbm_hard` | 0.015 | 0.000000 | 0.011540 | 0.184649 | 2 | 0.983333 |
| `overlap_heavy_8c_large_feat` | 0.20 | 0.076480 | 0.155030 | 0.326616 | 22 | 0.238750 |
| `overlap_extreme_4c` | 0.20 | 0.000107 | 0.022903 | 0.127224 | 7 | 0.983333 |
| `cont_lowrank_pggn_shrinkage` | 0.10 | 0.000655 | 0.046873 | 0.125850 | 3 | 0.975000 |

## Interpretation

The overlap diagnostic p-values do react: overlap cases show extremely strong
edge p-values and very small dense/sparse diagnostic sibling p-values. The
failure is specifically in the active corrected sibling gate used by the frozen
run.

However, alpha-only relaxation is not a clean solution:

- `overlap_heavy_8c_large_feat` needs `sibling_alpha = 0.20` to produce a
  nontrivial result, and the best rerun over-fragments to `22` clusters with
  only ARI `0.076480`.
- `overlap_extreme_4c` also needs `sibling_alpha = 0.20`, but the best result
  remains effectively unrecovered: ARI `0.000107` and largest cluster fraction
  `0.983333`.
- `cont_lowrank_pggn_shrinkage` opens at `0.10`, but the split is still a
  dominant-cluster result with ARI `0.000655`.
- `sbm_hard` opens near the nominal threshold, but the resulting split is a
  tiny dominant-cluster partition with nonpositive ARI.
- `sbm_moderate` is the only case where modest alpha relaxation gives a
  coherent partial rescue, but the result remains weak.

Therefore a global alpha optimization is not scientifically adequate. It can
force the gate open, but it does not solve the overlap topology/recovery
problem.

## Recommendation

Do not promote a larger global `sibling_alpha`.

The correct development direction is a constrained adaptive-alpha or
family-specific calibration layer:

1. Keep the current fail-closed production alpha values as the default.
2. Treat dense/sparse overlap p-values as evidence that the overlap signal is
   present, not as direct production permission to split.
3. Develop an overlap-specific validation rule that combines active sibling
   p-values with structural constraints: largest-cluster guard, cluster-count
   plausibility, fragmentation penalty, and label-free stability across
   topology cells.
4. Re-run the full 121-case grid only after the rule can reject the
   `sbm_hard`, low-rank, and dominant-cluster false-open patterns seen in this
   alpha sweep.

In short: the p-values show the tests detect overlap signal, but the current
corrected sibling gate is too conservative for overlap, and naive alpha
relaxation is too blunt to be production-valid.
