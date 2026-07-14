# Selected-Node Adaptive Law Benchmark Replay

Generated UTC: `20260709_180648Z`

## Status

This is a benchmark replay, not a production method change. The strict
selected-node law still fails closed because exact selected conditional
p-values or conditional Monte Carlo samples are absent. The adaptive policies
below are diagnostic proxies that test whether the available benchmark traces
support spending more sibling alpha after conditioning on edge evidence,
diagnostic covariance evidence, topology stability, NNLS branch-length
stability, and child-size balance.

## Policy Summary

- `strict_selected_law_required` (Strict selected conditional law required): `requires_exact_selected_node_p_value`, split cases `0/14`, sibling split cases `0/5`, total split cells `0`, max local alpha `0.01`, min topology spend `0.03555`.
- `adaptive_topology_branch_spending_proxy` (Adaptive topology/branch spending proxy): `diagnostic_proxy`, split cases `0/14`, sibling split cases `0/5`, total split cells `0`, max local alpha `0.05776`, min topology spend `0.03555`.
- `adaptive_no_stability_ablation` (Adaptive no-stability ablation): `diagnostic_ablation`, split cases `1/14`, sibling split cases `1/5`, total split cells `3`, max local alpha `0.2`, min topology spend `1`.

## Candidate Split Cases

- `adaptive_no_stability_ablation` / `sbm_moderate`: status `candidate_split_found`, split cells `3`, max clusters `3`, max alpha `0.2`.

## External Metric Audit

- `adaptive_no_stability_ablation` / `sbm_moderate` at alpha `0.2`: ARI `0.07184`, NMI `0.1716`, macro F1 `0.473`, clusters `6`, audit `nearest_available_rerun`.

## Interpretation

The strict selected law is scientifically honest but does not rescue benchmark
cases without a selected conditional p-value object. The topology/branch
spending proxy shows what the formal law would need to adjudicate: if topology
stability and branch-length stability suppress local alpha, the method remains
closed; if those penalties are ablated, any newly opened splits are diagnostic
only and must be checked against actual reruns before promotion.

## Sources

- `reports/tree_consensus_fail_closed_pvalues_20260709/fail_closed_traversal_trace.csv`
- `reports/tree_consensus_topology_difference_diagnosis_20260709/topology_difference_case_diagnosis.csv`
- `reports/tree_consensus_alpha_sensitivity_20260709/sibling_alpha_sweep_summary.csv`
- `benchmarks/validation/selected_node_sibling_null_law.py`
