# Topology Vector Benchmark 2026-06-15

## Inputs

- Regression gate KL profile benchmark: `raw/assets/benchmark-results/topology_vector_benchmark_20260615/regression_gate_kl_profile/regression_gate_kl_profile_comparison.csv`
- Context-negative topology law: `raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/context_negative_bayesian_topology_law`
- Context-negative topology sensitivity: `raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/context_negative_bayesian_topology_sensitivity`
- Conditional topology law: `raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/conditional_topology_law`
- Conditional topology regression gate: `raw/assets/benchmark-results/conditional_topology_law_20260615/regression_gate/regression_gate_comparison.csv`
- Conditional topology Julia run: `raw/assets/benchmark-results/conditional_topology_law_20260615/julia_selected_family`

## Regression Gate Result

The refined fixed-coordinate global pass-through profile ran on all 17 regression-gate cases with zero skips. Default projected-Wald KL skipped 6/17 rows. The refined profile improved exact-K count from 3/17 to 5/17. Counting skipped rows as ARI 0, mean ARI improves from 0.388484 to 0.445742; among ok-only rows, default projected-Wald is higher (0.600384 vs 0.445742). This profile is more runnable under calibration constraints, but not a universal performance improvement.

## Topology Vector Result

The context-negative topology law top-ranked the focused truth row with posterior log-odds margin 4.954869. Sensitivity supports the structural interpretation: 60/65 profile-weight combinations separate, topology-only separates in 5/5, outgoing-topology-only separates in 5/5, and selected-family plus context separates in 0/5.

## Conditional Topology Law Result

The directed incidence-aware conditional topology law keeps the focused truth row at rank 1 with conditional log-odds margin 4.694486. It remains diagnostic-only: the support status is `support_insufficient_fail_closed` because the internal incidence stratum still has only one truth-recovery row.

The benchmark-facing `kl_conditional_topology_diagnostic` method id runs the 17-case regression gate with 17/17 ok rows, mean ARI 0.460293, median ARI 0.480000, and exact-K count 4/17. On the Julia binary matrix it returns 410 final clusters and 412 stable regions; the generated UMAP overlay still shows heavy fragmentation, so this is not production evidence.

## Interpretation

The benchmark supports using the vector as a diagnostic conditioning object, especially outgoing balance and outgoing edge-norm balance conditioned by the incoming/family event and directed root/internal/pass-through incidence. It does not support promoting the refined profile or the conditional topology law as a production clustering default: the profile closes skips and improves some high-dimensional/phylogenetic rows, but under-splits several regression cases and loses ARI on strong Gaussian/SBM examples; the law still lacks support in the relevant internal stratum.
