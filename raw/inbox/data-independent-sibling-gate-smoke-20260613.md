# Data-Independent Sibling Gate Smoke 2026-06-13

Command:

```bash
python -m benchmarks.diagnostics.calibration.data_independent_sibling_gate_panel \
  --output-dir /tmp/kl_te_data_independent_gate \
  --suite binary \
  --case-names binary_2clusters \
  --data-roles null,signal \
  --candidate-methods coordinate_bonferroni,coordinate_bh \
  --sibling-alpha 0.01 \
  --selected-topology-penalty 10 \
  --replicates 50 \
  --base-seed 20260613
```

Summary:

| case_id | data_role | topology_mode | candidate_method | n_rows | rejection_rate_at_sibling_alpha | rejection_rate_at_effective_alpha | large_parent_rejection_rate_at_effective_alpha | data_independent_gate_status |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| binary_2clusters | null | selected_topology | coordinate_bh | 2450 | 0.026122 | 0.007755 | 0.075697 | data_independent_gate_null_candidate |
| binary_2clusters | null | selected_topology | coordinate_bonferroni | 2450 | 0.025714 | 0.007755 | 0.075697 | data_independent_gate_null_candidate |
| binary_2clusters | signal | selected_topology | coordinate_bh | 2450 | 0.316735 | 0.194694 | 0.738019 | data_independent_gate_signal_retained |
| binary_2clusters | signal | selected_topology | coordinate_bonferroni | 2450 | 0.269388 | 0.175510 | 0.738019 | data_independent_gate_signal_retained |

Production-admissibility summary:

| contract_id | n_components | n_required_components | n_required_ready | n_required_fail_closed | n_required_diagnostic_only | production_decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| data_independent_sibling_gate_candidate | 4 | 4 | 0 | 0 | 4 | diagnostic_only |

Interpretation:

The panel is a diagnostic same-data repair candidate, not a production
calibration rule. Removing same-sample adaptive PCA/dimension selection from
the sibling gate and using fixed coordinate-wise p-value aggregation with a
selected-topology penalty controls the binary selected-null smoke near the
`0.01` target while retaining signal in large parents. The contract remains
`diagnostic_only`.
