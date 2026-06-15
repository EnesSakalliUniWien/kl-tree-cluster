# Data-Independent Sibling Gate Transfer 2026-06-13

## Binary Transfer Smoke

Command:

```bash
python -m benchmarks.diagnostics.calibration.data_independent_sibling_gate_panel \
  --output-dir /tmp/kl_te_data_independent_gate_binary_transfer \
  --suite binary \
  --case-names binary_2clusters,binary_many_clusters,binary_unbalanced_low \
  --data-roles null,signal \
  --candidate-methods coordinate_bonferroni,coordinate_bh \
  --sibling-alpha 0.01 \
  --selected-topology-penalties 5,10,20,50 \
  --replicates 20 \
  --base-seed 20260613
```

Scoped production-admissibility re-evaluation after separating each
method/penalty/family candidate:

| candidate_method | penalty | source_family | max_null_rejection_rate_at_effective_alpha | min_signal_rejection_rate_at_effective_alpha | transfer_status | production_decision |
| --- | ---: | --- | ---: | ---: | --- | --- |
| coordinate_bh | 5 | binary_template | 0.013265 | 0.206355 | data_independent_gate_penalty_null_inflated | fail_closed_undefined |
| coordinate_bh | 10 | binary_template | 0.008163 | 0.151171 | data_independent_gate_penalty_transfer_candidate | diagnostic_only |
| coordinate_bh | 20 | binary_template | 0.007143 | 0.084114 | data_independent_gate_penalty_signal_weak | fail_closed_undefined |
| coordinate_bh | 50 | binary_template | 0.004082 | 0.046823 | data_independent_gate_penalty_signal_weak | fail_closed_undefined |
| coordinate_bonferroni | 5 | binary_template | 0.013265 | 0.046823 | data_independent_gate_penalty_null_inflated | fail_closed_undefined |
| coordinate_bonferroni | 10 | binary_template | 0.008163 | 0.046823 | data_independent_gate_penalty_signal_weak | fail_closed_undefined |
| coordinate_bonferroni | 20 | binary_template | 0.007143 | 0.046823 | data_independent_gate_penalty_signal_weak | fail_closed_undefined |
| coordinate_bonferroni | 50 | binary_template | 0.004082 | 0.046823 | data_independent_gate_penalty_signal_weak | fail_closed_undefined |

Interpretation: in this binary smoke, `coordinate_bh` with penalty `10` is the
only candidate that controls all three selected-null cases while retaining all
three signal cases under the predeclared `0.15` all-parent signal threshold.
The status is still `diagnostic_only`, not production promotion.

## Binary/Categorical Transfer Smoke

Command:

```bash
python -m benchmarks.diagnostics.calibration.data_independent_sibling_gate_panel \
  --output-dir /tmp/kl_te_data_independent_gate_transfer \
  --suite full \
  --case-names binary_2clusters,cat_clear_3cat_4c \
  --data-roles null,signal \
  --candidate-methods coordinate_bonferroni,coordinate_bh \
  --sibling-alpha 0.01 \
  --selected-topology-penalties 5,10,20 \
  --replicates 20 \
  --base-seed 20260613
```

Key rows:

| case_id | data_role | candidate_method | penalty | source_family | rejection_rate_at_effective_alpha | large_parent_rejection_rate_at_effective_alpha | status |
| --- | --- | --- | ---: | --- | ---: | ---: | --- |
| binary_2clusters | null | coordinate_bh | 10 | binary_template | 0.008163 | 0.080000 | data_independent_gate_null_candidate |
| binary_2clusters | signal | coordinate_bh | 10 | binary_template | 0.202041 | 0.778689 | data_independent_gate_signal_retained |
| cat_clear_3cat_4c | null | coordinate_bh | 10 | categorical_multinomial | 0.003535 | 0.034483 | data_independent_gate_null_candidate |
| cat_clear_3cat_4c | signal | coordinate_bh | 10 | categorical_multinomial | 0.057071 | 0.465686 | data_independent_gate_signal_weak |

Interpretation: the direct categorical smoke is null-conservative but weak
under the all-parent signal threshold. Large-parent signal rejection remains
visible, so the categorical blocker is power/aggregation and traversal
targeting rather than null inflation in this small smoke.

## Categorical Block-Gate Smoke

Command:

```bash
python -m benchmarks.diagnostics.calibration.data_independent_sibling_gate_panel \
  --output-dir /tmp/kl_te_data_independent_gate_categorical_block \
  --suite full \
  --case-names cat_clear_3cat_4c,cat_mod_3cat_4c,cat_highcard_10cat_4c \
  --data-roles null,signal \
  --candidate-methods coordinate_bh,block_bh \
  --sibling-alpha 0.01 \
  --selected-topology-penalties 5,10,20,50 \
  --replicates 20 \
  --base-seed 20260613
```

Transfer summary:

| candidate_method | penalty | max_null_rejection_rate_at_effective_alpha | min_signal_rejection_rate_at_effective_alpha | transfer_status |
| --- | ---: | ---: | ---: | --- |
| block_bh | 5 | 0.008824 | 0.033166 | data_independent_gate_penalty_signal_weak |
| block_bh | 10 | 0.006303 | 0.029899 | data_independent_gate_penalty_signal_weak |
| block_bh | 20 | 0.003361 | 0.027638 | data_independent_gate_penalty_signal_weak |
| block_bh | 50 | 0.002020 | 0.025879 | data_independent_gate_penalty_signal_weak |
| coordinate_bh | 5 | 0.007983 | 0.049580 | data_independent_gate_penalty_signal_weak |
| coordinate_bh | 10 | 0.005462 | 0.044538 | data_independent_gate_penalty_signal_weak |
| coordinate_bh | 20 | 0.003361 | 0.039916 | data_independent_gate_penalty_signal_weak |
| coordinate_bh | 50 | 0.002764 | 0.036134 | data_independent_gate_penalty_signal_weak |

Interpretation:

Feature-block categorical gates are statistically well-defined fixed-subspace
comparators: each original categorical feature contributes a chi-square block
with `df = n_categories - 1`, and block p-values are aggregated by BH or
Bonferroni. In this smoke, block BH remains null-conservative but does not
recover all-parent signal retention. Coordinate BH is more powerful for the
high-cardinality case, but it also remains signal-weak. The remaining
categorical blocker is therefore not only one-hot coordinate aggregation; it is
the selected-topology/traversal signal distribution across many weak sibling
contexts.
