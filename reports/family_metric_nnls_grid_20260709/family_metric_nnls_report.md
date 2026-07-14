# Family-Specific Metric and NNLS Grid

- schema_version: `family_metric_nnls_grid/v1`
- generated_at_utc: `20260709_191834Z`
- decision_scope: `evidence_only_no_production_promotion`
- selector_audit_status: `no_promotion`
- cells: `96`
- successful_cells: `86`
- skipped_cells: `10`

The graph kernel and NNLS target use the same selected family embedding. The original observations remain the input to TBS distributional tests.

## Family Results

| source_case_id | metric_family | branch_time_mode | ok_cells | mean_ari | mean_nmi | mean_macro_f1 | mean_silhouette_score | mean_davies_bouldin_index | mean_largest_cluster_fraction | mean_nnls_normalized_residual_rmse | mean_topology_kcut_ari | mean_topology_exact_true_cluster_clade_fraction | mean_topology_weighted_true_cluster_clade_purity | median_root_sibling_p_value_corrected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| binary_low_noise_4c | balanced_binary | none | 8 | 1.0000 | 1.0000 | 1.0000 | 0.5640 | 0.6596 | 0.2500 | 0.0592 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| binary_low_noise_4c | balanced_binary | normalized_branch_length | 8 | 1.0000 | 1.0000 | 1.0000 | 0.5640 | 0.6596 | 0.2500 | 0.0592 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| cat_clear_3cat_4c | categorical | none | 8 | 0.8540 | 0.9286 | 0.8333 | 0.1005 | 2.7781 | 0.3750 | 0.0881 | -0.0001 | 0.9688 | 0.9844 | 0.0000 |
| cat_clear_3cat_4c | categorical | normalized_branch_length | 8 | 0.8540 | 0.9286 | 0.8333 | 0.1005 | 2.7781 | 0.3750 | 0.0881 | -0.0001 | 0.9688 | 0.9844 | 0.0000 |
| cont_lowrank_pggn_shrinkage | high_dimensional_continuous | none | 3 | 0.0209 | 0.0510 | 0.2060 | -0.0080 | 3.7124 | 0.8292 | 0.0132 | 0.0008 | 0.0000 | 0.2675 | 0.4946 |
| cont_lowrank_pggn_shrinkage | high_dimensional_continuous | normalized_branch_length | 3 | 0.0209 | 0.0510 | 0.2060 | -0.0080 | 3.7124 | 0.8292 | 0.0132 | 0.0008 | 0.0000 | 0.2675 | 0.4944 |
| gauss_clear_medium_continuous | continuous_gaussian | none | 8 | 0.0000 | 0.0000 | 0.1000 |  |  | 1.0000 | 0.0329 | 1.0000 | 1.0000 | 1.0000 | 0.1286 |
| gauss_clear_medium_continuous | continuous_gaussian | normalized_branch_length | 8 | 0.0000 | 0.0000 | 0.1000 |  |  | 1.0000 | 0.0329 | 1.0000 | 1.0000 | 1.0000 |  |
| overlap_mod_4c_small | sparse_overlap_binary | none | 8 | 0.6632 | 0.6907 | 0.7760 | 0.0304 | 4.9042 | 0.4053 | 0.1116 | 0.0000 | 0.0000 | 0.2969 | 0.0000 |
| overlap_mod_4c_small | sparse_overlap_binary | normalized_branch_length | 8 | 0.6632 | 0.6907 | 0.7760 | 0.0304 | 4.9042 | 0.4053 | 0.1116 | 0.0000 | 0.0000 | 0.2969 | 0.0000 |
| sbm_moderate | sbm | none | 8 | 0.0000 | 0.0000 | 0.1961 |  |  | 1.0000 | 0.6883 | 0.0193 | 0.0000 | 0.3551 | 0.4110 |
| sbm_moderate | sbm | normalized_branch_length | 8 | 0.0000 | 0.0000 | 0.1961 |  |  | 1.0000 | 0.6883 | 0.0193 | 0.0000 | 0.3551 | 0.3177 |

## Label-Free Selections

| case_id | selected_tree_inference | selected_found_clusters | selected_mean_partition_agreement | selected_ari | selected_nmi | selected_macro_f1 | selector_status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gauss_clear_medium_continuous::none | nan |  |  |  |  |  | skip_no_valid_topology |
| gauss_clear_medium_continuous::normalized_branch_length | nan |  |  |  |  |  | skip_no_valid_topology |
| binary_low_noise_4c::none | weighted | 4.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | selected |
| binary_low_noise_4c::normalized_branch_length | weighted | 4.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | selected |
| sbm_moderate::none | nan |  |  |  |  |  | skip_no_valid_topology |
| sbm_moderate::normalized_branch_length | nan |  |  |  |  |  | skip_no_valid_topology |
| cat_clear_3cat_4c::none | complete | 4.0000 | 0.8331 | 1.0000 | 1.0000 | 1.0000 | selected |
| cat_clear_3cat_4c::normalized_branch_length | complete | 4.0000 | 0.8331 | 1.0000 | 1.0000 | 1.0000 | selected |
| overlap_mod_4c_small::none | average | 4.0000 | 0.6662 | 0.9148 | 0.8883 | 0.9676 | selected |
| overlap_mod_4c_small::normalized_branch_length | average | 4.0000 | 0.6662 | 0.9148 | 0.8883 | 0.9676 | selected |
| cont_lowrank_pggn_shrinkage::none | centroid | 7.0000 | 0.0000 | 0.0627 | 0.1529 | 0.4179 | selected |
| cont_lowrank_pggn_shrinkage::normalized_branch_length | centroid | 7.0000 | 0.0000 | 0.0627 | 0.1529 | 0.4179 | selected |

## Branch-Time Paired Effect

| paired_cells | mean_delta_ari | mean_delta_nmi | mean_delta_macro_f1 | mean_delta_silhouette | mean_delta_largest_cluster_fraction |
| --- | --- | --- | --- | --- | --- |
| 43 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Skipped Cells

| source_case_id | branch_time_mode | tree_inference | skip_reason |
| --- | --- | --- | --- |
| cont_lowrank_pggn_shrinkage | none | average | ValueError: Cannot fit sibling inflation model: no strict-null or stopped-edge empirical-null calibration records with positive weight. Found 79 selected non-null positive-weight record(s), which are not valid empirical-null calibration support. |
| cont_lowrank_pggn_shrinkage | none | complete | ValueError: Cannot fit sibling inflation model: no strict-null or stopped-edge empirical-null calibration records with positive weight. Found 79 selected non-null positive-weight record(s), which are not valid empirical-null calibration support. |
| cont_lowrank_pggn_shrinkage | none | weighted | ValueError: Cannot fit sibling inflation model: no strict-null or stopped-edge empirical-null calibration records with positive weight. Found 79 selected non-null positive-weight record(s), which are not valid empirical-null calibration support. |
| cont_lowrank_pggn_shrinkage | none | median | ValueError: Cannot fit sibling inflation model: no strict-null or stopped-edge empirical-null calibration records with positive weight. Found 79 selected non-null positive-weight record(s), which are not valid empirical-null calibration support. |
| cont_lowrank_pggn_shrinkage | none | ward | ValueError: Cannot fit sibling inflation model: no strict-null or stopped-edge empirical-null calibration records with positive weight. Found 79 selected non-null positive-weight record(s), which are not valid empirical-null calibration support. |
| cont_lowrank_pggn_shrinkage | normalized_branch_length | average | ValueError: Cannot fit sibling inflation model: no strict-null or stopped-edge empirical-null calibration records with positive weight. Found 79 selected non-null positive-weight record(s), which are not valid empirical-null calibration support. |
| cont_lowrank_pggn_shrinkage | normalized_branch_length | complete | ValueError: Cannot fit sibling inflation model: no strict-null or stopped-edge empirical-null calibration records with positive weight. Found 79 selected non-null positive-weight record(s), which are not valid empirical-null calibration support. |
| cont_lowrank_pggn_shrinkage | normalized_branch_length | weighted | ValueError: Cannot fit sibling inflation model: no strict-null or stopped-edge empirical-null calibration records with positive weight. Found 79 selected non-null positive-weight record(s), which are not valid empirical-null calibration support. |
| cont_lowrank_pggn_shrinkage | normalized_branch_length | median | ValueError: Cannot fit sibling inflation model: no strict-null or stopped-edge empirical-null calibration records with positive weight. Found 79 selected non-null positive-weight record(s), which are not valid empirical-null calibration support. |
| cont_lowrank_pggn_shrinkage | normalized_branch_length | ward | ValueError: Cannot fit sibling inflation model: no strict-null or stopped-edge empirical-null calibration records with positive weight. Found 79 selected non-null positive-weight record(s), which are not valid empirical-null calibration support. |
