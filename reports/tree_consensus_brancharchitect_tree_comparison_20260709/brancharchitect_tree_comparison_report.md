# BranchArchitect Tree Comparison Gate

- schema_version: `brancharchitect_tree_comparison/v1`
- generated_by: `benchmarks.validation.brancharchitect_tree_comparison`
- generated_at_utc: `20260709_140620Z`
- target_method: `tbs_diffusion_graphtools_adaptive_nnls`
- case_count: `14`
- ok_tree_cells: `101`
- skipped_cells: `11`
- pairwise_comparisons: `315`
- brancharchitect_path: `/private/tmp/BranchArchitect-inspect`
- brancharchitect_status: `interpolation_available`
- brancharchitect_interpolation_requested: `True`

## Interpretation

This report compares complete rebuilt tree objects, not only stopped traversal traces. The cluster-path columns are label-free with respect to ground truth: they compare branch-length tree distances within and between the predicted clusters.

BranchArchitect is optional. When configured, the pairwise table adds BranchArchitect RF and weighted RF metrics; interpolation movement-path metrics are filled only when the local BranchArchitect checkout and optional dependencies are present.

## Pairwise Summary

- median_rooted_internal_rf_relative: `0.551839`
- median_rooted_weighted_split_l1: `2.49375`
- median_leaf_path_rmse: `0.00213968`
- median_predicted_label_ari_between_topologies: `1`

## Largest Topology Differences

- dim_diffuse_6c_136f_continuous: average vs centroid, relative_rooted_RF=0.994, leaf_path_RMSE=0.00236, cluster_label_ARI=1
- dim_diffuse_6c_136f_continuous: average vs median, relative_rooted_RF=0.994, leaf_path_RMSE=0.00236, cluster_label_ARI=1
- dim_diffuse_6c_136f_continuous: centroid vs complete, relative_rooted_RF=0.994, leaf_path_RMSE=0.00236, cluster_label_ARI=1
- dim_diffuse_6c_136f_continuous: centroid vs single, relative_rooted_RF=0.994, leaf_path_RMSE=0.00236, cluster_label_ARI=1
- dim_diffuse_6c_136f_continuous: centroid vs ward, relative_rooted_RF=0.994, leaf_path_RMSE=0.00236, cluster_label_ARI=1
- dim_diffuse_6c_136f_continuous: centroid vs weighted, relative_rooted_RF=0.994, leaf_path_RMSE=0.00236, cluster_label_ARI=1
- dim_diffuse_6c_136f_continuous: complete vs median, relative_rooted_RF=0.994, leaf_path_RMSE=0.00236, cluster_label_ARI=1
- dim_diffuse_6c_136f_continuous: median vs single, relative_rooted_RF=0.994, leaf_path_RMSE=0.00236, cluster_label_ARI=1
- dim_diffuse_6c_136f_continuous: median vs ward, relative_rooted_RF=0.994, leaf_path_RMSE=0.00236, cluster_label_ARI=1
- dim_diffuse_6c_136f_continuous: median vs weighted, relative_rooted_RF=0.994, leaf_path_RMSE=0.00236, cluster_label_ARI=1

## Case Summary

- dim_diffuse_6c_136f_continuous: ok_cells=7, cluster_count_range=1-1, max_relative_rooted_RF=0.994, max_leaf_path_RMSE=0.00236, min_cluster_label_ARI=1
- gauss_outlier_cluster_4c_continuous: ok_cells=7, cluster_count_range=1-1, max_relative_rooted_RF=0.994, max_leaf_path_RMSE=0.00267, min_cluster_label_ARI=1
- gauss_single_outlier_4c_continuous: ok_cells=7, cluster_count_range=1-1, max_relative_rooted_RF=0.994, max_leaf_path_RMSE=0.00195, min_cluster_label_ARI=1
- dim_consolidated_4c_24f_continuous: ok_cells=7, cluster_count_range=1-1, max_relative_rooted_RF=0.994, max_leaf_path_RMSE=0.00633, min_cluster_label_ARI=1
- dim_consolidated_4c_72f_continuous: ok_cells=7, cluster_count_range=1-1, max_relative_rooted_RF=0.994, max_leaf_path_RMSE=0.00393, min_cluster_label_ARI=1
- phylo_brownian_null_16taxa: ok_cells=7, cluster_count_range=1-1, max_relative_rooted_RF=0.994, max_leaf_path_RMSE=0.00214, min_cluster_label_ARI=1
- mp_spike_above_bbp_continuous: ok_cells=7, cluster_count_range=1-1, max_relative_rooted_RF=0.992, max_leaf_path_RMSE=0.00171, min_cluster_label_ARI=1
- cont_lowrank_pggn_shrinkage: ok_cells=7, cluster_count_range=1-1, max_relative_rooted_RF=0.987, max_leaf_path_RMSE=2.28e-05, min_cluster_label_ARI=1
- gauss_clear_medium_continuous: ok_cells=7, cluster_count_range=1-1, max_relative_rooted_RF=0.983, max_leaf_path_RMSE=0.0175, min_cluster_label_ARI=1
- gauss_moderate_3c_continuous: ok_cells=7, cluster_count_range=1-1, max_relative_rooted_RF=0.977, max_leaf_path_RMSE=0.000508, min_cluster_label_ARI=1
- overlap_heavy_8c_large_feat: ok_cells=7, cluster_count_range=1-1, max_relative_rooted_RF=0.798, max_leaf_path_RMSE=0.0616, min_cluster_label_ARI=1
- sbm_moderate: ok_cells=8, cluster_count_range=1-1, max_relative_rooted_RF=0.78, max_leaf_path_RMSE=0.0493, min_cluster_label_ARI=1
- overlap_extreme_4c: ok_cells=8, cluster_count_range=1-1, max_relative_rooted_RF=0.764, max_leaf_path_RMSE=0.0606, min_cluster_label_ARI=1
- sbm_hard: ok_cells=8, cluster_count_range=1-1, max_relative_rooted_RF=0.737, max_leaf_path_RMSE=0.0404, min_cluster_label_ARI=1

## Skipped Cells

- `10`: MAD rooting requires positive distances between leaf pairs.
- `1`: Cannot fit sibling inflation model: no strict-null or stopped-edge empirical-null calibration records with positive weight. Found 799 selected non-null positive-weight record(s), which are not valid empirical-null calibration support.
