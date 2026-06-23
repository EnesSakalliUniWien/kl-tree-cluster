# GO Annotation Dataset and Directory Naming Inventory

## Dataset Location Contract

- Canonical reusable feature matrices live under `data/feature_matrices/`.
- New GO annotation matrices should use `feature_matrix_<dataset_slug>.tsv`.
- `<dataset_slug>` is lowercase snake case and contains no spaces, parentheses, or date suffixes.
- Root-level copies and files with download suffixes such as `(1)` are non-canonical.
- Large/private external inputs should first be captured or documented, then promoted to `data/feature_matrices/` before canonical reruns.

## Result Directory Contract

- Canonical wrapper runs: `results/analyses/<dataset_slug>_go_annotation_pipeline_<YYYYMMDD_HHMMSS>/`.
- Direct current-method runs: `results/analyses/<dataset_slug>_current_adaptive_diffusion_subspace_tree_<YYYYMMDD_HHMMSS>/`.
- Retained historical evidence under `raw/assets/benchmark-results/` should keep `<dataset_slug>_<analysis_kind>_<YYYYMMDD>/`.
- The canonical current stage contains `rankings/`, `plots/`, `subspaces/<weighting>/<block_name>/`, `connected_results_manifest.json`, `ARTIFACT_INDEX.md`, `<dataset_slug>_quality_aware_go_ic_by_method/`, and `<dataset_slug>_quality_aware_go_ic_plots/`.
- Reader-facing names should use `adaptive_diffusion_cosine_subspace` or `raw_cosine_subspace`, not internal `kak` labels.

## Inventory Counts

- Matrices inspected: `11`
- Matrices directly usable for the GO pipeline: `11`
- Duplicate-content groups: `0`
- Result roots inspected: `16`
- Result roots with naming/base mismatches: `11`

## Matrices

```text
                                                                      path                  dataset_slug                                                       recommended_path  n_rows  n_features  usable_for_go_pipeline  tracked_by_git        naming_status  has_duplicate_copy
            data/feature_matrices/feature_matrix_allGO_new_interactome.tsv         allgo_new_interactome         data/feature_matrices/feature_matrix_allGO_new_interactome.tsv     339        5873                    True            True            canonical               False
                                  data/feature_matrices/feature_matrix.tsv                feature_matrix                               data/feature_matrices/feature_matrix.tsv     626         456                    True            True            canonical               False
                  data/feature_matrices/feature_matrix_julia_allGO_new.tsv               julia_allgo_new               data/feature_matrices/feature_matrix_julia_allGO_new.tsv     602        6368                    True            True            canonical               False
                       data/feature_matrices/feature_matrix_julia_GOBP.tsv                    julia_gobp                    data/feature_matrices/feature_matrix_julia_GOBP.tsv     703        4922                    True            True            canonical               False
    data/feature_matrices/feature_matrix_julia_GOCC_GOBP_GOMF_combined.tsv julia_gocc_gobp_gomf_combined data/feature_matrices/feature_matrix_julia_GOCC_GOBP_GOMF_combined.tsv     703       14766                    True            True            canonical               False
                        data/feature_matrices/CMS_GO_BP_feature_matrix.tsv                     cms_go_bp                     data/feature_matrices/feature_matrix_cms_go_bp.tsv     225        3101                    True            True  legacy_tracked_name               False
                        data/feature_matrices/CMS_GO_CC_feature_matrix.tsv                     cms_go_cc                     data/feature_matrices/feature_matrix_cms_go_cc.tsv     213         376                    True            True  legacy_tracked_name               False
                        data/feature_matrices/CMS_GO_MF_feature_matrix.tsv                     cms_go_mf                     data/feature_matrices/feature_matrix_cms_go_mf.tsv     216         695                    True            True  legacy_tracked_name               False
                     data/feature_matrices/CMS_Reactome_feature_matrix.tsv                  cms_reactome                  data/feature_matrices/feature_matrix_cms_reactome.tsv     226        1740                    True            True  legacy_tracked_name               False
                         data/feature_matrices/HC_feature_matrix_GO_CC.tsv                      hc_go_cc                      data/feature_matrices/feature_matrix_hc_go_cc.tsv     317        1834                    True            True  legacy_tracked_name               False
/Users/berksakalli/Downloads/feature_matrix_AML_michel_reactome_msigdb.tsv    aml_michel_reactome_msigdb    data/feature_matrices/feature_matrix_AML_michel_reactome_msigdb.tsv     150        1665                    True           False outside_feature_root               False
```

## Duplicate Copies

No duplicate-content matrix groups found.

## Result Roots

```text
                                                                                                  path                              result_kind               dataset_slug  prefix_matches  under_preferred_base  has_connected_manifest  has_method_pdfs
       results/analyses/allgo_new_interactome_current_adaptive_diffusion_subspace_tree_20260618_175237 current_adaptive_diffusion_subspace_tree      allgo_new_interactome            True                  True                    True             True
 raw/assets/benchmark-results/julia_allGO_new_current_adaptive_diffusion_subspace_tree_20260618_163039 current_adaptive_diffusion_subspace_tree            julia_allgo_new           False                 False                   False            False
             results/analyses/julia_allGO_new_current_adaptive_diffusion_subspace_tree_20260618_163811 current_adaptive_diffusion_subspace_tree            julia_allgo_new           False                  True                    True             True
                          raw/assets/benchmark-results/julia_allGO_new_feature_matrix_quality_20260618                   feature_matrix_quality            julia_allgo_new           False                  True                   False            False
                         results/analyses/allgo_new_interactome_go_annotation_pipeline_dryrun_20260619                   go_annotation_pipeline      allgo_new_interactome            True                  True                   False            False
                    results/analyses/aml_michel_reactome_msigdb_go_annotation_pipeline_20260623_145252                   go_annotation_pipeline aml_michel_reactome_msigdb            True                  True                   False             True
                    results/analyses/aml_michel_reactome_msigdb_go_annotation_pipeline_20260623_151556                   go_annotation_pipeline aml_michel_reactome_msigdb            True                  True                   False            False
                                     results/analyses/gocc_iker_go_annotation_pipeline_20260619_144917                   go_annotation_pipeline                  gocc_iker            True                  True                   False             True
                               results/analyses/julia_allGO_new_go_annotation_pipeline_dryrun_20260619                   go_annotation_pipeline            julia_allgo_new           False                  True                   False            False
                                raw/assets/benchmark-results/julia_allGO_new_go_ic_tree_plots_20260618                         go_ic_tree_plots            julia_allgo_new           False                  True                   False             True
                      raw/assets/benchmark-results/julia_allGO_new_method_version_tree_matrix_20260618               method_version_tree_matrix            julia_allgo_new           False                  True                   False            False
                 raw/assets/benchmark-results/julia_allGO_new_method_version_tree_matrix_core_20260618               method_version_tree_matrix            julia_allgo_new           False                  True                   False            False
        raw/assets/benchmark-results/julia_allGO_new_method_version_tree_matrix_current_whole_20260618               method_version_tree_matrix            julia_allgo_new           False                  True                   False            False
raw/assets/benchmark-results/julia_allGO_new_method_version_tree_matrix_selected_plus_current_20260618               method_version_tree_matrix            julia_allgo_new           False                  True                   False             True
       raw/assets/benchmark-results/julia_allGO_new_method_version_tree_matrix_tfidf_selected_20260618               method_version_tree_matrix            julia_allgo_new           False                  True                   False            False
      raw/assets/benchmark-results/julia_allGO_new_c2ef_validate_cosine_subspace_split_legacy_20260618                             unclassified            julia_allgo_new           False                  True                   False            False
```
