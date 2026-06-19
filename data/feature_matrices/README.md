# Feature Matrices

Canonical input matrices used by clustering scripts and benchmark cases.

- `feature_matrix.tsv`: default feature matrix used by `scripts/analysis/run_feature_matrix_with_umap.py` and related diagnostics.
- `feature_matrix_julia_GOBP.tsv`: Julia GOBP matrix used in exploratory GO runs.
- `feature_matrix_julia_GOCC_GOBP_GOMF_combined.tsv`: combined Julia GOCC/GOBP/GOMF matrix used for the current blob analysis.
- `feature_matrix_julia_allGO_new.tsv`: active Julia allGO-new matrix used by the GO annotation feature-matrix pipeline.
- `feature_matrix_allGO_new_interactome.tsv`: active allGO-new interactome matrix used by the GO annotation feature-matrix pipeline.
- `CMS_*_feature_matrix.tsv`: CMS ontology/pathway matrices.
- `HC_feature_matrix_GO_CC.tsv`: HC gene-pathway matrix. This is the only tracked HC filename; the old `HC_feature_matrix_Reactome_Pathways.tsv` duplicate was removed.
