# GO Annotation Feature-Matrix Pipeline

Input: `/Users/berksakalli/Downloads/feature_matrix_AML_michel_reactome_msigdb.tsv`
Dry run: `False`

Canonical interpretation level:
- `30_canonical_current_subspace_pipeline`: current TBS gates on adaptive-diffusion cosine subspace trees.

## Stages

### 00_dataset_inventory
- Analysis level: `dataset_inventory_and_naming_preflight`
- Output: `results/analyses/aml_michel_reactome_msigdb_go_annotation_pipeline_20260623_151556/00_dataset_inventory`
- Optional: `False`
- Command: `/Users/berksakalli/Projects/kl-te-cluster/.venv/bin/python3 /Users/berksakalli/Projects/kl-te-cluster/scripts/analysis/inventory_go_annotation_datasets.py --output-dir results/analyses/aml_michel_reactome_msigdb_go_annotation_pipeline_20260623_151556/00_dataset_inventory --extra-paths /Users/berksakalli/Downloads/feature_matrix_AML_michel_reactome_msigdb.tsv`

### 05_matrix_quality
- Analysis level: `00_matrix_quality_only`
- Output: `results/analyses/aml_michel_reactome_msigdb_go_annotation_pipeline_20260623_151556/05_matrix_quality`
- Optional: `False`
- Command: `/Users/berksakalli/Projects/kl-te-cluster/.venv/bin/python3 /Users/berksakalli/Projects/kl-te-cluster/scripts/analysis/feature_matrix_quality_analysis.py --input /Users/berksakalli/Downloads/feature_matrix_AML_michel_reactome_msigdb.tsv --output-dir results/analyses/aml_michel_reactome_msigdb_go_annotation_pipeline_20260623_151556/05_matrix_quality`

### 10_current_adaptive_diffusion_subspace_tree
- Analysis level: `30_canonical_current_subspace_pipeline`
- Output: `results/analyses/aml_michel_reactome_msigdb_go_annotation_pipeline_20260623_151556/10_current_adaptive_diffusion_subspace_tree`
- Optional: `False`
- Command: `/Users/berksakalli/Projects/kl-te-cluster/.venv/bin/python3 /Users/berksakalli/Projects/kl-te-cluster/scripts/analysis/run_current_adaptive_diffusion_subspace_tree_experiment.py --input /Users/berksakalli/Downloads/feature_matrix_AML_michel_reactome_msigdb.tsv --output-dir results/analyses/aml_michel_reactome_msigdb_go_annotation_pipeline_20260623_151556/10_current_adaptive_diffusion_subspace_tree --dataset-label aml_michel_reactome_msigdb --max-rank 80 --min-segment-length 4 --max-segments 8 --diffusion-k-neighbors 15 --diffusion-time 3 --diffusion-components 30 --adaptive-bandwidth-type=-1/(d+2) --adaptive-epsilon median --adaptive-metric euclidean --weightings binary tfidf`

### 40_subspace_rosters_radial_trees
- Analysis level: `40_systematic_subspace_annotation_package`
- Output: `results/analyses/aml_michel_reactome_msigdb_go_annotation_pipeline_20260623_151556/10_current_adaptive_diffusion_subspace_tree/systematic_subspace_gene_annotations`
- Optional: `False`
- Command: `/Users/berksakalli/Projects/kl-te-cluster/.venv/bin/python3 /Users/berksakalli/Projects/kl-te-cluster/scripts/analysis/export_subspace_cluster_rosters.py --experiment-dir results/analyses/aml_michel_reactome_msigdb_go_annotation_pipeline_20260623_151556/10_current_adaptive_diffusion_subspace_tree --feature-matrix /Users/berksakalli/Downloads/feature_matrix_AML_michel_reactome_msigdb.tsv --dataset-label aml_michel_reactome_msigdb`

### 45_subspace_annotation_pdf
- Analysis level: `40_systematic_subspace_annotation_package`
- Output: `results/analyses/aml_michel_reactome_msigdb_go_annotation_pipeline_20260623_151556/10_current_adaptive_diffusion_subspace_tree/systematic_subspace_gene_annotations`
- Optional: `False`
- Command: `/Users/berksakalli/Projects/kl-te-cluster/.venv/bin/python3 /Users/berksakalli/Projects/kl-te-cluster/scripts/analysis/build_subspace_gene_annotation_pdf.py --feature-matrix /Users/berksakalli/Downloads/feature_matrix_AML_michel_reactome_msigdb.tsv --experiment-dir results/analyses/aml_michel_reactome_msigdb_go_annotation_pipeline_20260623_151556/10_current_adaptive_diffusion_subspace_tree --dataset-label aml_michel_reactome_msigdb`

### 50_cluster_meaningfulness_audit
- Analysis level: `40_systematic_subspace_annotation_package`
- Output: `results/analyses/aml_michel_reactome_msigdb_go_annotation_pipeline_20260623_151556/10_current_adaptive_diffusion_subspace_tree/systematic_subspace_gene_annotations/cluster_meaningfulness_audit`
- Optional: `False`
- Command: `/Users/berksakalli/Projects/kl-te-cluster/.venv/bin/python3 /Users/berksakalli/Projects/kl-te-cluster/scripts/analysis/audit_go_cluster_meaningfulness.py --dataset-label aml_michel_reactome_msigdb --feature-matrix /Users/berksakalli/Downloads/feature_matrix_AML_michel_reactome_msigdb.tsv --annotation-root results/analyses/aml_michel_reactome_msigdb_go_annotation_pipeline_20260623_151556/10_current_adaptive_diffusion_subspace_tree/systematic_subspace_gene_annotations`

### 90_package_for_github
- Analysis level: `90_github_upload_package`
- Output: `results/analyses/aml_michel_reactome_msigdb_go_annotation_pipeline_20260623_151556`
- Optional: `False`
- Command: `internal:finalize_go_annotation_results --input /Users/berksakalli/Downloads/feature_matrix_AML_michel_reactome_msigdb.tsv --output-dir results/analyses/aml_michel_reactome_msigdb_go_annotation_pipeline_20260623_151556 --dataset-label aml_michel_reactome_msigdb`

## Analysis-Level Audit

- CSV: `results/analyses/aml_michel_reactome_msigdb_go_annotation_pipeline_20260623_151556/30_analysis_level_audit/go_annotation_analysis_level_inventory.csv`
- Markdown: `results/analyses/aml_michel_reactome_msigdb_go_annotation_pipeline_20260623_151556/30_analysis_level_audit/go_annotation_analysis_level_summary.md`
