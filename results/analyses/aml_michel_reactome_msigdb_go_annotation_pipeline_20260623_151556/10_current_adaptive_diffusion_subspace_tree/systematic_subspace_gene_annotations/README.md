# Systematic Subspace Gene Annotation Report

Dataset: `aml_michel_reactome_msigdb`
Feature matrix: `/Users/berksakalli/Downloads/feature_matrix_AML_michel_reactome_msigdb.tsv`
Experiment directory: `results/analyses/aml_michel_reactome_msigdb_go_annotation_pipeline_20260623_151556/10_current_adaptive_diffusion_subspace_tree`

Files:
- `aml_michel_reactome_msigdb_systematic_subspace_gene_annotation_report.pdf`
  - each accepted TBS or diagnostic linkage-cut subspace has one plot page followed by one annotation page
- `subspace_cluster_gene_annotation_summary.csv`
- `subspace_cluster_roster.csv`
- `subspace_gene_membership_long.csv`
- `subspace_cluster_status.csv`
- `subspaces/`: one directory per accepted or diagnostic subspace with roster, annotations, source artifacts, radial tree, full-space embedding, subspace embedding, and tree-distance embedding when available
- `quickgo_term_interpretations.csv`
- `uniprot_representative_gene_interpretations.csv`
- `life_science_lookup_cache.json`

`assignment_source=accepted_tbs` rows are final accepted TBS assignments; `assignment_source=diagnostic_linkage_cut` rows are diagnostic linkage-tree cuts for failed gates.

Life-science interpretation uses QuickGO term records and UniProt reviewed human gene/protein lookups through `scripts/rest_request.py`.
