# aml_michel_reactome_msigdb GO Annotation Analysis

Input matrix: `/Users/berksakalli/Downloads/feature_matrix_AML_michel_reactome_msigdb.tsv`

Dataset shape: `150` genes x `1665` features.

Main output directory:
`results/analyses/aml_michel_reactome_msigdb_go_annotation_pipeline_20260623_151556/10_current_adaptive_diffusion_subspace_tree`

Key outputs:

- `10_current_adaptive_diffusion_subspace_tree/systematic_subspace_gene_annotations/aml_michel_reactome_msigdb_systematic_subspace_gene_annotation_report.pdf`
- `10_current_adaptive_diffusion_subspace_tree/systematic_subspace_gene_annotations/aml_michel_reactome_msigdb_radial_tree_clusters.pdf`
- `10_current_adaptive_diffusion_subspace_tree/systematic_subspace_gene_annotations/subspace_cluster_roster.csv`
- `10_current_adaptive_diffusion_subspace_tree/systematic_subspace_gene_annotations/subspace_gene_membership_long.csv`
- `10_current_adaptive_diffusion_subspace_tree/systematic_subspace_gene_annotations/subspace_cluster_status.csv`
- `10_current_adaptive_diffusion_subspace_tree/systematic_subspace_gene_annotations/cluster_meaningfulness_audit/coherence_and_top_annotations_summary.md`
- `10_current_adaptive_diffusion_subspace_tree/systematic_subspace_gene_annotations/cluster_meaningfulness_audit/highest_occurring_nonobsolete_annotations.csv`
- `10_current_adaptive_diffusion_subspace_tree/systematic_subspace_gene_annotations/cluster_meaningfulness_audit/highest_occurring_annotations_all_terms.csv`

Checks:

- Subspace assignment sources: `{'accepted_tbs': 12}`.
- All subspace gene-membership tables assign `150/150` genes: `True`.
- Radial tree PNGs recorded: `12`.
- The report PDF has `24` pages.
- Cluster meaningfulness audit covers `1105` clusters across `12` subspaces.
- Strict eigenband coherence counts: `{'above_null_mean_not_strong': 5, 'coherent': 1, 'insufficient_tested_clusters': 5, 'null_like_or_weak': 1}`.

Highest recurring non-obsolete strong annotations:

1. Synthesis of PC
2. IGF1R Signaling Cascade
3. Mitotic Telophase Cytokinesis
4. Transport of Mature mRNA Derived From an Intron-Containing Transcript
5. Epigenetic Regulation of Gene Expression
6. Interleukin-6 Family Signaling
7. CTNNB1 S45 Mutants Aren'T Phosphorylated
8. G2 M Transition
9. Defective Pyroptosis
10. RAS Signaling Downstream of NF1 Loss-Of-Function Variants
