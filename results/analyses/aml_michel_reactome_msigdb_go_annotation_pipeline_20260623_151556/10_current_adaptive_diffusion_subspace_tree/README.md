# Current Adaptive Diffusion Subspace Tree Experiment

Input: `/Users/berksakalli/Downloads/feature_matrix_AML_michel_reactome_msigdb.tsv`
Method version: `current`
Tree geometry: `adaptive_diffusion_cosine_subspace`
Experiment directory: `results/analyses/aml_michel_reactome_msigdb_go_annotation_pipeline_20260623_151556/10_current_adaptive_diffusion_subspace_tree`

Directory layout:
- `rankings/`: experiment-level ranking, spectrum, block metadata, and long quality tables.
- `plots/`: experiment-level summary plots.
- `subspaces/<weighting>/<block_name>/`: one folder per subspace with assignments, tree, quality CSVs, coordinates, and axis term-loading plots.
- `aml_michel_reactome_msigdb_quality_aware_go_ic_by_method/current__adaptive_diffusion_cosine_subspace/`: method-separated PDF, PNG pages, and copied ranking CSV.
- `aml_michel_reactome_msigdb_quality_aware_go_ic_plots/`: all-tree/ordered PDFs for the experiment.
- `ARTIFACT_INDEX.md`, `artifact_index.csv`, `subspace_plot_index.csv`, and `connected_results_manifest.json`: connected artifact tables.

Ranking:
- `display_rank` equals `specificity_aware_rank` for completed rows.
- `specificity_aware_rank` sorts by quality tier, cluster specificity score, specific-cluster fraction, weighted specificity delta, then lower GO-BIC active/gene.
- `old_display_rank` preserves the older quality-tier then GO-BIC order.
- `raw_go_ic_rank` preserves the raw GO-IC order for audit.
- `go_bic_active_per_gene` is lower-is-better only within comparable quality tiers.

Axis term loadings:
- `axis_term_loadings_all.csv` stores every GO term loading for every cosine mode in the subspace.
- `axis_top_terms.csv` stores the top positive, negative, and absolute GO-term loadings per axis.
- Positive and negative signs are orientation-dependent; the absolute loading is the stable strength score.

Ranking CSV: `results/analyses/aml_michel_reactome_msigdb_go_annotation_pipeline_20260623_151556/10_current_adaptive_diffusion_subspace_tree/rankings/current_adaptive_diffusion_subspace_tree_ranking.csv`
Artifact index: `results/analyses/aml_michel_reactome_msigdb_go_annotation_pipeline_20260623_151556/10_current_adaptive_diffusion_subspace_tree/ARTIFACT_INDEX.md`

Top ranked subspace:
- `binary / adaptive_modes_02_05`
- clusters: `5`
- quality tier: `quality_plausible`
- GO-BIC active/gene: `885.300327`
- coherent clusters: `5/5`
