# Feature Matrix Quality Analysis

Input: `data/feature_matrices/feature_matrix_julia_allGO_new.tsv`

Primary outputs:
- `feature_matrix_quality_report.pdf`
- `matrix_quality_summary.csv`
- `gene_annotation_burden.csv`
- `go_term_supports.csv`
- `entropy_summary.csv`
- `go_term_entropy_bins.csv`
- `gene_row_entropy_bins.csv`
- `feature_filter_sensitivity.csv`
- `pairwise_gene_similarity_summary.csv`
- `top_gene_cosine_pairs.csv`
- `top_go_term_cosine_pairs.csv`

Short interpretation:
- Matrix has `602` genes and `6368` GO terms with density `0.0249`.
- `1280` GO terms have support <= 2 genes; these are overfitting-prone for GO-IC.
- Median GO-term Bernoulli entropy is `0.1019` bits; `3180` terms are <= 0.10 bits.
- Total marginal GO-term entropy is `922.1` bits across `6368` terms.
- `0` GO terms occur in at least half of genes; these broad terms can dominate global geometry.
- Median gene has `112.5` active GO terms.
