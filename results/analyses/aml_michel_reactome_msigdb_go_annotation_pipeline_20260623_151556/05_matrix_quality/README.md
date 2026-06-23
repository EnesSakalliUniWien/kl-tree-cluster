# Feature Matrix Quality Analysis

Input: `/Users/berksakalli/Downloads/feature_matrix_AML_michel_reactome_msigdb.tsv`

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
- Matrix has `150` genes and `1665` GO terms with density `0.0844`.
- `335` GO terms have support <= 2 genes; these are overfitting-prone for GO-IC.
- Median GO-term Bernoulli entropy is `0.3274` bits; `191` terms are <= 0.10 bits.
- Total marginal GO-term entropy is `599.6` bits across `1665` terms.
- `9` GO terms occur in at least half of genes; these broad terms can dominate global geometry.
- Median gene has `119.0` active GO terms.
