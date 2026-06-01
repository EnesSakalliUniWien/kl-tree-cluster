# HC_feature_matrix_GO_CC.tsv

Binary gene × pathway membership matrix for **hypertrophic cardiomyopathy (HC)** associated genes, annotated with **Reactome** pathway terms (labelled GO CC in the filename for historical reasons).

| Property | Value |
|----------|-------|
| Rows | 317 genes (HGNC symbols) |
| Columns | 1 834 Reactome pathways |
| Values | Binary (0/1) — 1 = gene is annotated to that pathway |
| Density | 3.1 % (sparse) |
| Mean pathways/gene | 57.4 (median 29, range 1–458) |
| Mean genes/pathway | 9.9 (median 6, range 1–147) |
| Format | Tab-separated, first column = gene symbol (index) |

## Example genes

`KCND3`, `MYL4`, `TAZ`, `CACNB2`, `LMNA`, `RYR2`, `SCN5A`, `KCNQ1`, `DSC2`, `DSG2`

## Example pathways (columns)

- Muscle Contraction
- Cardiac Conduction
- Striated Muscle Contraction
- Neuronal System
- Signaling by Receptor Tyrosine Kinases

## Usage

```python
import pandas as pd

data = pd.read_csv("data/feature_matrices/HC_feature_matrix_GO_CC.tsv", sep="\t", index_col=0)
# data.shape → (317, 1834)
```

## Clustering Results

KL-tree decomposition was run at three significance levels. Results are stored alongside this file:

| α level | Directory | Clusters | Key files |
|---------|-----------|----------|-----------|
| 0.01 | `benchmarks/results/02_hc_cms_go_runs/data_alpha_runs/results_GO_CC_alpha_001/` | 79 | cluster_assignments.csv, data_with_clusters.tsv, tree_clusters.png/pdf, umap_clusters.png |
| 0.05 | `benchmarks/results/02_hc_cms_go_runs/data_alpha_runs/results_GO_CC_alpha_005/` | 85 | cluster_assignments.csv, data_with_clusters.tsv, tree_clusters.png/pdf, umap_clusters.png |
| 0.10 | `benchmarks/results/02_hc_cms_go_runs/data_alpha_runs/results_GO_CC_alpha_010/` | 85 | cluster_assignments.csv, data_with_clusters.tsv, tree_clusters.png/pdf, umap_clusters.png |
| default | `benchmarks/results/02_hc_cms_go_runs/data_alpha_runs/results_GO_CC_alpha_default/` | 85 | cluster_assignments.csv, umap_clusters.png |

### Output files per run

| File | Description |
|------|-------------|
| `cluster_assignments.csv` | Gene → cluster_id mapping with cluster root node and size |
| `data_with_clusters.tsv` | Original binary matrix with `cluster_id` column prepended, sorted by cluster |
| `cluster_sizes.csv` | Cluster ID and number of genes per cluster |
| `summary.json` | Run parameters (alpha, distance metric, linkage method) and output paths |
| `umap_clusters.png` | 2-D UMAP embedding coloured by cluster assignment |
| `tree_clusters.png` | Hierarchical tree with cluster colours, edge significance styling, and node halos |
| `tree_clusters.pdf` | Vector version of the tree plot |

### Biological analysis (α = 0.05)

The `benchmarks/results/02_hc_cms_go_runs/data_alpha_runs/results_GO_CC_alpha_005/bio_analysis/` subdirectory contains:

| File | Description |
|------|-------------|
| `pathway_enrichment_full.csv` | Fisher exact test (BH-corrected) for every pathway × cluster combination |
| `pathway_enrichment_top1_per_cluster.csv` | Top enriched pathway per cluster |
| `cluster_gene_lists.csv` | Genes per cluster with shared-pathway annotations |
| `cluster_sizes_histogram.png` | Histogram of cluster sizes |
| `cluster_jaccard_heatmap.png` | Pairwise Jaccard similarity of pathway profiles between clusters |
| `top_pathways_by_cluster.png` | Top 30 pathways stacked bar chart coloured by cluster |
| `biological_summary.txt` | High-level biological interpretation |

### Clustering summary

- **1 dominant cluster** (174 genes, 55%) — genes with sparse / generic pathway annotations
- **24 smaller clusters** (2–11 genes) — functionally coherent gene modules
- **60 singletons** — genes with unique pathway profiles
- **Key biological themes**: ion channels (KCNJ, SCN, CACN families), sarcomere / muscle contraction, desmosomes (DSC2, DSG2, PKP), nuclear pore (NUP), and signal transduction

### Reproducing

```bash
# Clustering
python scripts/analysis/run_feature_matrix_with_umap.py \
    --input data/feature_matrices/HC_feature_matrix_GO_CC.tsv \
    --output-dir benchmarks/results/02_hc_cms_go_runs/data_alpha_runs/results_GO_CC_alpha_005 \
    --edge-alpha 0.05 --sibling-alpha 0.05

# Biological analysis
python scripts/analysis/analyze_hc_clusters.py \
    --feature-matrix data/feature_matrices/HC_feature_matrix_GO_CC.tsv \
    --assignments benchmarks/results/02_hc_cms_go_runs/data_alpha_runs/results_GO_CC_alpha_005/cluster_assignments.csv \
    -o benchmarks/results/02_hc_cms_go_runs/data_alpha_runs/results_GO_CC_alpha_005/bio_analysis
```

## Note

This file is the canonical tracked HC gene-pathway matrix. The duplicate `HC_feature_matrix_Reactome_Pathways.tsv` filename was removed to avoid two names for the same matrix.
