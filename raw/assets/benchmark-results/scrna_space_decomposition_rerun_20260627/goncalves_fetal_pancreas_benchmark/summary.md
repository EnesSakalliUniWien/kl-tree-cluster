# Goncalves fetal pancreas progenitor benchmark

Generated at: 2026-06-27T21:29:15+02:00

## Data

- Dataset: Goncalves et al. human fetal pancreas development, UCSC Cell Browser
  fetal-pancreas matrix.
- Expression source: `https://cells.ucsc.edu/human-pancreas-dev/fetal-pancreas/exprMatrix.tsv.gz`.
- Metadata source: `https://cells.ucsc.edu/human-pancreas-dev/fetal-pancreas/meta.tsv`.
- Local expression path: `/Users/berksakalli/Projects/kl-te-cluster/raw/inbox/goncalves_human_pancreas_dev/fetal-pancreas/exprMatrix.tsv.gz`.
- Local metadata path: `/Users/berksakalli/Projects/kl-te-cluster/raw/inbox/goncalves_human_pancreas_dev/fetal-pancreas/meta.tsv`.
- Shape: 1465 cells x 2000 genes after feature selection.
- Input expression kind: `processed_scaled`.
- Cell-type labels: 8.
- Batches/samples: 4.

Label counts:

- trunk: 434
- mesenchyme: 349
- proliferating: 289
- tip: 188
- blood: 77
- unknown: 74
- endocrine: 28
- neurons: 26

## Workflow

The UCSC expression matrix is processed/scaled expression rather than raw counts, so the script preserves it in `layers["input_expression"]`, uses metadata `nCount_RNA` and `nFeature_RNA` for count QC summaries, skips count normalization/log1p, selects high-variance genes, and recomputes PCA, neighbors, UMAP, and Leiden clustering. The benchmark then runs the same classical and TBS method set
used for the adult pancreas comparison.

Requested PCs: 30. Effective PCs: 30. Maximum benchmark cells:
2500. Seed: 0. Elapsed seconds: 41.22.

TBS settings are sibling alpha `0.01`, edge alpha `0.001`, local adaptive
projected-Wald dimensions at 90% contrast energy, and separate topology-only,
recomputed-NNLS branch-time, and raw-linkage diagnostic rows.

## QC Notes

- Raw counts preserved in `layers["counts"]`: false.
- Input expression preserved in `layers["input_expression"]`: true.
- Raw/log-normalized snapshot present: True.
- Mitochondrial genes detected by `MT-` prefix: 0.
- Doublet status: not_run_prepackaged_post_count_object; R/Bioconductor scDblFinder was not available in this Python-only benchmark run.
- Ambient RNA status: not_run_no_empty_droplet_channel_inputs_in_prepackaged_h5ad.

## Results

| method                                                                                                 | status   |   significance_level |   edge_alpha |   n_clusters |   overcluster_ratio |   weighted_cluster_purity |   weighted_label_dominant_cluster_recall |   merge_error_rate |   split_error_rate |   v_measure |    nmi |    ari |   silhouette |   elapsed_sec | skip_reason   |
|:-------------------------------------------------------------------------------------------------------|:---------|---------------------:|-------------:|-------------:|--------------------:|--------------------------:|-----------------------------------------:|-------------------:|-------------------:|------------:|-------:|-------:|-------------:|--------------:|:--------------|
| Louvain                                                                                                | ok       |             nan      |     nan      |           12 |              1.5000 |                    0.6846 |                                   0.5092 |             0.3154 |             0.4908 |      0.4431 | 0.4431 | 0.3264 |       0.1209 |        0.1683 |               |
| Leiden                                                                                                 | ok       |             nan      |     nan      |           11 |              1.3750 |                    0.6669 |                                   0.5044 |             0.3331 |             0.4956 |      0.4394 | 0.4394 | 0.3126 |       0.1246 |        0.1275 |               |
| TBS adaptive diffusion topology projected adaptive-k90 alpha=0.01 edge=0.001                           | ok       |               0.0100 |       0.0010 |           24 |              3.0000 |                    0.6546 |                                   0.5399 |             0.3454 |             0.4601 |      0.4100 | 0.4100 | 0.2058 |       0.0817 |        3.8735 |               |
| TBS adaptive diffusion branch-time recomputed-NNLS projected adaptive-k90 alpha=0.01 edge=0.001        | ok       |               0.0100 |       0.0010 |           24 |              3.0000 |                    0.6546 |                                   0.5399 |             0.3454 |             0.4601 |      0.4100 | 0.4100 | 0.2058 |       0.0817 |        5.3808 |               |
| TBS adaptive diffusion raw-linkage branch-time diagnostic projected adaptive-k90 alpha=0.01 edge=0.001 | ok       |               0.0100 |       0.0010 |           24 |              3.0000 |                    0.6546 |                                   0.5399 |             0.3454 |             0.4601 |      0.4100 | 0.4100 | 0.2058 |       0.0817 |        3.5926 |               |
| Spectral true K                                                                                        | ok       |             nan      |     nan      |            8 |              1.0000 |                    0.5980 |                                   0.6164 |             0.4020 |             0.3836 |      0.3998 | 0.3998 | 0.3410 |       0.1145 |        0.0800 |               |
| TBS topology projected adaptive-k90 alpha=0.01 edge=0.001                                              | ok       |               0.0100 |       0.0010 |           36 |              4.5000 |                    0.6621 |                                   0.4464 |             0.3379 |             0.5536 |      0.3939 | 0.3939 | 0.2486 |       0.0446 |        3.8361 |               |
| K-means true K                                                                                         | ok       |             nan      |     nan      |            8 |              1.0000 |                    0.5870 |                                   0.5488 |             0.4130 |             0.4512 |      0.3851 | 0.3851 | 0.2988 |       0.1243 |        0.0924 |               |
| HDBSCAN                                                                                                | ok       |             nan      |     nan      |            4 |              0.5000 |                    0.3590 |                                   0.5857 |             0.6410 |             0.4143 |      0.1932 | 0.1932 | 0.0589 |       0.0524 |        0.0825 |               |
| TBS branch-time recomputed-NNLS projected adaptive-k90 alpha=0.01 edge=0.001                           | ok       |               0.0100 |       0.0010 |            1 |              0.1250 |                    0.2962 |                                   1.0000 |             0.7038 |             0.0000 |      0.0000 | 0.0000 | 0.0000 |     nan      |        5.5726 |               |
| TBS raw-linkage branch-time diagnostic projected adaptive-k90 alpha=0.01 edge=0.001                    | ok       |               0.0100 |       0.0010 |            1 |              0.1250 |                    0.2962 |                                   1.0000 |             0.7038 |             0.0000 |      0.0000 | 0.0000 | 0.0000 |     nan      |        3.4058 |               |

## Files

- `goncalves_fetal_pancreas_classical_pipeline.h5ad`
- `qc_cell_metrics.csv`
- `qc_metric_distributions.png`
- `method_metrics.csv`
- `method_assignments.csv`
- `celltype_fragmentation_by_method.csv`
- `cluster_composition_by_method.csv`
- `method_umap_clusters.png`
- `method_split_merge_diagnostic.png`
- `edge_gate_distance_time_model_analysis.md`
- `tbs_tree_branch_length_summary.csv`
- `*_tree_edges.csv`
- `*_traversal_trace.csv`
- `*_full_edge_traversal_trace.csv`
- `manifest.json`
