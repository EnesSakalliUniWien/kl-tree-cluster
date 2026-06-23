# aml_michel_reactome_msigdb cluster meaningfulness audit

## Method

This is an internal GO-coherence audit. For each cluster, the script tests whether
GO features are over-represented relative to the full feature matrix using a
one-sided hypergeometric test with Benjamini-Hochberg correction within the
cluster. It then compares each subspace to random partitions that preserve the
observed cluster-size distribution.

- Minimum tested cluster size: `3` genes.
- Minimum genes supporting an interpreted GO term: `2`.
- Significant enrichment threshold: `q <= 0.05`.
- Strong label: `q <= 0.01`, lift `>= 2.0`, and at least `3` supporting genes.
- Null iterations per subspace: `30`.

Rows with `assignment_source=diagnostic_linkage_cut` are failed-gate diagnostics,
not accepted final TBS assignments.

## Assignment-source summary

| assignment_source   |   n_subspaces |   mean_strong_fraction |   mean_enriched_fraction |   median_empirical_p_enriched_fraction |   median_empirical_p_strong_fraction |   mean_null_enriched_fraction |   mean_null_strong_fraction | eigenband_coherence_counts                                                                 |
|:--------------------|--------------:|-----------------------:|-------------------------:|---------------------------------------:|-------------------------------------:|------------------------------:|----------------------------:|:-------------------------------------------------------------------------------------------|
| accepted_tbs        |            12 |                  0.507 |                    0.945 |                                  0.452 |                                0.339 |                         0.859 |                       0.287 | above_null_mean_not_strong:5;coherent:1;insufficient_tested_clusters:5;null_like_or_weak:1 |

## Subspace summary

|   display_rank | weighting   | block_name              | assignment_source   |   n_tested_clusters |   strong_fraction |   enriched_fraction |   null_enriched_fraction_mean |   empirical_p_enriched_fraction |   empirical_p_strong_fraction | eigenband_coherence          |
|---------------:|:------------|:------------------------|:--------------------|--------------------:|------------------:|--------------------:|------------------------------:|--------------------------------:|------------------------------:|:-----------------------------|
|              1 | binary      | adaptive_modes_02_05    | accepted_tbs        |                   5 |             1     |               1     |                         0.84  |                          0.452  |                        0.0323 | above_null_mean_not_strong   |
|              2 | tfidf       | adaptive_modes_11_21    | accepted_tbs        |                   5 |             0.8   |               1     |                         0.787 |                          0.258  |                        0.0645 | above_null_mean_not_strong   |
|              3 | tfidf       | adaptive_modes_06_10    | accepted_tbs        |                  27 |             0.704 |               1     |                         0.891 |                          0.0323 |                        0.0323 | coherent                     |
|              4 | binary      | adaptive_modes_52_80    | accepted_tbs        |                  11 |             0.636 |               1     |                         0.93  |                          0.452  |                        0.0323 | above_null_mean_not_strong   |
|              5 | tfidf       | adaptive_modes_63_80    | accepted_tbs        |                  30 |             0.167 |               0.933 |                         0.867 |                          0.29   |                        0.774  | above_null_mean_not_strong   |
|              6 | tfidf       | adaptive_modes_02_05    | accepted_tbs        |                   4 |             0.25  |               0.75  |                         0.85  |                          0.871  |                        0.742  | null_like_or_weak            |
|              7 | binary      | adaptive_modes_06_09    | accepted_tbs        |                   0 |           nan     |             nan     |                       nan     |                        nan      |                      nan      | insufficient_tested_clusters |
|              8 | binary      | adaptive_modes_10_21    | accepted_tbs        |                   0 |           nan     |             nan     |                       nan     |                        nan      |                      nan      | insufficient_tested_clusters |
|              9 | tfidf       | adaptive_modes_22_62    | accepted_tbs        |                   0 |           nan     |             nan     |                       nan     |                        nan      |                      nan      | insufficient_tested_clusters |
|             10 | binary      | adaptive_modes_22_51    | accepted_tbs        |                   0 |           nan     |             nan     |                       nan     |                        nan      |                      nan      | insufficient_tested_clusters |
|             11 | tfidf       | adaptive_common_mode_01 | accepted_tbs        |                   2 |             0.5   |               1     |                         0.867 |                          0.742  |                        0.613  | insufficient_tested_clusters |
|             12 | binary      | adaptive_common_mode_01 | accepted_tbs        |                   8 |             0     |               0.875 |                         0.838 |                          0.645  |                        1      | above_null_mean_not_strong   |

## Strong examples

|   display_rank | weighting   | block_name           | assignment_source   |   cluster_id |   cluster_size | top_term                                                                   | top_go_id   |   best_q_value |   top_lift |   top_hits |
|---------------:|:------------|:---------------------|:--------------------|-------------:|---------------:|:---------------------------------------------------------------------------|:------------|---------------:|-----------:|-----------:|
|              1 | binary      | adaptive_modes_02_05 | accepted_tbs        |            0 |             21 | IGF1R Signaling Cascade                                                    |             |       9.27e-26 |       7.14 |         21 |
|              1 | binary      | adaptive_modes_02_05 | accepted_tbs        |            1 |             20 | Mitotic Telophase Cytokinesis                                              |             |       3.41e-17 |       4.84 |         20 |
|              1 | binary      | adaptive_modes_02_05 | accepted_tbs        |            2 |             19 | Transport of Mature mRNA Derived From an Intron-Containing Transcript      |             |       5.27e-16 |       6.1  |         17 |
|              1 | binary      | adaptive_modes_02_05 | accepted_tbs        |            3 |             43 | Epigenetic Regulation of Gene Expression                                   |             |       2.63e-14 |       2.26 |         37 |
|              2 | tfidf       | adaptive_modes_11_21 | accepted_tbs        |            1 |             26 | Interleukin-6 Family Signaling                                             |             |       1.49e-10 |       5.77 |         12 |
|              2 | tfidf       | adaptive_modes_11_21 | accepted_tbs        |            2 |              8 | CTNNB1 S45 Mutants Aren'T Phosphorylated                                   |             |       1.62e-10 |      12.5  |          8 |
|              1 | binary      | adaptive_modes_02_05 | accepted_tbs        |            4 |             47 | G2 M Transition                                                            |             |       6.88e-10 |       2.57 |         25 |
|              2 | tfidf       | adaptive_modes_11_21 | accepted_tbs        |            0 |              9 | Defective Pyroptosis                                                       |             |       1.17e-09 |      11.1  |          8 |
|              2 | tfidf       | adaptive_modes_11_21 | accepted_tbs        |            3 |             22 | RAS Signaling Downstream of NF1 Loss-Of-Function Variants                  |             |       1.38e-08 |       6.2  |         10 |
|              3 | tfidf       | adaptive_modes_06_10 | accepted_tbs        |           38 |              8 | Processive Synthesis on the C-strand of the Telomere                       |             |       6.27e-07 |       7.72 |          7 |
|              3 | tfidf       | adaptive_modes_06_10 | accepted_tbs        |           48 |              4 | Oxygen-dependent Proline Hydroxylation of Hypoxia-inducible Factor Alpha   |             |       1.04e-06 |      25    |          4 |
|              6 | tfidf       | adaptive_modes_02_05 | accepted_tbs        |           37 |              6 | SLBP Dependent Processing of Replication-Dependent Histone Pre-mRNAs       |             |       1.32e-06 |      13.9  |          5 |
|              3 | tfidf       | adaptive_modes_06_10 | accepted_tbs        |           31 |              6 | RNA Polymerase II Transcription Termination                                |             |       1.33e-06 |       8.33 |          6 |
|              3 | tfidf       | adaptive_modes_06_10 | accepted_tbs        |           27 |              3 | Uptake and Function of Anthrax Toxins                                      |             |       2.55e-06 |      50    |          3 |
|              3 | tfidf       | adaptive_modes_06_10 | accepted_tbs        |           64 |              4 | Downregulation of ERBB4 Signaling                                          |             |       8.27e-06 |      16.7  |          4 |
|              3 | tfidf       | adaptive_modes_06_10 | accepted_tbs        |           37 |              3 | Class I Peroxisomal Membrane Protein Import                                |             |       8.33e-06 |      37.5  |          3 |
|              3 | tfidf       | adaptive_modes_06_10 | accepted_tbs        |           32 |              5 | mRNA 3'-End Processing                                                     |             |       1.19e-05 |       8.82 |          5 |
|              3 | tfidf       | adaptive_modes_06_10 | accepted_tbs        |            5 |              3 | HCMV Late Events                                                           |             |       1.99e-05 |      30    |          3 |
|              4 | binary      | adaptive_modes_52_80 | accepted_tbs        |            2 |              5 | CaMK IV-mediated Phosphorylation of CREB                                   |             |       3.33e-05 |      30    |          3 |
|              3 | tfidf       | adaptive_modes_06_10 | accepted_tbs        |           18 |              5 | Regulation of Lipid Metabolism by PPARalpha                                |             |       3.43e-05 |       7.5  |          5 |
|              3 | tfidf       | adaptive_modes_06_10 | accepted_tbs        |           10 |              4 | Recruitment of Mitotic Centrosome Proteins and Complexes                   |             |       3.66e-05 |      11.5  |          4 |
|              4 | binary      | adaptive_modes_52_80 | accepted_tbs        |            1 |              6 | Extra-nuclear Estrogen Signaling                                           |             |       4.07e-05 |       5.36 |          6 |
|              3 | tfidf       | adaptive_modes_06_10 | accepted_tbs        |           53 |              4 | MECP2 Regulates Transcription of Neuronal Ligands                          |             |       8.61e-05 |      10    |          4 |
|              3 | tfidf       | adaptive_modes_06_10 | accepted_tbs        |           26 |              3 | Ras Activation Upon Ca2+ Influx Through NMDA Receptor                      |             |       0.000123 |      18.8  |          3 |
|              3 | tfidf       | adaptive_modes_06_10 | accepted_tbs        |           50 |              4 | Synthesis of PC                                                            |             |       0.000128 |       8.82 |          4 |
|              5 | tfidf       | adaptive_modes_63_80 | accepted_tbs        |            4 |              3 | APC C Cdc20 Mediated Degradation of Securin                                |             |       0.00014  |      18.8  |          3 |
|              3 | tfidf       | adaptive_modes_06_10 | accepted_tbs        |           47 |              3 | SCF-beta-TrCP Mediated Degradation of Emi1                                 |             |       0.000192 |      16.7  |          3 |
|              3 | tfidf       | adaptive_modes_06_10 | accepted_tbs        |           19 |              5 | Epigenetic Regulation of Gene Expression by MLL3 and MLL4 Complexes        |             |       0.000285 |       5    |          5 |
|              3 | tfidf       | adaptive_modes_06_10 | accepted_tbs        |           41 |              3 | RUNX1 Regulates Transcription of Genes Involved in Differentiation of HSCs |             |       0.000337 |      13.6  |          3 |
|              3 | tfidf       | adaptive_modes_06_10 | accepted_tbs        |           57 |              4 | DNA Methylation                                                            |             |       0.000652 |      12.5  |          3 |

## Weak or broad examples

|   display_rank | weighting   | block_name              | assignment_source   |   cluster_id |   cluster_size | meaningfulness_label   | top_term                                                       | top_go_id   |   best_q_value |   top_lift |
|---------------:|:------------|:------------------------|:--------------------|-------------:|---------------:|:-----------------------|:---------------------------------------------------------------|:------------|---------------:|-----------:|
|              5 | tfidf       | adaptive_modes_63_80    | accepted_tbs        |           18 |              3 | weak_or_not_enriched   | Estrogen-dependent Gene Expression                             |             |       0.373    |       1.69 |
|             12 | binary      | adaptive_common_mode_01 | accepted_tbs        |          103 |              3 | weak_or_not_enriched   | SUMOylation of DNA Damage Response and Repair Proteins         |             |       0.0854   |       2.31 |
|              5 | tfidf       | adaptive_modes_63_80    | accepted_tbs        |           56 |              3 | weak_or_not_enriched   | Chk1 Chk2(Cds1) Mediated Inactivation of Cyclin B Cdk1 Complex |             |       0.0548   |       5    |
|              2 | tfidf       | adaptive_modes_11_21    | accepted_tbs        |            4 |             85 | statistical_but_broad  | Cell Cycle, Mitotic                                            |             |       0.000122 |       1.31 |

## Files

- `cluster_meaningfulness.csv`: one row per cluster.
- `subspace_meaningfulness_summary.csv`: one row per subspace/eigenband.
- `assignment_source_meaningfulness_summary.csv`: accepted TBS versus diagnostic linkage-cut summary.
- `null_partition_summary.csv`: size-preserving null partition summaries.
- `top_meaningful_cluster_examples.csv`: strongest cluster examples.
- `weak_or_broad_cluster_examples.csv`: weakest or broadest examples.
- `quickgo_top_term_ids.txt`: GO IDs worth checking in QuickGO for top examples.

## Cluster label counts

{'moderate': 45, 'statistical_but_broad': 1, 'strong': 42, 'too_small': 1013, 'untested': 1, 'weak_or_not_enriched': 3}
