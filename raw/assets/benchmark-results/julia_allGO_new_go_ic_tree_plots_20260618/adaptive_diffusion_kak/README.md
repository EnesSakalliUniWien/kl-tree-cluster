# Adaptive Cosine/KAK Separated-Space Diffusion Probe

Schema: `adaptive_cosine_kak_benchmark_probe/v1/matrix_block_diffusion`
Input: `data/feature_matrices/feature_matrix_julia_allGO_new.tsv`
Rows x columns: `602 x 6368`
Reference labels: `None`
Edge alpha: `0.001`
Sibling alpha: `0.01`
Diffusion k: `15`
Diffusion time: `3`
Diffusion components: `30`
Diffusion mode: `adaptive`
Adaptive bandwidth type: `-1/(d+2)`
Adaptive epsilon: `median`
Weightings: `binary, tfidf`

## Status Counts

{'ok': 14, 'failed_gate': 1}

## Compact Ok Rows

weighting              block_name n_clusters  largest_cluster_fraction  singleton_fraction  reference_ari  reference_nmi
   binary adaptive_common_mode_01         12                  0.290698            0.000000            NaN            NaN
   binary    adaptive_modes_07_11        563                  0.011628            0.955595            NaN            NaN
   binary    adaptive_modes_12_21         46                  0.126246            0.108696            NaN            NaN
   binary    adaptive_modes_22_40         56                  0.056478            0.142857            NaN            NaN
   binary    adaptive_modes_41_55         65                  0.079734            0.169231            NaN            NaN
   binary    adaptive_modes_56_80         60                  0.141196            0.200000            NaN            NaN
    tfidf adaptive_common_mode_01        524                  0.014950            0.912214            NaN            NaN
    tfidf    adaptive_modes_02_05         40                  0.232558            0.125000            NaN            NaN
    tfidf    adaptive_modes_06_10        235                  0.014950            0.331915            NaN            NaN
    tfidf    adaptive_modes_11_15        236                  0.019934            0.372881            NaN            NaN
    tfidf    adaptive_modes_16_19         69                  0.112957            0.246377            NaN            NaN
    tfidf    adaptive_modes_20_31        249                  0.034884            0.522088            NaN            NaN
    tfidf    adaptive_modes_32_52        592                  0.004983            0.984797            NaN            NaN
    tfidf    adaptive_modes_53_80         22                  0.372093            0.000000            NaN            NaN
