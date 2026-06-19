# Adaptive Cosine/KAK Matrix Probe

Schema: `adaptive_cosine_kak_benchmark_probe/v1/matrix`
Input: `data/feature_matrices/feature_matrix_julia_allGO_new.tsv`
Rows x columns: `602 x 6368`
Edge alpha: `0.001`
Sibling alpha: `0.01`
Internal support thresholds enforced: `False`
Max rank: `80`
Weightings: `binary, tfidf`

## Status Counts

{'ok': 15}

## Compact Ok Rows

weighting              block_name            block_type  block_start  block_end  block_energy_fraction  n_clusters  largest_cluster_fraction  singleton_fraction
   binary adaptive_common_mode_01           common_mode            1          1               0.152978           3                  0.548173            0.000000
   binary    adaptive_modes_02_06 adaptive_decay_regime            2          6               0.231756         108                  0.117940            0.425926
   binary    adaptive_modes_07_11 adaptive_decay_regime            7         11               0.100801         531                  0.013289            0.920904
   binary    adaptive_modes_12_21 adaptive_decay_regime           12         21               0.123429         362                  0.021595            0.729282
   binary    adaptive_modes_22_40 adaptive_decay_regime           22         40               0.159083         399                  0.013289            0.719298
   binary    adaptive_modes_41_55 adaptive_decay_regime           41         55               0.097983         592                  0.004983            0.984797
   binary    adaptive_modes_56_80 adaptive_decay_regime           56         80               0.133970           1                  1.000000            0.000000
    tfidf adaptive_common_mode_01           common_mode            1          1               0.117386         412                  0.014950            0.822816
    tfidf    adaptive_modes_02_05 adaptive_decay_regime            2          5               0.181947         256                  0.058140            0.691406
    tfidf    adaptive_modes_06_10 adaptive_decay_regime            6         10               0.112800         233                  0.064784            0.648069
    tfidf    adaptive_modes_11_15 adaptive_decay_regime           11         15               0.073731         214                  0.013289            0.299065
    tfidf    adaptive_modes_16_19 adaptive_decay_regime           16         19               0.049945          76                  0.088040            0.197368
    tfidf    adaptive_modes_20_31 adaptive_decay_regime           20         31               0.121887         203                  0.021595            0.334975
    tfidf    adaptive_modes_32_52 adaptive_decay_regime           32         52               0.167002         172                  0.019934            0.232558
    tfidf    adaptive_modes_53_80 adaptive_decay_regime           53         80               0.175303         133                  0.019934            0.030075
