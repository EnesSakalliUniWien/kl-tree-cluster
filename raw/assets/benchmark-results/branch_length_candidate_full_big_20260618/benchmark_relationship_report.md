# Benchmark Relationship Report

- Source: `/Users/berksakalli/Projects/kl-te-cluster/raw/assets/benchmark-results/branch_length_candidate_full_big_20260618/full_benchmark_comparison.csv`
- Rows analyzed: `305`
- Unique cases: `121`
- Methods: `kl, kl_internal_filter_branch_length_v1, kl_legacy_c2ef9a69`
- Sections: `binary, categorical, gaussian, method_proof, overlapping, phylogenetic, sbm`
- Audit-backed rows: `0`

## Headline Findings

- Best average ARI: `kl` at `0.819` with exact-K rate `0.710`.
- Easiest section: `binary` (mean ARI `0.994`).
- Hardest section: `method_proof` (mean ARI `0.328`).
- Strongest method/section cell: `kl_internal_filter_branch_length_v1` on `binary` (mean ARI `0.998`, exact-K `0.958`).
- Weakest method/section cell: `kl_legacy_c2ef9a69` on `method_proof` (mean ARI `0.250`, exact-K `0.182`).

- Best outlier recovery: `kl` (mean outlier F1 `1.000`, singleton hit rate `1.000`, grouped recovery `0.667`).

## Pairwise Method Contrasts

- Largest ARI gain over `kl`: `kl_internal_filter_branch_length_v1` (`delta=-0.019`, `win_rate=0.076`, `n=79`).
- Weakest method relative to `kl`: `kl_legacy_c2ef9a69` (`delta=-0.077`, `win_rate=0.075`, `n=93`).
- Strongest head-to-head win rate: `kl` over `kl_legacy_c2ef9a69` (`win_rate=0.172`, `delta=0.077`).

## Method Summary

```text
                             method  n_rows  mean_ari  mean_nmi  mean_outlier_f1  singleton_hit_rate  grouped_recovery_rate  exact_k_rate  over_split_rate  under_split_rate
                                 kl      93     0.819     0.825            1.000               1.000                  0.667         0.710            0.118             0.172
kl_internal_filter_branch_length_v1      91     0.801     0.804            0.800               0.500                  0.667         0.681            0.088             0.231
                 kl_legacy_c2ef9a69     121     0.727     0.775            1.000               1.000                  0.333         0.645            0.207             0.149
```

## Section Summary

```text
     section  n_rows  mean_ari  mean_nmi  mean_outlier_f1  singleton_hit_rate  grouped_recovery_rate  exact_k_rate  over_split_rate  under_split_rate
    gaussian      71     0.604     0.659            0.933               0.833                  0.556         0.507            0.225             0.268
      binary      73     0.994     0.993              NaN                 NaN                    NaN         0.932            0.068             0.000
         sbm       6     0.379     0.356              NaN                 NaN                    NaN         0.500            0.000             0.500
 categorical      28     0.811     0.859              NaN                 NaN                    NaN         0.571            0.000             0.429
phylogenetic      31     0.894     0.924              NaN                 NaN                    NaN         0.452            0.516             0.032
 overlapping      67     0.886     0.878              NaN                 NaN                    NaN         0.910            0.045             0.045
method_proof      29     0.328     0.369              NaN                 NaN                    NaN         0.276            0.138             0.586
```

## Correlation Highlights

- Strongest monotonic `ari` relationship: `log_samples_per_cluster` (`rho=-0.203`, `negative`, `p=0.000371`, `n=305`).
- Strongest monotonic `exact_k` relationship: `log_samples_per_feature` (`rho=0.104`, `positive`, `p=0.071`, `n=305`).
- Strongest monotonic `over_split_flag` relationship: `log_samples` (`rho=-0.157`, `negative`, `p=0.00604`, `n=305`).
- Strongest monotonic `under_split_flag` relationship: `log_samples_per_cluster` (`rho=0.103`, `positive`, `p=0.0722`, `n=305`).

## Model Highlights

- `ari` model: `r_squared=0.377` over `305` rows.
- `ari`: more `true cluster count` is associated with `higher` values (`beta=0.650`, `p=0.00916`).
- `exact_k` model: `pseudo_r_squared=0.289` over `305` rows.
- `exact_k`: more `true cluster count` is associated with `higher` odds (`OR=198.425`, `p=0.0208`).
- `over_split_flag` model: `pseudo_r_squared=0.327` over `291` rows.
- `under_split_flag` model: `pseudo_r_squared=0.467` over `291` rows.

## Regression Coefficients

### `ari`

```text
                                            term   coef  std_err  pvalue  conf_low  conf_high
                                       Intercept  0.996    0.045   0.000     0.906      1.071
                                         noise_z  0.071    0.075   0.342    -0.034      0.248
                                  log_features_z -2.787    2.595   0.283    -7.969      1.016
                                   log_samples_z  0.822    1.455   0.572    -1.167      3.646
                             log_true_clusters_z  0.650    0.249   0.009     0.235      1.234
                       log_samples_per_cluster_z -0.867    1.384   0.531    -3.510      1.023
                      log_features_per_cluster_z  2.519    2.374   0.289    -0.964      7.204
                       log_samples_per_feature_z -0.070    0.094   0.453    -0.275      0.078
                     noise_cluster_interaction_z -0.027    0.072   0.706    -0.217      0.073
                       C(section)[T.categorical] -0.073    0.062   0.244    -0.200      0.054
                          C(section)[T.gaussian] -0.453    0.068   0.000    -0.588     -0.336
                      C(section)[T.method_proof] -0.634    0.101   0.000    -0.771     -0.399
                       C(section)[T.overlapping]  0.095    0.115   0.406    -0.146      0.356
                      C(section)[T.phylogenetic] -0.068    0.046   0.143    -0.144      0.028
                               C(section)[T.sbm] -0.588    0.163   0.000    -0.860     -0.225
C(method)[T.kl_internal_filter_branch_length_v1] -0.031    0.044   0.479    -0.113      0.060
                 C(method)[T.kl_legacy_c2ef9a69] -0.099    0.047   0.036    -0.191     -0.013
```

### `exact_k`

```text
                                            term   coef  std_err  pvalue  conf_low  conf_high  odds_ratio  or_conf_low               or_conf_high
                                       Intercept  2.486    1.120   0.026     1.538      4.448      12.011        4.654                     85.433
                                         noise_z  1.923    1.462   0.188     0.009      5.359       6.841        1.009                    212.487
                                  log_features_z -2.931   22.801   0.898   -56.136     33.684       0.053        0.000        425297801719891.688
                                   log_samples_z -5.357   13.311   0.687   -29.051     25.171       0.005        0.000            85413323729.693
                             log_true_clusters_z  5.290    2.289   0.021     2.835     11.623     198.425       17.029                 111685.630
                       log_samples_per_cluster_z  4.354   12.150   0.720   -24.501     24.942      77.817        0.000            67914280880.146
                      log_features_per_cluster_z  2.673   20.491   0.896   -28.670     50.683      14.478        0.000 5184705528587072045056.000
                       log_samples_per_feature_z -0.147    1.363   0.914    -2.084      2.906       0.863        0.124                     18.275
                     noise_cluster_interaction_z -1.846    1.354   0.173    -5.162      0.084       0.158        0.006                      1.088
                       C(section)[T.categorical] -1.641    1.133   0.148    -3.435     -0.537       0.194        0.032                      0.585
                          C(section)[T.gaussian] -2.638    1.145   0.021    -4.612     -1.708       0.071        0.010                      0.181
                      C(section)[T.method_proof] -3.501    1.278   0.006    -5.485     -2.203       0.030        0.004                      0.111
                       C(section)[T.overlapping]  1.656    1.195   0.166    -0.419      4.497       5.238        0.657                     89.766
                      C(section)[T.phylogenetic] -3.116    1.316   0.018    -5.065     -2.244       0.044        0.006                      0.106
                               C(section)[T.sbm] -2.216    7.743   0.775   -12.119     33.368       0.109        0.000        310170093751167.875
C(method)[T.kl_internal_filter_branch_length_v1] -0.196    0.444   0.659    -1.097      0.613       0.822        0.334                      1.847
                 C(method)[T.kl_legacy_c2ef9a69] -0.529    0.443   0.232    -1.492      0.164       0.589        0.225                      1.178
```

### `over_split_flag`

```text
                                            term    coef  std_err  pvalue  conf_low  conf_high  odds_ratio  or_conf_low               or_conf_high
                                       Intercept  -2.842    1.428   0.047    -6.024     -1.370       0.058        0.002                      0.254
                                         noise_z  -4.083    2.866   0.154   -12.791     -1.832       0.017        0.000                      0.160
                                  log_features_z  -1.576   34.417   0.963   -85.785     66.926       0.207        0.000 5184705528587072045056.000
                                   log_samples_z  -2.125   20.905   0.919   -41.505     46.169       0.119        0.000  112423478668542623744.000
                             log_true_clusters_z   0.885    5.954   0.882   -11.296     11.448       2.423        0.000                  93755.119
                       log_samples_per_cluster_z   1.422   19.626   0.942   -43.256     37.840       4.145        0.000      27150066699216972.000
                      log_features_per_cluster_z   2.337   30.910   0.940   -59.281     77.450      10.353        0.000 5184705528587072045056.000
                       log_samples_per_feature_z   1.597    1.987   0.422    -2.889      4.776       4.938        0.056                    118.607
                     noise_cluster_interaction_z   3.923    2.609   0.133     1.753     11.549      50.535        5.773                 103663.624
                       C(section)[T.categorical]  -7.927   34.712   0.819  -111.811     -4.453       0.000        0.000                      0.012
                          C(section)[T.gaussian]   0.860    1.547   0.578    -1.006      4.042       2.362        0.366                     56.942
                      C(section)[T.method_proof]   0.817    7.234   0.910   -14.653      3.922       2.263        0.000                     50.515
                       C(section)[T.overlapping]  -1.909   11.221   0.865   -13.665      2.133       0.148        0.000                      8.440
                      C(section)[T.phylogenetic]   3.887    1.552   0.012     2.742      8.292      48.766       15.522                   3993.442
                               C(section)[T.sbm] -10.785   50.140   0.830  -190.463     -4.357       0.000        0.000                      0.013
C(method)[T.kl_internal_filter_branch_length_v1]  -0.436    0.683   0.523    -1.985      0.616       0.647        0.137                      1.852
                 C(method)[T.kl_legacy_c2ef9a69]   0.823    0.556   0.139    -0.066      2.017       2.278        0.936                      7.514
```

### `under_split_flag`

```text
                                            term   coef  std_err  pvalue  conf_low  conf_high  odds_ratio  or_conf_low               or_conf_high
                                       Intercept -9.685   52.389   0.853  -175.259    -10.058       0.000        0.000                      0.000
                                         noise_z  2.276    3.293   0.489    -1.878     11.969       9.741        0.153                 157846.446
                                  log_features_z -2.371   78.939   0.976  -119.255    172.813       0.093        0.000 5184705528587072045056.000
                                   log_samples_z  0.643   49.687   0.990  -132.231     70.408       1.902        0.000 5184705528587072045056.000
                             log_true_clusters_z  2.382    9.847   0.809    -9.724     29.149      10.825        0.000          4560835467890.422
                       log_samples_per_cluster_z  4.193   46.239   0.928   -56.216    128.396      66.233        0.000 5184705528587072045056.000
                      log_features_per_cluster_z -0.983   70.920   0.989  -158.000    101.860       0.374        0.000 5184705528587072045056.000
                       log_samples_per_feature_z -3.697    4.181   0.377   -12.174      2.970       0.025        0.000                     19.498
                     noise_cluster_interaction_z -2.400    3.382   0.478   -13.469      1.244       0.091        0.000                      3.471
                       C(section)[T.categorical]  9.203   52.321   0.860     9.684    175.024    9925.989    16065.056 5184705528587072045056.000
                          C(section)[T.gaussian]  9.943   52.500   0.850    10.858    176.916   20813.891    51953.631 5184705528587072045056.000
                      C(section)[T.method_proof]  9.721   52.523   0.853    10.004    175.606   16665.253    22111.016 5184705528587072045056.000
                       C(section)[T.overlapping]  2.145   53.232   0.968     0.156    165.843       8.541        1.169 5184705528587072045056.000
                      C(section)[T.phylogenetic]  7.091   43.224   0.870   -34.317    143.554    1201.701        0.000 5184705528587072045056.000
                               C(section)[T.sbm] 10.635   53.648   0.843    -2.964    177.870   41567.826        0.052 5184705528587072045056.000
C(method)[T.kl_internal_filter_branch_length_v1]  0.672    0.583   0.249    -0.427      1.879       1.958        0.652                      6.546
                 C(method)[T.kl_legacy_c2ef9a69] -0.249    0.549   0.650    -1.364      0.804       0.780        0.256                      2.235
```
