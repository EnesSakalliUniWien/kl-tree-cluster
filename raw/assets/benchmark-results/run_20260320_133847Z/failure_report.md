# Benchmark Failure Diagnosis

**Source**: `/Users/berksakalli/Projects/kl-te-cluster/benchmarks/results/run_20260320_133847Z/full_benchmark_comparison.csv`
**Audit Dir**: `/Users/berksakalli/Projects/kl-te-cluster/benchmarks/results/run_20260320_133847Z/audit`

| Case ID | ARI | Found / True | Mode | Diagnosis |
| :--- | :--- | :--- | :--- | :--- |
| gauss_extreme_noise_highd | 0.115 | 26 / 4 | MIXED | Root split OK, moderate complexity. |
| dim_consolidated_4c_24f | 0.000 | 1 / 4 | **UNDER-SPLIT** | Root split rejected (P=3.64e-03) |
| dim_consolidated_4c_72f | 0.000 | 1 / 4 | **UNDER-SPLIT** | Root split rejected (P=1.27e-01) |
| dim_consolidated_4c_272f | 0.000 | 1 / 4 | **UNDER-SPLIT** | Root split rejected (P=1.74e-01) |
| dim_diffuse_6c_36f | 0.000 | 1 / 6 | **UNDER-SPLIT** | Root split rejected (P=2.55e-01) |
| dim_diffuse_6c_136f | 0.000 | 1 / 6 | **UNDER-SPLIT** | Root split rejected (P=1.25e-01) |
| dim_diffuse_6c_536f | 0.001 | 2 / 6 | **UNDER-SPLIT** | Root split rejected (P=1.82e-01) |
| sbm_moderate | 0.000 | 1 / 3 | **UNDER-SPLIT** | Root split rejected (P=3.18e-02) |
| sbm_hard | 0.000 | 1 / 3 | **UNDER-SPLIT** | Root split rejected (P=5.33e-03) |
| cat_highcard_20cat_4c | 0.000 | 1 / 4 | **UNDER-SPLIT** | Root split rejected (P=1.13e-03) |
| cat_overlap_3cat_4c | 0.000 | 1 / 4 | **UNDER-SPLIT** | Root split rejected (P=1.51e-02) |
| overlap_heavy_4c_small_feat | 0.000 | 1 / 4 | **UNDER-SPLIT** | Root split rejected (P=1.26e-01) |
| overlap_heavy_4c_med_feat | 0.000 | 1 / 4 | **UNDER-SPLIT** | Root split rejected (P=9.03e-02) |
| overlap_heavy_8c_large_feat | 0.000 | 1 / 8 | **UNDER-SPLIT** | Root split rejected (P=1.19e-01) |
| overlap_extreme_4c | 0.000 | 1 / 4 | **UNDER-SPLIT** | Root split rejected (P=1.00e+00) |
| overlap_mod_4c_small | 0.000 | 1 / 4 | **UNDER-SPLIT** | Root split rejected (P=2.75e-03) |
| overlap_hd_4c_1k | 0.000 | 1 / 4 | **UNDER-SPLIT** | Root split rejected (P=nan) |
| overlap_unbal_4c_small | 0.000 | 1 / 4 | **UNDER-SPLIT** | Root split rejected (P=5.80e-01) |