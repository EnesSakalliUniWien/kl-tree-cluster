"""Compare benchmark results across edge calibration approaches."""

import pandas as pd

# Latest run (sibling-neighborhood calibration)
df_new = pd.read_csv('benchmarks/results/run_20260218_155342Z/full_benchmark_comparison.csv')
# Baseline (no calibration)
df_base = pd.read_csv('benchmarks/results/run_20260218_114417Z/full_benchmark_comparison.csv')
# Previous soft calibration
df_soft = pd.read_csv('benchmarks/results/run_20260218_133637Z/full_benchmark_comparison.csv')

print('=' * 100)
print(f'{"Run":30s} | {"Method":20s} | {"N":>4s} | {"Mean ARI":>9s} | {"Med ARI":>8s} | {"Exact K":>8s} | {"K=1":>4s}')
print('=' * 100)

for label, df in [('Baseline (no cal)', df_base), ('Soft local cal', df_soft), ('Sibling-nbr cal', df_new)]:
    for method in ['kl', 'kl_rogerstanimoto']:
        sub = df[df['method'] == method]
        n = len(sub)
        mean_ari = sub['ari'].mean()
        median_ari = sub['ari'].median()
        exact_k = (sub['found_clusters'] == sub['true_clusters']).sum()
        k1 = (sub['found_clusters'] == 1).sum()
        print(f'{label:30s} | {method:20s} | {n:4d} | {mean_ari:9.3f} | {median_ari:8.3f} | {exact_k:>3d}/{n:<3d} | {k1:4d}')
    print('-' * 100)
