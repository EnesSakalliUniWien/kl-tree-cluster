---
title: Dimensional Gaussian Representation Diagnostic
type: analysis
status: reviewed
updated: 2026-05-26
sources:
  - benchmarks/shared/generators/generate_dimensional_gaussian.py
  - benchmarks/results/run_20260526_095857Z_full/full_benchmark_comparison.csv
  - benchmarks/results/dimensional_continuous_oracle_20260526/oracle_tree_recoverability.csv
tags:
  - benchmark
  - representation
  - recoverability
---

# Dimensional Gaussian Representation Diagnostic

## Summary

The selected continuous dimensional examples show that representation matters,
but only when the Euclidean hierarchy is recoverable. Consolidated continuous
cases improve because the raw coordinates preserve a large block-mean signal
that median binarization compresses. The diffuse continuous case still fails
because the signal is weakly spread across few informative coordinates and is
overwhelmed by many Gaussian noise coordinates; the resulting average-linkage
tree is not recoverable even under an oracle subtree cut.

This is not primarily a projected-Wald kernel failure. In the diffuse
continuous case, the hierarchy itself does not contain clean truth-aligned
subtrees.

## Details

The generator has two signal geometries. In consolidated mode, each cluster
owns a block of informative coordinates with mean `separation`, while the other
informative coordinates sit at a shared negative background. For the
four-cluster, twelve-informative-dimensional cases with separation \(2.8\), the
mean squared informative separation between cluster centers is about \(83.6\).

In diffuse mode, each cluster mean is a random vector normalized to length
`separation`. For the six-cluster, eighteen-informative-dimensional cases with
separation \(2.2\), the mean squared informative separation is only about
\(11.6\). The diffuse cases therefore have much less between-cluster signal
before adding the noise dimensions.

The 2026-05-26 full benchmark and local geometry check gave the following
diagnostic values:

| Case | Representation | \(p\) | Oracle any-\(K\) ARI | Oracle true-\(K\) ARI | KL ARI | Distance gap / within SD |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `dim_consolidated_4c_24f` | median binary | 24 | 0.490 | 0.447 | 0.447 | 1.00 |
| `dim_consolidated_4c_24f_continuous` | continuous | 24 | 1.000 | 1.000 | 1.000 | 4.03 |
| `dim_consolidated_4c_72f` | median binary | 72 | 0.116 | 0.064 | skipped | 0.53 |
| `dim_consolidated_4c_72f_continuous` | continuous | 72 | 0.992 | 0.708 | 0.708 | 2.35 |
| `dim_diffuse_6c_36f` | median binary | 36 | 0.073 | 0.045 | 0.012 | 0.44 |
| `dim_diffuse_6c_136f_continuous` | continuous | 136 | 0.055 | 0.002 | -0.002 | 0.29 |

The nearest-neighbor and root-tree diagnostics agree with this interpretation.
For `dim_consolidated_4c_24f_continuous`, 10-nearest-neighbor label purity was
1.0 and the oracle cut recovered four pure clusters. For
`dim_consolidated_4c_72f_continuous`, nearest-neighbor purity was still about
0.995, but the root split isolated one sample from a 159-sample mixed subtree;
the best unconstrained oracle cut recovered five pure groups, while the
exact-\(K=4\) oracle had to merge one true cluster pair and dropped to ARI
about 0.708. For `dim_diffuse_6c_136f_continuous`, nearest-neighbor purity was
only about 0.307, the root split was one singleton versus a mixed 179-sample
subtree, and the exact-\(K=6\) oracle cut was effectively random.

The mathematical reason is distance concentration. In the diffuse continuous
case, 18 informative coordinates carry a small random mean signal, while 118
noise coordinates add large within- and between-pair variation equally. The
observed Euclidean within-pair mean was about 16.60, the between-pair mean was
about 16.99, and the gap was only 0.29 within-standard-deviation units. Average
linkage therefore builds a tree dominated by noise fluctuations and singleton
joins instead of truth-aligned clades.

In contrast, the consolidated continuous cases have a block signal with much
larger center separation. Even with 60 noise dimensions, the observed
between-minus-within gap was about 2.35 within-standard-deviation units, enough
for Euclidean linkage to preserve useful subtrees. Median binarization removes
much of this metric magnitude and lowers the gap: the 72-feature consolidated
median-binary case had a gap of only 0.53 within-standard-deviation units and a
poor oracle tree.

## Evidence

- `generate_dimensional_gaussian.py` defines the consolidated and diffuse mean
  constructions and records informative/noise dimensions in metadata.
- `full_benchmark_comparison.csv` records that consolidated continuous cases
  achieved KL ARI 1.0 and 0.708, while the diffuse continuous case achieved
  about -0.002.
- `oracle_tree_recoverability.csv` records that the two consolidated continuous
  trees were recoverable or near-recoverable, while the diffuse continuous tree
  had oracle true-\(K\) ARI about 0.002.

## Links

- [[open-mathematical-questions]]
- [[local-marchenko-pastur-rule]]
- [[oracle-gate-path-diagnostic]]

## Open Questions

1. Should diffuse Gaussian be treated as a hierarchy/metric benchmark rather
   than a gate benchmark?
2. Should continuous Gaussian benchmarks include a signal-to-noise design
   target such as distance-gap/within-spread, not only \(p,n,K\), and nominal
   separation?
3. Should the continuous production path use covariance whitening or subspace
   screening before tree construction when many coordinates are pure noise?
