---
title: Branch Length Candidate Full Big 2026-06-18
type: source
status: reviewed
updated: 2026-06-18
sources:
  - raw/assets/benchmark-results/branch_length_candidate_full_big_20260618/manifest.json
  - raw/assets/benchmark-results/branch_length_candidate_full_big_20260618/full_benchmark_comparison.csv
  - raw/assets/benchmark-results/branch_length_candidate_full_big_20260618/summary_by_method.csv
  - raw/assets/benchmark-results/branch_length_candidate_full_big_20260618/summary_by_category_method.csv
  - raw/assets/benchmark-results/branch_length_candidate_full_big_20260618/pairwise_branch_length_deltas.csv
  - raw/assets/benchmark-results/branch_length_candidate_full_big_20260618/skip_reason_summary.csv
  - raw/assets/benchmark-results/branch_length_candidate_full_big_20260618/benchmark_relationship_report.md
  - raw/assets/benchmark-results/branch_length_candidate_full_big_20260618/failure_report.md
tags:
  - source
  - benchmarks
  - guarded
  - branch-length
---

# Branch Length Candidate Full Big 2026-06-18

## Summary

This run compares `kl`, `kl_legacy_c2ef9a69`, and
`kl_internal_filter_branch_length_v1` across the full `121`-case benchmark
suite after the guarded runner-contract fix. Plots were disabled to avoid the
UMAP isolation subprocess crash observed when probing the default plotted full
runner; metrics, relationship analysis, and failure diagnosis completed.

The result keeps branch-length internal filtering as a candidate to test, but
does not promote it over current `kl`. Current KL has the best completed-row
mean ARI and exact-K count. Branch-length is close on mean ARI, improves a few
binary, categorical, outlier, and phylogenetic rows, and preserves fail-closed
behavior on unsupported severe overlap. Legacy completes every row but has the
lowest mean ARI and returns low-information outputs on unsupported overlap and
method-proof rows.

## Key Points

- The full run has `363` rows: `121` cases times three methods.
- `kl` records `93` OK rows, `28` skips, `66` exact-K rows, mean ARI
  `0.819354`, and median ARI `1.0`.
- `kl_internal_filter_branch_length_v1` records `91` OK rows, `30` skips,
  `62` exact-K rows, mean ARI `0.801364`, and median ARI `1.0`.
- `kl_legacy_c2ef9a69` records `121` OK rows, no skips, `78` exact-K rows,
  mean ARI `0.726685`, and median ARI `0.994656`.
- The relationship report identifies current `kl` as the best average-ARI
  method. The branch-length candidate has the strongest binary section cell
  with mean ARI `0.998` and exact-K rate `0.958`.
- Branch-length has notable gains over current on `phylo_dna_8taxa_low_mut`,
  `phylo_protein_8taxa`, `cat_mod_4cat_6c`, `binary_2clusters`, and
  `gauss_outlier_cluster_4c`.
- Branch-length has notable losses versus current on
  `gauss_clear_medium_continuous`, `dim_consolidated_4c_24f_continuous`,
  `phylo_dna_4taxa_low_mut`, `phylo_protein_4taxa`, and
  `phylo_dna_8taxa_med_mut`.
- On `overlap_extreme_4c`, current `kl` and branch-length skip for missing
  strict empirical-null support, while legacy completes with ARI `0.002729`.

## Evidence

- `summary_by_method.csv` records method-level OK, skip, exact-K, ARI, purity,
  and found-cluster aggregates.
- `pairwise_branch_length_deltas.csv` records case-level ARI deltas between
  branch-length and the current or legacy comparators.
- `skip_reason_summary.csv` separates strict empirical-null support failures,
  dense empirical-Gaussian covariance implementation limits, and branch-length
  internal sparse-context inadmissibility skips.
- `benchmark_relationship_report.md` records full-suite section summaries and
  model contrasts. It analyzes `305` completed rows and reports current `kl`
  as best average ARI.
- `failure_report.md` records low-ARI completed rows; no audit logs were
  generated because matrix audit exports were disabled for the full run.

## Links

- [[branch-length-candidate-run-gate-20260618]]
- [[benchmark-runner-guarded-contract-fix-20260618]]
- [[manual-guarded-benchmark-run-direct-20260617]]
