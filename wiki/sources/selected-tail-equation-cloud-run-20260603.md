---
title: Selected Tail Equation Cloud Run 2026-06-03
type: source
status: reviewed
updated: 2026-06-03
sources:
  - benchmarks/cloud/aws_selected_tail_equation_study.py
  - benchmarks/cloud/aws/batch-stack.yml
  - raw/assets/benchmark-results/selected_tail_equation_cloud_run_20260603_1000/aws_selected_tail_equation_study_manifest.json
  - raw/assets/benchmark-results/selected_tail_equation_cloud_run_20260603_1000/local_deployment_context.json
  - raw/assets/benchmark-results/selected_tail_equation_cloud_run_20260603_1000/selected_ratio_tail_law.csv
  - raw/assets/benchmark-results/selected_tail_equation_cloud_run_20260603_1000/candidate_equations.csv
  - raw/assets/benchmark-results/selected_tail_equation_cloud_run_20260603_1000/candidate_equation_holdout.csv
  - raw/assets/benchmark-results/selected_tail_equation_cloud_run_20260603_1000/geometry_summary_by_case.csv
tags:
  - source
  - aws
  - calibration
  - selection
---

# Selected Tail Equation Cloud Run 2026-06-03

## Summary

This AWS Batch diagnostic ran the selected-hierarchy geometry study as `20`
shards with `50` regenerations per shard, giving `1000` requested regenerated
hierarchies per case for `gauss_null_large`, `gauss_clear_medium`,
`binary_low_noise_4c`, and `cat_clear_3cat_4c`. It produced `269,318`
selected records and `2,967` independent selected-hierarchy simulation ids.

The row-level combined table is stored in S3 because it is `261.5 MiB`. The
repository keeps the compact merged summaries under
`raw/assets/benchmark-results/selected_tail_equation_cloud_run_20260603_1000/`.

## Key Points

- Seven base contexts satisfy the current production selected-tail support
  contract in this cloud run.
- The admissible contexts are small or medium high-edge selected-tail contexts:
  categorical small-parent projection `1` and `2`, Gaussian small-parent
  projection `1` and `2`, and Gaussian medium-parent projection `2`.
- Binary small-parent high-edge projection `1` is close but remains
  support-limited at `413/499` independent matching simulations in this
  four-case cloud panel.
- Gaussian root/high-edge/projection-2 and large-parent/high-edge/projection-2
  have abundant support (`1731` and `1454` independent matching simulations)
  but fail the strict held-out exceedance standard-error contract. This means
  exact support is not the only issue; tail homogeneity is still context
  dependent.
- The edge-plus-spectral candidate equation remains a strong descriptive tail
  score. In leave-one-source-family holdout its top-tail AUC is `0.965890`,
  but the fitted equations remain descriptive and do not define calibrated
  production probabilities.
- The full descriptive equation has better in-sample mean-log-ratio fit, but
  its leave-one-source-family mean-log-ratio \(R^2\) is negative. This is
  evidence against treating the broad fitted equation as a portable production
  law.

## Evidence

- `raw/assets/benchmark-results/selected_tail_equation_cloud_run_20260603_1000/aws_selected_tail_equation_study_manifest.json`
  records `20` shards, `50` replicates per shard, `1000` requested
  regenerations per case, `269,318` selected records, and `2,967` independent
  simulation ids.
- `raw/assets/benchmark-results/selected_tail_equation_cloud_run_20260603_1000/local_deployment_context.json`
  records the AWS account, image digest, Batch job ids, S3 output location, and
  local dirty-worktree state used to launch the run.
- `raw/assets/benchmark-results/selected_tail_equation_cloud_run_20260603_1000/selected_ratio_tail_law.csv`
  records `80` selected-tail contexts, including `7` production-admissible
  rows under the current support and precision contract.
- `raw/assets/benchmark-results/selected_tail_equation_cloud_run_20260603_1000/candidate_equation_holdout.csv`
  records replicate, case, and source-family holdout behavior for the
  candidate descriptive equations.
- `benchmarks/cloud/aws_selected_tail_equation_study.py` prefixes shard-local
  independent simulation ids before recomputing the merged summaries.

## Links

- [[aws-selected-tail-equation-study]]
- [[selected-hierarchy-null-support-contract]]
- [[selected-hierarchy-geometric-law-map]]
- [[open-mathematical-questions]]
