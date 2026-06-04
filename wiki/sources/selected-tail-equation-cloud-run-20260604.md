---
title: Selected Tail Equation Cloud Run 2026-06-04
type: source
status: reviewed
updated: 2026-06-04
sources:
  - benchmarks/cloud/aws_selected_tail_equation_study.py
  - raw/assets/benchmark-results/selected_tail_equation_cloud_run_20260604_1000/aws_selected_tail_equation_study_manifest.json
  - raw/assets/benchmark-results/selected_tail_equation_cloud_run_20260604_1000/local_deployment_context.json
  - raw/assets/benchmark-results/selected_tail_equation_cloud_run_20260604_1000/selected_ratio_tail_law.csv
  - raw/assets/benchmark-results/selected_tail_equation_cloud_run_20260604_1000/candidate_equations.csv
  - raw/assets/benchmark-results/selected_tail_equation_cloud_run_20260604_1000/candidate_equation_holdout.csv
  - raw/assets/benchmark-results/selected_tail_equation_cloud_run_20260604_1000/geometry_summary_by_case.csv
tags:
  - source
  - aws
  - calibration
  - selection
---

# Selected Tail Equation Cloud Run 2026-06-04

## Summary

This is a replication of the AWS selected-tail equation study using the rebuilt
manifest-aware Batch image. It used `20` shards, `50` regenerations per shard,
base seed `20260604`, and the same four cases as the 2026-06-03 run:
`gauss_null_large`, `gauss_clear_medium`, `binary_low_noise_4c`, and
`cat_clear_3cat_4c`.

The merged run produced `271,801` selected records and `2,997` independent
selected-hierarchy simulation ids. The row-level merged table is stored in S3
because it is `263.9 MiB`; compact merged summaries are stored locally under
`raw/assets/benchmark-results/selected_tail_equation_cloud_run_20260604_1000/`.

## Key Points

- The rerun again finds exactly seven production-admissible base contexts under
  the current support and held-out precision contract.
- No context changed admissibility status relative to the 2026-06-03 AWS run.
- The seven admissible contexts are the same families of contexts:
  categorical small-parent high-edge projection `1` and `2`, Gaussian
  medium-parent high-edge projection `2`, and Gaussian small-parent
  edge-action/projection contexts.
- Root and large Gaussian high-edge projection-2 contexts again have abundant
  support but fail held-out precision. The root context has `1734` independent
  simulations with standard error `0.002280`; the large-parent context has
  `1465` simulations with standard error `0.002668`.
- Binary small-parent high-edge projection-1 remains support-limited in this
  four-case cloud panel, with `423/499` independent simulations, although its
  held-out standard error is below the precision threshold.
- The edge-plus-spectral candidate equation remains the strongest portable
  descriptive tail score in leave-one-source-family holdout, with top-tail AUC
  `0.966050`. It is still descriptive and does not define calibrated
  production probabilities.
- The run evaluated exactly six descriptive candidate functions:
  `edge_action`, `edge_sampling_geometry`, `edge_spectral_modes`,
  `edge_angular_capture`, `selected_energy_candidate`, and
  `full_descriptive_candidate`. The AWS runner did not add any additional
  model family; it only sharded and merged the geometry-covariate diagnostic.
- Candidate-equation tables use `271,798` finite log-ratio rows rather than
  all `271,801` selected records. A row-level debug pass found three selected
  Gaussian rows with `selected_hierarchy_ratio = 0`, giving
  `log_selected_hierarchy_ratio_status =
  undefined_zero_selected_hierarchy_ratio`. Those three rows are excluded from
  every candidate-equation fit by the finite-response model-table contract.
- The test statistic path is:
  `selected_hierarchy_ratio = statistic / (reference_scale *
  degrees_of_freedom)`, where the projected-Wald kernel uses an orthonormal
  selected PCA subspace, `reference_scale = 1`, and
  `degrees_of_freedom = sibling_projection_dimension`. The edge-action
  coordinate is derived as
  `-log10(min_child_edge_bh_p_value)`.
- A row-weighting audit found a real interpretation skew. `gauss_null_large`
  contributes `30.8%` of independent selected simulations but `63.6%` of
  selected rows, and it contributes `85.2%` of the row-level top-decile
  candidate-equation tail. Therefore candidate-equation AUCs are selected-node
  weighted, not equal-hierarchy weighted.
- The skew does not reverse the main ranking. When each independent hierarchy
  simulation is given equal total weight, `edge_spectral_modes` remains the
  strongest candidate among the tested equations, but the weighted-fit
  top-tail AUC drops to about `0.927`. This supports treating it as the best
  search coordinate, not as a calibrated production law.
- The selected-tail support table checks independent simulation counts, but
  its held-out exceedance rate and standard error are row-level within each
  selected context. This is appropriate for a selected-node descriptive target,
  but it should not be read as an independent-hierarchy standard error without
  also reporting a simulation-weighted version.

## Evidence

- `raw/assets/benchmark-results/selected_tail_equation_cloud_run_20260604_1000/aws_selected_tail_equation_study_manifest.json`
  records the merged run, including `build_commit`, `build_branch`, `20`
  shards, `50` regenerations per shard, `271,801` selected records, and
  `2,997` independent simulation ids.
- `raw/assets/benchmark-results/selected_tail_equation_cloud_run_20260604_1000/selected_ratio_tail_law.csv`
  records `79` selected-tail contexts and `7` production-admissible rows.
- `raw/assets/benchmark-results/selected_tail_equation_cloud_run_20260604_1000/candidate_equation_holdout.csv`
  records replicate, case, and source-family holdout behavior for the
  descriptive candidate equations.
- `raw/assets/benchmark-results/selected_tail_equation_cloud_run_20260604_1000/candidate_equations.csv`
  records one row for each of the six candidate functions and confirms that
  each function used the same `271,798` finite response rows.
- The row-level skew audit used the S3-only
  `selected_geometry_records.csv` merged table for this run. The temporary
  local copy was removed after inspection.
- `raw/assets/benchmark-results/selected_tail_equation_cloud_run_20260604_1000/local_deployment_context.json`
  records the AWS job ids, image digest, S3 location, and local dirty-worktree
  state used to launch the rerun.

## Links

- [[selected-tail-equation-cloud-run-20260603]]
- [[aws-selected-tail-equation-study]]
- [[selected-hierarchy-null-support-contract]]
- [[open-mathematical-questions]]
