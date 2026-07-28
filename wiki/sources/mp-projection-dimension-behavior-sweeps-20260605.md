---
title: MP Projection Dimension Behavior Sweeps 2026-06-05
type: source
status: reviewed
updated: 2026-06-05
sources:
  - raw/inbox/gavish-donoho-optimal-hard-threshold-2014.txt
  - benchmarks/diagnostics/spectral/mp/mp_projection_dimension_behavior_sweep.py
  - tests/validation/spectral/mp/77_test_mp_projection_dimension_behavior_sweep.py
  - benchmarks/results/diagnostics/mp_projection_dimension_behavior_method_proof_20260605/mp_projection_dimension_behavior_summary.csv
  - benchmarks/results/diagnostics/mp_projection_dimension_behavior_method_proof_20260605/mp_projection_dimension_behavior_case_summary.csv
  - benchmarks/results/diagnostics/mp_projection_dimension_behavior_binary_20260605/mp_projection_dimension_behavior_summary.csv
  - benchmarks/results/diagnostics/mp_projection_dimension_behavior_categorical_20260605/mp_projection_dimension_behavior_summary.csv
  - benchmarks/results/diagnostics/mp_projection_dimension_behavior_debug_20260605/null_pvalue_ratio_summary.csv
  - benchmarks/results/diagnostics/mp_projection_dimension_behavior_debug_20260605/dimension_strata_null_false_split.csv
  - benchmarks/results/diagnostics/mp_projection_dimension_behavior_debug_20260605/paired_null_decision_summary.csv
  - benchmarks/results/diagnostics/mp_projection_dimension_behavior_debug_20260605/raw_mp_zero_positive_contrast.csv
  - benchmarks/results/diagnostics/mp_projection_dimension_behavior_debug_20260605/required_alpha_summary.csv
tags:
  - source
  - diagnostics
  - spectral
  - validation
---

# MP Projection Dimension Behavior Sweeps 2026-06-05

## Summary

A diagnostic runner now reruns raw sibling projected-Wald tests on the same TBS
tree under five candidate sibling projection-dimension rules: current
edge-derived dimension, parent test dimension, raw parent MP count, raw MP
floor-1, and raw MP floor-2. It adds null false-split and signal-retention
endpoints to the earlier Q17 distribution-only grid, but remains diagnostic
because the tested p-values are raw selected-tree sibling p-values, not
production-calibrated decisions.

## Key Points

- The method-proof, binary, and categorical sweeps produced `163100`, `235880`,
  and `59230` records per rule. Binary and categorical had no skipped
  case-replicates; method-proof skipped `100` replicates of
  `cont_lowrank_pggn_shrinkage` because dense empirical-Gaussian covariance
  state would require about `4852.3 MiB`, above the `512.0 MiB` memory contract.
- The dimension rules differ materially. In the method-proof panel, raw parent
  MP count differs from current on `73.7%` of rows and uses `k=0` on `55.4%`;
  binary differs on `48.3%` and uses `k=0` on `28.5%`; categorical differs on
  `88.9%` and uses `k=0` on `84.6%`.
- Suite-wide raw selected-tree false splits are high for every nonzero-floor
  rule, showing that dimension choice alone cannot make the raw sibling
  projected-Wald p-values production-admissible after tree selection. Current
  null false-split rates were `0.731`, `0.925`, and `0.966` in method-proof,
  binary, and categorical. Raw MP count reduced those to `0.496`, `0.672`, and
  `0.119`, but at substantial signal-retention cost.
- The targeted MP spike cases expose the core Q14/Q15 trade-off. In
  `mp_spike_below_bbp_continuous`, raw MP count chose `k=0` on all rows and
  reduced false splits from `0.07597` under the current rule to `0.0`. In
  `mp_spike_above_bbp_continuous`, raw MP count again chose `k=0` on all rows
  and reduced signal retention from `0.09152` to `0.0`. Floor-1 restored some
  power (`0.08258`) but kept below-BBP false splits at `0.06277`.
- The Gavish-Donoho optimal hard singular-value threshold is relevant as a
  more conservative rank-selection candidate than the ordinary MP edge. It is
  not a production justification here because its objective is low-rank
  denoising risk, not selected-tree null false-split control. A follow-up
  diagnostic can add its known-noise and median-noise thresholds as candidate
  rank rules.
- The post-hoc debug tables show the raw-MP improvement is mostly the
  zero-dimensional no-test branch. For raw parent MP count on null contexts,
  the `k=0` stratum has rejection rate `0.0`, while the `k>0` stratum rejects
  about `0.996`, `0.992`, and `0.993` in method-proof, binary, and
  categorical.
- Nonzero selected-tree tests are far from a fixed-subspace chi-square null.
  The median selected statistic ratio on null contexts is about `123`, `134`,
  and `72` for the current rule in method-proof, binary, and categorical.
  A nominal `1%` null false-split cutoff would need raw p-values around
  `1e-113` in method-proof and `1e-225` in binary; categorical has literal
  p-value underflow in the lower tail. This rules out a simple alpha tweak as
  a production fix.

## Evidence

- `mp_projection_dimension_behavior_records.csv` files contain row-level rule,
  dimension, raw p-value, true null/signal context, and selected-ratio data.
- `mp_projection_dimension_behavior_summary.csv` files report rule-level
  dimension frequencies, rejection rates, null false-split rates, and signal
  retention.
- `mp_projection_dimension_behavior_case_summary.csv` separates behavior by
  benchmark case, including the below-BBP and above-BBP MP spike cases.
- `mp_projection_dimension_behavior_debug_20260605/` contains post-hoc
  attribution tables for null p-value ratios, dimension strata, paired rule
  decisions, raw-MP zero/nonzero contrasts, and required alpha cutoffs.
- `raw/inbox/gavish-donoho-optimal-hard-threshold-2014.txt` records the
  external singular-value threshold paper and its diagnostic-only relevance.

## Links

- [[recursive-method-followups-20260604]]
- [[open-mathematical-questions]]
