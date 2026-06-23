---
title: Refined Profile All-Benchmark Tests 2026-06-14
type: source
status: reviewed
updated: 2026-06-14
sources:
  - raw/inbox/refined-profile-all-benchmark-tests-20260614.md
  - benchmarks/diagnostics/calibration/fixed_sibling_gate_profile_validation.py
  - benchmarks/shared/runners/tbs_runner.py
  - tree_break_selection/hierarchy_analysis/decomposition/gates/orchestrator.py
tags:
  - source
  - diagnostics
  - benchmark
  - validation
---

# Refined Profile All-Benchmark Tests 2026-06-14

## Summary

The refined global selected-family pass-through profile was tested on every
currently supported selected-null case and on the full 120-case benchmark as a
single-seed performance pass. The result is diagnostic but not production
admissible: ordinary binary/categorical performance is often strong, but
overlap-template nulls still inflate, several signal families remain weak, and
continuous feature families fail the current covariance contract.

## Key Points

- The selected-null all-supported smoke covered 53 full-suite cases that the
  selected-null runner can regenerate: 42 binary-template and 11 direct
  categorical multinomial cases.
- That smoke wrote 318 rows. Binary-template nulls had 11 false splits across
  126 null rows; direct categorical nulls had zero false splits across 33 null
  rows.
- Six null-inflated cases were overlap-template binary cases:
  `overlap_unbal_4c_small`, `overlap_heavy_4c_small_feat`,
  `overlap_mod_4c_small`, `overlap_part_4c_small`, `overlap_extreme_4c`, and
  `overlap_part_8c_large`.
- The selected-null smoke retained many signals, but seven signal summaries
  were weak, including high-cardinality/moderate categorical cases and
  heavy-overlap binary cases.
- Production admissibility remains fail-closed in the all-supported smoke for
  both binary-template and categorical-multinomial transfer summaries.
- The full 120-case performance pass returned 107 ok rows and 13 errors. The
  errors are continuous feature families blocked by the current continuous
  covariance contract or dense covariance work-state limits.
- Across the 107 ok full-suite rows, mean ARI was `0.782567`, median ARI was
  `0.955719`, and exact-K rate was `0.439252`.
- The ok rows confirm the profile routed as intended:
  `fixed_coordinate_global_passthrough_refined_v1`,
  `fixed_coordinate_bh`, `global_sibling_min_passthrough_descendant_refined`,
  with 99 base selected-family permutations.
- High-dimensional direct categorical performance was strong but expensive:
  `cat_highd_3cat_500feat` returned ARI `0.888278`, K=`14/4`, in about `362`
  seconds; `cat_highd_4cat_1000feat` returned ARI `0.944379`, K=`15/6`, in
  about `1263` seconds.
- Weak full-suite families include SBM, dimensional Gaussian median binaries,
  categorical Dirichlet-multinomial, heavy-overlap binary, and the planted
  deep-signal traversal case.

## Evidence

- `raw/inbox/refined-profile-all-benchmark-tests-20260614.md` records the
  commands, artifact paths, summaries, and interpretation.
- `/tmp/klte_global_passthrough_refined_all_supported_3rep_20260614` contains
  the selected-null all-supported run outputs.
- `/tmp/klte_refined_full120_performance_20260614` contains the full-suite
  single-seed performance outputs.
- `benchmarks/diagnostics/calibration/fixed_sibling_gate_profile_validation.py`
  defines the selected-null profile validation runner.
- `benchmarks/shared/runners/tbs_runner.py` exposes the profile-capable TBS
  runner used by the full-suite performance pass.
- `tree_break_selection/hierarchy_analysis/decomposition/gates/orchestrator.py`
  defines the refined profile and selected-family guard behavior.

## Links

- [[fixed-sibling-gate-profile-validation-20260613]]
- [[open-mathematical-questions]]
- [[projected-wald-statistic]]
