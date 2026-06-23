---
title: Path-Conditioned Barycentric Action Diagnostics 2026-06-06
type: source
status: reviewed
updated: 2026-06-06
sources:
  - benchmarks/diagnostics/math_trace/barycentric_action.py
  - benchmarks/diagnostics/math_trace/path_conditioned_barycentric_action.py
  - tests/validation/78_test_barycentric_action_diagnostics.py
  - benchmarks/results/diagnostics/path_conditioned_barycentric_action_20260606/path_conditioned_barycentric_action_summary.json
  - benchmarks/results/diagnostics/path_conditioned_barycentric_action_20260606/missing_equation_candidate_panel.csv
  - benchmarks/results/diagnostics/path_conditioned_barycentric_action_20260606_fresh_full/path_conditioned_barycentric_action_summary.json
  - benchmarks/results/diagnostics/path_conditioned_barycentric_action_20260606_fresh_full/missing_equation_candidate_panel.csv
  - benchmarks/results/diagnostics/path_conditioned_barycentric_action_20260606_guard_panel/path_conditioned_barycentric_action_summary.json
  - benchmarks/results/diagnostics/path_conditioned_barycentric_action_20260606_guard_panel/kak_radius_angle_action_annotations.csv
  - benchmarks/results/diagnostics/path_conditioned_barycentric_action_20260606_guard_panel/action_budget_guard_panel.csv
  - benchmarks/results/diagnostics/path_conditioned_barycentric_action_20260606_guard_panel/path_conditioned_barycentric_action_report.md
  - benchmarks/results/diagnostics/path_conditioned_barycentric_action_20260606_guard_utility/path_conditioned_barycentric_action_summary.json
  - benchmarks/results/diagnostics/path_conditioned_barycentric_action_20260606_guard_utility/action_budget_guard_utility_curve.csv
  - benchmarks/results/diagnostics/path_conditioned_barycentric_action_20260606_guard_utility/path_conditioned_barycentric_action_report.md
  - benchmarks/results/run_20260606_path_action_full/full_benchmark_comparison.csv
  - benchmarks/results/run_20260606_path_action_full/benchmark_relationship_report.md
  - benchmarks/results/run_20260606_path_action_full/failure_report.md
tags:
  - source
  - diagnostics
  - barycentric
  - traversal
  - benchmarks
---

# Path-Conditioned Barycentric Action Diagnostics 2026-06-06

## Summary

This diagnostic implements the path-conditioned barycentric action trace as a
descriptive math diagnostic. It adds pure barycentric residual and
parallel-axis action helpers, then reads cached KAK internal geometry,
fragmentation validation, and full-benchmark outputs to rank probable missing
equation paths. The diagnostic does not change production edge, sibling,
traversal, projection, or calibration behavior.

## Key Points

- The cached KAK validation result is reproduced exactly:
  `radius_angle_action` has median leave-one-block-out AUC `0.892810`.
- The fresh full benchmark rerun in
  `benchmarks/results/run_20260606_path_action_full/` completed `120` cases
  and `1080` method rows with plots disabled and relationship analysis
  enabled.
- In the fresh benchmark, `tbs` has `92` ok rows and `28` skips. The ok-row
  mean ARI is `0.824578`.
- Of the `28` `tbs` skips, `27` are calibration-support skips caused by
  selected non-null positive-weight records without strict-null or stopped-edge
  empirical-null calibration support.
- The diagnostic ranks the exact barycentric edge/sibling identity as the base
  equation, not as a missing production rule.
- The paths promoted to validation panels are the KAK angular-shell
  radius/angle/action diagnostic, the parallel-axis action-budget equation,
  and traversal survival as a reached-and-split path event.
- The follow-up guard-panel extension writes row-level radius, angle,
  independent-radius, common-axis gap, action-budget proxy, and
  angular-shell-risk annotations for `7722` KAK internal-merge rows.
- The guard panel uses post-hoc final-assignment labels only. It treats
  one-final-cluster parents as pure-fragment contexts and multi-final-cluster
  parents as mixed contexts, with label provenance
  `posthoc_final_assignment_parent_cluster_count`.
- In the guard panel, pure-fragment contexts have higher median capped
  action-budget proxy (`1.0`) than mixed contexts (`0.598794`), so the
  diagnostic guard direction is high-action angular-shell risk, not a minimum
  action threshold.
- The best candidate guard in the tested grid is
  `action_ge_0.9__angle_ge_75__ind_ge_0.85`, with pure-fragment precision
  `0.713537`, pure-fragment flag rate `0.208578`, and mixed-context flag rate
  `0.086202`.
- A cost-sensitive guard utility curve was added. With one prevented pure
  fragment valued the same as one delayed mixed context, the best guard is
  `action_ge_0.75__angle_ge_60__ind_ge_0.85`, with `1385` pure-fragment flags,
  `641` mixed-context flags, and net utility `744`.
- When a delayed mixed context costs twice a prevented pure fragment, the best
  guard tightens to `action_ge_0.9__angle_ge_60__ind_ge_0.85`, with `1259`
  pure-fragment flags, `531` mixed-context flags, and net utility `197`.
- The best tested break-even mixed-context cost ratio is `2.490854`; no tested
  guard is net-positive once a delayed mixed context costs `3x` or `5x` a
  prevented pure fragment.
- External selected-tail support remains a support gap requiring new data or a
  validated external selected-tree calibration object. It is not an admissible
  fallback.
- Selected-region tangent cones and projection-selection basis laws remain
  diagnostic objects that need separate validation.

## Evidence

- `benchmarks/diagnostics/math_trace/barycentric_action.py` contains the pure
  barycentric identity, signed z-identity, and action-budget helpers.
- `benchmarks/diagnostics/math_trace/path_conditioned_barycentric_action.py`
  contains the cached-output diagnostic runner.
- `tests/validation/78_test_barycentric_action_diagnostics.py` verifies the
  exact residual helpers, action-budget identity, and cached diagnostic writer.
- `benchmarks/results/diagnostics/path_conditioned_barycentric_action_20260606_fresh_full/path_conditioned_barycentric_action_report.md`
  records the fresh benchmark join and candidate ranking.
- `benchmarks/results/diagnostics/path_conditioned_barycentric_action_20260606_guard_panel/action_budget_guard_panel.csv`
  records the high-action angular-shell guard threshold grid.
- `benchmarks/results/diagnostics/path_conditioned_barycentric_action_20260606_guard_panel/kak_radius_angle_action_annotations.csv`
  records the row-level radius, angle, action-budget proxy, and post-hoc
  pure/mixed context annotations.
- `benchmarks/results/diagnostics/path_conditioned_barycentric_action_20260606_guard_utility/action_budget_guard_utility_curve.csv`
  records cost-sensitive utility and break-even mixed-context cost ratios for
  the guard grid.
- `benchmarks/results/run_20260606_path_action_full/full_benchmark_comparison.csv`
  contains the fresh full benchmark table used by the recursive join.

## Links

- [[barycentric-action-equation-diagnostic-20260606]]
- [[full-benchmark-run-20260606]]
- [[adaptive-cosine-kak-benchmark-probe-20260605]]
- [[open-mathematical-questions]]
