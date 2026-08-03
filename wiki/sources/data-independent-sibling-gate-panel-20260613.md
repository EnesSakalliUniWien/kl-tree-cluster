---
title: Data-Independent Sibling Gate Panel 2026-06-13
type: source
status: reviewed
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/sibling/gates/data_independent_sibling_gate_panel.py
  - tests/validation/calibration/sibling/gates/99_test_data_independent_sibling_gate_panel.py
  - raw/inbox/data-independent-sibling-gate-smoke-20260613.md
  - raw/inbox/data-independent-sibling-gate-transfer-20260613.md
tags:
  - source
  - diagnostics
  - calibration
  - statistics
---

# Data-Independent Sibling Gate Panel 2026-06-13

## Summary

`data_independent_sibling_gate_panel.py` tests a same-data repair
candidates for the selected sibling null-law failure. It removes the adaptive
parent PCA projection and adaptive projection dimension from the sibling gate,
then evaluates predeclared coordinate-wise Wald p-values over a
selected-topology penalty grid.

## Key Points

- The panel is diagnostic-only. Its statuses are wired into the conservative
  production-admissibility contract, where candidate/null-retained outcomes
  remain `diagnostic_only` and null inflation, weak signal, or insufficient
  rows fail closed.
- The implemented candidates are `coordinate_bonferroni` and `coordinate_bh`
  over fixed whitened contrast coordinates, plus `block_bonferroni` and
  `block_bh` over fixed `FeatureSpace` blocks. Categorical block gates use one
  chi-square p-value per original categorical feature with `df = K - 1`.
  All candidates avoid learning a tested subspace from the same sibling
  contrast.
- Candidate production contracts are scoped by method, penalty, topology mode,
  and feature family, so failed grid alternatives do not invalidate a separate
  candidate.
- In the `binary_2clusters` selected-topology smoke with `50` replicates and
  selected-topology penalty `10`, both null rows have effective-alpha rejection
  rate `0.007755` against sibling alpha `0.01`.
- In the same smoke, signal effective-alpha rejection is `0.175510` for
  Bonferroni and `0.194694` for BH; large-parent signal rejection is
  `0.738019` for both candidates.
- In the three-case binary transfer smoke, `coordinate_bh` with penalty `10`
  is the only grid candidate that controls all selected-null cases and retains
  all signal cases under the predeclared all-parent signal threshold. Its
  scoped production decision is still `diagnostic_only`.
- The direct categorical smoke is null-conservative but signal-weak under the
  same all-parent threshold. For `cat_clear_3cat_4c`, `coordinate_bh` with
  penalty `10` has null rejection `0.003535` and signal rejection `0.057071`,
  while large-parent signal rejection is `0.465686`. Categorical therefore
  remains open on power/aggregation, not selected-null inflation in this smoke.
- The categorical block-gate smoke confirms that feature-block aggregation does
  not by itself close categorical transfer. Across `cat_clear_3cat_4c`,
  `cat_mod_3cat_4c`, and `cat_highcard_10cat_4c`, `block_bh` remains
  null-conservative, but its best minimum signal rejection over the tested
  penalty grid is `0.033166`; `coordinate_bh` is stronger at `0.049580`, and
  still signal-weak.

## Evidence

- `tests/validation/calibration/sibling/gates/99_test_data_independent_sibling_gate_panel.py` verifies
  the coordinate-wise p-values, summary statuses, contract behavior, and output
  writing.
- `raw/inbox/data-independent-sibling-gate-smoke-20260613.md` records the
  `50`-replicate smoke command, summary rates, and production decision.
- `raw/inbox/data-independent-sibling-gate-transfer-20260613.md` records the
  binary, binary/categorical, and categorical block-gate transfer smokes,
  including the scoped production-admissibility interpretation.

## Links

- [[projected-wald-statistic]]
- [[null-law-decomposition-panel-20260613]]
- [[production-admissibility-contract-20260613]]
- [[open-mathematical-questions]]
