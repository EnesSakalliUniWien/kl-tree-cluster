---
title: Root Selected Binary Resolution 2026-06-17
type: source
status: draft
updated: 2026-06-17
sources:
  - benchmarks/diagnostics/calibration/root_selected_binary_resolution_panel.py
  - tests/validation/181_test_root_selected_binary_resolution_panel.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_binary_resolution_mild_replay_v3_smoke
tags:
  - source
  - diagnostics
  - root
  - spectral
  - selected-tail
---

# Root Selected Binary Resolution 2026-06-17

## Summary

`root_selected_binary_resolution_panel.py` encodes the corrected root idea for
the current method: Tree-Break Selection always returns a binary hierarchy, so the root is
always represented as \(r\to(L,R)\), but the first binary split can be only one
selected refinement of a weak or unresolved top-level event.

The panel records the descriptive coordinate
\[
\rho_r = T\min(A,E),
\]
where \(T\) is selected tie-rank fraction, \(A\) is selected-ratio/action
log1p, and \(E\) is edge-margin/action log1p. This is a root-resolution
coordinate, not a p-value and not a rescue rule. The selected-root spectral
tail still controls the method action.

## Key Points

- The mild replay writes `7` binary-root resolution rows.
- Resolution bands are `2` weak, `3` transition, and `2` strong.
- `2/7` roots have existing selected-root tail support.
- `5/7` roots remain fail-closed.
- Among the fail-closed roots, `1` is moment-only/geometrically feasible but
  underpowered, and `4` require new same-geometry nonzero spectral-tail support.
- The median binary-resolution strength is `5.173422`.

The important method implication is that apparent binary-root strength is not
itself calibration. A strong selected binary root still fails closed when the
selected-root \(S_{H_u}\) tail law has no same-stratum support.

## Evidence

- `root_selected_binary_resolution_summary.csv` reports
  `weak_resolution_count = 2`, `transition_resolution_count = 3`,
  `strong_resolution_count = 2`, `existing_tail_support_count = 2`,
  `new_spectral_support_required_count = 4`, and `fail_closed_count = 5`.
- `root_selected_binary_resolution_rows.csv` marks
  `overlap_part_4c_small` as a strong binary-resolution root that still
  requires new selected spectral-tail support.
- `181_test_root_selected_binary_resolution_panel.py` verifies that existing
  tail support overrides weak-resolution concern, while strong binary
  resolution still fails closed when the selected spectral law is missing.

## Links

- [[root-selected-external-law-equation-20260617]]
- [[root-selected-conditional-tilt-feasibility-20260617]]
- [[root-selected-spectral-tail-law-with-legacy-overlay-20260617]]
- [[root-selected-mixed-region-law-20260616]]
