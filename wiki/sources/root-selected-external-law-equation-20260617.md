---
title: Root Selected External Law Equation 2026-06-17
type: source
status: draft
updated: 2026-06-17
sources:
  - benchmarks/diagnostics/calibration/root_selected_external_law_equation_panel.py
  - tests/validation/180_test_root_selected_external_law_equation_panel.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_external_law_equation_mild_replay_v3_smoke/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_external_law_equation_mild_replay_v3_smoke/root_selected_external_law_equation_rows.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_external_law_equation_mild_replay_v3_smoke/root_selected_external_law_equation_summary.csv
tags:
  - source
  - diagnostics
  - root
  - spectral
  - selected-tail
---

# Root Selected External Law Equation 2026-06-17

## Summary

`root_selected_external_law_equation_panel.py` converts the failed finite-support
tilt check into explicit external-law equations for selected-root calibration.
It keeps the binary-root tree fixed, but states what the selected-null or
external law must generate:
\[
\mathbb E_Q[\phi_{\mathrm{root}}\mid R_{\mathrm{root}},T,A,E,B,H_u]
=
\phi_{\mathrm{target}}
\]
and
\[
Q(S_{H_u}\ge S_{H_u,\mathrm{target}}
\mid R_{\mathrm{root}},T,A,E,B,H_u)>0.
\]

The panel is diagnostic-only. It does not create p-values and does not rescue
root splits.

## Key Points

- The mild replay writes `7` equation rows.
- `2/7` roots have existing selected-root tail support and defer to the
  deformed tail panel.
- `1/7` root has required-axis moment feasibility but is still diagnostic-only
  because same-stratum tail support is underpowered.
- `4/7` roots require new same-stratum support with nonzero \(S_{H_u}\).
- The maximum required \(S_{H_u}\) gap to the current support hull is
  `0.864845`.
- The maximum required deformed-ratio lower bound is `2.374638`.
- With target alpha `0.01`, the minimum support count for one-exceedance
  alpha resolution is `99`.

This separates three states that were previously easy to conflate: existing
support, moment-only geometric feasibility, and true missing positive spectral
tail support.

## Evidence

- `root_selected_external_law_equation_summary.csv` reports
  `existing_tail_support_count = 2`,
  `moment_only_reweighting_possible_count = 1`, and
  `new_spectral_support_required_count = 4`.
- `root_selected_external_law_equation_rows.csv` marks
  `overlap_mod_4c_small`, `overlap_mod_6c_med`,
  `overlap_part_4c_small`, and `overlap_unbal_4c_small` as requiring new
  same-stratum nonzero \(S_{H_u}\) support.
- `180_test_root_selected_external_law_equation_panel.py` verifies the
  nonzero-tail-support requirement, the moment-only feasible state, and the
  existing-support defer state.

## Links

- [[root-selected-conditional-tilt-feasibility-20260617]]
- [[root-selected-deformed-external-law-target-20260617]]
- [[root-selected-deformed-tail-support-gap-20260617]]
