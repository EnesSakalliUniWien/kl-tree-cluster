---
title: Root Selected Action Dominance Tail 2026-06-17
type: source
status: draft
updated: 2026-06-17
sources:
  - benchmarks/diagnostics/calibration/root_selected_action_dominance_tail_panel.py
  - tests/validation/173_test_root_selected_action_dominance_tail_panel.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_action_dominance_tail_mild_accumulated
tags:
  - source
  - diagnostics
  - root
  - action
  - spectral
---

# Root Selected Action Dominance Tail 2026-06-17

## Summary

`root_selected_action_dominance_tail_panel.py` tests the one-sided version of
the action-conditioning idea. It keeps \(T,E,B,H_u\) fixed and asks whether
there are selected-null or external-null support rows with
\[
A_{\mathrm{support}}\ge A_{\mathrm{target}}.
\]
Those rows are action-dominating support, but they are not exact
\(T,A,E,B,H_u\) support and therefore do not produce production p-values. A
one-sided monotonicity theorem would be required before they could become a
calibration rule.

The mild accumulated run finds action-dominating support for four targets:
the exact-supported `overlap_unbal_6c_med` plus the three action-only gap
targets `overlap_mod_4c_small`, `overlap_mod_6c_med`, and
`overlap_part_4c_small`. None of the action-dominating support rows exceed the
observed root spectral excess. The three action-only roots therefore remain
fail-closed and their next mathematical step is to prove or reject one-sided
action spectral-tail monotonicity.

## Key Points

- The panel writes `7` target rows and one summary row.
- `exact_supported_target_count = 2`, unchanged from the root-tail panel.
- `action_dominating_supported_target_count = 4`.
- `action_dominating_no_spectral_exceedance_count = 4`.
- For `overlap_mod_4c_small`, `overlap_mod_6c_med`, and
  `overlap_part_4c_small`, action-dominating support exists, but
  `action_dominating_spectral_exceedance_count = 0`.
- The diagnostic plus-one p-values for those three roots are `1/3`, but they
  are labeled `diagnostic_one_sided_action_monotonicity_required`, not
  production p-values.
- `overlap_heavy_4c_small_feat` and `overlap_unbal_4c_small` still have no
  action-dominating support in the same \(T,E,B,H_u\) context.
- `overlap_mod_4c_small` remains the old-commit risk case:
  `legacy_full_method_leaks_selected_null_root_risk`.

## Evidence

The tested diagnostic condition is

\[
(T,E,B,H_u)_{\mathrm{support}}
=
(T,E,B,H_u)_{\mathrm{target}},
\qquad
A_{\mathrm{support}}\ge A_{\mathrm{target}}.
\]

The spectral tail is still evaluated as

\[
S_{\mathrm{support}}\ge S_{\mathrm{target}}.
\]

For the three action-only gap roots, the first condition is populated but the
second condition has zero exceedances. Thus one-sided action dominance is not
enough as an empirical shortcut. It is a candidate theorem obligation:
without a proof that the selected spectral tail under lower \(A\) is bounded
by the observed higher-\(A\) support distribution, these rows cannot calibrate
the root split.

## Links

- [[root-selected-action-conditioning-ladder-20260617]]
- [[root-selected-spectral-tail-nearest-support-20260617]]
- [[root-selected-importance-tail-support-20260617]]
- [[selected-neighborhood-measurability-law]]
