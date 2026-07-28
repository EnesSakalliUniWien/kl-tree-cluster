---
title: Root Selected Population Law Requirement 2026-06-17
type: source
status: draft
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/root/selected/root_selected_population_law_requirement_panel.py
  - tests/validation/calibration/root/selected/174_test_root_selected_population_law_requirement_panel.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_population_law_requirement_mild_accumulated
tags:
  - source
  - diagnostics
  - root
  - spectral
  - population-law
---

# Root Selected Population Law Requirement 2026-06-17

## Summary

`root_selected_population_law_requirement_panel.py` quantifies the local
population-law correction that would be required after the one-sided
action-dominance diagnostic. For each target, it compares the observed
\(S_{\mathrm{root}}\) with the largest supported spectral excess available in
the exact or action-dominating support set and computes

\[
\kappa_H
=
\exp\{S_{\mathrm{target}}-S_{\mathrm{support,max}}\}.
\]

Here \(\kappa_H\) is the MP-edge multiplier that would make the target no more
spectrally extreme than the available support. It is a diagnostic requirement
for \(H_u\), not a calibration p-value.

The mild accumulated run shows that the three action-dominance fail-closed
roots require modest local MP-edge movement after the root MP edge is recomputed
with the true active feature count:
`overlap_mod_4c_small` needs multiplier `1.327024`,
`overlap_mod_6c_med` needs `1.432777`, and
`overlap_part_4c_small` needs `1.459352`. Thus \(H_u\) is still a required
object for those roots, but the correction is no longer the large factor seen
under the earlier compressed-spectrum artifact.

## Key Points

- The panel writes `7` requirement rows and one summary row.
- `requirement_available_count = 5`: two exact-supported roots and three
  action-dominance roots have enough support to quantify a requirement.
- `exact_no_requirement_count = 1` for `overlap_extreme_4c`, because exact
  root-tail support already exists and no \(H_u\) correction is needed for
  production inference.
- `overlap_unbal_6c_med` has exact support but still has no spectral
  exceedance; its diagnostic multiplier is `1.194335`, while production
  inference defers to the exact root-tail panel.
- `overlap_heavy_4c_small_feat` and `overlap_unbal_4c_small` still have
  missing support, so an \(H_u\) requirement cannot yet be estimated.
- `overlap_mod_4c_small` remains the old-commit risk case:
  `legacy_full_method_leaks_selected_null_root_risk`.

## Evidence

The derivation is direct:

\[
S_{\mathrm{root}}
=
\log\left(\frac{\lambda}{\lambda_{\mathrm{MP}}}\right).
\]

If the local reference edge is replaced by
\(\lambda_H=\kappa_H\lambda_{\mathrm{MP}}\), then

\[
S_H
=
\log\left(\frac{\lambda}{\lambda_H}\right)
=
S_{\mathrm{root}}-\log\kappa_H.
\]

To make the target match the largest supported spectral excess,

\[
S_{\mathrm{target}}-\log\kappa_H
=
S_{\mathrm{support,max}},
\]

so

\[
\kappa_H
=
\exp\{S_{\mathrm{target}}-S_{\mathrm{support,max}}\}.
\]

The refreshed observed multipliers are all below `1.5` for the supported
contexts. This changes the interpretation from "large identity-MP failure" to
"modest deformed-edge calculation still required." The remaining production
rule is unchanged: these rows are diagnostic until \(H_u\) is estimated or a
selected spectral-tail law is derived.

## Links

- [[root-selected-action-dominance-tail-20260617]]
- [[root-selected-action-conditioning-ladder-20260617]]
- [[local-marchenko-pastur-rule]]
- [[selected-neighborhood-measurability-law]]
