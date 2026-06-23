---
title: Branch Length Candidate Run Gate 2026-06-18
type: source
status: reviewed
updated: 2026-06-18
sources:
  - raw/assets/benchmark-results/branch_length_candidate_run_gate_20260618/regression_gate_metadata.json
  - raw/assets/benchmark-results/branch_length_candidate_run_gate_20260618/regression_gate_comparison.csv
  - raw/assets/benchmark-results/branch_length_candidate_standard_gate_20260618/manifest.json
  - raw/assets/benchmark-results/branch_length_candidate_standard_gate_20260618/rows.csv
  - raw/assets/benchmark-results/branch_length_candidate_standard_gate_20260618/summary_by_method.csv
  - raw/assets/benchmark-results/branch_length_candidate_standard_gate_20260618/summary_by_case.csv
tags:
  - source
  - benchmarks
  - guarded
  - branch-length
---

# Branch Length Candidate Run Gate 2026-06-18

## Summary

This benchmark records the first standard `run_gate` comparison after the
guarded runner-contract fix, using `tbs`, `tbs_legacy_c2ef9a69`, and
`tbs_internal_filter_branch_length_v1`.

The canonical `run_gate` panel has `24` rows: eight gate-supported cases times
three methods. It confirms that the branch-length internal filter remains
fail-closed on unsupported severe overlap contexts, but it does not show a
broad completed-row ARI win over current TBS on this particular gate panel.
Branch-length remains the next candidate to test because its safety behavior is
better aligned with the guarded contract than legacy, not because this run
promotes it as production-ready.

## Key Points

- `run_gate` completed without the direct-dispatch workaround in `54.48`
  seconds.
- The canonical cases were `sbm_moderate`, `sbm_hard`,
  `cat_highcard_20cat_4c`, `cat_overlap_3cat_4c`,
  `overlap_heavy_4c_med_feat`, `overlap_unbal_4c_small`,
  `overlap_extreme_4c`, and `gauss_extreme_noise_3c`.
- `tbs` records mean ARI `0.4777`, median ARI `0.5001`, and exact K `2/8`.
  It skips `cat_highcard_20cat_4c`, `overlap_heavy_4c_med_feat`, and
  `overlap_extreme_4c` for missing strict-null or stopped-edge empirical-null
  support.
- `tbs_legacy_c2ef9a69` records mean ARI `0.3338`, median ARI `0.2481`, and
  exact K `4/8`. It completes every row, including severe unsupported overlap,
  where `overlap_extreme_4c` returns six clusters with ARI `0.002729`.
- `tbs_internal_filter_branch_length_v1` records mean ARI `0.3935`, median ARI
  `0.5001`, and exact K `1/8`. It skips both severe overlap rows for missing
  empirical-null support and skips several sparse-context rows as internally
  inadmissible.
- On rows where both current TBS and branch-length complete in the canonical
  gate, their labels and ARI match for `cat_overlap_3cat_4c` and
  `overlap_unbal_4c_small`. Branch-length additionally completes
  `cat_highcard_20cat_4c` as one cluster with ARI `0.0`, while current TBS
  skips that row.

## Supplemental Shared-Catalog Panel

A broader shared-catalog pass was also run because the `run_gate` case registry
does not include the binary controls and several categorical controls used in
the desired broader panel. That supplemental panel has `36` rows: twelve cases
times the same three methods.

On this broader panel, `tbs_internal_filter_branch_length_v1` records `11` OK
rows, `1` skip, `8` exact-K rows, mean completed-row ARI `0.784468`, and
median ARI `0.955960`. Current `tbs` records `9` OK rows, `3` skips, `6`
exact-K rows, mean completed-row ARI `0.848217`, and median ARI `0.960769`.
Legacy completes all `12` rows with `7` exact-K rows and mean ARI `0.716054`.

The supplemental result keeps branch-length in the candidate set because it
improves exact-K behavior on several binary and categorical controls while
remaining fail-closed on `overlap_extreme_4c`. It should not be read as a
production promotion: branch-length still over-skips some canonical gate
contexts and still returns a one-cluster `cat_highcard_20cat_4c` row in the
canonical run.

## Evidence

- `regression_gate_metadata.json` records the canonical `run_gate` case list,
  methods, elapsed time, thread environment, and row count.
- `regression_gate_comparison.csv` records the hard-overlap distinction:
  current TBS and branch-length skip `overlap_extreme_4c` for missing strict
  empirical-null support, while legacy returns six clusters with near-zero ARI.
- The same CSV records branch-length sparse-context skips with
  `status='undefined_sparse_context'`, separating internal-admissibility
  failures from missing strict empirical-null support.
- The supplemental `summary_by_method.csv` records the broader-panel exact-K
  and completed-row ARI aggregates used for the branch-length candidate
  interpretation.

## Links

- [[benchmark-runner-guarded-contract-fix-20260618]]
- [[manual-guarded-benchmark-run-direct-20260617]]
- [[legacy-c2ef9a69-edge-alpha-comparison-20260617]]
