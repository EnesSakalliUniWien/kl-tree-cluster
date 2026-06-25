---
title: Redundant and Legacy Code Map 2026-06-23
type: analysis
status: draft
updated: 2026-06-23
sources:
  - pyproject.toml
  - tree_break_selection/legacy_methods/commit_c2ef9a69/METADATA.md
  - benchmarks/shared/runners/legacy_commit_runner.py
  - benchmarks/shared/runners/method_registry.py
  - benchmarks/shared/util/method_sets.py
  - tree_break_selection/hierarchy_analysis/decomposition/gates/orchestrator.py
  - tree_break_selection/hierarchy_analysis/decomposition/gates/spectral_transport.py
  - benchmarks/validation/feature_covariance_calibration.py
  - benchmarks/validation/selected_pca_projected_wald_calibration.py
tags:
  - code-audit
  - legacy
  - redundancy
---

# Redundant and Legacy Code Map 2026-06-23

## Summary

The project has one intentional legacy code island and several repeated
diagnostic scaffolds.

The intentional legacy island is
`tree_break_selection/legacy_methods/commit_c2ef9a69/`, a full importable copy
of the old `tree_break_selection` package. Its metadata says it exists for
diagnostic old-versus-current comparisons and is not a production path.
However, it remains live through benchmark registries and tests, so it should
not be deleted without first retiring `tbs_legacy_c2ef9a69` and related
comparison panels.

The main redundancy outside that snapshot is repeated benchmark/report
plumbing: JSON serialization helpers, path builders, finite-number coercion,
CSV-grid parsers, validation-report contracts, AWS shard resolution, and small
test row factories. These are mostly maintenance-cost issues rather than
runtime hazards.

## Details

### Static inventory

A local static pass over `tree_break_selection`, `benchmarks`, `tests`, and
`scripts` found:

- `803` project Python files.
- `667` non-legacy Python files.
- `134` Python files inside the c2ef9a69 legacy snapshot.
- Approximately `187,672` non-legacy source lines and `8,369` legacy-snapshot
  source lines, counting nonblank noncomment lines.

Current-vs-legacy path comparison found:

- `19` files with identical bytes at the same relative path, mostly
  `__init__.py`, stopping-edge recovery helpers, and `plot/config.py`.
- `69` files present in both current and legacy packages but changed.
- `46` legacy-only files absent from the current package.
- `27` current-only files absent from the legacy snapshot.

### Legacy code map

The legacy package is intentionally vendored under a nested namespace. The live
bridge is `benchmarks/shared/runners/legacy_commit_runner.py`, which imports
the legacy commit metadata and legacy `PosetTree`, builds a SciPy linkage tree,
calls the old `decompose` method, and returns a standard `MethodRunResult`.

The benchmark registry keeps this callable through:

- `tbs_legacy_c2ef9a69` in `benchmarks/shared/runners/method_registry.py`.
- `tbs_legacy_internal_spectral_diagnostic` in the same registry, which is a
  current runner configured to imitate an older internal spectral diagnostic.
- `tbs_rescued_legacy_v1`, a guarded current-runner hybrid that combines
  branch-length state, internal support thresholds, spectral transport
  pass-through, and regional bandwidth support.
- `TBS_DISTANCE_TREE_METHODS` in `benchmarks/shared/util/method_sets.py`, which
  includes the legacy and rescued-legacy method ids.

Legacy-only implementation families in the c2ef9a69 snapshot include:

- `decomposition/backends/random_projection/`.
- `decomposition/core/contracts.py`.
- `statistics/categorical_mahalanobis.py`.
- `child_parent_divergence/single_feature_subtree_policy/`.
- `sibling_divergence/adjusted_wald_annotation/`.
- `sibling_divergence/pair_testing/sibling_null_prior_interpolation/`.
- Older pooled-variance, branch-length, and projection metadata helpers.

These should be treated as archived comparator implementation unless a current
module imports them directly through the legacy namespace.

### Redundancy map

Exact duplicate non-legacy Python files, excluding `__init__.py`, were not
found in the scanned project surfaces. The repeated code is function-level and
pattern-level.

High-priority source redundancy:

- `tree_break_selection/hierarchy_analysis/decomposition/gates/orchestrator.py`
  repeats `_annotation_bool`, `_node_split_prerequisites`, and
  `_node_sibling_gate_open` from
  `tree_break_selection/hierarchy_analysis/decomposition/gates/spectral_transport.py`.
  This is production-adjacent and should be extracted to a shared gate helper
  before either path diverges further.
- `benchmarks/validation/feature_covariance_calibration.py` and
  `benchmarks/validation/selected_pca_projected_wald_calibration.py` repeat
  validation-report helpers such as `_validate_run_inputs`,
  `_completed_target_entry`, and `_validate_complete_report_context`.
  A shared validation report contract module would reduce drift.

Medium-priority benchmark redundancy:

- AWS/cloud benchmark scripts repeat `resolve_shard_index` and command entry
  plumbing.
- Many calibration panels repeat `_write_manifest`, `_json_default`,
  `_finite_float`, `_require_columns`, `rows_path`, `summary_path`,
  `_parse_csv_list`, `_parse_float_grid`, `_bool_value`, and `_select_cases`.
  Existing shared surfaces such as `benchmarks/shared/audit_utils.py` are the
  natural place to consolidate stable versions, but only after the affected
  panels are no longer changing daily.
- `tests/validation` repeats small `_row`, `_target`, and `_rows` builders.
  These are low-risk test-local duplication unless schema drift becomes noisy.

Low-priority cleanup signals:

- Ruff `F401/F841` found three concrete unused-code items:
  `typing.Any` in
  `benchmarks/diagnostics/spectral/adaptive_cosine_kak_benchmark_probe.py`,
  unused `cluster_summary` assignment in
  `benchmarks/diagnostics/spectral/kak_feature_subspace_clustering.py`, and
  unused `numpy` import in `scripts/plot_pancreas_tbs_readable_umap_clusters.py`.
- Local `__pycache__` directories under source paths are generated noise. They
  should remain ignored rather than treated as code.

### Cleanup order

1. Fence the legacy snapshot explicitly: keep it importable only for named
   comparator methods, and document that production code must not import it.
2. Decide whether `tbs_legacy_c2ef9a69`, `tbs_legacy_internal_spectral_diagnostic`,
   and `tbs_rescued_legacy_v1` remain benchmark-contract methods. If yes, keep
   the snapshot. If no, remove their registry entries, tests, and dependent
   panels first.
3. Extract the duplicated gate annotation helpers from orchestrator and
   spectral transport into one shared gate module.
4. Consolidate validation-report helpers used by the two calibration validation
   scripts.
5. Move stable benchmark-panel plumbing into shared utilities only after the
   actively changing diagnostic panels settle.
6. Apply the small Ruff cleanup items opportunistically.

## Evidence

- `tree_break_selection/legacy_methods/commit_c2ef9a69/METADATA.md` says the
  snapshot is a full copy of commit `c2ef9a69e0888168950bdee4a41ae8ab9996e32f`,
  mechanically nested for same-process imports, and intended for diagnostic
  old-versus-current comparisons rather than production.
- `benchmarks/shared/runners/legacy_commit_runner.py` is the direct bridge from
  benchmark dispatch into the legacy `PosetTree`.
- `benchmarks/shared/runners/method_registry.py` and
  `benchmarks/shared/util/method_sets.py` keep legacy, legacy-diagnostic, and
  guarded-rescued legacy method ids available to the benchmark runner.
- `tree_break_selection/hierarchy_analysis/decomposition/gates/orchestrator.py`
  and `tree_break_selection/hierarchy_analysis/decomposition/gates/spectral_transport.py`
  contain duplicated gate annotation predicates.
- `benchmarks/validation/feature_covariance_calibration.py` and
  `benchmarks/validation/selected_pca_projected_wald_calibration.py` contain
  duplicated validation-report contract helpers.

## Links

- [[project-overview]]
- [[legacy-c2ef9a69-method-package-20260616]]
- [[legacy-c2ef9a69-method-comparison-panel-20260616]]
- [[old-current-method-difference-ledger-20260617]]

## Open Questions

- Should the c2ef9a69 snapshot remain an importable package, or should future
  legacy comparisons use archived outputs only?
- Which benchmark-panel helpers are stable enough to consolidate without
  slowing down ongoing diagnostic work?
