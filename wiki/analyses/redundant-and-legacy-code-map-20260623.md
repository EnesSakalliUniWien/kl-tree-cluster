---
title: Redundant and Legacy Code Map 2026-06-23
type: analysis
status: reviewed
updated: 2026-07-28
sources:
  - tree_break_selection/space_separation/diffusion.py
  - tree_break_selection/space_separation/adaptive_cosine.py
  - tree_break_selection/space_separation/invariant_equivariant.py
  - tree_break_selection/tree/construction/build.py
  - tree_break_selection/tree/construction/hierarchical.py
  - tree_break_selection/hierarchy_analysis/bootstrap_consensus.py
  - tree_break_selection/hierarchy_analysis/decomposition/gates/guards.py
  - tree_break_selection/hierarchy_analysis/decomposition/gates/orchestrator.py
  - tree_break_selection/hierarchy_analysis/statistics/child_parent_divergence/child_parent_divergence_annotation/child_parent_divergence_annotation.py
  - tree_break_selection/hierarchy_analysis/tree_decomposition.py
  - tree_break_selection/tree/poset_tree.py
  - tree_break_selection/plot/image_panel.py
  - applications/endotypes/_shared.py
  - applications/endotypes/pipelines/run_feature_matrix_with_umap.py
  - applications/endotypes/plots/kak_signal_adaptive_umap_tree_page.py
  - applications/mnist/_shared.py
  - benchmarks/shared/audit_utils.py
  - benchmarks/shared/plots/export.py
  - benchmarks/shared/runners/method_registry.py
  - benchmarks/shared/runners/tbs_runner.py
  - benchmarks/shared/tbs_tree_context.py
  - benchmarks/diagnostics/runner_support.py
  - benchmarks/diagnostics/oracle/gate_path_trace.py
  - benchmarks/diagnostics/calibration/sibling/nulls/runner_support.py
  - benchmarks/validation/statistics/feature_covariance_calibration.py
  - benchmarks/validation/statistics/selected_pca_projected_wald_calibration.py
  - tests/validation/calibration/sibling/nulls/49_test_sibling_null_runner_support.py
  - tests/visualization/73_test_report_export_layout.py
tags:
  - code-audit
  - legacy
  - redundancy
---

# Redundant and Legacy Code Map 2026-06-23

## Summary

The former importable c2ef9a69 legacy package and its registry bridges were
retired on 2026-06-25. The 2026-07-27 recheck found no high-confidence dead
code in project-owned package, application, script, or benchmark Python files.

The 2026-07-28 clone census initially found 384 weak-mode clone pairs and 7,587
duplicated lines, or 4.14% of scanned production lines. This is not a blanket
deletion target. Diagnostic research panels accounted for 299 of the initial
pairs and 6,287 paired lines. The plot-export, core annotation/traversal,
endotype reference-data, and validation-calibration tranches were completed on
2026-07-28. The first contract-tested diagnostic-family cleanup then
consolidated classified-case execution and sibling-null preparation. Under the
same 10-line/70-token weak threshold, the current production scan reports 353
pairs and 6,542 duplicated lines, or 3.58%. Broad diagnostic consolidation
should continue one output-contract-tested family at a time.

## Details

The adaptive cosine weighting, eigendecomposition, segmentation, and block
coordinate logic previously lived inside a benchmark probe and was consumed by
applications. It now has one implementation in
`tree_break_selection/space_separation/adaptive_cosine.py`. The scRNA
invariant/equivariant decomposition likewise moved from a dataset script into
`tree_break_selection/space_separation/invariant_equivariant.py`.
The fixed Hamming-neighbor diffusion engine is now public beside adaptive and
block diffusion in `tree_break_selection/space_separation/diffusion.py`, so
applications and experiments no longer import a private benchmark-runner
function.

Two copied image-panel renderers were replaced by
`tree_break_selection/plot/image_panel.py`. Repeated endotype filename and
matrix-slug rules now use `applications/endotypes/_shared.py`; duplicated MNIST
summary-selection and compact digit-count parsing use
`applications/mnist/_shared.py`.

The tree-construction recheck found no second implementation of neighbor
joining, IQ-TREE import, MAD rooting, or branch-length NNLS. The main runner's
repeated dispatch, fallback, and metadata handling now live behind
`tree/construction/build.py`; direct linkage conversions use the maintained
constructor in `construction/hierarchical.py`. The unused sklearn and edge-list
adapters and shallow `PosetTree.from_*` pass-through methods were deleted with
no compatibility aliases. See [[tree-construction-method-map]].

A structural AST comparison still finds repeated small helpers across the
large calibration-diagnostic surface, especially `_require_columns`,
`_finite_float`, `_string_value`, `_json_default`, CLI parsers, and shard
plumbing. Some shared implementations already exist in
`benchmarks/shared/audit_utils.py`. Migrating dozens of active research panels
without contract tests would create more risk than the duplication currently
does, so this pass records the seam instead of applying a bulk rewrite.

The two validation programs
`benchmarks/validation/statistics/feature_covariance_calibration.py` and
`benchmarks/validation/statistics/selected_pca_projected_wald_calibration.py`
now share report provenance, Wilson intervals, synthetic continuous covariance
profiles, and p-value calibration summaries. Their distinct target validation,
report fields, and command interfaces remain local to each program.

### 2026-07-28 clone census

The production scan classified clone pairs by responsibility. Pair counts and
paired lines can overlap when one source fragment participates in several
matches, so they are prioritization signals rather than removable-line totals.

| Surface pair | Clone pairs | Paired lines | Interpretation |
| --- | ---: | ---: | --- |
| diagnostics to diagnostics | 281 | 5,611 | Research-panel scaffolding still dominates the total |
| benchmark support to benchmark support | 31 | 630 | Aligned records, generators, and validation reports |
| applications to applications | 17 | 289 | Endotype and plotting workflows, plus small report helpers |
| core to core | 5 | 72 | Gate arguments, covariance validation, and result fields |
| core to diagnostics | 6 | 97 | Diagnostic copies of production spectral and gate calculations |
| diagnostics to benchmark support | 6 | 109 | Validation and diagnostic report scaffolding |
| applications to diagnostics | 4 | 48 | Small cross-surface application probes |
| scripts to scripts | 2 | 26 | Repeated provenance verification branches |
| applications to benchmark support | 1 | 13 | One small shared workflow prefix |

Tests contain 58 weak-mode clone pairs and 1,298 duplicated lines, or 2.81% of
46,177 scanned lines. Most are fixture construction and repeated contract
cases. They should be replaced only when a shared public-interface test
expresses the same behavior; line similarity alone is not evidence that a test
is redundant.

The actionable clusters, in cleanup order, are:

1. Completed on 2026-07-28: `benchmarks/shared/plots/export.py` had repeated
   UMAP label assembly, rendering, filename, and output routing across the
   UMAP-only, UMAP-then-tree, and tree-then-UMAP workflows. The two uncalled
   interfaces were deleted, the live workflow became
   `create_case_report_pages_from_results()`, and no compatibility aliases
   remain. Shared figure output routing now also serves the 3D UMAP and
   manifold engines. The change removed a net 174 lines from `export.py`.
2. Completed on 2026-07-28: `_run_tbs_method` was deleted and the method
   registry now dispatches directly to `run_tbs_on_distance()`. Registry
   entries own the linkage and trace policy that previously hid in the
   forwarding adapter.
3. Completed on 2026-07-28: gate configuration is owned by
   `run_gate_annotation_pipeline()` and its `GateAnnotationBundle`.
   `TreeDecomposition` now consumes either that completed bundle or an
   explicit three-column traversal decision frame; it does not reconstruct,
   compare, or silently recompute annotation configuration. The
   `PosetTree.decompose()` forwarding facade was removed, and callers now show
   annotation production and traversal as separate phases.
4. Completed on 2026-07-28: selected-root and selected-global-sibling
   permutation guards now share one private validation, null-sampling, and
   Monte Carlo driver while retaining their distinct statistic callbacks and
   public contracts.
5. Completed on 2026-07-28: child-parent annotation now has one canonical
   function and one return contract containing both the annotated frame and
   spectral context. The DataFrame-only forwarding path was deleted without
   an alias.
6. Completed on 2026-07-28: the apparent endotype plotting clone was traced to
   reference-data plumbing, not rendering. The feature-matrix pipeline and KAK
   page now consume one canonical reference-endotype parser and one
   symbol-to-Entrez batch resolver from `applications/endotypes/_shared.py`.
   Both copied implementations were deleted without aliases.
7. Completed on 2026-07-28: the two covariance-calibration programs now consume
   shared provenance and Wilson helpers from
   `benchmarks/validation/contracts/report_contract.py` and shared numerical
   primitives from
   `benchmarks/validation/statistics/calibration_support.py`. Exact CSV header
   tests preserve their different output schemas; target-specific validation
   and CLIs were not flattened into a generic driver.
8. Completed on 2026-07-28: sibling-null and gate-path runners now resolve
   classified benchmark cases, runtime defaults, and result directories through
   `benchmarks/diagnostics/runner_support.py`. Prepared tree, annotation,
   traversal, and oracle context belongs to `oracle/gate_path_trace.py`;
   sibling-null table preparation and target selection belong to the
   sibling-null `runner_support.py`. Four copied sibling setup implementations
   and the gate-path copy were deleted without aliases. This also corrected the
   sibling runners' copied `repo_root` calculation, which previously pointed
   default discovery and output under
   `benchmarks/diagnostics/benchmarks/...` instead of the project
   `benchmarks/results/` directory.
9. Overlap, root-selection, and traversal diagnostic families now contain the
   largest actionable clone groups. Each family needs one observable output
   contract and one domain-owned driver before copied scripts are removed.

Several high-ranked matches should not be mechanically consolidated:

- `benchmarks/shared/metrics.py`,
  `benchmarks/shared/result_records/models.py`,
  `benchmarks/shared/result_records/dataframe.py`, and plot/runtime modules
  repeat record-field sequences that define aligned schemas.
- `SpectralContext` and `SpectralDecompositionResult` share fields but have
  different ownership and lifecycle semantics.
- Bernoulli and grouped-categorical covariance validators share structural
  checks but enforce different feature-family invariants.
- Test setup that looks alike can cover different public contracts.

## Evidence

- Vulture at 90% confidence reported no unused project-owned code after the
  reorganization.
- A repository-wide `jscpd` weak-mode scan with a 10-line/70-token production
  threshold measured 384 clone pairs across 400 files, 7,587 duplicated lines
  out of 183,106, and 46,149 duplicated tokens out of 980,227.
- The corresponding 12-line/80-token test scan measured 58 clone pairs across
  246 files and 1,298 duplicated lines out of 46,177.
- Exact normalized AST function-body comparison found 48 repeated function
  groups. Its largest exact groups are diagnostic selectors and threshold
  helpers; the production-facing exact groups corroborate the guard and
  calibration seams above.
- Pylint's independent `duplicate-code` check corroborated the diagnostic and
  validation families while retaining a 9.94/10 repository score.
- After the plot-export consolidation, the same production clone scan fell
  from 384 to 381 pairs and from 7,587 to 7,489 duplicated lines while scanned
  production lines fell from 183,106 to 182,932. The only remaining
  `export.py` matches are common argument and initialization prefixes of
  distinct 3D UMAP, manifold, and tree interfaces, not copied rendering
  implementations.
- All 20 visualization tests pass through the maintained case-report interface,
  and direct search finds no callers or aliases for the deleted plotting names.
- After the runner, annotation/traversal, permutation, and child-parent
  consolidation, the production scan fell again to 373 clone pairs and 7,259
  duplicated lines across 182,914 lines (3.97%). The removed target pairs no
  longer appear; the remaining core matches are different internal or
  cross-surface seams.
- After the endotype reference-data and validation-calibration consolidation,
  the scan fell to 370 clone pairs and 7,175 duplicated lines across 182,862
  lines (3.92%). The copied endotype HTTP/parser blocks and validation
  covariance, p-value-summary, Wilson, Git-state, and UTC helpers no longer
  appear.
- After the classified diagnostic runner and sibling-null context
  consolidation, the scan fell to 353 clone pairs and 6,542 duplicated lines
  across 182,496 lines (3.58%). The former 330-line Gaussian/selection runner
  cluster, 231-line Gaussian/inflation cluster, and gate-path preparation
  copies no longer appear. The remaining sibling-runner matches are small
  command and report-schema prefixes.
- The shared-runner interface is covered by six direct tests, including a real
  prepared regression-gate case and the corrected default result-directory
  contract. All 45 focused oracle/sibling tests and all 1,278 ordered
  repository tests pass; Ruff, Vulture at 90% confidence, and wiki lint over
  271 pages also pass.
- Vulture reported no project-owned dead code at 90% confidence after the
  deletions. Ruff, wiki lint over 271 pages, the focused interface suites, and
  all 1,254 ordered repository tests pass.
- The complete ordered repository gate passes all 1,260 tests after the
  interface deletion and rename.
- Ruff passes over the relocated library, application, benchmark, and test
  surfaces.
- Focused method, endotype, scRNA, MNIST, and benchmark-interface tests pass.
- Exact function-body comparison identified the remaining repeated diagnostic
  helpers described above; it found and motivated the endotype and MNIST
  application helper consolidation.
- Exact constructor search separated the three live registered topology
  builders from direct representation conversions and proved the two removed
  adapters had no production callers.
- All 50 endotype and statistical-validation focused tests and all 1,272
  ordered repository tests pass after the latest consolidation. Ruff passes
  over every production and test surface, and Vulture again reports no
  project-owned dead code at 90% confidence.

## Links

- [[project-overview]]
- [[repository-hygiene-and-completion-audit-20260727]]
- [[method-application-and-plot-seams-20260727]]
- [[tree-construction-method-map]]

## Open Questions

- Which diagnostic helper contracts are stable enough to move into
  `benchmarks/shared/` without masking panel-specific validation semantics?
- Should the remaining internal argument overlap inside the gate orchestrator
  become a typed request object, or would that hide experimentally important
  gate inputs?
- After application output contracts are covered, should direct
  distance/linkage/`PosetTree` sequences share one deep construction interface
  in `tree_break_selection/tree/`?
