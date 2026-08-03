# Unsupported Benchmark Outcomes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export calibration-unsupported TBS runs as typed `unsupported` outcomes, keep them out of quality metrics while reporting coverage, and replace the misleading Gaussian case with explicit dense-signal and sparse-signal/noise cases.

**Architecture:** Extend the existing `MethodRunResult` → dispatcher → `BenchmarkResultRow` pipeline with one shared status enum and typed reason. Detect missing empirical-null support in the TBS runner immediately after gate annotation. Reuse the current dimensional Gaussian generator, performance grid, and plot/report pipeline.

**Tech Stack:** Python 3.11, dataclasses, Enum, NumPy, pandas, SciPy, scikit-learn, pytest, Ruff, Matplotlib, repository benchmark/wiki tooling.

## Global Constraints

- `unsupported` is a recognized unavailable scientific contract, not an exception or ordinary skip.
- Unsupported runs have no labels, `found_clusters=0`, `labels_length=0`, and `NaN` clustering metrics.
- Quality aggregates use only `status=ok`; coverage reports unsupported counts/rates by method and case family.
- The TBS runner owns the decision; dispatch and reporting only validate and propagate it.
- Do not promote fixed-coordinate BH, add cross-fitting, change alphas, or change tree/diffusion/NNLS behavior.
- Remove `gauss_extreme_noise_highd` from active registration without an alias; do not modify historical raw assets.
- Reuse `dimensional_gaussian` for the 12-informative/19,988-noise binary case.
- Preserve current dirty-worktree changes and never stage unrelated hunks.
- Require fresh verification evidence before every completion or commit claim.

---

### Task 1: Define the authoritative typed outcome contract

**Files:**
- Create: `benchmarks/shared/types/run_status.py`
- Create: `benchmarks/shared/types/unsupported_reason.py`
- Modify: `benchmarks/shared/types/method_run_result.py`
- Modify: `benchmarks/shared/types/__init__.py`
- Modify: `benchmarks/shared/result_records/models.py`
- Modify: `benchmarks/shared/result_records/__init__.py`
- Modify: `benchmarks/shared/result_records/factory.py`
- Test: `tests/validation/43_test_result_status_validation.py`

**Interfaces:**
- Produces: `BenchmarkRunStatus`, `UnsupportedReasonCode`, `UnsupportedEvidence`, `UnsupportedReason`, and validated `MethodRunResult` states.
- Consumes: existing runner call sites that pass string statuses; the enum remains string-valued.

- [ ] **Step 1: Write failing tests for all three states**

Add a valid unsupported construction and invalid state combinations:

```python
reason = UnsupportedReason(
    code=UnsupportedReasonCode.EMPIRICAL_NULL_NO_INTERNAL_SUPPORT,
    stage="sibling_calibration",
    message="No admissible internal empirical-null support.",
    evidence=UnsupportedEvidence(39, 0, 39, 78, 78),
)
result = MethodRunResult(None, 0, None, "unsupported", None, {}, reason)
assert result.status is BenchmarkRunStatus.UNSUPPORTED
```

Assert `ValueError` for unsupported-with-labels, unsupported-without-reason,
skip-with-reason, ok-without-labels, negative evidence, and unknown status.

- [ ] **Step 2: Run the focused test and confirm it fails for missing types**

Run: `uv run pytest tests/validation/43_test_result_status_validation.py -q`

Expected: import/assertion failures for the absent enum member and reason models.

- [ ] **Step 3: Implement shared models and result invariants**

Define the string enum values `ok`, `skip`, `unsupported`; define the initial
reason code `empirical_null_no_internal_support`; require stage
`sibling_calibration`, a nonempty message, five nonnegative counts, and zero
admissible support. `MethodRunResult.__post_init__` normalizes strings to the
enum and enforces exclusive labels, skip reason, unsupported reason, and cluster
count rules. Keep `extra` unchanged.

- [ ] **Step 4: Move all result rows to the shared enum**

Delete the enum definition in `result_records/models.py`, update direct imports,
and remove the old result-record re-export. Do not create a compatibility alias.

- [ ] **Step 5: Run focused contract tests**

Run: `uv run pytest tests/validation/43_test_result_status_validation.py tests/core/15_test_benchmark_decomposition_utils.py -q`

Expected: all selected tests pass.

### Task 2: Detect unsupported empirical-null calibration inside TBS

**Files:**
- Create: `benchmarks/shared/runners/tbs_support.py`
- Modify: `benchmarks/shared/runners/tbs_runner.py`
- Modify: `benchmarks/shared/runners/dispatch.py`
- Create: `tests/integration/64_test_tbs_unsupported_outcome.py`
- Modify: `tests/pipeline/51_test_dispatch_contract.py`

**Interfaces:**
- Consumes: stamped columns `Sibling_Gate_P_Value_Calibration`, `Sibling_Role_Supported`, `Sibling_Divergence_Invalid`, `Child_Parent_Divergence_Tested`, and `Child_Parent_Divergence_Significant`.
- Produces: `unsupported_empirical_null_reason(annotations, sibling_gate_method) -> UnsupportedReason | None` and an early unsupported TBS result.

- [ ] **Step 1: Write a failing compact stamped-table test**

Build 39 focal rows with `undefined_no_internal_support`, zero supported roles,
39 invalid siblings, and 78 tested/rejected child edges. Assert exact reason
counts. Assert `None` for `fixed_coordinate_bh` and for a supported table.

- [ ] **Step 2: Write the real high-dimensional regression**

Run current Hamming/average TBS on the dense 40×20,000 recipe at edge alpha
0.001 and sibling alpha 0.01. Require `unsupported`, no labels, zero clusters,
zero admissible support, and 78 upstream rejections. Require the same data with
`fixed_coordinate_bh` to remain `ok`.

- [ ] **Step 3: Run the regression and confirm the old one-cluster failure**

Run: `uv run pytest tests/integration/64_test_tbs_unsupported_outcome.py -q`

Expected: failure because empirical-null TBS currently returns `ok`, one cluster.

- [ ] **Step 4: Implement the helper and early TBS return**

After gate annotation, call the helper using the resolved sibling-gate method.
When it returns a reason, return `MethodRunResult` with no labels/report, zero
clusters, status unsupported, the typed reason, and existing tree/annotation/
gate/timing/branch metadata. Return before `TreeDecomposition`.

- [ ] **Step 5: Preserve unsupported in dispatch**

Give `_normalize_method_result` explicit `ok`, `skip`, and `unsupported`
branches. Preserve the typed reason and `extra`; do not convert invalid `ok`
results into skips.

- [ ] **Step 6: Run runner and dispatcher tests**

Run: `uv run pytest tests/integration/64_test_tbs_unsupported_outcome.py tests/pipeline/51_test_dispatch_contract.py -q`

Expected: all selected tests pass.

### Task 3: Propagate evidence through result rows and metric execution

**Files:**
- Modify: `benchmarks/shared/result_records/models.py`
- Modify: `benchmarks/shared/result_records/factory.py`
- Modify: `benchmarks/shared/result_records/dataframe.py`
- Modify: `benchmarks/shared/util/method_execution.py`
- Modify: `benchmarks/shared/relationship_analysis.py`
- Modify: `benchmarks/shared/plots/runtime.py`
- Create: `tests/pipeline/83_test_unsupported_result_pipeline.py`
- Modify: `tests/validation/40_test_cluster_validation_core.py`

**Interfaces:**
- Consumes: `MethodRunResult.unsupported_reason`.
- Produces: eight stable unsupported columns and no `ComputedResultRecord` for unsupported runs.

- [ ] **Step 1: Write a failing pipeline test with a stub unsupported runner**

Assert zero clusters/labels, `NaN` quality and split metrics, empty skip reason,
exact flattened reason/evidence, and no computed result.

- [ ] **Step 2: Run focused tests and confirm missing schema fields**

Run: `uv run pytest tests/pipeline/83_test_unsupported_result_pipeline.py tests/validation/40_test_cluster_validation_core.py -q`

Expected: failures for absent unsupported fields.

- [ ] **Step 3: Extend the canonical schema**

Add after `skip_reason`: `unsupported_reason_code`, `unsupported_stage`,
`unsupported_reason`, `unsupported_focal_record_count`,
`unsupported_admissible_support_count`, `unsupported_invalid_record_count`,
`unsupported_upstream_tested_count`, and `unsupported_upstream_rejected_count`.
Non-unsupported text serializes empty; numeric evidence serializes `NaN`.

- [ ] **Step 4: Branch execution on explicit status**

Calculate metrics and create `ComputedResultRecord` only for `ok`. Both skip and
unsupported get `NaN` metrics/cluster errors. Pass the typed reason to the row
factory. The existing computed-result boundary thereby prevents assignment and
clustering-report exports for unsupported runs.

- [ ] **Step 5: Update canonical consumers and tests**

Add unsupported text fields to relationship normalization and detailed-log
exclusions. Relationship modeling already filters to `ok`; leave that behavior.

- [ ] **Step 6: Run focused schema tests**

Run: `uv run pytest tests/pipeline/83_test_unsupported_result_pipeline.py tests/validation/40_test_cluster_validation_core.py tests/pipeline/52_test_method_execution_index_alignment.py -q`

Expected: all selected tests pass.

### Task 4: Report support coverage and filter validation plots

**Files:**
- Modify: `benchmarks/shared/performance_grid.py`
- Modify: `benchmarks/shared/plots/summary.py`
- Modify: `benchmarks/full/run.py`
- Modify: `tests/pipeline/80_test_benchmark_performance_grid.py`
- Create: `tests/pipeline/84_test_validation_plot_status.py`

**Interfaces:**
- Produces: per-run coverage columns, `benchmark_support_coverage_by_method_case_family.csv`, and plots containing numeric values only for `ok` rows.

- [ ] **Step 1: Write failing coverage tests**

Add unsupported and skip rows. Require `attempted_count`, `successful_count`,
`unsupported_count`, `unsupported_rate`, and `skip_count`, plus a method/category
coverage CSV. The denominator is successful plus unsupported.

- [ ] **Step 2: Write a failing plot-status test**

Pass one row of each status. Require only the ok row in plotted numeric series
and all three counts in the figure title.

- [ ] **Step 3: Run focused tests and confirm skip conflation**

Run: `uv run pytest tests/pipeline/80_test_benchmark_performance_grid.py tests/pipeline/84_test_validation_plot_status.py -q`

Expected: failures because every non-ok row is currently counted as skipped and found cluster zero is plotted.

- [ ] **Step 4: Implement coverage aggregation and artifact**

Compute:

```python
successful_count = count(status == "ok")
unsupported_count = count(status == "unsupported")
skip_count = count(status == "skip")
attempted_count = successful_count + unsupported_count
unsupported_rate = unsupported_count / attempted_count if attempted_count else np.nan
```

Write the same fields grouped by method and case category to the named CSV,
expose it from the artifacts dataclass, and link it in Markdown.

- [ ] **Step 5: Filter plots and full-run mean output**

Mask non-ok plot values to `NaN`, report status counts in the title, and render
no-success axes unavailable. In the full runner, explicitly filter mean ARI to
ok rows and print the coverage artifact path.

- [ ] **Step 6: Run reporting tests**

Run: `uv run pytest tests/pipeline/80_test_benchmark_performance_grid.py tests/pipeline/84_test_validation_plot_status.py -q`

Expected: all selected tests pass.

### Task 5: Correct case semantics and independent-noise scaling

**Files:**
- Modify: `benchmarks/shared/cases/gaussian.py`
- Modify: `benchmarks/shared/cases/dimensionality.py`
- Modify: `benchmarks/shared/cases/regression_gate.py`
- Modify: `benchmarks/shared/generators/generate_dimensional_gaussian.py`
- Modify: `benchmarks/shared/generators/case_data_contracts.py`
- Modify: active Python selectors returned by `rg -n "gauss_extreme_noise_highd" --glob '*.py'`
- Modify: `tests/core/test_benchmark_case_data_contract.py`
- Modify: `tests/core/test_dimensional_gaussian_generator.py`
- Modify: `tests/core/test_generator_geometry_audit.py`

**Interfaces:**
- Produces: `gauss_dense_signal_highd`, its continuous companion, and `gauss_sparse_signal_highd_noise`; direct normal sampling for zero-correlated noise.

- [ ] **Step 1: Write failing case/metadata tests**

Require the old ID absent, new IDs unique, dense numerical recipe unchanged,
exact dense intent/caution metadata, and exactly 12 informative plus 19,988
nuisance dimensions for the sparse case.

- [ ] **Step 2: Write the failing fast-path test**

Monkeypatch dense covariance construction to reject the 19,988-dimensional
noise block. Require deterministic `(40, 20000)` generation and nuisance data
whose distribution does not depend on truth labels within deterministic
aggregate tolerance.

- [ ] **Step 3: Run focused generator tests and confirm failure**

Run: `uv run pytest tests/core/test_benchmark_case_data_contract.py tests/core/test_dimensional_gaussian_generator.py tests/core/test_generator_geometry_audit.py -q`

Expected: failures for old IDs, absent case, absent metadata override, and dense noise covariance.

- [ ] **Step 4: Rename active cases and update selectors**

Rename the unchanged recipe and continuous selected name. Allow exact case-level
`benchmark_intent` and `scientific_caution` to override methodological defaults.
Update active Python selectors and regression gates; leave raw JSON/CSV untouched.

- [ ] **Step 5: Add sparse case and independent-noise path**

Add the exact specification parameters. For `noise_corr == 0.0`, sample:

```python
noise_block = rng.normal(0.0, config.noise_std, (cluster_size, config.noise_dims))
```

Keep the current exchangeable-covariance path for nonzero noise correlation.

- [ ] **Step 6: Run generator/geometry tests**

Run: `uv run pytest tests/core/test_benchmark_case_data_contract.py tests/core/test_dimensional_gaussian_generator.py tests/core/test_generator_geometry_audit.py -q`

Expected: all selected tests pass.

### Task 6: Update benchmark and wiki documentation

**Files:**
- Modify: `benchmarks/README.md`
- Modify: `wiki/analyses/oracle-gate-path-diagnostic.md`
- Modify: `wiki/questions/open-mathematical-questions.md`
- Modify: `wiki/index.md`
- Modify: `wiki/log.md`
- Preserve: `raw/**`

**Interfaces:**
- Produces: current operational documentation and historical-ID provenance.

- [ ] **Step 1: Document the three-state result contract**

Describe unsupported semantics, flattened fields, `NaN` quality, coverage
denominator, and the distinction from skip and clustering failure.

- [ ] **Step 2: Document current and historical case names**

Keep historical old-ID evidence but note the active unchanged recipe name. Add
the sparse case and state that geometry audit, not known labels, determines tree
recoverability.

- [ ] **Step 3: Update durable wiki navigation and chronology**

Update index coverage and append a dated `2026-08-03` implementation/verification
entry. Do not edit raw assets.

- [ ] **Step 4: Verify documentation**

Run: `make wiki-lint && git diff --check`

Expected: both exit zero.

### Task 7: Verify against every acceptance criterion

**Files:**
- Review: all changed files
- Output: ignored/generated focused benchmark artifacts

**Interfaces:**
- Produces: fresh evidence; no unverified completion claim.

- [ ] **Step 1: Run all focused tests together**

Run the union of Tasks 1–5 test files with `uv run pytest ... -q`.

Expected: zero failures.

- [ ] **Step 2: Run Ruff on changed Python surfaces**

Run: `uv run ruff check benchmarks/shared benchmarks/full tests/core tests/integration tests/pipeline tests/validation`

Expected: zero errors.

- [ ] **Step 3: Run and inspect generator geometry**

Use the existing geometry-audit CLI for both new cases. Confirm dense caution,
sparse dimensional metadata, and no imposed expected ARI.

- [ ] **Step 4: Run the continuation smoke**

Run dense case followed by another case with TBS and k-means. Inspect CSV and
coverage outputs for unsupported TBS, zero labels/clusters, `NaN` quality,
later rows, and one unsupported count.

- [ ] **Step 5: Run the full repository suite**

Run: `uv run pytest -q`

Expected: zero failures; record exact pass/skip/warning counts.

- [ ] **Step 6: Run final docs/diff verification**

Run: `make wiki-lint && git diff --check`

Expected: both exit zero.

- [ ] **Step 7: Review requirements and repository state**

Map every specification acceptance criterion to a diff and fresh command
output. Confirm this tranche added no raw-asset changes. Inspect every existing
dirty file before staging.

- [ ] **Step 8: Commit only safely isolatable changes**

Stage path-by-path and inspect `git diff --cached`. If overlapping pre-existing
hunks cannot be separated, leave them unstaged and report the exact state rather
than committing unrelated work.
