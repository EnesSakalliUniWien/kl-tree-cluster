# Unsupported Benchmark Outcome and Gaussian Case Taxonomy Design

## Objective

Make benchmark outputs distinguish a valid clustering result from a method run
whose scientific calibration contract is unavailable. The first affected path
is Tree-Break Selection (TBS) with the empirical-null projected-Wald sibling
gate. When the selected hierarchy contains no admissible internal calibration
support, the runner must return an explicit `unsupported` outcome instead of
turning the fail-closed traversal boundary into one cluster with ARI zero.

The same tranche corrects the misleading high-dimensional Gaussian benchmark
taxonomy. The current `gauss_extreme_noise_highd` recipe gives every coordinate
a cluster-dependent center and is therefore dense high-dimensional signal, not
an irrelevant-feature experiment. Its recipe remains available under an
accurate name, while a separate case uses the existing dimensional Gaussian
model to combine a fixed informative subspace with independent nuisance
dimensions.

## Confirmed Failure Being Addressed

For `gauss_extreme_noise_highd`, the current binary recipe has 40 samples and
20,000 median-binarized `make_blobs` coordinates. The Hamming/average-linkage
tree contains a perfect four-cluster cut. At 100 or more features in a controlled
dimensional sweep, however, all 78 child-parent edges reject. This leaves zero
admissible sibling empirical-null records, marks all 39 sibling tests
`undefined_no_internal_support`, and closes every sibling gate. Traversal then
returns the root as a boundary and the benchmark records one cluster with ARI
zero.

Changing only the diagnostic sibling gate to `fixed_coordinate_bh` returns
four clusters with ARI 1.0. This establishes that the immediate failure is the
calibration-support contract rather than Hamming geometry, tree topology,
diffusion, NNLS branch lengths, or traversal implementation. The diagnostic
gate is not a valid production fallback because its coordinates are evaluated
after same-data hierarchy selection.

## Existing Environment and Integration Constraints

The implementation must extend the current repository environment rather than
create parallel abstractions:

- `benchmarks.shared.types.MethodRunResult` is the method-run boundary.
- `benchmarks.shared.result_records.BenchmarkRunStatus` and
  `BenchmarkResultRow` are the typed exported-row boundary.
- `benchmarks.shared.runners.dispatch` normalizes method results.
- Existing metric, assignment, plotting, and report exporters consume the
  normalized result rows.
- `benchmarks/diagnostics/generators/case_geometry_audit.py` already audits
  source representation, duplicate geometry, within/between distances, and
  nearest-neighbor label purity.
- The in-progress `tbs-audit --mode calibration-contract` work already provides
  detailed tested-null, stopped-frontier, and nested-blocked record evidence.
  Runtime status must consume production-stamped support, not duplicate or
  independently rederive this audit.
- The current worktree contains unrelated and overlapping changes. An
  implementation must inspect each overlapping diff, preserve unrelated work,
  and stage only files belonging to this tranche.

There will be one authoritative status enum and one authoritative unsupported
reason model. No aliases, legacy case identifiers, fallback result paths, or
second benchmark dispatcher will be introduced.

## Outcome Model

### Run Status

The shared run-status enum has exactly three values:

```text
ok
skip
unsupported
```

`ok` means that the method produced its declared output under an available
scientific contract. `skip` means that the method was not attempted or could
not be executed for an operational reason. `unsupported` means that execution
reached a recognized inferential boundary and established that the method's
scientific support contract was unavailable for this run.

Define `BenchmarkRunStatus` in
`benchmarks/shared/types/run_status.py`, delete its definition from the result
row module, and import the shared enum in both `MethodRunResult` and
`BenchmarkResultRow`. Because it remains a string-valued enum, existing
comparisons and serialization remain direct. There is no compatibility alias
for the old location; all repository imports move to the authoritative module.

### Typed Unsupported Reason

Add an immutable shared model with the following conceptual shape:

```text
UnsupportedReason
  code: str
  stage: str
  message: str
  evidence: UnsupportedEvidence

UnsupportedEvidence
  focal_record_count: int | None
  admissible_support_count: int | None
  invalid_record_count: int | None
  upstream_tested_count: int | None
  upstream_rejected_count: int | None
```

The initial reason has:

```text
code = empirical_null_no_internal_support
stage = sibling_calibration
```

The message explains that the selected hierarchy contains focal sibling tests
but no admissible internal empirical-null support. The evidence counts describe
the recognized boundary without carrying node arrays, DataFrames, or arbitrary
diagnostic dictionaries into the stable result schema.

Define reason codes in a string-valued `UnsupportedReasonCode` enum beside the
reason model. The generic models allow later methods to use `unsupported`, but
only enum members may populate exported results. Each reason code defines which
evidence fields are required. Unknown codes or missing required evidence are
contract errors.

### Result Invariants

Central validation enforces mutually exclusive result states:

- `ok`: labels are present; `skip_reason` and `unsupported_reason` are absent;
  `found_clusters` agrees with the number of non-noise labels.
- `skip`: labels are absent; `found_clusters == 0`; `skip_reason` is present;
  `unsupported_reason` is absent.
- `unsupported`: labels are absent; `found_clusters == 0`; `skip_reason` is
  absent; `unsupported_reason` is present and valid.

The generic `ok` state permits zero non-noise clusters because density methods
may validly label every observation as noise. A successful TBS decomposition
still requires at least one cluster. This distinction preserves existing method
semantics while preventing unsupported TBS output from masquerading as a valid
zero- or one-cluster result.

## TBS Runner Decision Boundary

The TBS runner owns the `unsupported` decision. The benchmark dispatcher and
post-processing layers must not infer it from cluster count, ARI, raw p-values,
or method names.

The runner proceeds through tree construction, optional branch-length fitting,
node-distribution population, and gate annotation. Immediately after gate
annotation it reads the production-stamped sibling calibration state. If the
active empirical-null gate reports focal records with
`undefined_no_internal_support`, the runner:

1. counts focal sibling records;
2. counts role-supported calibration records;
3. counts invalid sibling records;
4. counts tested child-parent edges;
5. counts rejected child-parent edges;
6. constructs the registered `UnsupportedReason`;
7. returns `status=unsupported`, no labels, and `found_clusters=0` before tree
   traversal or clustering-report construction.

The runner does not recompute whether a sibling record is null-like,
edge-blocked, or role-supported. Those roles are determined and stamped by the
gate pipeline. The separate calibration-contract audit remains responsible for
detailed structural support analysis.

Annotations, stage timings, tree-build diagnostics, and branch-length
optimization diagnostics remain available under `extra` when diagnostic
artifacts are requested. They are not substitutes for the typed unsupported
reason.

This boundary applies only to recognized scientific states. Unexpected
exceptions, inconsistent annotations, missing columns required by the active
gate, and unknown reason codes propagate as errors in strict execution. They do
not become `skip` or `unsupported` fallbacks.

## Dispatch, Metrics, and Export Data Flow

The dispatcher validates and propagates the runner outcome without inspecting
TBS internals. Normalized processing follows this sequence:

1. Validate the `MethodRunResult` state invariants.
2. For `ok`, align labels to the input index and calculate metrics normally.
3. For `skip`, write the operational skip row and continue the sweep.
4. For `unsupported`, bypass label alignment and metric computation, write the
   unsupported row, and continue the sweep.

An unsupported result row has:

```text
found_clusters = 0
labels_length = 0
all clustering-quality metrics = NaN
```

It also adds these stable columns:

```text
unsupported_reason_code
unsupported_stage
unsupported_reason
unsupported_focal_record_count
unsupported_admissible_support_count
unsupported_invalid_record_count
unsupported_upstream_tested_count
unsupported_upstream_rejected_count
```

For non-unsupported rows, unsupported text fields are empty and unsupported
numeric evidence fields are `NaN`. `skip_reason` remains exclusive to `skip`.

Assignment tables and clustering reports are not written for unsupported runs.
Diagnostic annotation or timing artifacts may still be written by explicit
audit/diagnostic paths. The general benchmark exporter must not create empty
assignment files that look like valid clusterings.

## Aggregation and Plotting Contract

Quality statistics use only `status=ok` rows. Unsupported rows are not converted
to zero scores and are not silently dropped from coverage reporting.

Every method summary and case-family summary reports:

```text
attempted_count
successful_count
unsupported_count
unsupported_rate
skip_count
```

`successful_count` and `unsupported_count` are scientifically attempted runs.
Therefore `attempted_count = successful_count + unsupported_count`.
`skip_count` reports scheduled runs that did not reach a scientific outcome and
is not part of `attempted_count`.

The denominator is:

```text
unsupported_rate = unsupported_count / (successful_count + unsupported_count)
```

If there are no successful or unsupported runs, `unsupported_rate` is `NaN`.
Operational skips are shown separately and do not change the scientific-support
denominator.

Plots and leaderboard tables must make support coverage visible alongside
quality. Mean ARI, NMI, and related metrics cannot rank a method without also
showing how many attempted scientific runs were unsupported. Pure plotting
code may choose layout behavior for an empty quality subset, but it must not
invent scores or change statuses.

## Gaussian Case Taxonomy

### Dense-Signal Stress Case

Remove the active case identifier `gauss_extreme_noise_highd`. There is no alias.
Retain the existing all-informative `make_blobs` recipe under:

```text
gauss_dense_signal_highd
```

Its current continuous representation becomes:

```text
gauss_dense_signal_highd_continuous
```

Both cases declare these exact metadata values:

```text
benchmark_intent = dense_high_dimensional_signal_and_calibration_saturation
scientific_caution = all_coordinates_are_cluster_dependent_not_irrelevant_noise
```

The current 40 samples, 20,000 features, four clusters, standard deviation 7.5,
and seed 43 remain unchanged so the calibration-saturation stress remains
reproducible.

### Sparse-Signal High-Dimensional Noise Case

Add the binary case `gauss_sparse_signal_highd_noise` to the dimensionality
case family. It reuses the existing `dimensional_gaussian` generator with:

```text
n_samples = 40
n_clusters = 4
informative_dims = 12
n_features = 20_000
noise_dims = 19_988
separation = 2.8
informative_std = 1.0
noise_std = 1.0
informative_corr = 0.0
noise_corr = 0.0
signal_mode = consolidated
balanced_clusters = true
seed = 43
observation = per-coordinate median binarization
```

The case is an irrelevant-feature robustness stress, not a promised-success
case. Its geometry audit determines whether the generated Hamming hierarchy is
tree-recoverable. Known labels do not imply an expected ARI, and a
tree-unrecoverable case must not drive gate changes.

### Independent-Noise Fast Path

The existing dimensional Gaussian generator constructs dense covariance
matrices. A 19,988-dimensional independent-noise covariance would require
roughly 3.2 GB before sampling. For exactly zero correlation, generate the
block directly with independent normal draws of the configured standard
deviation. This is mathematically identical to sampling from the diagonal
Gaussian covariance and preserves the existing public generator contract.

The informative block continues to use its current covariance implementation.
Nonzero exchangeable noise correlation remains on the existing path. No new
generator or duplicate case-data dispatcher is added.

### References and Historical Evidence

Current source selectors, regression gates, diagnostics, and tests move to the
new dense-signal ID. The old ID disappears from active case registration.
Historical raw benchmark assets are immutable and retain the ID under which
they were generated. Wiki pages citing those assets retain the historical ID
and add a concise note mapping it to the current dense-signal case where needed.
This records provenance without maintaining a runtime alias or rewriting raw
evidence.

## Testing Design

### Outcome Contract Tests

- Accept valid `ok`, `skip`, and `unsupported` states.
- Reject unsupported results with labels, nonzero clusters, skip reasons,
  unknown reason codes, or missing required evidence.
- Reject skipped results with unsupported reasons.
- Preserve valid all-noise density-method `ok` results.
- Require successful TBS results to contain at least one cluster.

### TBS and Dispatch Tests

- Use a small deterministic gate-annotation fixture to prove that zero
  admissible sibling support returns `unsupported` before traversal.
- Assert the initial reason code, stage, message, and all five evidence counts.
- Prove fixed-coordinate and other non-empirical gates do not trigger this
  reason merely because they lack empirical-null records.
- Prove dispatch propagates unsupported without label normalization or metric
  calculation.
- Prove a later method and later case still execute after an unsupported row.
- Prove unexpected runner errors still propagate in strict execution.

### Result, Aggregation, and Export Tests

- Assert unsupported rows contain zero clusters, zero labels, and `NaN` quality
  metrics.
- Assert unsupported fields are empty/`NaN` for `ok` and `skip` rows.
- Assert quality means use only successful rows.
- Assert unsupported counts and rates are grouped correctly by method and case
  family.
- Assert skips remain outside the unsupported-rate denominator.
- Assert no assignment or clustering-report artifact is written for an
  unsupported run.
- Assert plotting/report code records unsupported coverage and does not replace
  missing metrics with zero.

### Generator and Geometry Tests

- Assert the dense case preserves the current deterministic recipe and records
  its all-informative scientific caution.
- Assert the sparse case records exactly 12 informative and 19,988 nuisance
  dimensions.
- Assert independent nuisance dimensions have the same distribution across
  truth labels, within deterministic statistical tolerances.
- Assert the zero-correlation path does not call dense covariance construction
  for the noise block.
- Assert both cases are deterministic for their declared seeds.
- Run the geometry audit and record its tree-signal assessment without hard
  coding an expected clustering score.
- Assert the removed case ID is absent from active case registration and the
  new IDs are unique.

## Verification Before Completion

Fresh evidence is required before a completion or commit claim. The
implementation verification sequence is:

1. Run focused outcome-model, generator, dispatcher, aggregation, export, and
   plotting tests.
2. Run Ruff on every changed Python surface.
3. Run the focused generator-geometry audit and inspect the dense and sparse
   rows.
4. Run a focused benchmark smoke containing the dense TBS run plus subsequent
   methods/cases; inspect the result CSV for status, `NaN` metrics, evidence,
   coverage counts, and continuation.
5. Run the full repository test suite and report exact pass/fail counts.
6. Run `make wiki-lint` after wiki changes.
7. Run `git diff --check`.
8. Review the approved requirements line by line against the final diff.

A partial test run cannot support a full-suite claim. A passing linter cannot
support a runtime or scientific claim. If any command fails, report the actual
failure and do not describe the tranche as complete.

## Expected File Boundaries

The implementation plan should prefer these existing surfaces:

- shared status and reason models under `benchmarks/shared/types/`;
- existing method and result-row models under `benchmarks/shared/types/` and
  `benchmarks/shared/result_records/`;
- TBS decision logic in `benchmarks/shared/runners/tbs_runner.py`;
- result propagation in the existing dispatcher and benchmark execution path;
- existing metric, summary, plot, and export modules rather than new report
  engines;
- Gaussian and dimensional case definitions under
  `benchmarks/shared/cases/`;
- the existing dimensional Gaussian generator and geometry audit;
- focused tests in the repository's existing core, pipeline, integration, and
  validation test categories.

Exact files for aggregation and export changes are resolved during the
implementation plan by following the current result-row consumers. No broad
directory reorganization or unrelated refactor belongs in this tranche.

## Non-Goals

- Do not promote `fixed_coordinate_bh` or another diagnostic gate to the
  empirical-null production default.
- Do not add cross-fitting, feature splitting, or sample splitting.
- Do not change edge or sibling alpha values.
- Do not change tree construction, diffusion embeddings, or NNLS branch-length
  fitting.
- Do not reuse selected non-null observations as null calibration data.
- Do not add an external-calibration fallback.
- Do not rewrite historical raw result assets.
- Do not retain the removed case ID as an alias.
- Do not use unsupported rows as zero-quality observations.
- Do not combine this contract change with unrelated repository cleanup.

## Risks and Mitigations

- **Schema ripple:** a new status affects runners, rows, plots, and exports.
  Central validation and focused consumer tests prevent partial propagation.
- **Coverage inflation:** excluding unsupported rows could make mean quality look
  better. Mandatory unsupported counts and rates keep coverage visible.
- **Status abuse:** methods could use unsupported to hide poor clusterings.
  Registered reason codes, runner-owned decisions, and required evidence limit
  it to recognized scientific contracts.
- **Generator memory:** a dense independent-noise covariance is infeasible at
  20,000 dimensions. The zero-correlation direct-sampling path removes that
  allocation without changing the distribution.
- **Benchmark impossibility:** 12 informative dimensions among 19,988 nuisance
  dimensions may destroy Hamming tree signal. The geometry audit records this
  as part of the experiment instead of forcing a success expectation.
- **Historical ambiguity:** raw assets retain the original case ID. Wiki notes
  distinguish historical provenance from current active names without a
  runtime alias.
- **Dirty worktree collision:** implementation files already contain user work.
  Inspect and patch overlapping diffs, stage path-by-path, and never reset or
  overwrite unrelated changes.

## Acceptance Criteria

The tranche is acceptable only when all of the following are evidenced:

- A zero-support empirical-null TBS run is exported as `unsupported`, not `ok`,
  `skip`, or one cluster.
- Its labels are absent, cluster count is zero, quality metrics are `NaN`, and
  its typed reason contains the required numeric evidence.
- The benchmark continues to later runs.
- Aggregate quality excludes unsupported rows while support coverage reports
  them by method and case family.
- No assignment or clustering report is emitted for unsupported output.
- The dense case is accurately named and documented, with no active old-ID
  alias.
- The sparse high-dimensional case uses a fixed informative block and truly
  label-independent nuisance dimensions from the existing generator.
- The independent-noise fast path avoids the dense covariance allocation.
- Historical raw evidence remains unchanged.
- Focused tests, full tests, lint, geometry audit, benchmark smoke, wiki lint,
  and diff hygiene have fresh recorded outcomes.

## Sequence After This Tranche

After this specification is planned, implemented, and verified, a separate
brainstorming/specification cycle will address a no-cross-fitting
selected-hierarchy calibration law. That work must define the selected
conditional target, simulator or analytic law, context variables, tail
precision, support thresholds, and admissibility criteria before any production
integration. The unsupported outcome created here remains necessary even if a
future calibration model supports more cases.

## Approval Gate

This document specifies the first benchmark-honesty tranche only. After the
user reviews the committed specification, the next skill is `writing-plans`.
No implementation begins before that review and plan.
