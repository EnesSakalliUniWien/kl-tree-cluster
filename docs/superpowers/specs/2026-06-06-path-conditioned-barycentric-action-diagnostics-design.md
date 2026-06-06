# Path-Conditioned Barycentric Action Diagnostics Design

## Objective

Develop diagnostics that trace the full equation path behind KL-TE split
decisions and identify likely missing equations. The diagnostics are
descriptive only. They must not promote radius, angle, KAK, cosine, or external
selected-tail variables into production calibration unless a separate support
and validation contract is met.

## Assumptions

- The local barycentric identity is exact for every binary parent:
  `theta_u = beta theta_L + (1 - beta) theta_R`.
- In the default no-branch-scaling path, edge and sibling z-vectors share the
  same whitened barycentric direction up to sign.
- Fixed chi-square projected-Wald references are valid only for fixed or
  conditionally valid projection bases, not automatically for selected trees,
  selected sibling contexts, or selected PCA/KAK bases.
- The restored KAK/cosine geometry outputs are diagnostic inputs. They explain
  traversal geometry but are not calibration evidence by themselves.
- Existing benchmark outputs and restored KAK geometry tables may be reused
  when present; the first implementation should avoid rerunning expensive full
  benchmarks.

## Non-Goals

- Do not change production edge, sibling, traversal, or calibration behavior.
- Do not add a fallback external calibration rule.
- Do not use KAK or cosine decomposition as a production split basis.
- Do not tune method constants or support thresholds.
- Do not create a broad new abstraction for diagnostics.

## Equation Trace

For each binary parent `u` with children `L` and `R`, the diagnostic records:

```text
beta = n_L / (n_L + n_R)
delta = theta_L - theta_R
theta_L - theta_u = (1 - beta) delta
theta_R - theta_u = -beta delta
```

It then verifies the implemented no-branch-scaling edge/sibling identity:

```text
z_edge_left ~= z_sibling
z_edge_right ~= -z_sibling
```

The selected law being diagnosed is:

```text
Law(T_u | tree selected, edge path opened, projection selected, u traversed)
```

not the unconditional fixed-subspace reference:

```text
T_u ~ chi_square(k_u)
```

## Diagnostic 1: Barycentric Identity Trace

Purpose: verify that every analyzable parent obeys the exact barycentric
contrast algebra and quantify numerical residuals.

Inputs:

- Existing tree node distributions and leaf counts from the KL-TE gate or
  decomposition path.
- Existing edge and sibling projected-Wald records when available.

Outputs:

- One row per binary parent.
- Parent id, child ids, sizes, `beta`, sibling projection dimension, edge
  projection dimensions.
- Raw barycentric residual norms.
- Edge/sibling z-identity residual norms when comparable vectors are available.
- Status values: `ok`, `missing_edge_context`, `incompatible_projection`,
  `unsupported_feature_space`.

Success criteria:

- Residuals should be near numerical precision in supported default paths.
- Any failure must be reported as diagnostic status, not repaired silently.

## Diagnostic 2: Action Budget Trace

Purpose: test whether traversal fragmentation is better explained by a missing
inertia/action budget than by sibling significance alone.

For parent coordinates in a declared basis `B`, record the parallel-axis split
energy:

```text
split_action_B = n_u * beta * (1 - beta) * ||P_B(theta_L - theta_R)||^2
parent_inertia_B = sum_{i in L(u)} ||P_B(X_i - theta_u)||^2
action_fraction_B = split_action_B / parent_inertia_B
```

Inputs:

- Current feature-space coordinates for ordinary benchmark cases.
- Restored KAK/cosine block coordinates when analyzing KAK outputs.
- Existing assignment labels only for post-hoc fragmentation labels.

Outputs:

- One row per internal parent.
- `action_fraction_B`, sibling statistic, edge action, split balance, parent
  radius, child radii, and parent size.
- Optional post-hoc labels: pure fragment, pass-through fragment, mixed split.

Success criteria:

- The diagnostic should show whether high sibling evidence with low remaining
  parent inertia predicts over-fragmentation.
- Labels derived from assignments must be explicitly marked post-hoc.

## Diagnostic 3: KAK Radius/Angle Traversal Trace

Purpose: explain the `radius_angle_action` median AUC result by tracing the
actual geometry variables that predict traversal fragmentation.

Inputs:

- `kak_signal_adaptive_internal_tree_geometry.csv`.
- `kak_internal_traversal_fragmentation_validation.csv`.

Recorded variables:

```text
parent_centroid_norm
parent_centroid_angle_to_leading_axis_deg
parent_centroid_independent_fraction
sibling_separation_to_parent_radius_ratio
sibling_separation_to_child_radius_ratio
abs_sibling_common_axis_mean_delta
balance_fraction
parent_size
```

Outputs:

- A compact summary table comparing `size_balance_only`, `radius_action`, and
  `radius_angle_action`.
- A per-block table showing which variables improve leave-one-block-out AUC.
- A short markdown report explaining that the AUC is traversal diagnostic
  evidence, not calibration evidence.

Success criteria:

- Reproduce the existing median AUC `0.892810` from cached outputs when those
  files exist.
- If the exact label generator is not available, report the label provenance
  as cached/opaque rather than reverse-engineering it incorrectly.

## Diagnostic 4: Missing Equation Candidate Panel

Purpose: rank probable missing equation paths without promoting any of them.

Candidate paths:

- selected-region path: merge inequalities, margins, tangent cone, tie-cell
  status;
- edge-opening path: weakest child edge action and radial boundary distance;
- projection-selection path: selected PCA dimension, MP floor, selected basis
  alignment;
- action-budget path: split action divided by remaining parent inertia;
- angular-shell path: KAK radius, angle, independent-radius fraction, and
  common-axis gap;
- traversal-survival path: parent reached and split after upstream openings.

Outputs:

- A ranked CSV and markdown report with each candidate's evidence role:
  `exact_identity`, `diagnostic_association`, `support_gap`,
  `validation_required`, or `rejected_for_now`.

Success criteria:

- The report should identify which missing paths are mathematically exact,
  which are only predictive diagnostics, and which remain unsupported.

## Data Flow

1. Read cached benchmark/KAK outputs if present.
2. Compute lightweight derived tables in a new diagnostics output directory.
3. Write CSV summaries plus one markdown trace report.
4. Do not mutate benchmark result inputs.
5. Do not wire results into production calibration.

## Benchmark And Recursive Analysis Phase

After the diagnostics pass its focused tests, run the benchmark suite with the
new trace outputs enabled as analysis artifacts. The benchmark phase should
answer a narrower question than the leaderboard: which missing equation
candidate best explains observed skips, under-splits, and fragmentation?

Required outputs:

- Benchmark result directory using the existing benchmark runner.
- Joined diagnostic table keyed by case, method, parent context, and available
  trace status.
- Recursive analysis markdown report that revisits each candidate path after
  seeing benchmark evidence.
- Updated candidate ranking with one of: `promote_to_validation_panel`,
  `keep_diagnostic`, `needs_new_data`, or `deprioritize`.

The recursive analysis should be bounded to two passes:

1. First pass: run diagnostics and benchmark joins.
2. Second pass: identify the strongest remaining blocker and write the next
   smallest validation question.

Further recursion requires a new explicit request so the analysis does not
turn into open-ended benchmark fishing.

## Proposed Files

- Add one diagnostic script under `benchmarks/diagnostics/math_trace/`.
- Add focused tests under `tests/validation/`.
- Optionally add one wiki source summary after the diagnostic is run.

## Testing

- Unit-test barycentric residual calculations on a small hand-built binary
  tree.
- Unit-test action-budget formulas against the parallel-axis identity.
- Smoke-test the KAK cached-output reader against existing CSV columns.
- Verify the diagnostic report writes status values instead of failing on
  missing optional inputs.

## Risks

- Post-hoc fragmentation labels can be mistaken for calibration truth. The
  report must label them explicitly as post-hoc diagnostics.
- KAK/cosine variables may overfit one restored matrix analysis. The
  diagnostic should separate cached reproduction from general validation.
- Projection-selection and selected-region objects are not solved by this
  diagnostic. They should remain listed as missing equation paths unless
  future validation closes them.

## Approval Gate

Implementation may start only after this spec is reviewed. The first
implementation should be small: cached-output KAK trace plus reusable
parallel-axis/action-budget helpers, with no production method changes.
