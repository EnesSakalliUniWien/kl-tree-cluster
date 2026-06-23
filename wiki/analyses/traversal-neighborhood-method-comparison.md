---
title: Traversal Neighborhood Method Comparison
type: analysis
status: reviewed
updated: 2026-06-15
sources:
  - tree_break_selection/hierarchy_analysis/tree_decomposition.py
  - tree_break_selection/hierarchy_analysis/decomposition/gates/gate_evaluator.py
  - tree_break_selection/hierarchy_analysis/decomposition/gates/orchestrator.py
  - tree_break_selection/hierarchy_analysis/statistics/sibling_divergence/pair_testing/collection/child_parent_edge_metadata.py
  - tree_break_selection/hierarchy_analysis/statistics/sibling_divergence/pair_testing/collection/record_collection.py
  - tree_break_selection/hierarchy_analysis/statistics/sibling_divergence/inflation_correction/empirical_null_inflation_estimation.py
  - wiki/sources/old-vs-current-method-stack-comparison-20260615.md
  - wiki/sources/overlap-conditional-topology-law-panel-20260615.md
  - wiki/sources/specific-small-method-benchmark-20260615.md
  - wiki/sources/selected-neighborhood-distribution-panel-20260615.md
  - wiki/sources/retained-pass-through-topology-likelihood-panel-20260615.md
  - wiki/sources/overlap-selected-pass-through-fixture-miner-20260615.md
  - wiki/sources/selected-pass-through-branch-recovery-conditioning-20260615.md
tags:
  - analysis
  - traversal
  - neighborhood
  - calibration
---

# Traversal Neighborhood Method Comparison

## Summary

The previous and current Tree-Break Selection method stacks share the same basic top-down
traversal skeleton. They differ mainly in how the sibling gate is produced and
calibrated. The old stack used richer tree-neighborhood bandwidth ideas around
stopping and nearby stable/signal nodes, but with permissive support handling.
The current stack uses stricter support and fixed-coordinate diagnostics, but
its default internal calibration neighborhood is narrower.

The correct next object is therefore not a new global traversal rule. It is a
directed selected-neighborhood distribution law for internal recovery that
keeps the old tree-neighborhood insight while preserving the current
fail-closed support contract.

## Details

The traversal decision is:

\[
\operatorname{split}(u)
=
\operatorname{binary}(u)
\land
\operatorname{edge\_open}(u)
\land
\operatorname{sibling\_open}(u).
\]

`binary(u)` means the node has exactly two children. `edge_open(u)` means at
least one child-parent edge is significant. `sibling_open(u)` means the sibling
BH gate is open and the sibling test was not skipped.

If pass-through is disabled, every failed split becomes a final boundary. If
pass-through is enabled, a node with binary plus edge evidence but closed
sibling evidence can still be traversed when a descendant can fully split. In
current code this is represented by `TraversalDecision.PASS_THROUGH`, and it
continues to the children exactly like a split while marking the current node
as not being a final boundary.

This means neighborhood information historically affected traversal
indirectly. It changed the sibling decision, which then changed whether a node
was a split, boundary, or pass-through ancestor. It was not an explicit
directed recovery law at the traversal evaluator boundary.

The current strict empirical-null inflation path builds sibling records from
child-parent edge annotations. It marks sibling pairs as null-like when neither
child edge is significant, edge-blocked when testing is stopped or
ancestor-blocked, and assigns a sibling-null weight as the product of the two
child edge BH p-values or inherited stopped-edge weights. Calibration then
uses only strict-null or edge-blocked support records. Its local kernel uses
log sibling projection dimension and log parent sample size.

The older c2ef neighborhood idea was broader. It carried topology-aware
bandwidths such as stopping-edge distance, nearest stable/signal tree
distance, projection-scale spread, and adaptive local bandwidth. This captured
real structural context that the current strict kernel omits. The failure was
not that the variables were irrelevant; it was that they were computationally
pathological on full Julia without cached tree distances and not tied to a
strict support contract.

In the old sibling-null-prior interpolation, a blocked child \(c\) was
conditioned on four local bandwidths:

\[
\tau_b=\operatorname{median}_c b(c), \quad
\tau_t=\operatorname{median}_c \min_{t\in T} d(c,t), \quad
\tau_s=\operatorname{median}_c \min_{s\in S} d(c,s), \quad
h_k=\operatorname{sd}_{t\in T}\log k_t .
\]

Here \(b(c)\) is distance to the stopping edge, \(T\) is the set of stable
tested edges, \(S\) is the set of significant signal edges, \(d(\cdot,\cdot)\)
is undirected tree distance, and \(k_t\) is the edge projection dimension or
edge Wald degrees of freedom. The old child prior was then approximately:

\[
w_b(c)=\exp\{-\max(b(c)-1,0)/\tau_b\},
\]

\[
w_T(c,t)=
\exp\{-d(c,t)/\tau_t\}
\exp\left\{-\frac{1}{2}\left(\frac{\log k_t-\log k_c}{h_k}\right)^2\right\},
\]

\[
\bar p_T(c)=
\frac{\sum_{t\in T} w_T(c,t)p_t}{\sum_{t\in T}w_T(c,t)},
\quad
\tilde p(c)=
\frac{w_b(c)p_{\operatorname{stop}}(c)+\sum_{t\in T}w_T(c,t)p_t}
{w_b(c)+\sum_{t\in T}w_T(c,t)},
\]

\[
\rho_S(c)=\max_{s\in S}(1-p_s)\exp\{-d(c,s)/\tau_s\},
\quad
\pi_0(c)=\operatorname{clip}\{\tilde p(c)(1-\rho_S(c)),0,1\}.
\]

For a sibling pair, the old stack used the more signal-permissive child as the
pair-level bottleneck, effectively the smaller child null prior. This is why
the old method could look stronger: it could lower a blocked sibling-null prior
near signal neighborhoods, without first proving that the selected family was a
valid null-conditioning stratum.

The current empirical-null inflation law is a different object. For supported
strict-null or stopped/edge-blocked sibling records \(i\), it uses
\(x_i=(\log k_i,\log n_i)\) and a Gaussian local kernel in that context:

\[
K_i(u)=w_i
\exp\left\{-\frac{1}{2}\left\|
\frac{x_i-x_u}{\sigma}
\right\|^2\right\},
\quad
\hat c(u)=
\frac{\sum_i K_i(u)T_i}{\sum_i K_i(u)r_i\nu_i}.
\]

This is safer but less structural: selected non-null records are counted as
excluded support rather than used as calibration. The conditional-topology
diagnostic reintroduces the old tree-neighborhood variables as
`topology_neighborhood_log_component`, but as a support-gated diagnostic term.
It uses cached all-pairs tree distances and reports `tau_b`, `tau_t`, `tau_s`,
`h_k`, nearest stable distance, nearest signal distance, support count, signal
count, and selected-nonnull exclusions. It does not restore the old permissive
prior update.

The current conditional-topology diagnostics reintroduce that missing
structure safely. They expose directed incidence, incoming branch balance,
outgoing balance, outgoing edge-norm balance, fragment risk, neighborhood
scale, topology-neighborhood support, and guarded recovery fields. The guarded
recovery rule is intentionally diagnostic:

```text
recover_internal_split =
  root/null guards pass
  AND support sufficient
  AND balance_product/outgoing_edge evidence high
```

The focused multi-positive fixture supports this rule, but the real
context-negative overlap slice does not yet have enough truth support. The
only real truth row has coherent `balance_product` and outgoing edge-norm
evidence, but remains `support_insufficient_fail_closed`.

The two-method understanding is therefore:

```text
old stack:
  traversal skeleton:
    binary + edge + sibling, with pass-through
  neighborhood role:
    broad tree-neighborhood bandwidths modify sibling calibration
  strength:
    captures stopping/stable/signal neighborhood geometry
  weakness:
    permissive selected-nonnull support and expensive tree-distance loops

current stack:
  traversal skeleton:
    same binary + edge + sibling, with pass-through
  neighborhood role:
    strict support over log(k) and log(parent size), plus diagnostic topology rows
  strength:
    fail-closed support and fixed-coordinate non-cross-fit sibling gates
  weakness:
    insufficient explicit distribution law for internal selected neighborhoods
```

The candidate-level stop-rule comparison gives the sharpest current form of
this distinction. The pattern
`left_pass_through_downstream_split_right_stops` appears when the
conditional-topology profile keeps walking through an edge-open but locally
sibling-closed candidate and reaches downstream accepted splits, while the
refined global pass-through profile stops or guard-blocks the same candidate.
This pattern appears in both selected-null conservative suppressions and in
the `overlap_unbal_4c_small` possible signal over-suppression bucket. Therefore
the disagreement is not whether pass-through can expose downstream signal; it
can, and does, under both null-like and signal-like settings. The disagreement
is whether the selected pass-through neighborhood carries enough conditional
evidence to retain that walk without reopening selected-null false positives.

The retention-evidence summary shows the asymmetry that still has to be
explained. For `left_pass_through_downstream_split_right_stops`, selected-null
rows have one finite `balance_product` value, while the signal rows are all
`traversal_only` and have no finite `balance_product` or outgoing-edge topology
coverage. The current method therefore fails closed not because it lacks local
p-value evidence, but because the structural topology evidence needed to
condition the selected pass-through law is missing exactly on the deep signal
walks.

The retention-gap summary turns this into the next method contract. The refined
profile should stay closed on selected-null rows until a null-side
selected-pass-through false-positive law exists. The conditional profile should
not be promoted on signal rows until a signal-side topology likelihood exists
for retained pass-through walks. This is now the minimal difference between the
two methods: same traversal skeleton, different treatment of selected
pass-through neighborhoods under missing topology evidence.

The method-contract summary encodes that directly. `base_traversal_skeleton`
is shared by both methods. The two non-shared components are
`selected_null_pass_through_control`, where the refined profile is still the
required fail-closed behavior, and `signal_pass_through_retention`, where the
conditional profile may be recovering signal but remains unvalidated because
the topology likelihood is missing.

The profile-config contract checks this against the actual runtime profiles.
Both methods use `fixed_coordinate_bh`, sibling alpha penalty `50`, and the
same selected-root stability settings. The concrete configuration difference is
that `fixed_coordinate_global_passthrough_refined_v1` enables a selected-family
global sibling-min pass-through guard with `99` draws, alpha `0.01`, and scope
`global_sibling_min_passthrough_descendant_refined`, while
`fixed_coordinate_conditional_topology_diagnostic_v1` leaves that guard off and
emits topology-law evidence diagnostically.

The readiness summary gives the working verdict. Both profiles are ready only
as diagnostics for the shared traversal skeleton. The refined profile is ready
as a fail-closed guard candidate for selected-null pass-through control. The
conditional topology profile is not ready for production signal pass-through
retention because the signal-side topology likelihood is still missing.

The retained-pass-through likelihood panel checks that missing object directly.
It conditions on the shared selected event
`left_pass_through_downstream_split_right_stops`, matches signal rows to
selected-null controls by depth and descendant mass, and asks whether finite
topology evidence exists on both sides. In the compact run, `5/7` signal rows
can be matched, but signal retained-pass-through rows have no finite topology
features and matched controls also lack finite topology features. Thus the
likelihood is not identifiable, and the fail-closed readiness verdict is
empirically supported.

The branch-recovery conditioning panel adds the next guard against a false
interpretation of the old bandwidth method. Generated support shows that
feature-subspace geometry can identify clean retained pass-through branch
recovery when such rows are present. But the targeted real traversal-selected
search over clean binary four-cluster cases and `overlap_part_4c_small` mined
`12` selected pass-through event rows with `0` full or partial branch
recoveries: all `7` signal rows are false fragments and the other `5` rows are
selected-null controls. The old bandwidth lesson is therefore not that
selected-event walks with low balance should be retained. It is that tree
neighborhood should enter as context, while branch recovery needs a
feature-subspace likelihood and real traversal-selected branch-positive
support.

The traversal-network extension keeps that verdict but sharpens the failure
mode. The diagnostic now also records the directed distance to pass-through
context and the directed distance to the next downstream accepted split. In the
compact run, all `7` signal retained-pass-through rows and all `5`
selected-null controls have finite values for both distances, and all `5`
matched rows are traversal-neighborhood matched. Exact tree-network distance
is still unavailable, because the matches compare signal rows to selected-null
controls from separately selected trees. Therefore the current blocker is not
missing traversal context; it is the missing topology likelihood on the
signal-side retained walks and the absence of same-tree selected-null network
controls.

The selected pass-through fixture miner makes the first half of that blocker
operational. It extracts `12` rows in the selected event stratum, with `7`
signal candidates and `5` selected-null controls. All rows have finite directed
traversal context, but direct topology rows remain sparse: `0` finite rows on
the signal side and `1` on the selected-null side. The miner therefore adds a
structural fallback based on selected-tree descendant masses, with incoming
node-versus-sibling balance, outgoing child balance, and their product. This
completed topology support is observed for `7/7` signal candidates and `3/5`
selected-null controls. The completed balance product separates the compact
finite selected-event rows in the low direction, retaining `7/7` signal
candidates and `0` finite selected-null controls at threshold `0.02594`. Thus
the next clean enhancement was to expand selected pass-through fixture support.
The expanded seven-case, five-replicate overlap run mines `50` selected-event
rows and shows the compact separator does not transfer: completed topology
support is observed for `17/19` signal candidates and `20/31` selected-null
controls, but the low-direction separator overlaps `10` finite selected-null
controls. The fallback is therefore a useful conditioning variable, not a
standalone retention law.

The truth-labeled expanded rerun adds the missing overlap interpretation. The
`19` signal selected-event rows contain `0` full branch recoveries, `0` partial
branch recoveries, `1` barycentric mixture candidate, `14` false fragments, and
`4` unresolved signal candidates. The barycentric row
`overlap_unbal_4c_small` replicate `4` node `N790` has downstream truth ARI
`0.364506` and mean child purity `0.734586`, but the two downstream children
share the same majority truth label. Thus the conditional law cannot just ask
whether there is downstream heterogeneity. It must distinguish branch recovery
from barycentric mixture and fragmenting high-purity regions.

The branch-recovery conditioning panel isolates that target. In a focused
analytical fixture with `3` full branch recoveries and `1` partial recovery,
the oracle branch indicators separate all branch rows from barycentric,
fragment, and selected-null controls. Balance-only observable topology metrics
do not: `structural_balance_product` has AUC `0.9` but still leaks a
selected-null control. The feature-geometry extension adds a non-oracle
candidate: it reconstructs selected path membership, computes downstream
homogeneity gain and same-subspace consensus, and reports
`feature_branch_geometry_score`. In the focused fixture this score and its
components separate all branch recoveries from hard negatives. On the real
expanded overlap rows, feature geometry is observed for all `50` rows, but the
panel still reports `full_branch_recovery_support_missing`. The next
generated support fixture fills the first half of that gap: it creates `3`
full branch recoveries, `1` partial branch recovery, barycentric/fragment
negatives, and `3` selected-null controls through actual benchmark feature
matrices and synthetic selected paths. `feature_homogeneity_gain_min` and
`feature_branch_geometry_score` separate those branch rows from controls, while
balance product still leaks. The remaining enhancement is therefore not
another threshold on structural balance or bandwidth; it is mining or producing
real traversal-selected branch-positive cases with matched selected-null
controls under the selected-neighborhood law.

The distribution panel that follows from this comparison should measure
neighborhood distributions at the point where traversal actually stops or
passes through, not only at accepted split nodes. It should stratify rows by
root, internal, pass-through, and leaf incidence; strict-null, stopped-edge,
selected-nonnull, and truth role; and shallow versus deep/large-parent
context.

The minimum distribution variables are:

```text
traversal state:
  split, boundary, pass_through
  no_binary_structure
  edge_closed
  sibling_closed
  sibling_skipped
  explicit_guard_blocked

edge neighborhood:
  incoming edge p-value/action
  incoming sibling edge p-value/action
  outgoing left/right edge p-values/actions
  outgoing edge action balance

topology neighborhood:
  depth
  parent size
  child-size balance
  nearest stopped/stable/signal tree distance
  distance to pass-through ancestor
  distance to explicit guard block

calibration neighborhood:
  sibling projection dimension
  parent sample size
  strict-null support count
  stopped/edge-blocked support count
  selected-nonnull excluded count
  local effective support

internal recovery evidence:
  incoming_branch_balance
  outgoing_balance
  balance_product
  outgoing_edge_norm_balance
  fragment risk
```

The key statistical question is whether internal recovery positives occupy a
repeatable selected-neighborhood distribution, or whether the focused
`balance_product` separation is a one-row artifact.

## Evidence

- `GateEvaluator` defines the split, boundary, and pass-through actions.
- `TreeDecomposition` records traversal traces and delegates all split
  decisions to the gate evaluator.
- The child-parent edge metadata collector defines null-like, edge-blocked,
  and edge-weighted sibling support.
- The empirical-null inflation model uses strict-null or edge-blocked support
  and a log projection-dimension/log parent-size neighborhood kernel.
- [[old-vs-current-method-stack-comparison-20260615]] records the old c2ef
  topology-aware bandwidth variables and the current strict support path.
- [[specific-small-method-benchmark-20260615]] records that focused
  `balance_product` recovery works on synthetic positives but not yet on the
  real sparse context-negative slice.

## Links

- [[old-vs-current-method-stack-comparison-20260615]]
- [[overlap-conditional-topology-law-panel-20260615]]
- [[specific-small-method-benchmark-20260615]]
- [[open-mathematical-questions]]

## Open Questions

- How should the selected-neighborhood distribution be pooled across overlap
  cases without borrowing from selected-nonnull rows as empirical null support?
- Which topology distance variables from the older bandwidth layer remain
  useful after all-pairs tree distances are cached and support labels are
  enforced?
- What prospective focused overlap-positive generator gives enough internal
  recovery truth rows to validate the guarded recovery law?
