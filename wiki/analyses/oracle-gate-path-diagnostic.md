---
title: Oracle Gate-Path Diagnostic
type: analysis
status: reviewed
updated: 2026-05-24
sources:
  - benchmarks/shared/oracle_tree_recoverability.py
  - benchmarks/shared/gate_path_trace.py
  - benchmarks/shared/sibling_inflation_diagnostic.py
  - benchmarks/shared/kl_tree_context.py
  - benchmarks/run_oracle_tree_recoverability.py
  - benchmarks/run_gate_path_trace.py
  - benchmarks/run_sibling_inflation_diagnostic.py
  - benchmarks/results/oracle_tree_recoverability_20260524_155532Z/oracle_tree_recoverability.csv
  - benchmarks/results/gate_path_trace_20260524_155551Z/gate_path_trace.csv
  - benchmarks/results/gate_path_trace_20260524_155551Z/gate_path_trace_summary.csv
  - benchmarks/results/sibling_inflation_diagnostic_20260524_180332Z/sibling_inflation_targets.csv
  - benchmarks/results/sibling_inflation_diagnostic_20260524_180332Z/sibling_inflation_summary.csv
  - benchmarks/results/sibling_inflation_diagnostic_20260524_183607Z/sibling_inflation_targets.csv
  - benchmarks/results/sibling_inflation_diagnostic_20260524_183607Z/sibling_inflation_summary.csv
  - kl_clustering_analysis/config.py
  - kl_clustering_analysis/hierarchy_analysis/decomposition/gates/gate_evaluator.py
  - kl_clustering_analysis/hierarchy_analysis/tree_decomposition.py
tags:
  - analysis
  - oracle
  - gates
  - traversal
  - statistics
---

# Oracle Gate-Path Diagnostic

## Summary

The oracle gate-path diagnostic separates two mathematical questions that were
previously entangled. The first question is whether the fixed KL hierarchy
contains a good subtree cut. The second question is whether the implemented
statistical gates select that cut. The current evidence shows that the main
active gate failures are not tree-construction failures: twelve full-benchmark
cases have high oracle recoverability but are missed by the statistical
traversal. Five are under-splits caused by sibling calibration blocking a split
even when both child-parent edge gates are open. Seven are over-splits caused
by traversal continuing through oracle boundaries, either because the sibling
test directly opens inside a true clade or because pass-through follows
descendant split evidence after an ancestor sibling test says the children are
not different.

## Details

### Mathematical Object

Let \(T=(V,E)\) be the rooted binary hierarchy built by the KL benchmark
runner for a case. Its leaves are the samples \(1,\ldots,n\), and the true
labels are \(y_i\in\{1,\ldots,K\}\). A valid subtree cut \(C\subset V\) is a
set of pairwise-disjoint nodes whose descendant leaf sets partition all leaves.
Each node \(c\in C\) induces one predicted cluster. The implemented method
chooses one such cut through top-down gate decisions, while the oracle asks
which cut in the same tree would maximize agreement with the truth.

For a cut \(C\), define

\[
A(C)=\sum_{c\in C}\sum_{k=1}^{K}\binom{n_{ck}}{2},
\qquad
B(C)=\sum_{c\in C}\binom{n_c}{2},
\qquad
D=\sum_{k=1}^{K}\binom{n_k}{2},
\qquad
M=\binom{n}{2},
\]

where \(n_{ck}\) is the number of leaves in cut node \(c\) with true label
\(k\), \(n_c\) is the number of leaves under \(c\), and \(n_k\) is the total
number of leaves with true label \(k\). The adjusted Rand index for the cut is

\[
\operatorname{ARI}(C)
=
\frac{A(C)-B(C)D/M}
{\frac{1}{2}\{B(C)+D\}-B(C)D/M}.
\]

The numerator and the denominator are functions of additive quantities over cut
nodes. The oracle implementation uses Dinkelbach-style fractional
optimization: for a candidate ratio \(q\), it maximizes the additive objective

\[
A(C)-\left(\frac{D}{M}+q\left(\frac{1}{2}-\frac{D}{M}\right)\right)B(C)
-qD/2.
\]

The constant term \(-qD/2\) does not affect the chosen cut, so the dynamic
program can decide bottom-up whether each node is better accepted as a single
cluster or split into its children. The exact-\(K\) version adds a cluster-count
state and therefore restricts the dynamic program to cuts with exactly \(K\)
selected nodes. This oracle is not a competing clustering method. It is a
recoverability diagnostic for the fixed hierarchy.

### Gate Model Being Diagnosed

For each internal node \(v\), the implemented traversal evaluates three
conditions. The structural gate requires that \(v\) have two children. The
edge gate requires at least one child-parent projected-Wald edge test to be
significant. The sibling gate requires the sibling projected-Wald test to pass
the traversal-aligned sibling FDR decision. With `PASSTHROUGH=True`, a node
with open structural and edge prerequisites but closed sibling gate can still
be traversed when some descendant has a full split available.

The trace compares the oracle true-\(K\) cut against the actual traversal. A
node is marked as an under-split blocker when the actual traversal stops at an
ancestor of an oracle boundary. A node is marked as an over-split boundary when
the actual traversal splits or passes through an oracle boundary. The trace
also records child-parent p-values, sibling raw and adjusted p-values, sibling
projection dimension, empirical inflation factor, null weight, and whether
pass-through is responsible for continuing below a node.

### Classification Results

The corrected full-benchmark recoverability classification is:

```text
gate_over_split                  7
gate_under_split                 5
oracle_matched_below_solved      5
solved                          81
tree_unrecoverable              24
```

The `oracle_matched_below_solved` class is important. These cases are not gate
failures: the implemented KL cut matches the exact-\(K\) oracle within
tolerance, but both are below the solved ARI threshold. They should not drive
gate changes. The `tree_unrecoverable` class also should not drive gate
changes, because the current hierarchy does not contain a good exact-\(K\) cut.

The gate-path trace therefore targets twelve cases: five under-splits and seven
over-splits. Their summaries are:

```text
case_id                              class              KL ARI   oracle true-K ARI
gauss_extreme_noise_highd            under-split        0.0000   1.0000
gauss_extreme_noise_highd_continuous under-split        0.0000   1.0000
binary_balanced_low_noise            under-split        0.7054   0.9624
cat_mod_4cat_6c                      under-split        0.8194   0.9865
cat_highd_4cat_1000feat              under-split        0.8211   1.0000
cat_highd_3cat_500feat               over-split         0.1944   1.0000
phylo_dna_4taxa_low_mut              over-split         0.7672   1.0000
phylo_dna_8taxa_med_mut              over-split         0.9500   1.0000
phylo_protein_4taxa                  over-split         0.6684   1.0000
phylo_protein_12taxa                 over-split         0.2626   1.0000
phylo_conserved_8taxa                over-split         0.8028   1.0000
overlap_hd_4c_1k                     over-split         0.0000   0.9947
```

### Under-Split Mechanism

All five under-split blocker rows have the same mathematical form. The
child-parent edge tests are open for both children, but the sibling gate is
closed. The actual traversal therefore declares the current node a boundary,
even though the oracle cut lies below that node.

The blocking rows show that raw sibling evidence can be extremely strong while
the inflation-adjusted sibling p-value lands above `SIBLING_ALPHA = 0.01`.
For example, `gauss_extreme_noise_highd` has both child edges significant with
BH p-values recorded as zero, raw sibling p-value recorded as zero, projection
dimension 6, and empirical inflation around 7131. The adjusted sibling p-value
is about 0.122, so the sibling gate closes. The continuous companion has the
same structure with empirical inflation around 14608 and adjusted sibling
p-value about 0.423. The moderate binary and categorical under-splits show
the same pattern at smaller scale: open edges, strong raw sibling evidence,
inflation factors around 36 to 409, and corrected sibling p-values just above
the 0.01 threshold.

Mathematically, these are not edge-detection failures. They are failures of the
post-selection sibling calibration model. The inflation factor
\(\hat c(v)\) multiplies the projected quadratic reference scale so strongly
that the adjusted statistic no longer rejects, even when the edge evidence
already indicates that both children differ from their parent. The immediate
question is therefore whether \(\hat c(v)\) is estimating a real conditional
variance inflation in high-evidence split contexts or whether the empirical
null weighting degenerates and over-penalizes exactly the nodes that should be
split.

### Over-Split Mechanism

The over-split cases divide into two mechanisms.

The first mechanism is direct sibling false splitting inside oracle clades. In
the phylogenetic protein and conserved cases, most oracle boundary nodes have
open edge prerequisites and open sibling gates. For `phylo_protein_12taxa`, all
twelve oracle boundaries are directly split by the sibling gate, with median
sibling corrected p-values on the order of \(10^{-40}\) and median empirical
inflation around 1.35. This is not mainly a pass-through artifact. The local
sibling statistic is detecting within-taxon or within-clade variation as
split-worthy structure.

The second mechanism is pass-through fragmentation. In `overlap_hd_4c_1k`, all
four oracle boundaries are passed through, not directly split. The sibling gate
at those boundaries is closed, but descendant split evidence exists, so
pass-through continues downward and eventually produces 500 terminal clusters.
`cat_highd_3cat_500feat` is mixed: three oracle boundaries are passed through
and one is directly split. This means high-dimensional categorical and overlap
failures should not be treated as the same mathematical error as phylogenetic
false splitting.

The pass-through rule can be written schematically as

\[
\text{pass-through}(v)=
G_{12}(v)\land \neg G_3(v)\land
\exists u\prec v:\;G_{123}(u),
\]

where \(G_{12}\) denotes structural plus edge prerequisites, \(G_3\) denotes
the sibling gate, and \(G_{123}\) denotes a full descendant split. This rule
contains no comparison between the strength of the local sibling-same evidence
and the descendant split evidence. The trace shows that this omission matters:
when \(G_3(v)\) is closed at an oracle boundary, pass-through can still fragment
the true clade solely because some descendant looks locally splittable.

### Method Implications

The under-split and over-split findings should not be solved by one global
alpha change. Raising `SIBLING_ALPHA` would help the under-splits whose
corrected p-values are around 0.013 to 0.018, but it would worsen direct
sibling false splitting in the phylogenetic cases. Lowering `SIBLING_ALPHA`
would reduce over-splitting but further suppress the already blocked
under-splits. The trace therefore argues against a scalar threshold tweak as
the next principled method change.

The first mathematical development was a split-context-aware sibling inflation
diagnostic. It tests whether the empirical inflation estimate is too large when
both child-parent edges are strongly significant and the raw sibling p-value is
extremely small. The relevant object is not only the sibling statistic \(W_v\),
but the conditional context
\((p_{L|v},p_{R|v},d_v,\hat c(v),\pi_v)\), where \(p_{L|v}\) and \(p_{R|v}\)
are child-parent edge p-values, \(d_v\) is projection dimension, \(\hat c(v)\)
is empirical inflation, and \(\pi_v\) is the empirical sibling-null weight.

### Sibling Inflation Follow-Up

The follow-up diagnostic compares the implemented estimator with three
calibration variants for each focal sibling record:

\[
\hat c_{\mathrm{current}}(u),\qquad
\hat c_{-u}(u),\qquad
\hat c_{\mathrm{strict\ null}}(u),\qquad
\hat c_{\mathrm{blocked\ or\ null}}(u).
\]

Here \(\hat c_{-u}\) excludes the focal record from its own calibration set,
\(\hat c_{\mathrm{strict\ null}}\) uses only records marked `is_null_like`, and
\(\hat c_{\mathrm{blocked\ or\ null}}\) uses records that are either
`is_null_like` or edge-blocked. The diagnostic records explicit status values
such as `ok`, `no_calibration_records`, and `zero_calibration_weight`; missing
calibration support is therefore not converted into a neutral fallback.

The blocker rows have three different interpretations:

```text
case_id                              stage                    current c-hat  c-alpha    leave-one-out result
gauss_extreme_noise_highd            inflation-adjusted test      7131.4     4267.5    ok; still blocks
gauss_extreme_noise_highd_continuous inflation-adjusted test     14608.3     5213.6    zero calibration weight
binary_balanced_low_noise            sibling FDR                    35.8       37.0    ok; does not block before FDR
cat_mod_4cat_6c                      sibling FDR                    66.1       67.3    ok; does not block before FDR
cat_highd_4cat_1000feat              sibling FDR                   409.5      413.7    ok; does not block before FDR
```

For `gauss_extreme_noise_highd`, leave-one-out inflation remains above the
level needed to block at \(\alpha_{\mathrm{sib}}=0.01\), so the failure is not
explained only by self-calibration of the focal record. However, strict-null
calibration is unavailable, which means the current tree does not provide a
local empirical-null sample in that context.

For `gauss_extreme_noise_highd_continuous`, leave-one-out has zero positive
calibration weight and strict-null calibration is unavailable. This is a
calibration-support failure, not evidence that a neutral or strict-null
estimate should be silently substituted.

For the binary and categorical blockers, the current, leave-one-out, strict
null-like, and blocked-or-null-like estimates agree on the blocker row. The
inflation-adjusted p-value remains just below `SIBLING_ALPHA`, and the final
block occurs when sibling FDR correction moves the corrected p-value above the
threshold. These cases should drive a sibling-FDR analysis rather than another
inflation-estimator change.

The next recorded method step is therefore not to replace the production
inflation estimator immediately. It is to define the calibration-support
contract and then split the development path:

1. For high-dimensional Gaussian contexts, specify what the method should do
   when local empirical-null calibration is unavailable or has only
   underflow-level weight. The candidate outcomes are an explicit
   calibration-data error, a separately named conservative mode, or a validated
   external/null-simulation calibration model. A neutral fallback is not
   acceptable.
2. For binary and categorical under-splits, analyze the sibling FDR layer
   separately, because the inflation estimator is not the blocking stage.
3. Keep pass-through over-splitting and direct phylogenetic false splits on
   separate tracks; neither is explained by the under-split inflation
   diagnostic.

The manuscript/code contract also needed correction: the manuscript previously
described an effective-sample penalty multiplier as implemented. The code only
records effective sample size; it does not apply that multiplier. The draft now
marks the penalty as proposed and unimplemented unless separately validated.

### Calibration-Support Contract

For a focal sibling record \(u\), define the local non-focal support set

\[
\mathcal C_{-u}
=
\{q\neq u:\ f_q=f_u,\ \nu_q>0,\ w_q(u)>0\},
\]

where \(f_q\) is the feature family, \(\nu_q\) is the sibling-test degrees of
freedom, and \(w_q(u)=\pi_qK(z_q,z_u)\) is the local empirical-null calibration
weight. The focal record is excluded by definition.

The strict empirical-null support set is

\[
\mathcal C_{0,u}
=
\{q\in\mathcal C_{-u}:\ S_{l(q)}^{\mathrm{edge}}=0
\ \text{and}\ S_{r(q)}^{\mathrm{edge}}=0\},
\]

where \(S_{l(q)}^{\mathrm{edge}}\) and \(S_{r(q)}^{\mathrm{edge}}\) are the two
child-parent edge rejection indicators. The weak stopped-or-null support set is

\[
\mathcal C_{\mathrm{stop},u}
=
\{q\in\mathcal C_{-u}:\ q\in\mathcal C_{0,u}
\ \text{or}\ B_q^{\mathrm{edge}}=1\},
\]

where \(B_q^{\mathrm{edge}}\) indicates that the sibling record is edge-blocked
by the ancestor testing path.

The support-status contract is:

```text
strict_empirical_null_supported
  iff C_{0,u} has positive local calibration mass.

stopped_or_strict_empirical_null_supported
  iff C_{0,u} has no positive mass but C_{stop,u} does.

unsupported_without_empirical_null_support
  iff C_{stop,u} has no positive local calibration mass.
```

Only the first two statuses support an internal empirical-null interpretation.
If \(\mathcal C_{-u}\) contains only selected non-null context, that context
can be reported descriptively but is not a calibration state. The local
empirical-null calibration contract is unsatisfied whenever
\(\mathcal C_{\mathrm{stop},u}\) has no positive mass. A production method
must raise a calibration-data error; it must not silently use \(\hat c=1\),
recycle the target record, or call an external calibration model that has not
been validated under the full selection event.

This contract is necessary, not sufficient. It defines when an internal
empirical-null estimate is interpretable. It does not yet define a validated
minimum effective sample size, nor does it solve the high-dimensional Gaussian
case where strict support is absent.

Interpreting the pre-strict diagnostic rows for the two high-dimensional
Gaussian blockers gives:

```text
case_id                              support status                  action
gauss_extreme_noise_highd            unsupported_without_empirical_null_support   raise; research full-selection conditioning
gauss_extreme_noise_highd_continuous unsupported_without_empirical_null_support   raise; research full-selection conditioning
```

Thus the binary high-dimensional case has non-focal selected-context support,
but no strict empirical-null support. The continuous case does not even have
positive non-focal local calibration weight. Neither blocker should be
described as internally empirical-null supported. In the current strict
production path, selected non-null records are not admitted into the fitted
model, so both situations are production calibration failures rather than
runtime calibration states.

The production contract now follows this interpretation. The fitted inflation
model admits only positive-weight records that are strict null-like or
edge-blocked/stopped. Positive-weight selected non-null records are rejected as
calibration data. When the available support is selected-non-null only, the
method raises a calibration-data error instead of estimating \(\hat c\), using
\(\hat c=1\), or silently borrowing those records.

### Fixed-Subspace Gaussian Null Check

The next mathematical question was whether the huge inflation factors could be
explained by the projected-Wald reference law itself. The diagnostic
`benchmarks/run_gaussian_sibling_null_calibration.py` isolates the fixed
orthonormal-subspace null:

\[
Z\sim N(0,I_d),\qquad
Q_k=\lVert R_k Z\rVert^2,\qquad R_kR_k^\top=I_k.
\]

Under this null, \(Q_k\sim\chi^2_k\), so the external mean ratio is

\[
\hat c_{\mathrm{fixed}} =
\frac{\mathbb E_\ast[Q_k]}{a_u\nu_u}.
\]

This is not a full selected-tree null. It does not condition on the same
hierarchy, child-parent edge openings, or the fact that the sibling record was
selected as a blocker. For non-continuous observed features it is only a
standardized-\(z\) proxy, not a feature-family null calibration.

Applied to the two high-dimensional Gaussian blockers with 50,000 Monte Carlo
replicates:

```text
case_id                              scope                                      current c-hat  fixed-subspace c-hat  external blocks?
gauss_extreme_noise_highd            standardized z proxy for Bernoulli data        7131.4              0.9989       no
gauss_extreme_noise_highd_continuous continuous fixed-subspace Gaussian null        14608.3              1.0005       no
```

The fixed-subspace Gaussian null therefore does not explain the observed
inflation. The projected-Wald chi-square kernel is behaving as expected in the
same projected coordinate dimension. The open mathematical object is the
selection-conditioned null:

\[
\mathcal L\!\left(
T_u\mid
\text{same hierarchy},\
\text{child-parent prerequisites open},\
u\ \text{is a focal selected sibling context}
\right),
\]

not a replacement of the chi-square reference by another unconditioned
Gaussian law. Any future external calibration model would have to make that
conditioning explicit and be validated before entering production. A
fixed-subspace Gaussian check can be used as a lower-level diagnostic, but not
as a production fallback for missing calibration support.

### Local Edge-Selection Null Check

The next diagnostic conditioned on the first selection event that is local to
the blocker: the child-parent edge gate opens at the target parent. The
diagnostic keeps the observed tree and parent projection dimension fixed and
simulates

\[
Y\sim N(0,I_{k_{\mathrm{edge}}}),\qquad
Q_{\mathrm{edge}}=\sum_{j=1}^{k_{\mathrm{edge}}}Y_j^2,\qquad
Q_{\mathrm{sib}}=\sum_{j=1}^{k_{\mathrm{sib}}}Y_j^2.
\]

It accepts only draws with
\[
Q_{\mathrm{edge}}\ge \chi^2_{k_{\mathrm{edge}},1-\alpha_{\mathrm{edge}}},
\]
and estimates
\[
\hat c_{\mathrm{sel},1}(u)
=
\frac{\mathbb E_\ast[
Q_{\mathrm{sib}}\mid Q_{\mathrm{edge}}
\ge \chi^2_{k_{\mathrm{edge}},1-\alpha_{\mathrm{edge}}}
]}{a_u\nu_u}.
\]

This is a level-1 selection diagnostic. It conditions on the local edge event,
but not on Tree-BH over all edges, not on rebuilding the hierarchy, and not on
the event that the target appears as a blocker after the full traversal.

With 1,000,000 null candidates per target:

```text
case_id                              k_edge  k_sib  accepted  c_sel,1  current c-hat  selection p-value  blocks?
gauss_extreme_noise_highd                13      6      1011     2.80        7131.4           ~0.001      no
gauss_extreme_noise_highd_continuous      9      6      1000     3.40       14608.3           ~0.001      no
```

Conditioning on the local child-parent edge opening increases the null mean
from approximately one reference unit to roughly three reference units. That
is mathematically plausible, but it is nowhere near the thousands estimated by
the current empirical inflation model. The two blocker statistics remain far
outside this local selection-conditioned null. This makes the current failure
look like over-penalization from invalid calibration support, not like a
necessary correction for the projected-Wald reference or for the local
child-parent edge selection event alone.

The remaining mathematical object is therefore narrower:

\[
\mathcal L\!\left(
T_u\mid
\text{Tree-BH edge selection over the whole hierarchy},\
\text{observed tree construction},\
u\ \text{selected as a focal blocker context}
\right).
\]

If that full selection-conditioned null still gives \(c\) near the low
single digits, the production empirical inflation model should fail closed
when strict support is absent instead of borrowing selected non-null records.
If the full selection-conditioned null produces much larger \(c\), then the
external calibration model must be named and validated as a selection model,
not as an empirical-null fallback. The current diagnostics have not shown that
large-\(c\) full-selection effect, so no named external production model is
justified now.

### Root Tree-BH Selection Check

After enforcing the strict production contract, the Tree-BH selection
diagnostic was rerun without fitting the production sibling inflation model.
For the two high-dimensional blockers the target parent is the root. In a
binary root sibling group with equal target child-edge p-values, the fixed-tree
Tree-BH edge-path event is equivalent to the local root edge event:

\[
\text{TreeBH opens the target root}
\quad\Longleftrightarrow\quad
p_{\mathrm{edge}}\le \alpha_{\mathrm{edge}}.
\]

The diagnostic records this equivalence explicitly and reports the production
calibration status as unsupported. With 1,000,000 null candidates:

```text
case_id                              accepted  c_treeBH-root  production calibration
gauss_extreme_noise_highd                 984          2.81   unsupported
gauss_extreme_noise_highd_continuous     1021          3.45   unsupported
```

This confirms that, for the current two root blockers, replacing local edge
conditioning by the fixed-tree Tree-BH root edge path does not generate a large
inflation factor. A deeper non-root blocker would require simulating ancestor
Tree-BH selection along the observed path; these two cases do not exercise that
additional conditioning. Therefore the current production action remains a
strict calibration-data error, not a named selection-conditioned calibration
fallback.

The second mathematical development should be a stricter pass-through rule. A
candidate rule should require descendant split evidence to overcome local
sibling-same evidence, rather than treating any descendant split as sufficient.
This suggests a cut-level or evidence-ratio rule of the form

\[
\text{pass-through}(v)
\quad\text{only if}\quad
S_{\mathrm{desc}}(v) > \tau\{S_{\mathrm{same}}(v), n_v, d_v\},
\]

where \(S_{\mathrm{desc}}(v)\) summarizes descendant split evidence and
\(S_{\mathrm{same}}(v)\) summarizes the local evidence that the two children of
\(v\) should remain together. The diagnostic does not yet define the correct
functional form of \(\tau\); that remains an open method question.

The third mathematical development is separate from gate tuning. Cases in
`tree_unrecoverable` should be assigned to a hierarchy/metric track, not a
gate track. Diffuse dimensional Gaussian cases, heavy overlap cases, and hard
SBM cases require a different analysis of whether the distance, linkage, or
feature representation creates a hierarchy that can contain the target
partition.

## Evidence

- `benchmarks/shared/oracle_tree_recoverability.py` implements the exact
  subtree-cut oracle and the failure classification.
- `benchmarks/shared/gate_path_trace.py` constructs the node-level comparison
  between actual traversal, oracle boundaries, edge gates, sibling gates,
  inflation, and pass-through status.
- `benchmarks/shared/sibling_inflation_diagnostic.py` constructs the
  leave-one-out, strict-null, and blocked-or-null-like inflation variants and
  records explicit calibration-support statuses.
- `benchmarks/shared/gaussian_sibling_null_calibration.py` defines the
  fixed-subspace Gaussian sibling-null diagnostic contract and keeps the
  non-continuous case labeled as a standardized-\(z\) proxy rather than a
  feature-family null.
- `benchmarks/run_gaussian_sibling_null_calibration.py` runs that external
  diagnostic on selected sibling blocker contexts.
- `benchmarks/shared/selection_conditioned_sibling_null.py` defines the
  level-1 local edge-selection null diagnostic for sibling blocker contexts.
- `benchmarks/run_selection_conditioned_sibling_null.py` runs the fixed-tree,
  fixed-projection local edge-selection diagnostic on selected sibling
  blockers.
- `benchmarks/run_tree_bh_selection_conditioned_sibling_null.py` runs the
  fixed-tree root Tree-BH edge-path diagnostic without fitting unsupported
  sibling calibration.
- `benchmarks/shared/kl_tree_context.py` centralizes the benchmark KL tree
  construction contract so oracle and trace diagnostics use the same distance
  and linkage path.
- `benchmarks/results/oracle_tree_recoverability_20260524_155532Z/oracle_tree_recoverability.csv`
  records the corrected five-way classification for the full KL benchmark.
- `benchmarks/results/gate_path_trace_20260524_155551Z/gate_path_trace.csv`
  records the node-level gate evidence for the twelve true gate/stopping
  failures.
- `benchmarks/results/gate_path_trace_20260524_155551Z/gate_path_trace_summary.csv`
  records the case-level trace counts: under-split blocker counts, direct
  oracle-boundary splits, and fragmentation inside oracle boundaries.
- `benchmarks/results/sibling_inflation_diagnostic_20260524_180332Z/sibling_inflation_targets.csv`
  records current, leave-one-out, strict-null, and blocked-or-null-like
  estimates for focal sibling tests.
- `benchmarks/results/sibling_inflation_diagnostic_20260524_180332Z/sibling_inflation_summary.csv`
  records case-level counts of inflation blocking, FDR blocking, and
  unavailable strict-null calibration.
- `benchmarks/results/sibling_inflation_diagnostic_20260524_183607Z/sibling_inflation_targets.csv`
  records the calibration-support status and required action for the two
  high-dimensional Gaussian blockers.
- `benchmarks/results/sibling_inflation_diagnostic_20260524_183607Z/sibling_inflation_summary.csv`
  records case-level support counts for the high-dimensional Gaussian
  calibration-support check.
- `benchmarks/results/gaussian_sibling_null_calibration_20260524_192107Z/gaussian_sibling_null_targets.csv`
  records that the fixed-subspace Gaussian mean ratio is approximately one
  for both high-dimensional Gaussian blockers, while the runtime empirical
  inflation factors remain in the thousands.
- `benchmarks/results/gaussian_sibling_null_calibration_20260524_192107Z/gaussian_sibling_null_case_summary.csv`
  records the case-level external-null summary for the two blocker contexts.
- `benchmarks/results/selection_conditioned_sibling_null_20260524_200829Z/selection_conditioned_sibling_null_targets.csv`
  records that local edge-gate conditioning raises \(c\) only to about
  2.8--3.4 for the two high-dimensional Gaussian blockers.
- `benchmarks/results/selection_conditioned_sibling_null_20260524_200829Z/selection_conditioned_sibling_null_case_summary.csv`
  records acceptance rates near `EDGE_ALPHA = 0.001` and confirms that the
  selection-conditioned diagnostic does not block either target at
  `SIBLING_ALPHA`.
- `benchmarks/results/tree_bh_selection_conditioned_sibling_null_20260524_215606Z/tree_bh_selection_conditioned_sibling_null_targets.csv`
  records the fixed-tree root Tree-BH selection diagnostic and the unsupported
  production calibration status for the two high-dimensional Gaussian
  blockers.
- `kl_clustering_analysis/config.py` records `SIBLING_ALPHA = 0.01`,
  `EDGE_ALPHA = 0.001`, and `PASSTHROUGH = True`.
- `kl_clustering_analysis/hierarchy_analysis/decomposition/gates/gate_evaluator.py`
  implements the split and pass-through gate logic described here.

## Links

- [[kl-te-method]]
- [[projected-wald-statistic]]
- [[top-down-traversal]]
- [[tree-decomposition]]
- [[poset-tree]]

## Open Questions

- How should the external calibration law condition on the hierarchy,
  child-parent edge openings, and focal sibling selection event?
- For non-root blockers, does conditioning on the ancestor Tree-BH edge path
  still keep \(c\) in the low single digits, or does it introduce a larger
  finite-sample selected-context distortion?
- Which deeper full-selection diagnostic is needed for non-root blockers:
  fixed observed tree with ancestor Tree-BH path, focal blocker selection, or
  full hierarchy reconstruction under resampling?
- What sibling-FDR target should replace or justify the current flat
  traversal-aligned correction for binary and categorical blockers?
- Should pass-through be controlled by an evidence comparison between local
  sibling-same evidence and descendant split evidence?
- How should direct sibling false splitting inside phylogenetic clades be
  modeled: as a branch-length/effect-size stopping problem, a sibling
  covariance problem, or a missing hierarchical FDR constraint?
- Which hierarchy diagnostics should be built for `tree_unrecoverable` cases
  before changing metrics or linkage?
