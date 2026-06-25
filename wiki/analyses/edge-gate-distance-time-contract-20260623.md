---
title: Edge Gate Distance Time Contract
type: analysis
status: reviewed
updated: 2026-06-24
sources:
  - tree_break_selection/hierarchy_analysis/statistics/branch_length_utils.py
  - tree_break_selection/hierarchy_analysis/statistics/child_parent_divergence/child_parent_divergence_annotation/tree_testing.py
  - tree_break_selection/hierarchy_analysis/decomposition/gates/orchestrator.py
  - tree_break_selection/hierarchy_analysis/tree_decomposition.py
  - tree_break_selection/tree/optimized_branch_lengths.py
  - tree_break_selection/hierarchy_analysis/statistics/contrast_covariance.py
  - benchmarks/shared/runners/tbs_runner.py
  - scripts/pancreas_scrna_cluster_benchmark.py
  - tests/statistics/22_test_edge_branch_length_regression.py
  - tests/statistics/45_test_edge_gate_math_contract.py
  - tests/statistics/46_test_continuous_covariance_numerical_psd.py
  - tests/tree/test_optimized_branch_lengths.py
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/edge_gate_distance_time_model_analysis.md
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/method_metrics.csv
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_branch_time_sensitivity.csv
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_tree_branch_length_summary.csv
tags:
  - analysis
  - edge-gate
  - branch-length
  - clustering
---

# Edge Gate Distance Time Contract

## Summary

The edge gate now separates tree topology from stochastic time. Linkage and
adaptive-diffusion distances can build a topology and record branch lengths,
but those branch lengths do not inflate child-parent Wald variance by default.
Branch-time variance is an explicit opt-in policy:
`edge_branch_length_variance_policy="normalized_branch_length"`.

The earlier fixed-coordinate compatibility workaround has been rejected. Fixed
coordinate evidence remains diagnostic, but it no longer changes the production
edge projection floor. The current production-compatible path is shared
projected-Wald geometry: both edge and sibling tests can use a local adaptive
PCA prefix selected by projected contrast energy.

## Details

For a binary parent with child means \(L,R\) and parent mean
\[
P = \frac{n_L L+n_R R}{n_L+n_R},
\]
the child-parent edge contrast satisfies
\[
L-P = \frac{n_R}{n_L+n_R}(L-R).
\]
With the same coordinate chart, covariance model, and no extra branch-time
variance multiplier, the edge gate and sibling gate are therefore testing
closely related evidence. A strong sibling test combined with two weak
child-parent tests is not automatically impossible, but it is a red flag when
the gates use different projection dimensions or different variance models.

The pancreas failure showed that mismatch directly: the sibling gate saw a
strong difference, while the edge gate used a tiny adaptive projection and also
had an ambiguous branch-time interpretation. The current production contract
fixes the first issue by making both gates projected/adaptive and makes the
second explicit.

### Distance, Topology, and Time

For a hierarchical system, there are three related but distinct objects:

- a dissimilarity \(d(i,j)\) between observed leaves;
- a selected topology \(\mathcal{T}(d)\), produced by linkage,
  neighbor-joining, adaptive diffusion, or another tree builder;
- an edge-time map \(\tau:E(\mathcal{T})\rightarrow \mathbb{R}_{\ge 0}\) used
  in a stochastic model.

The current production contract lets \(d\) choose \(\mathcal{T}\) and records
linkage/adaptive-diffusion heights as branch-length diagnostics, but it does
not automatically identify those heights with \(\tau\). This matters because
linkage heights are selected order statistics of the same data used for tests.
Treating them as independent elapsed variance time can close real child-parent
contrasts solely because a selected merge was far apart in the tree-builder's
geometry.

Candidate time models have different null implications:

- **Topology-only sampling model:** \(\tau_e=0\) for variance purposes. Edge
  tests use finite-sample contrast covariance. This is the default and is the
  least committal model for Scanpy PCA linkage trees.
- **Brownian branch-time model:** edge contrast variance grows with time,
  usually through \(m_e=1+\tau_e/\bar{\tau}\). This is appropriate only if
  branch lengths are calibrated elapsed stochastic time, not just selected
  linkage distances.
- **Bounded or robust time model:** \(m_e=1+c\min(\tau_e,q_p)/\bar{\tau}\) or
  \(m_e=1+c(\tau_e/\bar{\tau})^\gamma\). This is useful diagnostically because
  it exposes sensitivity to long selected branches, but its tuning constants
  need an unsupervised criterion.
- **Diffusion-time topology model:** diffusion parameters choose a geometry and
  topology; diffusion time \(t\) is not automatically the edge-time map
  \(\tau_e\). It can be a scale parameter for neighborhood smoothing while edge
  variance remains finite-sample.
- **Fixed-topology additive-distance model:** a selected topology is held fixed
  and non-negative edge lengths are fit so that patristic distances approximate
  continuous-data pairwise distances. This is now implemented as native NNLS in
  `tree_break_selection/tree/optimized_branch_lengths.py`. It is a better
  branch-length contract than raw linkage heights because it directly optimizes
  an additive tree metric, but it still does not by itself prove that the edge
  lengths are Brownian stochastic time.

The edge gate therefore accepts branch time only through an explicit policy.
Any future production length optimization must specify the target objective
without curated labels, for example a null-calibrated stability criterion,
held-out likelihood, or selective-inference support condition.

### Policy

- `none`: branch lengths are topology/support diagnostics only; edge variance is
  finite-sample contrast covariance. This is the default for linkage and
  adaptive-diffusion topologies.
- `normalized_branch_length`: edge variance is multiplied by
  \(1 + \ell_e/\bar{\ell}\), where \(\ell_e\) is the edge branch length and
  \(\bar{\ell}\) is the positive mean branch length. This is a sensitivity
  model, not the default for Scanpy PCA linkage trees.
- `fixed_topology_nnls`: branch lengths are first refit by non-negative least
  squares against continuous-data pairwise distances on the selected topology,
  then used only when branch-time variance scaling is explicitly requested.
  This is the current native branch-time candidate for linkage-built continuous
  trees in this codebase.

### Pancreas Evidence

The adaptive projected rerun no longer collapses at the root and no longer uses
the rejected full-rank edge workaround. Adaptive-diffusion TBS gives `43`
clusters, weighted cluster purity `0.9288`, dominant-cluster recall `0.4840`,
split error `0.5160`, and V-measure `0.6620`. Topology-only standardized-PCA
linkage TBS gives `45` clusters, purity `0.7928`, recall `0.5800`, split error
`0.4200`, and V-measure `0.6020`. The raw-linkage branch-time diagnostic
collapses too aggressively to `9` clusters, purity `0.3772`, recall `0.9932`,
and V-measure `0.3124`.

The key conclusion is not that ARI is low. The repaired adaptive projection
improves the split/merge frontier, especially with adaptive-diffusion topology,
but topology choice and branch-time variance can still swing TBS between
over-splitting and over-merging.

### Native NNLS Branch-Time Evidence

The native fixed-topology NNLS rerun confirms that the bad branch-time row is a
raw-linkage-height problem, not an unavoidable consequence of using continuous
branch lengths. The benchmark contract now treats recomputed-NNLS as the
branch-time linkage-tree method and raw linkage heights as an explicit
diagnostic-only negative control. On standardized-PCA topology,
recomputed-NNLS branch time returns the same final clusters as topology-only
TBS: `45` clusters, purity `0.7928`, dominant-cluster recall `0.5800`, and
V-measure `0.6020`. On adaptive-diffusion topology, recomputed-NNLS branch
time returns the same final clusters as adaptive-diffusion topology-only TBS:
`43` clusters, purity `0.9288`, recall `0.4840`, and V-measure `0.6620`. The
raw-linkage diagnostic row remains the outlier with `9` clusters, purity
`0.3772`, recall `0.9932`, and V-measure `0.3124`.

The two NNLS fits used `50,000` sampled leaf pairs. The standardized-PCA fit
has residual RMSE `0.3018` and MAE `0.2017`; the adaptive-diffusion fit has
residual RMSE `0.3075` and MAE `0.2013`. Optimizer runtime was about `3.3`
seconds in both rows. This is fast enough to keep as a native benchmark option.

The adaptive-diffusion NNLS row also exposed a covariance numerical issue:
continuous covariance blocks that were PSD up to tiny floating-point negatives
could fail Cholesky after branch-time scaling. The repaired covariance path
symmetrizes scaled full covariance blocks and adds only the jitter needed to
make the block positive definite. This repair is covered by
`tests/statistics/46_test_continuous_covariance_numerical_psd.py`.

### Branch-Time Transform Scan

The benchmark now includes a supervised fixed-topology sensitivity scan over
candidate branch-time transforms. For each scanned model, the edge Wald
statistic is rescaled as
\[
W_e(m) = \frac{W_e(1)}{m_e},
\]
Tree-BH is recomputed, and traversal is rerun with the same sibling gate. This
is a failure-analysis diagnostic, not an unsupervised production rule, because
the reported optimum is selected against curated cell-type labels.

The scan confirms that length transforms act mainly as a split/merge dial. On
adaptive diffusion topology, `linear_scale_2` gives the best scanned V-measure,
`0.6700`, with `36` clusters, purity `0.9284`, and dominant-cluster recall
`0.4992`. On standardized-PCA topology, the scanned length transforms did not
improve over the unscaled adaptive-projected topology row.

This means branch-time optimization can improve the benchmark frontier, but it
does not supply the missing mathematical contract. A production rule still needs
an unsupervised time model or an independent traversal regularizer that controls
fragmentation without tuning to labels.

### Tree Inference Methods Capable of Continuous Edge-Length Fitting

The methods that are compatible with the current TBS setup are distance-tree or
Gaussian-covariance tree methods, not sequence-only phylogenetic likelihood
engines. The practical ordering is:

- Fixed-topology NNLS or weighted least squares on pairwise continuous
  distances. This is the closest native fit because it preserves the selected
  topology and solves directly for non-negative additive edge lengths.
- Neighbor joining, BIONJ, minimum evolution, and FastME-style balanced minimum
  evolution on Euclidean, Mahalanobis, or diffusion distances. These can infer
  both topology and branch lengths from continuous-data distance matrices, but
  the branch lengths are distance-fit lengths rather than automatically valid
  Brownian variance times.
- Brownian-motion or OU continuous-trait likelihood on a fixed or searched
  tree. This is statistically closest to the edge-gate covariance model, but it
  is heavier and needs decisions about dimensionality, trait covariance,
  feature splitting, and whether genes or PCs are independent replicated traits.
- Principal-graph and pseudotime methods, such as MST plus principal curves or
  elastic principal trees, optimize continuous trajectories but do not natively
  return a binary additive covariance tree. They are useful topology
  comparators, not immediate branch-time replacements.

## Evidence

- `tests/statistics/22_test_edge_branch_length_regression.py` verifies that
  topology branch lengths are ignored by default and only enter edge variance
  under the normalized branch-length policy.
- `tests/statistics/45_test_edge_gate_math_contract.py` verifies that
  fixed-coordinate sibling gates no longer raise the edge projection floor, that
  the adaptive projection fraction reaches both projected gates, and that the
  branch-time policy is part of gate-reuse metadata.
- `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/edge_gate_distance_time_model_analysis.md`
  records the benchmark interpretation and branch-length summaries with
  `Generated at: 2026-06-24T20:07:07+02:00`.
- `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/method_metrics.csv`
  records the split/merge metrics used to diagnose the remaining fragmentation.
- `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_branch_time_sensitivity.csv`
  records the fixed-topology branch-time transform scan and confirms the
  split/merge tradeoff.
- `tree_break_selection/tree/optimized_branch_lengths.py` implements the native
  fixed-topology NNLS branch-length fit.
- `tests/tree/test_optimized_branch_lengths.py` verifies the NNLS optimizer on
  a known additive continuous embedding.
- `tests/statistics/46_test_continuous_covariance_numerical_psd.py` verifies the
  covariance scaling repair needed by adaptive-diffusion NNLS branch time.
- `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_tree_branch_length_summary.csv`
  records the six TBS tree summaries, recomputed-NNLS residual diagnostics, and
  raw-linkage diagnostic rows.

## Open Questions

- Which branch-time null, if any, is justified for diffusion-derived trees in
  continuous scRNA PCA space?
- Should the native branch-time model be cross-fit, using disjoint features for
  topology construction, branch-length fitting, and projected gates?
- Can a Brownian or OU continuous-trait likelihood be made computationally
  practical for the pancreas-scale benchmark without collapsing the feature
  covariance into an arbitrary low-rank summary?
- What traversal regularizer should reduce TBS fragmentation while preserving
  the high cluster purity seen in the pancreas run?
- Should the benchmark suite promote split/merge diagnostics to first-class
  acceptance criteria alongside legacy ARI summaries?

## Links

- [[pancreas-scrna-clustering-benchmark-20260623]]
- [[continuous-tree-geometry-rethink-20260623]]
