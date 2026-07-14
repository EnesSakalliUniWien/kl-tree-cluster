---
title: Full Graphtools Tree Consensus Gate 2026-07-06
type: source
status: reviewed
updated: 2026-07-14
sources:
  - benchmarks/results/run_20260706_170815Z_full_graphtools_tree_consensus/full_benchmark_comparison.csv
  - benchmarks/results/run_20260706_170815Z_full_graphtools_tree_consensus/benchmark_performance_grid_summary.csv
  - benchmarks/results/run_20260706_170815Z_full_graphtools_tree_consensus/tree_consensus_report.md
  - benchmarks/results/run_20260706_170815Z_full_graphtools_tree_consensus/tree_consensus_summary.csv
  - benchmarks/results/run_20260706_170815Z_full_graphtools_tree_consensus/tree_consensus_selection.csv
  - benchmarks/results/run_20260706_170815Z_full_graphtools_tree_consensus/tree_consensus_rankings.csv
  - benchmarks/results/run_20260706_170815Z_full_graphtools_tree_consensus/tree_consensus_pairwise_agreement.csv
  - benchmarks/results/run_20260706_170815Z_full_graphtools_tree_consensus/tree_consensus_stability.csv
  - benchmarks/results/run_20260706_170815Z_full_graphtools_tree_consensus/tree_consensus_label_assignments.csv
  - benchmarks/results/run_20260706_170815Z_full_graphtools_tree_consensus/tree_consensus_fail_closed_analysis.csv
  - reports/tree_consensus_fail_closed_pvalues_20260709/fail_closed_pvalue_report.md
  - reports/tree_consensus_fail_closed_pvalues_20260709/fail_closed_loss_taxonomy.csv
  - reports/tree_consensus_fail_closed_pvalues_20260709/fail_closed_pvalue_cells.csv
  - reports/tree_consensus_fail_closed_pvalues_20260709/fail_closed_traversal_trace.csv
  - reports/tree_consensus_fail_closed_pvalues_20260709/fail_closed_fallback_external_audit.csv
  - reports/tree_consensus_alpha_sensitivity_20260709/alpha_sensitivity_report.md
  - reports/tree_consensus_alpha_sensitivity_20260709/alpha_threshold_requirements.csv
  - reports/tree_consensus_alpha_sensitivity_20260709/sibling_alpha_sweep_summary.csv
  - reports/tree_consensus_alpha_sensitivity_20260709/sibling_alpha_sweep_cells.csv
  - reports/tree_consensus_literature_policy_replay_20260709/literature_policy_replay_report.md
  - reports/tree_consensus_literature_policy_replay_20260709/literature_policy_replay_method_summary.csv
  - reports/tree_consensus_literature_policy_replay_20260709/literature_policy_replay_case_summary.csv
  - reports/tree_consensus_literature_policy_replay_20260709/literature_policy_replay_requirement_audit.csv
  - reports/tree_consensus_literature_policy_replay_20260709/literature_policy_replay_decisions.csv
  - reports/tree_consensus_literature_policy_replay_20260709/traversal_sibling_fdr_binary_smoke/traversal_sibling_fdr_summary.csv
  - reports/tree_consensus_brancharchitect_tree_comparison_20260709/brancharchitect_tree_comparison_report.md
  - reports/tree_consensus_brancharchitect_tree_comparison_20260709/brancharchitect_tree_case_summary.csv
  - reports/tree_consensus_brancharchitect_tree_comparison_20260709/brancharchitect_tree_cells.csv
  - reports/tree_consensus_brancharchitect_tree_comparison_20260709/brancharchitect_tree_pairwise.csv
  - reports/tree_consensus_brancharchitect_tree_comparison_20260709/brancharchitect_tree_newick.csv
  - reports/tree_consensus_topology_difference_diagnosis_20260709/topology_difference_diagnosis_report.md
  - reports/tree_consensus_topology_difference_diagnosis_20260709/topology_difference_method_summary.csv
  - reports/tree_consensus_topology_difference_diagnosis_20260709/topology_difference_pair_family_summary.csv
  - reports/tree_consensus_topology_difference_diagnosis_20260709/topology_difference_subtree_support.csv
  - reports/tree_consensus_topology_difference_diagnosis_20260709/topology_difference_subtree_highlights.csv
  - reports/tree_consensus_topology_difference_diagnosis_20260709/topology_difference_case_diagnosis.csv
  - reports/tree_consensus_topology_difference_diagnosis_20260709/topology_difference_null_recommendations.csv
  - reports/selected_node_sibling_null_law_20260709/selected_node_sibling_null_law_report.md
  - reports/selected_node_sibling_null_law_20260709/selected_node_sibling_null_law_conditions.csv
  - reports/selected_node_sibling_null_law_20260709/selected_node_sibling_null_law_examples.csv
  - reports/selected_node_sibling_null_law_20260709/selected_node_sibling_null_law_occurrences.csv
  - reports/selected_node_adaptive_law_replay_20260709/selected_node_adaptive_law_replay_report.md
  - reports/selected_node_adaptive_law_replay_20260709/selected_node_adaptive_law_method_summary.csv
  - reports/selected_node_adaptive_law_replay_20260709/selected_node_adaptive_law_case_summary.csv
  - reports/selected_node_adaptive_law_replay_20260709/selected_node_adaptive_law_external_audit.csv
  - .gitmodules
  - pyproject.toml
  - benchmarks/validation/brancharchitect_tree_comparison.py
  - tests/validation/194_test_brancharchitect_tree_comparison.py
  - benchmarks/shared/runners/method_registry.py
  - benchmarks/shared/runners/tbs_diffusion_runner.py
  - benchmarks/shared/runners/tbs_runner.py
  - tree_break_selection/tree/optimized_branch_lengths.py
  - tree_break_selection/hierarchy_analysis/statistics/contrast_covariance.py
tags:
  - source
  - benchmarks
  - diffusion
  - graphtools
  - adaptive-k
  - tree-inference
  - consensus
  - nnls
---

# Full Graphtools Tree Consensus Gate 2026-07-06

## Summary

The frozen label-free topology selector from the adaptive-K graphtools NNLS
focus panel was run as a full benchmark gate over the `121`-case suite and the
eight existing topology cells for `tbs_diffusion_graphtools_adaptive_nnls`.
The run produced the expected `968` result rows, persisted sample labels during
the benchmark path, and wrote consensus artifacts that select one topology per
case before external labels are audited.

The hard gate status is `pass`: completeness, label integrity, label-free
invariance, paired-valid mean external metrics, median external metrics, and
category-level regression checks all pass. This supports the frozen selector as
the next scientifically validated adaptive-K graphtools NNLS topology layer,
but it is not by itself a production-default promotion for the broader method.

## Key Points

- The full run used only `tbs_diffusion_graphtools_adaptive_nnls` with the
  `graphtools_adaptive_k_tree_strategy` grid: average, complete, weighted,
  single, centroid, median, Ward, and neighbor joining.
- `full_benchmark_comparison.csv` contains `968` rows: `121` cases times `8`
  run cells. There are `121` unique cases, `8` unique run ids, and no duplicate
  `case_id`/`run_id` pairs.
- The benchmark status profile is `925` `ok` rows and `43` `skip` rows.
  Skipped topology cells remain represented in the result table and fail
  closed in the selector.
- Persisted label rows pass integrity exactly: `221,800` combined label
  assignments equal the sum of `labels_length` over `ok` rows.
- The selector produced `101` selected cases and `20`
  `skip_no_valid_topology` cases.
- The `20` fail-closed selections separate into three classes: `14` real
  one-cluster under-splits against multi-cluster truth, `4` expected/null
  one-cluster rows where the selector is invalid only because it requires at
  least two clusters for internal metrics, and `2` full topology-cell support
  failures before labels are produced.
- The `43` skipped run cells are not equivalent to `43` failed cases: `30`
  occur inside the `20` fail-closed selection cases, while `13` occur in cases
  that still have a selected topology.
- Skipped run cells are concentrated in neighbor joining (`15` cells), Ward
  (`8`), complete (`6`), average (`4`), centroid (`3`), weighted (`3`),
  median (`2`), and single (`2`).
- Selected topology counts are weighted `45`, Ward `14`, neighbor joining
  `12`, single `7`, median `6`, centroid `6`, complete `6`, and average `5`.
- The selected rows have mean ARI `0.868327`, median ARI `1.0`, mean NMI
  `0.880812`, and mean macro F1 `0.921718`.
- Against average linkage on paired-valid rows, selected rows improve mean ARI
  from `0.838002` to `0.865667`, mean NMI from `0.861688` to `0.878405`, and
  mean macro F1 from `0.902504` to `0.920137`; medians remain tied at `1.0`.
- Against weighted linkage on paired-valid rows, selected rows improve mean
  ARI from `0.844943` to `0.868158`, mean NMI from `0.863204` to `0.880949`,
  and mean macro F1 from `0.904543` to `0.921373`; medians remain tied at
  `1.0`.
- No benchmark category with at least five paired-valid cases loses mean ARI
  against both average and weighted baselines by more than `0.02`.
- The label-free integrity check passes: blanking or permuting external
  columns does not change the selected topology output.
- The full-grid result strengthens the focus-panel conclusion that a single
  global tree topology is weaker than a frozen label-free selector over all
  accessible topology builders, followed by fixed-topology NNLS branch fitting.
- The focused 2026-07-09 p-value audit of the `14` real multi-cluster
  fail-closed under-splits separates them into `9` global edge-gate closures
  and `5` sibling-gate closures after edge support.
- In the `9` edge-gate failures, no successful topology cell has any
  all-node corrected child-parent edge p-value below `edge_alpha = 0.001`.
  The closest case is `dim_diffuse_6c_136f_continuous` with
  `min_any_edge_bh = 0.0119`.
- In the `5` sibling-gate failures, full-edge traversal can continue below
  the root, but no node opens the active sibling gate at
  `sibling_alpha = 0.01`. The closest case is `sbm_hard` with best corrected
  sibling p-value `0.0126`.
- The fail-closed p-value audit shows that fallback to the weighted topology
  cell inside the same adaptive-K graphtools NNLS grid does not rescue these
  cases: weighted returns one cluster where it runs. Current default `tbs`
  helps only a subset and has its own support skips, so production fallback
  should be explicit rather than silent.
- The skipped topology cells in the p-value audit are runtime support or
  precondition failures, not licensing failures: ten are MAD-rooted neighbor
  joining positive-distance precondition failures and one is a Ward empirical
  null calibration-support failure. The separate packaging concern is that the
  local `graphtools` and `tasklogger` package metadata report GPLv2 licenses.
- The 2026-07-09 alpha-sensitivity audit shows that alpha-only relaxation is
  not a scientifically adequate overlap fix. `overlap_heavy_8c_large_feat`
  needs `sibling_alpha = 0.20` to produce a nontrivial result, but the best
  rerun has ARI `0.076480` and `22` clusters; `overlap_extreme_4c` also needs
  `sibling_alpha = 0.20` but remains effectively unrecovered with ARI
  `0.000107` and largest-cluster fraction `0.983333`.
- The alpha audit supports the interpretation that dense/sparse overlap
  diagnostic p-values react to overlap signal, but the active corrected
  sibling gate is too conservative for overlap and naive global alpha
  relaxation is too blunt for production.
- The 2026-07-09 literature-policy replay tests eight policy families on the
  same fail-closed traversal traces: current traversal, Yekutieli-style
  hierarchical BH, TreeBH-style multiresolution BH, a dependence-robust BY
  proxy, Bretz-style graphical alpha recycling, Gao-style selective-inference
  fail-closed constraints, Wu/Bien/Panigrahi randomized alpha-spending
  fail-closed constraints, and one trace-only adaptive alpha-spending proxy.
- At the current `0.01` sibling budget, hierarchical FDR, TreeBH-style
  multiresolution BH, dependence-robust BY, and graphical alpha recycling
  rescue `0/14` fail-closed cases. This is expected from the observed active
  sibling p-values: preserving the local `0.01` budget cannot open overlap.
- The exact selective-inference and randomized-dendrogram alpha-spending
  families fail closed in the replay because the trace does not contain the
  paper-specific selected or randomized node p-values needed for truthful
  testing.
- The trace-only adaptive proxy with cap `0.20` opens candidate splits for
  `2/14` cases: `sbm_moderate` in `3` topology cells and
  `overlap_heavy_8c_large_feat` in `1` topology cell. The overlap candidate is
  weak and dominant (`3` clusters, largest-cluster fraction `0.95125`), so it
  is mechanism-discovery evidence only, not a production rescue.
- A companion `20`-replicate binary null FDR smoke shows the algorithmic
  synthetic-valid-p layer controls in this run (`mean_fdp = 0.0`), while
  fixed-tree Wald (`mean_fdp = 0.25`) and selected-tree Wald
  (`mean_fdp = 0.95`) over-reject; selected-tree inflated testing has
  `18/20` support failures. This reinforces that the missing object is not a
  larger global alpha, but selected-hierarchy calibration support.
- The 2026-07-09 BranchArchitect comparison rebuilds the same `14`
  multi-cluster fail-closed under-split cases across the eight adaptive-K
  graphtools NNLS topology cells and compares complete tree objects rather
  than stopped traversal traces.
- The BranchArchitect comparison writes `112` tree-cell rows: `101` rebuilt
  ok trees and `11` skipped cells. The skips match support/precondition
  issues: ten MAD-rooted neighbor-joining positive-distance failures and one
  Ward empirical-null calibration-support failure.
- All `101` ok rebuilt topology cells still return exactly one cluster.
  Consequently, no pairwise topology comparison changes the predicted cluster
  labels (`min_predicted_label_ari_between_topologies = 1.0` for every case)
  and no inter-vs-intra cluster path separation is finite.
- The pairwise tree layer records `315` topology comparisons. Median rooted
  internal RF-relative distance is `0.551839`, median rooted weighted split L1
  is `2.49375`, and median sampled leaf-path RMSE is `0.00213968`.
- Some topology cells differ strongly as rooted hierarchies without changing
  the clustering outcome. The largest case-level rooted RF-relative maxima
  occur in continuous fail-closed rows such as
  `dim_diffuse_6c_136f_continuous` (`0.994382`), while all cluster-count
  ranges remain `1-1`.
- BranchArchitect was loaded from a local checkout through
  `TBS_BRANCHARCHITECT_PATH` and was available for interpolation. The report
  records BranchArchitect distance metrics for all `315` pairwise comparisons;
  interpolation movement events are computed for `140` leaf-limited pairs and
  skipped by the explicit leaf cap for `175` larger pairs.
- The repository now pins that external implementation at a concrete commit
  under `vendor/BranchArchitect` as a git submodule. The optional
  `brancharchitect` dependency extra contains only the additional runtime
  packages required by the distance/interpolation adapter. BranchArchitect is
  MIT-licensed; the previously recorded GPL concern remains specific to the
  graphtools/tasklogger stack.
- The BranchArchitect comparison supports the diagnosis that the real
  fail-closed under-splits are not solved by choosing among the current
  topology cells or by inspecting NNLS branch lengths after the fact: the gate
  still returns the root cluster in every successful cell. The remaining issue
  is upstream edge/sibling support and selected-hierarchy calibration, not
  label-free topology consensus.
- The 2026-07-09 topology-difference diagnosis parses the
  BranchArchitect Newick exports and shows why the topology cells differ:
  topology is inferred before NNLS, while fixed-topology NNLS only fits branch
  lengths on the already selected hierarchy.
- Across all pairwise topology-cell comparisons, median rooted RF-relative
  distance is `0.5518`, while the median of per-case medians is `0` because
  the stable linkage core often shares topology. The maximum rooted
  RF-relative distance is `0.9944`.
- The method-family split is stable-core average/weighted/complete/Ward,
  nonmonotone centroid/median topology outliers, single-linkage high
  divergence, and neighbor joining as an additive-distance/MAD-rooted outlier
  with positive-distance precondition failures.
- The subtree-support layer emits `9,106` rooted internal subtree support rows
  and `168` ranked subtree highlights. Continuous edge-closed cases are mostly
  stable-core-vs-centroid/median ladder-like clade disagreements, while
  overlap/SBM cases show large unstable subtrees supported by one or a small
  subset of topology methods.
- The largest unstable subtree highlights include
  `overlap_heavy_8c_large_feat` with a `400`-leaf smaller-side subtree
  supported only by single linkage, `overlap_extreme_4c` with a `299`-leaf
  smaller-side subtree supported only by average linkage, and continuous
  consolidated/diffuse cases with `80`- to `90`-leaf stable-core or
  centroid/median clade contrasts.
- The topology/subtree diagnosis links these differences to the statistical
  failure mode: edge-closed cases need a selected-pipeline edge null before
  sibling testing is reached, while edge-supported overlap/SBM/low-rank cases
  need a selected-node sibling null conditioned on topology, NNLS branch
  lengths, local covariance, and topology stability rather than a blunt global
  alpha increase.
- The selected-node sibling-null law report formalizes that target. With
  selected topology `T`, selected parent `v`, child sample sizes `n_A,n_B`,
  NNLS branch lengths `ell_A,ell_B`, mean branch length `bar_ell`, local
  covariance `Sigma_v`, and fixed parent projection `U_{v,k}`, the current
  whitening scale is
  `(1/n_A + 1/n_B) * (1 + (ell_A + ell_B)/(2 * bar_ell))`. Under fixed
  conditioning the projected statistic has the implemented
  `chi_square(k)` law; the missing edge-supported law is instead
  `P_0(W_v >= W_obs | E_sel(T,v,k,U,ell,Sigma,topology_stability),
  H0_sibling(v))`. The report anchors this to selective-inference literature:
  hierarchical-clustering p-values must condition on the selected clusters,
  and post-selection inference requires the tested estimator's law conditional
  on the selection event.
- The selected-node law examples break the observed Wald sensitivity into
  smaller mechanisms: sample-size imbalance and long NNLS branch time reduce
  `W` for the same contrast, projection alignment can make the same norm
  visible or invisible, cutting through an eigenvalue multiplicity block is
  not orientation-invariant, and accepting extra open dimensions changes the
  chi-square tail even at nearly the same projected mass.
- The occurrence table keeps the case split unchanged: `5` edge-supported
  cases require the selected-node sibling law
  (`cont_lowrank_pggn_shrinkage`, `overlap_extreme_4c`,
  `overlap_heavy_8c_large_feat`, `sbm_hard`, `sbm_moderate`), while `9`
  edge-closed cases still need selected-pipeline edge calibration first. For
  the edge-supported bucket, the minimum active sibling p-value is `0.012573`
  but the minimum diagnostic sibling p-value is `1.028823e-25`, with median
  root branch-length ratio `157.352873`. The report also includes a
  requirement-coverage section mapping the requested topology, branch-length,
  covariance, eigenvector/multiplicity, open-dimension, occurrence, and
  smaller-example requirements to generated evidence.
- The adaptive-law benchmark replay applies that law target to the existing
  fail-closed benchmark traces without changing production behavior. The
  strict selected-law policy opens `0/14` cases because exact selected-node
  conditional p-values are absent. The topology/branch/covariance spending
  proxy also opens `0/14`, so conditioning on topology stability and NNLS
  branch-length stability remains fail-closed. The no-stability ablation opens
  only `sbm_moderate` in `3` topology cells; the nearest real rerun at
  `sibling_alpha = 0.20` has ARI `0.071841`, NMI `0.171602`, macro F1
  `0.472963`, and `6` clusters, so the opened split is diagnostic rather than
  promotion evidence.
- A follow-up diagnosis shows that the strict `0/14` result is not an observed
  failure of an implemented selected-node test. The strict policy is a
  fail-closed placeholder: `_evaluate_adaptive_node` returns
  `missing_exact_selected_node_conditional_p_value` unconditionally for that
  policy, and the trace schema has no selected-node conditional p-value field.
  Only `5` cases and `38` topology cells have full-edge replay traces; among
  their root decisions, `20` cells are edge-open with an ordinary active
  sibling p-value and `18` are edge-closed. The other `9` cases have no
  selected-node replay trace because they fail at the upstream edge gate.
  Therefore `0/14` combines `9` edge-law failures with `5` genuinely missing
  sibling-law cases and must not be interpreted as fourteen evaluated
  selected-node p-values.
- The current formal event is also not directly usable for rejection-sampling
  conditional Monte Carlo. It defines exact equality of the full selected
  state, including continuous NNLS lengths, covariance, eigenvectors, and
  stability values; for continuous inputs that exact event has probability
  zero. An executable law must condition exactly on the discrete selection
  state (adaptive K, topology/rooting, selected child sets, edge path, and
  accepted dimension) while either recomputing continuous nuisance quantities
  under each null draw or conditioning on a justified sufficient/ancillary
  statistic. Existing selected-hierarchy simulation does not fill this gap: it
  rebuilds default Hamming/average TBS without adaptive graphtools, topology
  consensus, or NNLS, and its validated external selected-tail domain excludes
  root/large-parent, continuous, and precomputed-distance contexts. The five
  relevant cases contain two large-root binary overlap contexts, two dependent
  SBM adjacency contexts, and one continuous low-rank context, so they also
  require distinct local-null generators rather than a single global iid
  permutation null.
- The frozen method uses four distinct geometric layers. Raw rows first enter
  a graphtools KNN graph with coordinate Hamming distance and adaptive
  `K` selected from `(5, 10, 15, 25, 40, 80, 160)` by connectivity and
  fragmentation criteria. The normalized graph is then mapped to `30`
  diffusion coordinates at diffusion time `3`, and Euclidean distance between
  those coordinates is passed to topology inference. Seven cells use SciPy
  linkage (`average`, `complete`, `weighted`, `single`, `centroid`, `median`,
  `ward`); the eighth uses neighbor joining followed by MAD rooting. Finally,
  fixed-topology NNLS fits path lengths to squared standardized Euclidean
  distance in the original feature matrix. Thus topology and branch length are
  currently estimated from different geometries.
- Reconstructed inputs show that Hamming is degenerate for all ten continuous
  fail-closed cases: every sampled off-diagonal pair has Hamming distance
  exactly `1`. The adaptive-K connectivity check still returns `K=10` because
  tied neighbors form a connected arbitrary graph, but that does not recover
  continuous geometry. In the two clear continuous Gaussian cases, the
  resulting linkage roots are `1/59` and `1/44` sample splits for the stable
  linkage methods, while standardized Euclidean distance has strong
  between/within separation. This identifies the raw graph metric as the
  primary upstream defect for the clear Gaussian, outlier, and phylogenetic
  continuous failures; high-dimensional diffuse, spike, and low-rank cases
  additionally have weak distance separation even under standardized
  Euclidean geometry.
- Hamming is not mathematically degenerate for the binary cases, but the
  observed separation is weak. The between/within Hamming-distance ratio is
  approximately `1.024` for heavy overlap, `1.002` for extreme overlap,
  `1.023` for moderate SBM, and `1.004` for hard SBM. Jaccard or cosine changes
  the geometry but does not by itself create strong separation in this audit.
  For SBM, row-wise Hamming is additionally poorly matched to graph data
  because common absent edges dominate and adjacency rows are dependent; a
  regularized graph-spectral distance is the more coherent candidate.
- Across the fourteen cases, the topology methods reach an edge-open root in
  `3/14` average, `4/14` complete, `2/14` weighted, `3/14` single, `1/14`
  centroid, `1/14` median, `3/13` Ward, and `3/4` successful neighbor-joining
  cells. No method opens a sibling gate. Median smaller-root-child fractions
  are only about `0.6%` to `1.3%` for the linkage methods, versus `7.7%` for
  the four successful neighbor-joining cells, showing that most linkage roots
  are outlier-versus-rest boundaries rather than plausible top-level cluster
  divisions.
- The five edge-supported cases separate by inference method. Low-rank
  continuous reaches sibling testing under average, complete, weighted,
  single, and Ward. Heavy binary overlap reaches it under average, complete,
  weighted, single, and neighbor joining. Extreme overlap reaches it under
  complete, median, Ward, and neighbor joining. Hard SBM reaches it under
  average, single, and centroid. Moderate SBM reaches it under complete, Ward,
  and neighbor joining. All active sibling p-values remain above `0.01`.
- NNLS changes only edge lengths on the fixed topology. The live normalized
  branch-time policy multiplies child-parent variance by
  `1 + ell/mean(ell)` and sibling variance by
  `1 + (ell_left + ell_right)/(2 mean(ell))`; it therefore keeps or reduces
  significance and cannot rescue an under-split by itself. Median observed
  root sibling variance multipliers are roughly `1.1` to `1.6` by topology
  method, with case maxima up to about `2.3`. A root-only no-branch
  counterfactual leaves all nine edge-closed cases above `edge_alpha=0.001`;
  the closest is `dim_diffuse_6c_136f_continuous` at approximately `0.00245`.
  Branch time contributes conservatism but is not the primary cause of the
  global closures.
- Branch lengths should therefore be estimated whenever a downstream method
  needs an additive tree metric, but they should enter hypothesis-test
  variance only when topology/target geometry are aligned and the additive fit
  is validated. A production branch-time gate should check normalized NNLS
  residual, path-distance correlation, pair-resampling stability, and topology
  stability. A poor or cross-geometry fit should use branch lengths as
  diagnostics and set branch-time variance to `none`, rather than treating
  fitted path length as calibrated stochastic time.

## Evidence

- `benchmarks/results/run_20260706_170815Z_full_graphtools_tree_consensus/full_benchmark_comparison.csv`
  records the `968` benchmark result rows.
- `benchmarks/results/run_20260706_170815Z_full_graphtools_tree_consensus/benchmark_performance_grid_summary.csv`
  records the eight run-cell performance summaries.
- `benchmarks/results/run_20260706_170815Z_full_graphtools_tree_consensus/tree_consensus_summary.csv`
  records `gate_status = pass` and the hard-gate criteria.
- `benchmarks/results/run_20260706_170815Z_full_graphtools_tree_consensus/tree_consensus_selection.csv`
  records one selected topology per valid case and
  `skip_no_valid_topology` for cases with no valid candidate.
- `benchmarks/results/run_20260706_170815Z_full_graphtools_tree_consensus/tree_consensus_rankings.csv`
  records all candidate ranks, penalties, scores, selector status, and
  external audit metrics.
- `benchmarks/results/run_20260706_170815Z_full_graphtools_tree_consensus/tree_consensus_pairwise_agreement.csv`
  records label-free adjusted Rand agreement between topology partitions.
- `benchmarks/results/run_20260706_170815Z_full_graphtools_tree_consensus/tree_consensus_stability.csv`
  records per-case method stability summaries used as tie-breaks.
- `benchmarks/results/run_20260706_170815Z_full_graphtools_tree_consensus/tree_consensus_label_assignments.csv`
  records the persisted sample labels used for agreement analysis.
- `benchmarks/results/run_20260706_170815Z_full_graphtools_tree_consensus/tree_consensus_fail_closed_analysis.csv`
  records the post-gate breakdown of the `20` fail-closed selected cases by
  null-vs-under-split-vs-support-failure class.
- `benchmarks/results/run_20260706_170815Z_full_graphtools_tree_consensus/tree_consensus_report.md`
  provides the human-readable pass/fail report.
- `reports/tree_consensus_fail_closed_pvalues_20260709/fail_closed_pvalue_report.md`
  records the detailed p-value loss-site analysis for the `14` real
  multi-cluster fail-closed under-splits.
- `reports/tree_consensus_fail_closed_pvalues_20260709/fail_closed_loss_taxonomy.csv`
  records the per-case edge-vs-sibling loss bucket, root edge counts, all-node
  edge significance counts, full-edge traversal counts, and best active
  sibling p-value.
- `reports/tree_consensus_fail_closed_pvalues_20260709/fail_closed_traversal_trace.csv`
  records live and full-edge traversal rows used to identify sibling-gate
  closures after edge support.
- `reports/tree_consensus_fail_closed_pvalues_20260709/fail_closed_fallback_external_audit.csv`
  records the external-metric fallback audit against current `tbs`, weighted
  topology, and the default-grid best method by ARI.
- `reports/tree_consensus_alpha_sensitivity_20260709/alpha_sensitivity_report.md`
  records the focused alpha-threshold and actual rerun analysis for the
  edge-supported fail-closed cases.
- `reports/tree_consensus_alpha_sensitivity_20260709/sibling_alpha_sweep_summary.csv`
  records the actual rerun outcomes for `sibling_alpha` values `0.015`,
  `0.025`, `0.05`, `0.10`, and `0.20` on topology cells with existing
  edge-open traversal.
- `reports/tree_consensus_literature_policy_replay_20260709/literature_policy_replay_report.md`
  records the literature-policy mapping, pass-through-aware replay summary,
  candidate split cases, companion FDR smoke, and production caveats.
- `reports/tree_consensus_literature_policy_replay_20260709/literature_policy_replay_method_summary.csv`
  records one row per policy family, including candidate split counts, total
  opened nodes, pass-through nodes, cluster-count proxies, and local-alpha
  maxima.
- `reports/tree_consensus_literature_policy_replay_20260709/literature_policy_replay_case_summary.csv`
  records one case/policy row for all `14` fail-closed under-splits.
- `reports/tree_consensus_literature_policy_replay_20260709/literature_policy_replay_requirement_audit.csv`
  records one requirement row for each of the six requested literature
  families, separating executable analogues, conservative proxies,
  absent-required-p-value constraints, and diagnostic-only outcomes.
- `reports/tree_consensus_literature_policy_replay_20260709/traversal_sibling_fdr_binary_smoke/traversal_sibling_fdr_summary.csv`
  records the companion algorithmic/fixed-tree/selected-tree/inflated sibling
  FDR smoke over `20` binary null replicates.
- `reports/tree_consensus_brancharchitect_tree_comparison_20260709/brancharchitect_tree_comparison_report.md`
  records the complete rebuilt-tree comparison across the `14` real
  fail-closed under-splits, including BranchArchitect availability and
  pairwise topology/path summaries.
- `reports/tree_consensus_brancharchitect_tree_comparison_20260709/brancharchitect_tree_case_summary.csv`
  records one row per fail-closed case with ok/skipped tree-cell counts,
  cluster-count range, rooted RF-relative maxima, weighted split distances,
  sampled leaf-path RMSE, and BranchArchitect interpolation status counts.
- `reports/tree_consensus_brancharchitect_tree_comparison_20260709/brancharchitect_tree_cells.csv`
  records one row per rebuilt or skipped topology cell, including branch-length
  totals, cluster-size summaries, and sampled predicted-cluster path metrics.
- `reports/tree_consensus_brancharchitect_tree_comparison_20260709/brancharchitect_tree_pairwise.csv`
  records the `315` pairwise tree comparisons with rooted split distances,
  weighted split distances, sampled leaf-path distances, predicted-label
  agreement, BranchArchitect RF, BranchArchitect weighted RF, and
  BranchArchitect interpolation event totals where leaf caps allow.
- `reports/tree_consensus_brancharchitect_tree_comparison_20260709/brancharchitect_tree_newick.csv`
  records branch-length Newick strings for the `101` rebuilt ok trees.
- `reports/tree_consensus_topology_difference_diagnosis_20260709/topology_difference_diagnosis_report.md`
  records the inferred mechanism for topology differences, subtree
  instability, and the resulting edge/sibling null-hypothesis requirements.
- `reports/tree_consensus_topology_difference_diagnosis_20260709/topology_difference_method_summary.csv`
  records method-family roles and topology/path/p-value summaries for the
  eight accessible topology methods.
- `reports/tree_consensus_topology_difference_diagnosis_20260709/topology_difference_pair_family_summary.csv`
  records one row per unordered topology-method pair with RF, weighted split,
  path-distance, and predicted-label agreement summaries.
- `reports/tree_consensus_topology_difference_diagnosis_20260709/topology_difference_subtree_support.csv`
  records rooted internal subtree support across topology cells with subtree
  hashes, smaller-side sizes, supporting/absent methods, branch lengths, and
  contrast type.
- `reports/tree_consensus_topology_difference_diagnosis_20260709/topology_difference_subtree_highlights.csv`
  records the ranked high-impact unstable subtrees per case.
- `reports/tree_consensus_topology_difference_diagnosis_20260709/topology_difference_case_diagnosis.csv`
  records per-case topology, root-branch, p-value, subtree, and null-model
  diagnosis fields.
- `reports/tree_consensus_topology_difference_diagnosis_20260709/topology_difference_null_recommendations.csv`
  records the selected-tree edge null, branch-length-conditioned sibling null,
  and topology-stability alpha-spending recommendations.
- `reports/selected_node_sibling_null_law_20260709/selected_node_sibling_null_law_report.md`
  records the mathematical selected-node sibling-null target, fixed-subspace
  theorem boundary, smaller Wald-reaction examples, and observed occurrence
  split.
- `reports/selected_node_sibling_null_law_20260709/selected_node_sibling_null_law_conditions.csv`
  records the seven conditioning components: fixed topology/subspace,
  selected topology/node, NNLS branch-length scale, local covariance
  eigensystem, multiplicity projector, adaptive dimension mixture, and
  topology-stability alpha spending.
- `reports/selected_node_sibling_null_law_20260709/selected_node_sibling_null_law_examples.csv`
  records ten deterministic examples that isolate sample-size, branch-length,
  projection-alignment, multiplicity, and accepted-dimension effects on the
  projected-Wald statistic.
- `reports/selected_node_sibling_null_law_20260709/selected_node_sibling_null_law_occurrences.csv`
  records the two observed law buckets: `5` selected-node sibling-null cases
  and `9` selected-pipeline edge-null cases.
- `reports/selected_node_adaptive_law_replay_20260709/selected_node_adaptive_law_replay_report.md`
  records the benchmark replay of strict selected law, topology/branch
  adaptive spending, and no-stability ablation policies.
- `reports/selected_node_adaptive_law_replay_20260709/selected_node_adaptive_law_method_summary.csv`
  records one row per replay policy with split-case counts, opened-node
  counts, cluster-count proxies, local-alpha maxima, and topology-spending
  minima.
- `reports/selected_node_adaptive_law_replay_20260709/selected_node_adaptive_law_case_summary.csv`
  records one row per fail-closed case and replay policy.
- `reports/selected_node_adaptive_law_replay_20260709/selected_node_adaptive_law_external_audit.csv`
  joins replay candidate splits to the prior sibling-alpha sweep external
  metrics where available.

## Links

- [[graphtools-adaptive-k-tree-consensus-focus-benchmark-20260630]]
- [[graphtools-adaptive-k-tree-inference-focus-benchmark-20260630]]
- [[full-graphtools-nnls-benchmark-run-20260630]]
