# Selected-Node Sibling Null Law

Generated UTC: `20260709_180648Z`

## Status

This report is a mathematical diagnostic and validation target. It does not
promote a production p-value. The fixed-subspace chi-square law remains valid only after conditioning on topology, NNLS branch-time scale, local covariance chart, selected node, projection subspace, and accepted dimension.

`5` case(s) require the selected-node sibling law; minimum active sibling p-value `0.01257`, minimum diagnostic sibling p-value `1.03e-25`, median topology branch-ratio `157.4`.

## Conditional Law

Let `T` be the selected rooted topology, `v` the selected binary parent, `A_v`
and `B_v` its child leaf sets, `n_A,n_B` their effective sample sizes, and
`ell_A,ell_B` their NNLS branch lengths. Let `bar_ell` be the mean branch
length. The current whitening scale is

`a_v = (1/n_A + 1/n_B) * (1 + (ell_A + ell_B)/(2 * bar_ell))`.

For local covariance `Sigma_v`, choose `L_v L_v^T = a_v Sigma_v`. With fixed
parent projection `U_{v,k}`, the fixed conditional statistic is

`W_v = || U_{v,k} L_v^{-1}(hat_mu_A - hat_mu_B) ||^2`.

If all selected objects are fixed or independent of the tested contrast, then
`W_v | T,v,k,U,Sigma,ell ~ chi_square(k)`. The selected-node law needed for
edge-supported cases is instead

`p_sel(v) = P_0(W_v >= W_obs | E_sel(T,v,k,U,ell,Sigma,topology_stability), H0_sibling(v))`.

The event `E_sel` includes adaptive KNN diffusion, topology inference method,
rooting, NNLS branch fitting, edge-open path, selected node, accepted dimension,
eigenvalue multiplicity handling, and predeclared topology-stability spending.

## Fixed-Subspace Proposition

Assume `C_v = (T,v,A_v,B_v,ell_A,ell_B,bar_ell,Sigma_v,U_{v,k},k)` is fixed.
Under the local sibling null, assume the contrast satisfies

`hat_delta_v = hat_mu_A - hat_mu_B | C_v,H0 ~ N(0, a_v Sigma_v)`.

Let `L_v L_v^T = a_v Sigma_v` and let the rows of `U_{v,k}` be orthonormal in
the whitened tangent chart. Then

`Z_v = U_{v,k} L_v^{-1} hat_delta_v ~ N(0, I_k)`

because `L_v^{-1} hat_delta_v ~ N(0,I)` and
`U_{v,k} U_{v,k}^T = I_k`. Therefore

`W_v = Z_v^T Z_v ~ chi_square(k)`.

This proves the current fixed-subspace reference. It also proves its boundary:
if `C_v` is learned from the same data in a way that depends on
`hat_delta_v`, then the unconditional or selected conditional law is no longer
this plain chi-square law unless the selection event is independent of the
tested whitened contrast.

## Selected Probability Mass

Let `S(X)` be the full construction map from data to adaptive KNN graph,
topology method, rooted topology, NNLS branch lengths, selected node, parent
projection, accepted dimension, and topology-stability state. For the observed
state `s_obs`, define `E_sel = {X: S(X)=s_obs}`. The selected p-value is the
conditional mass

`p_sel(v) = integral 1{W_v(X) >= W_v(X_obs)} dP_0(X | X in E_sel, H0_sibling(v))`.

Equivalently, it is a conditional Monte Carlo target over null draws that
rerun the same construction and keep only draws that reproduce the selected
state, or an analytic selective-inference target if `E_sel` is written as a
tractable selected region. This is why a larger global alpha is too blunt: it
does not define the conditional mass, it only changes a threshold.

## Literature Anchor

This construction follows the selective-inference principle that the tested
null must be conditioned on the data-dependent selection event. Gao, Bien, and
Witten's [Selective Inference for Hierarchical Clustering](https://arxiv.org/abs/2012.02936)
shows that classical mean-difference tests can be badly anti-conservative when
clusters are selected by hierarchical clustering and replaces them with
cluster-selection-conditioned p-values. Lee, Sun, Sun, and Taylor's
[Exact post-selection inference](https://arxiv.org/abs/1311.6238) gives the
general post-selection template: characterize the law of the tested estimator
conditional on the selection event. Our topology-conditioned sibling law is the
Tree-Break analogue of that template, with `E_sel` containing adaptive KNN,
topology inference, rooting, NNLS branch lengths, selected node, projection,
dimension, and topology-stability state.

## Eigenvalue Multiplicity And Open Dimensions

If `Sigma_v` has an eigenvalue block with multiplicity `m`, individual
eigenvectors inside that block are identifiable only up to an orthogonal
rotation. A statistic that accepts only part of the block can change when the
basis is rotated while the invariant covariance and signal norm are unchanged.
The invariant object is the whole projector `P_G` onto the tied eigenspace, with
`||P_G Z||^2 ~ chi_square(rank(P_G))` under fixed conditioning.

If the accepted dimension `K` is data-selected, the selected law is

`P_0(W_K >= w | E_sel) = sum_k P_0(W_k >= w, K=k | E_sel)`.

A chi-square tail with `df=k` is admissible only after conditioning on the
accepted value `K=k` and on the rule that made `K` observable. This is the
mathematical place where the number of eigenvalues, eigenvector stability,
multiplicity, and open dimensions enter the sibling law.

## Law Components

- `fixed_topology_fixed_subspace`: This is the implemented local fixed-subspace law; it is not by itself a selected-node law.
- `selected_topology_selected_node`: Required for edge-supported cases before extra alpha can be spent.
- `nnls_branch_length_scale`: Short selected branches sharpen W; long branches absorb contrast as branch-time noise.
- `local_covariance_eigensystem`: Covariance errors change the mass of W before topology selection is even considered.
- `multiplicity_projector`: Open dimensions should be block-stable; otherwise p-values depend on arbitrary eigenvector orientation.
- `adaptive_dimension_mixture`: Dimension adaptivity explains why the same Wald mass can move between significant and non-significant regimes.
- `topology_stability_alpha_spending`: A branch-supported sibling split still fails closed when its topology support is weak.

## Smaller Examples

- `balanced_short_branch_same_signal`: W `4.083`, p `0.04331`, note: Same contrast and fixed k=1; balanced size and short branch leave the largest W.
- `imbalanced_size_same_signal`: W `1.815`, p `0.1779`, note: Same contrast but larger 1/n_A+1/n_B; W drops because the null variance is wider.
- `long_nnls_branch_same_signal`: W `1.633`, p `0.2012`, note: Same contrast but longer conditioned branch time; W drops because branch-time noise absorbs contrast.
- `signal_aligned_with_selected_eigenvector`: W `7.84`, p `0.00511`, note: A spiked stable eigensystem keeps aligned contrast inside the accepted one-dimensional subspace.
- `signal_orthogonal_to_selected_eigenvector`: W `0`, p `1`, note: The same norm is invisible at k=1 when topology/projection select the wrong direction.
- `multiplicity_arbitrary_first_vector`: W `4`, p `0.0455`, note: A k=1 cut inside a tied eigenspace is not orientation-invariant.
- `multiplicity_rotated_first_vector`: W `2`, p `0.1573`, note: A rotation inside the same tied eigenspace changes k=1 W without changing the invariant signal norm.
- `multiplicity_full_projector`: W `4`, p `0.2615`, note: Testing the whole tied projector restores orientation invariance but changes degrees of freedom.
- `adaptive_energy_fraction_k2`: W `6.25`, p `0.04394`, note: An energy-fraction rule that accepts k=2 changes both W and the tail df; K must be conditioned on.
- `same_mass_fixed_k3`: W `6.29`, p `0.09832`, note: The same projected mass can become less significant when the accepted open dimension includes an extra null coordinate.

## Observed Occurrences

- `sibling_gate_closed_after_edge_open`: `5` case(s), required law `selected_node_sibling_null`, cases `cont_lowrank_pggn_shrinkage,overlap_extreme_4c,overlap_heavy_8c_large_feat,sbm_hard,sbm_moderate`.
- `edge_gate_closed_global`: `9` case(s), required law `selected_pipeline_edge_null`, cases `dim_consolidated_4c_24f_continuous,dim_consolidated_4c_72f_continuous,dim_diffuse_6c_136f_continuous,gauss_clear_medium_continuous,gauss_moderate_3c_continuous,gauss_outlier_cluster_4c_continuous,gauss_single_outlier_4c_continuous,mp_spike_above_bbp_continuous,phylo_brownian_null_16taxa`.

## Requirement Coverage

- `conditioned selected-node law`: covered by the conditional-law, fixed-subspace proposition, and selected probability mass sections.
- `topology and topology stability`: covered by `E_sel`, the `topology_stability_alpha_spending` component, and occurrence split.
- `NNLS branch lengths`: covered by the branch-time whitening scale and branch-length examples.
- `local covariance`: covered by `Sigma_v`, `L_v`, and the local-covariance eigensystem component.
- `eigenvalues/eigenvectors/multiplicity/open dimensions`: covered by the multiplicity-projector and adaptive-dimension-mixture sections.
- `experiment occurrences`: covered by the occurrence table joined from the topology-difference and traversal-trace reports.
- `smaller examples`: covered by the ten deterministic Wald-reaction rows.

## Interpretation

The overlap/SBM edge-supported failures are not solved by a single larger
global alpha. The missing object is a conditional tail probability for the
selected node. Branch lengths change the variance scale, covariance controls
the mass in each feature direction, eigenvalue multiplicity determines which
subspace is identifiable, accepted dimension changes the degrees of freedom,
and topology stability determines whether a selected split is eligible for
alpha at all.

## Sources

- `reports/tree_consensus_topology_difference_diagnosis_20260709/topology_difference_case_diagnosis.csv`
- `reports/tree_consensus_fail_closed_pvalues_20260709/fail_closed_traversal_trace.csv`
- `tree_break_selection/hierarchy_analysis/statistics/contrast_covariance.py`
- `tree_break_selection/hierarchy_analysis/statistics/projection/projected_wald/projected_wald_reference_distribution.py`
- `tree_break_selection/hierarchy_analysis/statistics/sibling_divergence/pair_testing/wald_statistic/sibling_divergence_test.py`
- `https://arxiv.org/abs/2012.02936`
- `https://arxiv.org/abs/1311.6238`
