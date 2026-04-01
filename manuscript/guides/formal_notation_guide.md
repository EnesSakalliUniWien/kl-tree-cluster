# Child-Wise Sibling Null Prior Interpolation

This guide formalizes the child-wise null prior interpolation method used in the adjusted sibling divergence pipeline. The relevant implementation lives in [kl_clustering_analysis/hierarchy_analysis/statistics/sibling_divergence/pair_testing/sibling_null_prior_interpolation/child_prior_estimation.py](/Users/berksakalli/Projects/kl-te-cluster/kl_clustering_analysis/hierarchy_analysis/statistics/sibling_divergence/pair_testing/sibling_null_prior_interpolation/child_prior_estimation.py), with helper kernels in [kl_clustering_analysis/hierarchy_analysis/statistics/sibling_divergence/pair_testing/sibling_null_prior_interpolation/kernel_interpolation.py](/Users/berksakalli/Projects/kl-te-cluster/kl_clustering_analysis/hierarchy_analysis/statistics/sibling_divergence/pair_testing/sibling_null_prior_interpolation/kernel_interpolation.py).

## Purpose

The interpolation method estimates how null-like each child of a sibling pair appears when the direct Gate 2 child-parent edge evidence is unavailable, blocked, or too crude to use by itself. The output is not a Bayesian prior distribution. It is an empirical calibration weight in $[0,1]$ that says how strongly a child should contribute to the null-like reference pool used for post-selection inflation correction in the sibling Wald test.

The baseline pair-level null-likeness score is the conservative edge-based rule

$$
\pi_u^{\mathrm{edge}} = \min(q_\ell, q_r),
$$

where $u = (\ell, r)$ is a sibling pair and $q_\ell, q_r$ are the BH-corrected Gate 2 child-parent p-values for the left and right child. The interpolation layer replaces this crude estimate with a child-wise smoothed estimate whenever stopping-edge context is available.

## Notation

| Symbol | Meaning |
| --- | --- |
| $u = (\ell, r)$ | A sibling pair with left child $\ell$ and right child $r$ |
| $p(u)$ | Parent node of sibling pair $u$ |
| $T_u$ | Raw sibling Wald statistic for pair $u$ |
| $k_u$ | Degrees of freedom for the sibling test |
| $q_c$ | BH-corrected Gate 2 child-parent p-value for child $c$ |
| $a(c)$ | Nearest stopping edge recovered above child $c$ |
| $q_{a(c)}$ | P-value attached to the stopping edge above $c$ |
| $d_T(i,j)$ | Tree distance between nodes $i$ and $j$ |
| $d_a(c)$ | Distance from child $c$ to its stopping edge |
| $\mathcal{S}$ | Stable reference set: tested, non-significant Gate 2 edges |
| $\mathcal{G}$ | Signal reference set: tested, significant Gate 2 edges |
| $\kappa_c$ | Structural scale assigned to child $c$ |
| $x_c = \log(\max(\kappa_c, 1))$ | Log structural scale for child $c$ |
| $\tau_b$ | Ancestor-distance bandwidth |
| $\tau_t$ | Stable-neighbor tree-distance bandwidth |
| $\tau_s$ | Signal-neighbor tree-distance bandwidth |
| $h_k$ | Structural matching bandwidth on the log-scale axis |
| $\pi_c$ | Final child-wise null-likeness estimate |
| $\pi_u$ | Final pair-wise null-likeness weight |

## Structural Scale

Each child receives a local structural scale used to match it to comparable stable neighbors. The implementation uses the available Gate 2 edge projection dimension if possible, otherwise the Gate 2 edge degrees of freedom, and otherwise falls back to $1$:

$$
\kappa_c =
\begin{cases}
k_c^{\mathrm{edge}}, & \text{if a positive edge projection dimension is available}, \\
df_c^{\mathrm{edge}}, & \text{if no projection dimension is available but a positive edge df exists}, \\
1, & \text{otherwise.}
\end{cases}
$$

The value actually fed into the kernel is

$$
x_c = \log(\max(\kappa_c, 1)).
$$

This log transform keeps the matching rule stable across children with very different effective dimensions.

## Adaptive Bandwidths

The method computes bandwidths directly from the current tree and the available stopping-edge metadata. Let the recovered stopping-edge distance for a blocked child be $d_a(c)$. Then the ancestor bandwidth is the median stopping-edge distance across children with stopping-edge context:

$$
	au_b = \operatorname{median}\{d_a(c)\}.
$$

For stable and signal references, the tree-distance bandwidths are defined from nearest-neighbor distances:

$$
	au_t = \operatorname{median}_i \min_{s \in \mathcal{S}} d_T(i,s),
\qquad
	au_s = \operatorname{median}_i \min_{g \in \mathcal{G}} d_T(i,g).
$$

The structural matching bandwidth is the sample standard deviation of the stable-reference log scales:

$$
h_k = \operatorname{sd}\{x_s : s \in \mathcal{S}\}.
$$

When the sample is degenerate, the implementation falls back to numerically safe defaults rather than allowing division by zero in the kernels.

## Ancestor Trust Weight

If child $c$ has a stopping edge above it, the ancestor contribution is discounted exponentially by distance:

$$
w_a(c) = \exp\!\left(-\frac{\max(d_a(c)-1, 0)}{\tau_b}\right).
$$

This gives weight near $1$ when the stopping edge is very close and smaller weight as the stopping edge moves farther away.

## Stable-Neighbor Interpolation

For each stable reference edge $s \in \mathcal{S}$, the method forms a product kernel with one factor on the tree and one factor on structural scale:

$$
w_s(c) = \exp\!\left(-\frac{d_T(c,s)}{\tau_t}\right) K_k(x_s, x_c),
$$

where the structural matching kernel is Gaussian when $h_k > 0$,

$$
K_k(x_s, x_c) = \exp\!\left(-\frac{1}{2}\left(\frac{x_s - x_c}{h_k}\right)^2\right),
$$

and reduces to an exact-match indicator when $h_k = 0$.

The total trusted stable-neighbor mass is

$$
W_{\mathcal{S}}(c) = \sum_{s \in \mathcal{S}} w_s(c).
$$

The stable-neighbor estimate of the child null p-value is then

$$
\hat{q}_{\mathcal{S}}(c) =
\frac{\sum_{s \in \mathcal{S}} w_s(c) q_s}{W_{\mathcal{S}}(c)}
\quad \text{if } W_{\mathcal{S}}(c) > 0.
$$

If no trusted stable neighbors are found, the implementation falls back to the ancestor p-value $q_{a(c)}$.

## Ancestor-Neighbor Blending

The ancestor and neighborhood estimates are combined by a convex interpolation:

$$
\hat{q}_{\mathrm{mix}}(c) =
\frac{w_a(c) q_{a(c)} + W_{\mathcal{S}}(c) \hat{q}_{\mathcal{S}}(c)}{w_a(c) + W_{\mathcal{S}}(c)}.
$$

This quantity says how null-like the child would appear if we used only ancestor support and nearby stable references.

## Signal Suppression

That mixed estimate is then suppressed when nearby significant Gate 2 edges suggest that the local region is signal-rich. The signal penalty is defined as

$$
\gamma(c) =
\max_{g \in \mathcal{G}}
\left[(1 - q_g) \exp\!\left(-\frac{d_T(c,g)}{\tau_s}\right)\right].
$$

This is large when there is a nearby edge with very small Gate 2 p-value. The final child-wise null-likeness estimate is

$$
\pi_c = \operatorname{clip}\left(\hat{q}_{\mathrm{mix}}(c) (1 - \gamma(c)), 0, 1\right).
$$

The returned diagnostics track the mixed neighborhood estimate, the ancestor support, and the fraction of evidence attributable to the neighborhood term:

$$
\omega_{\mathrm{nbr}}(c) =
\frac{W_{\mathcal{S}}(c)}{w_a(c) + W_{\mathcal{S}}(c)}.
$$

## Pair Aggregation

The pair-level null-likeness weight is conservative. After estimating both children, the sibling pair receives

$$
\pi_u = \min(\pi_\ell, \pi_r).
$$

The same conservative reduction is used for the pair-level diagnostics written back to the record: the smoothed neighborhood estimate is the minimum across children, the ancestor support is the minimum across children, and the neighborhood reliance is the maximum across children.

## Methodological Interpretation

Methodologically, this interpolation layer exists because the sibling divergence calibration step needs a reliable set of null-like examples, but not every child arrives with a clean directly tested Gate 2 edge. Some children are blocked by ancestor decisions, some are skipped, and some have weakly informative edge-level p-values. In those cases the method reconstructs a local null-likeness score by borrowing information from the nearest stopping edge and from nearby stable edges in the tree, while simultaneously discounting the estimate when nearby significant edges indicate local signal.

The method is therefore local, nonparametric, and conservative. It is local because every estimate depends on nearby nodes and nearby stopping edges rather than on a global regression alone. It is nonparametric because the key interpolation step is a kernel smoother over tree distance and structural scale rather than a rigid parametric model. It is conservative because the final pair weight is the minimum of the two child estimates, which prevents a sibling pair from being treated as strongly null-like unless both children look null-like.

## Downstream Role in the Adjusted Sibling Wald Test

These child-wise and pair-wise weights are used downstream to estimate the post-selection inflation factor for the sibling Wald statistic. For each sibling pair $u$, define the raw inflation ratio

$$
r_u = \frac{T_u}{k_u}.
$$

The pair weight $\pi_u$ controls how much that pair contributes to the weighted inflation estimate. The global estimate takes the weighted mean of these ratios and clips the result into a safe range:

$$
\hat{c}_{\mathrm{global}} = \operatorname{clip}\left(
\frac{\sum_u \pi_u r_u}{\sum_u \pi_u},
1,
\max_u r_u
\right).
$$

The calibrated deflation factor is then used to form the adjusted sibling statistic

$$
T_u^{\mathrm{adj}} = \frac{T_u}{\hat{c}(u)},
\qquad
p_u^{\mathrm{adj}} = \Pr\left(\chi^2_{k_u} \ge T_u^{\mathrm{adj}}\right),
$$

where $\hat{c}(u)$ is either the global estimate or the locally smoothed estimate used by the conditional deflation layer.

In short, the child-wise interpolation method does not itself decide whether siblings differ. Instead, it estimates how much confidence the calibration step should place in each pair as a null-like example. That makes it a critical component of the sibling divergence test even though it sits one layer upstream from the final adjusted p-value.
