# TBS edge-projection diagnostic for requested thresholds

Original generation timestamp: not recorded.
Provenance timestamp added at: 2026-06-24T20:22:42+02:00.

Requested TBS thresholds:

- sibling alpha: `0.01`
- edge alpha: `0.001`

## Original failure

The original TBS run returned one cluster because traversal stopped at the root.
The root sibling test saw a strong difference, but the edge gate used an
adaptive 2-dimensional projected edge statistic and blocked both child-parent
edges.

Original root traversal record:

- root node: `N4998`
- left child: `N4988`
- right child: `N4997`
- decision: `boundary`
- sibling p-value: `0.0`
- sibling statistic: `25053.918567354016`
- sibling degrees of freedom: `30`
- left child-parent edge p-value: `0.4390961140731243`
- left child-parent edge BH p-value: `0.4390961140731243`
- right child-parent edge p-value: `0.044176246562597124`
- right child-parent edge BH p-value: `0.08835249312519425`

Root child composition:

- Left child `N4988`: `10` cells; `8` macrophage and `2` ductal. Batch
  composition is `80%` batch `0` and `20%` batch `1`.
- Right child `N4997`: `2,490` cells; broad mixture dominated by alpha
  (`795`), beta (`633`), ductal (`338`), acinar (`258`), delta (`173`), and
  gamma (`108`). Batch composition is `64.3%` batch `0`, `15.9%` batch `1`,
  `15.9%` batch `2`, and `3.8%` batch `3`.

## Historical benchmark-only adjustment

Before the production fix, a benchmark-only replay set
`spectral_minimum_dimension = 20`, so the child-parent edge gate tested a
larger subspace than the original 2-dimensional adaptive edge statistic. This
kept the requested alpha levels unchanged and showed that the root collapse was
caused by the edge-subspace mismatch.

Edge-projection sensitivity at sibling alpha `0.01` and edge alpha `0.001`:

- minimum edge dimension `2`: `1` cluster, ARI `0.0000`, NMI `0.0000`
- minimum edge dimension `5`: `1` cluster, ARI `0.0000`, NMI `0.0000`
- minimum edge dimension `10`: `21` clusters, ARI `0.1148`, NMI `0.3796`
- minimum edge dimension `15`: `21` clusters, ARI `0.1148`, NMI `0.3796`
- minimum edge dimension `20`: `76` clusters, ARI `0.2710`, NMI `0.6263`
- minimum edge dimension `30`: `69` clusters, ARI `0.2605`, NMI `0.6069`

## Production correction

The production correction is stricter than the intermediate benchmark-only
workaround:

- fixed-coordinate sibling gates now raise the edge projection floor to the
  fixed feature-space dimension, which is `30` for this PCA benchmark;
- linkage/adaptive-diffusion branch lengths are topology/support diagnostics by
  default, not automatic variance time;
- normalized branch-length variance is available only through the explicit
  `edge_branch_length_variance_policy="normalized_branch_length"` sensitivity
  model.

With the production-compatible `30`-PC edge projection and sibling alpha `0.01`
/ edge alpha `0.001`, the rerun gives:

- topology-only TBS: `102` clusters, weighted purity `0.9368`,
  dominant-cluster recall `0.2852`, split error `0.7148`, V-measure `0.5806`;
- normalized branch-time TBS: `69` clusters, weighted purity `0.9076`,
  dominant-cluster recall `0.3392`, split error `0.6608`, V-measure `0.6069`;
- adaptive diffusion topology TBS: `72` clusters, weighted purity `0.9448`,
  dominant-cluster recall `0.3496`, split error `0.6504`, V-measure `0.6124`.

## Branch-time sensitivity

The benchmark now also scans candidate branch-time transforms on fixed TBS
topologies. This is supervised failure analysis, not a production clustering
rule: each transform rescales edge Wald statistics, reapplies Tree-BH, reruns
traversal, and is then scored against curated cell-type labels.

Best scanned rows:

- standardized-PCA topology with `linear_scale_2`: `55` clusters, weighted
  purity `0.9044`, dominant-cluster recall `0.4684`, split error `0.5316`,
  V-measure `0.6356`;
- adaptive diffusion topology with `quadratic_scale_1`: `37` clusters,
  weighted purity `0.9252`, dominant-cluster recall `0.3876`, split error
  `0.6124`, V-measure `0.6481`;
- standardized-PCA topology with stronger `linear_scale_4`: `33` clusters and
  dominant-cluster recall `0.6268`, but weighted purity collapses to `0.6696`.

This shows that branch-time transforms can trade over-splitting for merging,
but length scaling alone is not a justified unsupervised fix.

Interpretation:

The immediate failure was a mismatch between the edge gate's adaptive low-rank
projection and the 30-PC fixed-coordinate benchmark representation. Correcting
the edge projection contract opens traversal at the requested strict alphas, but
the result is still over-fragmented relative to curated pancreas cell types.
ARI is therefore a weak primary diagnostic: the useful signal is that TBS has
high cluster purity/homogeneity and poor cell-type consolidation/completeness.
