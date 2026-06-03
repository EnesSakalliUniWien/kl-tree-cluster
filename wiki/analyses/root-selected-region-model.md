---
title: Root Selected Region Model
type: analysis
status: reviewed
updated: 2026-06-03
sources:
  - wiki/analyses/method-proof-web.md
  - wiki/analyses/selected-hierarchy-selection-geometry.md
  - wiki/analyses/selected-hierarchy-geometric-law-map.md
  - wiki/sources/edge-selection-null-audit-20260601.md
  - wiki/sources/selected-hierarchy-null-audit-20260601.md
  - wiki/sources/selected-ratio-tail-law-diagnostic-20260602.md
  - wiki/sources/root-selected-region-margins-20260603.md
  - benchmarks/diagnostics/calibration/selected_hierarchy_geometry_covariates.py
  - benchmarks/diagnostics/calibration/root_selected_region_margins.py
  - benchmarks/shared/kl_tree_context.py
tags:
  - analysis
  - selection
  - geometry
  - proof
---

# Root Selected Region Model

## Summary

This page defines the first tractable selected-region object for KL-TE: a root
sibling context with a fixed hierarchy-construction procedure, fixed feature
chart, fixed projected-Wald kernel, no sibling FDR, and no traversal. It does
not solve the full production selected-hierarchy law. It defines the event
that must be conditioned on before a fixed-object Wald reference can be used
honestly after same-data hierarchy selection.

The target object is
\[
\mathcal L\!\left(R_\rho\mid X\in\mathcal S_\rho\right),
\qquad
R_\rho=\frac{W_\rho}{a_\rho\nu_\rho},
\]
where \(\rho\) is the root parent, \(W_\rho\) is the raw root sibling
projected-Wald statistic, and \(\mathcal S_\rho\) is the event that the same
data select the observed root split and open the child-parent edge path.

## Details

### Simplified Setting

Fix the following objects before analysis:

1. a feature-space chart and null covariance contract;
2. a deterministic hierarchy-construction algorithm \(H\);
3. a linkage score \(D_X(A,B)\) for every candidate pair of current clusters;
4. a deterministic tie-breaking rule;
5. an edge alpha \(\alpha_{\mathrm{edge}}\);
6. a fixed projected-Wald edge and sibling kernel.

Ignore sibling FDR, empirical inflation, pass-through traversal, and downstream
non-root focal selection. The only selection target is the observed root split
\[
\rho \to (L_\rho,R_\rho).
\]

### Hierarchy Selection Event

Agglomerative hierarchy construction is a sequence of merges. At merge step
\(t\), let \(\mathcal P_t(X)\) be the set of available cluster pairs and let
\((A_t,B_t)\) be the pair merged by the observed run. With deterministic
tie-breaking, the event that the same merge sequence occurs is
\[
\mathcal H_\rho
=
\bigcap_t
\bigcap_{(A,B)\in\mathcal P_t(X)\setminus\{(A_t,B_t)\}}
\left\{
D_X(A_t,B_t)\le D_X(A,B)
\right\}.
\]

This is already a selected region in data space. The sets
\(\mathcal P_t(X)\) are determined recursively by earlier merge inequalities,
but for a fixed observed merge sequence they become fixed candidate sets
inside that sequence cell.

For Euclidean average linkage, each linkage score is an average of squared
pairwise distances or Euclidean distances. In the squared-distance form,
\(D_X(A,B)\) is quadratic in \(X\), so each merge comparison is a quadratic
inequality. Therefore the fixed-sequence root-selection region is
semi-algebraic:
\[
\mathcal H_\rho
=
\{X:\ g_i(X)\le 0,\ i=1,\ldots,m_H\}
\]
for finitely many polynomial inequalities \(g_i\). For Hamming/binary
distances, the region is piecewise linear after fixing the sign/coordinate
cells of the absolute-value terms.

### Edge-Opening Event

Let \(Q_{\rho\to L}(X)\) and \(Q_{\rho\to R}(X)\) be the child-parent
projected-Wald edge statistics for the two root children, with corresponding
reference thresholds \(q_{L,\alpha}\) and \(q_{R,\alpha}\). The root edge-path
opening event is
\[
\mathcal E_\rho
=
\left\{
Q_{\rho\to L}(X)\ge q_{L,\alpha}
\right\}
\cap
\left\{
Q_{\rho\to R}(X)\ge q_{R,\alpha}
\right\},
\]
or the Tree-BH-adjusted analogue when the root path is embedded in the full
tree correction. In this simplified root model, the unadjusted form is the
minimal conditioning object.

Each \(Q\) is a quadratic form in the null-whitened child-parent contrast when
the projection is fixed. When the projection is selected from local spectral
rows, the event also includes the spectral-selection cell that fixes the
chosen projection dimension and basis.

### Root Sibling Selected Region

The root selected region is
\[
\mathcal S_\rho=\mathcal H_\rho\cap\mathcal E_\rho.
\]

The correct root selected law is therefore
\[
\mathcal L\!\left(
W_\rho
\mid
X\in \mathcal H_\rho\cap\mathcal E_\rho
\right),
\]
not the unconditional fixed-object law.

### Proposition: The Root Selected Law Is Generally Not Chi-Square

Assume the fixed-object root sibling statistic satisfies
\[
W_\rho\sim\chi^2_k
\]
when the root split, edge path, and projection are fixed independently of the
tested data. If \(\mathcal S_\rho\) depends on the same data and has nonzero
association with \(W_\rho\), then
\[
\mathcal L(W_\rho\mid X\in\mathcal S_\rho)
\ne
\chi^2_k
\]
in general.

Proof. This is an application of the conditioning counterexample in
[[method-proof-web]]. Since \(\mathcal E_\rho\) contains lower-bound
constraints on child-parent Wald energies, and \(\mathcal H_\rho\) selects
clusters by small between-cluster linkage or large induced barycentric
separation, the event is a nontrivial function of the same empirical
coordinates that enter \(W_\rho\). Unless the event is independent of
\(W_\rho\), conditioning changes the law. Independence is not implied by the
method construction and is contradicted by the edge-selection null audit.

### Differential-Geometric Objects

Inside a fixed smooth sequence cell, write
\[
\mathcal S_\rho=\{X:\ g_i(X)\le 0,\ i=1,\ldots,m\}.
\]
At an observed point \(x\), the active set is
\[
\mathcal A(x)=\{i:\ g_i(x)=0\}.
\]
The tangent cone is
\[
T_{\mathcal S_\rho}(x)
=
\{h:\ \nabla g_i(x)^\top h\le 0\ \text{for all } i\in\mathcal A(x)\}.
\]
The signed distance in the null-whitened metric is
\[
d_{\mathcal S_\rho}(x)
=
\inf_{y\in\partial\mathcal S_\rho}
\left\|
\Sigma_0^{-1/2}(x-y)
\right\|.
\]
Curvature is carried by the Hessians \(\nabla^2 g_i(x)\) of active smooth
constraints after projecting onto the local tangent boundary. These are the
differential-geometric objects suggested by the selected-inference literature:
selected region, boundary, signed distance/action, tangent cone, and curvature.

The current diagnostic proxy
\[
A_\rho=-\log p_{\min,\mathrm{edge}}
\]
is therefore best interpreted as a large-deviation or signed-distance proxy,
not as the signed distance itself.

### First-Order Merge-Boundary Law

For a fixed merge step, write the selected average-linkage pair as
\((A_t,B_t)\) and the nearest competitor pair as \((C_t,D_t)\). The observed
merge inequality is
\[
g_t(X)
=
D_X(A_t,B_t)-D_X(C_t,D_t)
\le 0.
\]
The observed score-space margin is
\[
m_t(X)
=
D_X(C_t,D_t)-D_X(A_t,B_t)
=
-g_t(X).
\]
When \(D_X\) is average Euclidean linkage and all involved pairwise distances
are nonzero, this cell is smooth. The first-order signed distance from the
observed point \(x\) to the nearest merge boundary in the ambient Euclidean
data metric is
\[
d_t^{(1)}(x)
=
\frac{m_t(x)}{\|\nabla g_t(x)\|_2}.
\]
This is not yet the null-whitened selected-region distance and not a
calibration law. It is the first local geometric object that can be computed
from the current tree builder without changing production inference.

For continuous Euclidean cases with independent leaves and root
empirical-Gaussian feature covariance \(\widehat\Sigma_\rho\), schema `v3`
also records the first-order null-whitened scale
\[
\sigma_{g,t}^2
=
\nabla g_t(x)^\top
\left(I_n\otimes\widehat\Sigma_\rho\right)
\nabla g_t(x),
\]
and the corresponding diagnostic distance
\[
d_{\Sigma,t}^{(1)}(x)
=
\frac{m_t(x)}{\sigma_{g,t}}.
\]
This is still a diagnostic geometry coordinate. It is not a production
selected-tail law because it conditions only on the local merge inequality and
does not include edge-opening, spectral selection, sibling FDR, or traversal.

For Hamming/discrete metrics, or when the selected merge is tied with another
minimum pair, the smooth formula is not the right object. The local selected
region is a discrete or nonsmooth tie cell. In that regime the next law is not
a curvature correction to \(d_t^{(1)}\); it is a discrete selected-region law
or a polyhedral/tie-breaking tangent-cone model.

### Connection To Existing Diagnostics

The selected-hierarchy null audit simulates from the conditional region
approximately by regenerating null data and reapplying \(H\) and the edge
tests. The geometry covariate diagnostic then records proxies for the
mathematical objects above:

- edge action as a signed-distance or large-deviation proxy;
- eigenvalue excess and effective rank as local spectral-mode descriptors;
- \(\cos^2\theta\) as selected-subspace angular alignment;
- parent size and child balance as finite-sample fluctuation descriptors.

The next proof-level development is not to add another scalar correction. It
is to decide whether these proxies can be replaced by actual functions of
\(\mathcal S_\rho\): active constraints, distances to boundaries, tangent
cones, and curvature terms.

### Observed Root Margin Diagnostic

`benchmarks/diagnostics/calibration/root_selected_region_margins.py` now
extracts one concrete part of \(\mathcal H_\rho\): for each average-linkage
merge step, it checks that the observed merge is a minimum active pair and
records the nearest-competitor margin. The final root merge has no competitor,
so the diagnostic summarizes the inequalities that construct the two root
child clusters.

The 2026-06-03 representative run separates two geometries. In Hamming,
discretized, and categorical cases, many root-child construction merges are
exact or numerical ties, and schema `v2` classifies those rows as requiring
discrete tie-cell geometry. In the continuous Euclidean diffuse representative,
all `178` root-child construction constraints have smooth first-order
geometry. The minimum merge margin is about `4.05e-4`, and the minimum
first-order signed distance after gradient normalization is about `2.24e-4`.
This means the selected-region boundary is not one uniform object: tie-heavy
discrete hierarchy cells and positive-margin continuous cells should not be
collapsed into a single scalar selected-tail explanation.

The same run records large root selected sibling ratios in all representatives
(`32.7` to `755`). Therefore merge margins alone do not explain the selected
tail. They are one component of the selected region and must be read together
with edge-opening strength, spectral excess, feature family, and covariance
geometry. Curvature remains explicit missing work: the continuous diagnostic
marks the high-dimensional Hessian operator as not materialized.

The supported eight-case continuous rerun makes that limitation sharper. Raw
merge margin, ambient first-order distance, and null-whitened first-order
merge distance have weak descriptive relationships with the log root sibling
selected ratio. The edge-action proxy has Spearman correlation `1.0` in this
small panel. Therefore the next selected-region object should be the
edge-opening boundary distance/action, not another adjustment to merge-margin
geometry.

### Minimal Diagnostic Contract

A focused root selected-region diagnostic should record, for each selected
root context:

```text
root merge sequence id
root child leaf sets
root sibling W
root selected ratio R
edge action proxy
active or near-active merge inequalities
smooth first-order signed distances or discrete tie-cell status
null-whitened first-order merge distances for continuous Euclidean cells
edge-statistic margins to threshold
lambda_k / lambda_plus
selected eigenvalue mass
effective rank
cos^2 theta
parent size
child balance
diagnostic status
```

The diagnostic remains descriptive unless it can estimate the conditional tail
\[
\Pr(R_\rho\ge r\mid X\in\mathcal S_\rho)
\]
with a declared support and precision contract.

## Evidence

- [[method-proof-web]] records the fixed-object projected-Wald proof and the
  selected-region proof gap.
- [[selected-hierarchy-selection-geometry]] records why same-data hierarchy
  selection creates selected barycentric contrasts.
- [[selected-hierarchy-geometric-law-map]] records the current edge-action,
  spectral, angular, parent-size, branch-length, and barycentric variables.
- `wiki/sources/edge-selection-null-audit-20260601.md` records that same-data
  hierarchy selection opens nearly all tested null edges, while fixed-tree
  permutations do not.
- `wiki/sources/selected-hierarchy-null-audit-20260601.md` records selected
  root and non-root null simulations with large selected-hierarchy ratios.
- `benchmarks/diagnostics/calibration/selected_hierarchy_geometry_covariates.py`
  records the current diagnostic proxy variables.
- [[root-selected-region-margins-20260603]] records the first concrete replay
  of root merge-selection inequalities and their margins for representative
  benchmark contexts.

## Links

- [[method-proof-web]]
- [[selected-hierarchy-selection-geometry]]
- [[selected-hierarchy-geometric-law-map]]
- [[selected-hierarchy-null-support-contract]]
- [[selected-ratio-tail-law-diagnostic-20260602]]
- [[root-selected-region-margins-20260603]]
- [[open-mathematical-questions]]

## Open Questions

- For the current Hamming and Euclidean benchmark metrics, which constraints
  dominate the null-whitened distance from \(x\) to
  \(\partial\mathcal S_\rho\)?
- How should exact/tie-heavy Hamming cells be represented geometrically:
  tangent cones of polyhedral cells, random tie-breaking cells, or a discrete
  selected-region law?
- Is edge action asymptotically equivalent to signed distance for selected
  edge opening?
- Does the tangent cone explain the selected-ratio tail better than the
  current edge-plus-spectral proxy equation?
- Can this root model be extended to non-root focal sibling contexts without
  making exact depth matching too sparse?
