---
title: Selected Hierarchy Selection Geometry
type: analysis
status: reviewed
updated: 2026-06-02
sources:
  - wiki/analyses/oracle-gate-path-diagnostic.md
  - wiki/questions/open-mathematical-questions.md
  - wiki/sources/edge-selection-null-audit-20260601.md
  - wiki/sources/feature-split-selection-audit-20260601.md
  - wiki/sources/selected-hierarchy-null-audit-20260601.md
  - wiki/sources/selected-hierarchy-external-calibration-contract-20260602.md
  - wiki/sources/selected-hierarchy-stratification-diagnostic-20260602.md
  - benchmarks/diagnostics/calibration/selected_hierarchy_null_audit.py
  - kl_clustering_analysis/tree/distributions.py
  - kl_clustering_analysis/hierarchy_analysis/statistics/contrast_covariance.py
  - kl_clustering_analysis/hierarchy_analysis/statistics/projection/projected_wald/projected_wald_kernel.py
tags:
  - analysis
  - geometry
  - selection
  - calibration
---

# Selected Hierarchy Selection Geometry

## Summary

The current geometric explanation is that the fixed projected-Wald test is not
the main source of the calibration problem. In a fixed tree and fixed
projection, the method tests a contrast in null-whitened tangent coordinates
and compares a projected squared norm with a chi-square reference. The observed
calibration failure appears when the same feature matrix first selects the
hierarchy and then supplies the child-parent and sibling tests. Hierarchy
selection turns internal node distributions into selected empirical
barycenters, so child-parent edges and focal sibling pairs are conditioned on
having large geometric separation in the same coordinates being tested.

## Details

For a fixed node, fixed feature-space chart, and fixed projection, the local
test geometry is simple. Let \(x_A\) and \(x_B\) be two empirical node
distributions in the active feature-space coordinates, and let
\(\Sigma_0(A,B)\) be the null contrast covariance used by the feature-space
contract. The projected-Wald kernel forms a whitened contrast

\[
z_{A,B}=\Sigma_0(A,B)^{-1/2}(x_A-x_B)
\]

and then tests

\[
W_{A,B}(P)=\lVert Pz_{A,B}\rVert^2,
\qquad
PP^\top=I_k.
\]

Under a fixed-subspace isotropic null, \(W_{A,B}(P)\) has a
\(\chi^2_k\)-type reference. This is the geometry validated by the fixed
projected-Wald and selected-PCA diagnostics: the whitened tangent coordinates
and orthonormal projection are not enough by themselves to explain the very
large sibling inflation factors seen in unsupported high-dimensional contexts.

The hierarchy adds a second geometric object. KL-TE builds a sample-leaf tree
from the same feature matrix that later supplies the tests. Each internal node
distribution is the leaf-count-weighted empirical barycenter of its descendant
leaves. If \(D(v)\) is the descendant leaf set under node \(v\), then
schematically

\[
\bar x_v=\frac{1}{|D(v)|}\sum_{i\in D(v)}x_i.
\]

The child-parent contrast at a selected edge is therefore not a generic null
contrast between independently named groups. It is the contrast between a
selected descendant subset and the selected parent barycenter:

\[
z_{c,v}
=
\Sigma_0(c,v)^{-1/2}(\bar x_c-\bar x_v),
\qquad
c\prec v,
\]

where both \(c\) and \(v\) are functions of the same observed feature matrix.
Under a global null, agglomerative hierarchy construction still orders leaves
by empirical noise. The selected children are geometrically separated because
the tree construction found them as separated subsets. Testing those selected
edges as if the tree were fixed conditions on an extreme-search event but uses
the fixed-tree reference.

This explains the edge-selection audit. In pure null data, same-data hierarchy
construction plus edge testing opens nearly all child-parent edges, while
fixed-tree feature permutations do not. In geometric terms, the selected tree
has already rotated the problem toward observed empirical separation. The edge
test then measures that selected separation in the same null-whitened tangent
space.

The sibling calibration problem is downstream of this geometry. The internal
empirical-null support set for focal sibling context \(u\) is

\[
\mathcal C_{0,u}
=
\{q: S^{\mathrm{edge}}_{l(q)}=0
\ \text{and}\
S^{\mathrm{edge}}_{r(q)}=0\}.
\]

When same-data edge selection opens almost every child-parent edge, this set
can have no positive local calibration mass. Positive-weight records then come
only from selected non-null-like contexts. Treating those records as empirical
null support would be mathematically dishonest because they are selected
high-contrast barycentric splits, not null sibling examples.

The selected-hierarchy null diagnostic estimates a closer geometric object:

\[
\mathcal L_0\!\left(
W_u
\mid
T=T(X),\
\text{edge path selected},\
u\ \text{selected as focal sibling context}
\right),
\]

implemented diagnostically by regenerating null feature matrices, rebuilding
the hierarchy inside each replicate, rerunning edge tests, and collecting
selected focal sibling statistics. Its first representative run produced
selected-hierarchy correction factors in the tens, not near one. This supports
the explanation that the missing correction is a hierarchy-selection geometry
effect rather than a failure of the projected-Wald kernel alone.
A stricter rerun requiring an open child-parent edge path and matching
projection dimension, parent size, and parent depth keeps the same qualitative
root conclusion. The non-root rerun is informative but incomplete: Gaussian
targets have matched selected-null support, while the binary and categorical
non-root targets have no matched selected-null records at 100 replicates under
that strict context. These unsupported rows describe the phenomenon that the
diagnostic support is itself sparse; they are not fallback calibration
decisions. The same rerun also shows why a production external calibration
contract needs an explicit precision target: the estimated scale \(c\) is
stable to roughly `3%`--`10%` relative simulation standard error in matched
rows, but the empirical-tail resolution is only about `0.011`--`0.026`, which
is too coarse for an \(\alpha=0.01\) tail decision.

The 500-replicate descriptive precision run strengthens the geometric
description without creating a method fallback. Root strict-context rows have
hundreds of matched simulations and \(c\) estimates in the tens. Non-root
Gaussian strict-context rows also have matched support. Non-root binary and
categorical rows expose the geometry of support sparsity: exact depth matching
can remove most or all selected-null support, while relaxing depth and then
parent-size restores support and still leaves \(c\) in the tens. This means
that support failure is partly a context-matching geometry problem, not proof
that the selected-hierarchy phenomenon is absent.

The 2026-06-02 stratification diagnostic further separates depth from parent
size. Parent-size bins show that small selected parent nodes often have much
higher selected-null scale than root-like selected nodes. Exact depth is
therefore a sparse descriptive coordinate, not yet a validated exact
conditioning variable.

The same diagnostic characterizes the selected-ratio distribution
\(R=W/(a\nu)\). Across reliable parent-size rows, \(R\) has q95 values in the
tens to low hundreds and the unconditioned projected-Wald reference rejects
nearly all selected-null records in most rows. Geometrically, the same
coordinates used to construct and open the hierarchy have already selected
large projected contrasts before the sibling statistic is interpreted. This is
why the fixed-subspace projected-Wald reference remains mathematically clean
but insufficient for the selected hierarchy without an explicit conditional
law.

The external calibration contract diagnostic sharpens this conclusion. With a
strict production tail-resolution target, no current 500-replicate stratum is
admissible as an external production calibration object. Scalar mean scaling is
also not a calibrated law shape: in reliable rows it gives zero rejection at
\(\alpha_{\mathrm{sib}}=0.01\), but its p-values are not uniform. The selected
law is therefore not just "chi-square times a constant"; it is a selected-ratio
tail distribution that would need its own validated support contract.

Feature-split cross-fit is useful evidence for the same reason, although it is
not the chosen production method. When one feature block selects the tree and
a held-out feature block supplies node distributions and tests, the
tree-selection noise is no longer the same noise being tested. In the null
case, edge rejection collapses and internal sibling calibration support
returns. Geometrically, cross-fit breaks the alignment between selected
barycentric separation and the tested tangent coordinates.

The current method therefore has three distinct geometric regimes:

1. Fixed tree, fixed projection: projected-Wald geometry is clean and locally
   chi-square under the stated null assumptions.
2. Fixed or selected projection inside fixed membership: selected-PCA
   validation supports the leaf-only inferential spectral basis in tested
   Gaussian settings.
3. Same-data selected hierarchy: node barycenters, edge openings, and focal
   sibling contexts are conditioned on a high-contrast search over the same
   feature matrix. This is the unresolved selection-conditioned geometry.

The practical method rule follows from that split. Internal empirical-null
calibration is valid only when strict or stopped-edge null-like support exists.
If the local support consists only of selected non-null contexts, production
must fail closed unless a separately validated selected-hierarchy calibration
model is provided.

## Evidence

- `wiki/sources/edge-selection-null-audit-20260601.md` records that same-data
  hierarchy construction opens about `99%` of tested child-parent null edges,
  while fixed-tree permutations have median rejection rate `0.0`.
- `wiki/sources/feature-split-selection-audit-20260601.md` records that
  feature-split selection/testing restores null support in the null case and
  preserves signal in the representative signal cases.
- `wiki/sources/selected-hierarchy-null-audit-20260601.md` records the first
  selected-hierarchy null run, the richer 100-replicate root/non-root reruns,
  and the 500-replicate descriptive precision/context-relaxation study, where
  selected-hierarchy correction factors are large enough to block the null
  case but still allow matched signal examples.
- `wiki/sources/selected-hierarchy-stratification-diagnostic-20260602.md`
  records the depth and parent-size stratification study showing that small
  selected parent nodes generally have larger selected-null scale and that the
  selected-ratio law has large upper quantiles under same-data hierarchy
  selection.
- `wiki/sources/selected-hierarchy-external-calibration-contract-20260602.md`
  records the explicit production-admissibility thresholds and the
  scalar-vs-tail diagnostic.
- `wiki/analyses/oracle-gate-path-diagnostic.md` records the support-status
  contract and the distinction between invalid selected-non-null calibration
  support and valid empirical-null support.
- `kl_clustering_analysis/tree/distributions.py` implements internal node
  distributions as empirical subtree barycenters.
- `kl_clustering_analysis/hierarchy_analysis/statistics/contrast_covariance.py`
  implements the null-whitened tangent maps used by edge, sibling, and spectral
  calculations.
- `kl_clustering_analysis/hierarchy_analysis/statistics/projection/projected_wald/projected_wald_kernel.py`
  implements the shared projected-Wald kernel.

## Links

- [[oracle-gate-path-diagnostic]]
- [[open-mathematical-questions]]
- [[edge-selection-null-audit-20260601]]
- [[feature-split-selection-audit-20260601]]
- [[selected-hierarchy-null-audit-20260601]]
- [[selected-hierarchy-null-support-contract]]
- [[projected-wald-statistic]]
- [[local-marchenko-pastur-rule]]
- [[selected-pca-projected-wald-validation]]

## Open Questions

- What exact conditioning event defines the production selected-hierarchy
  sibling null: full hierarchy reconstruction, fixed observed tree, ancestor
  Tree-BH path, focal sibling selection, or all of these?
- Can the selected-hierarchy correction be derived analytically, or must it be
  simulated as an external conditional null?
- How should continuous selected-hierarchy null data be generated without
  reintroducing an unvalidated covariance model?
- What context variables must be matched for a selected-hierarchy calibration
  law: \(n_L,n_R,p,k_u,a_u,\nu_u\), feature family, tree depth, edge path, or
  local barycentric separation?
- What precision target should be predeclared for external calibration:
  relative standard error of \(c\), empirical-tail resolution near
  \(\alpha_{\mathrm{sib}}\), or both?
- Should non-root selected-hierarchy matching condition exactly on depth, or is
  depth a descriptive stratifier whose exact matching makes support too sparse?
- When selected-hierarchy calibration is available, should it replace internal
  empirical-null calibration or remain a separate diagnostic mode?
