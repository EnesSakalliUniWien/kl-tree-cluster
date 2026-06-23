---
title: Method Proof Web
type: analysis
status: reviewed
updated: 2026-06-03
sources:
  - manuscript/guides/full_method_logic_map.md
  - manuscript/guides/edge_sibling_derivation_guide.md
  - manuscript/sections/method/edge_test.tex
  - manuscript/sections/method/sibling_test.tex
  - wiki/concepts/projected-wald-statistic.md
  - wiki/analyses/selected-hierarchy-selection-geometry.md
  - wiki/analyses/selected-hierarchy-geometric-law-map.md
  - wiki/analyses/local-marchenko-pastur-rule.md
  - wiki/analyses/selected-pca-projected-wald-validation.md
  - wiki/sources/edge-selection-null-audit-20260601.md
  - wiki/sources/selected-hierarchy-null-audit-20260601.md
  - wiki/sources/selected-ratio-tail-law-diagnostic-20260602.md
  - wiki/sources/local-mp-identity-law-diagnostic-20260602.md
  - wiki/sources/hierarchy-gate-separation-20260603.md
tags:
  - analysis
  - proof
  - method
  - selection
  - geometry
---

# Method Proof Web

## Summary

The method has a clean proof spine for fixed objects and an explicit proof gap
for same-data selected hierarchies. The fixed-object spine is:

```text
feature-space covariance -> null-whitened contrast -> fixed projection
-> projected-Wald chi-square reference -> local edge/sibling p-values
```

The selected-hierarchy problem begins when the same feature matrix first
chooses the tree and then supplies the tested node distributions. The correct
conditional object is
\[
\mathcal L\!\left(W_u\mid X\in\mathcal S_u\right),
\]
where \(\mathcal S_u\) is the selected region defined by hierarchy
construction, edge-path opening, and focal sibling-context selection. The
repository currently has diagnostics and candidate geometry for this object,
not a production theorem.

## Details

### Definitions

Let \(X=(X_1,\ldots,X_n)\) be the feature matrix in a declared feature-space
chart. For a node \(v\), let \(D(v)\) be its descendant leaves and
\(\hat\mu_v\) the empirical feature distribution or mean attached to that
subtree. For a contrast between two node distributions, write
\[
\Delta_{A,B}=\hat\mu_A-\hat\mu_B,\qquad
z_{A,B}=\Sigma_0(A,B)^{-1/2}\Delta_{A,B},
\]
where \(\Sigma_0(A,B)\) is the null contrast covariance from the feature-space
contract. For an orthonormal projection \(P\in\mathbb R^{k\times d}\),
\[
W_{A,B}(P)=\lVert Pz_{A,B}\rVert^2 .
\]

The selected-hierarchy sibling problem for a focal parent \(u\) is represented
by
\[
R_u=\frac{W_u}{a_u\nu_u},
\]
where \(a_u\) and \(\nu_u\) are the analytic scale and degrees of freedom
reported by the projected-Wald kernel. In the current orthonormal raw
reference, \(a_u=1\) and \(\nu_u=k_u\) before empirical inflation.

### Lemma 1: Fixed-Projection Wald Law

If \(z\sim N(0,I_d)\) and \(P\in\mathbb R^{k\times d}\) is fixed with
\(PP^\top=I_k\), then
\[
W(P)=\lVert Pz\rVert^2\sim\chi^2_k .
\]

Proof. Since \(P\) has orthonormal rows, \(Pz\) is a \(k\)-dimensional normal
vector with mean zero and covariance
\[
\operatorname{Cov}(Pz)=P I_d P^\top=I_k .
\]
Therefore \(Pz\sim N(0,I_k)\), and the squared Euclidean norm of a standard
\(k\)-variate normal vector is \(\chi^2_k\).

This is the local reference law used by the projected-Wald kernel. It proves a
fixed-projection result only.

### Lemma 2: Feature-Space Whitening Reduces Fixed Contrasts To Lemma 1

If, under a fixed node pair and fixed feature chart,
\[
\Delta_{A,B}\sim N(0,\Sigma_0(A,B))
\]
with \(\Sigma_0(A,B)\) positive definite on the active contrast coordinates,
then \(z_{A,B}=\Sigma_0(A,B)^{-1/2}\Delta_{A,B}\sim N(0,I_d)\). Lemma 1 then
gives the fixed-projection Wald law.

Proof. Linear transformation gives
\[
\operatorname{Cov}(z_{A,B})
=
\Sigma_0^{-1/2}\Sigma_0\Sigma_0^{-1/2}
=I_d .
\]
Normality is preserved by linear maps.

This connects Bernoulli, categorical, and continuous covariance contracts to
the same projected-Wald proof. Bernoulli and multinomial settings rely on
finite-sample or asymptotic normal approximations for empirical frequencies;
continuous Gaussian settings can satisfy the normal contrast assumption
directly under fixed covariance conditions. Where these assumptions fail, the
lemma does not apply.

### Lemma 3: Nested Child-Parent Variance Cancellation

Assume independent leaf observations with common covariance \(\Sigma\). Let
child \(c\) have \(n_c\) leaves and parent \(u\) have \(n_u\) leaves. Let
\(o=u\setminus c\) have \(n_o=n_u-n_c\) leaves and mean \(\hat\mu_o\). Since
\[
\hat\mu_u=\frac{n_c\hat\mu_c+n_o\hat\mu_o}{n_u},
\]
the child-parent contrast is
\[
\hat\mu_c-\hat\mu_u
=
\frac{n_o}{n_u}(\hat\mu_c-\hat\mu_o).
\]
Therefore
\[
\operatorname{Var}(\hat\mu_c-\hat\mu_u)
=
\frac{n_o^2}{n_u^2}
\left(\frac{\Sigma}{n_c}+\frac{\Sigma}{n_o}\right)
=
\frac{n_o}{n_c n_u}\Sigma
=
\left(\frac{1}{n_c}-\frac{1}{n_u}\right)\Sigma .
\]

This proves why the edge contrast uses a nested child-parent covariance rather
than the disjoint two-sample sibling covariance.

### Lemma 4: Sibling Variance For Disjoint Children

If children \(l(u)\) and \(r(u)\) are disjoint subtrees with independent leaf
means under the fixed-membership null, then
\[
\operatorname{Var}(\hat\mu_{l(u)}-\hat\mu_{r(u)})
=
\left(\frac{1}{n_l}+\frac{1}{n_r}\right)\Sigma .
\]

Proof. The two empirical means use disjoint independent leaves, so variances
add. Feature-space block covariance replaces \(\Sigma\) with Bernoulli,
multinomial drop-last, or empirical-Gaussian block covariance in the active
chart.

### Proposition 1: Same-Data Selection Changes The Null Law

Let \(W\sim\chi^2_k\) under an unselected fixed-object null. For a selection
event \(\mathcal S\) that depends on the same data as \(W\), the conditional
law \(W\mid \mathcal S\) is not generally \(\chi^2_k\).

Proof by counterexample. Take \(\mathcal S=\{W>c\}\) for any finite
\(c>0\). Then
\[
\Pr(W\le c\mid \mathcal S)=0,
\]
whereas \(\Pr(\chi^2_k\le c)>0\). Hence the conditional law differs from the
unconditional chi-square law.

The Tree-Break Selection same-data hierarchy event is more complex than \(\{W>c\}\), but the
logic is the same: hierarchy construction, edge opening, and focal sibling
selection are functions of the same feature matrix that supplies the tested
contrast. A fixed-object proof cannot be reused without conditioning on that
selection event.

### Proposition 2: Child-Mean Internal Rows Break The Fixed-Subspace Proof

If the projection \(P\) is allowed to depend on the tested contrast direction
\(z\), Lemma 1 no longer applies as stated.

Proof by counterexample. Let \(z\sim N(0,I_d)\) and set
\[
P(z)=\frac{z^\top}{\lVert z\rVert}
\]
as a one-row projection. Then \(P(z)P(z)^\top=1\), but
\[
\lVert P(z)z\rVert^2=\lVert z\rVert^2\sim\chi^2_d,
\]
not \(\chi^2_1\).

This is the proof-level explanation for the selected-PCA validation result:
deterministic child-mean/internal rows can align the spectral basis with the
same child contrast being tested. Leaf-only spectral rows are therefore the
active inferential contract; internal rows are historical diagnostics, not a
production basis.

### Proposition 3: Scalar Mean Inflation Does Not Prove Tail Calibration

Let \(R\) be a selected-ratio statistic. Choosing \(c=\mathbb E(R)\) can match
the first moment, but it does not imply that
\[
R \stackrel{d}{=} c\,\chi^2_\nu/\nu
\]
or that scalar-\(c\) p-values are uniform.

Proof. Distributional equality is stronger than equality of means. Two
nonnegative random variables can have the same expectation and different
upper-tail probabilities. A p-value transform is uniform only when it uses the
correct null distribution, not merely a distribution with the same mean.

This connects the selected-hierarchy external calibration diagnostics to the
method: scalar \(c\) may be descriptive, but the production object must be a
selected-ratio tail law whenever scalar-scaled p-values fail distributional
checks.

### Proposition 4: Oracle Recoverability Separates Tree Failures From Gate Failures

Fix a hierarchy \(T\) and restrict the final output to subtree cuts of \(T\).
If the best exact-\(K\) oracle subtree cut has ARI below the recoverability
threshold, then no gate or p-value rule operating on the same fixed tree and
same cut family can exceed that oracle target.

Proof. The oracle is defined as the maximum ARI over the allowed subtree-cut
family. Any gate traversal output is one member of that same family. Therefore
its ARI is bounded above by the oracle maximum.

This proves why tree/metric unrecoverable benchmark rows must not drive
statistical gate changes. They require representation, distance, linkage, or
benchmark-construction analysis before local-test changes.

### Theorem: Current Proof Boundary

The following statement is supported by proof under fixed-object assumptions:

\[
\text{fixed chart}+\text{fixed node pair}+\text{fixed projection}
\Longrightarrow
W(P)\sim\chi^2_k
\]
after valid covariance whitening.

The following statement is not yet proved:

\[
W_u\mid
\{\text{same-data hierarchy selected, edge path open, focal }u\}
\sim
\text{known calibrated law}.
\]

The missing theorem is a selected-inference theorem for
\[
\mathcal S_u
=
\{X:\ T(X)\text{ contains }u,\ E_u(X)\text{ is open},\
u\text{ is the focal sibling context}\}.
\]
The target is
\[
\mathcal L\!\left(R_u\mid X\in\mathcal S_u\right),
\qquad
R_u=\frac{W_u}{a_u\nu_u}.
\]

### How The Method Pieces Connect

1. Feature-space covariance defines the tangent metric.
2. Whitening maps valid fixed contrasts to isotropic coordinates.
3. Orthonormal projection gives the fixed projected-Wald chi-square law.
4. MP dimension selection chooses the projection size; its identity edge is
   valid only under local isotropic spectral assumptions.
5. Same-data hierarchy selection replaces the fixed-object law with a
   conditional selected-region law.
6. Internal empirical-null calibration is valid only when strict or
   stopped-edge null-like support exists.
7. If support is absent, production must fail closed unless a separately
   validated selected-tail law exists.
8. Oracle recoverability decides whether a bad benchmark row is even eligible
   to motivate statistical changes.

### Current Proof Gaps

- Derive the selected region \(\mathcal S_u\) for at least one simple root
  split model.
- Express edge-selection severity as an actual signed distance or action to
  the selected boundary, not only as \(-\log p_{\mathrm{edge}}\).
- Determine whether curvature or tangent-cone corrections are needed for
  selected sibling p-values.
- Derive or validate the full selected-ratio tail law; scalar \(c\) is not
  sufficient as a proof object.
- Derive a selected or deformed spectral law when the local population
  spectrum is not \(H=\delta_1\).
- Treat categorical selected extreme-node geometry separately from
  Bernoulli/discretized selected spectral inflation.
- Validate continuous covariance and continuous selected-hierarchy null
  generation before using continuous selected-tail diagnostics in production.

## Evidence

- `manuscript/guides/full_method_logic_map.md` and
  `manuscript/guides/edge_sibling_derivation_guide.md` record the existing
  proof gaps around projected chi-square, selected PCA, sibling inflation, and
  final traversal.
- `manuscript/sections/method/edge_test.tex` and
  `manuscript/sections/method/sibling_test.tex` state the current edge and
  sibling projected-Wald contracts.
- [[selected-hierarchy-selection-geometry]] records the selected-region
  interpretation and the same-data barycentric selection problem.
- [[selected-hierarchy-geometric-law-map]] records the current edge-action,
  spectral-mode, angular, sampling, branch-length, and barycentric variables.
- [[local-marchenko-pastur-rule]] records the MP algebra, identity-law
  limitations, finite identity-null screen, and categorical/continuous split.
- [[selected-pca-projected-wald-validation]] records the empirical validation
  that leaf-only selected PCA is calibrated in tested Gaussian settings while
  child-mean internal rows break the reference.
- [[hierarchy-gate-separation-20260603]] records the current proof-use rule:
  tree/metric unrecoverable cases must not motivate statistical gate changes.

## Links

- [[projected-wald-statistic]]
- [[selected-hierarchy-selection-geometry]]
- [[selected-hierarchy-geometric-law-map]]
- [[selected-hierarchy-null-support-contract]]
- [[local-marchenko-pastur-rule]]
- [[selected-pca-projected-wald-validation]]
- [[hierarchy-gate-separation-20260603]]
- [[open-mathematical-questions]]

## Open Questions

- Can \(\mathcal S_u\) be written explicitly for a root split under a simple
  Gaussian or Bernoulli null and a fixed agglomerative linkage?
- Is edge action \(-\log p_{\mathrm{edge}}\) asymptotically equivalent to a
  signed distance or large-deviation rate for \(\mathcal S_u\)?
- Which selected-tail contexts admit enough support for production
  calibration without borrowing across incompatible regimes?
- Can the selected spectral law be derived from a deformed MP population
  spectrum plus a selected-region condition?
- What theorem, if any, justifies traversal-aligned sibling BH over the
  edge-reachable frontier or the pass-through traversal rule under the selected
  hierarchy?
