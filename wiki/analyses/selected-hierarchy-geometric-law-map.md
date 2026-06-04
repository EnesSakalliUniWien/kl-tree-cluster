---
title: Selected Hierarchy Geometric Law Map
type: analysis
status: reviewed
updated: 2026-06-03
sources:
  - wiki/sources/selected-hierarchy-geometry-covariates-20260602.md
  - wiki/analyses/selected-hierarchy-selection-geometry.md
  - wiki/analyses/selected-hierarchy-null-support-contract.md
  - wiki/sources/selected-geometry-mp-integral-literature-20260602.md
  - kl_clustering_analysis/hierarchy_analysis/statistics/projection/projected_wald/projected_wald_reference_distribution.py
  - kl_clustering_analysis/hierarchy_analysis/statistics/projection/projection_dimension_estimation/projection_dimension_estimators.py
  - kl_clustering_analysis/hierarchy_analysis/statistics/contrast_covariance.py
  - kl_clustering_analysis/hierarchy_analysis/statistics/branch_length_utils.py
  - kl_clustering_analysis/tree/distributions.py
  - raw/assets/benchmark-results/selected_hierarchy_geometry_covariates_20260602_100/candidate_equations.csv
  - raw/assets/benchmark-results/selected_hierarchy_geometry_covariates_20260602_holdout_100/candidate_equation_holdout.csv
  - raw/assets/benchmark-results/selected_hierarchy_geometry_covariates_20260602_broad_200/candidate_equation_holdout.csv
  - raw/assets/benchmark-results/selected_hierarchy_geometry_covariates_20260602_broad_200/geometry_summary_by_case.csv
  - raw/assets/benchmark-results/selected_hierarchy_topology_refinement_input_20260603_300/candidate_equation_holdout.csv
  - raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_broad_200/selected_ratio_tail_law.csv
  - raw/assets/benchmark-results/selected_hierarchy_tail_law_admissibility_boundary_20260603_500/selected_ratio_tail_law.csv
  - raw/assets/benchmark-results/selected_hierarchy_tail_law_binary_boundary_20260603_600/selected_ratio_tail_law.csv
  - wiki/sources/selected-ratio-tail-law-diagnostic-20260602.md
  - wiki/sources/selected-tail-admissibility-domain-20260603.md
tags:
  - analysis
  - geometry
  - selection
  - physics
---

# Selected Hierarchy Geometric Law Map

## Summary

The selected-hierarchy geometry variables are not governed by independent
physical laws in the literal sense. They are statistical-geometric variables.
Some have direct mathematical laws used by the method: chi-square projected
energy, Marchenko--Pastur spectral thresholding, Pythagorean projection, sample
variance scaling, and explicit branch-length descriptors for tree geometry.
Other "physical" connections are useful analogies: energy/action,
coarse-graining, entropy/effective rank, and modal decomposition.

The current evidence says the leading observed variable is edge-selection
strength. In the 100-replicate geometry covariate diagnostic,
`negative_log10_min_child_edge_bh_p_value` has Spearman correlation about
`0.712` with \(\log R_u\), where
\[
R_u=\frac{W_u}{a_u\nu_u}.
\]
This makes edge-selection severity the closest object to a selected
large-deviation or energy-barrier coordinate. Eigenvalue and angular variables
then describe which selected modes carry that selected contrast energy.
The candidate-equation pass sharpens this: a compact edge-plus-spectral
equation is nearly as strong as the full descriptive equation for top-tail
discrimination, while the full equation explains more mean log-ratio variation
within replicate folds. A held-out transfer diagnostic keeps the
edge-plus-spectral equation strong under replicate folds and makes it the most
stable compact candidate under leave-one-case-out transfer. This is still a
descriptive law-finding result, not a production external-null calibration.

The broader 200-replicate panel makes the target sharper. Tail ranking and
absolute selected-ratio scale are different problems. Edge-selection severity
almost perfectly ranks broad-panel top-tail records, but selected-ratio means
range from about `31` in the clear Gaussian case to about `580` in the
high-dimensional categorical case. A production law would therefore need to
model the selected-ratio tail distribution within admissible contexts, not just
rank the global upper tail.

The selected-ratio tail-law diagnostic tests that next object directly. It
uses contexts defined by source family, feature family, parent-size bin,
sibling projection dimension, and binned edge action. In the 200-replicate
broad run, no context is production-admissible under the current support
contract. Some high-support small-parent, high-edge-action contexts have
held-out exceedance near \(\alpha_{\mathrm{sib}}=0.01\), but sparse root and
low-edge-action contexts are unstable. This describes a possible conditional
tail law; it does not license a production external calibration model.
Focused follow-ups show that this support target is reachable but narrow. The
300-replicate run admits two Gaussian small-parent, high-edge-action contexts.
The 500-replicate boundary expansion adds two categorical small-parent,
high-edge-action contexts under the same exact context rule. The 600-replicate
binary boundary expansion adds one binary projection-1 small-parent,
high-edge-action context. Thus the open problem is no longer whether an
admissible selected-tail context can exist; it is defining the domain of
admissible contexts and leaving external calibration undefined outside that
domain.

The no-bootstrap analytic direction is now sharper. The useful object in the
selective-inference literature is the selected region in the same local
tangent coordinate system used by the statistic. Shimodaira and Terada use
multiscale bootstrap to estimate signed distance and curvature, but KL-TE does
not need that estimator as a production path. The retained mathematical
objects are the selected region, tangent cone or local boundary, signed
distance/action, curvature, and conditional selected-error law.

## Details

### Core Energy Law

The projected-Wald statistic is a squared norm in null-whitened tangent
coordinates. For a sibling contrast \(z_u\) and selected orthonormal basis
\(V_k\),

\[
W_u=\lVert V_k^\top z_u\rVert^2.
\]

Conditional on a fixed projection under an isotropic standardized null,

\[
W_u\sim\chi^2_k.
\]

This is the method's actual reference law in
`projected_wald_reference_distribution.py`. The physics analogy is kinetic
energy of a whitened displacement: the statistic is energy in selected modes.
That analogy is only explanatory; the operative law is the chi-square law.

### Edge Selection As Action Or Large Deviation

Child-parent edge p-values are monotone functions of projected edge energy.
For large chi-square statistics, the tail probability decays approximately
exponentially. Therefore

\[
-\log p_{\mathrm{edge}}
\]

acts like a large-deviation action or energy barrier: small p-values indicate
that the hierarchy selected a split with unusually high whitened separation.
The diagnostic records this as
`negative_log10_min_child_edge_bh_p_value`.

This is the strongest current relationship with \(\log R_u\). The implication
is mathematical, not merely metaphorical: conditioning only on parent size,
depth, and projection dimension misses the severity of the selected edge
event.

### Eigenvalues As Spectral Modes

Eigenvalues describe variance carried by parent-local null-whitened tangent
modes. The method uses the Marchenko--Pastur upper edge

\[
\lambda_+ = \left(1+\sqrt{d/m}\right)^2
\]

to count raw spectral signal directions before applying the test-dimension
floor. This is an actual random-matrix law, with historical roots in
mathematical physics and statistical mechanics, but in the code it is used as
a covariance-spectrum threshold.

The related physical analogy is modal decomposition: eigenvectors are modes,
and eigenvalues are mode energies or variances. Current diagnostics show
spectral variables are secondary but visible: selected eigenvalue mass and
selected eigenvalue over the MP upper bound correlate with selected-ratio
scale.

The current MP edge is the identity-population special case. If local
null-whitened tangent rows were independent and exactly isotropic, the
population spectral law would be \(H=\delta_1\), and the upper support edge
would be \((1+\sqrt{d/m})^2\). The analytic generalization is the
Silverstein--Choi Stieltjes-transform equation for a general population
spectrum \(H\),
\[
m(z)=-
\left(
z-c\int \frac{t}{1+t\,m(z)}\,dH(t)
\right)^{-1},
\]
with density recovered from
\[
f(x)=\pi^{-1}\operatorname{Im}m(x+i0)
\]
and support edges obtained from
\[
z(m)=-\frac{1}{m}+c\int\frac{t}{1+t\,m}\,dH(t).
\]
This integral-based Marchenko--Pastur route should be studied before changing
KL-TE's production dimension rule. It asks whether each local tangent spectrum
is close enough to \(H=\delta_1\), or whether categorical, continuous, or
selected contexts need a deformed local spectral edge.

The candidate-equation diagnostic suggests spectral variables are especially
important for the upper tail. The edge-plus-spectral equation

\[
\log R_u
\sim
A_u+
\log(\lambda_{k,u}/\lambda_{+,u})+
m_{k,u}+
r_{\mathrm{eff},u}
\]

has top-10% tail AUC about `0.969`, nearly matching the larger full
descriptive equation. This is not a validated law; it is the current best
compact equation family to test in a larger selected-hierarchy study.

The 100-replicate holdout diagnostic keeps this interpretation. Under
replicate-modulo folds, the edge-plus-spectral equation has holdout tail AUC
about `0.969` and holdout \(R^2 \approx 0.104\); the full descriptive equation
has tail AUC about `0.969` and holdout \(R^2 \approx 0.181\). Under
leave-one-case-out transfer, the edge-plus-spectral equation has tail AUC
about `0.922` and holdout \(R^2 \approx 0.198\), while the full descriptive
equation has tail AUC about `0.877` and holdout \(R^2 \approx 0.184\). The
compact equation therefore transfers better as a tail descriptor than the
larger equation in this small four-case panel.

The broad 200-replicate run changes the interpretation from "best compact
equation" to "separate tail ranking from calibration scale." Replicate-fold
holdout gives the full descriptive equation \(R^2\approx0.429\), while the
selected-energy candidate has the highest top-tail AUC, about `0.9999`.
Leave-one-source-family-out gives the full descriptive equation the best
absolute-scale \(R^2\), about `0.458`, but its tail AUC drops to about `0.944`.
Edge and edge-plus-spectral equations keep near-perfect global tail AUCs but
fit absolute \(\log R\) scale worse. This means edge action is likely a
necessary selection coordinate, but not a sufficient calibrated law.

The 300-replicate Gaussian/categorical topology-input rerun adds an important
verification check. Under leave-one-source-family-out transfer, the
edge-sampling and edge-spectral equations keep tail AUCs near `0.995`, while
the selected-energy candidate falls to about `0.152` and the full descriptive
equation to about `0.050`. Therefore the selected-energy and full equations
should be treated as panel-specific descriptive fits, not as stable laws. The
more defensible next object is a predeclared low-dimensional selected-tail
coordinate system built around edge action, sampling geometry, and spectral
excess, followed by within-context tail calibration.

### Eigenvectors And Trigonometric Projection

The angular variables are ordinary orthogonal-projection geometry:

\[
\cos^2\theta_u=
\frac{\lVert V_k^\top z_u\rVert^2}{\lVert z_u\rVert^2},
\qquad
\sin^2\theta_u=1-\cos^2\theta_u.
\]

This is the Pythagorean law for decomposing a vector into selected-subspace
and residual components. \(\cos^2\theta_u\) is the fraction of sibling
contrast energy captured by the selected spectral subspace. It is high in the
representative run, so selected siblings usually align strongly with the
selected PCA modes. However, angular variables alone explain less log-ratio
variation than edge-selection severity or spectral summaries.

### Parent Size And Child Balance

Parent size and child balance belong to sampling-variance geometry. For
two-sample contrasts, variance scales with the sample-size terms

\[
\frac{1}{n_L}+\frac{1}{n_R}.
\]

Balanced children minimize this term for a fixed parent size. Very small or
unbalanced selected nodes have higher noise and are more vulnerable to
selection extremes. The physical analogy is finite-size fluctuation: smaller
systems fluctuate more strongly. The statistical law is central-limit and
standard-error scaling, not a separate physical law.

### Branch Length

Branch length is related to Brownian-motion variance accumulation on a
phylogenetic tree. Under an explicit Brownian phylogenetic covariance model,
independent branch variances would add:

\[
\operatorname{Var}(X_L-X_R)\propto b_L+b_R.
\]

The current KL-TE Wald kernel does not implement that covariance model.
Branch lengths are recorded as descriptive tree covariates for topology and
selection diagnostics; they are not injected into edge or sibling
projected-Wald variance.

### Barycenters And Coarse-Graining

Internal node distributions are empirical subtree barycenters:

\[
\bar x_v=
\frac{1}{|D(v)|}\sum_{i\in D(v)}x_i.
\]

This is mathematically a weighted center-of-mass calculation. The useful
physics analogy is coarse-graining: many leaves are compressed into one
subtree state. The selected-hierarchy problem arises because the same data
both chooses the coarse-grained hierarchy and tests the resulting selected
barycentric contrasts.

### Entropy And Effective Rank

Effective rank uses the entropy of the normalized eigenvalue spectrum:

\[
r_{\mathrm{eff}}=\exp\left(-\sum_j q_j\log q_j\right),
\qquad
q_j=\lambda_j/\sum_\ell\lambda_\ell.
\]

This is directly related to Shannon entropy and has a statistical-mechanics
analogy: concentrated spectra have low entropy and diffuse spectra have high
entropy. In the method it is a spectral-spread descriptor, not a production
calibration law.

## Evidence

- `wiki/sources/selected-hierarchy-geometry-covariates-20260602.md` records
  the 100-replicate geometry covariate diagnostic and relationship summaries.
- `raw/assets/benchmark-results/selected_hierarchy_geometry_covariates_20260602_100/candidate_equations.csv`
  records the nested candidate equations for mean log-ratio and top-tail
  discrimination.
- `raw/assets/benchmark-results/selected_hierarchy_geometry_covariates_20260602_holdout_100/candidate_equation_holdout.csv`
  records replicate-fold and leave-one-case-out transfer diagnostics for the
  same candidate equations.
- `raw/assets/benchmark-results/selected_hierarchy_geometry_covariates_20260602_broad_200/candidate_equation_holdout.csv`
  records broad-panel replicate, leave-one-case, and leave-one-source-family
  holdout diagnostics.
- `raw/assets/benchmark-results/selected_hierarchy_topology_refinement_input_20260603_300/candidate_equation_holdout.csv`
  records the newer Gaussian/categorical topology-input transfer check where
  selected-energy and full descriptive equations fail leave-one-source-family
  tail transfer.
- `raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_broad_200/selected_ratio_tail_law.csv`
  records within-context selected-ratio tail support, held-out exceedance, and
  production-admissibility failures.
- `wiki/sources/selected-geometry-mp-integral-literature-20260602.md` records
  the no-bootstrap selected-region geometry direction and the
  Stieltjes-transform integral route for deformed MP spectra.
- `kl_clustering_analysis/hierarchy_analysis/statistics/projection/projected_wald/projected_wald_reference_distribution.py`
  defines the fixed-subspace chi-square reference.
- `kl_clustering_analysis/hierarchy_analysis/statistics/projection/projection_dimension_estimation/projection_dimension_estimators.py`
  defines the Marchenko--Pastur signal-count rule and effective rank.
- `kl_clustering_analysis/hierarchy_analysis/statistics/branch_length_utils.py`
  validates and aggregates observed branch-length metadata.
- `kl_clustering_analysis/tree/distributions.py` implements internal node
  distributions as empirical subtree barycenters.

## Links

- [[selected-hierarchy-geometry-covariates-20260602]]
- [[selected-hierarchy-selection-geometry]]
- [[selected-hierarchy-null-support-contract]]
- [[selected-ratio-tail-law-diagnostic-20260602]]
- [[local-marchenko-pastur-rule]]
- [[projected-wald-statistic]]

## Open Questions

- Is edge-selection severity a required conditioning variable in a selected
  external null law, or can it be summarized through a lower-dimensional
  selected-ratio tail model?
- Are eigenvalue concentration and angular alignment sufficient tail-shape
  variables once edge-selection severity is included?
- Does the edge-plus-spectral equation remain stable under more cases,
  non-root contexts, high-dimensional categorical failures, phylogenetic
  families, and a row-level held-out selected-hierarchy null study?
- What is the right evaluation target for a production selected law: global
  top-tail ranking, within-context tail calibration, absolute selected-ratio
  prediction, or a full selected-ratio tail distribution?
- Which selected-ratio tail-law contexts can reach production-admissible
  support without borrowing across incompatible source families or selection
  regimes?
- Can the selected-hierarchy law be derived as a large-deviation or
  extreme-value problem over selected barycentric contrasts, using analytic
  selected-region geometry rather than bootstrap-estimated geometry?
- Does the local null-whitened tangent spectrum have \(H\approx\delta_1\), or
  does KL-TE need a deformed Marchenko--Pastur edge computed from a local
  population-spectrum integral?
- Should branch-length geometry enter the selected law for phylogenetic cases
  only through descriptive topology covariates, or through a future explicit
  phylogenetic covariance model?
