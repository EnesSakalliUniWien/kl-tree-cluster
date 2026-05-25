---
title: Open Mathematical Questions
type: question
status: reviewed
updated: 2026-05-25
sources:
  - manuscript/guides/full_method_logic_map.md
  - manuscript/guides/edge_sibling_derivation_guide.md
  - manuscript/sections/method/assumptions_validation.tex
  - manuscript/sections/method/sibling_test.tex
  - manuscript/sections/method/representation.tex
  - manuscript/sections/experiments/section.tex
  - wiki/analyses/oracle-gate-path-diagnostic.md
  - wiki/analyses/manuscript-life-science-readiness.md
  - benchmarks/validation/method_constants_manifest.py
tags:
  - method
  - math
  - validation
---

# Open Mathematical Questions

## Question

Which mathematical questions remain open for KL-TE before the method can be
treated as publication-ready rather than a gap-marked methods draft?

## Current State

The central unresolved calibration question is the missing external
conditional-null object for sibling tests when internal empirical-null support
is absent. The production estimator may use internal empirical-null calibration
only when the local support is strict null-like or stopped-edge supported. For
selected-non-null-only contexts, especially the high-dimensional Gaussian
blockers, the honest mathematical object is an external conditional law such as
\[
\mathcal L\!\left(
T_u\mid
\text{tree construction},\
\text{edge selection},\
u\ \text{selected as focal sibling context}
\right),
\]
with an external scale
\[
c_{\mathrm{external}}(u)
=
\frac{\mathbb E_0[W_u^{\mathrm{selected}}]}{a_u\nu_u}.
\]
The open work is to derive or simulate this law while preserving the local
context \(n_L,n_R,p,k_u,a_u,\nu_u\), feature family, and selection regime. The
existing fixed-subspace and root/local edge-selection diagnostics do not
produce inflation factors in the thousands, so they do not justify a production
external calibration model.

The projected-Wald reference also remains mathematically conditional. For a
fixed orthonormal projection, \(\lVert Pz\rVert^2\sim\chi^2_k\) under an
isotropic standardized null. The manuscript still needs either a clean
assumption statement that treats the selected PCA rows as fixed, or a
derivation/validation of the data-selected projection effect. This question is
coupled to the Marchenko--Pastur dimension rule, the minimum spectral dimension
floor, and the inclusion of internal subtree rows in local spectral matrices.

The feature-space model is only partly settled. Bernoulli coordinates have a
clear variance model under fixed membership. Categorical variables now have an
explicit multinomial drop-last covariance chart, but high-cardinality
categorical calibration still needs validation. Continuous data require an
explicit covariance contract; whether KL-TE should provide a production
continuous covariance estimator or keep continuous use as diagnostic remains
open.

The sibling multiplicity and traversal questions are separate from inflation.
Some binary and categorical under-splits are sibling-FDR blockers rather than
inflation-estimator failures. The current flat sibling BH correction therefore
needs either a traversal-aligned justification or a replacement hierarchical
target. Pass-through traversal also remains empirical: it can recover
descendant signal, but it can fragment oracle clades because it does not yet
compare descendant split evidence against local sibling-same evidence.

The failure modes must stay separated. Diffuse Gaussian, heavy-overlap, and SBM
cases often look like hierarchy or recoverability failures rather than gate
failures. Phylogenetic false splitting may require branch-length stopping,
sibling covariance changes, or hierarchical FDR rather than an inflation
change. These cases should not be collapsed into one threshold-tuning problem.

The current concrete open questions are:

1. How should the external calibration law condition on hierarchy construction,
   child-parent edge openings, and focal sibling selection?
2. For non-root blockers, does conditioning on the ancestor Tree-BH path keep
   \(c\) small, or create larger selected-context inflation?
3. Which full-selection diagnostic is needed: fixed observed tree, ancestor
   Tree-BH path, focal blocker selection, or full hierarchy reconstruction?
4. Can external Gaussian sibling-null calibration explain the Gaussian
   blockers?
5. Does external inflation depend mainly on \(p/n\), projection dimension, tree
   selection, covariance whitening, or feature family?
6. When should the calibration hierarchy return
   \(\hat c_{\mathrm{internal}}\), \(\hat c_{\mathrm{external}}\), or fail
   undefined?
7. What minimum effective calibration support is required before internal
   empirical-null inflation is trustworthy?
8. Is the empirical-null weight rule calibrated enough to use beyond
   diagnostics?
9. Is the context bandwidth stable when calibration support is sparse?
10. Under what exact assumptions is the projected chi-square reference valid?
11. Is the fixed-projection proof sufficient, or must the data-selected PCA
    effect be derived or simulated?
12. Does the local Marchenko--Pastur rule preserve calibration and power?
13. Is the minimum spectral dimension \(k_{\min}=2\) justified?
14. Does including internal subtree rows improve stability or bias local tests?
15. Is the sibling projection dimension rule mathematically justified?
16. How should one-hot categorical dependence be modeled and validated?
17. Are high-cardinality categorical failures caused by covariance modeling,
    sibling FDR, or projection dimension policy?
18. Should continuous data get a production covariance estimator?
19. How should discretized continuous data be validated under approximate
    Bernoulli assumptions?
20. What is the clean general feature-space formulation for mixed Bernoulli,
    categorical, and continuous blocks?
21. What sibling-FDR target should replace or justify flat BH across focal
    sibling pairs?
22. Should sibling FDR be traversal-aligned, hierarchical, or conditioned on
    the edge path?
23. Should pass-through require descendant split evidence to overcome local
    sibling-same evidence?
24. What functional form should
    \(S_{\mathrm{desc}}(v)>\tau(S_{\mathrm{same}}(v),n_v,d_v)\) take?
25. Does pass-through improve under-splits without unacceptable over-splitting?
26. How should direct sibling false splitting inside phylogenetic clades be
    modeled?
27. Which failures are gate failures and which are hierarchy/metric
    recoverability failures?
28. Which hierarchy diagnostics should be built for `tree_unrecoverable` cases
    before changing metrics or linkage?
29. For diffuse Gaussian, heavy-overlap, and SBM cases, is the problem
    representation, distance metric, linkage, or gate logic?
30. Which validation outputs become the locked manuscript evidence set?
31. Which method constants become justified defaults rather than implementation
    defaults?
32. What null calibration study supports edge alpha?
33. What null calibration and power study supports sibling alpha?
34. What simulation validates final cluster-count control, not only local
    p-values?
35. What planted-structure simulations support power claims?
36. Which real biological application should become the first
    manuscript-ready result?
37. Which generated outputs should be promoted into a locked
    manuscript-results manifest?
38. How should the manuscript state every remaining approximation without
    overstating Type-I error control?

## Evidence

- `manuscript/guides/full_method_logic_map.md` records the central estimator
  chain, method constants, and submission gaps.
- `manuscript/guides/edge_sibling_derivation_guide.md` records proof and
  validation gaps for edge variance, selected PCA, sibling inflation, and final
  traversal.
- `manuscript/sections/method/assumptions_validation.tex` states the current
  assumptions and method-constant validation table.
- `manuscript/sections/method/sibling_test.tex` defines the strict internal
  empirical-null support contract and the fail-closed calibration behavior.
- `manuscript/sections/method/representation.tex` marks one-hot compositional
  constraints as requiring validation.
- `manuscript/sections/experiments/section.tex` records that results are still
  prospective and require locked outputs.
- `wiki/analyses/oracle-gate-path-diagnostic.md` records the oracle,
  gate-path, sibling-inflation, fixed-subspace null, local edge-selection null,
  and root Tree-BH selection diagnostics.
- `wiki/analyses/manuscript-life-science-readiness.md` records the open
  biological application choice.
- `benchmarks/validation/method_constants_manifest.py` enumerates the active
  method constants that still need validation artifacts.

## Links

- [[kl-te-method]]
- [[projected-wald-statistic]]
- [[oracle-gate-path-diagnostic]]
- [[top-down-traversal]]
- [[manuscript-life-science-readiness]]
