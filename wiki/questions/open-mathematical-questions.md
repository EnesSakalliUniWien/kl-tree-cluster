---
title: Open Mathematical Questions
type: question
status: reviewed
updated: 2026-06-02
sources:
  - manuscript/guides/full_method_logic_map.md
  - manuscript/guides/edge_sibling_derivation_guide.md
  - manuscript/sections/method/assumptions_validation.tex
  - manuscript/sections/method/sibling_test.tex
  - manuscript/sections/method/representation.tex
  - manuscript/sections/experiments/section.tex
  - kl_clustering_analysis/tree/feature_space.py
  - kl_clustering_analysis/hierarchy_analysis/statistics/contrast_covariance.py
  - benchmarks/shared/generators/case_data_contracts.py
  - kl_clustering_analysis/tree/distributions.py
  - wiki/analyses/oracle-gate-path-diagnostic.md
  - wiki/sources/edge-selection-null-audit-20260601.md
  - wiki/sources/feature-split-selection-audit-20260601.md
  - wiki/sources/selected-hierarchy-null-audit-20260601.md
  - wiki/analyses/selected-hierarchy-null-support-contract.md
  - wiki/sources/selected-hierarchy-stratification-diagnostic-20260602.md
  - wiki/analyses/local-marchenko-pastur-rule.md
  - wiki/analyses/dimensional-gaussian-representation-diagnostic.md
  - wiki/analyses/manuscript-life-science-readiness.md
  - benchmarks/validation/method_constants_manifest.py
  - benchmarks/validation/feature_covariance_calibration.py
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

The problem decomposition is now sharper: missing sibling calibration support
is caused upstream by tree and edge selection. Under pure Bernoulli null data,
the edge-selection audit shows that using the same data to build the hierarchy
and test child-parent edges rejects about `99%` of tested edges at
`EDGE_ALPHA = 0.001`. Holding the hierarchy fixed and permuting feature
columns gives median rejection rate `0.0`. Thus the production issue is not
only a sibling inflation estimator question; it is a selected-hierarchy
conditional inference question.

The first cross-fit diagnostic supports this interpretation. Because the KL
tree is a sample-leaf hierarchy, literal sample splitting is undefined without
a held-out-sample assignment model. The implemented feature-split audit builds
the tree from one feature block and tests node distributions on a held-out
feature block over the same leaves. In `gauss_null_large`, this changes the
edge rejection rate from `1.0` in-sample to `0.0`, restores 199 supported
calibration records, and returns one cluster. Signal examples retain high or
perfect ARI while recovering many supported records. This points toward
cross-fit or selected-tree conditional inference as the next mathematical
development.

The selected-hierarchy null audit then removes cross-fitting from the candidate
method and simulates the same-data selected hierarchy directly. It regenerates
Bernoulli or categorical null data, rebuilds the tree in every replicate,
reruns edge tests, and collects selected focal sibling statistics. In the first
four representative cases, selected-hierarchy correction factors are large
(`28.8` to `58.5`). The mean-scaled chi-square p-value blocks
`gauss_null_large` while still rejecting the three signal examples. This is
the first evidence that a named selected-hierarchy calibration model may be
mathematically relevant. It is not yet a production rule because replicate
counts are small, context matching is preliminary, and continuous/null
covariance generation is unsupported.
[[selected-hierarchy-selection-geometry]] records the current geometric
explanation: fixed projected-Wald tangent geometry is not the blocker; the
unresolved object is the same-data selected hierarchy, where empirical
subtree barycenters, edge openings, and focal sibling contexts are conditioned
on high-contrast selection in the same coordinates that are later tested.
[[selected-hierarchy-null-support-contract]] records the current diagnostic
support interpretation: no matched selected-hierarchy records means unsupported,
and relaxed contexts explain support geometry rather than providing a fallback
calibration path.
The 100-replicate richer rerun keeps this conclusion for root targets and for
matched Gaussian non-root targets. It also exposes a sharper support problem:
binary and categorical non-root targets can have zero matched selected-null
records under projection, parent-size, and depth matching, so the external
selected-hierarchy diagnostic itself needs a minimum-support contract. Those
rows must be treated as descriptive unsupported states, not as fallback
calibration estimates.
The same run shows that support is not only a count problem. Matched rows have
relative simulation standard errors for \(c\) around `3%` to `10%`, but their
empirical-tail p-value resolution is only about `0.011` to `0.026`. Therefore
the present 100-replicate run can describe the selected-hierarchy phenomenon,
but it is not a locked production external-calibration run for
`SIBLING_ALPHA = 0.01`.
The 500-replicate descriptive precision run improves this picture. Strict root
rows and strict non-root Gaussian rows have hundreds of matched simulations and
\(c\) estimates in the tens. Non-root binary and categorical rows reveal that
exact context matching is the bottleneck: exact depth matching gives zero or
near-zero support, but dropping depth and then parent-size restores support
while keeping selected-hierarchy \(c\) in the tens. This is descriptive
evidence about support geometry, not a rule for borrowing relaxed contexts in
production.
The 2026-06-02 stratification diagnostic shows that parent size is a stronger
visible heterogeneity coordinate than exact depth in the current representative
cases. Small selected parent nodes often have \(c\) in the `50`--`70` range,
while root-like selected nodes are lower but still far above one. Exact depth
matching remains useful for description, but it is too sparse to promote to a
validated exact conditioning variable.
The regenerated stratification table records the selected-ratio law
\(R=W/(a\nu)\) rather than only the mean scale \(c\). In reliable parent-size
rows, the q95 of \(R\) ranges from about `38` to `103`, and the unconditioned
projected-Wald reference rejects almost all selected-null records in most
strata. This is descriptive evidence that the open object is a selected
conditional law, not a tuned scalar correction to the unselected reference.

The projected-Wald reference also remains mathematically conditional. For a
fixed orthonormal projection, \(\lVert Pz\rVert^2\sim\chi^2_k\) under an
isotropic standardized null. The manuscript still needs either a clean
assumption statement that treats the selected PCA rows as fixed, or a
derivation/validation of the data-selected projection effect. This question is
coupled to the Marchenko--Pastur dimension rule, the minimum spectral dimension
floor, and the row set used to estimate the local spectral basis.
The repository now includes a strict scaffold for this specific validation
target in
`benchmarks/validation/selected_pca_projected_wald_calibration.py`. That
scaffold simulates a fixed-membership Gaussian sibling null while selecting the
PCA rows and MP dimension from the same local null-whitened rows used by the
contrast. It is intentionally not evidence for hierarchy construction,
tree-selected sibling pairs, sibling FDR, traversal, or empirical-null
inflation.
[[selected-pca-projected-wald-validation]] records the first locked run. In the
tested Gaussian settings, leaf-only selected PCA was calibrated at
\(\alpha=0.05\), but appending deterministic child-mean internal rows produced
severe anti-conservative rejection rates from 0.648 to 1.000. Production now
uses descendant leaf rows only for the inferential PCA basis. The remaining
question exposed by that stricter contract is calibration support: several
high-dimensional or high-cardinality contexts have selected non-null records
but no strict empirical-null records.

The feature-space covariance contract is now explicit. Bernoulli coordinates
use a Bernoulli variance model under fixed membership. Categorical variables
use a multinomial covariance in a drop-last simplex chart. Continuous
benchmark inputs use empirical-Gaussian covariance blocks estimated on each
node's descendant leaves. The active implementation is an exact dense
covariance implementation; it now fails explicitly for continuous blocks whose
dense scatter state would exceed the supported implementation envelope. The
remaining mathematical question is validation and high-dimensional extension,
not contract shape: high-cardinality categorical calibration, continuous
finite-sample covariance behavior, and a validated low-rank or regularized
continuous covariance model still need targeted simulation evidence.
`benchmarks/validation/feature_covariance_calibration.py` now provides the
strict evidence generator for the first two targets. Its current scope is the
local sibling-null Wald statistic in the full tangent space, so it does not
close questions about selected PCA projections, MP dimension selection, sibling
FDR, traversal, tree construction, empirical-null inflation, or high-dimensional
continuous covariance.

The sibling multiplicity and traversal questions are separate from inflation.
Some binary and categorical under-splits are sibling-FDR blockers rather than
inflation-estimator failures. The current flat sibling BH correction therefore
needs either a traversal-aligned justification or a replacement hierarchical
target. Pass-through traversal also remains empirical: it can recover
descendant signal, but it can fragment oracle clades because it does not yet
compare descendant split evidence against local sibling-same evidence.

The failure modes must stay separated. Diffuse Gaussian, heavy-overlap, and SBM
cases often look like hierarchy or recoverability failures rather than gate
failures. [[dimensional-gaussian-representation-diagnostic]] records the
clearest current example: continuous coordinates make consolidated dimensional
Gaussian cases recoverable, but the diffuse continuous case still has an
unrecoverable average-linkage tree because weak spread-out mean signal is
dominated by many noise dimensions. Phylogenetic false splitting may require
branch-length stopping, sibling covariance changes, or hierarchical FDR rather
than an inflation change. These cases should not be collapsed into one
threshold-tuning problem.

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
7. What minimum matched selected-hierarchy support and Monte Carlo precision
   are required before an external calibration estimate is admissible?
8. Which context variables should be exact matching variables, and which should
   be descriptive stratifiers, for selected-hierarchy null studies?
   Current evidence says parent-size scale is informative, exact depth is
   sparse, and selected-ratio upper quantiles must be part of the support
   contract if a production external selected-hierarchy null is ever attempted.
9. What minimum effective calibration support is required before internal
   empirical-null inflation is trustworthy?
10. Is the empirical-null weight rule calibrated enough to use beyond
   diagnostics?
11. Is the context bandwidth stable when calibration support is sparse?
12. Under what exact assumptions is the projected chi-square reference valid?
13. Is the fixed-projection proof sufficient, or must the data-selected PCA
    effect be derived or simulated?
14. Does the local Marchenko--Pastur rule preserve calibration and power?
    [[local-marchenko-pastur-rule]] partially resolves the algebraic
    correctness question: the \(d_u/m_u\) edge matches the backend eigenvalue
    scale, and the code now separates raw MP signal count, projected-Wald test
    dimension, effective independent row count, and MP threshold row count. A
    targeted finite-null smoke did not support a simple finite-null swap:
    finite-null thresholding matched `binary_many_features`, worsened
    `cat_highcard_20cat_4c`, and under-split dimensional Gaussian cases in the
    historical diagnostic. The open question is whether a genuinely
    selection-aware threshold can be derived or validated.
15. Is the minimum spectral dimension \(k_{\min}=2\) justified?
    The 2026-06-01 leaf-only regression-gate diagnostic shows the trade-off:
    \(k_{\min}=2\) keeps higher mean/median ARI among runnable rows but creates
    six unsupported-calibration skips, while \(k_{\min}=1\) reduces skips to
    two but lowers aggregate ARI. This is not solved by a default change.
16. What calibration-support contract should be used when every local sibling
    record is selected non-null under the leaf-only spectral basis?
17. Is the sibling projection dimension rule mathematically justified?
18. How should one-hot categorical dependence be modeled and validated?
19. Are high-cardinality categorical failures caused by covariance modeling,
    sibling FDR, or projection dimension policy?
20. Is the per-node empirical-Gaussian covariance estimator calibrated well
    enough for production continuous data?
21. What validated low-rank or regularized continuous covariance model should
    replace dense empirical covariance for \(p \gg n\) continuous blocks?
22. How should discretized Gaussian benchmark variants be validated under
    approximate Bernoulli or categorical assumptions?
23. What is the clean general feature-space formulation for mixed Bernoulli,
    categorical, and continuous blocks?
24. What sibling-FDR target should replace or justify flat BH across focal
    sibling pairs?
25. Should sibling FDR be traversal-aligned, hierarchical, or conditioned on
    the edge path?
26. Should pass-through require descendant split evidence to overcome local
    sibling-same evidence?
27. What functional form should
    \(S_{\mathrm{desc}}(v)>\tau(S_{\mathrm{same}}(v),n_v,d_v)\) take?
28. Does pass-through improve under-splits without unacceptable over-splitting?
29. How should direct sibling false splitting inside phylogenetic clades be
    modeled?
30. Which failures are gate failures and which are hierarchy/metric
    recoverability failures?
31. Which hierarchy diagnostics should be built for `tree_unrecoverable` cases
    before changing metrics or linkage?
32. For diffuse Gaussian, heavy-overlap, and SBM cases, is the problem
    representation, distance metric, linkage, or gate logic?
33. Which validation outputs become the locked manuscript evidence set?
34. Which method constants become justified defaults rather than implementation
    defaults?
35. What null calibration study supports edge alpha?
36. What null calibration and power study supports sibling alpha?
37. What simulation validates final cluster-count control, not only local
    p-values?
38. What planted-structure simulations support power claims?
39. Which real biological application should become the first
    manuscript-ready result?
40. Which generated outputs should be promoted into a locked
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
- `manuscript/sections/method/representation.tex` defines one-hot categorical
  variables as drop-last multinomial blocks and continuous inputs as
  empirical-Gaussian covariance blocks; both still require validation.
- `manuscript/sections/experiments/section.tex` records that results are still
  prospective and require locked outputs.
- `wiki/analyses/oracle-gate-path-diagnostic.md` records the oracle,
  gate-path, sibling-inflation, fixed-subspace null, local edge-selection null,
  and root Tree-BH selection diagnostics.
- `wiki/analyses/local-marchenko-pastur-rule.md` records the MP dimension-rule
  audit, including backend eigenvalue scale, finite-sample null probes,
  the leaf-only production row contract, historical internal-row effects, and
  enhancement options.
- `benchmarks/validation/selected_pca_projected_wald_calibration.py` defines
  the selected-PCA projected-Wald validation scaffold.
- `wiki/analyses/selected-pca-projected-wald-validation.md` summarizes the
  first locked selected-PCA validation run and records the internal-row
  calibration failure.
- `wiki/analyses/dimensional-gaussian-representation-diagnostic.md` records
  the continuous-versus-median-binary dimensional Gaussian benchmark result and
  separates consolidated representation gain from diffuse hierarchy
  unrecoverability.
- `wiki/analyses/manuscript-life-science-readiness.md` records the open
  biological application choice.
- `benchmarks/validation/method_constants_manifest.py` enumerates the active
  method constants that still need validation artifacts.

## Links

- [[kl-te-method]]
- [[projected-wald-statistic]]
- [[local-marchenko-pastur-rule]]
- [[dimensional-gaussian-representation-diagnostic]]
- [[oracle-gate-path-diagnostic]]
- [[top-down-traversal]]
- [[manuscript-life-science-readiness]]
