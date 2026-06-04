---
title: Open Mathematical Questions
type: question
status: reviewed
updated: 2026-06-04
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
  - wiki/sources/selected-edge-type1-geometry-pilot-20260604.md
  - wiki/sources/selected-edge-binary-categorical-pilot-20260604.md
  - wiki/sources/feature-split-selection-audit-20260601.md
  - wiki/sources/selected-hierarchy-null-audit-20260601.md
  - wiki/analyses/selected-hierarchy-null-support-contract.md
  - wiki/sources/selected-hierarchy-external-calibration-contract-20260602.md
  - wiki/sources/selected-hierarchy-stratification-diagnostic-20260602.md
  - wiki/sources/selected-hierarchy-geometry-covariates-20260602.md
  - wiki/sources/selected-ratio-tail-law-diagnostic-20260602.md
  - wiki/sources/selected-geometry-mp-integral-literature-20260602.md
  - wiki/sources/local-mp-identity-law-diagnostic-20260602.md
  - wiki/sources/hierarchy-gate-separation-20260603.md
  - wiki/analyses/method-proof-web.md
  - wiki/analyses/root-selected-region-model.md
  - wiki/sources/root-selected-region-margins-20260603.md
  - wiki/sources/internal-vs-selected-hierarchy-inflation-20260603.md
  - raw/assets/benchmark-results/internal_vs_selected_hierarchy_inflation_20260603/internal_vs_selected_hierarchy_inflation.csv
  - wiki/sources/sibling-null-prior-interpolation-audit-20260604.md
  - wiki/sources/alpha-grid-full-20260604.md
  - wiki/sources/traversal-sibling-fdr-smoke-20260604.md
  - raw/assets/benchmark-results/selected-edge-type1-pilot-20260604/merged/selected_edge_geometry_edges.csv
  - raw/assets/benchmark-results/selected-edge-type1-pilot-20260604/merged/selected_edge_geometry_final.csv
  - raw/assets/benchmark-results/selected-edge-type1-binary-categorical-pilot-20260604/merged/selected_edge_geometry_edges.csv
  - raw/assets/benchmark-results/selected-edge-type1-binary-categorical-pilot-20260604/merged/selected_edge_geometry_final.csv
  - raw/assets/benchmark-results/sibling_null_prior_interpolation_audit_20260604/case_summary.csv
  - raw/assets/benchmark-results/sibling_null_prior_interpolation_audit_full_skips_20260604/case_summary.csv
  - raw/assets/benchmark-results/traversal_sibling_fdr_smoke_20260604/synthetic/traversal_sibling_fdr_summary.csv
  - raw/assets/benchmark-results/traversal_sibling_fdr_smoke_20260604/binary/traversal_sibling_fdr_summary.csv
  - wiki/sources/selected-tail-admissibility-domain-20260603.md
  - raw/assets/benchmark-results/selected_tail_admissibility_domain_20260603/context_admissibility_domain.csv
  - raw/assets/benchmark-results/selected_hierarchy_tail_law_binary_boundary_20260603_600/selected_ratio_tail_law.csv
  - wiki/sources/phylogenetic-ml-topological-selected-tail-literature-20260603.md
  - wiki/sources/selected-tail-topology-refinement-20260603.md
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

The 2026-06-04 selected-edge Type-I geometry pilot confirms this direction in
an AWS-sharded binary global-null run with two cases and `40` replicates. In
same-data selected-tree mode, about `99.6%` to `99.8%` of Tree-BH-tested
frontier edges reject, and about `80%` to `91%` of all edge rows reject. In
the fixed-tree control, all final decompositions return one cluster, but
frontier edge rejection is not itself a complete proof baseline: among the
smaller tested frontier, fixed-tree rejection rates range from about `15%` to
`54%`. The pilot therefore supports the selected-edge boundary/action problem
without closing Type-I theory. It also shows that strict sibling-inflation
support failure remains the honest production outcome for selected-tree null
rows when only selected non-null calibration records are available.

The direct categorical extension keeps the same selected-tree conclusion and
adds a second open problem. In the 2026-06-04 binary/categorical pilot,
same-data selected-tree mode rejects about `99.6%` to `99.9%` of
Tree-BH-tested frontier edges across binary and categorical nulls. However,
`cat_highcard_20cat_4c` also shows high fixed-tree frontier rejection rates:
about `77.8%` at edge alpha `0.0001` and `92.3%` at edge alpha `0.001`.
Thus high-cardinality categorical calibration is not only a same-data
selection problem. It also needs a finite-sample categorical one-hot
projected-Wald/Tree-BH calibration analysis under a fixed hierarchy.

The thresholds are now recorded as canonical method constants in
`kl_clustering_analysis/hierarchy_analysis/statistics/alpha_contract.py`, not
as mutable `config.py` values: `DEFAULT_EDGE_ALPHA = 0.001` and
`DEFAULT_SIBLING_ALPHA = 0.01`. These values are deliberately conservative
and still need validation as method constants; the cleanup only made their
assignment explicit and reportable, it did not tune them.

The 2026-06-04 AWS alpha grid gives benchmark evidence for those constants
without closing the Type-I validation gap. In the tested full-suite grid, the
current pair `edge_alpha = 0.001` and `sibling_alpha = 0.01` has the best mean
ARI (`0.893811`). A lower edge alpha (`0.0001` or `0.0003`) gives more exact
cluster-count hits but lower mean ARI. The grid supports keeping sibling alpha
`0.01` as the benchmark default for now, while leaving edge alpha as a
tradeoff between mean ARI and exact cluster-count control.

The traversal sibling-FDR smoke splits the FDR question into four layers. With
valid synthetic null p-values on an always-open three-level traversal tree,
mean FDP is `0.03` at sibling alpha `0.01`, so repeated BH over traversal
depths is not automatically a global FDR guarantee. On `binary_2clusters`,
fixed-tree raw sibling projected-Wald has mean FDP `0.25`; selected-tree raw
Wald has mean FDP `0.95`; and the active inflated selected-tree layer has
`18/20` empirical-inflation support failures. This does not close the final
FDR question, but it shows that we must validate algorithmic FDR, fixed-tree
Wald calibration, selected-tree effects, and inflation support separately.

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
The external calibration contract diagnostic defines the first explicit
production admissibility rule for this object. At \(\alpha_{\mathrm{sib}}=0.01\),
the default tail-resolution target requires at least `499` independent matching
simulations and `499` matched selected records, plus relative simulation
SE(\(\hat c\)) at or below `5%`. No current 500-replicate stratum passes this
production contract. In reliable rows, scalar mean scaling gives zero
rejections at \(\alpha_{\mathrm{sib}}=0.01\), but the resulting p-values are
not uniform; this points toward a selected-ratio tail law if an external
production model is ever attempted.
The selected-hierarchy geometry covariate diagnostic adds the first row-level
decomposition of candidate context variables. In the 100-replicate
representative run, edge-selection strength is the strongest recorded
univariate correlate of \(\log R\), with Spearman correlation about `0.712`
for `negative_log10_min_child_edge_bh_p_value`. Spectral variables are
secondary but visible, and angular alignment with the selected PCA subspace is
high in absolute terms. This refines the open problem: parent size, depth, and
projection dimension alone are unlikely to be enough. A production external
law, if pursued, must test whether edge-selection severity belongs in the
conditioning context and whether eigenvalue/angular variables are needed for
the selected-ratio tail shape.
The candidate-equation pass makes the next equation more concrete. The best
compact top-tail candidate is edge action plus spectral modes:
\[
\log R_u
\sim
A_u+
\log(\lambda_{k,u}/\lambda_{+,u})+
m_{k,u}+
r_{\mathrm{eff},u}.
\]
The full descriptive equation fits mean \(\log R\) better, but this compact
edge-plus-spectral form nearly matches it for top-10% tail discrimination in
the current representative run. This is still a diagnostic equation family,
not a production calibration law.
The held-out 100-replicate rerun strengthens the diagnostic but not the
production claim. Replicate-fold holdout preserves top-tail AUC around `0.969`
for the compact edge-plus-spectral form. Leave-one-case-out transfer reduces
the larger full descriptive equation more strongly, while the compact
edge-plus-spectral form keeps tail AUC around `0.922`. The next open question
is whether this compact law survives a larger panel with non-root contexts,
high-cardinality categorical failures, phylogenetic cases, and strict
row-level selected-null support.
The broad 200-replicate panel shows that this question has two parts. Global
top-tail ranking is easy in the current broad panel: edge-action and
edge/spectral equations have near-perfect held-out tail AUCs. Absolute
selected-ratio scale is harder and varies strongly by family, with mean \(R\)
rising to about `580` in `cat_highd_3cat_500feat`. The full descriptive
equation gives the best broad-panel holdout \(R^2\), but its
leave-one-source-family tail AUC is lower than the simpler edge/spectral
scores. This means the open production target is not "find a high-AUC
equation"; it is to define a selected-ratio tail law with admissible contexts
and calibrated absolute probabilities.
The selected-ratio tail-law diagnostic implements the first within-context
holdout version of that target. It conditions on source family, feature
family, parent-size bin, sibling projection dimension, and binned edge action.
In the 200-replicate broad run, `39` contexts have descriptive held-out tail
folds, `65` have no valid folds, and `0` are production-admissible. Several
high-support small-parent, high-edge-action contexts have held-out exceedance
near `0.01`, but all rows still fail the independent matching-simulation
threshold. The independent unit is now explicit as
`selected_hierarchy_simulation_id`; the largest source-family context reaches
`376` matching simulations, below the `499` production threshold. Sparse root
and low-edge-action contexts remain unstable. This
makes the next open question one of admissible support and context design, not
another scalar heuristic.
Focused follow-ups show that admissibility is possible but narrow. The
300-replicate run admits two small-parent, high-edge-action `gaussian_blobs`
contexts with sibling projection dimension `1` and `2`. The 500-replicate
boundary expansion admits the corresponding `categorical_multinomial`
small-parent, high-edge-action contexts with projection dimension `1` and `2`.
The 600-replicate binary boundary expansion admits `binary_template` only for
projection dimension `1`; the projection-2 context remains support-limited at
`445/499` simulations. Root, medium-parent, large-parent, lower-edge-action,
continuous, and precomputed-distance contexts remain outside the current
admissible production domain. Therefore the open object is no longer simply
"whether support can ever be reached"; it is a context-specific selected-tail
law with explicit admissible and non-admissible regions.
The AWS Batch 1000-replicate per-case run sharpens this point. Seven base
contexts are admissible, but support-rich root and large-parent high-edge
Gaussian contexts still fail the held-out precision contract. Binary
small-parent projection-1 now fails mainly by support in this four-case cloud
panel. The open mathematical question is therefore tail homogeneity: which
geometric variables define a selected region whose ratio tail is stable enough
for calibrated probabilities?
The 2026-06-04 rebuilt-image rerun reproduces the same admissibility set, so
the next question is not whether the seven contexts were a seed artifact. It is
why root/large high-edge Gaussian contexts are support-rich but precision-poor,
and what geometric variable splits those selected regions without turning the
contract into data-adaptive borrowing.

The internal-vs-selected-hierarchy comparison gives one concrete root-level
bridge between the current production estimator and the external diagnostic.
For `dim_diffuse_6c_136f`, internal empirical-null support exists and the
active internal scale agrees with selected-hierarchy scale:
\(c_{\mathrm{internal}}=85.27\) versus \(c_{\mathrm{sel}}=86.23\), with
required blocking scale `82.21`. For `gauss_null_large` and
`cat_highcard_20cat_4c`, the selected-hierarchy diagnostic has matched records,
but internal production support is absent. This separates two questions:
whether selected hierarchy explains observed scale in supported diffuse
Gaussian contexts, and whether unsupported contexts can ever receive a
production-valid external law without borrowing from an invalid support set.

The 2026-06-04 sibling null-prior interpolation audit rechecks the old
tree-neighborhood interpolation idea without restoring it as a production
fallback. It reconstructs old-style interpolated priors from current explicit
edge columns and compares them with strict internal support. In the initial
representative run, `binary_perfect_4c`, `cat_highcard_20cat_4c`,
`overlap_heavy_4c_small_feat`, and `phylo_large_32taxa` all lack strict
internal support, while the diagnostic score still assigns positive weights to
many selected non-null records. In the 24-case full calibration-skip audit,
every case again has `no_strict_internal_support`, and 23 of 24 cases receive
positive interpolated selected-nonnull weights. The exception is
`gauss_extreme_noise_highd`, where even the diagnostic score has no positive
interpolated support. This explains why the earlier interpolation could often
avoid hard support failure, but it also identifies the mathematical problem:
those records are selected non-null evidence and are not admissible
empirical-null calibration support. The diagnostic is therefore useful for
describing the phenomenon, not for changing production calibration.

The bootstrap estimator is not the desired production direction. The selected
inference literature is still useful because it identifies the mathematical
objects: selected regions, signed distances, mean curvature, tangent cones, and
conditional selected-error probabilities. For KL-TE, the analytic target is to
express hierarchy construction, edge opening, and focal sibling selection as a
selected region in the null-whitened tangent chart, then derive or validate the
selected-ratio tail from that geometry. Resampling may remain a diagnostic to
understand the phenomenon, but it should not become a fallback calibration
rule.

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
[[method-proof-web]] makes this proof boundary explicit. The fixed-object
spine is a theorem under fixed feature chart, fixed node pair, valid
covariance whitening, and fixed orthonormal projection. The unproved object is
the selected law
\[
\mathcal L(R_u\mid X\in\mathcal S_u),
\]
where \(\mathcal S_u\) includes hierarchy construction, edge-path opening, and
focal sibling selection. Scalar mean inflation and edge-action proxies are
therefore diagnostics until this selected-region law or an admissible
selected-tail estimator is validated.
[[root-selected-region-model]] defines the first tractable version of that
object: a root sibling context where the selected region is written as the
intersection of hierarchy merge inequalities and root edge-opening
inequalities. This narrows the next proof target to active constraints,
signed-distance/action proxies, tangent cones, curvature terms, and the
conditional root selected-ratio tail.
[[root-selected-region-margins-20260603]] implements the first concrete
observed-root extraction for average linkage. It verifies the selected merge
sequence against the condensed distance matrix and records nearest-competitor
margins for the merge inequalities that construct the two root child clusters.
The representative run shows two selected-region geometries:
tie-heavy Hamming/discretized/categorical root construction cells and
positive-margin continuous Euclidean cells. Schema `v5` computes the
first-order signed distance \(m_t/\|\nabla g_t\|\) and the root
empirical-Gaussian null-whitened distance \(m_t/\sigma_{g,t}\) for smooth
Euclidean average-linkage constraints, while marking the discrete cases as
tie-cell geometry. Schema `v5` also records the fixed-subspace edge-opening
boundary coordinate \(\sqrt Q-\sqrt{q_{1-\alpha,k}}\), its statistic margin,
the root edge-path Tree-BH action, and the local edge/sibling z-identity at a
binary parent. Root selected sibling ratios remain large in both groups, so
merge margins alone are not the law; they are one piece of the selected-region
conditioning object. In the supported eight-case continuous panel, edge-path
radial distance, edge statistic margin, and edge path action each have
Spearman correlation `1.0` with log root sibling ratio, while merge-margin
distances are weak. The same panel verifies that edge and sibling tests use
the same projected barycentric energy before thresholds/FDR/inflation; the
open object is the selected conditioning layer around that shared direction.
Schema `v6` isolates the fixed-projection law conditional only on root
edge-opening:
\[
\Pr(X\ge w\mid X+Y\ge q_{1-\alpha,k_e}).
\]
This law does not explain the current diffuse-dimensional blockers. In the
observed root rows, edge-conditioned sibling p-values remain far below
`SIBLING_ALPHA = 0.01`; the block appears only after current internal
empirical inflation. Therefore the next open object is the relationship
between empirical inflation and the fuller selected-hierarchy law, not another
root edge-opening truncation.
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
inflation-estimator failures. The active code uses traversal-aligned sibling BH
over the edge-reachable frontier; the remaining question is whether this
reachable-frontier family is the right mathematical target under selected
hierarchy traversal.
Pass-through traversal also remains empirical: it can recover descendant
signal, but it can fragment oracle clades because it does not yet compare
descendant split evidence against local sibling-same evidence.

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

The 2026-06-03 hierarchy/gate separation makes this a current full-suite rule,
not only a historical oracle observation. In the latest strict KL-only
full-suite output, `62` of `110` cases are solved, `24` are explicit
calibration-support-undefined skips, `14` are tree/metric unrecoverable,
`4` are oracle-matched below the solved threshold, `4` are gate over-splits,
`1` is a gate under-split, and `1` is a continuous covariance boundary. Thus
only five runnable cases are currently eligible for gate-path statistical
changes before additional hierarchy/metric work. The tree/metric rows must be
handled through representation, distance, linkage, or benchmark construction
analysis first.

The phylogenetic, machine-learning selective-inference, and topological graph
literature sharpens the medium/large-parent selected-tail question. When a
medium or large parent has enough matching simulations but fails held-out tail
precision, the likely problem is not only replicate count. It is that a single
parent-size bin mixes different selected regions: different local subtree
balances, descendant merge topologies, merge-persistence gaps, ancestor
edge-action paths, covariance condition numbers, and spectral alignments.
Phylogenetic comparative methods also warn that precomputed tree or distance
objects require an explicit covariance, branch-length, evolutionary, kernel, or
permutation model before they define a null law. Therefore medium/large and
precomputed-distance contexts need topology-aware diagnostic stratification
before any production external calibration can be considered.

The first topology-refinement diagnostic shows that exact topology-aware
stratification is not yet a production path. On the 300-replicate
Gaussian/categorical panel, balance, topology, merge-persistence, edge-path,
spectral-alignment, and combined exact refinements use data-adaptive bins
learned from the diagnostic rows. Those refined contexts can diagnose
heterogeneity, but they cannot be production-admissible calibration contexts.
The combined exact refinement creates `1,587` contexts, no diagnostic
support-contract passes, and no production-admissible contexts. This shifts
the open question from "which exact topology bins should be matched" to
"which predeclared low-dimensional topology/selection coordinates can model
the tail without destroying independent support."
The same 300-replicate input also rechecks the descriptive equation family:
edge-sampling and edge-spectral equations transfer across source families,
whereas the selected-energy and full descriptive equations fail that split.
Thus the next law should not be a high-dimensional fitted equation; it should
be a low-dimensional predeclared selected-region coordinate model.

The current concrete open questions are:

1. How should the external calibration law condition on hierarchy construction,
   child-parent edge openings, and focal sibling selection?
2. For non-root blockers, does conditioning on the ancestor Tree-BH path keep
   \(c\) small, or create larger selected-context inflation?
3. Which full-selection diagnostic is needed: fixed observed tree, ancestor
   Tree-BH path, focal blocker selection, or full hierarchy reconstruction?
4. Can external Gaussian sibling-null calibration explain the Gaussian
   blockers?
5. Does external inflation depend mainly on edge-selection severity, \(p/n\),
   projection dimension, eigenvalue concentration, angular alignment, tree
   selection, covariance whitening, or feature family?
6. When should the calibration hierarchy return
   \(\hat c_{\mathrm{internal}}\), \(\hat c_{\mathrm{external}}\), or fail
   undefined?
7. What minimum matched selected-hierarchy support and Monte Carlo precision
   are required before an external calibration estimate is admissible?
   Current diagnostic contract: at \(\alpha_{\mathrm{sib}}=0.01\), require at
   least `499` independent matching simulations, `499` matched selected
   records, and relative simulation SE(\(\hat c\)) no larger than `5%`.
   Current evidence fails this production contract.
8. Which context variables should be exact matching variables, and which should
   be descriptive stratifiers, for selected-hierarchy null studies?
   Current evidence says parent-size scale is informative, exact depth is
   sparse, and selected-ratio upper quantiles must be part of the support
   contract if a production external selected-hierarchy null is ever attempted.
   Current external-contract stratum variables are case, feature family, \(n\),
   \(p\), sibling projection dimension, and parent-size bin. The geometry
   covariate diagnostic adds edge-selection severity as the strongest current
   candidate context variable, with eigenvalue and angular summaries as
   candidate tail-shape variables rather than validated matching rules. The
   broad geometry panel further separates global top-tail ranking from
   calibrated selected-ratio scale; a high held-out AUC alone is not enough for
   production external calibration. The selected-ratio tail-law diagnostic
   confirms that explicit edge-action contexts are still support-limited at
   200 replicates.
   Focused selected-tail runs show that small-parent, high-edge-action
   Gaussian, categorical, and binary projection-1 contexts are admissible.
   Binary projection-2, root, medium-parent, large-parent, lower-edge-action,
   continuous, and precomputed-distance contexts remain outside the current
   admissible domain.
   Literature on selective clustering inference, phylogenetic covariance, and
   topological graph summaries says medium/large-parent failures should test
   subtree topology, balance, merge persistence, branch/covariance condition,
   and ancestor edge-action variables rather than treating parent size as the
   only scale coordinate.
   The first exact-refinement test is negative for production use: exact
   topology-aware bins expose heterogeneity, but they are data-adaptive and
   fragment medium/large support. The next version should test a modeled
   selected-tail law or a coarser predeclared topology coordinate, not a
   many-way exact match.
   The root selected-region margin diagnostic adds merge-inequality geometry
   as a candidate descriptive variable. It also shows that discrete
   tie-heavy hierarchy cells and continuous positive-margin hierarchy cells
   should not be pooled without a mathematical reason.
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
    selection-aware threshold can be derived or validated. The no-bootstrap
    analytic route is to move from the identity-population MP edge
    \(H=\delta_1\) to a local deformed MP law when the null-whitened tangent
    spectrum has non-identity population law \(H_u\). The relevant calculation
    is the Stieltjes-transform integral equation and support-edge inverse map,
    not an empirical bootstrap threshold.
    The 2026-06-02 local identity-law screen confirms that this is a real
    validation question: Bernoulli/discretized selected spectra often sit near
    or above the identity MP edge, while continuous empirical-covariance
    spectra sit far below the identity MP positive support because same-node
    empirical covariance whitening gives positive eigenvalues at the centered
    self-whitening scale \((m_u-1)/m_u\). The finite identity-null comparison
    further separates selected spectral inflation from ordinary finite-sample
    fluctuation: Gaussian-null, high-dimensional Bernoulli, and diffuse
    discretized Gaussian selected nodes exceed the finite-null 95% top edge in
    about `54.5%`, `72.1%`, and `70.5%` of evaluated nodes, while the two
    categorical screens are closer to ordinary finite fluctuation at about
    `7.0%` and `12.0%`.
    The 2026-06-03 selected-tree spectral-law rerun refines the categorical
    branch: `cat_highcard_20cat_4c` behaves like a small set of selected
    extreme nodes with strong node-size/aspect-ratio relationships, while
    `cat_highd_3cat_500feat` has a few very large root/half-tree extremes and
    many tiny selected extreme nodes. Therefore Bernoulli/discretized spectra
    need a selected-tree spectral inflation law, but categorical spectra need
    an extreme-node analysis tied to multinomial covariance, projection
    dimension, and sibling-testing decisions.
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
24. Does traversal-aligned sibling BH over the edge-reachable frontier control
    the intended sibling false-split target? The 2026-06-04 smoke says this
    must be decomposed into algorithmic repeated-BH behavior, fixed-tree
    projected-Wald calibration, selected-tree p-value distortion, and
    empirical-inflation support failure.
25. Should sibling FDR additionally condition on the edge path, traversal
    depth, or selected hierarchy, or use a more explicit hierarchical target?
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
    defaults? The full alpha grid supports the current sibling alpha and gives
    evidence for the current edge alpha, but it is still benchmark-performance
    evidence rather than a selected-tree null calibration proof.
35. What null calibration study supports edge alpha beyond benchmark ARI?
36. What null calibration and power study supports sibling alpha beyond the
    current full-suite grid?
37. What simulation validates final cluster-count control, not only local
    p-values?
38. What planted-structure simulations support power claims?
39. Which real biological application should become the first
    manuscript-ready result?
40. Which generated outputs should be promoted into a locked
    manuscript-results manifest?
41. How should the manuscript state every remaining approximation without
    overstating Type-I error control?
42. How should observed root merge margins be lifted from ambient first-order
    Euclidean distance to the actual selected-region law: null-whitened signed
    distance, active-set and tangent-cone structure, curvature, or a discrete
    tie-cell law?
43. Given the verified local edge/sibling barycentric z-identity, how should
    the fixed-subspace edge-opening radial distance and edge-path Tree-BH
    action be lifted to the full selected law with selected projection,
    Tree-BH selection cells, sibling FDR/inflation, and non-root focal sibling
    contexts?
44. Why does the internal empirical-inflation layer block some root diffuse
    dimensional contexts when the raw and edge-conditioned sibling tails are
    still strongly significant? The current diagnostic says the blocker is not
    the fixed-projection root edge-opening law alone. The open question is
    whether internal empirical inflation is estimating a real fuller
    selected-hierarchy tail, over-penalizing selected non-null contexts, or
    mixing incompatible support regimes.

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
- `wiki/sources/selected-geometry-mp-integral-literature-20260602.md` records
  the selected-region geometry literature and the Stieltjes-transform integral
  route for general Marchenko--Pastur spectra.
- `wiki/sources/local-mp-identity-law-diagnostic-20260602.md` records the
  representative production-spectrum screen against the identity MP law.
- `wiki/sources/root-selected-region-margins-20260603.md` records observed
  root merge-selection margins, ambient and null-whitened first-order
  Euclidean signed-distance geometry for supported continuous representatives,
  fixed-subspace edge-opening boundary/action fields, edge/sibling
  z-relationship fields, and tie-cell status for discrete/nonsmooth
  representatives.
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
- `wiki/sources/alpha-grid-full-20260604.md` records the AWS full-suite alpha
  grid over `25` alpha pairs and separates benchmark evidence from Type-I
  calibration proof.

## Links

- [[kl-te-method]]
- [[projected-wald-statistic]]
- [[local-marchenko-pastur-rule]]
- [[dimensional-gaussian-representation-diagnostic]]
- [[oracle-gate-path-diagnostic]]
- [[top-down-traversal]]
- [[manuscript-life-science-readiness]]
