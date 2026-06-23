# Tree-Break Selection Method Logic Map

This guide records the current mathematical contract behind the manuscript.
It is not a results section.

## Central Mathematical Object

Tree-Break Selection starts from a typed sample-feature matrix and a rooted binary
agglomerative hierarchy. The active feature-space contract may contain
Bernoulli coordinates, categorical one-hot blocks with drop-last multinomial
covariance, continuous empirical-Gaussian blocks, or future mixed blocks. Each
internal node defines a parent subtree with two child subtrees. The method
computes subtree feature distributions, annotates child-parent edges and sibling
pairs with projected-Wald tests, corrects the edge and sibling p-values, and
walks the tree top down to decide the final partition.

## Main Assumptions

- The working feature representation has an explicit feature-space contract.
- Subtree membership is treated as fixed when a local p-value is evaluated.
- Coordinatewise Bernoulli variance formulas are exact only for independent
  Bernoulli coordinates with fixed groups; categorical and continuous blocks use
  their declared covariance models and require separate validation.
- Local PCA directions are treated as fixed once selected.
- The orthonormal projected-Wald reference uses a chi-square law for a fixed
  projected subspace under an isotropic standardized null.
- The empirical sibling inflation model is a calibration model, not an exact
  selective-inference theorem.
- The sibling contrast covariance used to standardize a left-right test and
  the parent spectral covariance used to choose projection directions are
  different covariance objects. Diagnostics may compare both, but production
  currently uses the orthonormal projected-Wald reference after standardization
  rather than a covariance-weighted quadratic reference.

## Estimator and Statistic Chain

1. Estimate node-level feature distributions from descendant leaves.
2. Form child-parent and sibling contrast vectors.
3. Standardize contrasts using the covariance model declared by the active
   feature-space block.
4. Build parent-local PCA directions from the local subtree representation.
5. Select the projection dimension with the local Marchenko-Pastur rule plus
   implementation floor and cap.
6. Compute the projected quadratic statistic in an orthonormal basis.
7. Use a chi-square reference for the uninflated projected-Wald statistic.
8. For sibling tests, estimate context-weighted empirical-null inflation from
   sibling records and divide the raw statistic by that inflation.
9. Correct p-values and traverse the tree.

## Covariance Object Separation

The sibling contrast covariance is the feature-space covariance of the
left-right contrast being tested. For Bernoulli blocks this is coordinatewise
under the current feature-space contract, and its graph-Laplacian diagnostic is
therefore diagonal unless a non-diagonal block model is declared.

The parent spectral covariance is the covariance of descendant null-whitened
tangent rows used to build the parent-local spectral basis and choose the
projection dimension. It can be dense even when the sibling contrast covariance
is diagonal, because it summarizes the selected local geometry of descendants
rather than the direct two-sample contrast covariance.

Covariance-inferred Satterthwaite degrees of freedom and scales are diagnostic
comparators for distribution-shape analysis. They do not replace the
implemented chi-square reference unless a separate production-admissibility
contract promotes such a rule.

## Canonical Terminology

- Use **projected-Wald statistic** for the projected quadratic form.
- Use **orthonormal projected-Wald reference** for the chi-square reference
  after projection onto orthonormal rows.
- Use **context-weighted empirical-null inflation** for the sibling calibration
  model.
- Do not call the sibling inflation model an exact selective p-value.
- Do not call the empirical-null weight a posterior null probability.

## Constants and Defaults Requiring Validation

- Edge alpha: `0.001`.
- Sibling alpha: `0.01`.
- Marchenko-Pastur upper-edge dimension rule.
- Minimum spectral dimension: `2`.
- Spectral row set: descendant leaves only in the inferential PCA basis.
- Sibling projection dimension: geometric mean of child edge dimensions.
- Empirical-null weight from joint child-edge null-evidence weights.
- Context bandwidth within feature family over active log-context axes.
- Effective calibration sample size as a diagnostic.
- External selected-hierarchy selected-tail law for unsupported internal
  calibration contexts: partially demonstrated only in narrow small-parent,
  high-edge-action contexts for Gaussian, categorical, and binary
  projection-1 settings; undefined elsewhere.
- Pass-through traversal.

## Empirical Claims Not Yet Manuscript-Ready

- Null calibration of the edge stage.
- Null calibration of the sibling stage after empirical-null inflation.
- Full-pipeline control of the final number of clusters.
- Power under planted binary subtree structure.
- Robustness to sparse high-dimensional feature matrices.
- Robustness to one-hot categorical dependence.
- Robustness to categorical one-hot covariance, continuous empirical-Gaussian
  covariance, and discretized Gaussian benchmark variants.
- External selected-tail calibration outside the currently admissible narrow
  small-parent, high-edge-action contexts.
- Real-data interpretability.

## Required Before Submission

[RESULTS GAP: add locked simulation and benchmark outputs with seeds, commit,
configuration manifest, and confidence intervals.]

[VALIDATION GAP: add ablations for every implementation constant listed in the
method-constants table.]

[PROOF GAP: state the exact assumptions under which the projected chi-square
reference is valid after data-dependent local projection, or present it as a
working approximation requiring empirical validation.]
