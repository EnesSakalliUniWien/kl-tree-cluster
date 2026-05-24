# Edge and Sibling Derivation Audit

This guide tracks the derivation chain used by the manuscript and marks the
places where mathematics, implementation, and validation still need to meet.

## Child-Parent Edge Test

For a parent node `u` and child `c`, the method compares the child feature-rate
vector to the parent feature-rate vector. Because the child leaves are included
inside the parent leaves, this is a nested comparison rather than an ordinary
independent two-sample comparison.

Derivation chain:

1. Define descendant-leaf rates for `u` and `c`.
2. Compute the nested child-parent contrast.
3. Use the nested Bernoulli-style variance formula.
4. Standardize coordinates.
5. Project onto parent-local orthonormal PCA directions.
6. Sum squared projected coordinates.
7. Compare to a chi-square law with the selected projected dimension.

[PROOF GAP: the manuscript still needs a clean derivation showing the exact
finite-sample conditions under which the nested variance formula is unbiased or
consistent.]

[VALIDATION GAP: correlated one-hot features and discretized continuous
features violate the independent Bernoulli coordinate model and need simulation
or sensitivity checks.]

## Projection and Eigenvectors

The local spectral representation computes eigenvectors from the local
correlation-scale subtree matrix. The eigenvalues select candidate directions
through the Marchenko-Pastur upper edge. The current projected-Wald statistic
does not weight the null distribution by those eigenvalues; it uses the
orthonormal projection rows and a chi-square reference.

[PROOF GAP: if the PCA directions are selected from the same data being tested,
the fixed-subspace chi-square argument is approximate. The manuscript must
either derive the effect of this selection or validate it empirically.]

## Sibling Test

For a binary parent, the sibling test compares the two child feature-rate
vectors after coordinatewise standardization. The projected statistic is
computed in a parent-local orthonormal basis. The raw sibling statistic is
anti-conservative when the hierarchy has selected unusually separated sibling
pairs from the same data.

The implemented correction is context-weighted empirical-null inflation:

1. Build sibling records across binary parents.
2. Assign each record an empirical-null weight from the two child-parent
   adjusted p-values.
3. Define context by the sibling projection dimension.
4. Smooth over nearby contexts.
5. Estimate inflation as a weighted ratio of observed raw statistic to
   projected-Wald reference scale.
6. Divide the focal raw statistic by the estimated inflation.

[DEFINITION GUARD: the empirical-null weight is a monotone calibration weight,
not a posterior probability of the sibling null.]

[VALIDATION GAP: the inflation estimator needs null calibration, sensitivity to
few calibration records, sensitivity to the bandwidth rule, and comparison
against uninflated sibling testing.]

## Multiplicity and Traversal

The edge stage uses tree-aware multiple-testing correction. The sibling stage
uses corrected focal-pair p-values. The final traversal combines corrected edge
and sibling decisions with pass-through enabled.

[VALIDATION GAP: local-test calibration is not enough. The final number of
clusters must be evaluated under null and planted-structure simulations.]
