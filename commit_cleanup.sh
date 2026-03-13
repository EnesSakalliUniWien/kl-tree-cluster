#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

git add -A

git commit -m "refactor: remove dead code — unused sibling methods, spectral estimators, edge calibration, and defensive wrappers

Removed modules:
- cousin_calibrated_test.py (cousin F-test sibling method)
- cousin_tree_guided.py (tree-guided cousin sibling method)
- cousin_weighted_wald.py (weighted Wald sibling method)
- edge_calibration.py (edge calibration module)
- compare_sibling_methods.py (benchmark comparing deleted methods)

Removed functions/classes:
- estimate_k_effective_rank, estimate_k_active_features, count_active_features (k_estimators.py)
- EFFECTIVE_RANK, ACTIVE_FEATURES enum members (SpectralKMethod)
- Poisson KL divergence (_kl_poisson), composite KL fields
- WeightedCalibrationModel (consolidated to CalibrationModel)
- Satterthwaite path (hardcoded eigenvalue whitening)

Removed defensive code:
- Triple-layer try/except around projected Wald kernel and both callers
- hasattr(R, 'dot') dispatch (R is always np.ndarray)

Removed dead config options:
- FELSENSTEIN_SCALING, EDGE_CALIBRATION_*
- SIBLING_TEST_METHOD options: cousin_ftest, cousin_tree_guided, cousin_weighted_wald
- SPECTRAL_METHOD options: effective_rank, active_features

Kept intact:
- effective_rank() utility (used by adaptive projection floor)
- marchenko_pastur (default spectral method) + None (JL fallback)
- wald + cousin_adjusted_wald (sibling methods)
- Branch length scaling (Felsenstein)

Tests: 191 passed, 4 skipped, 0 failures
Net: -4,727 lines across 47 files"
