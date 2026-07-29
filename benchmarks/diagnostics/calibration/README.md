# Calibration Diagnostics

This package contains investigation and validation tools for calibration
behavior. It is not a production calibration layer. Production tree
construction, testing, and plotting interfaces remain under
`tree_break_selection/`.

## Categories

- `edge/`: edge-null behavior and selected-edge post-run analysis.
- `overlap/`: overlap, junction, recovery, and structural-threshold studies.
- `root/center/`: robust root-center and rooted-tree geometry checks.
- `root/selected/`: selected-root region, support, and spectral-tail studies.
- `root/tie_rank/`: tie-rank-conditioned root proposals and replay panels.
- `selected/family/`: selected-family traversal and candidate-law studies.
- `selected/hierarchy/`: hierarchy selection and external-calibration studies.
- `selected/neighborhood/`: neighborhood conditioning and spectral-flow studies.
- `selected/tail/`: selected-tail law and promotion-gate studies.
- `sibling/gates/`: data-independent sibling-gate profiles.
- `sibling/nulls/`: sibling-null models, calibration panels, and their runners.
- `spectral_transport/`: spectral-transport dispatch and threshold studies.
- `statistics/`: covariance, test-statistic, and null-law studies.
- `traversal/`: traversal guards, path conditioning, and admissibility checks.

The matching tests live under `tests/validation/calibration/` with the same
category path. Import modules from their owning category; the package roots do
not re-export moved names.

Shared diagnostic contracts remain at this package root:

- `reporting.py` writes named table bundles and their manifest envelope.
- `overlap/binary_threshold_scan.py` owns two-direction threshold scans for
  binary truth/negative roles.
- `root/root_tail_values.py` owns shared selected-root value parsing,
  support-role classification, log-action transforms, spectral-excess
  transforms, and T,A,E,B,H_u stratum keys.
- `root/selected/root_tail_action_support.py` owns the common selected-root
  action-support geometry used by the conditioning and dominance panels.

## Maintained runners

- `selected/family/run_selected_family_matrix.py`
- `sibling/nulls/run_gaussian_sibling_null_calibration.py`
- `sibling/nulls/run_selection_conditioned_sibling_null.py`
- `sibling/nulls/run_sibling_inflation_diagnostic.py`
- `sibling/nulls/run_tree_bh_selection_conditioned_sibling_null.py`

Multiscale UMAP rendering is owned by
`selected/family/multiscale_umap.py` beside the selected-family matrix runner.
It is a calibration-specific overlay, not a production plotting interface.
