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

## Maintained runners

- `selected/family/run_selected_family_matrix.py`
- `sibling/nulls/run_gaussian_sibling_null_calibration.py`
- `sibling/nulls/run_selection_conditioned_sibling_null.py`
- `sibling/nulls/run_sibling_inflation_diagnostic.py`
- `sibling/nulls/run_tree_bh_selection_conditioned_sibling_null.py`

Multiscale UMAP rendering is owned by
`tree_break_selection/plot/multiscale_umap.py` and is exercised by the
selected-family matrix runner. Calibration diagnostics should use that engine
instead of adding another plotting wrapper.
