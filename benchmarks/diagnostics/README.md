# Benchmark Diagnostics

This directory contains investigation tools for benchmark behavior. These
scripts may read benchmark outputs, build oracle traces, or simulate
calibration diagnostics, but they are not part of the production clustering
method.

Use `benchmarks/shared/` for reusable benchmark execution contracts. Add files
here only when they are diagnostic entrypoints or diagnostic-only helpers.

## Layout

- `oracle/`: subtree-cut recoverability and gate-path traces.
- `calibration/`: sibling inflation and selection-conditioned null diagnostics.
- `spectral/`: Marchenko-Pastur and projection-dimension diagnostics.
- `math_trace/`: trace-schema validation and deterministic mathematical
  failure attribution for benchmark run artifacts.
- `failure/`: benchmark failure-report tracing used by the full benchmark.
- `analysis/`: post-run result analysis and durable diagnostic notes.

Maintained entrypoints:

- `oracle/run_oracle_tree_recoverability.py`
- `oracle/run_gate_path_trace.py`
- `calibration/run_sibling_inflation_diagnostic.py`
- `calibration/run_gaussian_sibling_null_calibration.py`
- `calibration/run_selection_conditioned_sibling_null.py`
- `calibration/run_tree_bh_selection_conditioned_sibling_null.py`
- `calibration/sample_split_selection_audit.py`
- `calibration/selected_hierarchy_null_audit.py`
- `calibration/selected_hierarchy_external_calibration_contract.py`
- `calibration/selected_hierarchy_stratification_diagnostic.py`
- `calibration/selected_hierarchy_geometry_covariates.py`
- `calibration/root_selected_region_margins.py`
- `calibration/selected_tail_parent_size_balance_stability.py`
- `calibration/selected_tail_promotion_gate.py`
- `calibration/internal_support_threshold_validation.py`
- `calibration/sibling_null_weight_rule_validation.py`
- `calibration/edge_null_calibration_panel.py`
- `calibration/sibling_null_calibration_panel.py`
- `calibration/traversal_guard_validation_panel.py`
- `calibration/production_admissibility_contract.py`
- `calibration/selected_edge_sibling_null_equation.py`
- `calibration/selected_edge_sibling_postrun_analysis.py`
- `calibration/differential_statistic_validity_panel.py`
- `calibration/regularized_wald_statistic_panel.py`
- `calibration/null_law_decomposition_panel.py`
- `calibration/statistic_distribution_shape_panel.py`
- `calibration/covariance_laplacian_panel.py`
- `calibration/data_independent_sibling_gate_panel.py`
- `calibration/data_independent_sibling_gate_traversal_panel.py`
- `calibration/fixed_sibling_gate_profile_validation.py`
- `calibration/selected_family_traversal_panel.py`
- `spectral/cosine_band_coherence_comparator.py`
- `analysis/analyze_relationships.py`
- `spectral/compare_mp_dimension_contracts.py`
- `spectral/mp_projection_dimension_behavior_sweep.py`
- `spectral/sibling_projection_dimension_rule_grid.py`
- `open_questions/full_diagnostic_contract.py`
- `math_trace/infer_benchmark_math.py`

`calibration/fixed_sibling_gate_profile_validation.py` is the current shared
runner smoke for fixed sibling-gate profiles, root-stability metadata, and the
default-off selected-root permutation guard. Its successful rows remain
diagnostic evidence unless the production-admissibility outputs pass the
confidence contract. Use `fixed_coordinate_selective_root_v1` to exercise the
packaged selected-root permutation candidate directly, and
`fixed_coordinate_selective_passthrough_v1` to exercise the narrower
pass-through descendant selected-subtree guard.

`calibration/selected_family_traversal_panel.py` compares baseline traversal
with the fixed-coordinate selected-root/pass-through profile family and writes
multi-scale node, region, and sample outputs. Use
`fixed_coordinate_global_passthrough_refined_v1` as the primary binary V1
selected-family diagnostic candidate; it remains validation-only, not a
production default.

`scripts/analysis/multiscale_umap_overlay.py` joins
`multiscale_gene_assignments.csv` to existing UMAP coordinates and renders a
stable-region-first overlay with pass-through or guard zones marked separately.

`spectral/cosine_band_coherence_comparator.py` ports the old c2ef fixed
cosine-band and biological-coherence checks into the current diagnostic stack.
It keeps the historical predeclared bands and runs current gate/decomposition
code on each band tree; the outputs are comparator evidence, not production
calibration.
