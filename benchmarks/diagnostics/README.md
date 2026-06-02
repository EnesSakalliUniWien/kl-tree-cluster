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
- `analysis/analyze_relationships.py`
- `spectral/compare_mp_dimension_contracts.py`
