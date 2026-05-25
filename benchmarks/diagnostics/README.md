# Benchmark Diagnostics

This directory contains investigation tools for benchmark behavior. These
scripts may read benchmark outputs, build oracle traces, or simulate
calibration diagnostics, but they are not part of the production clustering
method.

Use `benchmarks/shared/` for reusable benchmark execution contracts. Add files
here only when they are diagnostic entrypoints or diagnostic-only helpers.

Maintained entrypoints:

- `run_oracle_tree_recoverability.py`
- `run_gate_path_trace.py`
- `run_sibling_inflation_diagnostic.py`
- `run_gaussian_sibling_null_calibration.py`
- `run_selection_conditioned_sibling_null.py`
- `run_tree_bh_selection_conditioned_sibling_null.py`
- `analyze_relationships.py`

