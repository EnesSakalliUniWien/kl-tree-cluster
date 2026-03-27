# Tests Overview

The suite is organized by pipeline layer so you can validate fast structural behavior first, then
statistics, traversal, pipeline contracts, integration, and visualization.

## Directory Structure

```text
tests/
├── conftest.py
├── test_cases_config.py
├── core/
├── statistics/
├── localization/
├── validation/
├── pipeline/
├── integration/
└── visualization/
```

## Current Suite Map

### Shared fixtures and repo-wide regressions

- `conftest.py`
- `test_cases_config.py`
- `test_methodology_fixes.py`

### Core tree/decomposition primitives (`core/`)

- `10_test_poset_tree.py`
- `12_test_cluster_assignments.py`
- `13_test_cluster_decomposer_threshold.py`
- `16_test_data_utils_bool_extraction.py`

### Statistical kernels and config wiring (`statistics/`)

- `22_test_edge_branch_length_regression.py`
- `25_test_per_test_projection_seeding.py`
- `26_test_invalid_nonfinite_handling.py`
- `27_test_categorical_distributions.py`
- `29_test_method_k_estimators_parity.py`
- `30_test_gate_adapter_parity.py`
- `31_test_registry_config_wiring.py`

### Traversal and localization behavior (`localization/`)

- `33_test_skip_reason_propagation_integration.py`
- `35_test_gates_traversal.py`

### Cluster validation stack (`validation/`)

- `40_test_cluster_validation_core.py`
- `41_test_independent_cluster_validation.py`
- `42_test_cluster_validation_integration.py`
- `43_test_result_status_validation.py`

### Pipeline contracts and reporting artifacts (`pipeline/`)

- `50_test_attention_pipeline.py`
- `51_test_dispatch_contract.py`
- `52_test_method_execution_index_alignment.py`
- `53_test_runner_contract_alignment.py`
- `55_test_case_run_audit_env_restore.py`
- `57_test_pdf_utils.py`
- `58_test_pipeline_pdf_behavior.py`
- `59_test_pipeline_pdf_naming.py`
- `60_test_case_execution_pdf_cover_mode.py`
- `61_test_benchmark_relationship_analysis.py`

### Benchmark and generator smoke tests (`integration/`)

- `60_test_benchmark_methods_smoke.py`
- `61_test_sbm_integration.py`
- `62_test_phylogenetic_generator.py`

### Visualization and layout (`visualization/`)

- `70_test_cluster_tree_layout.py`
- `71_test_cluster_tree_visualization.py`

## Recommended `pytest` Execution Order

Use the helper script when you want the staged order the repository expects:

```bash
python scripts/run_tests_ordered.py --list
python scripts/run_tests_ordered.py --stage 1
python scripts/run_tests_ordered.py
```

Or run the suites directly:

```bash
# 1) Core structure + decomposition
pytest tests/core/

# 2) Statistical engines + calibration
pytest tests/statistics/

# 3) Traversal and gate behavior
pytest tests/localization/

# 4) Cluster validation stack
pytest tests/validation/

# 5) Pipeline contracts + reporting artifacts
pytest tests/pipeline/

# 6) Integration smoke + visualization
pytest tests/integration/ tests/visualization/

# 7) Full suite
pytest
```

## Notes

- Grouping is by intent, not by a strict import dependency graph.
- If you touched decomposition logic, start with `tests/core/`, `tests/statistics/`, and `tests/localization/`.
- For benchmark or artifact-generation changes, prioritize `tests/pipeline/` and `tests/integration/`.
