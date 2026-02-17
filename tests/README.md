# Tests Overview

This document organizes test files by **purpose** in dedicated subdirectories, providing a practical
execution order (fast/core first, then broader integration).

## Directory Structure

```
tests/
├── conftest.py                    # pytest fixtures & configuration
├── improved_test_cases.py         # shared test case definitions
├── test_cases_config.py           # shared test configuration
├── core/                          # Stage 1: Core tree/decomposition
├── statistics/                    # Stage 2: Statistical validity/calibration
├── localization/                  # Stage 3: Signal localization/merge
├── validation/                    # Stage 4: Cluster validation
├── pipeline/                      # Stage 5: Pipeline contracts/artifacts
├── integration/                   # Stage 6: Benchmark integration
└── visualization/                 # Stage 7: Visualization
```

## Purpose-Based Test Organization

### 1) Test infrastructure & shared fixtures
- `conftest.py` (pytest special file)
- `improved_test_cases.py` (imported by other tests)
- `test_cases_config.py` (imported by other tests)

### 2) Core tree/decomposition primitives (`core/`)
- `10_test_poset_tree.py`
- `11_test_tree_decomposition_distance.py`
- `12_test_cluster_assignments.py`
- `13_test_cluster_decomposer_threshold.py`
- `14_test_local_kl_utils.py`

### 3) Statistical validity & calibration internals (`statistics/`)
- `20_test_clt_validity.py`
- `21_test_random_projection.py`
- `22_test_edge_branch_length_regression.py`
- `23_test_weighted_calibration.py`
- `24_test_weighted_calibration_diagnostic.py`
- `25_test_per_test_projection_seeding.py`
- `26_test_invalid_nonfinite_handling.py`
- `27_test_categorical_distributions.py`

### 4) Signal localization & merge behavior (`localization/`)
- `30_test_signal_localization.py`
- `31_test_posthoc_merge.py`
- `32_test_posthoc_merge_calibration.py`
- `33_test_skip_reason_propagation_integration.py`

### 5) Cluster validation stack (`validation/`)
- `40_test_cluster_validation_core.py`
- `41_test_independent_cluster_validation.py`
- `42_test_cluster_validation_integration.py`
- `43_test_result_status_validation.py`

### 6) Pipeline/runner/contracts & artifact behavior (`pipeline/`)
- `50_test_attention_pipeline.py`
- `51_test_dispatch_contract.py`
- `52_test_method_execution_index_alignment.py`
- `53_test_runner_contract_alignment.py`
- `54_test_compare_sibling_methods_contract.py`
- `55_test_case_run_audit_env_restore.py`
- `56_test_run_weighted_full_import_safety.py`
- `57_test_pdf_utils.py`
- `58_test_pipeline_pdf_behavior.py`
- `59_test_pipeline_pdf_naming.py`

### 7) Benchmark/generator integration smoke (`integration/`)
- `60_test_benchmark_methods_smoke.py`
- `61_test_sbm_integration.py`
- `62_test_phylogenetic_generator.py`

### 8) Visualization/layout (`visualization/`)
- `70_test_cluster_tree_layout.py`
- `71_test_cluster_tree_visualization.py`

## Recommended `pytest` Execution Order

Run these in sequence when debugging or validating incremental changes:

You can also run the same stages via the helper script:

```bash
python scripts/run_tests_ordered.py --list
python scripts/run_tests_ordered.py --stage 1
python scripts/run_tests_ordered.py
```

```bash
# 1) Core structure + decomposition
pytest tests/core/

# 2) Statistical engines + calibration
pytest tests/statistics/

# 3) Localization + post-hoc merge behavior
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

- Grouping is by intent, not by strict dependency graph.
- If you only touched decomposition logic, run sections 1–4 first.
- For benchmark/pipeline changes, prioritize sections 6 and 7.
