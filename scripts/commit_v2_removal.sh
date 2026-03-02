#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

git add -A

git commit -m "refactor: remove signal localization (v2) from codebase

v2 achieved mean ARI 0.431 vs v1's 0.757 across 74 benchmark cases.
The localization sub-tests lack power (small sub-samples + BH penalty
-> false similarity edges -> incorrect merges). Keeping it as dead code
creates maintenance burden with no practical benefit.

Deleted:
- kl_clustering_analysis/hierarchy_analysis/signal_localization.py
- tests/localization/30_test_signal_localization.py
- tests/localization/34_test_v2_bug_fixes.py
- tests/core/11_test_tree_decomposition_distance.py (tested _test_node_pair_divergence, v2-only)
- notebooks/run_v2_digits.py
- notebooks/run_v2_benchmark_small.py
- debug_scripts/diagnose_v2.py

Modified (v2 code removed):
- gates.py: removed V2TraversalState, should_split_v2, process_node_v2, _check_edge_significance
- tree_decomposition.py: removed decompose_tree_v2, _test_node_pair_divergence, use_signal_localization params
- poset_tree.py: removed v2 dispatch branch
- config.py: removed USE_SIGNAL_LOCALIZATION, LOCALIZATION_MAX_DEPTH, LOCALIZATION_MAX_PAIRS
- kl_runner.py: removed _run_kl_v2_method
- 35_test_gates_traversal.py: removed TestV2TraversalState, TestProcessNodeV2, v2-related tests
- 32_test_posthoc_merge_calibration.py, 12, 13, 41, 42, 50 tests: removed use_signal_localization=False kwargs
- scripts/run_tests_ordered.py: removed stale test file references

Added:
- debug_scripts/diagnostics/analyze_overlap_cases.py
- debug_scripts/diagnostics/sweep_overlap_configs.py

299 passed, 4 skipped"
