# Benchmark Diagnostics

This directory contains investigation tools for benchmark behavior. These
scripts may read benchmark outputs, build oracle traces, or simulate
calibration diagnostics, but they are not part of the production clustering
method.

Use `benchmarks/shared/` for reusable benchmark execution contracts. Add files
here only when they are diagnostic entrypoints or diagnostic-only helpers.

## Layout

- `oracle/`: subtree-cut recoverability and gate-path traces.
- `calibration/`: responsibility-grouped calibration diagnostics; see
  `calibration/README.md` for the source and mirrored-test map.
- `spectral/`: Marchenko-Pastur and projection-dimension diagnostics.
- `math_trace/`: trace-schema validation and deterministic mathematical
  failure attribution for benchmark run artifacts.
- `failure/`: benchmark failure-report tracing used by the full benchmark.
- `analysis/`: post-run result analysis and durable diagnostic notes.

`runner_support.py` owns classified-case selection, serial runtime defaults,
and correctly rooted timestamped result directories shared by diagnostic
entrypoints. `oracle/gate_path_trace.py` owns prepared tree/gate/traversal/oracle
context. `calibration/sibling/nulls/runner_support.py` adds the sibling-null
tables and target-mode selection to that prepared context.

Maintained entrypoints:

- `oracle/run_oracle_tree_recoverability.py`
- `oracle/run_gate_path_trace.py`
- `calibration/sibling/nulls/run_sibling_inflation_diagnostic.py`
- `calibration/sibling/nulls/run_gaussian_sibling_null_calibration.py`
- `calibration/sibling/nulls/run_selection_conditioned_sibling_null.py`
- `calibration/sibling/nulls/run_tree_bh_selection_conditioned_sibling_null.py`
- `calibration/selected/family/run_selected_family_matrix.py`
- `spectral/cosine_band_coherence_comparator.py`
- `analysis/analyze_relationships.py`
- `spectral/compare_mp_dimension_contracts.py`
- `spectral/mp_projection_dimension_behavior_sweep.py`
- `spectral/sibling_projection_dimension_rule_grid.py`
- `open_questions/full_diagnostic_contract.py`
- `math_trace/infer_benchmark_math.py`

`calibration/sibling/gates/fixed_sibling_gate_profile_validation.py` is the current shared
runner smoke for fixed sibling-gate profiles, root-stability metadata, and the
default-off selected-root permutation guard. Its successful rows remain
diagnostic evidence unless the production-admissibility outputs pass the
confidence contract. Use `fixed_coordinate_selective_root_v1` to exercise the
packaged selected-root permutation candidate directly, and
`fixed_coordinate_selective_passthrough_v1` to exercise the narrower
pass-through descendant selected-subtree guard.

`calibration/selected/family/selected_family_traversal_panel.py` compares baseline traversal
with the fixed-coordinate selected-root/pass-through profile family and writes
multi-scale node, region, and sample outputs. Use
`fixed_coordinate_global_passthrough_refined_v1` as the primary binary V1
selected-family diagnostic candidate; it remains validation-only, not a
production default.

`tree_break_selection/plot/multiscale_umap.py` joins
`multiscale_gene_assignments.csv` to existing UMAP coordinates and renders a
stable-region-first overlay with pass-through or guard zones marked separately.

`spectral/cosine_band_coherence_comparator.py` ports the old c2ef fixed
cosine-band and biological-coherence checks into the current diagnostic stack.
It keeps the historical predeclared bands and runs current gate/decomposition
code on each band tree; the outputs are comparator evidence, not production
calibration.
