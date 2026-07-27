# New Contributor Map

This file gives a first-pass route through the repository. It is intentionally
short: follow it before opening deep implementation directories.

## First 30 Minutes

1. Read `README.md` for the method summary and repository path policy.
2. Install the locked lean development environment:

   ```bash
   uv venv --python 3.11 .venv
   uv sync --extra dev --extra benchmark --extra viz --locked
   ```

3. Run the quick smoke path:

   ```bash
   uv run python quick_start.py
   uv run pytest tests/core tests/statistics tests/localization
   ```

4. If you need benchmark behavior, read `benchmarks/README.md`, then run:

   ```bash
   uv run python -m benchmarks.smoke.run_subset
   ```

   Before running the full `uv run pytest` suite, switch to the full test
   environment:

   ```bash
   uv sync --extra all --extra experimental-gpl --locked
   ```

5. If you need current mathematical context, start at `wiki/index.md`, then
   read `wiki/concepts/tree-break-selection.md`,
   `wiki/concepts/projected-wald-statistic.md`, and
   `wiki/analyses/oracle-gate-path-diagnostic.md`.

6. If you need manuscript context, start with
   `manuscript/guides/full_method_logic_map.md` before reading the TeX
   sections.

## Where Things Belong

| Need | Go to |
| ---- | ----- |
| Importable method code | `tree_break_selection/` |
| Tree structure and feature-space data contracts | `tree_break_selection/tree/` |
| Decomposition traversal and gate orchestration | `tree_break_selection/hierarchy_analysis/` |
| Statistical kernels, projection, inflation, and FDR | `tree_break_selection/hierarchy_analysis/statistics/` |
| Invariant/equivariant and adaptive-cosine separation | `tree_break_selection/space_separation/` |
| Reusable plotting engines | `tree_break_selection/plot/` |
| Full, smoke, and regression benchmark execution | `benchmarks/full/`, `benchmarks/smoke/`, `benchmarks/regression/` |
| Standalone benchmark experiments | `benchmarks/experiments/` |
| Benchmark-only investigation tools | `benchmarks/diagnostics/` |
| Dataset applications and entry points | `applications/` |
| Endotype/GO applications | `applications/endotypes/` |
| Single-cell applications | `applications/scrna/` |
| MNIST report applications | `applications/mnist/` |
| Canonical tracked input matrices | `data/feature_matrices/` |
| External reference tables | `data/reference/` |
| Curated evidence snapshots from generated outputs | `raw/assets/` |
| Durable project memory and open questions | `wiki/` |
| Paper draft and derivation notes | `manuscript/` |
| Manuscript-specific figure generators | `manuscript/tools/figures/` |
| Local-only/generated work | `local_data/`, root `analysis/`, `benchmarks/results/`, `reports/` |

## Do Not Start Here

- `benchmarks/results/`: generated outputs and historical run products.
- `raw/assets/benchmark-results/`: small promoted evidence snapshots cited by
  wiki or analysis notes.
- `manuscript/build/`: build output.
- root `analysis/`: ignored local analysis workspace if present.
- hidden tool directories such as `.venv/`, `.pytest_cache/`, `.ruff_cache/`,
  `.kiro/`, `.playwright-mcp/`, and `.github/skills/`.

## Main Code Route

For the active method, read in this order:

1. `tree_break_selection/tree/feature_space.py`
2. `tree_break_selection/tree/poset_tree.py`
3. `tree_break_selection/hierarchy_analysis/tree_decomposition.py`
4. `tree_break_selection/hierarchy_analysis/decomposition/gates/orchestrator.py`
5. `tree_break_selection/hierarchy_analysis/decomposition/gates/gate_evaluator.py`
6. `tree_break_selection/hierarchy_analysis/statistics/README.md`
7. `tree_break_selection/space_separation/README.md`
8. `tree_break_selection/plot/README.md`

## Main Benchmark Route

For benchmark behavior, read in this order:

1. `benchmarks/README.md`
2. `benchmarks/shared/README.md`
3. `benchmarks/shared/cases/__init__.py`
4. `benchmarks/shared/generators/generate_case_data.py`
5. `benchmarks/shared/runners/method_registry.py`
6. `benchmarks/full/run.py`

## Testing Route

- Small implementation change: run the nearest test file, then one adjacent
  directory such as `tests/statistics/` or `tests/pipeline/`.
- Gate/traversal change: run `tests/core/`, `tests/statistics/`, and
  `tests/localization/`.
- Benchmark/report change: run `tests/pipeline/` and `tests/integration/`.
- Before committing: sync the full test environment, then run `uv run pytest`,
  `make wiki-lint`, and `git diff --check`.
