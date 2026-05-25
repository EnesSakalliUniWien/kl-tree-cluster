# New Contributor Map

This file gives a first-pass route through the repository. It is intentionally
short: follow it before opening deep implementation directories.

## First 30 Minutes

1. Read `README.md` for the method summary and repository path policy.
2. Run the quick smoke path:

   ```bash
   python quick_start.py
   pytest tests/core tests/statistics tests/localization
   ```

3. If you need benchmark behavior, read `benchmarks/README.md`, then run:

   ```bash
   python benchmarks/run_subset.py
   ```

4. If you need current mathematical context, start at `wiki/index.md`, then
   read `wiki/concepts/kl-te-method.md`,
   `wiki/concepts/projected-wald-statistic.md`, and
   `wiki/analyses/oracle-gate-path-diagnostic.md`.

5. If you need manuscript context, start with
   `manuscript/guides/full_method_logic_map.md` before reading the TeX
   sections.

## Where Things Belong

| Need | Go to |
| ---- | ----- |
| Importable method code | `kl_clustering_analysis/` |
| Tree structure and feature-space data contracts | `kl_clustering_analysis/tree/` |
| Decomposition traversal and gate orchestration | `kl_clustering_analysis/hierarchy_analysis/` |
| Statistical kernels, projection, inflation, and FDR | `kl_clustering_analysis/hierarchy_analysis/statistics/` |
| Full and subset benchmark execution | `benchmarks/` |
| Benchmark-only investigation tools | `benchmarks/diagnostics/` |
| User-facing real-data commands | `scripts/analysis/` |
| Canonical tracked input matrices | `data/feature_matrices/` |
| External reference tables | `data/reference/` |
| Curated evidence snapshots from generated outputs | `raw/assets/` |
| Durable project memory and open questions | `wiki/` |
| Paper draft and derivation notes | `manuscript/` |
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

1. `kl_clustering_analysis/tree/feature_space.py`
2. `kl_clustering_analysis/tree/poset_tree.py`
3. `kl_clustering_analysis/hierarchy_analysis/tree_decomposition.py`
4. `kl_clustering_analysis/hierarchy_analysis/decomposition/gates/orchestrator.py`
5. `kl_clustering_analysis/hierarchy_analysis/decomposition/gates/gate_evaluator.py`
6. `kl_clustering_analysis/hierarchy_analysis/statistics/README.md`

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
- Before committing: run `pytest`, `python3 scripts/wiki/lint.py`, and
  `git diff --check`.
