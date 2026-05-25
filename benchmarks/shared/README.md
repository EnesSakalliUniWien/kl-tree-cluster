# benchmarks/shared/

Reusable benchmark infrastructure. Benchmark runners import from here; one-off
investigation tools belong in `benchmarks/diagnostics/`.

## Main Flow

1. `cases/` defines benchmark case dictionaries.
2. `generators/generate_case_data.py` turns a case dictionary into data,
   labels, and metadata.
3. `runners/method_registry.py` defines available method IDs.
4. `runners/dispatch.py` routes a method ID to the matching runner.
5. `util/method_execution.py` runs one method on one generated case.
6. `pipeline.py` coordinates case generation, method execution, metrics, and
   optional plotting.
7. `result_records/` defines the canonical row shape written to benchmark CSVs.

## Directory Map

| Path | Purpose |
| ---- | ------- |
| `cases/` | Default and specialized case definitions. |
| `generators/` | Synthetic, categorical, Gaussian, phylogenetic, SBM, and real-data loaders. |
| `runners/` | Method implementations and method registry. |
| `result_records/` | Typed benchmark result record and DataFrame conversion. |
| `types/` | Small benchmark dataclasses. |
| `util/` | Case execution, method execution, parameter parsing, PDF helpers, and timing. |
| `plots/` | Benchmark PDF pages, embeddings, runtime plots, and report export. |
| `relationship_analysis.py` | Post-run benchmark factor analysis. |
| `metrics.py` | ARI, NMI, purity, exact-K, and outlier metrics. |

## Contracts

- A case dictionary is not a result row. It is only an input recipe.
- `generate_case_data()` is the boundary that converts a case recipe into
  matrix data plus metadata such as `feature_space` or precomputed distances.
- Method runners return `MethodRunResult`.
- `result_records/` owns the CSV row contract.
- Missing or unsupported method contexts should be recorded explicitly as
  skipped/unsupported statuses, not hidden by default values.
