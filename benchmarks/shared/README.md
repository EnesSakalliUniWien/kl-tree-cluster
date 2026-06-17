# benchmarks/shared/

Reusable benchmark infrastructure. Benchmark runners import from here; one-off
investigation tools belong in `benchmarks/diagnostics/`.

## Main Flow

1. `cases/` defines benchmark case dictionaries.
2. `generators/generate_case_data.py` is the stable dispatch entry point that
   turns a case dictionary into data, labels, and metadata. Family-specific
   implementations live in `generators/*_cases.py`.
3. `util/case_inputs.py` validates the generated matrix contract and returns a
   typed `PreparedCaseInputs` object with shared distance objects only when a
   selected method actually needs them.
4. `runners/method_registry.py` defines available method IDs.
5. `runners/dispatch.py` routes a method ID to the matching runner.
6. `util/method_execution.py` runs one method on one generated case and records
   the tree-distance source used by KL-family methods.
7. `util/case_run.py` loops over selected methods for a single case.
8. `pipeline.py` coordinates case iteration, metrics, and optional plotting.
9. `result_records/` defines the canonical row shape written to benchmark CSVs.

`benchmarks/full/run.py` is outside this shared package. It is the report
orchestrator: it chooses a suite, resumes/appends CSV output, decides plotting
policy, calls the shared pipeline one case at a time, and assembles PDFs and
post-run diagnostics.

## Directory Map

| Path | Purpose |
| ---- | ------- |
| `cases/` | Default and specialized case definitions. |
| `cases/geometry.py` | Canonical recipe-level `n`, `p`, and true-K helpers used by runners and report pages. |
| `generators/` | Stable dispatch, shared case-data contracts, and family-specific synthetic/real-data loaders. |
| `runners/` | Method implementations and method registry. |
| `result_records/` | Typed benchmark result record and DataFrame conversion. |
| `types/` | Small benchmark dataclasses. |
| `util/` | Case execution, method execution, parameter parsing, PDF helpers, and timing. |
| `plots/` | Benchmark PDF pages, embeddings, runtime plots, and report export. |
| `relationship_analysis.py` | Post-run benchmark factor analysis. |
| `metrics.py` | External, fragmentation, internal-geometry, and outlier clustering metrics. |

## Contracts

- A case dictionary is not a result row. It is only an input recipe.
- `generate_case_data()` is the boundary that converts a case recipe into
  matrix data plus metadata such as `feature_space` or precomputed distances.
- Case recipe geometry belongs in `cases/geometry.py`; report pages and runner
  large-case decisions must not duplicate generator-specific shape rules.
- Generated metadata distinguishes `source_family` from
  `feature_representation`. For example, Gaussian blob cases can return either
  `median_binary` features or `continuous` features while sharing the same
  `source_family`.
- A precomputed KL tree distance is valid only when generated metadata sets
  `requires_precomputed_kl_distance=True` and provides
  `precomputed_distance_condensed`. Otherwise KL-family methods compute their
  tree distance from the feature matrix and their own run parameters.
- `get_test_cases_by_suite()` separates mathematical input contracts:
  `binary`, `categorical`, `continuous`, `discretized_gaussian`, `graph`, and
  `full`.
- The `continuous` suite contains selected representation-forwarding examples,
  not one continuous clone for every historical Gaussian stress case.
- Method runners return `MethodRunResult`.
- `result_records/` owns the CSV row contract and preserves `source_family`
  plus `feature_representation` in every result row. KL-family rows also expose
  fixed stage timing columns for tree build, node-divergence population, edge gate,
  edge gate substeps, spectral whitening/eigensolve/projection, sibling gate, sibling gate
  substeps, and traversal.
- Missing or unsupported method contexts should be recorded explicitly as
  skipped/unsupported statuses, not hidden by default values.
