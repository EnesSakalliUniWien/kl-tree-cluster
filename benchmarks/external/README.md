# External Benchmark Sources

This folder holds optional adapters for benchmark sources that are useful for
scientific comparison but should not become mandatory runtime dependencies.

Install them with:

```bash
uv sync --extra benchmark-infra
```

## clustbench / clustering-benchmarks

`clustering-benchmarks` is the best fit for standardized clustering benchmark
batteries. The adapter preserves clustbench's multiple-reference-partition
contract and requires callers to choose `label_index` explicitly.

```python
from benchmarks.external.clustbench_adapter import load_clustbench_dataset

case = load_clustbench_dataset("wut", "x2", label_index=0)
data = case.data
labels = case.labels
metadata = case.as_case_metadata()
```

Use this for scientific clustering benchmark comparisons where the data are
already designed as clustering tasks.

## OpenML

OpenML benchmark suites are useful for reproducible external datasets and
shared task collections. The local adapter treats supervised classification
targets as reference partitions for clustering evaluation; no train/test split
is used because clustering is unsupervised.

```python
from benchmarks.external.openml_adapter import (
    list_openml_suite_task_ids,
    load_openml_classification_task,
)

task_ids = list_openml_suite_task_ids("OpenML-CC18")
case = load_openml_classification_task(task_ids[0])
```

Use this for broad reproducibility checks, not as a replacement for
clustering-specific benchmark batteries.

## ASV

ASV tracks runtime and memory regressions across commits. The project-level
configuration is `asv.conf.json`, and benchmark functions live under
`asv_benchmarks/`.

```bash
uv run --extra benchmark-infra asv check
uv run --extra benchmark-infra asv run --quick --show-stderr
```
