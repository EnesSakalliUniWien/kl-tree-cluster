---
title: AWS Selected-Tail Equation Study
type: tool
status: draft
updated: 2026-06-03
sources:
  - benchmarks/cloud/aws_selected_tail_equation_study.py
  - benchmarks/cloud/aws/README.md
  - benchmarks/cloud/aws/Dockerfile
  - benchmarks/cloud/aws/batch-stack.yml
tags:
  - aws
  - benchmark
  - calibration
---

# AWS Selected-Tail Equation Study

## Summary

Use the AWS Batch runner when the selected-hierarchy selected-tail equation
study needs more independent regenerations than are practical on the local
machine. The runner shards the existing row-level geometry diagnostic and then
recomputes the combined candidate-equation and tail-law tables from namespaced
simulation ids.

This tool is research infrastructure only. It does not define an external
calibration fallback and it does not change production inference.

## Usage

Build the container:

```bash
docker build -f benchmarks/cloud/aws/Dockerfile -t tree-break-selection-selected-tail:latest .
```

Run one local shard for smoke testing:

```bash
python -m benchmarks.cloud.aws_selected_tail_equation_study run-shard \
  --output-dir raw/assets/benchmark-results/selected_tail_equation_cloud_run \
  --shard-count 20 \
  --replicates-per-shard 50 \
  --shard-index 0
```

In AWS Batch, submit an array job whose size equals `--shard-count`. The array
job command omits `--shard-index` because AWS Batch provides
`AWS_BATCH_JOB_ARRAY_INDEX`.

After all shards finish, run:

```bash
python -m benchmarks.cloud.aws_selected_tail_equation_study merge \
  --output-dir raw/assets/benchmark-results/selected_tail_equation_cloud_run \
  --shard-count 20 \
  --replicates-per-shard 50
```

The merged output is valid only if all shard directories are present. The merge
step prefixes independent simulation ids with the shard id before recomputing
tail-law support, preventing duplicate replicate labels from inflating
precision.

## Evidence

- `benchmarks/cloud/aws_selected_tail_equation_study.py` owns shard resolution,
  row-level record loading, namespaced simulation ids, combined summary
  recomputation, and optional S3 sync.
- `benchmarks/cloud/aws/README.md` documents the local and AWS Batch commands.
- `benchmarks/cloud/aws/Dockerfile` defines the reproducible container entry
  point.
- `benchmarks/cloud/aws/batch-stack.yml` defines the Fargate Batch queue,
  job definition, roles, log group, and S3 output permissions.

## Links

- [[selected-hierarchy-geometric-law-map]]
- [[selected-hierarchy-null-support-contract]]
- [[open-mathematical-questions]]
