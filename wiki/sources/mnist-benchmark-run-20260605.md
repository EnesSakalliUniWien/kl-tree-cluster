---
title: MNIST Benchmark Run 2026-06-05
type: source
status: reviewed
updated: 2026-06-05
sources:
  - benchmarks/results/experiments/mnist/mnist_benchmark_summary.csv
tags:
  - source
  - benchmarks
  - mnist
  - validation
---

# MNIST Benchmark Run 2026-06-05

## Summary

The MNIST benchmark example completed on 2026-06-05 using the canonical
`benchmarks.experiments.mnist.run` entrypoint. It sampled `2000` MNIST images,
binarized pixels at threshold `0.0`, and evaluated `20` distance/linkage
configurations.

## Key Points

- Only the four single-linkage configurations completed decomposition. They
  heavily over-split the `10` digit classes into `1133` to `1155` predicted
  clusters.
- Best ARI was `0.069192` for `jaccard + single` and `dice + single`; both had
  NMI `0.519758` and `1133` predicted clusters.
- `rogerstanimoto + single` and `hamming + single` produced ARI `0.055932`,
  NMI `0.512789`, and `1155` predicted clusters.
- The other `16` configurations failed closed under the sibling empirical-null
  support contract: each reported no strict-null or stopped-edge calibration
  records with positive weight and `1999` selected non-null positive-weight
  records.

## Evidence

- `benchmarks/results/experiments/mnist/mnist_benchmark_summary.csv` contains
  the 20-row result table from the completed run.
- `benchmarks/experiments/mnist/run.py` defines the run grid and output path.

## Links

- [[full-benchmark-run-20260605]]
- [[selected-hierarchy-null-support-contract]]
- [[benchmark-pipeline-contract]]
