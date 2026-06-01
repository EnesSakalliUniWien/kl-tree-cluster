# Benchmark Experiments

This directory contains standalone experiments that are not part of the
canonical full benchmark contract.

Use `uv run python -m benchmarks.full.run` for comparable method-wide
benchmark claims. Use these experiment runners when asking a narrower
scientific question.

| Path | Question |
| ---- | -------- |
| `branch_length/` | How does phylogenetic divergence affect recovery? |
| `branch_length_3d/` | How do divergence and feature count interact? |
| `multi_split/` | How well does KL-TE recover K in a balanced star phylogeny? |
| `mnist/` | How does the method behave on MNIST image-derived inputs? |
| `umap_datasets/` | How does it behave on small public datasets used in UMAP examples? |

Experiment outputs are written under `benchmarks/results/experiments/`.
