# MNIST Application

MNIST benchmark generation remains under `benchmarks/experiments/mnist/`.
Application-facing interactive and static report renderers live here and read
the retained benchmark outputs. The generators can benchmark thresholded binary
features with configurable distance and linkage, while the retained alpha-sweep
reports reconstruct continuous PCA50 Euclidean, average-linkage trees. The
renderer follows the saved result's geometry; MNIST does not have one universal
tree-construction route.

```bash
python applications/mnist/plot_interactive.py
python applications/mnist/plot_report.py
```
