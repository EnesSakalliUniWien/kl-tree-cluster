# KL-Divergence Clustering Analysis

A Python library for KL-divergence based hierarchical clustering with NetworkX integration and PosetTree representations.

## Features

- **Simulation**: Generate synthetic binary feature matrices with controllable clustering structure
- **PosetTree**: Transform SciPy hierarchical clustering results into NetworkX directed graphs
- **SageMath Integration**: Convert tree structures to SageMath Poset objects

## Installation

```bash
# Basic installation
uv sync

# With SageMath support (requires Python >=3.11)
uv sync --extra sage

# Development installation
uv sync --extra dev
```

## Quick Start

```python
from simulation import generate_random_feature_matrix
from tree.poset_tree import PosetTree
from scipy.cluster.hierarchy import linkage

# Generate synthetic data
data, clusters = generate_random_feature_matrix(
    n_rows=100, n_cols=50, entropy_param=0.2, n_clusters=3
)

# Create linkage matrix
Z = linkage(list(data.values()), method='complete')

# Transform to PosetTree
tree = PosetTree.from_linkage(Z, leaf_names=list(data.keys()))

# Optional: Convert to SageMath Poset
poset = tree.to_sagemath_poset()
```

## Project Structure

```
kl-te-cluster/
├── simulation/          # Data generation utilities
│   └── generate_random_feature_matrix.py
├── tree/               # Tree transformation modules
│   └── poset_tree.py  # PosetTree class
├── tests/             # Test suite
└── docs/              # Documentation
```

## Requirements

- Python >=3.11
- NumPy, SciPy, NetworkX, scikit-learn, pandas
- Optional: SageMath (for `to_sagemath_poset()`)

## License

MIT
