# KL-TE Clustering Benchmarks

This directory contains the various benchmark suites for evaluating the KL-TE clustering algorithm. Each benchmark type is organized into its own folder containing the execution script and its results.

## Structure

Each benchmark folder follows this pattern:
- `run.py`: The executable script for the benchmark.
- `results/`: Directory where CSVs, plots, and other outputs are saved.

## Benchmark Types

### 1. Branch Length ([branch_length/](branch_length/))
Measures clustering performance (ARI/NMI) as a function of evolutionary divergence between two groups (Fixed leaves).

### 2. Branch Length 3D ([branch_length_3d/](branch_length_3d/))
Varies both branch length and the number of features to create a performance surface.

### 3. Multi-Split ([multi_split/](multi_split/))
Tests the ability to recover the correct number of clusters (k) in a star phylogeny.

### 4. Temporal ([temporal/](temporal/))
Evaluates incremental clustering as sequences evolve along a growing branch over time.

### 5. UMAP Datasets ([umap_datasets/](umap_datasets/))
Benchmarks the algorithm on standard datasets (Palmer Penguins, Sklearn Digits) used in UMAP documentation.

### 6. MNIST ([mnist/](mnist/))
Evaluates performance on a subset of the MNIST handwritten digits dataset.

### 7. Full Suite ([full/](full/))
Runs the complete default test case suite from the `kl_clustering_analysis` library.

## Running Benchmarks

To run a benchmark, execute the `run.py` script from the project root:

```bash
python benchmarks/branch_length/run.py
```

Results will be automatically saved to `benchmarks/<type>/results/`.
