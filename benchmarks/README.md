# Tree-Break Selection Clustering Benchmarks

This directory contains the benchmark infrastructure for Tree-Break Selection clustering.

## Quick Start

```bash
# Run the canonical default benchmark
uv run python -m benchmarks.full.run

# Run a contract-specific case suite
TBS_CASE_SUITE=binary uv run python -m benchmarks.full.run
TBS_CASE_SUITE=continuous uv run python -m benchmarks.full.run
TBS_CASE_SUITE=categorical uv run python -m benchmarks.full.run

# Run specific benchmark suites
uv run python -m benchmarks.experiments.branch_length.run
uv run python -m benchmarks.experiments.branch_length_3d.run
uv run python -m benchmarks.experiments.multi_split.run

# Quick subset for fast iteration (~15 cases)
uv run python -m benchmarks.smoke.run_subset

# Fast regression gate
uv run python -m benchmarks.regression.run_gate

# Real-world datasets (MNIST, Penguins, Digits)
uv run python -m benchmarks.experiments.mnist.run
uv run python -m benchmarks.experiments.umap_datasets.run

# Analyze benchmark-factor relationships for the latest run
uv run python -m benchmarks.diagnostics.analysis.analyze_relationships
```

## Architecture: Shared System Plus Diagnostics

Benchmark runners should import reusable infrastructure from
`benchmarks.shared.*`. Investigation-only tools live under
`benchmarks.diagnostics.*`.

### Authoritative Defaults

The authoritative default benchmark definition lives in code:

- Cases: `benchmarks.shared.cases.get_default_test_cases()`
- Contract suites: `benchmarks.shared.cases.get_test_cases_by_suite()`
- Default methods: `benchmarks.shared.util.method_sets.DEFAULT_METHODS`

Both `benchmarks/full/run.py` and
`benchmarks/shared/pipeline.py::benchmark_cluster_algorithm()` should reflect
those same defaults. The README documents those code-backed defaults and is not
an independent source of truth.

### Benchmark Classification

Benchmark code is classified by role, not by age or filename:

| Class | Location | Role |
| ----- | -------- | ---- |
| `canonical` | `benchmarks/full/`, `benchmarks/smoke/`, `benchmarks/regression/`, `benchmarks/shared/` | Current benchmark contract, shared runners, default method set, and fast gates. |
| `experiment` | `benchmarks/experiments/` | Standalone scientific studies with explicit sweeps or real datasets. |
| `validation` | `benchmarks/validation/` | Focused calibration or method-constant checks with manifests. |
| `diagnostic` | `benchmarks/diagnostics/` | Investigation-only panels, failure analysis, and post-run diagnosis. |
| `cloud` | `benchmarks/cloud/` | AWS wrappers for large validation or diagnostic jobs. |
| `result` | `benchmarks/results/` | Generated outputs; never a source of benchmark defaults. |

Method result rows carry `run_id`, `benchmark_class`, `benchmark_grid`, and
`benchmark_repeat` columns. Resume logic uses `run_id`, so grid cells and
repeats do not collapse into a single method-level completion flag.

### Grid Contract

Method variants should be represented as parameter grids, not duplicated method
IDs. Use `benchmarks.shared.benchmark_grid.benchmark_grid()` for Cartesian
sweeps and repeats. Each grid cell gets one stable method-qualified `run_id`;
duplicate `run_id` values fail during method selection.

Use separate method IDs only for genuinely different runner families, for
example `tbs_diffusion_graphtools` versus `tbs_diffusion_graphtools_nnls`.
Use grid axes for settings such as linkage, tree builder, alpha, nearest-neighbor
K, and repeat index.

### Shared Features

The `benchmarks/shared/` system provides:
- `matrix_audit` parameter for TensorBoard-style matrix exports
- `evolution.py` for temporal evolution tracking
- `audit_utils.py` for matrix export infrastructure
- `linkage_matrix` captured in TBS runner's `extra` dict

The `benchmarks/diagnostics/` system provides failure diagnosis, oracle
recoverability, gate-path tracing, sibling-calibration diagnostics, and
standalone benchmark-analysis tools in purpose-named subdirectories. Standalone
experiments that are not part of the canonical full-suite contract live under
`benchmarks/experiments/`.

## Key Files

| File                       | Purpose                                                         |
| -------------------------- | --------------------------------------------------------------- |
| `shared/cases/__init__.py` | All test case definitions (Gaussian, binary, SBM, phylogenetic) |
| `shared/cases/geometry.py` | Canonical case-recipe shape and true-K helpers                  |
| `shared/pipeline.py`       | `benchmark_cluster_algorithm()` shared execution pipeline       |
| `full/run.py`              | Canonical suite/report orchestrator                             |
| `shared/util/case_inputs.py` | Matrix contract validation and shared distance preparation    |
| `shared/util/method_execution.py` | One method+parameter run and result-row construction     |
| `shared/runners/tbs_runner.py` | TBS-specific runner with SBM `distance_condensed` handling    |
| `shared/metrics.py`        | ARI, NMI, Purity calculations                                   |
| `shared/generators/`       | Data generators (phylogenetic, Gaussian, etc.)                  |
| `smoke/run_subset.py`      | Small fast smoke runner                                         |
| `regression/run_gate.py`   | Historically sensitive benchmark regression gate                |
| `experiments/`             | Standalone branch-length, MNIST, UMAP, and multi-split studies  |
| `diagnostics/`             | Oracle, calibration, spectral, failure, and post-run diagnostics |
| `validation/`              | Method-constant validation manifests and checks                 |

## Running Benchmarks

### Critical: Run Module Commands From Project Root

All benchmark commands **must** be run from the project root through the locked
`uv` environment:

```bash
# Correct
uv run python -m benchmarks.full.run

# Wrong
cd benchmarks/full && python run.py
```

### Output Structure

Results saved to timestamped directories under each benchmark's `results/` folder:
```
benchmarks/results/run_YYYYMMDD_HHMMSSZ_<case_suite>/
├── <case_suite>_benchmark_comparison.csv    # Main results
├── benchmark_performance_grid.md            # Ranked grid-cell report
├── benchmark_performance_grid_summary.csv   # One row per run_id/grid cell
├── benchmark_performance_grid_*.csv         # Case-by-run ARI/NMI/Purity/status grids
├── failure_report.md              # Failed cases analysis
├── benchmark_relationship_report.md   # Factor/method relationship summary
├── benchmark_relationship_*.csv       # Method/section summaries + modeled effects
├── benchmark_relationship_plots.pdf   # Relationship plots (if enabled)
├── audit/                         # Matrix exports (if enabled)
└── plots/
    └── case_N.pdf                 # Per-case visualizations
```

## Benchmark Types

### 1. Full Suite ([full/](full/))

**Purpose**: Canonical end-to-end benchmark — runs the complete default test
case suite and compares the default benchmark methods.

**Data generation**: By default the runner uses the `full` suite from
`benchmarks.shared.cases.get_test_cases_by_suite()`, which currently resolves to
121 cases. Each case specifies a generator, sample count, feature count,
cluster count, and noise level. The dispatcher (`generate_case_data`) routes to
the appropriate generator, binarizes, one-hot-encodes, or keeps continuous
coordinates under an explicit `FeatureSpace`, and feeds the resulting matrix to
each clustering method. Generated metadata records both `source_family` and
`feature_representation` so reports can distinguish, for example,
Gaussian-source median-binary cases from Gaussian-source continuous cases. The
benchmark CSV preserves both fields in every result row.

**Experiment setup**:

- Default methods: `benchmarks.shared.util.method_sets.DEFAULT_METHODS`, currently
  `tbs`, `tbs_diffusion`, `tbs_diffusion_adaptive_nnls`, `leiden`, `louvain`,
  `kmeans`, `spectral`, `dbscan`, `optics`, and `hdbscan`.
- Additional registered methods such as `tbs_complete` and `tbs_single` are
  available through `TBS_METHODS`, but they are not part of the canonical
  default benchmark unless explicitly requested.
- Optional GPL diffusion methods such as `tbs_diffusion_graphtools`,
  `tbs_diffusion_graphtools_nnls`, and
  `tbs_diffusion_graphtools_adaptive_nnls` require the `experimental-gpl` extra:
  `uv sync --extra experimental-gpl`.
  `tbs_diffusion_graphtools_adaptive_nnls` exposes linkage and
  neighbor-joining tree strategies as one parameter grid, not as duplicate
  method IDs.
- Each case runs in an isolated subprocess (optional), with configurable timeout (default 1800 s) and retry count (default 4).
- Per-case PDF plots (tree, UMAP embedding, manifold comparison) are generated and merged into `<case_suite>_benchmark_report.pdf`.
- Set `TBS_CASE_SUITE` to run one mathematical input-contract suite:
  `binary`, `categorical`, `continuous`, `discretized_gaussian`, `graph`, or
  `full`.

**Evaluation**: ARI, NMI, Purity, Exact-K match. Results saved to
`<case_suite>_benchmark_comparison.csv` with a `failure_report.md` for cases
that error or time out.
Relationship analysis is also generated by default: method/section summaries, effect tables, a markdown report, and a compact PDF that explains how metrics move with noise, sample size, feature count, and true cluster count.
For a focused diagnosis of the current TBS gap on categorical and overlapping families, see `benchmarks/diagnostics/analysis/categorical_overlapping_gap_diagnosis.md`.

---

### 2. Branch Length ([experiments/branch_length/](experiments/branch_length/))

**Purpose**: Measures how clustering performance degrades as evolutionary divergence increases between two groups.

**Data generation**: Two-group Jukes–Cantor substitution model. A root sequence of length `n_features` (default 200) over `n_categories` states (default 4) is generated uniformly. Group 1 keeps the root; Group 2 evolves along a branch of variable length. Within-group variation is added via short terminal branches (default 0.05). Total samples: `2 × n_leaves/2` (default 200).

**Experiment setup**:

- Sweeps over 16 branch lengths from 0.01 (nearly identical) to 2.0 (saturated divergence).
- 5 replicates per branch length (different random seeds).
- TBS method only (`hamming` + `average`).

**Evaluation**: ARI and NMI plotted as a function of branch length, with mean ± std error bands across replicates. Produces a performance curve showing the method's sensitivity range.

---

### 3. Branch Length 3D ([experiments/branch_length_3d/](experiments/branch_length_3d/))

**Purpose**: Extends the branch-length benchmark to a 2D parameter sweep — varies both evolutionary divergence and the number of features simultaneously to create a performance surface.

**Data generation**: Same Jukes–Cantor model as Branch Length, but with a grid of `(branch_length, n_features)` combinations.

**Experiment setup**:

- Branch lengths: 9 values from 1 to 30.
- Feature counts: 6 values from 50 to 500.
- 3 replicates per grid cell.
- Fixed `n_leaves=200`, `n_categories=4`.

**Evaluation**: ARI surface plotted as a 3D heatmap (branch length × features × ARI). Shows the joint effect of signal strength and dimensionality on clustering accuracy.

---

### 4. Multi-Split ([experiments/multi_split/](experiments/multi_split/))

**Purpose**: Tests the method's ability to recover the correct number of clusters (K) in a balanced star phylogeny.

**Data generation**: Star-topology tree where all K groups diverge from a common ancestor. A root sequence (`n_features=200`, `n_categories=4`) is evolved along `between_group_branch=0.3` to create K group ancestors, then each ancestor generates `n_total/K` samples via short terminal branches (`within_group_branch=0.05`).

**Experiment setup**:

- Sweeps K over {2, 4, 6, 7, 8, 10, 12} groups.
- Fixed `n_total_samples=200` (divided equally among groups).
- 10 replicates per K.
- TBS method only.

**Evaluation**: K-recovery accuracy (found K vs. true K), ARI, NMI. Plotted as a function of true K. Tests whether the statistical gates correctly identify all split points in a balanced tree.

---

### 5. Quick Subset ([smoke/run_subset.py](smoke/run_subset.py))

**Purpose**: Fast iteration benchmark — runs ~15 representative cases from the full suite for quick validation during development.

**Data generation**: Draws from the same default case pool used by the full
suite (`get_default_test_cases()`), which currently contains 121 cases.
Hand-picked subset covers Gaussian, Binary, Categorical, SBM, and Overlapping
families.

**Experiment setup**:

- ~15 cases, TBS method only.
- Plots enabled (UMAP comparison pages).
- Uses `benchmark_cluster_algorithm()` from `shared/pipeline.py`.

**Evaluation**: Same metrics as full suite (ARI, NMI, Exact K). Printed as a summary table to stdout.

---

### 6. MNIST ([experiments/mnist/](experiments/mnist/))

**Purpose**: Real-world benchmark on handwritten digit images (10 classes).

**Data generation**: Downloads a random subset of MNIST via `sklearn.datasets.fetch_openml` (default 1000 samples, 784 features). Pixel intensities are normalized to [0, 1], then binarized with threshold 0.1 (dark pixels → 0, light pixels → 1). Optional PCA pre-reduction to 50 components.

**Experiment setup**:

- Distance metric: `rogerstanimoto` (double-weights mismatches, better for sparse binary images).
- Linkage: `average`.
- α = 0.05 for both edge and sibling tests.

**Evaluation**: ARI, NMI, K-found vs. K-true (10 digits). Results saved to CSV and printed as a summary. Tests the method on high-dimensional, real-world data where cluster boundaries are not perfectly separable.

---

### 7. UMAP Datasets ([experiments/umap_datasets/](experiments/umap_datasets/))

**Purpose**: Benchmarks on the standard datasets featured in the UMAP documentation — Palmer Penguins and Sklearn Digits.

**Data generation**:

- **Palmer Penguins**: 333 samples × 4 numeric features (bill length/depth, flipper length, body mass), 3 species. Continuous features are discretized into 5 ordinal bins via `KBinsDiscretizer`, then treated as binary.
- **Sklearn Digits**: 1797 samples × 64 features (8×8 pixel images), 10 digit classes. Pixel values are binarized at threshold.

**Experiment setup**:

- TBS method with `hamming` + `average`.
- Generates interactive Bokeh HTML plots for per-dataset embedding visualization.
- Results saved to timestamped output directory.

**Evaluation**: ARI, NMI, K-found vs. K-true. Per-dataset PDF and interactive HTML plots.

---

### 8. Calibration ([calibration/](calibration/))

**Purpose**: Empirical statistical calibration — verifies that the edge and sibling tests maintain correct Type I error rates under the null hypothesis, and that TreeBH FDR control is valid.

**Data generation**:

- **Null scenarios**: Pure-noise binary matrices with `p_one=0.5` (no cluster structure). Three sizes: 64×32, 128×64, 192×96.
- **TreeBH scenarios**: Tree-structured synthetic p-values with known alt/null partition. Varying alt fractions (0–50%) and signal strengths (beta distribution shape parameter 0.2–0.5).

**Experiment setup**:

- Null calibration: 30 replicates per scenario (configurable via `TBS_CAL_NULL_REPS`). Full pipeline runs (linkage → tree → decompose) on each replicate.
- TreeBH calibration: 200 replicates per scenario (configurable via `TBS_CAL_TREEBH_REPS`).
- Also includes crossfit permutation diagnostics.

**Evaluation**: Per-scenario Type I error rates with 95% binomial CIs. Edge and sibling rejection rates compared to nominal α = 0.05. TreeBH FDR and power summaries. Results written to CSV + `calibration_plots.pdf` + `calibration_report.md`.

---

## Test Case Categories

The full suite currently resolves to 121 cases from the shared case registry.
The major families represented in that registry are summarized below.

### Contract Suites

Use these suites when making method claims. They separate mathematical input
contracts instead of mixing every historical stress case into one score.

| Suite | Cases | Contract | Main interpretation |
| ----- | ----- | -------- | ------------------- |
| `binary` | 42 | Native Bernoulli `{0,1}` matrices from the binary generator. | Primary benchmark for the mature Bernoulli Tree-Break Selection path. |
| `categorical` | 27 | Multinomial/categorical blocks represented by explicit one-hot `FeatureSpace` metadata. | Tests block-covariance categorical support, not independent Bernoulli columns. |
| `continuous` | 9 | Selected raw Gaussian-coordinate examples with explicit continuous `FeatureSpace` and Euclidean tree distances. | Experimental empirical-Gaussian path; report separately from binary and discretized Gaussian results. |
| `discretized_gaussian` | 32 | Gaussian sources transformed to binary or quantile one-hot features. | Discretization stress tests, not evidence for native continuous performance. |
| `graph` | 3 | SBM adjacency features with precomputed modularity distance. | Graph-distance/recoverability stress tests. |
| `full` | 121 | Union of the registry. | Broad smoke/reporting suite; avoid using its aggregate as a single method claim. |

### Gaussian

The Gaussian blob, dimensional Gaussian, and Gaussian-outlier families keep the
historical median-binarized cases. A small set of selected `*_continuous`
representation-forwarding examples keeps the raw coordinates, carries an
explicit continuous `FeatureSpace`, and provides Euclidean tree distances
through benchmark metadata. These examples are not cloned for every historical
Gaussian stress case; they are a focused check of the empirical-Gaussian path.

| Family | Discretized cases | Selected continuous examples |
| ------ | ----------------- | ---------------------------- |
| Gaussian blobs (`blobs`) | `gaussian_extreme_noise`, `improved_gaussian`, `gaussian_null`, `overlapping_gaussian`, `overlapping_gaussian_quantile` | `continuous_gaussian_examples` |
| Dimensional Gaussian | `gaussian_dimensionality_consolidated`, `gaussian_dimensionality_diffuse` | `continuous_dimensional_gaussian_examples` |
| Gaussian outliers | `gaussian_outlier_singleton`, `gaussian_outlier_contamination` | `continuous_gaussian_outlier_examples` |

### Core Binary (28 cases)

Generated directly as binary {0,1} matrices via `generate_random_feature_matrix`. Each cluster owns a distinctive subset of features with controlled bit-flip probabilities. `entropy_param` controls noise (0 = perfect separation, 0.5 = random). This is the most natural input format for the Bernoulli TBS pipeline.

The `binary` contract suite contains these 28 core binary cases plus the 14
native-binary overlapping cases described in the [Overlapping](#overlapping-25-cases)
section.

| Subcategory                  | Cases | n_rows  | n_cols  | K    | entropy   |
| ---------------------------- | ----- | ------- | ------- | ---- | --------- |
| `binary_balanced_low_noise`  | 2     | 72      | 72–120  | 4    | 0.25      |
| `binary_sparse_features`     | 2     | 72–100  | 72–500  | 4    | 0.10      |
| `improved_binary_perfect`    | 3     | 40–160  | 50–200  | 2–8  | 0.00      |
| `improved_binary_low_noise`  | 4     | 40–240  | 50–300  | 2–12 | 0.05–0.10 |
| `improved_binary_moderate`   | 3     | 80–200  | 100–250 | 4–8  | 0.12–0.15 |
| `improved_binary_hard`       | 2     | 100–280 | 200–400 | 4–8  | 0.15–0.20 |
| `improved_binary_unbalanced` | 2     | 100–150 | 150–200 | 4–6  | 0.10–0.12 |
| `improved_binary_edge_cases` | 3     | 50–300  | 60–2000 | 2–15 | 0.10      |

### SBM (3 cases)

Generated via Stochastic Block Model (`generate_sbm`). The adjacency matrix is converted to a **modularity-based distance** (`1 - B_norm` where `B = A - ddᵀ/2m`). This pre-computed distance is passed directly to the TBS runner — there is no fallback to `pdist()` on raw adjacency data.

| Cases | Sizes (nodes)      | p_intra   | p_inter    | K   |
| ----- | ------------------ | --------- | ---------- | --- |
| 3     | [30,30]–[50,40,30] | 0.05–0.12 | 0.005–0.04 | 2–3 |

### Categorical (11 cases)

Generated via `generate_categorical_feature_matrix` as integer category indices (0 to K−1), then **one-hot encoded** into `(n_rows × n_cols × n_categories)` binary indicators before the TBS pipeline. Tests the algorithm's handling of multi-valued features.

| Subcategory                    | Cases | n_rows  | n_cols   | K   | n_categories | entropy   |
| ------------------------------ | ----- | ------- | -------- | --- | ------------ | --------- |
| `categorical_clear`            | 3     | 100–150 | 50–80    | 4–6 | 3–5          | 0.05–0.08 |
| `categorical_moderate`         | 2     | 120–180 | 60–100   | 4–6 | 3–4          | 0.15–0.18 |
| `categorical_high_cardinality` | 2     | 200     | 40–50    | 4   | 10–20        | 0.10–0.12 |
| `categorical_unbalanced`       | 1     | 150     | 60       | 4   | 3            | 0.12      |
| `categorical_overlapping`      | 1     | 400     | 100      | 4   | 3            | 0.35      |
| `categorical_high_dimensional` | 2     | 200–300 | 500–1000 | 4–6 | 3–4          | 0.12–0.15 |

### Phylogenetic (13 cases)

Generated via `generate_phylogenetic_data` — simulates trait evolution along a random phylogenetic tree using a Jukes–Cantor-like substitution model. Each taxon is a cluster; `samples_per_taxon` samples are drawn from the evolved distribution at each leaf. Category-index matrix is **one-hot encoded** before the TBS pipeline.

| Subcategory              | Cases | n_taxa (=K) | n_features | n_categories | samples/taxon | mutation_rate |
| ------------------------ | ----- | ----------- | ---------- | ------------ | ------------- | ------------- |
| `phylogenetic_dna`       | 4     | 4–16        | 100–500    | 4            | 15–25         | 0.2–0.4       |
| `phylogenetic_protein`   | 3     | 4–12        | 50–150     | 20           | 15–30         | 0.3–0.4       |
| `phylogenetic_divergent` | 2     | 4–8         | 100–200    | 4            | 20–25         | 0.7–0.8       |
| `phylogenetic_conserved` | 2     | 4–8         | 100–200    | 4            | 20–25         | 0.05–0.08     |
| `phylogenetic_large`     | 2     | 32–64       | 500–1000   | 4            | 8–10          | 0.3–0.35      |

**Known issue**: Severe over-splitting on many phylogenetic cases due to near-zero branch lengths in the constructed clustering tree (see [Known Issues](#known-issues--structural-problems)).

### Overlapping (29 cases)

Binary subcategories use the `binary` generator with high `entropy_param` (0.22–0.48) to create clusters whose feature profiles significantly overlap. Gaussian subcategory uses `blobs` with high `cluster_std` (3.0–6.0). Tests the algorithm's ability to correctly **merge** overlapping groups rather than over-split.

| Subcategory                     | Cases | n_samples | n_features | K    | Noise param       |
| ------------------------------- | ----- | --------- | ---------- | ---- | ----------------- |
| `overlapping_binary_heavy`      | 5     | 500–600   | 50–1000    | 4–6  | entropy 0.40–0.48 |
| `overlapping_binary_moderate`   | 4     | 400–1000  | 80–800     | 4–10 | entropy 0.28–0.32 |
| `overlapping_binary_partial`    | 4     | 400–1000  | 60–600     | 4–10 | entropy 0.22–0.26 |
| `overlapping_binary_highd`      | 4     | 500–1000  | 1000–5000  | 4–10 | entropy 0.28–0.35 |
| `overlapping_binary_unbalanced` | 4     | 400–1000  | 100–1000   | 4–10 | entropy 0.28–0.35 |
| `overlapping_gaussian`          | 8     | 300–1000  | 30–300     | 3–10 | std 3.0–6.0       |

### Real Data (1 case)

Loads `data/feature_matrices/feature_matrix.tsv` from the repo root — a pre-existing binary GO-term feature matrix. No ground-truth labels; used for qualitative evaluation only.

---

## Evaluation Methods

All benchmarks (except calibration) evaluate clustering quality using:

| Metric                                  | Range   | Interpretation                                               |
| --------------------------------------- | ------- | ------------------------------------------------------------ |
| **ARI** (Adjusted Rand Index)           | [−1, 1] | 1 = perfect, 0 = random, negative = worse than random        |
| **NMI** (Normalized Mutual Information) | [0, 1]  | 1 = perfect correspondence between predicted and true labels |
| **Purity**                              | [0, 1]  | Fraction of samples in the dominant true class per cluster   |
| **Exact K**                             | count   | Number of cases where found K equals true K                  |

Note: K-Means and Spectral Clustering are given the **true K** as input, making them oracle baselines rather than fully unsupervised competitors.

## Clustering Methods

The method registry exposes the canonical methods plus additional diagnostic
TBS variants, while the default full benchmark uses the method subset in
`benchmarks.shared.util.method_sets.DEFAULT_METHODS`.

| Key                 | Name                 | Distance          | Linkage  | Notes                               |
| ------------------- | -------------------- | ----------------- | -------- | ----------------------------------- |
| `tbs`                | TBS Divergence        | hamming           | average  | Default — binary-native             |
| `tbs_complete`       | TBS (Complete)        | hamming           | complete | Complete-linkage variant            |
| `tbs_single`         | TBS (Single)          | hamming           | single   | Single-linkage variant              |
| `tbs_diffusion`      | TBS (Hamming NN Diffusion) | Hamming nearest-neighbor diffusion tree | average | Default diffusion method; branch lengths default to `linkage_ultrametric` |
| `tbs_diffusion_adaptive` | TBS (Adaptive pydiffmap Diffusion) | pydiffmap adaptive diffusion tree | average | Adaptive diffusion baseline with linkage-ultrametric branch lengths |
| `tbs_diffusion_adaptive_nnls` | TBS (Adaptive pydiffmap Diffusion, NNLS Branch-Time) | pydiffmap adaptive diffusion tree | average | Default adaptive diffusion branch-time candidate; fixed-topology NNLS lengths plus normalized branch-length variance |
| `tbs_diffusion_graphtools` | TBS (graphtools Kernel Diffusion) | graphtools kernel diffusion tree | average | Optional GPL backend; requires `uv sync --extra experimental-gpl`; linkage-ultrametric branch lengths |
| `tbs_diffusion_graphtools_nnls` | TBS (graphtools Kernel Diffusion, NNLS Branch-Time) | graphtools kernel diffusion tree | average | Optional GPL backend; fixed-topology NNLS lengths plus normalized branch-length variance |
| `tbs_diffusion_graphtools_adaptive_nnls` | TBS (graphtools Kernel Diffusion, Adaptive-K NNLS Branch-Time) | graphtools kernel diffusion tree with fragmentation-guard K selection | grid | Optional GPL backend; one tree-strategy grid over average, complete, weighted, single, centroid, median, Ward, and MAD-rooted neighbor joining; fixed-topology NNLS lengths plus normalized branch-length variance |
| `leiden`            | Leiden               | KNN graph         | —        | Community detection, resolution=1.0 |
| `louvain`           | Louvain              | KNN graph         | —        | Community detection, resolution=1.0 |
| `kmeans`            | K-Means              | —                 | —        | **Oracle**: uses true K             |
| `spectral`          | Spectral             | nearest_neighbors | —        | **Oracle**: uses true K             |
| `dbscan`            | DBSCAN               | —                 | —        | Density-based, auto eps             |
| `optics`            | OPTICS               | —                 | —        | Density-based, xi=0.05              |
| `hdbscan`           | HDBSCAN              | —                 | —        | Density-based, min_cluster=5        |

NNLS branch-time methods fail closed if fixed-topology branch-length fitting
does not apply fitted lengths to the `PosetTree`. Non-converged solver output is
not allowed to continue with topology-only/linkage branch lengths unless
`branch_length_optimization_apply_nonconverged=True` is set for an explicit
diagnostic run. NNLS benchmark output should retain the solver status,
`applied_to_tree`, design density/nonzero counts, zero design-column counts, and
normalized residual columns so quality comparisons can distinguish topology
effects from failed branch-time fitting.

Benchmark dispatch is fail-fast for method exceptions. A runner may still return
an explicit `status=skip` result for a known unsupported method/case contract,
but unexpected exceptions are not converted into benchmark rows. Regression and
maintained validation/NNLS grids therefore stop at the first execution error.

## Adding New Benchmark Suites

### 1. Create Directory Structure

```bash
mkdir benchmarks/my_suite
mkdir benchmarks/my_suite/results
```

### 2. Create cases.py

```python
MY_CASES = [
    {
        "name": "my_custom_case",
        "generator": "gaussian",
        "n_samples": 200,
        "n_features": 30,
        "n_clusters": 3,
        "cluster_std": 1.0,
        "random_state": 42,
    },
]

ALL_CASES = MY_CASES
```

### 3. Create run.py

```python
from benchmarks.shared.pipeline import benchmark_cluster_algorithm
from benchmarks.my_suite.cases import ALL_CASES

if __name__ == "__main__":
    results_df, fig = benchmark_cluster_algorithm(
        test_cases=ALL_CASES,
        methods=["tbs", "tbs_diffusion"],
    )
```

### SBM-Specific: Pre-computed Distance Required

SBM cases **require** pre-computed modularity distance:
```python
# In tbs_runner.py - distance_condensed is REQUIRED for SBM
if case["generator"] == "sbm":
    if distance_condensed is None:
        raise ValueError("SBM cases require pre-computed distance_condensed")
```

There is **no fallback** to `pdist()` on raw adjacency data.

## Known Issues & Structural Problems

### Critical Issues

1. **Categorical/Phylogenetic Type Mismatch**  
   Categorical and phylogenetic cases now enter the TBS runner through explicit
   one-hot `FeatureSpace` contracts. Remaining failures should be interpreted
   as calibration, projection-dimension, FDR, traversal, metric, or
   recoverability problems, not as a Bernoulli/categorical install or schema
   mismatch.

2. **Gradient Templates Design Flaw**  
   `_create_gradient_templates()` assigns `prob_ones = cluster_id / (n_clusters - 1)`. For k>4, adjacent templates differ by ≈1/k fraction — below noise floor after bit-flip. Prefer `_create_sparse_templates` for k>4.

3. **Median Binarization Power Loss**  
   `(X > median(X, axis=0)).astype(int)` forces θ≈0.5, maximizing Wald variance denominator θ(1-θ)=0.25 and minimizing test power.

4. **Small Sample Size Failures**  
   Gaussian cases with n/K < 20 (e.g., n=30, K=3 → 10/cluster) are below practical power threshold for projected Wald χ² test. Explains K=1 under-splitting (`gauss_clear_medium`, `binary_perfect_2c`).

## Debugging Failed Cases

### Enable Debug Trace

Use the canonical runner outputs for diagnosis:

- `benchmarks/results/run_<timestamp>/full_benchmark_comparison.csv`
- `benchmarks/results/run_<timestamp>/failure_report.md`
- `benchmarks/results/run_<timestamp>/audit/`

### Common Failure Patterns

| Symptom                      | Likely Cause                              | Investigation                              |
| ---------------------------- | ----------------------------------------- | ------------------------------------------ |
| K=1 (under-splitting)        | n/K < 20, or signal below noise floor     | Check `n_samples / n_clusters` ratio       |
| K>>expected (over-splitting) | Categorical data without one-hot encoding | Verify data type going into `decompose()`  |
| All cases K=1                | Branch length handling bug                | Check `mean_branch_length` computation     |
| SBM failures                 | Missing `distance_condensed`              | Verify case data has pre-computed distance |

### Analyze Specific Case

See the maintained diagnostic utilities under `benchmarks/diagnostics/`.
Prefer generated benchmark artifacts over hard-coded README tables when
inspecting current behavior.

## Integration with Main Library

The benchmark system uses explicit method parameters rather than mutable
package globals:
```python
from tree_break_selection.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_SIBLING_ALPHA,
)
from benchmarks.shared.pipeline import benchmark_cluster_algorithm

# Benchmark runs use the canonical edge threshold and an explicit sibling
# threshold argument. Tree geometry and linkage are explicit entries in each
# registered method specification.
df_results, _ = benchmark_cluster_algorithm(
    significance_level=DEFAULT_SIBLING_ALPHA,
)

# Lower-level method experiments pass tree construction, traversal, and alpha
# choices directly. Production benchmark rows record the values used.
```

## Calibration Diagnostics

Calibration diagnostics are maintained as explicit standalone scripts under
`benchmarks/diagnostics/calibration/`, including edge-calibration helpers.
Oracle and gate-path diagnostics live under `benchmarks/diagnostics/oracle/`;
spectral-dimension diagnostics live under `benchmarks/diagnostics/spectral/`;
post-run result analyses live under `benchmarks/diagnostics/analysis/`. The
full benchmark runner only runs the clustering comparison and relationship
analysis.
