# KL-Divergence Hierarchical Clustering Toolkit

This repository provides tooling to analyse binary feature datasets by constructing hierarchical cluster trees,
quantifying each merge with Kullback–Leibler divergence, and applying statistical tests that identify the branches
forming stable clusters. It is intended for workflows that require statistically defensible cluster boundaries,
per-merge diagnostics (local KL score, chi-square p-value, sibling independence outcome), and reproducible reports that
support downstream analyses of binary observations.

## Overview

- Analyse binary feature data through hierarchical clustering and information-theoretic scoring.
- Annotate each merge with KL-divergence, chi-square significance, and conditional mutual information.
- Support reproducible validation through permutation tests and multiple-testing control.
- Produce annotated trees, cluster assignments, and tabular diagnostics for downstream analysis.

The toolkit combines distance-based tree construction, KL scoring utilities, and statistical testing helpers. A
dedicated decomposition routine converts significant nodes into stable cluster assignments by deciding where the
hierarchy should stop splitting and which sibling branches stay merged.

## Key Concepts

- **Hierarchical tree** – the analysis revolves around a `PosetTree`, a directed structure that records parent/child
  relationships alongside per-node distributions.
- **Decomposition** – cluster boundaries appear at nodes where the statistical gates say “stop splitting”; descendants
  that fail a gate remain merged and form a cluster together.
- **Information tests** – KL divergence and conditional mutual information supply the scores that determine whether a
  branch is informative enough to justify a split.

### Implementation Map

- Tree construction, distribution propagation, and KL metrics: `hierarchy_analysis/divergence_metrics.py`.
- Local KL chi-square helpers and statistical utilities: `hierarchy_analysis/statistics/kl_tests/utils.py`.
- Sibling independence via conditional mutual information: `hierarchy_analysis/statistics/conditional_sibling_independence.py`.
- Decomposition logic that applies the statistical gates: `hierarchy_analysis/tree_decomposition.py`.

### Pipeline Workflow

Starting from a binary matrix $X \in \{0,1\}^{n \times p}$, the pipeline proceeds through four checkpoints:

1. **Pairwise linkage** – compute the Hamming distance between rows to drive clustering.

   $$D_{ij} = \sum_{k=1}^{p} \lvert X_{ik} - X_{jk} \rvert$$

2. **Node distributions** – average the descendant leaves $C_u$ to obtain Bernoulli parameters.

   $$\theta_{u,k} = \frac{1}{|C_u|} \sum_{i \in C_u} X_{ik}$$

3. **Local KL scoring** – quantify how a child $c$ diverges from its parent $u$.

   $$D_{\mathrm{KL}}(\theta_c \| \theta_u) = \sum_{k=1}^{p} \left[\theta_{c,k} \log \frac{\theta_{c,k}}{\theta_{u,k}} + (1-\theta_{c,k}) \log \frac{1-\theta_{c,k}}{1-\theta_{u,k}}\right]$$

   The chi-square gate uses the approximation

   $$2\,|C_c|\,D_{\mathrm{KL}}(\theta_c \| \theta_u) \sim \chi^{2}_{p}$$

   to decide whether the child diverges from its parent. Intuitively, the KL term asks, “How surprised would we be to
   see the child’s feature rates if the parent’s pattern were still true?” Scaling by the child’s sample count turns
   that surprise into a likelihood-ratio statistic, which large-sample theory says behaves like a chi-square random
   variable with one degree of freedom per feature. A chi-square goodness-of-fit check then compares the parent’s
   expected counts with the child’s observed counts and flags cases where the mismatch is too large to blame on random
   noise.

4. **Sibling independence** – evaluate how often the siblings co-occur relative to what you would expect from the parent
   alone. For each feature $k$, count the proportions of $(c_1, c_2)$ taking every combination of
   $\{0,1\} \times \{0,1\}$ among the samples where the parent equals 0 or 1. The conditional mutual information
   aggregates those log-ratio contributions to show whether the siblings carry extra information about each other once
   the parent is known.

## Statistical Gates and Independence Checks

The `TreeDecomposition` treats every internal node as a checkpoint—called a gate—that decides whether the tree is
allowed to split at that spot. Each gate represents a statistical question about the parent/child relationship; if the
answer is “yes,” the walk continues into the children, and if the answer is “no,” the branch stays merged and forms a
cluster boundary.

- **Gate 1 – local divergence check**: For each child, the decomposer either reuses or recomputes the
  child-versus-parent KL divergence and converts it into a chi-square p-value. Both children must show a real shift away
  from their parent before the branch is allowed to split.
- **Gate 2 – sibling independence check**: When both children pass Gate 1, the decomposer looks up
  `Sibling_BH_Different`. This flag comes from the sibling divergence test (Jensen–Shannon divergence with a
  chi-square approximation plus Benjamini–Hochberg correction). If the flag is `False`, the children remain merged.
- **Optional parent gate**: Setting `parent_gate="strict"` adds one more requirement—only parents already marked
  significant can split. Leave it `off` to ignore this extra guard.

If any gate fails, the algorithm labels the parent node as the cluster boundary and stops there. When every active gate
passes, it continues the walk into each child so the process can repeat deeper in the tree.

Mathematically, let the hierarchy be a directed tree $T = (V, E)$ with root $r$ and binary children for each internal
node. For an internal node $u$ with children $c_1$ and $c_2$, the walk evaluates

$$
\text{Gate}_1(u, c_i) = \mathbf{1}\!\left[2\,|C_{c_i}|\,D_{\mathrm{KL}}(\theta_{c_i}\,\|\,\theta_u) \gt \chi^2_{p, \alpha}\right],
$$

and

$$
\text{Gate}_2(u) = \mathbf{1}\!\left[p_{\mathrm{JSD}}(c_1, c_2) \le \alpha \text{ after BH correction}\right].
$$

The walk follows a depth-first rule:

1. If $u$ is a leaf, record its cluster label and return.
2. If $\text{Gate}_1(u, c_1) \land \text{Gate}_1(u, c_2) \land \text{Gate}_2(u)$ is `True`, recurse on $c_1$ and $c_2$.
3. Otherwise, stop at $u$ and assign all leaves beneath $u$ to the same cluster.

This simple “if-then” recursion ensures that every branch of the tree either terminates at the earliest node that fails
a gate or keeps splitting while both gates continue to approve the children.

The function `annotate_sibling_divergence` computes the Jensen–Shannon divergence between sibling distributions,
converts it into a chi-square p-value, and applies Benjamini–Hochberg correction across all tested parents. Nodes with
`Sibling_BH_Different = True` are eligible to split; nodes with `Sibling_BH_Different = False` are merged at the parent.

### Worked Example

See [worked_example.md](worked_example.md) for a detailed step-by-step numerical example of the KL-divergence hierarchical clustering process.

## Highlights

- Build a hierarchy from binary data using SciPy linkage and NetworkX-backed `PosetTree` (a directed tree structure that
  records parent/child links plus per-node metadata such as distributions).
- Quantify KL-divergence at every internal node to detect informative feature splits.
- Run multiple statistical tests to flag significant child-parent and sibling relationships.
- Decompose the resulting tree into cluster assignments you can validate against ground truth.

## Getting Started

### Prerequisites

- Python `>=3.11`
- A virtual environment tool such as `uv` or `venv`
- Optional: SageMath if you want the Sage-specific tooling (`uv sync --extra sage`)

### Install Dependencies

Using `uv` (recommended):

```bash
uv sync
```

Using `pip` inside a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Install From GitHub

Install directly from a GitHub repo (no PyPI needed):

```bash
pip install git+https://github.com/<org>/<repo>.git
```

Pin a branch, tag, or commit:

```bash
pip install git+https://github.com/<org>/<repo>.git@main
pip install git+https://github.com/<org>/<repo>.git@v0.1.0
pip install git+https://github.com/<org>/<repo>.git@<commit_sha>
```

Example `requirements.txt` entry that installs everything (dev + benchmark extras):

```text
kl-clustering-analysis[dev,benchmark] @ git+https://github.com/<org>/<repo>.git
```

## Run the Quick Start Pipeline

`quick_start.py` wires together the full analysis pipeline on a synthetic dataset to illustrate each stage.

```bash
python quick_start.py
```

What the script does:

1. **Generate data** – creates a binary feature matrix by thresholding Gaussian blobs so you can reproduce demo data
   with known clusters.
2. **Build the hierarchy** – computes pairwise Hamming distances, runs SciPy `linkage`, and wraps the result in a
   `PosetTree` so each node keeps track of its distribution, significance markers, and children.
3. **Score nodes** – applies `calculate_hierarchy_kl_divergence` to quantify how informative each split is for the
   generated features.
4. **Annotate significance** – runs multiple hypothesis tests to identify statistically significant branches:
   - `annotate_child_parent_divergence`
   - `annotate_sibling_divergence`
5. **Decompose clusters** – uses `TreeDecomposition` to turn significant nodes into cluster assignments and prints a
   concise report.
6. **Validate results** – compares discovered clusters with the synthetic ground truth using Adjusted Rand Index (ARI)
   so you know how well the decomposition performed.

The script prints console output summarizing each step, reports the discovered clusters, and ends with the ARI score
(`1.0` denotes a perfect match; `0.0` indicates random assignment). The demo does not create files, so reruns can be
performed without cleanup.

## Working With Your Own Data

- Replace the synthetic data block in `quick_start.py` with your dataframe (binary feature matrix).
- Keep sample names as the index so the reporting remains readable.
- If your data is not binary, adapt the preprocessing section to binarize or adjust the distance metric in `pdist`.
- Preserve the overall pipeline order so the statistical annotations stay in sync with the calculated metrics.

## Validation & Testing

- Run the automated tests with `pytest`.
- Inspect `tests/test_cluster_validation.py` for examples of how to assert cluster quality in custom scenarios.
- Consider recording ARI or other metrics alongside your experiments to compare runs.

## Benchmark Methods (Optional)

The benchmarking suite can run additional clustering baselines side-by-side with the KL pipeline:

- Graph community detection: Leiden, Louvain
- Density-based clustering: DBSCAN, OPTICS, HDBSCAN (optional)

Optional dependencies (skipped automatically if missing):

```bash
pip install leidenalg igraph python-louvain hdbscan
```

## License

MIT
