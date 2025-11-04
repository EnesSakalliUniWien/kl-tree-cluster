# KL-Divergence Hierarchical Clustering Toolkit

Pipeline utilities for exploring hierarchical cluster structure with KL-divergence based scoring,
statistical significance testing, and tree decomposition helpers.

## Overview

- Analyse binary feature data through hierarchical clustering and information-theoretic scoring.
- Annotate each merge with KL-divergence, chi-square significance, and conditional mutual information.
- Support reproducible validation through permutation tests and multiple-testing control.
- Produce annotated trees, cluster assignments, and tabular diagnostics for downstream analysis.

The toolkit combines distance-based tree construction, KL scoring utilities, and statistical testing helpers.
A dedicated decomposition routine converts significant nodes into stable cluster assignments.

## Implementation Map

- Tree construction, distribution propagation, and KL metrics: `hierarchy_analysis/divergence_metrics.py`.
- Local KL chi-square helpers and shared statistical utilities: `hierarchy_analysis/statistics/shared_utils.py`.
- Sibling independence via conditional mutual information: `hierarchy_analysis/statistics/sibling_independence.py`.
- Decomposition logic that applies the statistical gates: `hierarchy_analysis/cluster_decomposition.py`.

## Mathematical Workflow and Sibling Tests

Starting from a binary matrix $X \in \{0,1\}^{n \times p}$, the pipeline proceeds through four checkpoints:

1. **Pairwise linkage** – compute the Hamming distance between rows to drive clustering:

   $$
   D_{ij} = \sum_{k=1}^{p} \lvert X_{ik} - X_{jk} \rvert .
   $$

2. **Node distributions** – average the descendant leaves $C_u$ to obtain Bernoulli parameters:

   $$
   \theta_{u,k} = \frac{1}{|C_u|} \sum_{i \in C_u} X_{ik} .
   $$

3. **Local KL scoring** – quantify how a child $c$ diverges from its parent $u$:

   $$
   D_{\mathrm{KL}}(\theta_c \| \theta_u) = \sum_{k=1}^{p} \theta_{c,k} \log \frac{\theta_{c,k}}{\theta_{u,k}} + (1-\theta_{c,k}) \log \frac{1-\theta_{c,k}}{1-\theta_{u,k}} .
   $$

   The chi-square gate uses the approximation

   $$
   2\,|C_c|\,D_{\mathrm{KL}}(\theta_c \Vert \theta_u) \sim \chi^{2}_{p}
   $$

   to decide whether the child diverges from its parent.

4. **Sibling independence** – evaluate conditional mutual information for siblings $c_1$ and $c_2$ given their parent $u$:

   $$
   I(c_1; c_2 \mid u) = \sum_{k=1}^{p} \sum_{a,b \in \{0,1\}} \hat{P}_{u,k}(a,b) \log \frac{\hat{P}_{u,k}(a,b)}{\hat{P}_{u,k}(a)\,\hat{P}_{u,k}(b)} ,
   $$

   where $\hat{P}_{u,k}$ denotes the empirical joint distribution conditioned on membership in $C_u$.

The function `annotate_sibling_independence_cmi` first thresholds node distributions using `binary_threshold`. It then obtains binary arrays $(X, Y, Z)$ for sibling 1, sibling 2, and the parent before evaluating

$$
I(X;Y \mid Z) = \frac{|Z=0|}{|Z|} I(X;Y \mid Z=0) + \frac{|Z=1|}{|Z|} I(X;Y \mid Z=1)
$$

via the batched routine `_cmi_binary_vec`.

Under the null hypothesis that siblings are conditionally independent given their parent, shuffling $Y$ within each parent stratum leaves both marginals unchanged while breaking dependence. The module therefore draws permutation replicates

$$
\widehat{I}^{(b)}(X;Y \mid Z), \qquad b = 1,\dots,B .
$$

The observed value $\widehat{I}_{\text{obs}}$ is compared against the permutation distribution to estimate

$$
p = \frac{1 + \sum_{b=1}^{B} \mathbf{1}\{\widehat{I}^{(b)} \ge \widehat{I}_{\text{obs}}\}}{1 + B} .
$$

Because each parent node produces one sibling test, the code collects the $p$-values and applies Benjamini–Hochberg control through `apply_benjamini_hochberg_correction`.
The corrected decision marks a parent as dependent or independent.
Those decisions yield `Sibling_BH_Dependent` and `Sibling_BH_Independent` flags that inform the decomposition pass.
The `ClusterDecomposer` traverses the hierarchy and splits a node only when both the local KL gate and the CMI-based independence gate pass.

### Worked Example

1. **Toy data**

   | sample | $f_1$ | $f_2$ |
   | ------ | ----- | ----- |
   | $A$    | 1     | 0     |
   | $B$    | 1     | 1     |
   | $C$    | 0     | 1     |

   Pairwise distances are

   $$
   D_{AB} = 1,\quad D_{BC} = 1,\quad D_{AC} = 2 .
   $$

   SciPy linkage therefore merges $A$ with $B$ before attaching $C$, producing

   ```text
   root
   ├─ u_AB
   │  ├─ A
   │  └─ B
   └─ C
   ```

2. **Node distributions** – Using `calculate_hierarchy_kl_divergence`, Bernoulli parameters propagate upward:

   $$
   \theta_A = (1, 0), \qquad \theta_B = (1, 1), \qquad \theta_C = (0, 1),
   $$

   $$
   \theta_{u_{AB}} = \frac{1}{2}\big((1,0) + (1,1)\big) = (1, 0.5),
   $$

   $$
   \theta_{\text{root}} = \frac{1}{3}\big(2 \cdot (1,0.5) + (0,1)\big) = \left(\frac{2}{3}, \frac{2}{3}\right).
   $$

3. **KL-based scoring** – For each edge, the module evaluates the KL divergence in nats:

   $$
   D_{\mathrm{KL}}(\theta_A \| \theta_{u_{AB}}) = 0.693,
   $$

   $$
   D_{\mathrm{KL}}(\theta_{u_{AB}} \| \theta_{\text{root}}) = 0.464,
   $$

   $$
   D_{\mathrm{KL}}(\theta_C \| \theta_{\text{root}}) = 1.504.
   $$

   Multiplying by $2\,|C_c|$ yields chi-square statistics that feed the local significance gate.

4. **Sibling independence** – `annotate_sibling_independence_cmi` thresholds each distribution at $0.5$, obtaining binary vectors

   $$
   u_{AB} \mapsto (1, 1), \qquad C \mapsto (0, 1), \qquad \text{root} \mapsto (1, 1).
   $$

   Conditional mutual information evaluates to

   $$
   I(u_{AB}; C \mid \text{root}) = 0.0,
   $$

   every permutation replicate achieves the same value, and the Benjamini–Hochberg step keeps `Sibling_BH_Dependent` set to `False`. The decomposer therefore treats the siblings as independent and recurses on each branch.

## Highlights

- Build a hierarchy from binary data using SciPy linkage and NetworkX-backed `PosetTree`.
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

## Run the Quick Start Pipeline

`quick_start.py` wires together the full analysis pipeline on a synthetic dataset so you can see each
stage in action.

```bash
python quick_start.py
```

What the script does:

1. **Generate data** – creates a binary feature matrix by thresholding Gaussian blobs so you can
   reproduce demo data with known clusters.
2. **Build the hierarchy** – computes pairwise Hamming distances, runs SciPy `linkage`, and wraps the
   result in a `PosetTree`.
3. **Score nodes** – applies `calculate_hierarchy_kl_divergence` to quantify how informative each
   split is for the generated features.
4. **Annotate significance** – runs multiple hypothesis tests to identify statistically significant
   branches:
   - `annotate_nodes_with_statistical_significance_tests`
   - `annotate_child_parent_divergence`
   - `annotate_sibling_independence_cmi`
5. **Decompose clusters** – uses `ClusterDecomposer` to turn significant nodes into cluster
   assignments and prints a concise report.
6. **Validate results** – compares discovered clusters with the synthetic ground truth using
   Adjusted Rand Index (ARI) so you know how well the decomposition performed.

You should see console output describing each step, a summary of discovered clusters, and the final
ARI score (`1.0` is a perfect match; `0.0` indicates random assignment). No files are written by this
demo; it is safe to rerun repeatedly.

## Working With Your Own Data

- Replace the synthetic data block in `quick_start.py` with your dataframe (binary feature matrix).
- Keep sample names as the index so the reporting remains readable.
- If your data is not binary, adapt the preprocessing section to binarize or adjust the distance
  metric in `pdist`.
- Preserve the overall pipeline order so the statistical annotations stay in sync with the
  calculated metrics.

## Project Layout

```text
.
├── quick_start.py                 # End-to-end reference pipeline
├── tree/                          # PosetTree utilities and graph adapters
├── hierarchy_analysis/            # KL divergence, statistical tests, decomposition
├── simulation/                    # Alternative data generators and helpers
├── notebooks/                     # Exploratory notebooks (see clustering_pipe_line.ipynb)
├── tests/                         # Pytest suite (validation helpers, decomposition checks)
└── docs/                          # Additional conceptual documentation and figures
```

## Validation & Testing

- Run the automated tests with `pytest`.
- Inspect `tests/test_cluster_validation.py` for examples of how to assert cluster quality in
  custom scenarios.
- Consider recording ARI or other metrics alongside your experiments to compare runs.

## License

MIT
