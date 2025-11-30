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
- Local KL chi-square helpers and shared statistical utilities: `hierarchy_analysis/statistics/shared_utils.py`.
- Sibling independence via conditional mutual information: `hierarchy_analysis/statistics/conditional_sibling_independence.py`.
- Decomposition logic that applies the statistical gates: `hierarchy_analysis/cluster_decomposition.py`.

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

The `ClusterDecomposer` treats every internal node as a checkpoint—called a gate—that decides whether the tree is
allowed to split at that spot. Each gate represents a statistical question about the parent/child relationship; if the
answer is “yes,” the walk continues into the children, and if the answer is “no,” the branch stays merged and forms a
cluster boundary.

- **Gate 1 – local divergence check**: For each child, the decomposer either reuses or recomputes the
  child-versus-parent KL divergence and converts it into a chi-square p-value. Both children must show a real shift away
  from their parent before the branch is allowed to split.
- **Gate 2 – sibling independence check**: When both children pass Gate 1, the decomposer looks up
  `Sibling_BH_Independent`. This flag comes from the conditional mutual information test and confirms that the two
  children behave independently once the parent is known. If the flag is `False`, the children remain merged.
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
\text{Gate}_2(u) = \mathbf{1}\!\left[\text{CMI}_{u}(c_1, c_2) \text{ is non-significant after BH correction}\right].
$$

The walk follows a depth-first rule:

1. If $u$ is a leaf, record its cluster label and return.
2. If $\text{Gate}_1(u, c_1) \land \text{Gate}_1(u, c_2) \land \text{Gate}_2(u)$ is `True`, recurse on $c_1$ and $c_2$.
3. Otherwise, stop at $u$ and assign all leaves beneath $u$ to the same cluster.

This simple “if-then” recursion ensures that every branch of the tree either terminates at the earliest node that fails
a gate or keeps splitting while both gates continue to approve the children.

The function `annotate_sibling_independence_cmi` chooses a threshold for each node with `binary_threshold`—either the
default 0.5 cutoff or an adaptive rule such as Otsu (maximizes the separation between low and high values) or Li
(balances the information content of the two sides)—so the parent/child probabilities can be expressed as binary arrays
$(X, Y, Z)$ during the independence check.

From a tree perspective, the conditional mutual information (CMI) test supplies a single question at every parent node:
“Do these siblings still communicate once we know the parent?” It answers that question through four high-level steps:

1. **Focus on relevant branches** – Only parents whose children already passed the local divergence gate are examined,
   so the tree walk spends time on splits that looked promising in Gate 1.
2. **Contrast sibling behaviour inside each parent state** – The parent node carries a binary distribution, so every
   feature either falls into the part of the subtree where the parent’s probability rounded to 0 or the part where it
   rounded to 1. The test compares how often the siblings agree within each of those two slices to see whether the
   branches add new information after the parent is fixed.
3. **Build a “no extra information” reference** – For each parent slice, repeatedly permute the order of one sibling’s
   features while holding the parent and the other sibling fixed. This reshuffling is performed separately within the
   parent-equals-0 subset and the parent-equals-1 subset so the parent’s conditioning is respected. The resulting CMI
   values describe how often siblings would appear to agree if any dependence were purely accidental. For example,
   suppose $Z = [0, 0, 1, 1]$, $X = [1, 0, 1, 0]$, and $Y = [1, 0, 0, 1]$. A valid permutation keeps the indices with
   $Z=0$ (positions 1 and 2) together and the indices with $Z=1$ (positions 3 and 4) together. Shuffling only within
   those strata might produce $Y' = [0, 1, 0, 1]$ (swap inside the $Z=0$ block), $Y' = [1, 0, 1, 0]$ (swap inside the
   $Z=1$ block), or the original arrangement. Each distinct shuffled $Y'$ yields a new CMI value that contributes to the
   reference distribution.
4. **Record the decision on the node** – Convert the observed CMI into a p-value by repeatedly shuffling one sibling
   within each parent state and recomputing the CMI. Count how many shuffled outcomes are at least as large as the
   observed value, add one to both that count and the total number of trials to avoid zero-probability artefacts, and
   take their ratio to obtain the p-value. Adjust the collection of sibling p-values across all parents with
   Benjamini–Hochberg control. A corrected flag of `Sibling_BH_Dependent = True` tells the decomposer to keep the parent
   merged; `Sibling_BH_Independent = True` tells it the branch may open when the walk reaches that node.

Before running this test, make sure the tree nodes already contain their feature distributions, the project dependencies
from `pyproject.toml` (NumPy, SciPy, pandas, NetworkX, StatsModels) are available, and each parent has enough examples
where it is 0 and where it is 1 to allow the shuffling step. Run the sibling independence annotator after computing
local KL metrics and before building the final clusters; that way, when the `ClusterDecomposer` walks the tree it sees
both the divergence scores and the independence flags it needs to decide where to split.

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

### Node distributions – Using `calculate_hierarchy_kl_divergence`, Bernoulli parameters propagate upward:

**Leaf node parameters:**

$$
\theta_A = (1, 0), \qquad \theta_B = (1, 1), \qquad \theta_C = (0, 1),
$$

**Internal node parameters:**

$$
\theta_{u_{AB}} = \frac{1}{2}\left((1,0) + (1,1)\right) = (1, 0.5),
$$

**Root node parameters:**

$$
\theta_{\text{root}} = \frac{1}{3}\left(2 \cdot (1,0.5) + (0,1)\right) = \left(\frac{2}{3}, \frac{2}{3}\right).
$$

#### KL-based scoring – For each edge, the module evaluates the KL divergence in nats:

**Child-to-parent divergences:**

$$
\begin{align}
D_{\mathrm{KL}}(\theta_A \| \theta_{u_{AB}}) &= 0.693, \\
D_{\mathrm{KL}}(\theta_{u_{AB}} \| \theta_{\text{root}}) &= 0.464, \\
D_{\mathrm{KL}}(\theta_C \| \theta_{\text{root}}) &= 1.504.
\end{align}
$$

Multiplying by $2\,|C_c|$ yields chi-square statistics that feed the local significance gate.

### Sibling independence

`annotate_sibling_independence_cmi` thresholds each distribution at $0.5$, obtaining binary vectors

$$
u_{AB} \mapsto (1, 1), \qquad C \mapsto (0, 1), \qquad \text{root} \mapsto (1, 1).
$$

Conditional mutual information evaluates to

$$
I(u_{AB}; C \mid \text{root}) = 0.0,
$$

every permutation replicate achieves the same value, and the Benjamini–Hochberg step keeps `Sibling_BH_Dependent` set to
`False`. The decomposer therefore treats the siblings as independent and recurses on each branch.

A concrete numerical example of the KL-divergence hierarchical clustering process using specific data points follows.

## Numerical Example: KL-Divergence Hierarchical Clustering

### Sample Dataset

Consider a binary feature matrix with 5 samples and 3 features:

| Sample | $f_1$ | $f_2$ | $f_3$ |
| ------ | ----- | ----- | ----- |
| A      | 1     | 1     | 0     |
| B      | 1     | 0     | 0     |
| C      | 1     | 0     | 1     |
| D      | 0     | 1     | 1     |
| E      | 0     | 1     | 0     |

### Step 1: Pairwise Hamming Distances

Calculate distances between all sample pairs:

**Distances involving sample A:**

$$
\begin{align}
D_{AB} &= |1-1| + |1-0| + |0-0| = 0 + 1 + 0 = 1 \\
D_{AC} &= |1-1| + |1-0| + |0-1| = 0 + 1 + 1 = 2 \\
D_{AD} &= |1-0| + |1-1| + |0-1| = 1 + 0 + 1 = 2 \\
D_{AE} &= |1-0| + |1-1| + |0-0| = 1 + 0 + 0 = 1
\end{align}
$$

**Distances involving sample B:**

$$
\begin{align}
D_{BC} &= |1-1| + |0-0| + |0-1| = 0 + 0 + 1 = 1 \\
D_{BD} &= |1-0| + |0-1| + |0-1| = 1 + 1 + 1 = 3 \\
D_{BE} &= |1-0| + |0-1| + |0-0| = 1 + 1 + 0 = 2
\end{align}
$$

**Distances involving samples C, D, E:**

$$
\begin{align}
D_{CD} &= |1-0| + |0-1| + |1-1| = 1 + 1 + 0 = 2 \\
D_{CE} &= |1-0| + |0-1| + |1-0| = 1 + 1 + 1 = 3 \\
D_{DE} &= |0-0| + |1-1| + |1-0| = 0 + 0 + 1 = 1
\end{align}
$$

### Step 2: Hierarchical Tree Construction

Using minimum distances for linkage:

1. First merge: A-B (distance = 1), A-E (distance = 1), B-C (distance = 1), D-E (distance = 1)
2. One possible sequence merges A-B first, then D-E, then (A,B)-C, and finally ((A,B),C)-(D,E)

Resulting tree structure:

```text
        root
       /    \
    u_ABC   u_DE
    /   \    /  \
  u_AB   C  D    E
  / \
 A   B
```

### Step 3: Node Distribution Calculation

Calculate Bernoulli parameters for each node:

**Leaf nodes:**

- $\theta_A = (1, 1, 0)$
- $\theta_B = (1, 0, 0)$
- $\theta_C = (1, 0, 1)$
- $\theta_D = (0, 1, 1)$
- $\theta_E = (0, 1, 0)$

**Internal nodes:**

- $\theta_{u_{AB}} = \frac{1}{2}[(1,1,0) + (1,0,0)] = (1.0, 0.5, 0.0)$
- $\theta_{u_{DE}} = \frac{1}{2}[(0,1,1) + (0,1,0)] = (0.0, 1.0, 0.5)$
- $\theta_{u_{ABC}} = \frac{1}{3}[2 \cdot (1.0,0.5,0.0) + (1,0,1)] = (\frac{3}{3}, \frac{1}{3}, \frac{1}{3})$
  $= (1.0, 0.333, 0.333)$
- $\theta_{\text{root}} = \frac{1}{5}[3 \cdot (1.0,0.333,0.333) + 2 \cdot (0.0,1.0,0.5)] = (0.6, 0.6, 0.4)$

### Step 4: KL-Divergence Calculations

For each child-parent pair, calculate KL divergence:

**A vs u_AB:**

$$D_{\mathrm{KL}}(\theta_A \| \theta_{u_{AB}}) = \sum_{k=1}^{3} \left[\theta_{A,k} \log\left(\frac{\theta_{A,k}}{\theta_{u_{AB},k}}\right) + (1-\theta_{A,k}) \log\left(\frac{1-\theta_{A,k}}{1-\theta_{u_{AB},k}}\right)\right]$$

For feature 1: $1 \cdot \log(1/1) + 0 \cdot \log(0/0) = 0 + 0 = 0$

For feature 2: $1 \cdot \log(1/0.5) + 0 \cdot \log(0/0.5) = \log(2) + 0 = 0.693$

For feature 3: $0 \cdot \log(0/0) + 1 \cdot \log(1/1) = 0 + 0 = 0$

$$
D_{\mathrm{KL}}(\theta_A \| \theta_{u_{AB}}) = 0.693 \text{ nats}
$$

**B vs u_AB:** For feature 1: $1 \cdot \log(1/1) + 0 \cdot \log(0/0) = 0$ For feature 2:
$0 \cdot \log(0/0.5) + 1 \cdot \log(1/0.5) = 0 + \log(2) = 0.693$ For feature 3:
$0 \cdot \log(0/0) + 1 \cdot \log(1/1) = 0$

$$
D_{\mathrm{KL}}(\theta_B \| \theta_{u_{AB}}) = 0.693 \text{ nats}
$$

### Step 5: Chi-Square Statistics

Convert KL divergences to chi-square statistics:

For child A: $T_A = 2 \cdot |C_A| \cdot D_{\mathrm{KL}}(\theta_A \| \theta_{u_{AB}}) = 2 \cdot 1 \cdot 0.693 = 1.386$

For child B: $T_B = 2 \cdot |C_B| \cdot D_{\mathrm{KL}}(\theta_B \| \theta_{u_{AB}}) = 2 \cdot 1 \cdot 0.693 = 1.386$

### Step 6: Statistical Significance

The statistical test used in Gate 1 treats the KL divergence as a likelihood-ratio statistic. For a child \(c\) and
parent \(u\), the hypotheses are \(H_0: \theta_c = \theta_u\) versus \(H_1: \theta_c \neq \theta_u\). The likelihood
ratio

$$
\Lambda = \frac{L(\theta_u \mid \text{child data})}{L(\theta_c \mid \text{child data})}
$$

leads, via Wilks’ theorem, to the chi-square approximation

$$
-2\log(\Lambda) = 2|C_c|D_{\mathrm{KL}}(\theta_c \| \theta_u) \xrightarrow{d} \chi^2_p ,
$$

as established by Wilks’ theorem for likelihood-ratio tests under suitable regularity conditions. For Bernoulli data the
approximation holds when:

- each feature exhibits non-degenerate variability (neither all zeros nor all ones),
- the effective sample size \(|C_c|\) is moderately large relative to the number of features \(p\),
- the null hypothesis \(H_0: \theta_c = \theta_u\) provides a reasonable approximation.

The degrees of freedom \(p\) correspond to the \(p\) equality constraints imposed under \(H_0\).

### 4. Detailed Step 6 Analysis for the Example

#### **Child A vs Parent u_AB**

**Child distribution**: $\theta_A = (1, 1, 0)$ **Parent distribution**: $\theta_{u_{AB}} = (1.0, 0.5, 0.0)$

**Feature-by-feature analysis**:

**Feature 1**:

- Child: $\theta_{A,1} = 1$, Parent: $\theta_{u_{AB},1} = 1$
- Contribution: $1 \cdot \log(1/1) + 0 \cdot \log(0/0) = 0$
- **No divergence** - perfect agreement

**Feature 2**:

- Child: $\theta_{A,2} = 1$, Parent: $\theta_{u_{AB},2} = 0.5$
- Contribution: $1 \cdot \log(1/0.5) + 0 \cdot \log(0/0.5) = \log(2) = 0.693$
- **High divergence** - child is always 1, parent is 50/50

**Feature 3**:

- Child: $\theta_{A,3} = 0$, Parent: $\theta_{u_{AB},3} = 0$
- Contribution: $0 \cdot \log(0/0) + 1 \cdot \log(1/1) = 0$
- **No divergence** - perfect agreement

**Total KL divergence**: $D_{\mathrm{KL}}(\theta_A \| \theta_{u_{AB}}) = 0 + 0.693 + 0 = 0.693$ nats

#### **Converting to Chi-Square Statistic**

$$T_A = 2 \cdot |C_A| \cdot D_{\mathrm{KL}}(\theta_A \| \theta_{u_{AB}}) = 2 \cdot 1 \cdot 0.693 = 1.386$$

**Why multiply by 2?** This comes from the likelihood ratio test theory - the factor of 2 ensures the statistic follows
a chi-square distribution.

**Why multiply by |C_A|?** More samples provide more evidence. With only 1 sample under A, we have limited evidence of
divergence.

#### **Statistical Decision**

With $p = 3$ features, we compare against $\chi^2_3$ distribution:

- **Critical value** at $\alpha = 0.05$: $\chi^2_{3,0.05} = 7.815$
- **Our statistic**: $T_A = 1.386$
- **Decision**: $1.386 < 7.815$ → **Not significant**

**P-value calculation**: $$\text{p-value} = \Pr(\chi^2_3 \geq 1.386) = 0.709$$

### 5. **Why KL Divergence is Superior to Alternatives**

#### **A. Compared to Simple Distance Metrics**

**Euclidean distance**: $\|\theta_A - \theta_{u_{AB}}\| = \|(0, 0.5, 0)\| = 0.5$

- **Problem**: No statistical framework, no p-values, no principled cutoff

**Manhattan distance**: $\|\theta_A - \theta_{u_{AB}}\|_1 = 0 + 0.5 + 0 = 0.5$

- **Problem**: Same issues as Euclidean

#### **B. Information-Theoretic Advantages**

1. **Asymmetric**: $D_{\mathrm{KL}}(A \| B) \neq D_{\mathrm{KL}}(B \| A)$
   - This matters! We care about how surprised we'd be seeing the child given the parent, not vice versa

2. **Scale-invariant**: KL divergence naturally handles features with different base rates

3. **Additive**: Features contribute independently to the total divergence

4. **Connects to entropy**: $D_{\mathrm{KL}}(P \| Q) = H(P,Q) - H(P)$ where $H(P,Q)$ is cross-entropy

The KL divergence approach therefore assesses whether the observed data provide sufficient evidence to justify treating
the groups as distinct. For the worked example, the split between A and B is not statistically significant at the 0.05
level.

### Step 7: P-values

$$
\begin{align}
\text{p-value}_A &= \Pr(\chi^2_3 \geq 1.386) \approx 0.709 \\
\text{p-value}_B &= \Pr(\chi^2_3 \geq 1.386) \approx 0.709
\end{align}
$$

Since both p-values > 0.05, **Gate 1 fails** for the u_AB node, meaning A and B should remain merged as a single
cluster.

### Final Cluster Assignment

Based on the statistical gates:

- **Cluster 1:** {A, B} (failed to split)
- **Cluster 2:** {C}
- **Cluster 3:** {D, E} (would need similar analysis)

This numerical example shows how the KL-divergence approach combines information-theoretic scoring with statistical
testing to decide where to cut the hierarchical tree.

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
   - `annotate_root_node_significance`
   - `annotate_child_parent_divergence`
   - `annotate_sibling_independence_cmi`
5. **Decompose clusters** – uses `ClusterDecomposer` to turn significant nodes into cluster assignments and prints a
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

## Project Layout

```text
.
├── quick_start.py                 # End-to-end reference pipeline
├── tree/                          # PosetTree utilities and graph adapters (construct and traverse the hierarchy graph)
├── hierarchy_analysis/            # KL divergence, statistical tests, decomposition
├── simulation/                    # Alternative data generators and helpers
├── notebooks/                     # Exploratory notebooks (see clustering_pipe_line.ipynb)
├── tests/                         # Pytest suite (validation helpers, decomposition checks)
└── docs/                          # Additional conceptual documentation and figures
```

## Validation & Testing

- Run the automated tests with `pytest`.
- Inspect `tests/test_cluster_validation.py` for examples of how to assert cluster quality in custom scenarios.
- Consider recording ARI or other metrics alongside your experiments to compare runs.

## License

MIT
