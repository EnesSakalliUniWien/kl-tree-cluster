# Hierarchical Clustering with Statistical Decomposition and Random Projections

## 1. Introduction

This manuscript defines a method for clustering high-dimensional categorical data (including binary). The approach combines standard hierarchical clustering with a statistically grounded top-down decomposition. Clusters are separated only if there is statistical evidence of distributional differences, leveraging random projections to maintain power in specific high-dimensional settings.

## 2. Notation

- Let $X$ be the data matrix with $N$ samples and $M$ features.
- Each feature $j \in \{1, \dots, M\}$ can take values in a categorical set of size $K_j$.
- For binary data, $K_j=2$ (values $\{0,1\}$).
- For general categorical data, we represent the state of a node $u$ as a probability distribution matrix $\hat{\mu}_u$ or a concatenation of probability vectors vs all categories.
- Let $\mathcal{T}$ be a hierarchical tree where leaves correspond to rows of $X$.
- For any node $u \in \mathcal{T}$, let $L(u)$ be the set of leaf indices in the subtree rooted at $u$.
- Let $n_u = |L(u)|$ be the number of samples in node $u$.
- Let $\hat{\mu}_u$ be the centroid of node $u$, defined as the component-wise average of the leaf distributions.

## 3. Algorithm

### 3.1. Hierarchy Construction

We construct the initial hierarchy $\mathcal{T}$ using Agglomerative Clustering on dataset $X$ with the following metric and linkage:

- **Metric**: Hamming Distance. For categorical vectors $x, y$ (where $x_j$ denotes the category of feature $j$):
  $$ d(x, y) = \frac{1}{M} \sum_{j=1}^M \mathbb{I}(x_j \neq y_j) $$
  *(Note: This applies to binary and general categorical encoding).*
- **Linkage**: Average Linkage. The distance between clusters $A, B$ is:
  $$ D(A, B) = \frac{1}{|A||B|} \sum_{a \in A, b \in B} d(a, b) $$

### 3.2. State Propagation

After building the tree, we calculate probability distributions for every node:

1. **Leaf Nodes**: A single sample with category $c$ in feature $j$ is treated as a deterministic distribution. The probability vector $\hat{\mu}_j$ is the **one-hot encoding** of $c$ (a vector of zeros with a single 1 at the index corresponding to $c$).
    - *Example*: If Feature $j$ has domains $\{A, B, C\}$ and the sample is $B$, the vector is $[0, 1, 0]$.

2. **Internal Nodes**: The distribution of an internal node is the sample-weighted average of its children's Distributions.
    - *Example*: Merging a leaf $[0, 1, 0]$ (B) and a leaf $[1, 0, 0]$ (A) results in $[0.5, 0.5, 0]$, representing a 50/50 mixture.

### 3.3. Tree Decomposition

We perform a top-down traversal to identify valid clusters. Every internal node $u$ connects to two children $L$ and $R$. A node $u$ splits if it satisfies these gates:

1. **Child-Parent Divergence Gate**: Check if a potential sub-cluster ($L$ or $R$) differs significantly from parent $u$. If children are identical to the parent, the split separates noise; stop.
2. **Sibling Divergence Gate**: The distributions $\hat{\mu}_L$ and $\hat{\mu}_R$ differ significantly.

#### Child-Parent Divergence Test

1. **Hypothesis**: Test if the child is a random subset (noise) or unique pattern (signal).
   - $H_0: \mu_{child} = \mu_{parent}$
   - $H_1: \mu_{child} \neq \mu_{parent}$

2. **Standardization (Z-Score)**: Calculate the standardized difference for every feature based on the parent's variance ($H_0$ population).
   $$ Z_j = \frac{\hat{\mu}_{child, j} - \hat{\mu}_{parent, j}}{\sqrt{\frac{\hat{\mu}_{parent, j}(1 - \hat{\mu}_{parent, j})}{n_{child}}}} $$
   where $\hat{\mu}$ is feature probability and $n_{child}$ is child sample count.

**Composite Divergence Score**:
   To distinguish deep splits from noise, we augment raw probability difference $Z$ with topological branch length $b_u$ (Hamming distance to parent). The Composite Divergence Score $S(u)$ is:
   $$ S(u) = D_{KL}(\hat{\mu}_{child} || \hat{\mu}_{parent}) + \lambda \frac{\bar{D}_{KL}}{\bar{b}} b_u $$
   Under $H_0$, both terms should be small. This score acts as an initial filter.

**Formula Intuition**: Measures child probability "surprise" relative to parent.

- **Numerator**: The raw probability "shift".
- **Denominator**: The expected "noise" for group size $n_{child}$.
- **Result ($Z_j$)**: A value of $Z=2.0$ means the shift is twice as large as random noise. Squaring and summing yields a total distance representing signal, ignoring random variation.

*Numerical Stability*: Clip Z-scores to $[-100, 100]$ to bounds infinite Z-scores from zero variance.

1. **Random Projection**: Project the Z-vector onto a lower-dimensional subspace ($k \approx \log n_{child}$) using fixed random matrix $R$.
   $$ Y = R \cdot Z $$

**Distance Interpretation**:
We quantify distance as **Standardized Euclidean Distance** in Z-score space.

1. **Euclidean Embedding**: Squared norm $\|Z\|^2 = \sum Z_i^2$ represents distance in standardized space.
2. **KL Equivalence**: This squared distance is asymptotically equivalent to scaled Kullback-Leibler divergence via second-order Taylor expansion: $\sum Z^2 \approx 2N \cdot D_{KL}$.

**Random Projection Rationale**:
Random Projections ($Y = RZ$) preserve Euclidean norms ($\|Y\|^2 \approx \|Z\|^2$) via the Johnson-Lindenstrauss lemma. This compresses the feature space, mitigating the **curse of dimensionality** by concentrating scattered signal into a comprehensive metric ($T$). While linear projection fails on raw probability vectors, it accurately preserves the information-theoretic divergence ($D_{KL}$) when applied to the standardized Z-space.

**Projection Mechanism**:
The linear mapping $Y = R \cdot Z$ transforms the high-dimensional difference vector into a compressed representation.

1. **Definitions**:
   - **Input ($Z$)**: Standardized difference vector of dimension $M$ (features).
   - **Output ($Y$)**: Projected vector of dimension $k$.
   - **Matrix ($R$)**: Orthonormal random matrix of size $k \times M$.

2. **Dimensionality ($k$)**:
   The target dimension is derived from the **Johnson-Lindenstrauss Lemma**, which guarantees distance preservation dependent on sample count $N$ rather than feature count $M$. For error tolerance $\epsilon$:
   $$ k \ge \frac{4 \ln N}{\epsilon^2} \approx O(\log N) $$

3. **Matrix Construction**: We construct the projection matrix $R$ via **Gaussian Orthogonalization**:
   - **Gaussian Random Matrix ($G$)**: A matrix initialized with independent standard normal entries $g_{ij} \sim \mathcal{N}(0, 1)$. It serves as a source of uniformly distributed random directions (isotropy).
   - **Orthonormal Random Matrix ($R$)**: The result of orthogonalizing $G$ via **QR Decomposition**, scaled by $\sqrt{\frac{M}{k}}$. The QR factorization produces perfectly perpendicular unit vectors (Q). Scaling them ensures that the total energy (squared norm) of the projected vector matches the original space, compensating for the dimensionality reduction.
4. **Preservation of Geometry**: Orthonormality and proper scaling ensure that the length of the vector $Z$ is preserved in expectation: $\|Y\|^2 \approx \|Z\|^2$.

*Note on Categorical Data*: For $K > 2$, apply Z-score logic to One-Hot encoding. This treats the problem as a sum of $K$ "One-vs-Rest" binary divergences. Though distinct from Pearson $\chi^2$ (accounting for covariance), this statistic is monotonically related to divergence. It provides a conservative metric compatible with Random Projection linear geometry.

5. **Decision (Statistical Significance)**:
   We evaluate the test statistic $T = \|Y\|^2$ against the **Chi-Squared Distribution** ($\chi^2_k$), which modeIs the "energy" of random noise in $k$ dimensions.
   - **Theoretical Basis**: Under $H_0$ (no difference), each projected component $Y_i$ is approximately $\mathcal{N}(0,1)$. Thus, their sum of squares $T$ follows $\chi^2_k$.
   - **Noise Baseline ($T \approx k$)**: The expected value $\mathbb{E}[T] = k$. A score near $k$ implies the observed divergence $Z$ is typical of random sampling variations.
   - **Signal Detection ($T \gg k$)**: A score significantly exceeding $k$ indicates the difference vector has more "energy" than chance allows ($p < \alpha$), confirming a real structural split. We reject $H_0$.

#### Sibling Divergence Test

If the child-parent test passes, we test if siblings differ ($H_0: \mu_L = \mu_R$ vs $H_1: \mu_L \neq \mu_R$). The procedure uses **pooled variance** with optional **branch-length adjustment** based on Felsenstein's (1985) Phylogenetic Independent Contrasts.

1. **Standardization**: Compute element-wise Z-score vector $Z$.
    - For Binary Features ($M$ values):
      $$ Z_j = \frac{\hat{\mu}_{L,j} - \hat{\mu}_{R,j}}{\sqrt{\hat{\sigma}^2_j}} $$
    - For Categorical: Flatten standardized differences of all category probabilities.
    - **Base Variance** $\hat{\sigma}^2$ uses pooled variance approximation:
      $$ \hat{\sigma}^2_{j,k} = \hat{\mu}_{pool, j, k}(1 - \hat{\mu}_{pool, j, k}) \left(\frac{1}{n_L} + \frac{1}{n_R}\right) $$

    **Branch-Length Adjustment (Felsenstein, 1985)**:
    Raw contrasts between siblings are not identically distributed; their expected variance depends on topological distance. Following the Phylogenetic Independent Contrasts method, we scale the variance by the sum of branch lengths from parent to each sibling:
    $$ \hat{\sigma}^2_{adjusted} = \hat{\sigma}^2_{base} \cdot (b_L + b_R) $$
    Where $b_L, b_R$ are the branch lengths (Hamming distance to parent) for each sibling.

    **Effect**: This normalization makes the test **stricter** for topologically close siblings (short branches require larger raw differences) and **more lenient** for distant siblings (long branches imply expected divergence, so moderate differences suffice).

    **Intuitive Explanation**:
    Pooled Variance compares "equal" groups (siblings) to estimate shared variability.
    - **Logic**: Under identity, pooling data yields a more accurate variance estimate than separating.
    - **Pooled Mean ($\hat{\mu}_{pool}$)**:
       $$ \hat{\mu}_{pool} = \frac{N_L \mu_L + N_R \mu_R}{N_L + N_R} $$
    - **Pooled Variance Formula**: Calculate difference variance using shared mean.
       $$ \text{Variance} = \underbrace{\hat{\mu}_{pool}(1 - \hat{\mu}_{pool})}_{\text{Variance of single sample}} \times \underbrace{\left(\frac{1}{N_L} + \frac{1}{N_R}\right)}_{\text{Adjustment for sample sizes}} $$

    **Difference from Child-Parent Test**:
    - **Child-Parent Test**: Uses **Parent's Variance** (treating Parent as Population).
    - **Sibling Test**: Uses **Pooled Variance** (treating siblings as peers from a common value).

2. **Random Projection**: If dimensionality is large, project $Z$ onto lower-dimensional subspace $\mathbb{R}^k$ using Gaussian random matrix $R_{proj}$, where $k \approx O(\log N)$.
    $$ Y = R_{proj} Z $$

3. **Test Statistic**:
    $$ T = \|Y\|^2 = \sum_{i=1}^k Y_i^2 $$
    Under $H_0$, $T \sim \chi^2_k$. Reject $H_0$ if p-value $< \alpha$.