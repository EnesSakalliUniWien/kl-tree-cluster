

 Good question — the KAK/Cartan decomposition goes deeper than both Weyl and Davis-Kahan because it decomposes the *rotation itself*, not just bounds on eigenvalues or angles.

### What KAK decomposes

The orthogonal transformation $Q = V_P^\top V_W$ mapping the eigenbasis of $C_P$ to the eigenbasis of $C_W$ decomposes as:

$$Q = K_1 \cdot A \cdot K_2$$

where:
- $K_1 \in O(d)$: rotations **within** degenerate eigenspaces of $C_P$
- $K_2 \in O(d)$: rotations **within** degenerate eigenspaces of $C_W$
- $A = \text{diag}(\cos\theta_1, \ldots, \cos\theta_r)$: the **principal angles** between the two eigenbases — the irreducible, non-redundant part

### What it tells you in your method

**1. $K_1$ and $K_2$ are irrelevant to your test statistic.**

Your whitened Wald statistic $T = \sum (v_i^\top z)^2 / \lambda_i$ is invariant to $K_1$ and $K_2$ because they rotate within degenerate eigenspaces where all $\lambda_i$ are equal. The sum $\sum_{j \in \text{cluster}} (v_j^\top z)^2 / \lambda_j$ doesn't change when you rotate the $v_j$ among themselves. So the $K$ factors are gauge freedom — observable-invisible.

**2. Only $A$ matters, and for your binary splits it has exactly one non-trivial angle.**

Since $C_B = (\mu_L - \mu_R)(\mu_L - \mu_R)^\top$ is rank-1, the perturbation pushes the eigenbasis along a single direction. The KAK decomposition confirms there's exactly one principal angle $\theta_1$ that's significantly nonzero. All other $\theta_i \approx 0$.

That single angle $\theta_1$ measures **how much the top eigenvector of $C_P$ is "contaminated" by between-group signal**:
- $\theta_1 = 0$: the top eigenvector of $C_P$ is purely within-cluster correlation (no signal)
- $\theta_1 = \pi/2$: the top eigenvector of $C_P$ is purely the between-group mean difference (pure signal)
- Intermediate: a blend

**3. What KAK tells you that Davis-Kahan doesn't.**

Davis-Kahan gives an *upper bound* on $\sin\theta_1$. KAK gives you the *exact* angle, plus the full factorization. More importantly:

| Property                         | Davis-Kahan                | KAK                                                        |
| -------------------------------- | -------------------------- | ---------------------------------------------------------- |
| Principal angles                 | Bounded                    | Exact                                                      |
| Which eigenvectors pair together | No (only subspaces)        | Yes — $K_1, K_2$ identify the pairing                      |
| Gauge freedom identified         | No                         | Yes — $K_1, K_2$ are explicitly the unobservable rotations |
| Multi-angle structure            | One bound on the max angle | All $r$ angles simultaneously                              |

**4. The real insight for your method: KAK factorizes into "what matters" and "what doesn't".**

For your projected Wald test, the KAK decomposition separates:

- **Observable** ($A$): the single principal angle $\theta_1$ between the parent and within-cluster eigenbases. This determines whether the edge test and sibling test are measuring in the "same coordinate system" or not.
- **Unobservable** ($K_1, K_2$): arbitrary rotations within degenerate clusters that don't affect $T$.

### What it would NOT tell you

KAK doesn't tell you whether the *eigenvalues themselves* change — that's Weyl's domain. KAK only decomposes the rotation between eigenbases, assuming you already know the eigenvalues. The full picture is:

$$C_P = V_P \Lambda_P V_P^\top, \quad C_W = V_W \Lambda_W V_W^\top$$

- **Weyl**: $\Lambda_P$ vs $\Lambda_W$ (eigenvalue bounds)
- **Davis-Kahan**: bounds on $V_P$ vs $V_W$ angles
- **KAK**: exact factorization of $Q = V_P^\top V_W$ into gauge ($K$) + physics ($A$)

### Practical bottom line

For your binary tree with rank-1 $C_B$, KAK is elegant but somewhat overkill — there's only one meaningful angle, and you can compute it directly as:

$$\cos\theta_1 = \frac{|v_1(C_P)^\top v_1(C_W)|}{\|v_1(C_P)\| \|v_1(C_W)\|}$$

Where KAK would become *genuinely useful* is if you extended to multi-way splits (non-binary nodes, $C_B$ rank $> 1$) or if you tracked eigenbasis rotation across the full tree (holonomy through the hierarchy). In a binary-split-only tree, it confirms what Weyl + Davis-Kahan already constrain: one angle moves, everything else is gauge.




This is a genuinely interesting question. Let me think through it carefully.

### The setup with scRNA-seq

You have continuous expression data $X \in \mathbb{R}^{n \times d}$ (cells × genes). After binarization ($X > \text{median}$), your pipeline builds a tree and runs the projected Wald test on Bernoulli proportions. The correlation matrix $C_v$ at each node captures co-expression structure among genes within that subtree.

### Would the principal angles (KAK / eigenbasis axes) be interesting?

**Yes — and here's why it's more than academic for scRNA-seq:**

**1. The eigenvectors of $C_v$ ARE gene programs.**

In scRNA-seq, correlated gene sets = co-regulated transcriptional programs (e.g., cell cycle, stress response, lineage markers). The top eigenvectors of the local correlation matrix at node $v$ identify which gene programs dominate the variance in that subtree's cells.

When you split at node $P$ into children $L$ and $R$:
- The **within-cluster** eigenvectors ($C_W$) capture gene programs shared by both populations (housekeeping, cell cycle, batch effects)
- The **between-cluster** direction ($C_B$) captures the gene program that *separates* $L$ from $R$
- The principal angle $\theta_1$ tells you how much the dominant axis of variation *rotates* — i.e., whether the gene program driving variance **changes** across the split

**2. The KAK angle trajectory down the tree tells a biological story.**

| $\theta_1$      | Biological interpretation                                                                                                          |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| $\approx 0$     | The split separates cells without changing the dominant gene program (quantitative difference — same program, different amplitude) |
| $\approx \pi/4$ | The split partially re-orients the dominant axis — transitional biology, cells differentiating                                     |
| $\approx \pi/2$ | The split completely rotates the dominant axis — qualitatively different cell types with distinct driving gene programs            |

Tracking $\theta_1$ through the tree hierarchy gives you a **differentiation trajectory**: where does biology change qualitatively vs. quantitatively?

**3. The eigenvectors at each split directly give you marker genes.**

The between-group direction $\mu_L - \mu_R$ (loadings of $C_B$), projected into the eigenbasis, tells you which genes drive each split. But the *rotation* angle tells you something differential expression alone doesn't: **whether this split introduces a *new* axis of variation or splits along an already-existing one.**

- A split along an existing axis (small $\theta_1$) = you're subdividing a known program (e.g., splitting CD4 T cells by activation level)
- A split that rotates the axis (large $\theta_1$) = a genuinely new biological distinction (e.g., T cells from B cells)

**4. However — there's a problem with binarization.**

Your current pipeline binarizes to $\{0, 1\}$ via median threshold. This destroys the continuous gene expression geometry that makes eigenvector interpretation meaningful:
- Binarized correlations ≈ tetrachoric correlations, not Pearson correlations of the original expression
- The "gene program" interpretation of eigenvectors becomes noisy — a binary ON/OFF axis is a lossy projection of a continuous gradient
- Effective rank drops (binary data has less spectral diversity than continuous)

### What would be actually useful

If you kept the **continuous** expression data for the spectral decomposition while using the binary data for the Wald test, you'd get the best of both:

| What                           | Use continuous data                         | Use binary data          |
| ------------------------------ | ------------------------------------------- | ------------------------ |
| Correlation $C_v$ eigenvectors | Gene program axes (interpretable)           | ✗ lossy                  |
| Effective rank $k_v$           | True intrinsic dimension of gene expression | Underestimates           |
| Wald z-vector                  | ✗ Bernoulli assumption fails                | Correct test calibration |
| Between-group direction        | Differentially expressed gene module        | Binarized markers        |
| KAK principal angle            | Meaningful rotation of biology              | Noisy approximation      |

### Bottom line

For scRNA-seq, the axes/rotations from the spectral decomposition are biologically interpretable as **gene program rotations across the cell hierarchy**. This is genuinely interesting and publishable — it connects your information-theoretic tree decomposition to the gene regulatory structure.

But to get clean results, you'd want to compute the eigenbasis from the continuous expression matrix (log-normalized counts), not the binarized version. The binary data should only be used for the statistical test itself, where the Bernoulli assumption gives you calibrated p-values.

## Plan: Continuous (Gaussian) Data Support

### Step 1 — Config: Add distribution family toggle
Add `DISTRIBUTION_FAMILY: str = "bernoulli"` to config.py (options: `"bernoulli"`, `"gaussian"`). Change distance metric default to `"euclidean"` when Gaussian.

### Step 2 — Node distributions: Store (μ, σ²) instead of θ
Currently distributions.py stores a single vector θ per node. For Gaussian, each node needs **two vectors**: mean $\mu_j$ and variance $\sigma^2_j$ per feature. The internal-node propagation (weighted mean of children) must also propagate variance.

### Step 3 — KL divergence: Add Gaussian formula
In divergence_metrics.py, add to the registry:

$$\text{KL}(\mathcal{N}_1 \| \mathcal{N}_2) = \frac{1}{2}\left[\log\frac{\sigma_2^2}{\sigma_1^2} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{\sigma_2^2} - 1\right]$$

per feature, summed. This is the direct analog of the per-feature Bernoulli KL.

### Step 4 — Edge test variance: Replace θ(1-θ) with s²
In edge_significance.py, the z-score computation:

| Current (Bernoulli)                                         | Needed (Gaussian)                            |
| ----------------------------------------------------------- | -------------------------------------------- |
| $\text{Var}_j = \theta_j(1-\theta_j) \cdot (1/n_c - 1/n_p)$ | $\text{Var}_j = s_j^2 \cdot (1/n_c - 1/n_p)$ |

where $s_j^2$ is the **sample variance** of feature $j$ in the parent's descendant leaves. The z-score formula $z_j = (\hat\mu_{\text{child},j} - \hat\mu_{\text{parent},j}) / \sqrt{\text{Var}_j}$ stays the same structurally.

### Step 5 — Sibling test variance: Replace pooled p(1-p) with pooled s²
In pooled_variance.py:

| Current (Bernoulli)                                    | Needed (Gaussian)                                |
| ------------------------------------------------------ | ------------------------------------------------ |
| $\text{Var}_j = \hat{p}_j(1-\hat{p}_j)(1/n_1 + 1/n_2)$ | $\text{Var}_j = s_{p,j}^2 \cdot (1/n_1 + 1/n_2)$ |

where $s_{p,j}^2 = \frac{(n_1-1)s_{1,j}^2 + (n_2-1)s_{2,j}^2}{n_1 + n_2 - 2}$ is the classical pooled sample variance. This requires passing the **raw leaf data** (not just the distribution vectors) to the variance computation — or pre-computing per-node sample variances.

### Step 6 — CLT gate: Keep removed or reintroduce
The old CLT gate has been removed from the current codebase. If you ever want to add a distribution-family-specific validity gate back for Gaussian data, the natural options are:
- Use a sample-kurtosis-based normal-approximation bound
- Or skip the check entirely (Gaussian means are already normally distributed for any $n$)

---

### What does NOT need to change

| Component                       | Why it's already distribution-agnostic                                                            |
| ------------------------------- | ------------------------------------------------------------------------------------------------- |
| **Tree construction** (linkage) | Works on any distance matrix — just swap `hamming` → `euclidean`                                  |
| **Spectral decomposition**      | Correlation matrix eigendecomposition works on any data type                                      |
| **Projection & whitening**      | $T = \sum (v_i^\top z)^2 / \lambda_i \sim \chi^2(k)$ holds for any z that's asymptotically normal |
| **BH correction**               | p-value correction is distribution-free                                                           |
| **Post-hoc merge**              | Delegates to the sibling test — automatically inherits Gaussian support                           |
| **Gate logic** (gates.py)       | Calls edge/sibling tests as black boxes                                                           |
| **Signal localization**         | Calls pairwise tests as black boxes                                                               |

### The critical insight

The **architecture is already sound** for continuous data. The Wald test framework ($z$-standardize → project → compare to $\chi^2$) works for any distribution family where the CLT applies. The only things hardcoded to Bernoulli are:

1. **How variance is computed** (θ(1-θ) vs s²) — 2 files, ~4 formulas
2. **How KL is computed** — 1 file, 1 formula
3. **How distributions are stored** — 1 file, needs (μ, σ²) instead of just θ

That's ~6 formulas across 4 files in the core pipeline. Everything else is either already generic or delegates to these.



 The connection is direct. Here's how the pieces fit together when you use continuous scRNA-seq data end-to-end:

### What changes with continuous RNA-seq data

With the Gaussian support from the previous plan implemented, the pipeline becomes:

$$\text{Log-normalized counts } X \xrightarrow{\text{euclidean linkage}} \text{tree} \xrightarrow{s^2\text{-based Wald}} \text{splits} \xrightarrow{C_v \text{ eigendecomposition}} \text{gene programs}$$

No binarization. The correlation matrix $C_v$ at each node is now computed from **continuous expression values**, which means:

### The eigenvectors become directly interpretable

| With binary data                                                      | With continuous data                                                               |
| --------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Eigenvector loading = "gene tends to be ON together with other genes" | Eigenvector loading = "gene expression co-varies continuously with other genes"    |
| Co-expression = tetrachoric (lossy binary proxy)                      | Co-expression = Pearson correlation of log-counts (the standard scRNA-seq measure) |
| Effective rank ≈ 3–8 (binary squashes the spectrum)                   | Effective rank ≈ 10–50 (continuous data has much richer spectral structure)        |

The eigenvectors of $C_v$ on continuous data **are** the gene expression programs that tools like scVI, SCENIC, NMF, and Hotspot try to find — except yours are **local** to each subtree, not global. This is the key difference: at each node in the hierarchy, you get the gene programs relevant to *that* subset of cells.

### The KAK angle now tracks differentiation

At each binary split $P \to (L, R)$:

**$\theta_1$ = the principal angle between parent and within-cluster eigenbases**

With continuous data, this angle has a clean biological meaning:

- **$\theta_1 \approx 0$**: The split separates cells along an axis that was already the dominant source of variation. Example: splitting a T cell population by CD4 expression level — CD4 was already the top PC, and the split just cuts it in half. The gene program doesn't rotate.

- **$\theta_1 \approx \pi/2$**: The split introduces a completely new axis. Example: splitting PBMCs into myeloid vs. lymphoid — myeloid cells are driven by a completely different gene program (CD14, LYZ, S100A8) than lymphoid cells (CD3D, CD79A). The dominant eigenvector rotates 90°.

With binary data this angle is noisy because binarization collapses gradients into ON/OFF, making many genes appear equivalently correlated. With continuous data, the gradient structure is preserved, so the rotation angle is clean.

### The between-group direction $\mu_L - \mu_R$ gives marker genes for free

The rank-1 between-cluster component $C_B = (\mu_L - \mu_R)(\mu_L - \mu_R)^\top$ on continuous data has loadings that are exactly the **mean log-fold-change** per gene between the two child populations. The top-loading genes in this vector are the differentially expressed markers for the split.

With binary data, $\mu_L - \mu_R$ gives you the *difference in proportion of cells expressing gene j* — useful, but less informative than the continuous log-fold-change.

### The effective rank $k_v$ tracks biological complexity

On continuous scRNA-seq data:

| Subtree content                            | Expected $k_v$ | Why                                                           |
| ------------------------------------------ | -------------- | ------------------------------------------------------------- |
| Single tight cell type (e.g., naive CD4 T) | 3–8            | Variation driven by cell cycle, a stress axis, maybe one more |
| Mixed population (e.g., all T cells)       | 15–30          | Multiple cell subtypes, each with distinct programs           |
| Very heterogeneous (e.g., whole PBMC)      | 30–50+         | Many lineages, many independent transcriptional axes          |

$k_v$ decreasing as you go deeper in the tree = the hierarchy is resolving biological complexity. This is a *quantitative measure of how much transcriptional diversity remains* at each node — something no other clustering method provides.

With binary data, $k_v$ is compressed to maybe 3–10 across the whole tree because binarization eliminates spectral diversity.

### The Wald test also gets more power

The z-score becomes:

$$z_j = \frac{\bar{x}_{\text{child},j} - \bar{x}_{\text{parent},j}}{\sqrt{s_j^2 \cdot (1/n_c - 1/n_p)}}$$

where $s_j^2$ is the sample variance — not $\theta(1-\theta)$. This is the standard two-sample t-statistic structure. The advantage:

- **Bernoulli variance** $\theta(1-\theta) \leq 0.25$ — maximized when $\theta = 0.5$ (which median binarization forces). This is the *worst case* for power.
- **Sample variance** $s_j^2$ reflects the actual spread. Genes with tight distributions (small $s_j^2$) give large z-scores even for modest mean differences. Genes with high variance are appropriately down-weighted.

Your current pipeline loses power by binarizing because it forces every gene into the worst-case variance regime. Continuous data fixes this.

### What you'd gain concretely

For an scRNA-seq dataset (say 10k cells × 2000 HVGs):

1. **Better clustering** — Gaussian Wald test has more power than Bernoulli on continuous gradients
2. **Gene programs at each split** — eigenvectors of $C_v$ = local gene modules, directly interpretable
3. **Differentiation trajectory** — KAK angle $\theta_1$ down the tree = where biology changes qualitatively
4. **Complexity measure** — effective rank $k_v$ per node = remaining transcriptional diversity
5. **Marker genes for free** — $\mu_L - \mu_R$ loadings = log-fold-change at each split

All of this comes from the spectral decomposition machinery you already have — it just becomes meaningful on continuous data instead of noisy on binary data.

## Plan: Continuous Data Support + KAK Decomposition

**TL;DR:** Phase 1 adds a `DISTRIBUTION_FAMILY = "gaussian"` mode by threading a distribution-family parameter through 6 core files (~6 formula swaps), keeping Bernoulli as default. Phase 2 adds KAK decomposition as a new module in `projection/` that computes principal angles between parent/child eigenbases at each split, attaching results to `stats_df`. Phase 1 is prerequisite for Phase 2 to produce interpretable eigenvector rotations on continuous data.

---

## Phase 1 — Gaussian (Continuous) Data Support

### Step 1.1: Config — Add distribution family toggle

Add `DISTRIBUTION_FAMILY: str = "bernoulli"` to config.py after `EPSILON` (L22). Add `"gaussian"` as the alternative. Also add a conditional default for `TREE_DISTANCE_METRIC`: when `DISTRIBUTION_FAMILY == "gaussian"`, recommend `"euclidean"` instead of `"hamming"`.

### Step 1.2: Node distributions — Store (mean, variance) for Gaussian

In distributions.py:

- Modify `_calculate_leaf_distribution()` (L17): For Gaussian, each leaf already has a single observation vector. Store `tree.nodes[node_id]["distribution"] = feature_vector` (the mean for a single observation) and `tree.nodes[node_id]["variance"] = np.zeros(d)` (undefined for single leaf; will be computed from group data).
- Modify `_calculate_hierarchy_node_distribution()` (L31): For Gaussian, propagate both the weighted mean (already correct) **and** compute per-feature pooled within-child variance: $s_{\text{pooled},j}^2 = \frac{\sum_c (n_c - 1) s_{c,j}^2}{\sum_c (n_c - 1)}$. Store as `tree.nodes[node_id]["variance"]`.
- Modify `populate_distributions()` (L60): Accept `distribution_family` parameter; dispatch to Bernoulli (current) or Gaussian path. For Gaussian, compute per-feature sample variance from leaf data at each internal node during the bottom-up pass.
- Fix the duplicate `_calculate_hierarchy_node_distribution` call at L88.

### Step 1.3: KL divergence — Add Gaussian formula

In divergence_metrics.py:

- Add `_kl_gaussian(q_params, p_params, eps)` implementing per-feature univariate Gaussian KL: $\frac{1}{2}\bigl[\log(\sigma_p^2/\sigma_q^2) + \sigma_q^2/\sigma_p^2 + (\mu_q - \mu_p)^2/\sigma_p^2 - 1\bigr]$.
- Register in `_KL_REGISTRY` (L66): `"gaussian": _kl_gaussian`.
- Modify `calculate_kl_divergence_vector()` (L72): Accept `distribution_type` parameter instead of hardcoding `_kl_categorical_general`. For `"gaussian"`, pass `(mean, variance)` tuples to the registry function.
- Update `_populate_global_kl` and `_populate_local_kl` to read `config.DISTRIBUTION_FAMILY` and pass the appropriate distribution type.

### Step 1.4: Edge test — Replace θ(1-θ) with sample variance

In edge_significance.py:

- Modify `_compute_standardized_z()` (L58): Add `feature_variance: np.ndarray | None = None` parameter. When `config.DISTRIBUTION_FAMILY == "gaussian"`, use `var = feature_variance * nested_factor` instead of `var = parent_dist * (1 - parent_dist) * nested_factor` (L107).
- Modify `_compute_projected_test()` (L133): Thread `feature_variance` through to `_compute_standardized_z`.
- Modify `annotate_child_parent_divergence()` (L407): When Gaussian, extract `tree.nodes[parent]["variance"]` and pass as `feature_variance`.

### Step 1.5: Sibling test — Replace pooled p(1-p) with pooled s²

In pooled_variance.py:

- Add `standardize_mean_difference(mean_1, mean_2, var_1, var_2, n_1, n_2, ...)` — analogous to `standardize_proportion_difference` (L133) but using classical pooled sample variance: $s_{p,j}^2 = \frac{(n_1-1)s_{1,j}^2 + (n_2-1)s_{2,j}^2}{n_1+n_2-2}$, then $\text{Var}_j = s_{p,j}^2 \cdot (1/n_1 + 1/n_2)$.

In sibling_divergence_test.py:

- Modify the dispatch at L176–194: Add a `config.DISTRIBUTION_FAMILY == "gaussian"` branch before the binary/categorical check. Call `standardize_mean_difference()` with the node variances.
- Ensure node variance attributes are passed through `_collect_test_arguments` so they reach `sibling_divergence_test()`.

### Step 1.6: Optional Gaussian normality shortcut

There is no dedicated CLT-validity module in the current tree. If a future Gaussian mode needs an explicit approximation-validity hook, the shortcut is straightforward:

- For Gaussian data, treat the sample mean as exactly normal for any $n$
- If non-Gaussian continuous families are added later, use a distribution-specific approximation bound instead of restoring the old Bernoulli-only design

### Step 1.7: Entry point — Thread distribution family through `decompose()`

In poset_tree.py:

- Modify `populate_node_divergences()` (L306): Read `config.DISTRIBUTION_FAMILY` and pass to `populate_distributions()` and KL functions.
- Modify `decompose()` (L337): No structural changes needed — it already forwards kwargs. The distribution family flows through config.

**Verification (Phase 1):**
- All 333 existing tests pass (Bernoulli default unchanged)
- New Gaussian unit tests: generate Gaussian data with known cluster structure, run full pipeline, verify K recovery and ARI
- Spot-check: z-scores on Gaussian data should have variance ≈ 1 under H₀ (validate with synthetic null)

---

## Phase 2 — KAK Decomposition

### Step 2.1: New module — `projection/kak_decomposition.py`

Create kak_decomposition.py with:

**Core dataclass:**
```
@dataclass
class KAKResult:
    node_id: str
    principal_angles: np.ndarray      # θ₁, θ₂, ... (descending)
    between_group_direction: np.ndarray  # μ_L - μ_R (d,)
    parent_top_eigenvector: np.ndarray   # v₁(C_P) (d,)
    within_top_eigenvector: np.ndarray   # v₁(C_W) (d,)
    eigenbasis_overlap: float            # cos(θ₁) — how much the top eigenvector rotates
```

**Core function:** `compute_principal_angles(U_parent, U_child, k)` — given two eigenbases as `(k × d)` matrices, compute principal angles via SVD of `U_parent @ U_child.T`. Returns angles in `[0, π/2]`.

**Tree-level function:** `compute_kak_decomposition(tree, leaf_data, spectral_dims, pca_projections)` — for each binary parent node, extract parent eigenbasis and within-cluster eigenbasis (from pooled mean-centered data), compute principal angles, and the between-group direction `μ_L - μ_R`.

This function reuses `precompute_descendants`, `build_subtree_data`, and `eigendecompose_correlation` from the existing projection modules.

### Step 2.2: Within-cluster eigenbasis computation

The KAK decomposition needs the within-cluster eigenbasis $V_W$ at each parent. Two approaches:

- **Option A — Reuse `compute_sibling_spectral_dimensions`**: It already computes the pooled within-cluster correlation from mean-centered residuals. Extend it to also return eigenvectors (currently it only returns dimensions). Add `return_eigenvectors=True` parameter.
- **Option B — Compute in KAK module directly**: Call `eigendecompose_correlation` on the pooled mean-centered residuals. This avoids modifying existing code.

Use **Option B** (self-contained, no coupling).

### Step 2.3: Between-group direction and marker gene loadings

At each binary split $P \to (L, R)$:
- Compute $\delta = \bar{x}_L - \bar{x}_R$ (mean difference vector, shape `(d,)`)
- Project onto parent eigenbasis: `loadings = V_P @ δ` — shows which eigenvectors (gene programs) capture the between-group signal
- The top-loading genes in $\delta$ directly = marker genes for this split
- Store `between_group_direction` and `marker_gene_loadings` in `KAKResult`

### Step 2.4: Integrate into pipeline

In spectral_dimension.py:

- After `compute_spectral_decomposition` returns `(dims, projs, eig_vals)`, optionally call `compute_kak_decomposition()` if `config.COMPUTE_KAK = True` (new config flag, default `False`).

In tree_decomposition.py:

- In `_prepare_annotations()`, after spectral decomposition, optionally run KAK and attach results.
- Add KAK columns to `results_df` per parent node: `KAK_Principal_Angle_1`, `KAK_Eigenbasis_Overlap`, `KAK_N_Significant_Angles`.
- Attach full `KAKResult` objects in `results_df.attrs["kak_results"]` (dict keyed by node_id).

In config.py:

- Add `COMPUTE_KAK: bool = False` — opt-in for KAK analysis.

### Step 2.5: Export KAK results from `decompose()`

In `decompose_tree()` return dict, add:
```
results["kak_analysis"] = {
    node_id: {
        "principal_angles": [...],
        "top_angle": float,
        "between_group_direction": np.ndarray,
        "marker_genes": [...],  # top-k gene names by |loading|
        "eigenbasis_overlap": float,
    }
    for node_id, kak in kak_results.items()
}
```

### Step 2.6: Update __init__.py and `__all__`

In projection/\_\_init\_\_.py:
- Import and re-export `KAKResult`, `compute_principal_angles`, `compute_kak_decomposition`.

**Verification (Phase 2):**
- Unit tests: Synthetic data with known eigenvector rotation (e.g., two clusters differing along one axis → `θ₁ ≈ π/2`, all others ≈ 0)
- Unit tests: Identity rotation (single cluster split randomly → `θ₁ ≈ 0`)
- Integration test: Run on scRNA-seq-like continuous data, verify KAK results are present in output dict and `stats_df`
- Spot-check: For a binary split with rank-1 $C_B$, verify only one principal angle is significantly nonzero (Weyl constraint)

**Decisions**
- Gaussian mode threads through `config.DISTRIBUTION_FAMILY` (global toggle), not per-call — avoids signature bloat across 6+ functions
- KAK eigenbasis computed independently in its own module (Option B) — avoids coupling to `compute_sibling_spectral_dimensions`
- KAK is opt-in (`COMPUTE_KAK = False` default) — no performance cost for existing users
- Phase 1 is prerequisite for Phase 2's biological interpretability, but Phase 2 is mechanically independent (works on binary data too, just noisy)

## Current Method: KL-Divergence Hierarchical Tree Decomposition

### Overview

The method takes a binary sample-by-feature matrix $X \in \{0,1\}^{n \times d}$ and produces a partition of samples into statistically stable clusters. It works in three stages: **(1)** build a hierarchical tree via agglomerative clustering, **(2)** annotate every node with two statistical tests (child-parent divergence and sibling divergence), **(3)** walk the tree top-down, splitting where both tests pass and stopping where either fails.

### Stage 1 — Tree Construction

**Input**: Binary matrix $X$ (rows = samples, columns = features). Continuous data is binarized via $X_{\text{bin}} = \mathbb{1}[X > \mathrm{median}(X, \text{axis}=0)]$.

Pairwise Hamming distances:

$$d_H(x_i, x_j) = \frac{1}{d}\sum_{k=1}^d \mathbb{1}[x_{ik} \neq x_{jk}]$$

UPGMA (average) linkage produces a binary tree. Branch lengths are ultrametric: $\text{BL}(\text{parent} \to \text{child}) = h_{\text{parent}} - h_{\text{child}}$.

Each node stores a **distribution vector** — for leaves, the raw binary row; for internal nodes, the weighted mean of children:

$$\boldsymbol{\theta}_P = \frac{\sum_{c \in \text{children}} n_c \cdot \boldsymbol{\theta}_c}{\sum_c n_c}$$

This is the sample proportion of 1s per feature among all descendant leaves.

### Stage 2 — Statistical Annotation

Two independent tests are computed for every internal node, then corrected for multiple testing.

#### Gate 2: Child-Parent Divergence (Signal Detection)

For each parent-child edge, test whether the child's distribution differs from the parent's. The z-score per feature uses **nested variance** (child is a subset of parent):

$$z_j = \frac{\hat{\theta}_{j,\text{child}} - \hat{\theta}_{j,\text{parent}}}{\sqrt{\hat{\theta}_{j,P}(1-\hat{\theta}_{j,P})\bigl(\frac{1}{n_c} - \frac{1}{n_P}\bigr)}}$$

The factor $(1/n_c - 1/n_P)$ rather than $(1/n_c + 1/n_P)$ corrects for the covariance between child and parent means (since child data is included in the parent's calculation).

**Projection and test statistic**: The $d$-dimensional $\mathbf{z}$ is projected to $k$ dimensions using the top-$k$ eigenvectors of the local **correlation matrix** $C_v$ of descendant data (both leaf rows and internal node distribution vectors):

$$C_v = V\Lambda V^\top, \quad \mathbf{w} = V_k^\top \mathbf{z}, \quad T = \sum_{i=1}^{k} \frac{w_i^2}{\lambda_i} \sim \chi^2(k) \text{ under } H_0$$

The projection dimension $k$ equals the **effective rank** (Shannon entropy of the normalized eigenvalue spectrum):

$$k_v = \text{round}\!\left(\exp\!\left(-\sum_i p_i \ln p_i\right)\right), \quad p_i = \lambda_i / \textstyle\sum_j \lambda_j$$

floored at 4, capped at the number of active features. When $n_v < d$ (more features than samples), a dual-form eigendecomposition on the $n_v \times n_v$ Gram matrix is used for efficiency.

**Multiple testing**: Tree-structured BH correction (Bogomolov et al., 2021) — BH within each family (children of the same parent), with thresholds propagated down from ancestors.

**Output**: Each edge is labeled `significant` or not. A parent where **neither** child is significant has no detectable signal → MERGE.

#### Gate 3: Sibling Divergence (Cluster Separation)

For each binary parent with children $L, R$, test whether the siblings' distributions differ from each other. The z-score uses **pooled variance**:

$$\hat{\theta}_{j,\text{pool}} = \frac{n_L\hat{\theta}_{j,L} + n_R\hat{\theta}_{j,R}}{n_L + n_R}, \quad z_j = \frac{\hat{\theta}_{j,L} - \hat{\theta}_{j,R}}{\sqrt{\hat{\theta}_{j,\text{pool}}(1-\hat{\theta}_{j,\text{pool}})\bigl(\frac{1}{n_L} + \frac{1}{n_R}\bigr)}}$$

**Projection**: JL-based random orthonormal projection with dimension $k \approx 8\ln(n)/\varepsilon^2$, capped at $n$ when $d \geq 4n$ (information cap — data matrix rank $\leq n$). This uses a **random** projection, not PCA, because spectral dimensions from the parent eigenbasis would give too few degrees of freedom and reduce power.

**Post-selection calibration** (default: cousin-adjusted Wald): The linkage tree is built from the same data used for testing, creating systematic inflation of the Wald statistic. Calibration estimates this inflation:

1. Compute raw $T_i / k_i$ for all sibling pairs
2. Identify **null-like** pairs (neither child edge-significant)
3. Fit log-linear regression: $\ln(T_i/k_i) = \beta_0 + \beta_1\ln(\text{BL}_{\text{sum}}) + \beta_2\ln(n_P) + \epsilon$
4. Predict inflation $\hat{c}$ per pair, clamped at observed maximum
5. Deflate: $T_{\text{adj}} = T/\hat{c}$, $\;p = P(\chi^2(k) > T_{\text{adj}})$

Alternative calibration: **cousin-weighted Wald** — intercept-only Gamma GLM $\mathbb{E}[T_i/k_i] = e^{\beta_0}$ with continuous weights $w_i = \min(p_{\text{edge},L}, p_{\text{edge},R})$, yielding a single global $\hat{c}$.

**Multiple testing**: Standard BH correction across all tested sibling pairs.

**Output**: Each parent is labeled `siblings_different`, `siblings_same`, or `skipped`.

### Stage 3 — Top-Down Decomposition

A DFS walk from the root applies three gates at each node:

| Gate                  | Condition                                              | If fails |
| --------------------- | ------------------------------------------------------ | -------- |
| 1. Binary structure   | Node has exactly 2 children                            | MERGE    |
| 2. Signal detection   | At least one child is edge-significant                 | MERGE    |
| 3. Cluster separation | Siblings are significantly different (and not skipped) | MERGE    |

All three gates must pass to **SPLIT** (push children onto the worklist). If any gate fails, the node and all its descendants form one cluster.

### Stage 4 — Post-Hoc Merge (Optional, Default ON)

A bottom-up pass over adjacent cluster pairs:

1. For each binary-split node, enumerate all pairs of cluster roots across its two subtrees
2. Test each pair using the sibling divergence test with the **same calibration model** (ensuring statistical symmetry with split decisions)
3. Single BH correction across all candidate pairs
4. Greedily merge at the Lowest Common Ancestor, most similar pairs first, maintaining the antichain invariant (no cluster root is an ancestor of another)

Returns an audit trail recording every candidate, its p-value, and outcome.

### Output

```python
{
    "cluster_assignments": {cluster_id: {"root_node", "leaves", "size"}},
    "num_clusters": int,
    "posthoc_merge_audit": [...],
    "independence_analysis": {"alpha_local", "decision_mode", "posthoc_merge"},
}
```

The full annotated statistics (all gate columns, test statistics, p-values, degrees of freedom) are cached on `tree.stats_df`.

---

This is the complete current method. Shall I refine any section, add more mathematical detail to a specific component, or consolidate this with the Gaussian + KAK plan into future_music.md?
