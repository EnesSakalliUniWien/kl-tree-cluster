# Worked Example

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

### Sibling divergence

`annotate_sibling_divergence` compares sibling distributions with Jensen–Shannon divergence and converts the result to a
chi-square p-value. In this toy example, the sibling distributions are close enough that the corrected p-value is not
significant, so `Sibling_BH_Different` stays `False`. The decomposer therefore treats the siblings as similar and merges
at the parent.

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
