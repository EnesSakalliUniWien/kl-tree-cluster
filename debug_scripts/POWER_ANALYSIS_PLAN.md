# Power Analysis Plan for Hierarchical Clustering

## Overview

**Objective**: Understand why case 18 produces 69 clusters instead of 4 by analyzing statistical power at each split decision.

**Key Insight**: With small sample sizes (n_left=1, n_right=1 common), even large true effects cannot be detected reliably.

---

## Phase 1: Data Extraction (COMPLETED)

**Script**: `extract_case_18_for_analysis.py`

**What it does**:
1. Generates exact case 18 data (400 samples, 100 features, 4 true clusters)
2. Builds hierarchical tree using Hamming distance + average linkage
3. Annotates with child-parent and sibling divergence statistics
4. Computes power for each potential split
5. Saves comprehensive CSV for analysis

**Output**: `case_18_power_analysis.csv` with columns:
- `node_id`: Internal node identifier
- `n_parent`, `n_left`, `n_right`: Sample sizes
- `depth`: Tree depth
- `cp_pval`, `cp_significant`: Child-parent test results
- `sb_pval`, `sb_different`: Sibling test results
- `power`: Statistical power (0-1)
- `n_required`: Sample size needed for 80% power
- `is_sufficient`: Whether current power ≥ 80%

---

## Phase 2: Analysis (IN JUPYTER/R)

### Analysis 1: Power Distribution

```python
# Key questions:
# 1. What fraction of splits have power < 20%? < 50%? < 80%?
# 2. How does power vary with tree depth?
# 3. What's the relationship between n_left/n_right and power?

import matplotlib.pyplot as plt

# Plot 1: Power histogram
plt.hist(df['power'], bins=20)
plt.axvline(0.2, color='r', linestyle='--', label='20% (no power)')
plt.axvline(0.8, color='g', linestyle='--', label='80% (sufficient)')
plt.xlabel('Statistical Power')
plt.ylabel('Number of Split Decisions')
plt.title('Distribution of Statistical Power')

# Plot 2: Power vs sample size
plt.scatter(df['n_left'] + df['n_right'], df['power'], alpha=0.5)
plt.axhline(0.8, color='r', linestyle='--')
plt.xlabel('Total Samples in Parent')
plt.ylabel('Power')
plt.title('Power vs Sample Size')
```

### Analysis 2: Split Decisions vs Power

```python
# Key question: How many "significant" splits occurred with insufficient power?

crosstab = pd.crosstab(
    df['is_sufficient'],
    df['sb_different'],
    margins=True
)
print(crosstab)

# Expected result: Many splits with insufficient power were declared "different"
# This is the smoking gun for over-splitting!
```

### Analysis 3: False Discovery Rate

```python
# Key question: What's the actual Type I error rate when power is low?

# Under H0 (no true difference), power = alpha = 0.05
# But we observe many more "significant" results!

low_power = df[df['power'] < 0.2]
false_discovery_rate = low_power['sb_different'].mean()
print(f"False discovery rate with power < 20%: {false_discovery_rate:.2%}")
# Expected: >> 5%!
```

### Analysis 4: Effect Size vs Power

```python
# Key question: What effect sizes CAN we detect with our sample sizes?

# Compute minimum detectable effect size (MDES) for each split
# MDES ≈ (Z_α/2 + Z_β) × √[p(1-p)(1/n1 + 1/n2)]

# Plot: MDES distribution
# If MDES > typical cluster separation → we can't detect real clusters!
```

---

## Phase 3: Expected Findings

### Finding 1: Most Splits Have No Power

**Prediction**: 70-90% of split decisions have power < 20%

**Why**: 
- Case 18 has 400 samples, 69 found clusters
- Average cluster size = 400/69 ≈ 6 samples
- Many internal nodes have n_left = 1, n_right = 1

**Impact**: Cannot reliably detect ANY true differences

### Finding 2: High False Discovery Rate

**Prediction**: 30-50% of "significant" splits have power < 20%

**Why**:
- With power = 10%, P(declared significant | H0 true) = 10%
- But multiple testing amplifies this!

**Impact**: Massive over-splitting due to noise

### Finding 3: Power Increases with Depth (Paradox)

**Prediction**: Deeper nodes have LOWER power

**Why**:
- Root: n = 400 (high power)
- Level 1: n ≈ 200 (moderate power)
- Level 5+: n ≈ 6 (no power)

**Current Bug**: TreeBH gets MORE permissive at deeper levels!

---

## Phase 4: Recommendations

### Immediate Fix: Power-Based Pruning

```python
# In tree_decomposition.py

MIN_POWER_THRESHOLD = 0.5  # Must have 50% power to split

def _should_split(self, parent):
    # ... existing gates ...
    
    # NEW: Power gate
    n_left = self._leaf_count_cache[left]
    n_right = self._leaf_count_cache[right]
    
    # Estimate power for typical effect size
    min_effect = 0.2  # Cohen's h = 0.2 (small effect)
    power = estimate_power(n_left, n_right, min_effect)
    
    if power < MIN_POWER_THRESHOLD:
        # Insufficient power → merge (conservative)
        return False
    
    # Continue with other gates...
```

### Better Fix: Minimum Cluster Size

```python
MIN_CLUSTER_SIZE = 10  # Hyperparameter

def _should_split(self, parent):
    # ... 
    
    if n_left < MIN_CLUSTER_SIZE or n_right < MIN_CLUSTER_SIZE:
        return False  # Too small to split
```

### Best Fix: Both + More Conservative FDR

```python
# Replace TreeBH with stricter correction

def conservative_correction(p_values, alpha=0.05, level=1):
    """
    More stringent than TreeBH:
    - α decreases with tree depth
    - Bonferroni within families
    """
    # Option 1: α / level (linear decrease)
    adjusted_alpha = alpha / level
    
    # Option 2: Bonferroni across all tests at this level
    # adjusted_alpha = alpha / len(p_values)
    
    return multipletests(p_values, alpha=adjusted_alpha, method='fdr_bh')
```

---

## Next Steps

1. **Run extraction script** → Generate `case_18_power_analysis.csv`
2. **Load in Jupyter** → Perform analyses 1-4
3. **Visualize findings** → Create diagnostic plots
4. **Validate hypothesis** → Confirm power explains over-splitting
5. **Design fix** → Choose between power gate, min size, or FDR correction
6. **Test fix** → Re-run case 18 with new algorithm
7. **Validate on benchmark** → Check if overall ARI improves

---

## Success Criteria

- [ ] Power analysis explains >80% of over-splitting in case 18
- [ ] Fix reduces clusters from 69 to 4-8 (closer to true K=4)
- [ ] ARI improves from 0.056 to >0.5
- [ ] No regression on other test cases
