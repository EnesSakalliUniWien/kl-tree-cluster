"""Verify Felsenstein's PIC logic and our implementation.

Felsenstein (1985) Phylogenetic Independent Contrasts:
- When comparing traits at two tips, the expected variance of the
  difference is proportional to the sum of branch lengths.
- Var(X_L - X_R) = σ² × (b_L + b_R)
- Standardized contrast: (X_L - X_R) / √(b_L + b_R)

The INTENT:
- Longer branches = more evolutionary time = more expected divergence
- Finding a large difference is NOT surprising if branches are long
- So the test should be LESS sensitive (larger p-value) for longer branches

Let's verify the math and check if our implementation matches this intent.
"""

import numpy as np
from scipy.stats import chi2


def analyze_felsenstein_logic():
    """Walk through Felsenstein's logic step by step."""
    print("=" * 70)
    print("Felsenstein's Phylogenetic Independent Contrasts: Logic Check")
    print("=" * 70)

    print("""
    FELSENSTEIN'S MODEL:
    --------------------
    - Two species L and R share a common ancestor
    - Branch lengths b_L and b_R represent evolutionary time
    - Under Brownian motion: Var(X_L - X_R) ∝ (b_L + b_R)
    
    STANDARDIZATION:
    ----------------
    Contrast = (X_L - X_R) / √(b_L + b_R)
    
    This has unit variance regardless of branch lengths.
    
    FOR HYPOTHESIS TESTING:
    -----------------------
    If we're testing H₀: L and R come from same distribution
    
    - Longer branches → we EXPECT more divergence even under H₀
    - So a large observed difference is NOT evidence against H₀
    - The test should be LESS significant (larger p-value)
    
    OUR IMPLEMENTATION:
    -------------------
    variance = base_variance × (b_L + b_R)
    z = difference / √variance
    
    Let's check if this matches Felsenstein's intent...
    """)


def test_with_different_branch_lengths():
    """Test how branch length affects significance."""
    print("\n" + "=" * 70)
    print("Effect of Branch Length on Significance")
    print("=" * 70)

    # ...existing code...
    difference = 0.15  # Observed difference
    base_variance = 0.05  # Var = p(1-p)(1/n1 + 1/n2)

    print(f"\nFixed: difference = {difference}, base_variance = {base_variance}")
    print(f"\nVarying branch_sum:")
    print(
        f"{'branch_sum':<12} {'adjusted_var':<14} {'z-score':<10} {'p-value':<12} {'interpretation'}"
    )
    print("-" * 65)

    for branch_sum in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        adjusted_var = base_variance * branch_sum
        z = difference / np.sqrt(adjusted_var)
        # For simplicity, use 1 df (one feature)
        p_val = 2 * (1 - chi2.cdf(z**2, df=1))

        if p_val < 0.05:
            interp = "SIGNIFICANT"
        else:
            interp = "not significant"

        print(
            f"{branch_sum:<12.1f} {adjusted_var:<14.4f} {z:<10.2f} {p_val:<12.6f} {interp}"
        )

    print("""
    ✓ CORRECT BEHAVIOR:
    - branch_sum = 0.1 → z = 2.12, SIGNIFICANT
    - branch_sum = 10.0 → z = 0.21, not significant
    
    Longer branches → less significant. This MATCHES Felsenstein's intent!
    """)


def analyze_the_real_issue():
    """Explain what the actual issue is."""
    print("\n" + "=" * 70)
    print("THE REAL ISSUE: Branch Length Scale")
    print("=" * 70)

    print("""
    MY PREVIOUS STATEMENT WAS IMPRECISE.
    
    The Felsenstein formula (Var × branch_sum) is CORRECT. The behavior is:
    - branch_sum > 1 → variance increases → z decreases → LESS significant ✓
    - branch_sum < 1 → variance decreases → z increases → MORE significant
    
    Both are mathematically correct applications of the formula!
    
    THE QUESTION IS: What do our branch lengths mean?
    
    SCENARIO 1: Phylogenetic context (Felsenstein's original use)
    - Branch lengths = evolutionary time or expected substitutions
    - Typically b_L + b_R > 1 (measured in substitutions per site)
    - Longer branches → more divergence expected → less significant
    
    SCENARIO 2: Linkage/hierarchical clustering (our context)
    - Branch lengths = Hamming/Euclidean distances
    - Typically small: 0 < distance < 1 for normalized data
    - b_L + b_R often < 1
    
    THE ISSUE IS NOT THE FORMULA, BUT THE INTERPRETATION:
    
    In phylogenetics:
    - Long branch = much evolutionary time = expect large differences
    - "Don't be surprised by differences, they had time to diverge"
    
    In hierarchical clustering:
    - Small distance = samples are similar
    - Large distance = samples are different
    
    THESE ARE DIFFERENT CONCEPTS!
    
    In clustering, a LARGER distance (branch length) means:
    - The samples are ALREADY known to be different (that's why distance is large)
    - We SHOULD be more likely to split, not less!
    
    This is OPPOSITE to the phylogenetic interpretation.
    """)


def correct_interpretation():
    """Provide the correct interpretation."""
    print("\n" + "=" * 70)
    print("CORRECTED ANALYSIS")
    print("=" * 70)

    print("""
    CONCLUSION:
    -----------
    1. The formula Var × (b_L + b_R) is mathematically correct for Felsenstein's PIC.
    
    2. Felsenstein's INTENT is that longer branches mean LESS significance.
       This is because in phylogenetics, long branches = expected divergence.
    
    3. Our implementation correctly applies this formula.
    
    4. HOWEVER, using this in hierarchical clustering is conceptually problematic:
       
       - In phylogenetics: branch length = evolutionary time (prior expectation)
       - In clustering: branch length = measured distance (observed data)
       
       These are fundamentally different!
    
    5. In clustering context, large distance ALREADY tells us samples are different.
       Using it to make the test LESS sensitive is circular reasoning:
       "They're far apart, so finding them different is not surprising"
       But the distance IS the evidence they're different!
    
    RECOMMENDATION:
    ---------------
    The branch length adjustment should probably be DISABLED for clustering,
    unless we have a principled way to interpret branch lengths as
    "expected divergence under the null hypothesis" rather than
    "observed evidence of divergence."
    
    MY EARLIER STATEMENT CORRECTION:
    --------------------------------
    I said: "This is opposite to Felsenstein's intent"
    
    More precisely: The FORMULA works as Felsenstein intended (longer = less sig),
    but APPLYING it to clustering distances may not make sense because
    the semantic meaning of "branch length" is different in the two contexts.
    """)


if __name__ == "__main__":
    analyze_felsenstein_logic()
    test_with_different_branch_lengths()
    analyze_the_real_issue()
    correct_interpretation()
