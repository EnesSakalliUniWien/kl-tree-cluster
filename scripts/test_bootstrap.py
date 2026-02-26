"""Stress-test bootstrap consensus with hard cases.

Cases include:
  1. Heavy overlap (Gaussian clusters nearly touching)
  2. Correlated / redundant features (rank-deficient design)
  3. Block-correlated features with noise columns
  4. Unbalanced clusters (1 big + several tiny)
  5. Chain topology (elongated, non-spherical clusters)
  6. High-dimensional sparse (d >> n)
  7. AR(1) correlated features (rho=0.8)
  8. Near-singular (rank ~ 3, 60 features)

Metrics & Interpretation
========================

Bootstrap consensus metrics
----------------------------
- **Co-association matrix C** (n × n): Entry C[i,j] = fraction of bootstrap
  replicates where samples i and j co-clustered (given both were sampled).

- **Co-association fractions** summarize the distribution of C's upper triangle:
    - *>0.9*  : "hard core" — pairs that ALWAYS cluster together.  High = stable
                clusters with clear boundaries.
    - *<0.1*  : "separated" — pairs that NEVER cluster together.  High = clear
                between-cluster separation.
    - *0.3–0.7 (ambiguous)*: Pairs where the algorithm cannot decide.  This is the
                KEY instability indicator.  High ambiguous% means the cluster boundary
                is data-dependent — resampling shifts samples across clusters.

  Interpretation:
    - Clean block-diagonal C (high >0.9, high <0.1, low ambiguous) → robust.
    - Fuzzy C (high ambiguous) → the found clusters are artefacts of this particular
      sample, not stable structure.

- **Cluster stability**: Mean intra-cluster co-association.  A cluster with
  stability < 0.7 is likely not a real entity — its members scatter across
  different clusters under resampling.

- **Clade support**: Fraction of bootstrap trees containing this exact subtree
  (restricted to the samples present in each resample).  Analogous to Felsenstein
  bootstrap support in phylogenetics.

Classical clustering validation metrics
----------------------------------------
These metrics evaluate the FOUND clusters on the ORIGINAL data, regardless of
ground truth.  They help answer: "Are these noise-induced clusters or do they
have internal geometric structure?"

- **Silhouette score** (range [-1, 1]):
    Mean of (b - a) / max(a, b) where a = mean intra-cluster distance,
    b = mean nearest-cluster distance.
    - >0.5  : reasonable structure
    - 0.25–0.5: overlapping but distinguishable
    - <0.25 : no meaningful structure — clusters are likely Brownian/noise artefacts

- **Calinski-Harabasz (CH) index** (higher = better):
    Ratio of between-cluster dispersion to within-cluster dispersion, scaled by
    degrees of freedom: CH = [trace(B)/(K-1)] / [trace(W)/(n-K)].
    No absolute threshold — compare across K values.  Meaningless for K=1.

- **Davies-Bouldin (DB) index** (lower = better):
    Mean of max_j[(s_i + s_j) / d(c_i, c_j)] over clusters, where s_i is the
    mean distance of cluster i's members to its centroid and d(c_i,c_j) is the
    centroid distance.
    - <1.0  : well-separated clusters
    - >2.0  : heavy overlap or noise

The key diagnostic combination:
  - High silhouette + high stability     → real, robust clusters
  - High silhouette + LOW stability      → Brownian-motion artefact: the data's
    random walk creates local density bumps that look geometric but don't
    replicate under resampling
  - Low silhouette + high stability      → consistent but overlapping structure
    (possibly finer substructure within true clusters)
  - Low silhouette + low stability       → pure noise / over-splitting
"""

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.bootstrap_consensus import bootstrap_consensus


def _binarize(X):
    return (X > np.median(X, axis=0)).astype(int)


def _classical_validation(data, cluster_labels, metric="hamming"):
    """Compute classical clustering validation metrics on the found clusters.

    These are evaluated on the ORIGINAL (non-resampled) data using the
    clusters the algorithm actually found — NOT ground truth.  This tells us
    whether the found partition has real geometric structure or is just
    picking up noise / Brownian-motion artefacts.

    Parameters
    ----------
    data : pd.DataFrame
        Binary sample × feature matrix.
    cluster_labels : array-like of int
        Cluster assignment for each sample (from decompose output).
    metric : str
        Distance metric for silhouette/DB (default: hamming, matching the
        pipeline's tree-construction metric).

    Returns
    -------
    dict with keys: silhouette, calinski_harabasz, davies_bouldin, n_clusters
    """
    labels = np.asarray(cluster_labels)
    unique_k = len(np.unique(labels))
    X = data.values

    result = {"n_clusters": unique_k}

    if unique_k < 2 or unique_k >= len(labels):
        # Metrics undefined for K=1 or K=n
        result["silhouette"] = float("nan")
        result["calinski_harabasz"] = float("nan")
        result["davies_bouldin"] = float("nan")
        return result

    # Silhouette & Davies-Bouldin use a distance metric
    # For binary data, hamming distance is the natural choice
    result["silhouette"] = silhouette_score(X, labels, metric=metric)

    # Calinski-Harabasz uses Euclidean internally (variance ratio) —
    # still meaningful on binary data as it measures spread
    result["calinski_harabasz"] = calinski_harabasz_score(X, labels)

    # Davies-Bouldin uses Euclidean centroids
    result["davies_bouldin"] = davies_bouldin_score(X, labels)

    return result


def _report(name, data, y_true, n_boot=50):
    """Run bootstrap consensus and print a full diagnostic report.

    Combines:
    1. Bootstrap consensus metrics (co-association, stability, clade support)
    2. Classical validation on FOUND clusters (silhouette, CH, DB)
    3. Classical validation on TRUE clusters (for comparison)

    The comparison between found-cluster validation and bootstrap stability
    is the key diagnostic:
      - "Brownian artefacts" show high silhouette but low bootstrap stability
      - "Real clusters" show high silhouette AND high stability
      - "Noise splits" show low silhouette AND low stability
    """
    true_k = len(np.unique(y_true))
    print(f"\n{'='*70}")
    print(f"  CASE: {name}")
    print(f"  n={data.shape[0]}  p={data.shape[1]}  true_K={true_k}")
    print(f"{'='*70}")

    result = bootstrap_consensus(data, n_boot=n_boot, random_seed=42, verbose=False)

    orig = result["original_results"]
    k_orig = orig["num_clusters"]

    # --- Build predicted labels ---
    ca = orig.get("cluster_assignments", {})
    pred = {}
    for cid, info in ca.items():
        for leaf in info["leaves"]:
            pred[leaf] = cid
    y_pred = [pred.get(f"S{i}", -1) for i in range(len(y_true))]
    ari = adjusted_rand_score(y_true, y_pred)

    print(f"  Original K={k_orig}  ARI={ari:.3f}")

    # --- Classical validation on FOUND clusters ---
    cv_found = _classical_validation(data, y_pred, metric=config.TREE_DISTANCE_METRIC)
    print(f"\n  Classical validation (FOUND clusters, K={cv_found['n_clusters']}):")
    print(
        f"    Silhouette       = {cv_found['silhouette']:+.3f}   "
        f"{'good' if cv_found['silhouette'] > 0.5 else 'overlap' if cv_found['silhouette'] > 0.25 else 'WEAK'}"
    )
    print(f"    Calinski-Harabasz= {cv_found['calinski_harabasz']:.1f}")
    print(
        f"    Davies-Bouldin   = {cv_found['davies_bouldin']:.3f}   "
        f"{'good' if cv_found['davies_bouldin'] < 1.0 else 'overlap' if cv_found['davies_bouldin'] < 2.0 else 'POOR'}"
    )

    # --- Classical validation on TRUE clusters (baseline) ---
    cv_true = _classical_validation(data, y_true, metric=config.TREE_DISTANCE_METRIC)
    print(f"  Classical validation (TRUE clusters, K={cv_true['n_clusters']}):")
    print(f"    Silhouette       = {cv_true['silhouette']:+.3f}")
    print(f"    Calinski-Harabasz= {cv_true['calinski_harabasz']:.1f}")
    print(f"    Davies-Bouldin   = {cv_true['davies_bouldin']:.3f}")

    # --- Diagnostic: are the found clusters Brownian artefacts? ---
    mean_stability = np.mean(list(result["cluster_stability"].values()))
    sil_found = cv_found["silhouette"]
    print(f"\n  Diagnostic (mean stability={mean_stability:.3f}, silhouette={sil_found:+.3f}):")
    if np.isnan(sil_found):
        print("    -> K=1, no cluster structure detected")
    elif sil_found > 0.25 and mean_stability > 0.7:
        print("    -> REAL CLUSTERS: geometric structure + bootstrap stable")
    elif sil_found > 0.25 and mean_stability <= 0.7:
        print("    -> BROWNIAN/NOISE ARTEFACT: looks geometric but NOT stable")
        print("       The random walk in feature space creates local density")
        print("       bumps that don't replicate under resampling.")
    elif sil_found <= 0.25 and mean_stability > 0.7:
        print("    -> CONSISTENT OVERLAP: stable partition but clusters overlap")
        print("       Possibly finer substructure within ground-truth clusters.")
    else:
        print("    -> NOISE OVER-SPLIT: no structure, not stable — pure artefact")

    # --- Bootstrap K distribution ---
    k_dist = result["k_distribution"]
    print(f"\n  Bootstrap K distribution ({len(k_dist)} replicates):")
    for k, cnt in sorted(Counter(k_dist).items()):
        bar = "█" * cnt
        print(f"    K={k:2d}: {cnt:3d} ({100*cnt/len(k_dist):4.0f}%) {bar}")

    # --- Cluster stability ---
    print("  Cluster stability:")
    for cid, stab in sorted(result["cluster_stability"].items()):
        size = ca[cid]["size"]
        tag = ""
        if stab < 0.5:
            tag = " ← UNSTABLE (likely noise)"
        elif stab < 0.7:
            tag = " ← borderline"
        print(f"    Cluster {cid} (n={size:3d}): {stab:.3f}{tag}")

    # --- Clade support (top 5) ---
    clades_sorted = sorted(result["clade_support"].items(), key=lambda x: len(x[0]), reverse=True)
    print("  Top 5 clade supports:")
    for clade, support in clades_sorted[:5]:
        print(f"    size={len(clade):3d}  support={support:.2f}")

    # --- Co-association fractions ---
    # These summarize the upper triangle of the co-association matrix C.
    #
    # C[i,j] = (# replicates where i,j co-cluster) / (# replicates where
    #           both i,j are present in the bootstrap sample).
    #
    # The three bins partition ALL sample-pairs:
    #   >0.9  = "always together"    → stable cluster core
    #   <0.1  = "never together"     → clearly separated
    #   0.3–0.7 = "ambiguous"        → boundary instability
    #
    # A perfectly stable K-cluster solution has:
    #   >0.9 ≈ 1/K  (within-cluster pairs)
    #   <0.1 ≈ 1-1/K (between-cluster pairs)
    #   ambiguous ≈ 0%
    #
    # When noise/Brownian motion creates spurious clusters, the "ambiguous"
    # fraction rises because resampling collapses or rearranges those splits.
    co = result["co_association_matrix"].values
    mask = np.triu(np.ones_like(co, dtype=bool), k=1)
    upper = co[mask]
    print(
        f"  Co-association fractions:  "
        f">0.9: {(upper > 0.9).mean():.1%}  "
        f"<0.1: {(upper < 0.1).mean():.1%}  "
        f"0.3-0.7: {((upper >= 0.3) & (upper <= 0.7)).mean():.1%} (ambiguous)"
    )

    return result


def main():
    rng = np.random.default_rng(2026)

    # ================================================================
    # CASE 1: Heavy Gaussian overlap (cluster_std = 3.0, centers close)
    # ================================================================
    n, p, k = 120, 40, 4
    centers = rng.standard_normal((k, p)) * 1.5
    labels = np.repeat(np.arange(k), n // k)
    X = centers[labels] + rng.standard_normal((n, p)) * 3.0
    data = pd.DataFrame(_binarize(X), index=[f"S{i}" for i in range(n)])
    _report("Heavy Gaussian overlap (std=3.0, close centers)", data, labels)

    # ================================================================
    # CASE 2: Correlated / redundant features (rank ~ 5 out of 60)
    # ================================================================
    n, p_latent, p_observed, k = 99, 5, 60, 3
    centers_latent = rng.standard_normal((k, p_latent)) * 3.0
    labels = np.repeat(np.arange(k), n // k)
    rng.shuffle(labels)
    Z_latent = centers_latent[labels] + rng.standard_normal((n, p_latent)) * 0.8
    mixing = rng.standard_normal((p_latent, p_observed))
    X_full = Z_latent @ mixing + rng.standard_normal((n, p_observed)) * 0.3
    data = pd.DataFrame(_binarize(X_full), index=[f"S{i}" for i in range(n)])
    _report("Correlated features (rank~5 -> 60d)", data, labels)

    # ================================================================
    # CASE 3: Block-correlated features with noise columns
    # ================================================================
    n, k = 100, 4
    labels = np.repeat(np.arange(k), n // k)
    rng.shuffle(labels)
    X_signal = rng.standard_normal((n, 20)) * 0.5
    for ci in range(k):
        mask = labels == ci
        X_signal[mask, ci * 5 : (ci + 1) * 5] += 3.0
    corr_idx = rng.integers(0, 20, size=30)
    X_correlated = X_signal[:, corr_idx] + rng.standard_normal((n, 30)) * 0.5
    X_noise = rng.standard_normal((n, 20))
    X_full = np.hstack([X_signal, X_correlated, X_noise])
    data = pd.DataFrame(_binarize(X_full), index=[f"S{i}" for i in range(n)])
    _report("Block-correlated + 20 noise cols (70d)", data, labels)

    # ================================================================
    # CASE 4: Unbalanced clusters (60 + 15 + 10 + 5)
    # ================================================================
    sizes = [60, 15, 10, 5]
    n = sum(sizes)
    labels = np.concatenate([np.full(s, i) for i, s in enumerate(sizes)])
    p = 40
    centers = np.zeros((4, p))
    centers[0, :10] = 2.0
    centers[1, 10:20] = 2.5
    centers[2, 20:30] = 3.0
    centers[3, 30:40] = 3.5
    X = centers[labels] + rng.standard_normal((n, p)) * 1.0
    data = pd.DataFrame(_binarize(X), index=[f"S{i}" for i in range(n)])
    _report("Unbalanced clusters (60/15/10/5)", data, labels)

    # ================================================================
    # CASE 5: Chain / elongated topology (3 clusters in a line)
    # ================================================================
    n, p, k = 90, 30, 3
    labels = np.repeat(np.arange(k), n // k)
    centers = np.zeros((k, p))
    for ci in range(k):
        centers[ci, :] = ci * 0.8
    X = centers[labels] + rng.standard_normal((n, p)) * 1.2
    data = pd.DataFrame(_binarize(X), index=[f"S{i}" for i in range(n)])
    _report("Chain topology (3 clusters, small gap)", data, labels)

    # ================================================================
    # CASE 6: High-dimensional sparse (d=500, n=60)
    # ================================================================
    n, p, k = 60, 500, 3
    labels = np.repeat(np.arange(k), n // k)
    X = rng.binomial(1, 0.1, size=(n, p)).astype(float)
    for ci in range(k):
        mask = labels == ci
        feat_start = ci * 30
        X[mask, feat_start : feat_start + 30] = rng.binomial(1, 0.7, size=(mask.sum(), 30))
    data = pd.DataFrame(X.astype(int), index=[f"S{i}" for i in range(n)])
    _report("High-d sparse (n=60, p=500)", data, labels)

    # ================================================================
    # CASE 7: Two overlapping + AR(1) correlated features (rho=0.8)
    # ================================================================
    n, p, k = 100, 30, 2
    labels = np.repeat(np.arange(k), n // k)
    rng.shuffle(labels)
    cov = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            cov[i, j] = 0.8 ** abs(i - j)
    L = np.linalg.cholesky(cov)
    centers = np.zeros((k, p))
    centers[0, :15] = 1.5
    centers[1, 15:] = 1.5
    X = np.zeros((n, p))
    for i in range(n):
        X[i] = centers[labels[i]] + L @ rng.standard_normal(p)
    data = pd.DataFrame(_binarize(X), index=[f"S{i}" for i in range(n)])
    _report("AR(1) correlated features (rho=0.8, K=2)", data, labels)

    # ================================================================
    # CASE 8: Near-singular -- 3 clusters, features are almost copies
    # ================================================================
    n, k = 90, 3
    labels = np.repeat(np.arange(k), n // k)
    rng.shuffle(labels)
    X_latent = np.zeros((n, 3))
    for ci in range(k):
        X_latent[labels == ci, ci] = 2.0
    X_latent += rng.standard_normal((n, 3)) * 0.5
    X_full = np.hstack([X_latent[:, [j]] + rng.standard_normal((n, 20)) * 0.05 for j in range(3)])
    data = pd.DataFrame(_binarize(X_full), index=[f"S{i}" for i in range(n)])
    _report("Near-singular (rank~3, 60 features)", data, labels)

    print(f"\n{'='*70}")
    print("  ALL CASES COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
