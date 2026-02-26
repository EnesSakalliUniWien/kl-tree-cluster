"""Bootstrap consensus analysis on real HC gene-pathway data.

Dataset: 317 hypertrophic cardiomyopathy genes × 1834 Reactome pathways
(binary membership matrix, density 3.1%).

This is the real-data counterpart of test_bootstrap.py.  No ground truth
labels exist, so we rely entirely on:

  1. Bootstrap consensus — stability of the found clusters under resampling.
     Answers: "Are these clusters real or artefacts of this particular sample?"

  2. Classical validation (silhouette, CH, DB) — internal geometric quality.
     Answers: "Do the clusters have good separation in Hamming space?"

  3. Cross-diagnostic — combining both signals.
     Answers: "Are there clusters that LOOK geometric but DON'T replicate
     (Brownian artefacts)?  Or clusters that replicate but have poor
     geometry (consistent overlap)?"

Previous results at α=0.05:
  - 85 clusters found
  - 1 dominant cluster (174 genes, 55%) with sparse pathway annotations
  - 24 smaller clusters (2–11 genes) with coherent biology
  - 60 singletons

We expect the bootstrap to reveal:
  - The 24 coherent modules should have HIGH stability
  - Singletons trivially have stability=1.0 (vacuously)
  - The dominant cluster (174 genes) may show internal instability
    if it's a catch-all for heterogeneous low-signal genes
"""

import warnings
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.bootstrap_consensus import bootstrap_consensus

warnings.filterwarnings("ignore", category=RuntimeWarning)  # suppress statsmodels GLM warnings


def _classical_validation(data, cluster_labels, metric="hamming"):
    """Compute silhouette, Calinski-Harabasz, Davies-Bouldin on found clusters."""
    labels = np.asarray(cluster_labels)
    unique_k = len(np.unique(labels))
    X = data.values
    result = {"n_clusters": unique_k}

    if unique_k < 2 or unique_k >= len(labels):
        result["silhouette"] = float("nan")
        result["calinski_harabasz"] = float("nan")
        result["davies_bouldin"] = float("nan")
        return result

    result["silhouette"] = silhouette_score(X, labels, metric=metric)
    result["calinski_harabasz"] = calinski_harabasz_score(X, labels)
    result["davies_bouldin"] = davies_bouldin_score(X, labels)
    return result


def main():
    # --- Load real data ---
    data_path = "data/HC_feature_matrix_GO_CC.tsv"
    print(f"Loading {data_path} ...")
    data = pd.read_csv(data_path, sep="\t", index_col=0)
    n, p = data.shape
    print(f"  {n} genes × {p} pathways, density={data.values.mean():.3f}")
    print(f"  Already binary: unique values = {sorted(pd.unique(data.values.ravel()))}")

    # --- Run bootstrap consensus ---
    # Use 50 replicates (317 genes × 1834 features is heavier than synthetic cases)
    n_boot = 50
    print(f"\nRunning bootstrap consensus with {n_boot} replicates ...")
    print(f"  metric={config.TREE_DISTANCE_METRIC}  linkage={config.TREE_LINKAGE_METHOD}")
    print("  alpha_local=0.05  sibling_alpha=0.05")
    print()

    result = bootstrap_consensus(
        data,
        n_boot=n_boot,
        alpha_local=0.05,
        sibling_alpha=0.05,
        random_seed=42,
        verbose=True,
    )

    orig = result["original_results"]
    k_orig = orig["num_clusters"]
    ca = orig.get("cluster_assignments", {})

    print(f"\n{'='*70}")
    print("  HC Gene-Pathway Bootstrap Analysis")
    print(f"  n={n}  p={p}  found K={k_orig}")
    print(f"{'='*70}")

    # --- Build label vector ---
    gene_to_cluster = {}
    for cid, info in ca.items():
        for leaf in info["leaves"]:
            gene_to_cluster[leaf] = cid
    y_found = [gene_to_cluster.get(gene, -1) for gene in data.index]

    # --- Classical validation on found clusters ---
    cv = _classical_validation(data, y_found, metric=config.TREE_DISTANCE_METRIC)
    print(f"\n  Classical validation (FOUND clusters, K={cv['n_clusters']}):")
    print(
        f"    Silhouette       = {cv['silhouette']:+.3f}   "
        f"{'good' if cv['silhouette'] > 0.5 else 'overlap' if cv['silhouette'] > 0.25 else 'WEAK'}"
    )
    print(f"    Calinski-Harabasz= {cv['calinski_harabasz']:.1f}")
    print(
        f"    Davies-Bouldin   = {cv['davies_bouldin']:.3f}   "
        f"{'good' if cv['davies_bouldin'] < 1.0 else 'overlap' if cv['davies_bouldin'] < 2.0 else 'POOR'}"
    )

    # --- K distribution ---
    k_dist = result["k_distribution"]
    print(f"\n  Bootstrap K distribution ({len(k_dist)} replicates):")
    for k, cnt in sorted(Counter(k_dist).items()):
        pct = 100 * cnt / len(k_dist)
        bar = "█" * max(1, int(pct / 2))
        print(f"    K={k:3d}: {cnt:3d} ({pct:4.0f}%) {bar}")
    k_arr = np.array(k_dist)
    print(
        f"    mean K = {k_arr.mean():.1f}  std = {k_arr.std():.1f}  "
        f"range = [{k_arr.min()}, {k_arr.max()}]"
    )

    # --- Cluster stability ---
    stab = result["cluster_stability"]
    sizes = {cid: ca[cid]["size"] for cid in ca}

    # Partition clusters by stability
    stable = {cid: s for cid, s in stab.items() if s >= 0.7}
    borderline = {cid: s for cid, s in stab.items() if 0.5 <= s < 0.7}
    unstable = {cid: s for cid, s in stab.items() if s < 0.5}

    print(f"\n  Cluster stability summary ({len(stab)} clusters):")
    print(
        f"    Stable (≥0.7):     {len(stable):3d} clusters, "
        f"{sum(sizes[c] for c in stable):3d} genes"
    )
    print(
        f"    Borderline (0.5–0.7): {len(borderline):3d} clusters, "
        f"{sum(sizes[c] for c in borderline):3d} genes"
    )
    print(
        f"    Unstable (<0.5):   {len(unstable):3d} clusters, "
        f"{sum(sizes[c] for c in unstable):3d} genes"
    )

    # --- Show non-singleton clusters sorted by stability ---
    nonsingleton = [(cid, stab[cid], sizes[cid]) for cid in ca if sizes[cid] > 1]
    nonsingleton.sort(key=lambda x: x[1])  # lowest stability first

    print(f"\n  Non-singleton clusters sorted by stability " f"({len(nonsingleton)} clusters):")
    for cid, s, sz in nonsingleton:
        tag = ""
        if s < 0.5:
            tag = " ← UNSTABLE (likely noise/Brownian)"
        elif s < 0.7:
            tag = " ← borderline"
        genes = ca[cid]["leaves"]
        gene_str = ", ".join(genes[:5])
        if len(genes) > 5:
            gene_str += f" ... (+{len(genes)-5})"
        print(f"    Cluster {cid:3d} (n={sz:3d}): stab={s:.3f}  " f"genes=[{gene_str}]{tag}")

    # --- Clade support (top 15) ---
    clades_sorted = sorted(result["clade_support"].items(), key=lambda x: len(x[0]), reverse=True)
    print("\n  Top 15 clade supports:")
    for clade, support in clades_sorted[:15]:
        print(f"    size={len(clade):3d}  support={support:.2f}")

    # --- Co-association matrix summary ---
    co = result["co_association_matrix"].values
    mask = np.triu(np.ones_like(co, dtype=bool), k=1)
    upper = co[mask]
    print("\n  Co-association fractions:")
    print(f"    >0.9 (always together):    {(upper > 0.9).mean():.1%}")
    print(f"    <0.1 (never together):     {(upper < 0.1).mean():.1%}")
    print(f"    0.3-0.7 (AMBIGUOUS):       {((upper >= 0.3) & (upper <= 0.7)).mean():.1%}")
    print(f"    mean={upper.mean():.3f}  median={np.median(upper):.3f}")

    # --- Diagnostic ---
    mean_stability = np.mean(list(stab.values()))
    sil = cv["silhouette"]
    print(f"\n  Diagnostic (mean stability={mean_stability:.3f}, silhouette={sil:+.3f}):")
    if sil > 0.25 and mean_stability > 0.7:
        print("    -> REAL CLUSTERS: geometric structure + bootstrap stable")
    elif sil > 0.25 and mean_stability <= 0.7:
        print("    -> BROWNIAN/NOISE ARTEFACT: looks geometric but NOT stable")
    elif sil <= 0.25 and mean_stability > 0.7:
        print("    -> CONSISTENT OVERLAP: stable partition but clusters overlap")
        print("       Real biological modules with shared pathway annotations.")
    else:
        print("    -> NOISE OVER-SPLIT: no structure, not stable — pure artefact")

    # --- Investigate the dominant cluster ---
    biggest_cid = max(ca, key=lambda c: ca[c]["size"])
    biggest_size = ca[biggest_cid]["size"]
    biggest_stab = stab[biggest_cid]
    print(
        f"\n  Dominant cluster analysis (cluster {biggest_cid}, "
        f"n={biggest_size}, {100*biggest_size/n:.0f}% of genes):"
    )
    print(f"    Stability = {biggest_stab:.3f}")
    if biggest_stab < 0.7:
        print("    -> Catch-all cluster is UNSTABLE — heterogeneous low-signal genes")
        print("       that get assigned together by default, not because they're similar.")
    else:
        print("    -> Catch-all cluster is stable — these genes genuinely share")
        print("       sparse/background pathway profiles.")

    # --- Save co-association matrix for downstream use ---
    out_path = "data/results_GO_CC_alpha_005/bootstrap_co_association.tsv"
    result["co_association_matrix"].to_csv(out_path, sep="\t")
    print(f"\n  Co-association matrix saved to {out_path}")

    print(f"\n{'='*70}")
    print("  ANALYSIS COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
