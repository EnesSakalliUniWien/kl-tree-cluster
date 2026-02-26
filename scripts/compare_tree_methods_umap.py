"""UMAP comparison of three tree inference methods on HC gene-pathway data.

Methods compared:
  1. hamming   + average   → K=85 (174 biggest, 60 singletons)
  2. hamming   + complete  → K=23 (168 biggest,  6 singletons)
  3. rogerstanimoto + average → K=85 (174 biggest, 60 singletons)

Produces a 3×2 panel figure:
  Row 1: UMAP colored by cluster assignment (each method)
  Row 2: Same UMAP but highlighting singletons vs non-singletons

All three use the SAME UMAP embedding (hamming-based) so differences
are purely from the clustering, not from different projections.
"""

import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from collections import Counter

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import umap
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score

from kl_clustering_analysis.tree.poset_tree import PosetTree

warnings.filterwarnings("ignore")

# ── Load data ────────────────────────────────────────────────────────
DATA_PATH = "data/HC_feature_matrix_GO_CC.tsv"
data = pd.read_csv(DATA_PATH, sep="\t", index_col=0)
n, p = data.shape
X = data.values
genes = data.index.tolist()
print(f"Data: {n} genes × {p} pathways")

# ── Compute UMAP embedding (one shared embedding) ───────────────────
print("Computing UMAP embedding (hamming metric) ...")
reducer = umap.UMAP(
    n_components=2,
    metric="hamming",
    n_neighbors=15,
    min_dist=0.3,
    random_state=42,
)
embedding = reducer.fit_transform(X)
print(f"  Embedding shape: {embedding.shape}")

# ── Run three decompositions ─────────────────────────────────────────
METHODS = [
    ("hamming", "average", "hamming + average"),
    ("hamming", "complete", "hamming + complete"),
    ("rogerstanimoto", "average", "rogerstanimoto + average"),
]

decomps = {}
for metric, link, label in METHODS:
    print(f"\nDecomposing: {label} ...")
    d = pdist(X, metric=metric)
    Z = linkage(d, method=link)
    tree = PosetTree.from_linkage(Z, leaf_names=genes)
    res = tree.decompose(leaf_data=data, alpha_local=0.05, sibling_alpha=0.05)

    ca = res["cluster_assignments"]
    K = res["num_clusters"]

    # Build per-gene label vector
    gene_to_cid = {}
    for cid, info in ca.items():
        for leaf in info["leaves"]:
            gene_to_cid[leaf] = cid

    labels = np.array([gene_to_cid.get(g, -1) for g in genes])
    sizes = {cid: info["size"] for cid, info in ca.items()}
    singletons = set(cid for cid, sz in sizes.items() if sz == 1)
    is_singleton = np.array([gene_to_cid.get(g, -1) in singletons for g in genes])

    # Silhouette (only meaningful if 2 ≤ K < n)
    n_unique = len(np.unique(labels))
    sil = silhouette_score(X, labels, metric="hamming") if 2 <= n_unique < n else float("nan")

    decomps[label] = {
        "labels": labels,
        "K": K,
        "sizes": sizes,
        "singletons": singletons,
        "is_singleton": is_singleton,
        "ca": ca,
        "silhouette": sil,
    }
    n_sing = len(singletons)
    n_nonsingleton = K - n_sing
    biggest = max(sizes.values())
    print(
        f"  K={K}, biggest={biggest}, non-singleton={n_nonsingleton}, "
        f"singletons={n_sing}, silhouette={sil:.3f}"
    )


# ── Shared cluster coloring logic ────────────────────────────────────
def _make_cluster_colors(labels, sizes_dict, singletons_set):
    """Assign colors: large clusters get distinct hues, singletons are grey."""
    unique_cids = sorted(set(labels))
    # Sort non-singleton clusters by size (largest first) for color priority
    nonsingleton_cids = sorted(
        [c for c in unique_cids if c not in singletons_set],
        key=lambda c: sizes_dict.get(c, 0),
        reverse=True,
    )

    # Use tab20 + tab20b + tab20c for up to 60 distinct colors
    base_colors = (
        list(plt.cm.tab20(np.linspace(0, 1, 20)))
        + list(plt.cm.tab20b(np.linspace(0, 1, 20)))
        + list(plt.cm.tab20c(np.linspace(0, 1, 20)))
    )

    cid_to_color = {}
    for i, cid in enumerate(nonsingleton_cids):
        cid_to_color[cid] = base_colors[i % len(base_colors)]
    for cid in singletons_set:
        cid_to_color[cid] = (0.7, 0.7, 0.7, 0.5)  # grey, semi-transparent

    colors = np.array([cid_to_color.get(l, (0.5, 0.5, 0.5, 0.5)) for l in labels])
    return colors, nonsingleton_cids, cid_to_color


# ── Build figure ─────────────────────────────────────────────────────
print("\nBuilding figure ...")
fig, axes = plt.subplots(3, 2, figsize=(18, 24))
fig.suptitle(
    "HC Gene-Pathway Data: Tree Inference Method Comparison (UMAP projection)",
    fontsize=16,
    fontweight="bold",
    y=0.98,
)

for row, (metric, link, label) in enumerate(METHODS):
    info = decomps[label]
    labels_arr = info["labels"]
    is_sing = info["is_singleton"]
    colors, nonsingleton_cids, cid_to_color = _make_cluster_colors(
        labels_arr, info["sizes"], info["singletons"]
    )

    # ── Left panel: clusters colored ──
    ax = axes[row, 0]
    # Draw singletons first (background)
    mask_s = is_sing
    mask_ns = ~is_sing
    ax.scatter(
        embedding[mask_s, 0],
        embedding[mask_s, 1],
        c=[(0.7, 0.7, 0.7, 0.4)],
        s=12,
        marker="x",
        linewidths=0.5,
        zorder=1,
    )
    ax.scatter(
        embedding[mask_ns, 0],
        embedding[mask_ns, 1],
        c=colors[mask_ns],
        s=30,
        edgecolors="black",
        linewidths=0.3,
        zorder=2,
    )
    ax.set_title(
        f"{label}\nK={info['K']}  "
        f"(non-singleton={len(nonsingleton_cids)}, "
        f"singletons={len(info['singletons'])})  "
        f"sil={info['silhouette']:.3f}",
        fontsize=11,
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    # Legend: top-5 largest clusters
    top5 = nonsingleton_cids[:5]
    patches = []
    for cid in top5:
        sz = info["sizes"][cid]
        c = cid_to_color[cid]
        # Find a representative gene
        rep_genes = [g for g, l in zip(genes, labels_arr) if l == cid][:3]
        gene_str = ", ".join(rep_genes)
        patches.append(mpatches.Patch(color=c, label=f"C{cid} (n={sz}): {gene_str}"))
    patches.append(
        mpatches.Patch(
            color=(0.7, 0.7, 0.7, 0.5), label=f"singletons (n={len(info['singletons'])})"
        )
    )
    ax.legend(handles=patches, loc="lower right", fontsize=7, framealpha=0.8)

    # ── Right panel: singleton anatomy ──
    ax2 = axes[row, 1]

    # Color by category: biggest cluster, other non-singleton, singleton
    biggest_cid = max(info["sizes"], key=info["sizes"].get)
    cat_colors = []
    for g, l in zip(genes, labels_arr):
        if l in info["singletons"]:
            cat_colors.append("red")
        elif l == biggest_cid:
            cat_colors.append("steelblue")
        else:
            cat_colors.append("forestgreen")

    ax2.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=cat_colors,
        s=20,
        edgecolors="black",
        linewidths=0.2,
        alpha=0.7,
    )
    ax2.set_title(
        f"{label} — Cluster Anatomy",
        fontsize=11,
    )
    ax2.set_xlabel("UMAP 1")
    ax2.set_ylabel("UMAP 2")

    biggest_sz = info["sizes"][biggest_cid]
    n_other_ns = sum(1 for c in nonsingleton_cids if c != biggest_cid)
    n_other_genes = sum(info["sizes"][c] for c in nonsingleton_cids if c != biggest_cid)
    patches2 = [
        mpatches.Patch(
            color="steelblue", label=f"Dominant cluster (n={biggest_sz}, {100*biggest_sz/n:.0f}%)"
        ),
        mpatches.Patch(
            color="forestgreen",
            label=f"Other clusters ({n_other_ns} clusters, {n_other_genes} genes)",
        ),
        mpatches.Patch(color="red", label=f"Singletons (n={len(info['singletons'])})"),
    ]
    ax2.legend(handles=patches2, loc="lower right", fontsize=8, framealpha=0.8)

plt.tight_layout(rect=[0, 0, 1, 0.96])

out_path = "data/results_GO_CC_alpha_005/tree_method_umap_comparison.pdf"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nFigure saved to {out_path}")

# ── Numerical overlap analysis ───────────────────────────────────────
print(f"\n{'='*70}")
print("  Overlap Analysis: Which genes move between methods?")
print(f"{'='*70}")

# Compare hamming+average vs hamming+complete
ha = decomps["hamming + average"]
hc = decomps["hamming + complete"]
ra = decomps["rogerstanimoto + average"]

# Singletons that get absorbed in complete-linkage
ha_singletons = set(g for g, l in zip(genes, ha["labels"]) if l in ha["singletons"])
hc_singletons = set(g for g, l in zip(genes, hc["labels"]) if l in hc["singletons"])

absorbed = ha_singletons - hc_singletons
print(f"\n  hamming+average singletons: {len(ha_singletons)}")
print(f"  hamming+complete singletons: {len(hc_singletons)}")
print(f"  Singletons absorbed by complete linkage: {len(absorbed)}")

# Where did they go?
hc_gene_to_cid = {g: l for g, l in zip(genes, hc["labels"])}
absorbed_destinations = Counter()
for g in absorbed:
    dest_cid = hc_gene_to_cid[g]
    dest_size = hc["sizes"][dest_cid]
    absorbed_destinations[dest_cid] += 1

print("\n  Destination of absorbed singletons (complete linkage):")
for dest_cid, count in absorbed_destinations.most_common(10):
    sz = hc["sizes"][dest_cid]
    rep = [g for g, l in zip(genes, hc["labels"]) if l == dest_cid][:3]
    print(
        f"    Cluster {dest_cid} (n={sz}): absorbed {count} singletons  " f"[{', '.join(rep)}...]"
    )

# Compare hamming+average vs rogerstanimoto+average
print("\n  hamming+average vs rogerstanimoto+average:")
agree = (ha["labels"] == ra["labels"]).sum()
print(f"    Identical labels: {agree}/{n} ({100*agree/n:.0f}%)")
# Check if they're actually the same partition
from sklearn.metrics import adjusted_rand_score

# Check if they're actually the same partition

ari = adjusted_rand_score(ha["labels"], ra["labels"])
print(f"    ARI between partitions: {ari:.4f}")

# Non-singleton cluster genes overlap
ha_ns_genes = set(g for g, l in zip(genes, ha["labels"]) if l not in ha["singletons"])
hc_ns_genes = set(g for g, l in zip(genes, hc["labels"]) if l not in hc["singletons"])
print("\n  Non-singleton gene sets:")
print(f"    hamming+average:  {len(ha_ns_genes)} genes in non-singleton clusters")
print(f"    hamming+complete: {len(hc_ns_genes)} genes in non-singleton clusters")
print(f"    Overlap: {len(ha_ns_genes & hc_ns_genes)} genes")
print(f"    Only in average:  {len(ha_ns_genes - hc_ns_genes)} genes")
print(f"    Only in complete: {len(hc_ns_genes - ha_ns_genes)} genes")

# Which non-singleton clusters from average survive in complete?
print("\n  Cluster survival (average → complete):")
# For each non-singleton cluster in average, find best-matching cluster in complete
ha_clusters = {}
for cid in set(ha["labels"]):
    if cid not in ha["singletons"]:
        ha_clusters[cid] = set(g for g, l in zip(genes, ha["labels"]) if l == cid)
hc_clusters = {}
for cid in set(hc["labels"]):
    if cid not in hc["singletons"]:
        hc_clusters[cid] = set(g for g, l in zip(genes, hc["labels"]) if l == cid)

for ha_cid in sorted(ha_clusters, key=lambda c: len(ha_clusters[c]), reverse=True):
    ha_set = ha_clusters[ha_cid]
    best_cid, best_overlap = None, 0
    for hc_cid, hc_set in hc_clusters.items():
        overlap = len(ha_set & hc_set)
        if overlap > best_overlap:
            best_cid = hc_cid
            best_overlap = overlap
    if best_cid is not None:
        hc_set = hc_clusters[best_cid]
        jaccard = len(ha_set & hc_set) / len(ha_set | hc_set)
        status = (
            "EXACT"
            if jaccard == 1.0
            else "CLOSE" if jaccard > 0.7 else "SPLIT/MERGE" if jaccard > 0.3 else "LOST"
        )
        rep = list(ha_set)[:3]
        print(
            f"    avg C{ha_cid:2d} (n={len(ha_set):3d}) → "
            f"cpl C{best_cid:2d} (n={len(hc_set):3d})  "
            f"overlap={best_overlap}/{len(ha_set)}  J={jaccard:.2f}  {status}  "
            f"[{', '.join(rep)}]"
        )

print(f"\n{'='*70}")
print("  DONE")
print(f"{'='*70}")
