"""Evaluate whether the current tree inference method is appropriate for HC data.

Question: Is hamming + average linkage the right tree construction for
317 genes × 1834 Reactome pathways (binary, 3.1% density)?

This script compares:
  1. Distance metrics: hamming vs jaccard vs rogerstanimoto vs cosine
  2. Linkage methods: average vs ward vs complete vs single
  3. Tree quality: cophenetic correlation, branch length distribution
  4. Downstream impact: K found by decompose() under each combination
"""

import time
import warnings

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Data ─────────────────────────────────────────────────────────────
DATA_PATH = "data/HC_feature_matrix_GO_CC.tsv"
data = pd.read_csv(DATA_PATH, sep="\t", index_col=0)
n, p = data.shape
X = data.values

print(f"Dataset: {n} genes × {p} pathways, density={X.mean():.3f}")
print()

# ── 1. Distance metric comparison ────────────────────────────────────
METRICS = ["hamming", "jaccard", "rogerstanimoto", "cosine", "dice", "sokalsneath"]

print("=" * 70)
print("  PART 1: Distance Metric Comparison")
print("=" * 70)

dists = {}
for m in METRICS:
    try:
        d = pdist(X, metric=m)
        dists[m] = d
        print(
            f"  {m:20s}  mean={d.mean():.4f}  std={d.std():.4f}  "
            f"range=[{d.min():.4f}, {d.max():.4f}]  "
            f"dyn_range={(d.max()-d.min())/d.mean():.2f}"
        )
    except Exception as e:
        print(f"  {m:20s}  FAILED: {e}")

# Feature sparsity context
row_sums = X.sum(axis=1)
print("\n  Context — pathways per gene:")
print(
    f"    mean={row_sums.mean():.1f}  median={np.median(row_sums):.0f}  "
    f"range=[{row_sums.min()}, {row_sums.max()}]"
)
print(f"    ≤5 pathways: {(row_sums <= 5).sum()} genes ({100*(row_sums <= 5).mean():.0f}%)")
print(f"    ≤2 pathways: {(row_sums <= 2).sum()} genes ({100*(row_sums <= 2).mean():.0f}%)")
print(f"    =0 pathways: {(row_sums == 0).sum()} genes")

# Near-zero distances (effectively identical genes)
print("\n  Near-identical gene pairs:")
for m in ["hamming", "jaccard", "rogerstanimoto"]:
    if m in dists:
        d = dists[m]
        print(
            f"    {m:20s}  dist<0.01: {(d < 0.01).sum():5d} pairs  "
            f"dist<0.005: {(d < 0.005).sum():4d} pairs"
        )

# Dynamic range comparison — key for tree quality
print("\n  Dynamic range analysis (higher = better tree resolution):")
for m in ["hamming", "jaccard", "rogerstanimoto", "cosine"]:
    if m in dists:
        d = dists[m]
        iqr = np.percentile(d, 75) - np.percentile(d, 25)
        cv = d.std() / d.mean()
        print(f"    {m:20s}  CV={cv:.3f}  IQR={iqr:.4f}  " f"IQR/range={iqr/(d.max()-d.min()):.3f}")

# ── 2. Linkage method × metric comparison ────────────────────────────
LINKAGES = ["average", "complete", "ward", "single", "weighted"]
# Ward requires euclidean — we'll compute euclidean distance on the binary matrix
TEST_METRICS = ["hamming", "jaccard", "rogerstanimoto"]

print(f"\n{'='*70}")
print("  PART 2: Linkage × Metric — Cophenetic Correlation")
print(f"{'='*70}")
print("  (Cophenetic r: how well the tree preserves pairwise distances)")
print("  Higher = tree is more faithful to the original distances.\n")

header = f"  {'metric':20s} {'linkage':10s}  coph_r   coph_rho"
print(header)
print("  " + "-" * len(header.strip()))

results_coph = []
for m in TEST_METRICS:
    d = dists[m]
    for lm in LINKAGES:
        try:
            if lm == "ward":
                # Ward requires euclidean; skip non-euclidean metrics
                d_euc = pdist(X, metric="euclidean")
                Z = linkage(d_euc, method="ward")
                c, coph_dist = cophenet(Z, d_euc)
            else:
                Z = linkage(d, method=lm)
                c, coph_dist = cophenet(Z, d)
            rho, _ = spearmanr(d if lm != "ward" else d_euc, coph_dist)
            print(f"  {m:20s} {lm:10s}  {c:.4f}   {rho:.4f}")
            results_coph.append(
                {"metric": m, "linkage": lm, "coph_pearson": c, "coph_spearman": rho}
            )
        except Exception as e:
            print(f"  {m:20s} {lm:10s}  FAILED: {e}")

# ── 3. Branch length analysis ────────────────────────────────────────
print(f"\n{'='*70}")
print("  PART 3: Branch Length Distribution (hamming/average vs alternatives)")
print(f"{'='*70}")
print("  Short branches everywhere → compressed tree → statistical tests " "underpowered")
print("  Bimodal branches → clear cluster boundaries\n")

for m in ["hamming", "jaccard", "rogerstanimoto"]:
    d = dists[m]
    Z = linkage(d, method="average")
    bl = Z[:, 2]  # merge heights
    diffs = np.diff(np.sort(bl))
    print(f"  {m} + average:")
    print(
        f"    heights: mean={bl.mean():.4f}  std={bl.std():.4f}  "
        f"range=[{bl.min():.6f}, {bl.max():.4f}]"
    )
    print(f"    gaps:    max_gap={diffs.max():.6f}  " f"top5_gaps={sorted(diffs)[-5:][::-1]}")
    # How many merges happen below 1% of max height?
    frac_low = (bl < 0.01 * bl.max()).mean()
    frac_mid = ((bl >= 0.01 * bl.max()) & (bl < 0.5 * bl.max())).mean()
    frac_high = (bl >= 0.5 * bl.max()).mean()
    print(
        f"    merges:  <1%max: {100*frac_low:.0f}%  "
        f"1-50%max: {100*frac_mid:.0f}%  >50%max: {100*frac_high:.0f}%"
    )
    print()

# ── 4. Downstream impact: K from decompose() ─────────────────────────
print(f"{'='*70}")
print("  PART 4: Downstream K from tree.decompose()")
print(f"{'='*70}")
print("  Testing metric × linkage combinations (this takes a few minutes)...\n")

from kl_clustering_analysis.tree.poset_tree import PosetTree

combos = [
    ("hamming", "average"),
    ("hamming", "complete"),
    ("jaccard", "average"),
    ("jaccard", "complete"),
    ("rogerstanimoto", "average"),
    ("rogerstanimoto", "complete"),
]

for m, lm in combos:
    t0 = time.time()
    try:
        d_vec = dists[m] if m in dists else pdist(X, metric=m)
        if lm == "ward":
            d_vec = pdist(X, metric="euclidean")
        Z = linkage(d_vec, method=lm)
        tree = PosetTree.from_linkage(Z, leaf_names=data.index.tolist())
        res = tree.decompose(leaf_data=data, alpha_local=0.05, sibling_alpha=0.05)
        K = res["num_clusters"]
        ca = res["cluster_assignments"]
        sizes = sorted([ca[c]["size"] for c in ca], reverse=True)
        biggest = sizes[0]
        singletons = sum(1 for s in sizes if s == 1)
        nonsingleton = len(sizes) - singletons
        elapsed = time.time() - t0
        print(
            f"  {m:20s} + {lm:10s}  K={K:3d}  "
            f"(biggest={biggest}, non-singleton={nonsingleton}, "
            f"singletons={singletons})  [{elapsed:.1f}s]"
        )
    except Exception as e:
        print(f"  {m:20s} + {lm:10s}  FAILED: {e}")

# ── 5. Recommendation ────────────────────────────────────────────────
print(f"\n{'='*70}")
print("  PART 5: Assessment")
print(f"{'='*70}")
print(
    """
  For SPARSE BINARY data (density 3.1%), the key issue is:

  HAMMING distance counts BOTH shared-1s AND shared-0s as agreement.
  With 96.9% zeros, two unrelated genes already agree on ~94% of 
  features → hamming distances are compressed into a narrow band.
  This makes the tree shallow with many short branches, and the 
  statistical gates (especially Gate 2) struggle to distinguish
  real signal from noise at such low dynamic range.

  JACCARD distance IGNORES shared-0s (double-negatives), measuring
  only |intersection| / |union| of pathway memberships.  For sparse
  binary data this is the natural choice:
  - Higher dynamic range: genes sharing many pathways → low distance,
    genes sharing none → distance 1.0
  - Tree captures POSITIVE co-annotation, not absence patterns
  - More biologically interpretable: "these genes are in the same
    pathways" rather than "these genes are absent from the same pathways"

  ROGERS-TANIMOTO is a hamming variant (double-weights disagreements)
  but still counts shared-0s, so it inherits the compression problem.

  RECOMMENDATION for this dataset:
  → Jaccard + average linkage (or Jaccard + complete)
  → Compare K and cluster composition to current hamming + average
  → The 60 singletons likely collapse into the dominant cluster or
    merge into meaningful modules under Jaccard
"""
)

print("=" * 70)
print("  DONE")
print("=" * 70)
