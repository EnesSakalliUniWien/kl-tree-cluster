#!/usr/bin/env python3
"""Check whether median binarization destroys cluster signal."""
import sys

sys.path.insert(0, "/Users/berksakalli/Projects/kl-te-cluster")
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

X, y = make_blobs(n_samples=30, n_features=30, centers=3, cluster_std=0.5, random_state=100)
X_bin = (X > np.median(X, axis=0)).astype(int)

Z_cont = linkage(pdist(X, "euclidean"), "average")
Z_bin = linkage(pdist(X_bin, "hamming"), "average")

y_cont = fcluster(Z_cont, t=3, criterion="maxclust")
y_bin = fcluster(Z_bin, t=3, criterion="maxclust")

print(f"ARI continuous linkage 3-cut: {adjusted_rand_score(y, y_cont):.3f}")
print(f"ARI binarized linkage 3-cut:  {adjusted_rand_score(y, y_bin):.3f}")
print(
    f"Column means: min={X_bin.mean(0).min():.2f} max={X_bin.mean(0).max():.2f} mean={X_bin.mean(0).mean():.2f}"
)
print(f"Unique binary rows: {len(np.unique(X_bin, axis=0))}")
print()
for c in range(3):
    mask = y == c
    if mask.sum() > 1:
        intra = pdist(X_bin[mask], "hamming")
        print(f"Cluster {c} (n={mask.sum()}): intra-hamming mean={intra.mean():.3f}")
print(f"All pairs: hamming mean={pdist(X_bin, 'hamming').mean():.3f}")
print(f"All pairs: hamming mean={pdist(X_bin, 'hamming').mean():.3f}")
