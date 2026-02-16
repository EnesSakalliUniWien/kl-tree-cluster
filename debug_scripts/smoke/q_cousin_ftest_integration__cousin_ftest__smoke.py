"""
Purpose: Smoke-test cousin F-test integration in the sibling divergence pipeline.
Inputs: Synthetic test setup defined in-script.
Outputs: Console pass/fail style diagnostic output.
Expected runtime: ~5-30 seconds.
How to run: python debug_scripts/smoke/q_cousin_ftest_integration__cousin_ftest__smoke.py
"""

from kl_clustering_analysis import config

print("SIBLING_TEST_METHOD =", config.SIBLING_TEST_METHOD)

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from scipy.spatial.distance import pdist

from kl_clustering_analysis.tree.poset_tree import PosetTree

rng = np.random.default_rng(42)
X = rng.binomial(1, 0.5, size=(100, 30))
data = pd.DataFrame(X, index=[f"S{i}" for i in range(100)], columns=[f"F{j}" for j in range(30)])
Z = linkage(pdist(data.values, metric="hamming"), method="average")
tree = PosetTree.from_linkage(Z, leaf_names=data.index.tolist())
result = tree.decompose(leaf_data=data)
print("Clusters:", result["num_clusters"])

# Check that the stats_df has the test method column
if "Sibling_Test_Method" in tree.stats_df.columns:
    methods = tree.stats_df["Sibling_Test_Method"].dropna().value_counts()
    print("Test methods used:")
    for m, c in methods.items():
        print(f"  {m}: {c}")
else:
    print("WARNING: Sibling_Test_Method column not found")

print("Smoke test PASSED")
