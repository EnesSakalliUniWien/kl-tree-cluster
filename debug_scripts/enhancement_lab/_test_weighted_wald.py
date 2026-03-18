"""Quick smoke test for sibling-test methods on current HEAD.

`cousin_weighted_wald` is no longer a supported Gate 3 mode in the current
orchestrator, so this script exercises the supported methods instead.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import warnings

warnings.filterwarnings("ignore")
import traceback

from debug_scripts.enhancement_lab.lab_helpers import temporary_config
from kl_clustering_analysis import config

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.generators import generate_case_data
from kl_clustering_analysis.tree.poset_tree import PosetTree

tc = next(c for c in get_default_test_cases() if c["name"] == "gauss_moderate_5c")
data_df, y_true, _, _ = generate_case_data(tc)

old_method = config.SIBLING_TEST_METHOD
try:
    for method in ("wald", "cousin_adjusted_wald"):
        try:
            with temporary_config(SIBLING_TEST_METHOD=method):
                dist = pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC)
                Z = linkage(dist, method=config.TREE_LINKAGE_METHOD)
                tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())
                tree.populate_node_divergences(data_df)
                decomp = tree.decompose(leaf_data=data_df, alpha_local=0.05, sibling_alpha=0.05)
                print(f"{method}: K={decomp['num_clusters']}")
        except Exception:
            print(f"{method}: FAILED")
            traceback.print_exc()
finally:
    config.SIBLING_TEST_METHOD = old_method
