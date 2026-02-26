"""Bootstrap consensus analysis for hierarchical clustering stability.

Resamples the data *B* times, rebuilds the tree + decomposition each time,
and aggregates results into:

* **Co-association matrix** — fraction of replicates where two samples
  co-cluster (given both were present in the bootstrap sample).
* **Clade support values** — fraction of replicates containing each clade
  from the *original* tree.
* **Cluster stability scores** — per-cluster average co-association.

All heavy lifting (linkage, PosetTree, decompose) reuses the existing
pipeline — this module adds only the resampling loop and aggregation.

Example
-------
>>> from kl_clustering_analysis.hierarchy_analysis.bootstrap_consensus import (
...     bootstrap_consensus,
... )
>>> result = bootstrap_consensus(data, n_boot=100, random_seed=42)
>>> result["co_association_matrix"]   # pd.DataFrame (n × n)
>>> result["clade_support"]           # dict[frozenset, float]
>>> result["cluster_stability"]       # dict[int, float]
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Set

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from kl_clustering_analysis import config


def bootstrap_consensus(
    data: pd.DataFrame,
    *,
    n_boot: int = 100,
    alpha_local: float | None = None,
    sibling_alpha: float | None = None,
    metric: str | None = None,
    linkage_method: str | None = None,
    random_seed: int = 42,
    verbose: bool = False,
    decompose_kwargs: dict | None = None,
) -> Dict[str, Any]:
    """Run bootstrap consensus on a binary DataFrame.

    Parameters
    ----------
    data
        Binary sample × feature DataFrame.  Index values are sample IDs.
    n_boot
        Number of bootstrap replicates.
    alpha_local, sibling_alpha
        Significance levels forwarded to ``PosetTree.decompose()``.
        Defaults to ``config.ALPHA_LOCAL`` / ``config.SIBLING_ALPHA``.
    metric
        Distance metric for ``pdist``.  Defaults to ``config.TREE_DISTANCE_METRIC``.
    linkage_method
        Linkage method.  Defaults to ``config.TREE_LINKAGE_METHOD``.
    random_seed
        Seed for the bootstrap RNG.
    verbose
        Print progress every 10 replicates.
    decompose_kwargs
        Extra keyword arguments forwarded to ``decompose()``.

    Returns
    -------
    dict with keys
        ``co_association_matrix``
            pd.DataFrame (n × n) — pairwise co-clustering frequency.
        ``clade_support``
            dict mapping frozenset[str] → float (support ∈ [0, 1]).
        ``cluster_stability``
            dict mapping original cluster_id → mean intra-cluster co-association.
        ``original_results``
            The decomposition result dict from the *original* (non-resampled) tree.
        ``k_distribution``
            list[int] — number of clusters found in each replicate.
        ``n_boot``
            Echo of how many replicates were run.
    """
    from kl_clustering_analysis.tree.poset_tree import PosetTree  # local import to avoid cycles

    alpha_local = alpha_local or config.ALPHA_LOCAL
    sibling_alpha = sibling_alpha or config.SIBLING_ALPHA
    metric = metric or config.TREE_DISTANCE_METRIC
    linkage_method = linkage_method or config.TREE_LINKAGE_METHOD
    decompose_kwargs = decompose_kwargs or {}

    sample_ids = list(data.index)
    n = len(sample_ids)
    rng = np.random.default_rng(random_seed)

    # ---------- 0. original tree ----------
    Z_orig = linkage(pdist(data.values, metric=metric), method=linkage_method)
    tree_orig = PosetTree.from_linkage(Z_orig, leaf_names=sample_ids)
    results_orig = tree_orig.decompose(
        leaf_data=data,
        alpha_local=alpha_local,
        sibling_alpha=sibling_alpha,
        **decompose_kwargs,
    )

    # Extract original clades (sets of leaf labels) for support calculation
    original_clades = _extract_clades(tree_orig)

    # ---------- 1. bootstrap loop ----------
    # Co-association accumulators (dense — fine for n < ~5000)
    co_count = np.zeros((n, n), dtype=np.float64)  # both present
    co_same = np.zeros((n, n), dtype=np.float64)  # both present AND same cluster
    k_distribution: List[int] = []

    # Map sample_id → integer position
    id_to_idx = {sid: i for i, sid in enumerate(sample_ids)}

    # Clade hit counts
    clade_hits: Dict[FrozenSet[str], int] = {c: 0 for c in original_clades}

    for b in range(n_boot):
        if verbose and (b + 1) % 10 == 0:
            print(f"  bootstrap {b + 1}/{n_boot}")

        # --- resample rows with replacement ---
        boot_idx = rng.choice(n, size=n, replace=True)
        # Which original samples are present?
        present_set = set(boot_idx)
        # Deduplicated rows for this bootstrap (unique indices)
        unique_idx = np.array(sorted(present_set))

        X_boot = data.iloc[unique_idx]
        boot_labels = [sample_ids[i] for i in unique_idx]

        if len(unique_idx) < 4:
            # Degenerate bootstrap — skip
            continue

        # --- build tree & decompose ---
        try:
            Z_b = linkage(pdist(X_boot.values, metric=metric), method=linkage_method)
            tree_b = PosetTree.from_linkage(Z_b, leaf_names=boot_labels)
            res_b = tree_b.decompose(
                leaf_data=X_boot,
                alpha_local=alpha_local,
                sibling_alpha=sibling_alpha,
                **decompose_kwargs,
            )
        except Exception:
            # If decomposition fails on a degenerate resample, skip
            continue

        # --- accumulate co-association ---
        cluster_assignments_b = res_b.get("cluster_assignments", {})
        label_to_cluster = _build_label_map(cluster_assignments_b)

        present_list = [sample_ids[i] for i in unique_idx]
        present_idx_list = [id_to_idx[sid] for sid in present_list]

        # All pairs of present samples: co_count += 1; if same cluster co_same += 1
        for a_pos in range(len(present_list)):
            for b_pos in range(a_pos + 1, len(present_list)):
                i, j = present_idx_list[a_pos], present_idx_list[b_pos]
                co_count[i, j] += 1
                co_count[j, i] += 1
                sid_a = present_list[a_pos]
                sid_b = present_list[b_pos]
                if label_to_cluster.get(sid_a) == label_to_cluster.get(sid_b):
                    co_same[i, j] += 1
                    co_same[j, i] += 1

        k_distribution.append(res_b.get("num_clusters", 0))

        # --- clade support ---
        # A clade from the original tree is "recovered" if the subset of
        # its members present in this bootstrap sample forms a monophyletic
        # group (= appears as a clade) in the bootstrap tree.
        boot_clades = _extract_clades(tree_b)
        boot_leaf_set = set(boot_labels)
        for orig_clade in clade_hits:
            restricted = orig_clade & boot_leaf_set
            if len(restricted) < 2:
                continue  # trivial — skip
            if frozenset(restricted) in boot_clades:
                clade_hits[orig_clade] += 1

    # ---------- 2. aggregate ----------
    # Co-association matrix
    with np.errstate(divide="ignore", invalid="ignore"):
        co_assoc = np.where(co_count > 0, co_same / co_count, 0.0)
    np.fill_diagonal(co_assoc, 1.0)
    co_assoc_df = pd.DataFrame(co_assoc, index=sample_ids, columns=sample_ids)

    # Clade support
    clade_support = {c: hits / n_boot for c, hits in clade_hits.items()}

    # Cluster stability (mean intra-cluster co-association)
    cluster_stability = _compute_cluster_stability(
        results_orig.get("cluster_assignments", {}), co_assoc_df, id_to_idx
    )

    return {
        "co_association_matrix": co_assoc_df,
        "clade_support": clade_support,
        "cluster_stability": cluster_stability,
        "original_results": results_orig,
        "k_distribution": k_distribution,
        "n_boot": n_boot,
    }


# --------------- helpers ---------------


def _build_label_map(cluster_assignments: Dict) -> Dict[str, int]:
    """Map sample_label → cluster_id from decompose output."""
    label_map: Dict[str, int] = {}
    for cid, info in cluster_assignments.items():
        for leaf in info["leaves"]:
            label_map[leaf] = cid
    return label_map


def _extract_clades(tree) -> Set[FrozenSet[str]]:
    """Return the set of all clades (leaf-label sets) in *tree*.

    A clade is the set of leaf labels descended from an internal node.
    Single-leaf "clades" are excluded (trivially present in every tree).
    """
    clades: Set[FrozenSet[str]] = set()
    for node in tree.nodes():
        if tree.out_degree(node) == 0:
            continue  # leaf
        desc = _descendant_leaves(tree, node)
        if len(desc) > 1:
            clades.add(frozenset(desc))
    return clades


def _descendant_leaves(tree, node) -> List[str]:
    """Return leaf *labels* descended from *node*."""
    if tree.out_degree(node) == 0:
        return [tree.nodes[node].get("label", node)]
    leaves: List[str] = []
    for child in tree.successors(node):
        leaves.extend(_descendant_leaves(tree, child))
    return leaves


def _compute_cluster_stability(
    cluster_assignments: Dict,
    co_assoc_df: pd.DataFrame,
    id_to_idx: Dict[str, int],
) -> Dict[int, float]:
    """Mean pairwise co-association within each original cluster."""
    stability: Dict[int, float] = {}
    for cid, info in cluster_assignments.items():
        leaves = info["leaves"]
        if len(leaves) < 2:
            stability[cid] = 1.0
            continue
        vals = []
        for a_pos in range(len(leaves)):
            for b_pos in range(a_pos + 1, len(leaves)):
                sid_a, sid_b = leaves[a_pos], leaves[b_pos]
                if sid_a in co_assoc_df.index and sid_b in co_assoc_df.columns:
                    vals.append(co_assoc_df.loc[sid_a, sid_b])
        stability[cid] = float(np.mean(vals)) if vals else 0.0
    return stability
