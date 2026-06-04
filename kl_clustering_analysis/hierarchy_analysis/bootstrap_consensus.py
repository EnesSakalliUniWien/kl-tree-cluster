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

from collections import defaultdict
from typing import Any, Dict, FrozenSet, List, Set

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_EDGE_ALPHA,
    DEFAULT_SIBLING_ALPHA,
)


def bootstrap_consensus(
    data: pd.DataFrame,
    *,
    n_boot: int = 100,
    edge_alpha: float | None = None,
    sibling_alpha: float | None = None,
    metric: str | None = None,
    linkage_method: str | None = None,
    random_seed: int = 42,
    decompose_kwargs: dict | None = None,
) -> Dict[str, Any]:
    """Run bootstrap consensus on a binary DataFrame.

    Parameters
    ----------
    data
        Binary sample × feature DataFrame.  Index values are sample IDs.
    n_boot
        Number of bootstrap replicates.
    edge_alpha, sibling_alpha
        Significance levels forwarded to ``PosetTree.decompose()``.
        Defaults to ``DEFAULT_EDGE_ALPHA`` / ``DEFAULT_SIBLING_ALPHA``.
    metric
        Distance metric for ``pdist``.  Defaults to ``config.TREE_DISTANCE_METRIC``.
    linkage_method
        Linkage method.  Defaults to ``config.TREE_LINKAGE_METHOD``.
    random_seed
        Seed for the bootstrap RNG.
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

    if edge_alpha is None:
        edge_alpha = DEFAULT_EDGE_ALPHA
    if sibling_alpha is None:
        sibling_alpha = DEFAULT_SIBLING_ALPHA
    if metric is None:
        metric = config.TREE_DISTANCE_METRIC
    if linkage_method is None:
        linkage_method = config.TREE_LINKAGE_METHOD
    if decompose_kwargs is None:
        decompose_kwargs = {}

    sample_ids = list(data.index)
    n = len(sample_ids)
    rng = np.random.default_rng(random_seed)

    # ---------- 0. original tree ----------
    Z_orig = linkage(pdist(data.values, metric=metric), method=linkage_method)
    tree_orig = PosetTree.from_linkage(Z_orig, leaf_names=sample_ids)
    tree_orig.populate_node_divergences(data)
    results_orig = tree_orig.decompose(
        annotations_df=tree_orig.annotations_df,
        leaf_data=data,
        edge_alpha=edge_alpha,
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

    for _ in range(n_boot):
        # --- resample rows with replacement ---
        # Keep multiplicities (true bootstrap weighting) by retaining all draws.
        # Use unique temporary labels so repeated originals can coexist as leaves.
        boot_idx = rng.choice(n, size=n, replace=True)
        boot_labels = [f"{sample_ids[i]}__b{j}" for j, i in enumerate(boot_idx)]
        boot_original_labels = [sample_ids[i] for i in boot_idx]
        X_boot = data.iloc[boot_idx].copy()
        X_boot.index = boot_labels

        if len(set(boot_original_labels)) < 4:
            # Degenerate bootstrap — skip
            continue

        # --- build tree & decompose ---
        Z_b = linkage(pdist(X_boot.values, metric=metric), method=linkage_method)
        tree_b = PosetTree.from_linkage(Z_b, leaf_names=boot_labels)
        tree_b.populate_node_divergences(X_boot)
        res_b = tree_b.decompose(
            annotations_df=tree_b.annotations_df,
            leaf_data=X_boot,
            edge_alpha=edge_alpha,
            sibling_alpha=sibling_alpha,
            **decompose_kwargs,
        )

        # --- accumulate co-association ---
        cluster_assignments_b = res_b["cluster_assignments"]
        label_to_cluster = _build_label_map(cluster_assignments_b)
        present_list, present_idx_list, cluster_count_matrix = _build_original_cluster_count_matrix(
            boot_labels,
            boot_original_labels,
            label_to_cluster,
            id_to_idx,
        )

        if len(present_list) > 0 and cluster_count_matrix.size > 0:
            multiplicities = cluster_count_matrix.sum(axis=1).astype(np.float64, copy=False)
            same_weights = cluster_count_matrix @ cluster_count_matrix.T
            present_weights = np.outer(multiplicities, multiplicities)

            idx = np.asarray(present_idx_list, dtype=int)
            co_count[np.ix_(idx, idx)] += present_weights
            co_same[np.ix_(idx, idx)] += same_weights.astype(np.float64, copy=False)

        k_distribution.append(res_b["num_clusters"])

        # --- clade support ---
        # A clade from the original tree is "recovered" if the subset of
        # its members present in this bootstrap sample forms a monophyletic
        # group (= appears as a clade) in the bootstrap tree.
        boot_clades = _extract_clades(tree_b)
        boot_leaf_set = set(boot_original_labels)
        temp_to_original = dict(zip(boot_labels, boot_original_labels))
        boot_clades_original: Set[FrozenSet[str]] = set()
        for boot_clade in boot_clades:
            original_members = {temp_to_original[leaf] for leaf in boot_clade}
            if len(original_members) > 1:
                boot_clades_original.add(frozenset(original_members))
        for orig_clade in clade_hits:
            restricted = orig_clade & boot_leaf_set
            if len(restricted) < 2:
                continue  # trivial — skip
            if frozenset(restricted) in boot_clades_original:
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
        results_orig["cluster_assignments"], co_assoc_df, id_to_idx
    )

    result = {
        "co_association_matrix": co_assoc_df,
        "clade_support": clade_support,
        "cluster_stability": cluster_stability,
        "original_results": results_orig,
        "k_distribution": k_distribution,
        "n_boot": n_boot,
    }

    return result


# --------------- helpers ---------------


def _build_label_map(cluster_assignments: Dict) -> Dict[str, int]:
    """Map sample_label → cluster_id from decompose output."""
    label_map: Dict[str, int] = {}
    for cluster_identifier, cluster_metadata in cluster_assignments.items():
        for leaf in cluster_metadata["leaves"]:
            label_map[leaf] = cluster_identifier
    return label_map


def _build_original_cluster_count_matrix(
    boot_labels: List[str],
    boot_original_labels: List[str],
    label_to_cluster: Dict[str, int],
    id_to_idx: Dict[str, int],
) -> tuple[List[str], List[int], np.ndarray]:
    """Build original-sample × cluster duplicate-count matrix for one bootstrap replicate."""
    original_to_counts: dict[str, defaultdict[int, int]] = {}
    for temp_label, original_label in zip(boot_labels, boot_original_labels):
        cluster_id = label_to_cluster[temp_label]
        counts = original_to_counts.setdefault(original_label, defaultdict(int))
        counts[cluster_id] += 1

    if not original_to_counts:
        return [], [], np.zeros((0, 0), dtype=np.int32)

    present_originals = sorted(original_to_counts.keys(), key=lambda sid: id_to_idx[sid])
    present_idx = [id_to_idx[sid] for sid in present_originals]
    cluster_ids = sorted(
        {cluster_id for counts in original_to_counts.values() for cluster_id in counts},
        key=repr,
    )
    cluster_to_col = {cluster_id: j for j, cluster_id in enumerate(cluster_ids)}

    matrix = np.zeros((len(present_originals), len(cluster_ids)), dtype=np.int32)
    for i, original_label in enumerate(present_originals):
        for cluster_id, count in original_to_counts[original_label].items():
            matrix[i, cluster_to_col[cluster_id]] = int(count)
    return present_originals, present_idx, matrix


def _extract_clades(tree) -> Set[FrozenSet[str]]:
    """Return the set of all clades (leaf-label sets) in *tree*.

    A clade is the set of leaf labels descended from an internal node.
    Single-leaf "clades" are excluded (trivially present in every tree).
    """
    descendant_sets = tree.compute_descendant_sets(use_labels=True)
    clades: Set[FrozenSet[str]] = set()
    for node in tree.nodes():
        if tree.out_degree(node) == 0:
            continue  # leaf
        descendants = descendant_sets[node]
        if len(descendants) > 1:
            clades.add(frozenset(descendants))
    return clades


def _compute_cluster_stability(
    cluster_assignments: Dict,
    co_assoc_df: pd.DataFrame,
    id_to_idx: Dict[str, int],
) -> Dict[int, float]:
    """Mean pairwise co-association within each original cluster."""
    stability: Dict[int, float] = {}
    for cluster_identifier, cluster_metadata in cluster_assignments.items():
        leaves = cluster_metadata["leaves"]
        if len(leaves) < 2:
            stability[cluster_identifier] = 1.0
            continue
        vals = []
        for a_pos in range(len(leaves)):
            for b_pos in range(a_pos + 1, len(leaves)):
                sid_a, sid_b = leaves[a_pos], leaves[b_pos]
                if sid_a in co_assoc_df.index and sid_b in co_assoc_df.columns:
                    vals.append(co_assoc_df.loc[sid_a, sid_b])
        stability[cluster_identifier] = float(np.mean(vals)) if vals else 0.0
    return stability
