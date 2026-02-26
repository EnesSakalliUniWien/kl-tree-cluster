"""Tree decomposition logic for KL-based clustering.

This module contains :class:`~kl_clustering_analysis.hierarchy_analysis.tree_decomposition.TreeDecomposition`,
which traverses a hierarchy and decides where to split or merge to form clusters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Set, Tuple

if TYPE_CHECKING:
    from ..tree.poset_tree import PosetTree

import numpy as np
import pandas as pd

from .. import config
from ..core_utils.data_utils import extract_bool_column_dict
from .cluster_assignments import build_cluster_assignments as _build_cluster_assignments_func
from .cluster_assignments import build_sample_cluster_assignments
from .gate_annotations import compute_gate_annotations
from .gates import GateEvaluator, V2TraversalState, iterate_worklist, process_node, process_node_v2
from .pairwise_testing import test_cluster_pair_divergence, test_node_pair_divergence
from .posthoc_merge import apply_posthoc_merge
from .signal_localization import (
    extract_constrained_clusters,
    merge_difference_graphs,
    merge_similarity_graphs,
)
from .statistics.branch_length_utils import compute_mean_branch_length
from .statistics.projection.random_projection import resolve_min_k
from .statistics.sibling_divergence import CalibrationModel, WeightedCalibrationModel


class TreeDecomposition:
    """Annotate a hierarchy with significance tests and carve it into clusters.

    The decomposer walks a :class:`~tree.poset_tree.PosetTree` top-down and decides
    whether to split or merge at each internal node based on three statistical gates:

    #. **Binary structure gate** - parent must have exactly two children to split.
    #. **Child-parent divergence gate** - at least one child must significantly
       diverge from the parent (projected Wald chi-square test), confirming
       there is real signal to split on.
    #. **Sibling divergence gate** - siblings must have significantly different
       distributions according to a projected Wald chi-square test with
       Benjamini-Hochberg FDR correction.  If siblings are significantly
       different, the split proceeds; otherwise the children are merged
       into a single cluster.

    Nodes that pass all gates become cluster boundaries. Leaves under the same
    boundary node are assigned the same cluster identifier. The resulting report
    captures the cluster root node, member leaves, and cluster size.
    """

    def __init__(
        self,
        tree: PosetTree,
        results_df: pd.DataFrame | None = None,
        *,
        alpha_local: float = config.ALPHA_LOCAL,
        sibling_alpha: float = config.SIBLING_ALPHA,
        posthoc_merge: bool = config.POSTHOC_MERGE,
        posthoc_merge_alpha: float | None = config.POSTHOC_MERGE_ALPHA,
        use_signal_localization: bool = config.USE_SIGNAL_LOCALIZATION,
        localization_max_depth: int | None = config.LOCALIZATION_MAX_DEPTH,
        localization_max_pairs: int | None = config.LOCALIZATION_MAX_PAIRS,
        leaf_data: pd.DataFrame | None = None,
        spectral_method: str | None = config.SPECTRAL_METHOD,
    ):
        """Configure decomposition thresholds and pre-compute reusable metadata.

        Parameters
        ----------
        tree
            Directed hierarchy (typically a :class:`~tree.poset_tree.PosetTree`).
        results_df
            DataFrame of statistical annotations (e.g., columns produced by
            ``hierarchy_analysis.statistics`` helpers). May be ``None`` if the caller
            plans to rely on on-the-fly calculations.
        alpha_local
            Significance level used when the local Kullback-Leibler divergence gate
            falls back to raw chi-square tests.
        sibling_alpha
            Significance level used by sibling-independence annotations and gating.
        posthoc_merge
            If True, apply tree-respecting post-hoc merging after initial decomposition.
            This iteratively merges clusters whose underlying leaf clusters are NOT
            significantly different, working bottom-up through the tree.
        posthoc_merge_alpha
            Significance level for post-hoc merge tests. Defaults to sibling_alpha if None.
        use_signal_localization
            If True, use signal localization (v2) to find WHERE divergence originates,
            enabling cross-boundary partial merges for soft cluster boundaries.
        localization_max_depth
            Maximum recursion depth for signal localization.
        localization_max_pairs
            Maximum number of terminal cross-boundary pairs to collect per
            localization call.  Once reached, remaining stack items are
            recorded at their current granularity without further drilling.
        leaf_data
            Raw binary data matrix (samples × features).  Required for per-node
            spectral dimension estimation.  When ``None``, the legacy JL-based
            projection dimension is used regardless of *spectral_method*.
        spectral_method
            Per-node projection dimension estimator.  See ``config.SPECTRAL_METHOD``.
        """
        self.tree = tree
        self.results_df = results_df if results_df is not None else pd.DataFrame()
        self.alpha_local = float(alpha_local)
        self.sibling_alpha = float(sibling_alpha)
        self._leaf_data = leaf_data
        self._spectral_method = spectral_method if leaf_data is not None else None

        # --- Resolve adaptive projection floor ---
        # When config.PROJECTION_MIN_K == "auto", compute the data-driven
        # minimum from the effective rank of the full dataset.  The resolved
        # integer is stored and passed through to all annotation / test calls
        # so that the fixed floor never overrides the data's actual rank.
        self._resolved_min_k: int = resolve_min_k(config.PROJECTION_MIN_K, leaf_data)

        self.posthoc_merge = bool(posthoc_merge)
        self.posthoc_merge_alpha = (
            float(posthoc_merge_alpha) if posthoc_merge_alpha is not None else self.sibling_alpha
        )

        # Signal localization parameters
        self.use_signal_localization = bool(use_signal_localization)
        self.localization_max_depth = (
            None if localization_max_depth is None else int(localization_max_depth)
        )
        self.localization_max_pairs = (
            None if localization_max_pairs is None else int(localization_max_pairs)
        )

        # ----- root -----
        self._root = next(node_id for node_id, degree in self.tree.in_degree() if degree == 0)

        # ----- pre-cache node metadata -----
        self._cache_node_metadata()

        # ----- leaf partitions & counts (poset view) -----
        self._leaf_partition_by_node = self.tree.compute_descendant_sets(use_labels=True)

        self._leaf_count_cache: Dict[str, int] = {
            node_id: self.tree.nodes[node_id].get(
                "leaf_count", len(self._leaf_partition_by_node.get(node_id, ()))
            )
            for node_id in self.tree.nodes
        }

        # ----- mean branch length for Felsenstein normalization -----
        # Only consider edges that actually carry a branch_length attribute.
        # Edges without the attribute are NOT treated as zero-length — they
        # are simply absent.  If no edge has the attribute the Felsenstein
        # adjustment is disabled entirely (_mean_branch_length = None).
        #
        # config.FELSENSTEIN_SCALING gates the entire mechanism: when False
        # (default since 2026-02-17) the mean is forced to None so no
        # variance inflation is applied anywhere.
        if config.FELSENSTEIN_SCALING:
            self._mean_branch_length = compute_mean_branch_length(self.tree)
        else:
            self._mean_branch_length = None

        # ----- ensure statistical annotations are present -----
        self.results_df = self._prepare_annotations(self.results_df)

        # ----- extract calibration model for post-hoc merge symmetry -----
        # When using cousin-adjusted (or weighted) Wald, the annotation step
        # stores a CalibrationModel / WeightedCalibrationModel that deflates
        # raw Wald stats by the estimated post-selection inflation factor ĉ.
        # We reuse the same model in _test_cluster_pair_divergence so that the
        # post-hoc merge test has identical calibration to the decomposition —
        # otherwise the merge uses inflated raw T while the split used
        # deflated T_adj, making it systematically harder to merge than to split.
        self._calibration_model: CalibrationModel | WeightedCalibrationModel | None = (
            self.results_df.attrs.get("_calibration_model")
        )

        # ----- results_df → fast dictionary lookups (no .loc in hot paths) -----
        self._local_significant = extract_bool_column_dict(
            self.results_df, "Child_Parent_Divergence_Significant"
        )
        # Sibling divergence test: Sibling_BH_Different = True means siblings differ -> SPLIT
        self._sibling_different = extract_bool_column_dict(self.results_df, "Sibling_BH_Different")
        self._sibling_skipped = extract_bool_column_dict(
            self.results_df, "Sibling_Divergence_Skipped"
        )

        for column_name, mapping in (
            ("Child_Parent_Divergence_Significant", self._local_significant),
            ("Sibling_BH_Different", self._sibling_different),
            ("Sibling_Divergence_Skipped", self._sibling_skipped),
        ):
            missing = set(self.tree.nodes) - set(mapping.keys())
            if missing:
                preview = ", ".join(map(repr, list(missing)[:5]))
                raise ValueError(f"Missing {column_name!r} values for nodes: {preview}.")

        # Precompute children list (avoids rebuilding generator repeatedly)
        self._children: Dict[str, List[str]] = {
            n: list(self.tree.successors(n)) for n in self.tree.nodes
        }

        # ----- construct the GateEvaluator -----
        self._gate = GateEvaluator(
            tree=self.tree,
            local_significant=self._local_significant,
            sibling_different=self._sibling_different,
            sibling_skipped=self._sibling_skipped,
            children_map=self._children,
            descendant_leaf_sets=self._leaf_partition_by_node,
            root=self._root,
        )

    # ---------- initialization helpers ----------

    def _prepare_annotations(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Ensure statistical annotation columns are present on *results_df*.

        Delegates to :func:`~.gate_annotations.compute_gate_annotations`.
        """
        return compute_gate_annotations(
            self.tree,
            results_df,
            alpha_local=self.alpha_local,
            sibling_alpha=self.sibling_alpha,
            leaf_data=self._leaf_data,
            spectral_method=self._spectral_method,
            min_k=self._resolved_min_k,
        )

    def _cache_node_metadata(self) -> None:
        """Cache node attributes for fast repeated access during decomposition.

        Extracts and stores distributions, leaf flags, and labels.
        This one-time preprocessing avoids expensive NetworkX lookups in tight loops.
        """
        self._distribution_by_node: Dict[str, np.ndarray] = {}
        self._is_leaf: Dict[str, bool] = {}
        self._label: Dict[str, str] = {}

        for node_id in self.tree.nodes:
            node_data = self.tree.nodes[node_id]
            # Preserve shape: 1D for binary (Bernoulli), 2D for categorical (multinomial)
            distribution_array = np.asarray(node_data["distribution"], dtype=float)
            self._distribution_by_node[node_id] = distribution_array
            self._is_leaf[node_id] = node_data["is_leaf"]
            self._label[node_id] = node_data.get("label", node_id)

        # n_features is the first dimension (number of features/variables)
        if self._distribution_by_node:
            first_dist = next(iter(self._distribution_by_node.values()))
            self._n_features = first_dist.shape[0] if first_dist.ndim >= 1 else 0
        else:
            self._n_features = 0

    # ---------- utilities ----------

    def _get_all_leaves(self, node_id: str) -> set[str]:
        """Return the leaf partition beneath a node using precomputed cache.

        Parameters
        ----------
        node_id
            The node whose leaf partition to retrieve

        Returns
        -------
        set
            Set of leaf labels in the node's partition
        """
        return set(self._leaf_partition_by_node[node_id])

    # ---------- LCA ----------

    def _find_cluster_root(self, leaf_labels: Set[str]) -> str:
        """Identify the lowest common ancestor for a collection of leaf labels.

        This implementation delegates the LCA calculation to the tree object, which is
        expected to have an efficient implementation, and caches a label-to-node mapping
        to accelerate leaf lookups.
        """
        if not hasattr(self, "_label_to_node_map"):
            # This is a one-time cache population for all leaves.
            self._label_to_node_map = {
                self._label[n]: n for n in self.tree.nodes if self._is_leaf.get(n)
            }

        leaf_nodes = [
            self._label_to_node_map[label]
            for label in leaf_labels
            if label in self._label_to_node_map
        ]

        if not leaf_nodes:
            return self._root

        # The `self.tree` object is a PosetTree, which has `find_lca_for_set`.
        return self.tree.find_lca_for_set(leaf_nodes)

    # ---------- local KL (child vs parent) ----------

    def _leaf_count(self, node_id: str) -> int:
        return self._leaf_count_cache[node_id]

    def _child_diverges_from_parent(self, child: str, parent: str) -> bool:
        """Determine whether the local divergence test flags ``child`` as divergent.

        Relies on precomputed ``Child_Parent_Divergence_Significant`` annotations from
        :func:`~kl_clustering_analysis.hierarchy_analysis.statistics.kl_tests.edge_significance.annotate_child_parent_divergence`.
        No fallback computation is performed; missing annotations raise to signal
        a misconfigured pipeline.
        """
        annotated = self._local_significant.get(child)
        if annotated is None:
            raise ValueError(
                "Local divergence annotations missing for node "
                f"{child!r}; run annotate_child_parent_divergence first."
            )
        return bool(annotated)

    # ---------- core decomposition (iterative, no recursion) ----------

    def _build_cluster_assignments(
        self, final_leaf_sets: List[set[str]]
    ) -> dict[int, dict[str, object]]:
        """Build cluster assignment dictionary from collected leaf sets.

        Delegates to :func:`.cluster_assignments.build_cluster_assignments`.
        """
        return _build_cluster_assignments_func(final_leaf_sets, self._find_cluster_root)

    def _test_node_pair_divergence(self, node_a: str, node_b: str) -> Tuple[float, float, float]:
        """Test divergence between two arbitrary tree nodes.

        Delegates to :func:`~.pairwise_testing.test_node_pair_divergence`.
        Uses the same calibration model as the Gate 3 sibling test so
        that signal-localization sub-tests are on the same statistical
        scale as the decomposition.
        """
        return test_node_pair_divergence(
            self.tree,
            node_a,
            node_b,
            self._mean_branch_length,
            calibration_model=self._calibration_model,
        )

    # ---------- post-hoc merge helpers ----------

    def _test_cluster_pair_divergence(
        self, cluster_a: str, cluster_b: str, common_ancestor: str
    ) -> Tuple[float, float, float]:
        """Test if two clusters are significantly different.

        Delegates to :func:`~.pairwise_testing.test_cluster_pair_divergence`.
        """
        return test_cluster_pair_divergence(
            self.tree,
            cluster_a,
            cluster_b,
            common_ancestor,
            self._mean_branch_length,
            calibration_model=self._calibration_model,
        )

    def decompose_tree(self) -> dict[str, object]:
        """Return cluster assignments by iteratively traversing the hierarchy.

        Traversal order
        ---------------
        The traversal uses a last in, first out list (similar to an explicit stack),
        which produces a depth-first traversal order. When a node is split, its
        two children are appended in right-then-left order so that the left child
        is processed first on the next iteration.
        """
        nodes_to_visit: List[str] = [self._root]
        final_leaf_sets: List[set[str]] = []
        processed: Set[str] = set()

        for node in iterate_worklist(nodes_to_visit, processed):
            process_node(node, self._gate, nodes_to_visit, final_leaf_sets)

        cluster_assignments = self._build_cluster_assignments(final_leaf_sets)

        # Apply post-hoc merge and capture audit trail
        cluster_assignments, merge_audit = self._maybe_apply_posthoc_merge_with_audit(
            cluster_assignments
        )

        return {
            "cluster_assignments": cluster_assignments,
            "num_clusters": len(cluster_assignments),
            "posthoc_merge_audit": merge_audit,
            "independence_analysis": {
                "alpha_local": self.alpha_local,
                "decision_mode": "sibling_divergence",
                "posthoc_merge": self.posthoc_merge,
            },
        }

    def decompose_tree_v2(self) -> dict[str, object]:
        """Return cluster assignments using signal localization for soft boundaries.

        This enhanced version drills down when siblings are "different" to find
        WHERE the divergence originates. This enables cross-boundary partial merges
        when some sub-clusters are similar despite being on opposite sides of a split.

        Algorithm
        ---------
        1. Standard top-down traversal collecting split points
        2. For each split point, run signal localization to find soft boundaries
        3. Build combined similarity graph from all split points
        4. Extract final clusters from connected components

        Returns
        -------
        dict[str, object]
            Contains:
            - cluster_assignments: Mapping from cluster ID to cluster info
            - num_clusters: Number of clusters
            - localization_results: Dict of LocalizationResult per split point
            - similarity_graph: Combined similarity graph (edges = similar pairs)
            - independence_analysis: Configuration metadata
        """
        # Phase 1: Standard traversal to collect split points
        nodes_to_visit: List[str] = [self._root]
        processed: Set[str] = set()
        state = V2TraversalState(
            split_points=[],
            merge_points=[],
            localization_results={},
        )

        for node in iterate_worklist(nodes_to_visit, processed):
            process_node_v2(
                node,
                self._gate,
                nodes_to_visit,
                state,
                test_divergence=self._test_node_pair_divergence,
                sibling_alpha=self.sibling_alpha,
                localization_max_depth=self.localization_max_depth,
                localization_max_pairs=self.localization_max_pairs,
            )

        split_points = state.split_points
        merge_points = state.merge_points
        localization_results = state.localization_results

        # Phase 2: Build combined similarity and difference graphs
        combined_similarity = merge_similarity_graphs(localization_results)
        combined_difference = merge_difference_graphs(localization_results)

        # Phase 3: Extract soft clusters with constraints
        if combined_similarity.number_of_edges() > 0:
            # We have soft boundaries - extract clusters using constrained merge
            soft_clusters = extract_constrained_clusters(
                similarity_graph=combined_similarity,
                difference_graph=combined_difference,
                tree=self.tree,
                merge_points=merge_points,
            )
        else:
            # No soft boundaries - use standard hard cluster extraction
            soft_clusters = [self._get_all_leaves(node) for node in merge_points]

        # Build final cluster assignments
        cluster_assignments = self._build_cluster_assignments(soft_clusters)

        # Apply post-hoc merge and capture audit trail
        cluster_assignments, merge_audit = self._maybe_apply_posthoc_merge_with_audit(
            cluster_assignments
        )

        return {
            "cluster_assignments": cluster_assignments,
            "num_clusters": len(cluster_assignments),
            "localization_results": localization_results,
            "similarity_graph": combined_similarity,
            "difference_graph": combined_difference,
            "split_points": split_points,
            "posthoc_merge_audit": merge_audit,
            "independence_analysis": {
                "alpha_local": self.alpha_local,
                "decision_mode": "sibling_divergence_v2_localized",
                "posthoc_merge": self.posthoc_merge,
                "use_signal_localization": True,
                "localization_max_depth": self.localization_max_depth,
                "localization_max_pairs": self.localization_max_pairs,
            },
        }

    def _maybe_apply_posthoc_merge_with_audit(
        self, cluster_assignments: dict[int, dict[str, object]]
    ) -> Tuple[dict[int, dict[str, object]], List[Dict]]:
        """Optionally apply tree-respecting post-hoc merge and return audit trail.

        Parameters
        ----------
        cluster_assignments
            Mapping from cluster id to cluster metadata, including ``root_node``.

        Returns
        -------
        Tuple[dict[int, dict[str, object]], List[Dict]]
            - Updated cluster assignments after optional post-hoc merging.
            - Audit trail of merges tested.
        """
        if not (self.posthoc_merge and self._distribution_by_node):
            return cluster_assignments, []

        cluster_roots: Set[str] = set()
        for info in cluster_assignments.values():
            root_node = info.get("root_node")
            if isinstance(root_node, str) and root_node:
                cluster_roots.add(root_node)

        if not cluster_roots:
            return cluster_assignments, []

        merged_roots, audit_trail = apply_posthoc_merge(
            cluster_roots=cluster_roots,
            alpha=self.posthoc_merge_alpha,
            tree=self.tree,
            children=self._children,
            root=self._root,
            test_divergence=self._test_cluster_pair_divergence,
        )

        # Deterministic cluster ids: avoid iterating an unordered set.
        merged_leaf_sets: List[set[str]] = []
        for root_node in sorted(merged_roots):
            leaf_labels = self._get_all_leaves(root_node)
            if leaf_labels:
                merged_leaf_sets.append(set(leaf_labels))

        return self._build_cluster_assignments(merged_leaf_sets), audit_trail

    # build_sample_cluster_assignments is the module-level pure function
    # re-exported as a static method for backward compatibility.
    build_sample_cluster_assignments = staticmethod(build_sample_cluster_assignments)
