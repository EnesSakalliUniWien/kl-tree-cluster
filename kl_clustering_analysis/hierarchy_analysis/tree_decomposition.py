"""Tree decomposition logic for KL-based clustering.

This module contains :class:`~kl_clustering_analysis.hierarchy_analysis.tree_decomposition.TreeDecomposition`,
which traverses a hierarchy and decides where to split or merge to form clusters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterator, List, Set, Tuple

import networkx as nx

if TYPE_CHECKING:
    from ..tree.poset_tree import PosetTree

import numpy as np
import pandas as pd

from .. import config
from ..core_utils.data_utils import extract_bool_column_dict
from .cluster_assignments import build_cluster_assignments as _build_cluster_assignments_func
from .cluster_assignments import build_sample_cluster_assignments
from .posthoc_merge import apply_posthoc_merge
from .signal_localization import (
    LocalizationResult,
    extract_constrained_clusters,
    localize_divergence_signal,
    merge_difference_graphs,
    merge_similarity_graphs,
)
from .statistics import annotate_child_parent_divergence, annotate_sibling_divergence
from .statistics.branch_length_utils import (
    compute_mean_branch_length,
    sanitize_positive_branch_length,
)

# Statistical sibling test used for pairwise cluster comparisons
from .statistics.sibling_divergence.sibling_divergence_test import sibling_divergence_test


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
        localization_max_depth: int | None = None,
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
        """
        self.tree = tree
        self.results_df = results_df if results_df is not None else pd.DataFrame()
        self.alpha_local = float(alpha_local)
        self.sibling_alpha = float(sibling_alpha)

        self.posthoc_merge = bool(posthoc_merge)
        self.posthoc_merge_alpha = (
            float(posthoc_merge_alpha) if posthoc_merge_alpha is not None else self.sibling_alpha
        )

        # Signal localization parameters
        self.use_signal_localization = bool(use_signal_localization)
        self.localization_max_depth = (
            None if localization_max_depth is None else int(localization_max_depth)
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
        self._mean_branch_length = compute_mean_branch_length(self.tree)

        # ----- ensure statistical annotations are present -----
        self.results_df = self._prepare_annotations(self.results_df)

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

    # ---------- initialization helpers ----------

    def _prepare_annotations(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Ensure statistical annotation columns are present on *results_df*.

        If the required gate columns (``Child_Parent_Divergence_Significant``,
        ``Sibling_BH_Different``, ``Sibling_Divergence_Skipped``) already exist,
        the DataFrame is returned unchanged.  Otherwise the annotation pipeline
        is executed using the configured alpha levels and sibling test method.

        Parameters
        ----------
        results_df
            DataFrame indexed by node id.  May already contain the annotation
            columns (pre-annotated path) or only base KL/distribution columns
            (raw path).

        Returns
        -------
        pd.DataFrame
            The annotated DataFrame, ready for gate evaluation.
        """
        required = {
            "Child_Parent_Divergence_Significant",
            "Sibling_BH_Different",
            "Sibling_Divergence_Skipped",
        }
        if required.issubset(results_df.columns):
            return results_df

        # -- Gate 2: child-parent divergence --
        results_df = annotate_child_parent_divergence(
            self.tree,
            results_df,
            significance_level_alpha=self.alpha_local,
        )

        # -- Gate 3: sibling divergence (method selected via config) --
        if config.SIBLING_TEST_METHOD == "cousin_ftest":
            from .statistics.sibling_divergence import annotate_sibling_divergence_cousin

            results_df = annotate_sibling_divergence_cousin(
                self.tree,
                results_df,
                significance_level_alpha=self.sibling_alpha,
            )
        elif config.SIBLING_TEST_METHOD == "cousin_adjusted_wald":
            from .statistics.sibling_divergence import annotate_sibling_divergence_adjusted

            results_df = annotate_sibling_divergence_adjusted(
                self.tree,
                results_df,
                significance_level_alpha=self.sibling_alpha,
            )
        elif config.SIBLING_TEST_METHOD == "cousin_tree_guided":
            from .statistics.sibling_divergence import annotate_sibling_divergence_tree_guided

            results_df = annotate_sibling_divergence_tree_guided(
                self.tree,
                results_df,
                significance_level_alpha=self.sibling_alpha,
            )
        elif config.SIBLING_TEST_METHOD == "cousin_weighted_wald":
            from .statistics.sibling_divergence import annotate_sibling_divergence_weighted

            results_df = annotate_sibling_divergence_weighted(
                self.tree,
                results_df,
                significance_level_alpha=self.sibling_alpha,
            )
        else:
            results_df = annotate_sibling_divergence(
                self.tree,
                results_df,
                significance_level_alpha=self.sibling_alpha,
            )

        return results_df

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

    def _process_node_for_decomposition(
        self,
        node_id: str,
        nodes_to_visit: List[str],
        final_leaf_sets: List[set[str]],
    ) -> None:
        """Apply split-or-merge decision for one node during traversal.

        Parameters
        ----------
        node_id
            Node to process
        nodes_to_visit
            Pending nodes list used as a last in, first out worklist.
        final_leaf_sets
            Accumulator for cluster leaf sets
        """
        if self._should_split(node_id):
            children = self._children[node_id]
            left_child, right_child = children
            # Because this is a last in, first out worklist, pushing right then left
            # means the left child is processed first (left-first depth-first traversal).
            nodes_to_visit.append(right_child)
            nodes_to_visit.append(left_child)
        else:
            final_leaf_sets.append(self._get_all_leaves(node_id))

    def _iterate_nodes_to_visit(
        self, nodes_to_visit: List[str], processed: Set[str]
    ) -> Iterator[str]:
        """Yield nodes from a mutable last in, first out list exactly once.

        This is a small readability helper that encapsulates the common:
        pop → skip processed → mark processed → yield pattern.

        Notes
        -----
        The underlying algorithm still relies on mutating ``nodes_to_visit`` during
        traversal (e.g., pushing children when a node should split).
        """
        while nodes_to_visit:
            node_id = nodes_to_visit.pop()
            if node_id in processed:
                continue
            processed.add(node_id)
            yield node_id

    def _build_cluster_assignments(
        self, final_leaf_sets: List[set[str]]
    ) -> dict[int, dict[str, object]]:
        """Build cluster assignment dictionary from collected leaf sets.

        Delegates to :func:`.cluster_assignments.build_cluster_assignments`.
        """
        return _build_cluster_assignments_func(final_leaf_sets, self._find_cluster_root)

    def _should_split(self, parent: str) -> bool:
        """Evaluate statistical gates and return ``True`` when parent should split.

        Gates evaluated in order:

        1. **Binary structure gate** (always active)
           - OPEN: parent has exactly 2 children → proceed to statistical tests
           - CLOSED: parent has <2 or >2 children → merge at parent, form cluster boundary

        2. **Child-parent divergence gate** (child vs parent) - SIGNAL DETECTION
           - OPEN: at least one child significantly diverges from parent → proceed
           - CLOSED: neither child diverges from parent → MERGE
           - Checked FIRST to confirm there's actual signal (not just noise)
           - This matches the annotation order: child-parent test runs before sibling test

        3. **Sibling divergence gate** (sibling vs sibling) - CLUSTER SEPARATION
           - OPEN: siblings are significantly different → SPLIT
           - CLOSED: siblings are not significantly different → MERGE
           - Only checked if child-parent gate passes (consistent with annotation)

        Outcome Logic:
        - ALL gates OPEN → return True → children added to traversal stack, split continues recursively
        - ANY gate CLOSED → return False → collect all leaves under parent as single cluster, stop splitting here

        Rationale for gate order:
        - Child-parent test detects signal: "Did the children diverge from the parent?"
        - Sibling test confirms separation: "Did they diverge in different directions?"
        - This order matches annotation: sibling test is only computed when child-parent passes
        """
        # Gate 1: Binary structure requirement
        children = self._children[parent]
        if len(children) != 2:
            return False

        left_child, right_child = children

        # Gate 2: Child-parent divergence requirement (check FIRST - detects signal)
        # At least one child must significantly diverge from parent to confirm there's real signal
        left_diverges = self._local_significant.get(left_child)
        right_diverges = self._local_significant.get(right_child)

        if left_diverges is None or right_diverges is None:
            raise ValueError(
                "Missing child-parent divergence annotations for "
                f"{left_child!r} or {right_child!r}; annotate before decomposing."
            )

        # If neither child diverges from parent, it's just noise - merge
        if not (left_diverges or right_diverges):
            return False

        # Gate 3: Sibling divergence requirement (confirms cluster separation)
        # Sibling_BH_Different = True means siblings have significantly different distributions
        is_different = self._sibling_different.get(parent)

        if is_different is None:
            raise ValueError(
                "Sibling divergence annotations missing for node "
                f"{parent!r}; run annotate_sibling_divergence first."
            )

        if self._sibling_skipped.get(parent, False):
            # Cannot confirm separation — conservative: merge
            return False

        # Siblings must be significantly different to justify splitting
        return bool(is_different)

    def _test_node_pair_divergence(self, node_a: str, node_b: str) -> Tuple[float, float, float]:
        """Test divergence between two arbitrary tree nodes.

        Wrapper for sibling_divergence_test that extracts distributions
        and sample sizes from the tree.

        Parameters
        ----------
        node_a, node_b : str
            Tree node identifiers to compare.

        Returns
        -------
        Tuple[float, float, float]
            (test_statistic, degrees_of_freedom, p_value)
        """
        dist_a = self._distribution_by_node[node_a]
        dist_b = self._distribution_by_node[node_b]
        n_a = float(self._leaf_count(node_a))
        n_b = float(self._leaf_count(node_b))

        # Calculate patristic distances (sum of branch lengths) to LCA.
        # Only compute when the tree actually has branch length annotations;
        # otherwise nx.shortest_path_length defaults missing attributes to 1
        # (hop count), which is inconsistent with _mean_branch_length=None.
        dist_path_a = None
        dist_path_b = None
        if self._mean_branch_length is not None:
            lca = self.tree.find_lca(node_a, node_b)
            try:
                dist_path_a = nx.shortest_path_length(
                    self.tree, source=lca, target=node_a, weight="branch_length"
                )
                dist_path_b = nx.shortest_path_length(
                    self.tree, source=lca, target=node_b, weight="branch_length"
                )
                dist_path_a = sanitize_positive_branch_length(dist_path_a)
                dist_path_b = sanitize_positive_branch_length(dist_path_b)
            except nx.NetworkXNoPath:
                dist_path_a = None
                dist_path_b = None

        return sibling_divergence_test(
            left_dist=dist_a,
            right_dist=dist_b,
            n_left=n_a,
            n_right=n_b,
            branch_length_left=dist_path_a,
            branch_length_right=dist_path_b,
            mean_branch_length=self._mean_branch_length,
            test_id=f"nodepair:{node_a}|{node_b}",
        )

    def _check_edge_significance(self, node_id: str) -> bool:
        """Check if a node is significantly different from its parent.

        Wrapper around _local_significant dict for use as callback.
        """
        # Root has no parent edge to test, usually treated as significant/base
        if node_id == self._root:
            return True
        return bool(self._local_significant.get(node_id, False))

    def _should_split_v2(self, parent: str) -> Tuple[bool, LocalizationResult | None]:
        """Enhanced split decision with signal localization.

        Like _should_split but when siblings are "different", drills down
        to find WHERE the difference originates, enabling cross-boundary
        partial merges.

        Gates evaluated in order:

        1. **Binary structure gate** - parent must have exactly 2 children
        2. **Child-parent divergence gate** - at least one child diverges
        3. **Sibling divergence gate with localization**:
           - If siblings are "same" → MERGE (return False, None)
           - If siblings are "different" → LOCALIZE:
             - If all cross-pairs different → HARD SPLIT (return True, result)
             - If some cross-pairs similar → SOFT SPLIT (return True, result with edges)

        Parameters
        ----------
        parent : str
            The parent node to evaluate.

        Returns
        -------
        Tuple[bool, LocalizationResult | None]
            (should_split, localization_result)
            - (False, None): Don't split (merge at parent)
            - (True, None): Hard split without localization details
            - (True, LocalizationResult): Split with localization info
        """
        # Gate 1: Binary structure requirement
        children = self._children[parent]
        if len(children) != 2:
            return False, None

        left_child, right_child = children

        # Gate 2: Child-parent divergence requirement
        left_diverges = self._local_significant.get(left_child)
        right_diverges = self._local_significant.get(right_child)

        if left_diverges is None or right_diverges is None:
            raise ValueError(
                "Missing child-parent divergence annotations for "
                f"{left_child!r} or {right_child!r}; annotate before decomposing."
            )

        # If neither child diverges from parent, it's USUALLY noise - merge.
        # EXCEPTION: If the tree is very unbalanced (e.g. 1 vs 1000), the large child
        # will be statistically identical to the parent. In this case, we MUST check
        # if the siblings are different from each other.
        both_children_similar_to_parent = not (left_diverges or right_diverges)

        if both_children_similar_to_parent:
            # STRICT ENFORCEMENT of Child-Parent Gate (Equation 3.3.1 in Manuscript)
            # If children are statistically indistinguishable from parent (noise),
            # we must STOP and merge. Proceeding would split noise.
            return False, None

        # Gate 3: Sibling divergence with localization
        is_different = self._sibling_different.get(parent)

        if is_different is None:
            raise ValueError(
                "Sibling divergence annotations missing for node "
                f"{parent!r}; run annotate_sibling_divergence first."
            )

        if self._sibling_skipped.get(parent, False):
            # Cannot confirm separation — conservative: merge
            return False, None

        # If siblings are not different at aggregate level, merge
        if not is_different:
            return False, None

        # Siblings ARE different at aggregate level - now LOCALIZE the signal
        localization_result = localize_divergence_signal(
            tree=self.tree,
            left_root=left_child,
            right_root=right_child,
            test_divergence=self._test_node_pair_divergence,
            alpha=self.sibling_alpha,
            max_depth=self.localization_max_depth,
            is_edge_significant=self._check_edge_significance,
        )

        return True, localization_result

    # ---------- post-hoc merge helpers ----------

    def _compute_cluster_distribution(self, cluster_root: str) -> Tuple[np.ndarray, int]:
        """Compute the distribution for a cluster.

        Uses the precomputed distribution from the tree node.

        Parameters
        ----------
        cluster_root
            The root node of the cluster.

        Returns
        -------
        Tuple[np.ndarray, int]
            (distribution, sample_size)
        """
        distribution = self._distribution_by_node[cluster_root]
        sample_size = self._leaf_count(cluster_root)
        return distribution, sample_size

    def _test_cluster_pair_divergence(
        self, cluster_a: str, cluster_b: str, common_ancestor: str
    ) -> Tuple[float, float, float]:
        """Test if two clusters are significantly different.

        Wrapper around :func:`sibling_divergence_test`.

        Notes
        -----
        - Computes a Wald χ² statistic (optionally after JL projection) from the
          standardized difference between the two cluster mean vectors.
        - The sibling test returns ``(test_statistic, degrees_of_freedom, p_value)``.
        - The ``common_ancestor`` parameter is accepted for API symmetry but is
          not used by the current implementation.

        Parameters
        ----------
        cluster_a, cluster_b
            The two cluster root nodes to compare.
        common_ancestor
            The lowest common ancestor in the tree (present for API symmetry,
            not used by this wrapper).

        Returns
        -------
        Tuple[float, float, float]
            (test_statistic, degrees_of_freedom, p_value)
        """
        dist_a, size_a = self._compute_cluster_distribution(cluster_a)
        dist_b, size_b = self._compute_cluster_distribution(cluster_b)

        # Calculate branch lengths from common_ancestor for Felsenstein adjustment.
        # Only compute when the tree actually has branch length annotations;
        # otherwise nx.shortest_path_length defaults missing attributes to 1
        # (hop count), which is inconsistent with _mean_branch_length=None.
        branch_len_a = None
        branch_len_b = None
        if self._mean_branch_length is not None:
            try:
                branch_len_a = nx.shortest_path_length(
                    self.tree,
                    source=common_ancestor,
                    target=cluster_a,
                    weight="branch_length",
                )
                branch_len_b = nx.shortest_path_length(
                    self.tree,
                    source=common_ancestor,
                    target=cluster_b,
                    weight="branch_length",
                )
                branch_len_a = sanitize_positive_branch_length(branch_len_a)
                branch_len_b = sanitize_positive_branch_length(branch_len_b)
            except nx.NetworkXNoPath:
                branch_len_a = None
                branch_len_b = None

        test_stat, df, p_value = sibling_divergence_test(
            left_dist=dist_a,
            right_dist=dist_b,
            n_left=float(size_a),
            n_right=float(size_b),
            branch_length_left=branch_len_a,
            branch_length_right=branch_len_b,
            mean_branch_length=self._mean_branch_length,
            test_id=f"clusterpair:{cluster_a}|{cluster_b}|ancestor:{common_ancestor}",
        )

        return test_stat, df, p_value

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

        for node in self._iterate_nodes_to_visit(nodes_to_visit, processed):
            self._process_node_for_decomposition(node, nodes_to_visit, final_leaf_sets)

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
        split_points: List[Tuple[str, str, str]] = []  # (parent, left, right)
        merge_points: List[str] = []  # Nodes where we decided to merge
        processed: Set[str] = set()
        localization_results: Dict[str, LocalizationResult] = {}

        while nodes_to_visit:
            node = nodes_to_visit.pop()
            if node in processed:
                continue
            processed.add(node)

            should_split, loc_result = self._should_split_v2(node)

            if should_split:
                children = self._children[node]
                if len(children) == 2:
                    left_child, right_child = children
                    split_points.append((node, left_child, right_child))

                    if loc_result is not None:
                        localization_results[node] = loc_result

                    # Add children to traversal
                    for child in children:
                        if child not in processed:
                            nodes_to_visit.append(child)
            else:
                merge_points.append(node)

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
