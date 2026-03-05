"""Tree decomposition logic for KL-based clustering.

This module contains :class:`~kl_clustering_analysis.hierarchy_analysis.tree_decomposition.TreeDecomposition`,
which traverses a hierarchy and decides where to split or merge to form clusters.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Dict, List, Set, Tuple

if TYPE_CHECKING:
    from ..tree.poset_tree import PosetTree

import numpy as np
import pandas as pd

from .. import config
from ..core_utils.data_utils import extract_bool_column_dict
from .cluster_assignments import build_cluster_assignments as _build_cluster_assignments_func
from .cluster_assignments import build_sample_cluster_assignments
from .decomposition.backends.random_projection_backend import resolve_minimum_projection_dimension_backend
from .decomposition.gates.orchestrator import run_gate_annotation_pipeline
from .decomposition.gates.pairwise_testing import (
    build_branch_distance_cache,
    test_cluster_pair_divergence,
)
from .decomposition.gates.posthoc_merge import apply_posthoc_merge
from .decomposition.gates.traversal import GateEvaluator, iterate_worklist, process_node
from .statistics.branch_length_utils import (
    compute_mean_branch_length,
    sanitize_positive_branch_length,
)
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
        annotations_df: pd.DataFrame | None = None,
        *,
        alpha_local: float = config.EDGE_ALPHA,
        sibling_alpha: float = config.SIBLING_ALPHA,
        posthoc_merge: bool = config.POSTHOC_MERGE,
        posthoc_merge_alpha: float | None = config.POSTHOC_MERGE_ALPHA,
        leaf_data: pd.DataFrame | None = None,
        spectral_method: str | None = config.SPECTRAL_METHOD,
        passthrough: bool = config.PASSTHROUGH,
    ):
        """Configure decomposition thresholds and pre-compute reusable metadata.

        Parameters
        ----------
        tree
            Directed hierarchy (typically a :class:`~tree.poset_tree.PosetTree`).
        annotations_df
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
        leaf_data
            Raw binary data matrix (samples × features).  Required for per-node
            spectral dimension estimation.  When ``None``, the legacy JL-based
            projection dimension is used regardless of *spectral_method*.
        spectral_method
            Per-node projection dimension estimator.  See ``config.SPECTRAL_METHOD``.
        """
        self.tree = tree
        self.annotations_df = annotations_df if annotations_df is not None else pd.DataFrame()
        self.alpha_local = float(alpha_local)
        self.sibling_alpha = float(sibling_alpha)
        self._leaf_data = leaf_data
        self._spectral_method = spectral_method if leaf_data is not None else None

        # --- Resolve adaptive projection floor ---
        # When config.PROJECTION_MINIMUM_DIMENSION == "auto", compute the data-driven
        # minimum from the effective rank of the full dataset.  The resolved
        # integer is stored and passed through to all annotation / test calls
        # so that the fixed floor never overrides the data's actual rank.
        self._resolved_minimum_projection_dimension: int = resolve_minimum_projection_dimension_backend(
            config.PROJECTION_MINIMUM_DIMENSION,
            leaf_data=leaf_data,
        )

        self.posthoc_merge = bool(posthoc_merge)
        self.posthoc_merge_alpha = (
            float(posthoc_merge_alpha) if posthoc_merge_alpha is not None else self.sibling_alpha
        )

        # ----- root -----
        self._root = next(node_id for node_id, degree in self.tree.in_degree() if degree == 0)

        # ----- pre-cache node metadata -----
        self._cache_node_metadata()

        # ----- leaf partitions & counts (poset view) -----
        self._descendant_leaf_sets = self.tree.compute_descendant_sets(use_labels=True)

        self._leaf_count_cache: Dict[str, int] = {
            node_id: node_data.get("leaf_count", len(self._descendant_leaf_sets.get(node_id, ())))
            for node_id, node_data in self._node_attrs_by_id.items()
        }

        # ----- branch-length semantics and Felsenstein normalization -----
        self._mean_branch_length, branch_distance_mode = self._resolve_branch_length_scaling()
        self._branch_distance_cache = (
            build_branch_distance_cache(
                self.tree,
                root=self._root,
                distance_mode=branch_distance_mode,
            )
            if self._mean_branch_length is not None
            else None
        )

        # ----- ensure statistical annotations are present -----
        self.annotations_df = self._prepare_annotations(self.annotations_df)

        # ----- extract calibration model for post-hoc merge symmetry -----
        # When using cousin-adjusted (or weighted) Wald, the annotation step
        # stores a CalibrationModel / WeightedCalibrationModel that deflates
        # raw Wald stats by the estimated post-selection inflation factor ĉ.
        # We reuse the same model in _test_cluster_pair_divergence so that the
        # post-hoc merge test has identical calibration to the decomposition —
        # otherwise the merge uses inflated raw T while the split used
        # deflated T_adj, making it systematically harder to merge than to split.
        self._calibration_model: CalibrationModel | WeightedCalibrationModel | None = (
            self.annotations_df.attrs.get("_calibration_model")
        )

        # ----- annotations_df → fast dictionary lookups (no .loc in hot paths) -----
        self._local_significant = extract_bool_column_dict(
            self.annotations_df, "Child_Parent_Divergence_Significant"
        )
        # Sibling divergence test: Sibling_BH_Different = True means siblings differ -> SPLIT
        self._sibling_different = extract_bool_column_dict(
            self.annotations_df, "Sibling_BH_Different"
        )
        self._sibling_skipped = extract_bool_column_dict(
            self.annotations_df, "Sibling_Divergence_Skipped"
        )

        for column_name, mapping in (
            ("Child_Parent_Divergence_Significant", self._local_significant),
            ("Sibling_BH_Different", self._sibling_different),
            ("Sibling_Divergence_Skipped", self._sibling_skipped),
        ):
            missing = set(self._node_ids) - set(mapping.keys())
            if missing:
                preview = ", ".join(map(repr, list(missing)[:5]))
                raise ValueError(f"Missing {column_name!r} values for nodes: {preview}.")

        # Precompute children list (avoids rebuilding generator repeatedly)
        self._children: Dict[str, List[str]] = {
            n: list(self.tree.successors(n)) for n in self._node_ids
        }

        # ----- precompute has_descendant_split (bottom-up O(n)) -----
        self._passthrough = bool(passthrough)
        self._has_descendant_split: Dict[str, bool] = {}
        if self._passthrough:
            self._has_descendant_split = self._compute_has_descendant_split()

        # ----- construct the GateEvaluator -----
        self._gate = GateEvaluator(
            tree=self.tree,
            local_significant=self._local_significant,
            sibling_different=self._sibling_different,
            sibling_skipped=self._sibling_skipped,
            children_map=self._children,
            descendant_leaf_sets=self._descendant_leaf_sets,
            has_descendant_split=self._has_descendant_split,
            passthrough=self._passthrough,
        )

    # ---------- initialization helpers ----------

    def _resolve_branch_length_scaling(self) -> Tuple[float | None, str]:
        """Resolve branch-length mode and mean normalization for this run.

        Returns
        -------
        Tuple[float | None, str]
            ``(mean_branch_length, distance_mode)`` where *distance_mode* is
            passed to :func:`build_branch_distance_cache`.
        """
        if not config.FELSENSTEIN_SCALING:
            return None, "phylogeny"

        mode = config.FELSENSTEIN_BRANCH_LENGTH_MODE
        if mode not in {"phylogeny", "topology"}:
            raise ValueError(
                "config.FELSENSTEIN_BRANCH_LENGTH_MODE must be 'phylogeny' "
                f"or 'topology'. Got {mode!r}."
            )

        if mode == "topology":
            # Unit edge lengths by construction, so normalization mean is 1.
            return 1.0, "topology"

        policy = config.FELSENSTEIN_INCOMPLETE_BRANCH_POLICY
        if policy not in {"warn_disable", "error"}:
            raise ValueError(
                "config.FELSENSTEIN_INCOMPLETE_BRANCH_POLICY must be "
                f"'warn_disable' or 'error'. Got {policy!r}."
            )

        invalid_edges: List[Tuple[str, str, object]] = []
        for parent, child in self.tree.edges():
            raw_branch_length = self.tree.edges[parent, child].get("branch_length")
            if sanitize_positive_branch_length(raw_branch_length) is None:
                invalid_edges.append((parent, child, raw_branch_length))

        if invalid_edges:
            preview = ", ".join(
                f"{parent}->{child}:{raw!r}" for parent, child, raw in invalid_edges[:3]
            )
            message = (
                "FELSENSTEIN_SCALING is enabled with phylogeny branch-length mode, "
                f"but {len(invalid_edges)} of {self.tree.number_of_edges()} edges have "
                "missing/invalid 'branch_length'. "
                f"Examples: {preview}."
            )
            if policy == "error":
                raise ValueError(message)
            warnings.warn(
                message + " Disabling Felsenstein scaling for this decomposition run.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None, "phylogeny"

        mean_branch_length = compute_mean_branch_length(self.tree)
        if mean_branch_length is None:
            message = (
                "FELSENSTEIN_SCALING is enabled with phylogeny branch-length mode, "
                "but no valid positive branch lengths were found."
            )
            if policy == "error":
                raise ValueError(message)
            warnings.warn(
                message + " Disabling Felsenstein scaling for this decomposition run.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None, "phylogeny"

        return mean_branch_length, "phylogeny"

    def _compute_has_descendant_split(self) -> Dict[str, bool]:
        """Precompute bottom-up flag: does any descendant have a significant sibling split?

        For each internal node, ``has_descendant_split[node]`` is ``True``
        when at least one descendant (not the node itself) has
        ``Sibling_BH_Different == True`` and was not skipped.

        Computed in a single reverse-topological pass (O(n)):

        - Leaves → ``False``
        - Internal node → ``True`` if any child is itself a split node or
          has ``has_descendant_split == True``.

        Returns
        -------
        Dict[str, bool]
            Mapping from node ID to boolean flag.
        """
        import networkx as nx

        has_split: Dict[str, bool] = {}

        # Reverse topological order = leaves first, root last
        for node in reversed(list(nx.topological_sort(self.tree))):
            children = self._children.get(node, [])
            if not children:
                # Leaf node
                has_split[node] = False
                continue

            # Check if any child is itself a split point or has a descendant split
            found = False
            for child in children:
                # Child itself is a significant split point
                child_is_split = self._sibling_different.get(
                    child, False
                ) and not self._sibling_skipped.get(child, True)
                if child_is_split or has_split.get(child, False):
                    found = True
                    break

            has_split[node] = found

        return has_split

    def _prepare_annotations(self, annotations_df: pd.DataFrame) -> pd.DataFrame:
        """Ensure statistical annotation columns are present on *annotations_df*.

        Delegates directly to canonical gate orchestrator.
        """
        return run_gate_annotation_pipeline(
            self.tree,
            annotations_df,
            alpha_local=self.alpha_local,
            sibling_alpha=self.sibling_alpha,
            leaf_data=self._leaf_data,
            spectral_method=self._spectral_method,
            minimum_projection_dimension=self._resolved_minimum_projection_dimension,
            sibling_method=config.SIBLING_TEST_METHOD,
            # Preserve existing decomposition semantics.
            fdr_method="tree_bh",
            sibling_spectral_dims=None,
            sibling_pca_projections=None,
            sibling_pca_eigenvalues=None,
            edge_calibration=None,
        ).annotated_df

    def _cache_node_metadata(self) -> None:
        """Cache node attributes for fast repeated access during decomposition.

        Extracts and stores distributions, leaf flags, and labels.
        This one-time preprocessing avoids expensive NetworkX lookups in tight loops.
        """
        self._node_ids: Tuple[str, ...] = tuple()
        self._node_attrs_by_id: Dict[str, dict] = {}
        self._distribution_by_node: Dict[str, np.ndarray] = {}
        self._is_leaf: Dict[str, bool] = {}
        self._label: Dict[str, str] = {}

        node_ids: List[str] = []
        for node_id, node_data in self.tree.nodes(data=True):
            node_ids.append(node_id)
            self._node_attrs_by_id[node_id] = node_data
            # Preserve shape: 1D for binary (Bernoulli), 2D for categorical (multinomial)
            distribution_array = np.asarray(node_data["distribution"], dtype=float)
            self._distribution_by_node[node_id] = distribution_array
            self._is_leaf[node_id] = node_data["is_leaf"]
            self._label[node_id] = node_data.get("label", node_id)
        self._node_ids = tuple(node_ids)

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
        return set(self._descendant_leaf_sets[node_id])

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
                self._label[n]: n for n in self._node_ids if self._is_leaf.get(n)
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

    # ---------- core decomposition (iterative, no recursion) ----------

    def _build_cluster_assignments(
        self, final_leaf_sets: List[set[str]]
    ) -> dict[int, dict[str, object]]:
        """Build cluster assignment dictionary from collected leaf sets.

        Delegates to :func:`.cluster_assignments.build_cluster_assignments`.
        """
        return _build_cluster_assignments_func(final_leaf_sets, self._find_cluster_root)

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
            branch_distance_cache=self._branch_distance_cache,
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
        for cluster_metadata in cluster_assignments.values():
            root_node = cluster_metadata.get("root_node")
            if isinstance(root_node, str) and root_node:
                cluster_roots.add(root_node)

        if not cluster_roots:
            return cluster_assignments, []

        merged_roots, audit_trail = apply_posthoc_merge(
            cluster_roots=cluster_roots,
            alpha=self.posthoc_merge_alpha,
            tree=self.tree,
            children=self._children,
            test_divergence=self._test_cluster_pair_divergence,
        )

        # Deterministic cluster ids: avoid iterating an unordered set.
        merged_leaf_sets: List[set[str]] = []
        for root_node in sorted(merged_roots):
            leaf_labels = self._get_all_leaves(root_node)
            if leaf_labels:
                merged_leaf_sets.append(set(leaf_labels))

        return self._build_cluster_assignments(merged_leaf_sets), audit_trail
