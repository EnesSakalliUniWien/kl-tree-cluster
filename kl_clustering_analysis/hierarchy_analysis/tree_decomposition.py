"""Tree decomposition logic for KL-based clustering.

This module contains :class:`~kl_clustering_analysis.hierarchy_analysis.tree_decomposition.TreeDecomposition`,
which traverses a hierarchy and decides where to split or merge to form clusters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..tree.feature_space import FeatureSpace
    from ..tree.poset_tree import PosetTree

import pandas as pd

from .. import config
from ..core_utils.data_utils import extract_bool_column_dict
from .cluster_assignments import ClusterBoundary, build_cluster_assignments
from .decomposition.gates.annotation_bundle import GateAnnotationBundle
from .decomposition.gates.column_contracts import (
    validate_edge_gate_columns,
    validate_sibling_gate_columns,
)
from .decomposition.gates.gate_evaluator import GateEvaluator, TraversalDecision
from .decomposition.gates.orchestrator import (
    SiblingGateProfile,
    build_gate_annotation_config_metadata,
    build_gate_annotation_leaf_data_metadata,
    resolve_sibling_gate_profile_config,
    run_gate_annotation_pipeline,
)
from .decomposition.gates.spectral_transport import (
    DEFAULT_SPECTRAL_TRANSPORT_BLOCK_LOG_TOLERANCE,
    DEFAULT_SPECTRAL_TRANSPORT_MAX_COST,
    DEFAULT_SPECTRAL_TRANSPORT_UNMATCHED_MODE_PENALTY,
)
from .statistics.alpha_contract import DEFAULT_EDGE_ALPHA, DEFAULT_SIBLING_ALPHA
from .statistics.child_parent_divergence.child_parent_divergence_annotation.spectral_context import (
    EDGE_GATE_SPECTRAL_MINIMUM_PROJECTION_DIMENSION,
)
from .statistics.sibling_divergence.inflation_correction.empirical_null_inflation_estimation import (
    DEFAULT_INTERNAL_SUPPORT_THRESHOLDS,
)
from .statistics.sibling_divergence.inflation_correction.types.inflation_model import (
    CalibrationSupportThresholds,
)


class TreeDecomposition:
    """Annotate a hierarchy with significance tests and carve it into clusters.

    The decomposer walks a :class:`~tree.poset_tree.PosetTree` top-down and decides
    whether to split or stop at each internal node. A split requires one structure
    prerequisite and two statistical gates:

    #. **Binary structure prerequisite** - parent must have exactly two children.
    #. **Edge divergence gate** - at least one child must significantly diverge
       from the parent (projected Wald chi-square test), confirming there is
       edge-level signal to split on.
    #. **Sibling divergence gate** - siblings must have significantly different
       distributions according to a projected Wald chi-square test with
       empirical-null inflation and sibling FDR correction.

    Nodes that do not split become cluster boundaries. Leaves under the same
    boundary node are assigned the same cluster identifier. In pass-through mode,
    a closed sibling-divergence gate may still allow traversal to descendants when a deeper
    split is already supported.
    """

    def __init__(
        self,
        tree: PosetTree,
        annotations_df: pd.DataFrame | None = None,
        *,
        gate_annotation_bundle: GateAnnotationBundle | None = None,
        edge_alpha: float = DEFAULT_EDGE_ALPHA,
        sibling_alpha: float = DEFAULT_SIBLING_ALPHA,
        leaf_data: pd.DataFrame | None = None,
        feature_space: FeatureSpace | None = None,
        spectral_minimum_dimension: int = EDGE_GATE_SPECTRAL_MINIMUM_PROJECTION_DIMENSION,
        spectral_include_internal_barycenters: bool = False,
        sibling_gate_profile: str | SiblingGateProfile | None = None,
        sibling_gate_method: str = "projected_wald_inflation",
        sibling_gate_alpha_penalty: float = 1.0,
        root_stability_guard_threshold: float | None = None,
        root_stability_subsample_replicates: int = 0,
        root_stability_feature_fraction: float = 0.8,
        root_stability_seed: int = 0,
        root_stability_tree_distance_metric: str = "hamming",
        root_stability_tree_linkage_method: str = "average",
        root_selective_permutation_guard_replicates: int = 0,
        root_selective_permutation_guard_seed: int = 0,
        root_selective_permutation_guard_alpha: float | None = None,
        root_selective_permutation_guard_scope: str = "root",
        root_selective_permutation_guard_tree_distance_metric: str = "hamming",
        root_selective_permutation_guard_tree_linkage_method: str = "average",
        enforce_internal_support_thresholds: bool = False,
        internal_support_thresholds: CalibrationSupportThresholds = (
            DEFAULT_INTERNAL_SUPPORT_THRESHOLDS
        ),
        spectral_transport_passthrough_guard: bool = False,
        spectral_transport_max_cost: float = DEFAULT_SPECTRAL_TRANSPORT_MAX_COST,
        spectral_transport_require_mp_blocks: bool = True,
        spectral_transport_block_log_tolerance: float = (
            DEFAULT_SPECTRAL_TRANSPORT_BLOCK_LOG_TOLERANCE
        ),
        spectral_transport_unmatched_mode_penalty: float = (
            DEFAULT_SPECTRAL_TRANSPORT_UNMATCHED_MODE_PENALTY
        ),
        passthrough: bool = config.PASSTHROUGH,
    ):
        """Configure decomposition thresholds and pre-compute reusable metadata.

        Parameters
        ----------
        tree
            Directed hierarchy (typically a :class:`~tree.poset_tree.PosetTree`).
        annotations_df
            DataFrame of statistical annotations (e.g., columns produced by
            ``hierarchy_analysis.statistics`` helpers). Used as input to the
            gate annotation pipeline.
        gate_annotation_bundle
            Explicit reusable output from ``run_gate_annotation_pipeline``.
            This is the only cache-valid gate annotation contract.
        edge_alpha
            Significance level used by the child-parent edge-divergence gate.
        sibling_alpha
            Significance level used by sibling-divergence annotations and gating.
        sibling_gate_method
            Sibling gate strategy. The default keeps the current projected-Wald
            inflation path. Fixed-subspace methods avoid parent PCA/dimension
            selection in the sibling statistic.
        sibling_gate_profile
            Optional named profile that expands to an auditable fixed-gate
            method, selected-topology penalty, root-stability guard, and
            optional selected-root permutation guard.
        sibling_gate_alpha_penalty
            Positive divisor applied to ``sibling_alpha`` before sibling FDR.
            This exposes the selected-topology penalty used by fixed-gate
            diagnostics without changing the default effective alpha.
        root_stability_guard_threshold
            Optional fail-closed root guard. When configured, the root sibling
            gate is closed if its feature-subsample root stability ARI falls
            below this threshold.
        leaf_data
            Raw feature matrix required for per-node spectral dimension estimation.
            Missing leaf data is a contract error for the gate annotation pipeline.
        """
        if annotations_df is not None and gate_annotation_bundle is not None:
            raise ValueError("Pass either annotations_df or gate_annotation_bundle, not both.")

        self.tree = tree
        self._gate_annotation_bundle = gate_annotation_bundle
        if gate_annotation_bundle is not None:
            self.annotations_df = gate_annotation_bundle.annotated_df
        elif annotations_df is not None:
            self.annotations_df = annotations_df
        else:
            self.annotations_df = pd.DataFrame()
        self.edge_alpha = float(edge_alpha)
        self.sibling_alpha = float(sibling_alpha)
        self._leaf_data = leaf_data
        self._feature_space = feature_space
        self._spectral_minimum_dimension = int(spectral_minimum_dimension)
        self._spectral_include_internal_barycenters = bool(
            spectral_include_internal_barycenters
        )
        (
            self._sibling_gate_profile_id,
            self._sibling_gate_method,
            self._sibling_gate_alpha_penalty,
            self._root_stability_guard_threshold,
            self._root_stability_subsample_replicates,
            self._root_stability_feature_fraction,
            self._root_stability_seed,
            resolved_root_selective_permutation_guard_replicates,
            resolved_root_selective_permutation_guard_seed,
            resolved_root_selective_permutation_guard_alpha,
            resolved_root_selective_permutation_guard_scope,
            resolved_spectral_transport_passthrough_guard,
            resolved_spectral_transport_max_cost,
            resolved_spectral_transport_require_mp_blocks,
            resolved_spectral_transport_block_log_tolerance,
            resolved_spectral_transport_unmatched_mode_penalty,
        ) = resolve_sibling_gate_profile_config(
            sibling_gate_profile=sibling_gate_profile,
            sibling_gate_method=sibling_gate_method,
            sibling_gate_alpha_penalty=sibling_gate_alpha_penalty,
            root_stability_guard_threshold=root_stability_guard_threshold,
            root_stability_subsample_replicates=root_stability_subsample_replicates,
            root_stability_feature_fraction=root_stability_feature_fraction,
            root_stability_seed=root_stability_seed,
            root_selective_permutation_guard_replicates=(
                root_selective_permutation_guard_replicates
            ),
            root_selective_permutation_guard_seed=(
                root_selective_permutation_guard_seed
            ),
            root_selective_permutation_guard_alpha=(
                root_selective_permutation_guard_alpha
            ),
            root_selective_permutation_guard_scope=(
                root_selective_permutation_guard_scope
            ),
            spectral_transport_passthrough_guard=(
                spectral_transport_passthrough_guard
            ),
            spectral_transport_max_cost=spectral_transport_max_cost,
            spectral_transport_require_mp_blocks=spectral_transport_require_mp_blocks,
            spectral_transport_block_log_tolerance=(
                spectral_transport_block_log_tolerance
            ),
            spectral_transport_unmatched_mode_penalty=(
                spectral_transport_unmatched_mode_penalty
            ),
        )
        self._enforce_internal_support_thresholds = bool(
            enforce_internal_support_thresholds
        )
        self._root_stability_tree_distance_metric = str(
            root_stability_tree_distance_metric
        )
        self._root_stability_tree_linkage_method = str(
            root_stability_tree_linkage_method
        )
        self._root_selective_permutation_guard_replicates = int(
            resolved_root_selective_permutation_guard_replicates
        )
        self._root_selective_permutation_guard_seed = int(
            resolved_root_selective_permutation_guard_seed
        )
        self._root_selective_permutation_guard_alpha = (
            None
            if resolved_root_selective_permutation_guard_alpha is None
            else float(resolved_root_selective_permutation_guard_alpha)
        )
        self._root_selective_permutation_guard_scope = str(
            resolved_root_selective_permutation_guard_scope
        )
        self._root_selective_permutation_guard_tree_distance_metric = str(
            root_selective_permutation_guard_tree_distance_metric
        )
        self._root_selective_permutation_guard_tree_linkage_method = str(
            root_selective_permutation_guard_tree_linkage_method
        )
        self._internal_support_thresholds = internal_support_thresholds
        self._spectral_transport_passthrough_guard = bool(
            resolved_spectral_transport_passthrough_guard
        )
        self._spectral_transport_max_cost = float(
            resolved_spectral_transport_max_cost
        )
        self._spectral_transport_require_mp_blocks = bool(
            resolved_spectral_transport_require_mp_blocks
        )
        self._spectral_transport_block_log_tolerance = float(
            resolved_spectral_transport_block_log_tolerance
        )
        self._spectral_transport_unmatched_mode_penalty = float(
            resolved_spectral_transport_unmatched_mode_penalty
        )

        # ----- root -----
        self._root = self.tree.root()

        self._node_ids = tuple(self.tree.nodes)

        # ----- leaf partitions & counts (poset view) -----
        self._descendant_leaf_sets = self.tree.compute_descendant_sets(use_labels=True)

        # ----- ensure statistical annotations are present -----
        self.annotations_df = self._prepare_annotations(self.annotations_df)

        self._edge_divergent = self._extract_required_bool_annotation_column(
            "Child_Parent_Divergence_Significant"
        )
        # Sibling divergence test: Sibling_BH_Different = True means siblings differ -> SPLIT
        self._sibling_different = self._extract_required_bool_annotation_column(
            "Sibling_BH_Different"
        )

        self._sibling_skipped = self._extract_required_bool_annotation_column(
            "Sibling_Divergence_Skipped"
        )
        self._passthrough_supported = (
            self._extract_required_bool_annotation_column(
                "Spectral_Transport_Pass_Through_Supported"
            )
            if self._spectral_transport_passthrough_guard
            else None
        )
        self._passthrough_bottleneck = (
            {
                node: str(self.annotations_df.loc[node, "Spectral_Transport_Bottleneck"])
                for node in self._node_ids
            }
            if self._spectral_transport_passthrough_guard
            and "Spectral_Transport_Bottleneck" in self.annotations_df.columns
            else None
        )

        # Precompute children list (avoids rebuilding generator repeatedly)
        self._children: dict[object, list[object]] = {
            n: list(self.tree.successors(n)) for n in self._node_ids
        }

        # ----- construct the GateEvaluator -----
        self._gate = GateEvaluator(
            tree=self.tree,
            edge_divergent=self._edge_divergent,
            sibling_different=self._sibling_different,
            sibling_skipped=self._sibling_skipped,
            children_map=self._children,
            passthrough=bool(passthrough),
            passthrough_supported=self._passthrough_supported,
            passthrough_bottleneck=self._passthrough_bottleneck,
        )

    # ---------- initialization helpers ----------

    def _prepare_annotations(self, annotations_df: pd.DataFrame) -> pd.DataFrame:
        """Ensure statistical annotation columns are present on *annotations_df*.

        Reuses precomputed gate annotations only when they are supplied as an
        explicit ``GateAnnotationBundle`` whose metadata matches this
        decomposition request.
        """
        if self._gate_annotation_bundle is not None and self._can_reuse_gate_annotation_bundle(
            self._gate_annotation_bundle
        ):
            return self._gate_annotation_bundle.annotated_df

        annotation_bundle = run_gate_annotation_pipeline(
            self.tree,
            annotations_df,
            edge_alpha=self.edge_alpha,
            sibling_alpha=self.sibling_alpha,
            leaf_data=self._leaf_data,
            feature_space=self._feature_space,
            spectral_minimum_dimension=self._spectral_minimum_dimension,
            spectral_include_internal_barycenters=(
                self._spectral_include_internal_barycenters
            ),
            sibling_gate_profile=self._sibling_gate_profile_id,
            sibling_gate_method=self._sibling_gate_method,
            sibling_gate_alpha_penalty=self._sibling_gate_alpha_penalty,
            root_stability_guard_threshold=self._root_stability_guard_threshold,
            root_stability_subsample_replicates=(
                self._root_stability_subsample_replicates
            ),
            root_stability_feature_fraction=self._root_stability_feature_fraction,
            root_stability_seed=self._root_stability_seed,
            root_stability_tree_distance_metric=(
                self._root_stability_tree_distance_metric
            ),
            root_stability_tree_linkage_method=(
                self._root_stability_tree_linkage_method
            ),
            root_selective_permutation_guard_replicates=(
                self._root_selective_permutation_guard_replicates
            ),
            root_selective_permutation_guard_seed=(
                self._root_selective_permutation_guard_seed
            ),
            root_selective_permutation_guard_alpha=(
                self._root_selective_permutation_guard_alpha
            ),
            root_selective_permutation_guard_scope=(
                self._root_selective_permutation_guard_scope
            ),
            root_selective_permutation_guard_tree_distance_metric=(
                self._root_selective_permutation_guard_tree_distance_metric
            ),
            root_selective_permutation_guard_tree_linkage_method=(
                self._root_selective_permutation_guard_tree_linkage_method
            ),
            enforce_internal_support_thresholds=(
                self._enforce_internal_support_thresholds
            ),
            internal_support_thresholds=self._internal_support_thresholds,
            spectral_transport_passthrough_guard=(
                self._spectral_transport_passthrough_guard
            ),
            spectral_transport_max_cost=self._spectral_transport_max_cost,
            spectral_transport_require_mp_blocks=(
                self._spectral_transport_require_mp_blocks
            ),
            spectral_transport_block_log_tolerance=(
                self._spectral_transport_block_log_tolerance
            ),
            spectral_transport_unmatched_mode_penalty=(
                self._spectral_transport_unmatched_mode_penalty
            ),
        )
        self._gate_annotation_bundle = annotation_bundle
        return annotation_bundle.annotated_df

    def _can_reuse_gate_annotation_bundle(
        self,
        gate_annotation_bundle: GateAnnotationBundle,
    ) -> bool:
        """Return whether existing gate annotations can be trusted as current."""
        annotations_df = gate_annotation_bundle.annotated_df
        if annotations_df.empty:
            return False

        validate_edge_gate_columns(annotations_df)
        validate_sibling_gate_columns(annotations_df)

        required_gate_decision_columns = (
            "Child_Parent_Divergence_Significant",
            "Sibling_BH_Different",
            "Sibling_Divergence_Skipped",
        )
        if any(column not in annotations_df.columns for column in required_gate_decision_columns):
            return False
        if any(annotations_df[column].isna().any() for column in required_gate_decision_columns):
            return False
        if set(self._node_ids) - set(annotations_df.index):
            return False

        metadata = gate_annotation_bundle.metadata

        return (
            metadata.pipeline == "gate_annotation"
            and metadata.edge.alpha == self.edge_alpha
            and metadata.sibling.alpha == self.sibling_alpha
            and metadata.config
            == build_gate_annotation_config_metadata(
                spectral_minimum_dimension=self._spectral_minimum_dimension,
                spectral_include_internal_barycenters=(
                    self._spectral_include_internal_barycenters
                ),
                sibling_gate_profile_id=self._sibling_gate_profile_id,
                sibling_gate_method=self._sibling_gate_method,
                sibling_gate_alpha_penalty=self._sibling_gate_alpha_penalty,
                root_stability_guard_threshold=self._root_stability_guard_threshold,
                root_stability_subsample_replicates=(
                    self._root_stability_subsample_replicates
                ),
                root_stability_feature_fraction=self._root_stability_feature_fraction,
                root_stability_seed=self._root_stability_seed,
                root_stability_tree_distance_metric=(
                    self._root_stability_tree_distance_metric
                ),
                root_stability_tree_linkage_method=(
                    self._root_stability_tree_linkage_method
                ),
                root_selective_permutation_guard_replicates=(
                    self._root_selective_permutation_guard_replicates
                ),
                root_selective_permutation_guard_seed=(
                    self._root_selective_permutation_guard_seed
                ),
                root_selective_permutation_guard_alpha=(
                    self._root_selective_permutation_guard_alpha
                ),
                root_selective_permutation_guard_scope=(
                    self._root_selective_permutation_guard_scope
                ),
                root_selective_permutation_guard_tree_distance_metric=(
                    self._root_selective_permutation_guard_tree_distance_metric
                ),
                root_selective_permutation_guard_tree_linkage_method=(
                    self._root_selective_permutation_guard_tree_linkage_method
                ),
                enforce_internal_support_thresholds=(
                    self._enforce_internal_support_thresholds
                ),
                internal_support_thresholds=self._internal_support_thresholds,
                spectral_transport_passthrough_guard=(
                    self._spectral_transport_passthrough_guard
                ),
                spectral_transport_max_cost=self._spectral_transport_max_cost,
                spectral_transport_require_mp_blocks=(
                    self._spectral_transport_require_mp_blocks
                ),
                spectral_transport_block_log_tolerance=(
                    self._spectral_transport_block_log_tolerance
                ),
                spectral_transport_unmatched_mode_penalty=(
                    self._spectral_transport_unmatched_mode_penalty
                ),
            )
            and metadata.leaf_data
            == build_gate_annotation_leaf_data_metadata(
                self._leaf_data,
                feature_space=self._feature_space,
            )
        )

    def _extract_required_bool_annotation_column(self, column_name: str) -> dict[object, bool]:
        """Extract a required boolean annotation column keyed by tree node id."""
        return extract_bool_column_dict(
            self.annotations_df,
            column_name,
            coerce_index_to_str=False,
        )

    # ---------- core decomposition (iterative, no recursion) ----------

    def decompose_tree(self) -> dict[str, object]:
        """Return cluster assignments by iteratively traversing the hierarchy.

        Traversal order
        ---------------
        The traversal uses a last in, first out list (similar to an explicit stack),
        which produces a depth-first traversal order. When a node is split, its
        two children are appended in right-then-left order so that the left child
        is processed first on the next iteration.
        """
        nodes_to_visit: list[object] = [self._root]
        final_boundaries: list[ClusterBoundary] = []
        traversal_trace: list[dict[str, object]] = []
        processed: set[object] = set()

        while nodes_to_visit:
            node = nodes_to_visit.pop()
            if node in processed:
                continue
            processed.add(node)

            decision = self._gate.decision(node)
            children = self._children[node]
            passthrough_support = self._gate.passthrough_support_status(node)
            traversal_trace.append(
                {
                    "node_id": node,
                    "decision": decision.value,
                    "is_leaf": len(children) == 0,
                    "n_children": len(children),
                    "n_descendant_leaves": len(self._descendant_leaf_sets[node]),
                    "final_boundary": decision is TraversalDecision.BOUNDARY,
                    **passthrough_support,
                }
            )
            if decision in (TraversalDecision.SPLIT, TraversalDecision.PASS_THROUGH):
                left_child, right_child = children
                nodes_to_visit.append(right_child)
                nodes_to_visit.append(left_child)
                continue

            final_boundaries.append(
                ClusterBoundary(root_node=node, leaves=self._descendant_leaf_sets[node])
            )

        cluster_assignments = build_cluster_assignments(final_boundaries)

        return {
            "cluster_assignments": cluster_assignments,
            "num_clusters": len(cluster_assignments),
            "traversal_trace": traversal_trace,
            "independence_analysis": {
                "edge_alpha": self.edge_alpha,
                "sibling_alpha": self.sibling_alpha,
                "decision_mode": "sibling_divergence",
            },
        }
