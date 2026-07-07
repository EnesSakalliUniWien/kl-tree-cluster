"""Tree decomposition logic for TBS-based clustering.

This module contains :class:`~tree_break_selection.hierarchy_analysis.tree_decomposition.TreeDecomposition`,
which traverses a hierarchy and decides where to split or merge to form clusters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..tree.feature_space import FeatureSpace
    from ..tree.poset_tree import PosetTree

import pandas as pd

from .. import config as runtime_config
from ..core_utils.data_utils import extract_bool_column_dict
from ..tree.distributions import (
    DEFAULT_CONTINUOUS_COVARIANCE_MIN_CHILD_LEAF_COUNT,
    DEFAULT_CONTINUOUS_COVARIANCE_POLICY,
    validate_continuous_covariance_min_child_leaf_count,
    validate_continuous_covariance_policy,
)
from .cluster_assignments import ClusterBoundary, build_cluster_assignments
from .decomposition.config import (
    TRACE_LEVEL_COMPACT,
    TRACE_LEVEL_FULL,
    DecompositionConfig,
    TraceLevel,
    validate_trace_level,
)
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
from .statistics.branch_length_utils import (
    EDGE_BRANCH_LENGTH_VARIANCE_POLICY_NONE,
    validate_edge_branch_length_variance_policy,
)
from .statistics.child_parent_divergence.child_parent_divergence_annotation.spectral_context import (
    EDGE_GATE_SPECTRAL_MINIMUM_PROJECTION_DIMENSION,
)
from .statistics.projection.spectral.tree_estimator import (
    INTERNAL_DISTRIBUTION_EMPIRICAL_BARYCENTER,
    MP_ROW_COUNT_LEAF_EFFECTIVE_ROWS,
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
        adaptive_projection_dimension_energy_fraction: float | None = None,
        spectral_include_internal_barycenters: bool = False,
        spectral_internal_distribution_mode: str = INTERNAL_DISTRIBUTION_EMPIRICAL_BARYCENTER,
        spectral_mp_row_count_mode: str = MP_ROW_COUNT_LEAF_EFFECTIVE_ROWS,
        continuous_covariance_policy: str = DEFAULT_CONTINUOUS_COVARIANCE_POLICY,
        continuous_covariance_min_child_leaf_count: int = (
            DEFAULT_CONTINUOUS_COVARIANCE_MIN_CHILD_LEAF_COUNT
        ),
        edge_branch_length_variance_policy: str = EDGE_BRANCH_LENGTH_VARIANCE_POLICY_NONE,
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
        passthrough: bool = runtime_config.PASSTHROUGH,
        trace_level: TraceLevel = TRACE_LEVEL_COMPACT,
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
        self.decomposition_config = DecompositionConfig(
            edge_alpha=edge_alpha,
            sibling_alpha=sibling_alpha,
            spectral_minimum_dimension=spectral_minimum_dimension,
            adaptive_projection_dimension_energy_fraction=(
                adaptive_projection_dimension_energy_fraction
            ),
            spectral_include_internal_barycenters=spectral_include_internal_barycenters,
            spectral_internal_distribution_mode=spectral_internal_distribution_mode,
            spectral_mp_row_count_mode=spectral_mp_row_count_mode,
            continuous_covariance_policy=continuous_covariance_policy,
            continuous_covariance_min_child_leaf_count=(continuous_covariance_min_child_leaf_count),
            edge_branch_length_variance_policy=edge_branch_length_variance_policy,
            sibling_gate_profile=sibling_gate_profile,
            sibling_gate_method=sibling_gate_method,
            sibling_gate_alpha_penalty=sibling_gate_alpha_penalty,
            root_stability_guard_threshold=root_stability_guard_threshold,
            root_stability_subsample_replicates=root_stability_subsample_replicates,
            root_stability_feature_fraction=root_stability_feature_fraction,
            root_stability_seed=root_stability_seed,
            root_stability_tree_distance_metric=root_stability_tree_distance_metric,
            root_stability_tree_linkage_method=root_stability_tree_linkage_method,
            root_selective_permutation_guard_replicates=(
                root_selective_permutation_guard_replicates
            ),
            root_selective_permutation_guard_seed=root_selective_permutation_guard_seed,
            root_selective_permutation_guard_alpha=root_selective_permutation_guard_alpha,
            root_selective_permutation_guard_scope=root_selective_permutation_guard_scope,
            root_selective_permutation_guard_tree_distance_metric=(
                root_selective_permutation_guard_tree_distance_metric
            ),
            root_selective_permutation_guard_tree_linkage_method=(
                root_selective_permutation_guard_tree_linkage_method
            ),
            enforce_internal_support_thresholds=enforce_internal_support_thresholds,
            internal_support_thresholds=internal_support_thresholds,
            spectral_transport_passthrough_guard=spectral_transport_passthrough_guard,
            spectral_transport_max_cost=spectral_transport_max_cost,
            spectral_transport_require_mp_blocks=spectral_transport_require_mp_blocks,
            spectral_transport_block_log_tolerance=spectral_transport_block_log_tolerance,
            spectral_transport_unmatched_mode_penalty=spectral_transport_unmatched_mode_penalty,
            passthrough=passthrough,
            trace_level=trace_level,
        )
        self.edge_alpha = float(self.decomposition_config.edge_alpha)
        self.sibling_alpha = float(self.decomposition_config.sibling_alpha)
        self._leaf_data = leaf_data
        self._feature_space = feature_space
        self._trace_level = validate_trace_level(str(self.decomposition_config.trace_level))
        self._spectral_minimum_dimension = int(
            self.decomposition_config.spectral_minimum_dimension
        )
        self._adaptive_projection_dimension_energy_fraction = (
            None
            if self.decomposition_config.adaptive_projection_dimension_energy_fraction is None
            else float(self.decomposition_config.adaptive_projection_dimension_energy_fraction)
        )
        self._spectral_include_internal_barycenters = bool(
            self.decomposition_config.spectral_include_internal_barycenters
        )
        self._spectral_internal_distribution_mode = str(
            self.decomposition_config.spectral_internal_distribution_mode
        )
        self._spectral_mp_row_count_mode = str(
            self.decomposition_config.spectral_mp_row_count_mode
        )
        self._continuous_covariance_policy = validate_continuous_covariance_policy(
            self.decomposition_config.continuous_covariance_policy
        )
        self._continuous_covariance_min_child_leaf_count = (
            validate_continuous_covariance_min_child_leaf_count(
                self.decomposition_config.continuous_covariance_min_child_leaf_count
            )
        )
        self._edge_branch_length_variance_policy = validate_edge_branch_length_variance_policy(
            self.decomposition_config.edge_branch_length_variance_policy
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
            sibling_gate_profile=self.decomposition_config.sibling_gate_profile,
            sibling_gate_method=self.decomposition_config.sibling_gate_method,
            sibling_gate_alpha_penalty=self.decomposition_config.sibling_gate_alpha_penalty,
            root_stability_guard_threshold=(
                self.decomposition_config.root_stability_guard_threshold
            ),
            root_stability_subsample_replicates=(
                self.decomposition_config.root_stability_subsample_replicates
            ),
            root_stability_feature_fraction=(
                self.decomposition_config.root_stability_feature_fraction
            ),
            root_stability_seed=self.decomposition_config.root_stability_seed,
            root_selective_permutation_guard_replicates=(
                self.decomposition_config.root_selective_permutation_guard_replicates
            ),
            root_selective_permutation_guard_seed=(
                self.decomposition_config.root_selective_permutation_guard_seed
            ),
            root_selective_permutation_guard_alpha=(
                self.decomposition_config.root_selective_permutation_guard_alpha
            ),
            root_selective_permutation_guard_scope=(
                self.decomposition_config.root_selective_permutation_guard_scope
            ),
            spectral_transport_passthrough_guard=(
                self.decomposition_config.spectral_transport_passthrough_guard
            ),
            spectral_transport_max_cost=self.decomposition_config.spectral_transport_max_cost,
            spectral_transport_require_mp_blocks=(
                self.decomposition_config.spectral_transport_require_mp_blocks
            ),
            spectral_transport_block_log_tolerance=(
                self.decomposition_config.spectral_transport_block_log_tolerance
            ),
            spectral_transport_unmatched_mode_penalty=(
                self.decomposition_config.spectral_transport_unmatched_mode_penalty
            ),
        )
        self._enforce_internal_support_thresholds = bool(
            self.decomposition_config.enforce_internal_support_thresholds
        )
        self._root_stability_tree_distance_metric = str(
            self.decomposition_config.root_stability_tree_distance_metric
        )
        self._root_stability_tree_linkage_method = str(
            self.decomposition_config.root_stability_tree_linkage_method
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
            self.decomposition_config.root_selective_permutation_guard_tree_distance_metric
        )
        self._root_selective_permutation_guard_tree_linkage_method = str(
            self.decomposition_config.root_selective_permutation_guard_tree_linkage_method
        )
        self._internal_support_thresholds = self.decomposition_config.internal_support_thresholds
        self._spectral_transport_passthrough_guard = bool(
            resolved_spectral_transport_passthrough_guard
        )
        self._spectral_transport_max_cost = float(resolved_spectral_transport_max_cost)
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
        self._passthrough_supported = self._build_passthrough_support_map()
        self._passthrough_bottleneck = self._build_passthrough_bottleneck_map()

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
            passthrough=bool(self.decomposition_config.passthrough),
            passthrough_supported=self._passthrough_supported,
            passthrough_bottleneck=self._passthrough_bottleneck,
        )

    # ---------- initialization helpers ----------

    def _uses_selected_family_passthrough_guard(self) -> bool:
        return self._root_selective_permutation_guard_scope in {
            "passthrough_descendant",
            "global_sibling_min_passthrough_descendant",
            "global_sibling_min_passthrough_descendant_refined",
        }

    def _annotation_bool_value(self, node: object, column: str) -> bool:
        if column not in self.annotations_df.columns or node not in self.annotations_df.index:
            return False
        value = self.annotations_df.loc[node, column]
        return bool(pd.notna(value) and bool(value))

    def _build_passthrough_support_map(self) -> dict[object, bool] | None:
        selected_family_support = None
        if (
            self._uses_selected_family_passthrough_guard()
            and "Selective_Permutation_Guard_Would_Block" in self.annotations_df.columns
        ):
            selected_family_support = {
                node: not self._annotation_bool_value(
                    node, "Selective_Permutation_Guard_Would_Block"
                )
                for node in self._node_ids
            }

        spectral_support = None
        if self._spectral_transport_passthrough_guard:
            spectral_support = self._extract_required_bool_annotation_column(
                "Spectral_Transport_Pass_Through_Supported"
            )

        if selected_family_support is None and spectral_support is None:
            return None

        return {
            node: bool(
                (selected_family_support[node] if selected_family_support is not None else True)
                and (spectral_support[node] if spectral_support is not None else True)
            )
            for node in self._node_ids
        }

    def _build_passthrough_bottleneck_map(self) -> dict[object, str] | None:
        if self._passthrough_supported is None:
            return None

        bottlenecks: dict[object, str] = {}
        for node in self._node_ids:
            reasons: list[str] = []
            if self._uses_selected_family_passthrough_guard() and self._annotation_bool_value(
                node, "Selective_Permutation_Guard_Would_Block"
            ):
                reasons.append("selected_family_passthrough_guard_blocked")
            if (
                self._spectral_transport_passthrough_guard
                and "Spectral_Transport_Pass_Through_Supported" in self.annotations_df.columns
                and not self._annotation_bool_value(
                    node, "Spectral_Transport_Pass_Through_Supported"
                )
            ):
                spectral_bottleneck = (
                    str(self.annotations_df.loc[node, "Spectral_Transport_Bottleneck"])
                    if "Spectral_Transport_Bottleneck" in self.annotations_df.columns
                    and node in self.annotations_df.index
                    else "spectral_transport_bottleneck"
                )
                reasons.append(spectral_bottleneck)
            bottlenecks[node] = ";".join(reason for reason in reasons if reason)
        return bottlenecks

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
            adaptive_projection_dimension_energy_fraction=(
                self._adaptive_projection_dimension_energy_fraction
            ),
            spectral_include_internal_barycenters=(self._spectral_include_internal_barycenters),
            spectral_internal_distribution_mode=(self._spectral_internal_distribution_mode),
            spectral_mp_row_count_mode=self._spectral_mp_row_count_mode,
            continuous_covariance_policy=self._continuous_covariance_policy,
            continuous_covariance_min_child_leaf_count=(
                self._continuous_covariance_min_child_leaf_count
            ),
            edge_branch_length_variance_policy=(self._edge_branch_length_variance_policy),
            sibling_gate_profile=self._sibling_gate_profile_id,
            sibling_gate_method=self._sibling_gate_method,
            sibling_gate_alpha_penalty=self._sibling_gate_alpha_penalty,
            root_stability_guard_threshold=self._root_stability_guard_threshold,
            root_stability_subsample_replicates=(self._root_stability_subsample_replicates),
            root_stability_feature_fraction=self._root_stability_feature_fraction,
            root_stability_seed=self._root_stability_seed,
            root_stability_tree_distance_metric=(self._root_stability_tree_distance_metric),
            root_stability_tree_linkage_method=(self._root_stability_tree_linkage_method),
            root_selective_permutation_guard_replicates=(
                self._root_selective_permutation_guard_replicates
            ),
            root_selective_permutation_guard_seed=(self._root_selective_permutation_guard_seed),
            root_selective_permutation_guard_alpha=(self._root_selective_permutation_guard_alpha),
            root_selective_permutation_guard_scope=(self._root_selective_permutation_guard_scope),
            root_selective_permutation_guard_tree_distance_metric=(
                self._root_selective_permutation_guard_tree_distance_metric
            ),
            root_selective_permutation_guard_tree_linkage_method=(
                self._root_selective_permutation_guard_tree_linkage_method
            ),
            enforce_internal_support_thresholds=(self._enforce_internal_support_thresholds),
            internal_support_thresholds=self._internal_support_thresholds,
            spectral_transport_passthrough_guard=(self._spectral_transport_passthrough_guard),
            spectral_transport_max_cost=self._spectral_transport_max_cost,
            spectral_transport_require_mp_blocks=(self._spectral_transport_require_mp_blocks),
            spectral_transport_block_log_tolerance=(self._spectral_transport_block_log_tolerance),
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
                adaptive_projection_dimension_energy_fraction=(
                    self._adaptive_projection_dimension_energy_fraction
                ),
                spectral_include_internal_barycenters=(self._spectral_include_internal_barycenters),
                spectral_internal_distribution_mode=(self._spectral_internal_distribution_mode),
                spectral_mp_row_count_mode=self._spectral_mp_row_count_mode,
                continuous_covariance_policy=self._continuous_covariance_policy,
                continuous_covariance_min_child_leaf_count=(
                    self._continuous_covariance_min_child_leaf_count
                ),
                edge_branch_length_variance_policy=(self._edge_branch_length_variance_policy),
                sibling_gate_profile_id=self._sibling_gate_profile_id,
                sibling_gate_method=self._sibling_gate_method,
                sibling_gate_alpha_penalty=self._sibling_gate_alpha_penalty,
                root_stability_guard_threshold=self._root_stability_guard_threshold,
                root_stability_subsample_replicates=(self._root_stability_subsample_replicates),
                root_stability_feature_fraction=self._root_stability_feature_fraction,
                root_stability_seed=self._root_stability_seed,
                root_stability_tree_distance_metric=(self._root_stability_tree_distance_metric),
                root_stability_tree_linkage_method=(self._root_stability_tree_linkage_method),
                root_selective_permutation_guard_replicates=(
                    self._root_selective_permutation_guard_replicates
                ),
                root_selective_permutation_guard_seed=(self._root_selective_permutation_guard_seed),
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
                enforce_internal_support_thresholds=(self._enforce_internal_support_thresholds),
                internal_support_thresholds=self._internal_support_thresholds,
                spectral_transport_passthrough_guard=(self._spectral_transport_passthrough_guard),
                spectral_transport_max_cost=self._spectral_transport_max_cost,
                spectral_transport_require_mp_blocks=(self._spectral_transport_require_mp_blocks),
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

    # ---------- traversal audit helpers ----------

    def _edge_branch_length(self, parent: object, child: object) -> float | None:
        """Return the recorded branch length for an edge when present."""
        edge_attrs = self.tree.edges[parent, child]
        if "branch_length" not in edge_attrs:
            return None
        return float(edge_attrs["branch_length"])

    def _annotation_value(self, node: object | None, column: str) -> object | None:
        """Return one annotation value when the column and row are available."""
        if node is None or column not in self.annotations_df.columns:
            return None
        if node not in self.annotations_df.index:
            return None
        value = self.annotations_df.loc[node, column]
        if pd.isna(value):
            return None
        return value.item() if hasattr(value, "item") else value

    def _gate_p_value_debug_fields(
        self,
        *,
        node: object,
        left_child: object | None,
        right_child: object | None,
    ) -> dict[str, object]:
        """Return raw/corrected p-value diagnostics for edge and sibling gates."""
        return {
            "left_edge_p_value": self._annotation_value(
                left_child, "Child_Parent_Divergence_P_Value"
            ),
            "left_edge_p_value_bh": self._annotation_value(
                left_child, "Child_Parent_Divergence_P_Value_BH"
            ),
            "left_edge_tested": self._annotation_value(
                left_child, "Child_Parent_Divergence_Tested"
            ),
            "left_edge_ancestor_blocked": self._annotation_value(
                left_child, "Child_Parent_Divergence_Ancestor_Blocked"
            ),
            "right_edge_p_value": self._annotation_value(
                right_child, "Child_Parent_Divergence_P_Value"
            ),
            "right_edge_p_value_bh": self._annotation_value(
                right_child, "Child_Parent_Divergence_P_Value_BH"
            ),
            "right_edge_tested": self._annotation_value(
                right_child, "Child_Parent_Divergence_Tested"
            ),
            "right_edge_ancestor_blocked": self._annotation_value(
                right_child, "Child_Parent_Divergence_Ancestor_Blocked"
            ),
            "sibling_p_value": self._annotation_value(node, "Sibling_Divergence_P_Value"),
            "sibling_p_value_corrected": self._annotation_value(
                node, "Sibling_Divergence_P_Value_Corrected"
            ),
            "sibling_test_statistic": self._annotation_value(node, "Sibling_Test_Statistic"),
            "sibling_degrees_of_freedom": self._annotation_value(
                node, "Sibling_Degrees_of_Freedom"
            ),
            "sibling_test_method": self._annotation_value(node, "Sibling_Test_Method"),
            "sibling_gate_p_value_calibration": self._annotation_value(
                node, "Sibling_Gate_P_Value_Calibration"
            ),
            "sibling_gate_p_value_role": self._annotation_value(node, "Sibling_Gate_P_Value_Role"),
            "sibling_sparse_p_value": self._annotation_value(
                node, "Sibling_Sparse_Evidence_P_Value"
            ),
            "sibling_sparse_method": self._annotation_value(node, "Sibling_Sparse_Evidence_Method"),
            "sibling_sparse_calibration": self._annotation_value(
                node, "Sibling_Sparse_Evidence_Calibration"
            ),
            "sibling_dense_p_value": self._annotation_value(node, "Sibling_Dense_Evidence_P_Value"),
            "sibling_dense_method": self._annotation_value(node, "Sibling_Dense_Evidence_Method"),
            "sibling_dense_calibration": self._annotation_value(
                node, "Sibling_Dense_Evidence_Calibration"
            ),
            "sibling_dense_statistic": self._annotation_value(
                node, "Sibling_Dense_Evidence_Test_Statistic"
            ),
            "sibling_dense_degrees_of_freedom": self._annotation_value(
                node, "Sibling_Dense_Evidence_Degrees_of_Freedom"
            ),
            "sibling_fixed_coordinate_bh_p_value": self._annotation_value(
                node, "Sibling_Fixed_Coordinate_BH_P_Value"
            ),
            "sibling_fixed_block_bh_p_value": self._annotation_value(
                node, "Sibling_Fixed_Block_BH_P_Value"
            ),
            "sibling_fixed_global_p_value": self._annotation_value(
                node, "Sibling_Fixed_Global_P_Value"
            ),
        }

    def _full_edge_traversal_row(
        self,
        *,
        node: object,
        depth: int,
        path: tuple[object, ...],
        actual_decision: TraversalDecision,
        actual_visited: bool,
    ) -> dict[str, object]:
        """Return one audit row for edge-reachable traversal."""
        children = self._children[node]
        left_child = children[0] if len(children) == 2 else None
        right_child = children[1] if len(children) == 2 else None
        left_edge_test_tuple = (node, left_child) if left_child is not None else None
        right_edge_test_tuple = (node, right_child) if right_child is not None else None
        sibling_test_tuple = (
            (node, left_child, right_child)
            if left_child is not None and right_child is not None
            else None
        )

        left_edge_open = bool(self._edge_divergent[left_child]) if left_child is not None else False
        right_edge_open = (
            bool(self._edge_divergent[right_child]) if right_child is not None else False
        )
        edge_gate_open = bool(left_edge_open or right_edge_open)
        sibling_skipped = bool(self._sibling_skipped[node])
        sibling_different = bool(self._sibling_different[node])
        sibling_gate_open = bool(sibling_different and not sibling_skipped)
        passthrough_audit = self._gate.passthrough_audit_status(node)

        left_branch_length = (
            self._edge_branch_length(node, left_child) if left_child is not None else None
        )
        right_branch_length = (
            self._edge_branch_length(node, right_child) if right_child is not None else None
        )

        if not children:
            edge_traversal_action = "stop"
            edge_traversal_stop_reason = "leaf"
        elif len(children) != 2:
            edge_traversal_action = "stop"
            edge_traversal_stop_reason = "non_binary"
        elif edge_gate_open:
            edge_traversal_action = "continue"
            edge_traversal_stop_reason = "edge_open_continue"
        else:
            edge_traversal_action = "stop"
            edge_traversal_stop_reason = "edge_closed"

        return {
            "node_id": node,
            "left_child": left_child,
            "right_child": right_child,
            "left_edge_test_tuple": left_edge_test_tuple,
            "right_edge_test_tuple": right_edge_test_tuple,
            "sibling_test_tuple": sibling_test_tuple,
            **self._gate_p_value_debug_fields(
                node=node,
                left_child=left_child,
                right_child=right_child,
            ),
            "depth": int(depth),
            "path": path,
            "actual_visited": bool(actual_visited),
            "actual_decision": actual_decision.value,
            "edge_traversal_action": edge_traversal_action,
            "edge_traversal_stop_reason": edge_traversal_stop_reason,
            "is_leaf": len(children) == 0,
            "n_children": len(children),
            "n_descendant_leaves": len(self._descendant_leaf_sets[node]),
            "descendant_leaf_signature": tuple(sorted(self._descendant_leaf_sets[node])),
            "left_edge_open": left_edge_open,
            "right_edge_open": right_edge_open,
            "edge_gate_open": edge_gate_open,
            "sibling_different": sibling_different,
            "sibling_skipped": sibling_skipped,
            "sibling_gate_open": sibling_gate_open,
            **passthrough_audit,
            "left_branch_length": left_branch_length,
            "right_branch_length": right_branch_length,
            "left_branch_length_missing": (left_child is not None and left_branch_length is None),
            "right_branch_length_missing": (
                right_child is not None and right_branch_length is None
            ),
        }

    def _full_edge_traversal_trace(
        self,
        *,
        actual_decision_by_node: dict[object, TraversalDecision],
        actual_visited_nodes: set[object],
    ) -> list[dict[str, object]]:
        """Walk the tree until edge tests stop, independent of sibling gates."""
        nodes_to_visit: list[tuple[object, int, tuple[object, ...]]] = [
            (self._root, 0, (self._root,))
        ]
        processed: set[object] = set()
        trace: list[dict[str, object]] = []

        while nodes_to_visit:
            node, depth, path = nodes_to_visit.pop()
            if node in processed:
                continue
            processed.add(node)

            actual_decision = actual_decision_by_node.get(node)
            if actual_decision is None:
                actual_decision = self._gate.decision(node)

            row = self._full_edge_traversal_row(
                node=node,
                depth=depth,
                path=path,
                actual_decision=actual_decision,
                actual_visited=node in actual_visited_nodes,
            )
            trace.append(row)

            if row["edge_traversal_action"] == "continue":
                left_child, right_child = self._children[node]
                nodes_to_visit.append((right_child, depth + 1, (*path, right_child)))
                nodes_to_visit.append((left_child, depth + 1, (*path, left_child)))

        return trace

    @staticmethod
    def _empty_live_traversal_counters() -> dict[str, int]:
        """Return compact counters for the live decomposition traversal."""
        return {
            "live_nodes_visited": 0,
            "live_internal_tuples": 0,
            "live_split_count": 0,
            "live_pass_through_count": 0,
            "live_boundary_count": 0,
            "live_passthrough_candidate_count": 0,
            "live_passthrough_support_blocked_count": 0,
        }

    @staticmethod
    def _update_live_traversal_counters(
        counters: dict[str, int],
        *,
        decision: TraversalDecision,
        n_children: int,
        passthrough_audit: dict[str, object],
    ) -> None:
        counters["live_nodes_visited"] += 1
        counters["live_internal_tuples"] += int(n_children == 2)
        if decision is TraversalDecision.SPLIT:
            counters["live_split_count"] += 1
        elif decision is TraversalDecision.PASS_THROUGH:
            counters["live_pass_through_count"] += 1
        elif decision is TraversalDecision.BOUNDARY:
            counters["live_boundary_count"] += 1
        counters["live_passthrough_candidate_count"] += int(
            bool(passthrough_audit["passthrough_candidate"])
        )
        counters["live_passthrough_support_blocked_count"] += int(
            passthrough_audit["passthrough_decision_reason"]
            == "passthrough_support_blocked"
        )

    @staticmethod
    def _traversal_counters(
        *,
        traversal_trace: list[dict[str, object]],
        full_edge_traversal_trace: list[dict[str, object]],
    ) -> dict[str, int]:
        """Return compact counters for live and edge-reachable traversal."""
        live_decisions = [str(row["decision"]) for row in traversal_trace]
        full_stop_reasons = [
            str(row["edge_traversal_stop_reason"]) for row in full_edge_traversal_trace
        ]
        return {
            "live_nodes_visited": len(traversal_trace),
            "live_internal_tuples": sum(int(row["n_children"]) == 2 for row in traversal_trace),
            "live_split_count": live_decisions.count(TraversalDecision.SPLIT.value),
            "live_pass_through_count": live_decisions.count(TraversalDecision.PASS_THROUGH.value),
            "live_boundary_count": live_decisions.count(TraversalDecision.BOUNDARY.value),
            "live_passthrough_candidate_count": sum(
                bool(row["passthrough_candidate"]) for row in traversal_trace
            ),
            "live_passthrough_support_blocked_count": sum(
                row["passthrough_decision_reason"] == "passthrough_support_blocked"
                for row in traversal_trace
            ),
            "full_edge_nodes_visited": len(full_edge_traversal_trace),
            "full_edge_internal_tuples": sum(
                int(row["n_children"]) == 2 for row in full_edge_traversal_trace
            ),
            "full_edge_continue_count": sum(
                row["edge_traversal_action"] == "continue" for row in full_edge_traversal_trace
            ),
            "full_edge_stop_count": sum(
                row["edge_traversal_action"] == "stop" for row in full_edge_traversal_trace
            ),
            "full_edge_leaf_stop_count": full_stop_reasons.count("leaf"),
            "full_edge_non_binary_stop_count": full_stop_reasons.count("non_binary"),
            "full_edge_closed_stop_count": full_stop_reasons.count("edge_closed"),
            "full_edge_passthrough_candidate_count": sum(
                bool(row["passthrough_candidate"]) for row in full_edge_traversal_trace
            ),
            "full_edge_passthrough_support_blocked_count": sum(
                row["passthrough_decision_reason"] == "passthrough_support_blocked"
                for row in full_edge_traversal_trace
            ),
            "full_edge_missing_branch_length_count": sum(
                bool(row["left_branch_length_missing"]) + bool(row["right_branch_length_missing"])
                for row in full_edge_traversal_trace
            ),
        }

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
        collect_full_trace = self._trace_level == TRACE_LEVEL_FULL
        traversal_trace: list[dict[str, object]] = []
        traversal_counters = self._empty_live_traversal_counters()
        processed: set[object] = set()
        actual_decision_by_node: dict[object, TraversalDecision] = {}

        while nodes_to_visit:
            node = nodes_to_visit.pop()
            if node in processed:
                continue
            processed.add(node)

            decision = self._gate.decision(node)
            actual_decision_by_node[node] = decision
            children = self._children[node]
            left_child = children[0] if len(children) == 2 else None
            right_child = children[1] if len(children) == 2 else None
            left_edge_test_tuple = (node, left_child) if left_child is not None else None
            right_edge_test_tuple = (node, right_child) if right_child is not None else None
            sibling_test_tuple = (
                (node, left_child, right_child)
                if left_child is not None and right_child is not None
                else None
            )
            passthrough_audit = self._gate.passthrough_audit_status(node)
            if collect_full_trace:
                traversal_trace.append(
                    {
                        "node_id": node,
                        "left_child": left_child,
                        "right_child": right_child,
                        "left_edge_test_tuple": left_edge_test_tuple,
                        "right_edge_test_tuple": right_edge_test_tuple,
                        "sibling_test_tuple": sibling_test_tuple,
                        **self._gate_p_value_debug_fields(
                            node=node,
                            left_child=left_child,
                            right_child=right_child,
                        ),
                        "decision": decision.value,
                        "is_leaf": len(children) == 0,
                        "n_children": len(children),
                        "n_descendant_leaves": len(self._descendant_leaf_sets[node]),
                        "descendant_leaf_signature": tuple(
                            sorted(self._descendant_leaf_sets[node])
                        ),
                        "final_boundary": decision is TraversalDecision.BOUNDARY,
                        **passthrough_audit,
                    }
                )
            else:
                self._update_live_traversal_counters(
                    traversal_counters,
                    decision=decision,
                    n_children=len(children),
                    passthrough_audit=passthrough_audit,
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
        result = {
            "cluster_assignments": cluster_assignments,
            "num_clusters": len(cluster_assignments),
            "traversal_counters": traversal_counters,
            "independence_analysis": {
                "edge_alpha": self.edge_alpha,
                "sibling_alpha": self.sibling_alpha,
                "decision_mode": "sibling_divergence",
            },
        }
        if not collect_full_trace:
            return result

        full_edge_traversal_trace = self._full_edge_traversal_trace(
            actual_decision_by_node=actual_decision_by_node,
            actual_visited_nodes=processed,
        )
        result.update(
            {
                "traversal_trace": traversal_trace,
                "full_edge_traversal_trace": full_edge_traversal_trace,
                "traversal_counters": self._traversal_counters(
                    traversal_trace=traversal_trace,
                    full_edge_traversal_trace=full_edge_traversal_trace,
                ),
            }
        )
        return result
