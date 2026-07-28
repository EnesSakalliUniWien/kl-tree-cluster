"""Optional guard layers for gate annotation."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score

from tree_break_selection.core_utils.tree_utils import bottom_up_nodes
from tree_break_selection.hierarchy_analysis.statistics.contrast_covariance import (
    build_contrast_covariance,
    compute_whitened_wald_contrast,
)
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.fixed_subspace_annotation import (
    FIXED_SUBSPACE_SIBLING_GATE_METHODS,
    fixed_coordinate_bh_p_value,
    fixed_subspace_sibling_p_value,
)
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.pair_testing.collection.pair_observations import (
    identify_binary_sibling_children,
)
from tree_break_selection.tree.feature_space import FeatureSpace

from .annotation_predicates import (
    annotation_bool,
    node_sibling_gate_open,
    node_split_prerequisites,
)
from .profiles import GLOBAL_PASSTHROUGH_REFINED_REPLICATES


def validate_root_stability_guard_config(
    *,
    threshold: float | None,
    subsample_replicates: int,
    feature_fraction: float,
) -> None:
    replicates = int(subsample_replicates)
    if replicates < 0:
        raise ValueError("root_stability_subsample_replicates must be nonnegative.")
    fraction = float(feature_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("root_stability_feature_fraction must lie in (0, 1].")
    if threshold is None:
        return
    threshold_value = float(threshold)
    if not math.isfinite(threshold_value) or not -1.0 <= threshold_value <= 1.0:
        raise ValueError(
            f"root_stability_guard_threshold must be finite and lie in [-1, 1]; got {threshold!r}."
        )
    if replicates <= 0:
        raise ValueError(
            "root_stability_subsample_replicates must be positive when "
            "root_stability_guard_threshold is configured."
        )


def validate_root_selective_permutation_guard_config(
    *,
    replicates: int,
    alpha: float,
    scope: str = "root",
) -> None:
    count = int(replicates)
    if count < 0:
        raise ValueError("root_selective_permutation_guard_replicates must be nonnegative.")
    alpha_value = float(alpha)
    if not 0.0 < alpha_value < 1.0:
        raise ValueError(
            f"root_selective_permutation_guard_alpha must lie in (0, 1); got {alpha!r}."
        )
    allowed_scopes = {
        "root",
        "open_internal",
        "passthrough_descendant",
        "global_sibling_min_passthrough_descendant",
        "global_sibling_min_passthrough_descendant_refined",
    }
    if str(scope) not in allowed_scopes:
        raise ValueError(
            "root_selective_permutation_guard_scope must be 'root', "
            "'open_internal', 'passthrough_descendant', or "
            "'global_sibling_min_passthrough_descendant'/'..._refined'; "
            f"got {scope!r}."
        )


def _tree_root_node(tree):
    if hasattr(tree, "root"):
        return tree.root()
    if "root" in tree.graph:
        return tree.graph["root"]
    roots = [node for node, degree in tree.in_degree() if degree == 0]
    if len(roots) != 1:
        raise ValueError(f"Expected one root, got {roots!r}.")
    return roots[0]


def _descendant_leaf_label_sets(tree) -> dict[object, frozenset]:
    if hasattr(tree, "compute_descendant_sets"):
        return tree.compute_descendant_sets(use_labels=True)

    labels_by_node: dict[object, frozenset] = {}
    for node in tree.nodes:
        leaves: set[object] = set()
        for descendant in {node, *_nx_descendants(tree, node)}:
            if tree.out_degree(descendant) == 0:
                leaves.add(tree.nodes[descendant].get("label", descendant))
        labels_by_node[node] = frozenset(leaves)
    return labels_by_node


def _nx_descendants(tree, node) -> set[object]:
    frontier = list(tree.successors(node))
    seen: set[object] = set()
    while frontier:
        current = frontier.pop()
        if current in seen:
            continue
        seen.add(current)
        frontier.extend(tree.successors(current))
    return seen


def _tree_root_split_labels(tree, data_index: pd.Index) -> np.ndarray | None:
    root = _tree_root_node(tree)
    children = list(tree.successors(root))
    if len(children) != 2:
        return None

    descendant_sets = _descendant_leaf_label_sets(tree)
    left = set(descendant_sets[children[0]])
    right = set(descendant_sets[children[1]])

    labels: list[int] = []
    missing: list[object] = []
    for label in data_index:
        if label in left:
            labels.append(0)
        elif label in right:
            labels.append(1)
        else:
            missing.append(label)
    if missing:
        raise ValueError(
            "leaf_data index must match leaf labels under the supplied tree root; "
            f"missing={missing[:5]!r}."
        )
    values = np.asarray(labels, dtype=int)
    if np.unique(values).size < 2:
        return None
    return values


def _build_selected_linkage_tree(
    data: pd.DataFrame,
    *,
    distance_metric: str,
    linkage_method: str,
):
    from tree_break_selection.tree.construction import tree_from_linkage

    values = data.to_numpy(dtype=float)
    if values.ndim != 2:
        raise ValueError("leaf_data must be a 2D feature matrix.")
    if values.shape[0] < 2:
        raise ValueError("Selected-root permutation requires at least two leaves.")
    if not np.isfinite(values).all():
        raise ValueError("leaf_data contains non-finite values.")
    distances = pdist(values, metric=str(distance_metric))
    return tree_from_linkage(
        linkage(distances, method=str(linkage_method)),
        leaf_names=data.index.tolist(),
    )


def _fixed_root_sibling_p_value(
    tree,
    feature_space: FeatureSpace,
    *,
    method: str,
) -> float:
    root = _tree_root_node(tree)
    children = identify_binary_sibling_children(tree, root)
    if children is None:
        return 1.0
    left, right = children
    contrast = build_contrast_covariance(
        np.asarray(tree.nodes[left]["distribution"], dtype=float),
        np.asarray(tree.nodes[right]["distribution"], dtype=float),
        float(tree.nodes[left]["leaf_count"]),
        float(tree.nodes[right]["leaf_count"]),
        comparison="sibling",
        feature_space=feature_space,
    )
    return fixed_subspace_sibling_p_value(
        contrast.whitened_vector(),
        feature_space,
        method=method,  # type: ignore[arg-type]
    )


def _can_use_fast_discrete_coordinate_p_values(feature_space: FeatureSpace) -> bool:
    return feature_space.family_label in {"bernoulli", "categorical"}


def _fixed_discrete_coordinate_sibling_p_value(
    first: np.ndarray,
    second: np.ndarray,
    first_sample_size: float,
    second_sample_size: float,
    feature_space: FeatureSpace,
) -> float:
    z = compute_whitened_wald_contrast(
        np.asarray(first, dtype=float),
        np.asarray(second, dtype=float),
        float(first_sample_size),
        float(second_sample_size),
        comparison="sibling",
        feature_space=feature_space,
    )
    return fixed_coordinate_bh_p_value(z)


def _fixed_binary_sibling_p_values(
    tree,
    feature_space: FeatureSpace,
    *,
    method: str,
) -> list[float]:
    """Return fixed-subspace sibling p-values for every binary parent."""
    p_values: list[float] = []
    use_fast_discrete_coordinate = (
        method == "fixed_coordinate_bh"
        and _can_use_fast_discrete_coordinate_p_values(feature_space)
    )
    for parent in tree.nodes:
        children = identify_binary_sibling_children(tree, parent)
        if children is None:
            continue
        left, right = children
        if use_fast_discrete_coordinate:
            p_values.append(
                _fixed_discrete_coordinate_sibling_p_value(
                    np.asarray(tree.nodes[left]["distribution"], dtype=float),
                    np.asarray(tree.nodes[right]["distribution"], dtype=float),
                    float(tree.nodes[left]["leaf_count"]),
                    float(tree.nodes[right]["leaf_count"]),
                    feature_space,
                )
            )
            continue
        contrast = build_contrast_covariance(
            np.asarray(tree.nodes[left]["distribution"], dtype=float),
            np.asarray(tree.nodes[right]["distribution"], dtype=float),
            float(tree.nodes[left]["leaf_count"]),
            float(tree.nodes[right]["leaf_count"]),
            comparison="sibling",
            feature_space=feature_space,
        )
        p_values.append(
            fixed_subspace_sibling_p_value(
                contrast.whitened_vector(),
                feature_space,
                method=method,  # type: ignore[arg-type]
            )
        )
    return p_values


def _selected_tree_fixed_sibling_min_p_value(
    data: pd.DataFrame,
    feature_space: FeatureSpace,
    *,
    method: str,
    tree_distance_metric: str,
    tree_linkage_method: str,
) -> float:
    tree = _build_selected_linkage_tree(
        data,
        distance_metric=tree_distance_metric,
        linkage_method=tree_linkage_method,
    )
    tree.populate_node_divergences(data, feature_space=feature_space)
    p_values = _fixed_binary_sibling_p_values(
        tree,
        feature_space,
        method=method,
    )
    return float(min(p_values)) if p_values else 1.0


def _selected_root_fixed_sibling_p_value(
    data: pd.DataFrame,
    feature_space: FeatureSpace,
    *,
    method: str,
    tree_distance_metric: str,
    tree_linkage_method: str,
) -> float:
    tree = _build_selected_linkage_tree(
        data,
        distance_metric=tree_distance_metric,
        linkage_method=tree_linkage_method,
    )
    tree.populate_node_divergences(data, feature_space=feature_space)
    return _fixed_root_sibling_p_value(
        tree,
        feature_space,
        method=method,
    )


def _block_permutation_null_sample(
    leaf_data: pd.DataFrame,
    feature_space: FeatureSpace,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Return a feature/block permutation sample preserving block margins."""
    source = leaf_data.to_numpy(dtype=int)
    out = np.zeros_like(source)
    n_rows = int(source.shape[0])
    for block in feature_space.blocks:
        indices = np.asarray(block.column_indices, dtype=int)
        if block.family == "categorical":
            categories = np.argmax(source[:, indices], axis=1)
            permuted = rng.permutation(categories)
            out[np.arange(n_rows), indices[permuted]] = 1
            continue
        if block.family == "bernoulli":
            out[:, indices[0]] = rng.permutation(source[:, indices[0]])
            continue
        raise ValueError(
            "Root selective permutation supports Bernoulli and categorical "
            f"feature blocks only; got {block.family!r}."
        )
    return pd.DataFrame(
        out,
        index=leaf_data.index,
        columns=leaf_data.columns,
    )


def _selected_permutation_p_value(
    leaf_data: pd.DataFrame,
    feature_space: FeatureSpace,
    *,
    method: str,
    bootstrap_replicates: int,
    seed: int,
    tree_distance_metric: str = "hamming",
    tree_linkage_method: str = "average",
    observed_p_value: float | None = None,
    statistic: Callable[..., float],
    guard_name: str,
) -> dict[str, float]:
    """Estimate a selected-tree p-value with one shared permutation engine."""
    if method not in FIXED_SUBSPACE_SIBLING_GATE_METHODS:
        raise ValueError(
            f"{guard_name} requires a fixed-subspace sibling gate; got method={method!r}."
        )
    count = int(bootstrap_replicates)
    if count < 0:
        raise ValueError("bootstrap_replicates must be nonnegative.")
    observed = (
        float(observed_p_value)
        if observed_p_value is not None and np.isfinite(float(observed_p_value))
        else statistic(
            leaf_data,
            feature_space,
            method=method,
            tree_distance_metric=tree_distance_metric,
            tree_linkage_method=tree_linkage_method,
        )
    )
    if count <= 0:
        return {
            "root_observed_p_value": float(observed),
            "root_selective_p_value": np.nan,
            "root_selective_null_min_p_value": np.nan,
            "root_selective_null_q05_p_value": np.nan,
        }

    rng = np.random.default_rng(int(seed))
    null_p_values: list[float] = []
    for _ in range(count):
        null_sample = _block_permutation_null_sample(leaf_data, feature_space, rng)
        null_p_values.append(
            statistic(
                null_sample,
                feature_space,
                method=method,
                tree_distance_metric=tree_distance_metric,
                tree_linkage_method=tree_linkage_method,
            )
        )
    null = np.asarray(null_p_values, dtype=float)
    selected = (1.0 + float(np.sum(null <= observed))) / (float(null.size) + 1.0)
    return {
        "root_observed_p_value": float(observed),
        "root_selective_p_value": float(selected),
        "root_selective_null_min_p_value": float(np.min(null)),
        "root_selective_null_q05_p_value": float(np.quantile(null, 0.05)),
    }


def selected_root_permutation_p_value(
    leaf_data: pd.DataFrame,
    feature_space: FeatureSpace,
    *,
    method: str,
    bootstrap_replicates: int,
    seed: int,
    tree_distance_metric: str = "hamming",
    tree_linkage_method: str = "average",
    observed_p_value: float | None = None,
) -> dict[str, float]:
    """Estimate the selected-root p-value by feature/block permutation."""
    return _selected_permutation_p_value(
        leaf_data,
        feature_space,
        method=method,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
        tree_distance_metric=tree_distance_metric,
        tree_linkage_method=tree_linkage_method,
        observed_p_value=observed_p_value,
        statistic=_selected_root_fixed_sibling_p_value,
        guard_name="Selected-root permutation",
    )


def selected_global_sibling_min_permutation_p_value(
    leaf_data: pd.DataFrame,
    feature_space: FeatureSpace,
    *,
    method: str,
    bootstrap_replicates: int,
    seed: int,
    tree_distance_metric: str = "hamming",
    tree_linkage_method: str = "average",
    observed_p_value: float | None = None,
) -> dict[str, float]:
    """Estimate a whole selected-family minimum sibling p-value.

    This diagnostic null compares an observed selected sibling p-value against
    the minimum fixed-subspace sibling p-value over every binary parent in each
    reselected null tree. It is a conservative global-family correction for
    pass-through descendant searches.
    """
    return _selected_permutation_p_value(
        leaf_data,
        feature_space,
        method=method,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
        tree_distance_metric=tree_distance_metric,
        tree_linkage_method=tree_linkage_method,
        observed_p_value=observed_p_value,
        statistic=_selected_tree_fixed_sibling_min_p_value,
        guard_name="Selected-family permutation",
    )


def _selected_linkage_root_split_labels(
    data: pd.DataFrame,
    *,
    distance_metric: str,
    linkage_method: str,
) -> np.ndarray | None:
    values = data.to_numpy(dtype=float)
    if values.ndim != 2:
        raise ValueError("leaf_data must be a 2D feature matrix.")
    n_rows = int(values.shape[0])
    if n_rows < 2:
        return None
    if not np.isfinite(values).all():
        raise ValueError("leaf_data contains non-finite values.")

    distances = pdist(values, metric=str(distance_metric))
    linkage_matrix = linkage(distances, method=str(linkage_method))
    clusters: dict[int, set[int]] = {index: {index} for index in range(n_rows)}
    for step_index, row in enumerate(linkage_matrix):
        left_id = int(row[0])
        right_id = int(row[1])
        clusters[n_rows + step_index] = clusters[left_id] | clusters[right_id]

    left_child = int(linkage_matrix[-1, 0])
    right_child = int(linkage_matrix[-1, 1])
    labels = np.zeros(n_rows, dtype=int)
    labels[list(clusters[right_child])] = 1
    if not clusters[left_child] or not clusters[right_child]:
        return None
    return labels


def compute_root_feature_subsample_stability(
    tree,
    leaf_data: pd.DataFrame,
    feature_space: FeatureSpace,
    *,
    subsample_replicates: int,
    feature_fraction: float,
    seed: int,
    tree_distance_metric: str = "hamming",
    tree_linkage_method: str = "average",
) -> dict[str, float]:
    """Measure supplied-root stability under deterministic feature-block subsampling."""
    validate_root_stability_guard_config(
        threshold=None,
        subsample_replicates=subsample_replicates,
        feature_fraction=feature_fraction,
    )
    if int(subsample_replicates) <= 0:
        return {
            "root_stability_subsample_mean_ari": np.nan,
            "root_stability_subsample_median_ari": np.nan,
            "root_stability_subsample_q10_ari": np.nan,
        }

    original = _tree_root_split_labels(tree, leaf_data.index)
    if original is None:
        return {
            "root_stability_subsample_mean_ari": np.nan,
            "root_stability_subsample_median_ari": np.nan,
            "root_stability_subsample_q10_ari": np.nan,
        }

    blocks = tuple(tuple(block.column_indices) for block in feature_space.blocks)
    if not blocks:
        raise ValueError("Feature-space blocks are required for root stability.")
    n_blocks = len(blocks)
    n_selected = max(1, int(round(float(feature_fraction) * n_blocks)))
    rng = np.random.default_rng(int(seed))

    scores: list[float] = []
    for _ in range(int(subsample_replicates)):
        selected_blocks = rng.choice(n_blocks, size=n_selected, replace=False)
        columns = [column for block_index in selected_blocks for column in blocks[int(block_index)]]
        subsampled = leaf_data.iloc[:, columns]
        selected_labels = _selected_linkage_root_split_labels(
            subsampled,
            distance_metric=str(tree_distance_metric),
            linkage_method=str(tree_linkage_method),
        )
        if selected_labels is None:
            continue
        scores.append(float(adjusted_rand_score(original, selected_labels)))

    if not scores:
        return {
            "root_stability_subsample_mean_ari": np.nan,
            "root_stability_subsample_median_ari": np.nan,
            "root_stability_subsample_q10_ari": np.nan,
        }
    values = np.asarray(scores, dtype=float)
    return {
        "root_stability_subsample_mean_ari": float(np.mean(values)),
        "root_stability_subsample_median_ari": float(np.median(values)),
        "root_stability_subsample_q10_ari": float(np.quantile(values, 0.10)),
    }


def apply_root_stability_guard(
    tree,
    annotations_df: pd.DataFrame,
    root_stability: Mapping[str, float],
    *,
    threshold: float,
) -> pd.DataFrame:
    """Close an unstable root sibling gate without changing non-root decisions."""
    threshold_value = float(threshold)
    out = annotations_df.copy()
    for column in (
        "Root_Stability_Subsample_Mean_ARI",
        "Root_Stability_Subsample_Median_ARI",
        "Root_Stability_Subsample_Q10_ARI",
        "Root_Stability_Guard_Threshold",
    ):
        out[column] = np.nan
    out["Root_Stability_Guard_Blocked"] = False

    root = _tree_root_node(tree)
    if root not in out.index:
        return out

    mean_ari = float(root_stability.get("root_stability_subsample_mean_ari", np.nan))
    out.loc[root, "Root_Stability_Subsample_Mean_ARI"] = mean_ari
    out.loc[root, "Root_Stability_Subsample_Median_ARI"] = root_stability.get(
        "root_stability_subsample_median_ari",
        np.nan,
    )
    out.loc[root, "Root_Stability_Subsample_Q10_ARI"] = root_stability.get(
        "root_stability_subsample_q10_ari",
        np.nan,
    )
    out.loc[root, "Root_Stability_Guard_Threshold"] = threshold_value

    root_gate_open = bool(out.loc[root, "Sibling_BH_Different"])
    should_block = bool(np.isfinite(mean_ari) and mean_ari < threshold_value)
    if root_gate_open and should_block:
        out.loc[root, "Sibling_BH_Different"] = False
        out.loc[root, "Sibling_BH_Same"] = True
        out.loc[root, "Root_Stability_Guard_Blocked"] = True
    return out


def _root_stability_blocked(
    annotations_df: pd.DataFrame,
    node: object,
    root: object,
) -> bool:
    return node == root and annotation_bool(
        annotations_df,
        node,
        "Root_Stability_Guard_Blocked",
    )


def _selective_guard_root_candidate(
    annotations_df: pd.DataFrame,
    node: object,
    root: object,
) -> bool:
    return node_sibling_gate_open(annotations_df, node) or _root_stability_blocked(
        annotations_df,
        node,
        root,
    )


def _node_closed_by_explicit_guard(
    annotations_df: pd.DataFrame,
    node: object,
) -> bool:
    return any(
        annotation_bool(annotations_df, node, column)
        for column in (
            "Root_Stability_Guard_Blocked",
            "Root_Selective_Permutation_Guard_Blocked",
            "Selective_Permutation_Guard_Blocked",
        )
    )


def _passthrough_descendant_guard_candidates(
    tree,
    annotations_df: pd.DataFrame,
    *,
    root: object,
) -> list[object]:
    """Return open split nodes reachable only through a closed sibling ancestor."""
    node_ids = tuple(tree.nodes)
    children = {node: list(tree.successors(node)) for node in node_ids}
    split_prerequisites = {
        node: node_split_prerequisites(tree, annotations_df, node) for node in node_ids
    }
    sibling_open = {node: node_sibling_gate_open(annotations_df, node) for node in node_ids}
    can_split = {node: bool(split_prerequisites[node] and sibling_open[node]) for node in node_ids}
    has_descendant_split: dict[object, bool] = {}
    for node in bottom_up_nodes(tree):
        has_descendant_split[node] = any(
            can_split.get(child, False) or has_descendant_split.get(child, False)
            for child in children[node]
        )

    closed_passthrough_ancestors = {
        node
        for node in node_ids
        if split_prerequisites[node]
        and not can_split[node]
        and has_descendant_split.get(node, False)
        and not _node_closed_by_explicit_guard(annotations_df, node)
    }
    passthrough_reachable: dict[object, bool] = {}
    stack: list[tuple[object, bool]] = [(root, False)]
    while stack:
        node, ancestor_passthrough = stack.pop()
        passthrough_reachable[node] = bool(ancestor_passthrough)
        child_passthrough = bool(ancestor_passthrough or node in closed_passthrough_ancestors)
        for child in children[node]:
            stack.append((child, child_passthrough))

    return [
        node
        for node in annotations_df.index
        if node != root and can_split.get(node, False) and passthrough_reachable.get(node, False)
    ]


def apply_root_selective_permutation_guard(
    tree,
    annotations_df: pd.DataFrame,
    leaf_data: pd.DataFrame,
    feature_space: FeatureSpace,
    *,
    method: str,
    bootstrap_replicates: int,
    seed: int,
    alpha: float,
    scope: str = "root",
    tree_distance_metric: str = "hamming",
    tree_linkage_method: str = "average",
) -> pd.DataFrame:
    """Close selected-root or selected-subtree splits failing permutation evidence."""
    validate_root_selective_permutation_guard_config(
        replicates=bootstrap_replicates,
        alpha=alpha,
        scope=scope,
    )
    out = annotations_df.copy()
    root_columns = (
        "Root_Selective_Permutation_Observed_P_Value",
        "Root_Selective_Permutation_P_Value",
        "Root_Selective_Permutation_Null_Min_P_Value",
        "Root_Selective_Permutation_Null_Q05_P_Value",
        "Root_Selective_Permutation_Guard_Alpha",
        "Root_Selective_Permutation_Guard_Replicates",
        "Root_Selective_Permutation_Guard_Seed",
    )
    generic_columns = (
        "Selective_Permutation_Observed_P_Value",
        "Selective_Permutation_P_Value",
        "Selective_Permutation_Null_Min_P_Value",
        "Selective_Permutation_Null_Q05_P_Value",
        "Selective_Permutation_Guard_Alpha",
        "Selective_Permutation_Guard_Replicates",
        "Selective_Permutation_Guard_Seed",
        "Selective_Permutation_Guard_Scope",
    )
    for column in (*root_columns, *generic_columns):
        out[column] = np.nan
    out["Selective_Permutation_Guard_Scope"] = ""
    out["Selective_Permutation_Base_P_Value"] = np.nan
    out["Selective_Permutation_Guard_Refined"] = False
    out["Root_Selective_Permutation_Guard_Would_Block"] = False
    out["Root_Selective_Permutation_Guard_Blocked"] = False
    out["Selective_Permutation_Guard_Would_Block"] = False
    out["Selective_Permutation_Guard_Blocked"] = False

    root = _tree_root_node(tree)
    if root not in out.index:
        return out

    descendant_sets = _descendant_leaf_label_sets(tree)
    scope_value = str(scope)
    evaluated_nodes: set[object] = set()
    evaluation_index = 0

    def evaluate_node(node: object, seed_offset: int) -> bool:
        if node not in descendant_sets or len(descendant_sets[node]) < 2:
            return False
        descendant_labels = descendant_sets[node]
        labels = [label for label in leaf_data.index if label in descendant_labels]
        subset = leaf_data.loc[labels]
        observed = (
            float(out.loc[node, "Sibling_Divergence_P_Value"])
            if "Sibling_Divergence_P_Value" in out.columns
            and pd.notna(out.loc[node, "Sibling_Divergence_P_Value"])
            else None
        )
        out.loc[node, "Selective_Permutation_Guard_Alpha"] = float(alpha)
        out.loc[node, "Selective_Permutation_Guard_Replicates"] = int(bootstrap_replicates)
        out.loc[node, "Selective_Permutation_Guard_Seed"] = int(seed) + seed_offset
        out.loc[node, "Selective_Permutation_Guard_Scope"] = scope_value
        result = selected_root_permutation_p_value(
            subset,
            feature_space,
            method=method,
            bootstrap_replicates=int(bootstrap_replicates),
            seed=int(seed) + seed_offset,
            tree_distance_metric=tree_distance_metric,
            tree_linkage_method=tree_linkage_method,
            observed_p_value=observed,
        )
        out.loc[node, "Selective_Permutation_Observed_P_Value"] = result["root_observed_p_value"]
        out.loc[node, "Selective_Permutation_P_Value"] = result["root_selective_p_value"]
        out.loc[node, "Selective_Permutation_Null_Min_P_Value"] = result[
            "root_selective_null_min_p_value"
        ]
        out.loc[node, "Selective_Permutation_Null_Q05_P_Value"] = result[
            "root_selective_null_q05_p_value"
        ]
        selected_p = result["root_selective_p_value"]
        would_block = bool(np.isfinite(selected_p) and selected_p > float(alpha))
        out.loc[node, "Selective_Permutation_Guard_Would_Block"] = would_block
        if bool(out.loc[node, "Sibling_BH_Different"]) and would_block:
            out.loc[node, "Sibling_BH_Different"] = False
            out.loc[node, "Sibling_BH_Same"] = True
            out.loc[node, "Selective_Permutation_Guard_Blocked"] = True
        if node == root:
            out.loc[root, "Root_Selective_Permutation_Observed_P_Value"] = result[
                "root_observed_p_value"
            ]
            out.loc[root, "Root_Selective_Permutation_P_Value"] = result["root_selective_p_value"]
            out.loc[root, "Root_Selective_Permutation_Null_Min_P_Value"] = result[
                "root_selective_null_min_p_value"
            ]
            out.loc[root, "Root_Selective_Permutation_Null_Q05_P_Value"] = result[
                "root_selective_null_q05_p_value"
            ]
            out.loc[root, "Root_Selective_Permutation_Guard_Alpha"] = float(alpha)
            out.loc[root, "Root_Selective_Permutation_Guard_Replicates"] = int(bootstrap_replicates)
            out.loc[root, "Root_Selective_Permutation_Guard_Seed"] = int(seed) + seed_offset
            out.loc[root, "Root_Selective_Permutation_Guard_Would_Block"] = would_block
            out.loc[root, "Root_Selective_Permutation_Guard_Blocked"] = bool(
                out.loc[root, "Selective_Permutation_Guard_Blocked"]
            )
        return True

    def evaluate_global_pass_through_nodes(
        nodes: list[object],
        seed_offset: int,
    ) -> bool:
        observed_values = [
            float(out.loc[node, "Sibling_Divergence_P_Value"])
            for node in nodes
            if node in out.index
            and "Sibling_Divergence_P_Value" in out.columns
            and pd.notna(out.loc[node, "Sibling_Divergence_P_Value"])
        ]
        if not observed_values:
            return False
        observed = float(min(observed_values))
        result = selected_global_sibling_min_permutation_p_value(
            leaf_data,
            feature_space,
            method=method,
            bootstrap_replicates=int(bootstrap_replicates),
            seed=int(seed) + seed_offset,
            tree_distance_metric=tree_distance_metric,
            tree_linkage_method=tree_linkage_method,
            observed_p_value=observed,
        )
        base_selected_p = result["root_selective_p_value"]
        used_replicates = int(bootstrap_replicates)
        refined = False
        base_floor = 1.0 / (float(bootstrap_replicates) + 1.0)
        if (
            scope_value == "global_sibling_min_passthrough_descendant_refined"
            and np.isfinite(base_selected_p)
            and float(base_selected_p) <= base_floor + 1e-12
            and int(bootstrap_replicates) < GLOBAL_PASSTHROUGH_REFINED_REPLICATES
        ):
            result = selected_global_sibling_min_permutation_p_value(
                leaf_data,
                feature_space,
                method=method,
                bootstrap_replicates=GLOBAL_PASSTHROUGH_REFINED_REPLICATES,
                seed=int(seed) + seed_offset,
                tree_distance_metric=tree_distance_metric,
                tree_linkage_method=tree_linkage_method,
                observed_p_value=observed,
            )
            used_replicates = GLOBAL_PASSTHROUGH_REFINED_REPLICATES
            refined = True
        selected_p = result["root_selective_p_value"]
        would_block = bool(np.isfinite(selected_p) and selected_p > float(alpha))
        for node in nodes:
            out.loc[node, "Selective_Permutation_Observed_P_Value"] = result[
                "root_observed_p_value"
            ]
            out.loc[node, "Selective_Permutation_Base_P_Value"] = base_selected_p
            out.loc[node, "Selective_Permutation_P_Value"] = selected_p
            out.loc[node, "Selective_Permutation_Null_Min_P_Value"] = result[
                "root_selective_null_min_p_value"
            ]
            out.loc[node, "Selective_Permutation_Null_Q05_P_Value"] = result[
                "root_selective_null_q05_p_value"
            ]
            out.loc[node, "Selective_Permutation_Guard_Alpha"] = float(alpha)
            out.loc[node, "Selective_Permutation_Guard_Replicates"] = int(used_replicates)
            out.loc[node, "Selective_Permutation_Guard_Seed"] = int(seed) + seed_offset
            out.loc[node, "Selective_Permutation_Guard_Scope"] = scope_value
            out.loc[node, "Selective_Permutation_Guard_Refined"] = refined
            out.loc[node, "Selective_Permutation_Guard_Would_Block"] = would_block
            if bool(out.loc[node, "Sibling_BH_Different"]) and would_block:
                out.loc[node, "Sibling_BH_Different"] = False
                out.loc[node, "Sibling_BH_Same"] = True
                out.loc[node, "Selective_Permutation_Guard_Blocked"] = True
        return True

    if scope_value == "root":
        candidate_nodes = [root] if _selective_guard_root_candidate(out, root, root) else []
    elif scope_value == "open_internal":
        candidate_nodes = [
            node
            for node in out.index
            if node in descendant_sets
            and len(descendant_sets[node]) >= 2
            and (node_sibling_gate_open(out, node) or _root_stability_blocked(out, node, root))
        ]
    else:
        candidate_nodes = []
        if _selective_guard_root_candidate(out, root, root):
            candidate_nodes.append(root)

    for node in candidate_nodes:
        if evaluate_node(node, evaluation_index):
            evaluated_nodes.add(node)
            evaluation_index += 1

    if scope_value in {
        "passthrough_descendant",
        "global_sibling_min_passthrough_descendant",
        "global_sibling_min_passthrough_descendant_refined",
    }:
        pass_through_nodes = _passthrough_descendant_guard_candidates(
            tree,
            out,
            root=root,
        )
    else:
        pass_through_nodes = []

    if scope_value == "passthrough_descendant":
        for node in pass_through_nodes:
            if node in evaluated_nodes:
                continue
            if node not in descendant_sets or len(descendant_sets[node]) < 2:
                continue
            if evaluate_node(node, evaluation_index):
                evaluated_nodes.add(node)
                evaluation_index += 1
    elif scope_value in {
        "global_sibling_min_passthrough_descendant",
        "global_sibling_min_passthrough_descendant_refined",
    }:
        global_nodes = [
            node
            for node in pass_through_nodes
            if node not in evaluated_nodes
            and node in descendant_sets
            and len(descendant_sets[node]) >= 2
        ]
        if evaluate_global_pass_through_nodes(global_nodes, evaluation_index):
            evaluated_nodes.update(global_nodes)
    return out


__all__ = [
    "apply_root_selective_permutation_guard",
    "apply_root_stability_guard",
    "compute_root_feature_subsample_stability",
    "selected_global_sibling_min_permutation_p_value",
    "selected_root_permutation_p_value",
    "validate_root_selective_permutation_guard_config",
    "validate_root_stability_guard_config",
]
