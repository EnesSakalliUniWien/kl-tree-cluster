"""Classification logic for the Gate 2 single-feature subtree policy."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.mixture import GaussianMixture


def _find_low_variance_ratio_threshold(
    variance_ratios: list[float],
) -> tuple[bool, float | None]:
    valid_ratios = [float(r) for r in variance_ratios if np.isfinite(r) and r > 0]
    if len(valid_ratios) < 2:
        return False, None

    log_ratios = np.log(np.asarray(valid_ratios, dtype=np.float64)).reshape(-1, 1)
    one_component = GaussianMixture(n_components=1, random_state=0)
    two_component = GaussianMixture(n_components=2, random_state=0)
    one_component.fit(log_ratios)
    two_component.fit(log_ratios)

    if two_component.bic(log_ratios) + 1e-9 >= one_component.bic(log_ratios):
        return False, None

    means = np.asarray(two_component.means_, dtype=np.float64).ravel()
    low_label = int(np.argmin(means))
    labels = two_component.predict(log_ratios)
    low_values = np.exp(log_ratios[labels == low_label].ravel())
    high_values = np.exp(log_ratios[labels != low_label].ravel())
    if low_values.size == 0 or high_values.size == 0:
        return False, None

    low_max = float(np.max(low_values))
    high_min = float(np.min(high_values))
    if low_max < high_min:
        threshold = float(np.sqrt(low_max * high_min))
    else:
        threshold = float(np.exp(np.mean(np.sort(means))))
    return True, threshold


def _classify_low_information_subtrees(
    single_feature_candidates: list[dict[str, Any]],
) -> tuple[bool, float | None, set[str]]:
    """Return the fitted threshold and node ids for low-information subtrees."""
    has_low_group, low_ratio_threshold = _find_low_variance_ratio_threshold(
        [
            candidate["variance_ratio"]
            for candidate in single_feature_candidates
            if candidate["variance_ratio"] is not None
        ]
    )
    if not has_low_group or low_ratio_threshold is None:
        return has_low_group, low_ratio_threshold, set()

    low_node_ids = {
        str(candidate["node_id"])
        for candidate in single_feature_candidates
        if candidate["variance_ratio"] is not None
        and float(candidate["variance_ratio"]) <= float(low_ratio_threshold)
    }
    return has_low_group, low_ratio_threshold, low_node_ids


def _summarize_single_feature_candidate_groups(
    single_feature_candidates: list[dict[str, Any]],
    low_node_ids: set[str],
) -> dict[str, Any]:
    """Summarize low-information and retained candidate groups."""
    low_candidates = [
        candidate
        for candidate in single_feature_candidates
        if str(candidate["node_id"]) in low_node_ids
    ]
    high_candidates = [
        candidate
        for candidate in single_feature_candidates
        if str(candidate["node_id"]) not in low_node_ids
    ]

    low_count = int(len(low_candidates))
    high_count = int(len(high_candidates))
    low_rows = int(sum(int(candidate["n_rows"]) for candidate in low_candidates))
    high_rows = int(sum(int(candidate["n_rows"]) for candidate in high_candidates))
    low_internal_rows = int(sum(int(candidate["internal_rows"]) for candidate in low_candidates))
    dangerous_tree = bool(low_count > high_count and low_rows > high_rows and low_internal_rows > 0)

    blocked_node_id_set = set(low_node_ids) if dangerous_tree else set()
    allowed_node_ids = sorted(
        str(candidate["node_id"])
        for candidate in single_feature_candidates
        if str(candidate["node_id"]) not in blocked_node_id_set
    )
    return {
        "low_count": low_count,
        "high_count": high_count,
        "low_rows": low_rows,
        "high_rows": high_rows,
        "low_internal_rows": low_internal_rows,
        "dangerous_tree": dangerous_tree,
        "blocked_node_id_set": blocked_node_id_set,
        "blocked_node_ids": sorted(blocked_node_id_set),
        "allowed_node_ids": allowed_node_ids,
        "allowed_node_id_set": set(allowed_node_ids),
    }
