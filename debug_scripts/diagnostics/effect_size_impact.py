"""Diagnose the impact of effect-size gates on the KL pipeline.

For every benchmark case, this script:
1. Runs the standard KL decomposition.
2. At each internal node, computes two effect-size metrics:
   - Edge effect:    mean |p_child - p_parent|  (per child)
   - Sibling effect: mean |p_left  - p_right|   (per parent)
3. Records whether the node was SPLIT or MERGED by the current pipeline.
4. Simulates what would happen with various effect-size thresholds.

Output:
- Per-node CSV with effect sizes and gate decisions
- Summary table: for each threshold, how K and ARI change across all cases
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import logging
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.generators import generate_case_data
from kl_clustering_analysis import config
from kl_clustering_analysis.tree.poset_tree import PosetTree

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

# ---------------------------------------------------------------------------
# Effect-size computation
# ---------------------------------------------------------------------------


def compute_edge_effect(tree, child_id: str, parent_id: str) -> float:
    """Mean |p_child - p_parent| across all features."""
    child_dist = tree.nodes[child_id].get("distribution")
    parent_dist = tree.nodes[parent_id].get("distribution")
    if child_dist is None or parent_dist is None:
        return np.nan
    c = np.asarray(child_dist, dtype=float)
    p = np.asarray(parent_dist, dtype=float)
    return float(np.mean(np.abs(c - p)))


def compute_sibling_effect(tree, left_id: str, right_id: str) -> float:
    """Mean |p_left - p_right| across all features."""
    left_dist = tree.nodes[left_id].get("distribution")
    right_dist = tree.nodes[right_id].get("distribution")
    if left_dist is None or right_dist is None:
        return np.nan
    l = np.asarray(left_dist, dtype=float)
    r = np.asarray(right_dist, dtype=float)
    return float(np.mean(np.abs(l - r)))


def compute_max_sibling_effect(tree, left_id: str, right_id: str) -> float:
    """Max |p_left - p_right| across features."""
    left_dist = tree.nodes[left_id].get("distribution")
    right_dist = tree.nodes[right_id].get("distribution")
    if left_dist is None or right_dist is None:
        return np.nan
    l = np.asarray(left_dist, dtype=float)
    r = np.asarray(right_dist, dtype=float)
    return float(np.max(np.abs(l - r)))


# ---------------------------------------------------------------------------
# Collect per-node information from a decomposed tree
# ---------------------------------------------------------------------------


@dataclass
class NodeRecord:
    """One record per internal (non-leaf) node."""

    case_name: str
    node_id: str
    n_leaves: int
    gate_1_2_pass: bool
    gate_3_pass: bool
    decision: str  # "SPLIT", "MERGE", "PASSTHROUGH", "GATE12_FAIL"
    edge_effect_left: float
    edge_effect_right: float
    sibling_effect_mean: float
    sibling_effect_max: float
    # From annotations
    edge_p_left: float
    edge_p_right: float
    sibling_p: float
    sibling_skipped: bool


def collect_node_records(
    tree: PosetTree,
    annotations_df: pd.DataFrame,
    case_name: str,
) -> List[NodeRecord]:
    """Walk the tree and collect effect-size + gate info for every internal node."""
    root = next(n for n, deg in tree.in_degree() if deg == 0)
    children_map = {n: list(tree.successors(n)) for n in tree.nodes()}
    leaf_partition = tree.compute_descendant_sets(use_labels=True)

    records = []
    for node_id in tree.nodes():
        children = children_map[node_id]
        if len(children) == 0:
            continue  # leaf

        n_leaves = len(leaf_partition.get(node_id, set()))

        # Gate 1+2 evaluation
        if len(children) != 2:
            records.append(
                NodeRecord(
                    case_name=case_name,
                    node_id=node_id,
                    n_leaves=n_leaves,
                    gate_1_2_pass=False,
                    gate_3_pass=False,
                    decision="GATE12_FAIL",
                    edge_effect_left=np.nan,
                    edge_effect_right=np.nan,
                    sibling_effect_mean=np.nan,
                    sibling_effect_max=np.nan,
                    edge_p_left=np.nan,
                    edge_p_right=np.nan,
                    sibling_p=np.nan,
                    sibling_skipped=True,
                )
            )
            continue

        left, right = children

        # Edge effects
        parent_ids = [p for p in tree.predecessors(node_id)]
        parent_id = parent_ids[0] if parent_ids else None

        edge_eff_left = compute_edge_effect(tree, left, node_id)
        edge_eff_right = compute_edge_effect(tree, right, node_id)

        # Sibling effect
        sib_mean = compute_sibling_effect(tree, left, right)
        sib_max = compute_max_sibling_effect(tree, left, right)

        # Gate decisions from annotations_df
        left_sig = False
        right_sig = False
        edge_p_left = np.nan
        edge_p_right = np.nan
        if left in annotations_df.index:
            left_sig = bool(annotations_df.loc[left, "Child_Parent_Divergence_Significant"])
            edge_p_left = float(annotations_df.loc[left, "Child_Parent_Divergence_P_Value_BH"])
        if right in annotations_df.index:
            right_sig = bool(annotations_df.loc[right, "Child_Parent_Divergence_Significant"])
            edge_p_right = float(annotations_df.loc[right, "Child_Parent_Divergence_P_Value_BH"])

        gate_12_pass = left_sig or right_sig

        # Gate 3
        sibling_diff = False
        sibling_skip = True
        sibling_p = np.nan
        if node_id in annotations_df.index:
            sibling_diff = bool(annotations_df.loc[node_id, "Sibling_BH_Different"])
            sibling_skip = bool(annotations_df.loc[node_id, "Sibling_Divergence_Skipped"])
            p_col = "Sibling_Divergence_P_Value_Corrected"
            if p_col in annotations_df.columns:
                sibling_p = float(annotations_df.loc[node_id, p_col])

        gate_3_pass = gate_12_pass and sibling_diff and not sibling_skip

        if gate_12_pass and gate_3_pass:
            decision = "SPLIT"
        elif not gate_12_pass:
            decision = "GATE12_FAIL"
        else:
            decision = "MERGE"

        records.append(
            NodeRecord(
                case_name=case_name,
                node_id=node_id,
                n_leaves=n_leaves,
                gate_1_2_pass=gate_12_pass,
                gate_3_pass=gate_3_pass,
                decision=decision,
                edge_effect_left=edge_eff_left,
                edge_effect_right=edge_eff_right,
                sibling_effect_mean=sib_mean,
                sibling_effect_max=sib_max,
                edge_p_left=edge_p_left,
                edge_p_right=edge_p_right,
                sibling_p=sibling_p,
                sibling_skipped=sibling_skip,
            )
        )

    return records


# ---------------------------------------------------------------------------
# Simulate effect-size gate: re-run traversal with threshold filtering
# ---------------------------------------------------------------------------


def simulate_with_threshold(
    tree: PosetTree,
    annotations_df: pd.DataFrame,
    sibling_min_effect: float,
    edge_min_effect: float | None = None,
) -> int:
    """Re-run the DFS traversal with effect-size thresholds and return K.

    Parameters
    ----------
    sibling_min_effect
        Minimum mean |p_L - p_R| to allow a split at Gate 3.
    edge_min_effect
        Minimum mean |p_child - p_parent| to allow a split at Gate 2.
        If None, no edge-effect filtering (only sibling).
    """
    root = next(n for n, deg in tree.in_degree() if deg == 0)
    children_map = {n: list(tree.successors(n)) for n in tree.nodes()}
    leaf_partition = tree.compute_descendant_sets(use_labels=True)

    from kl_clustering_analysis.core_utils.data_utils import extract_bool_column_dict

    local_significant = extract_bool_column_dict(annotations_df, "Child_Parent_Divergence_Significant")
    sibling_different = extract_bool_column_dict(annotations_df, "Sibling_BH_Different")
    sibling_skipped = extract_bool_column_dict(annotations_df, "Sibling_Divergence_Skipped")

    nodes_to_visit = [root]
    cluster_count = 0
    processed = set()

    while nodes_to_visit:
        node_id = nodes_to_visit.pop()
        if node_id in processed:
            continue
        processed.add(node_id)

        children = children_map[node_id]
        if len(children) == 0:
            # leaf → single-sample cluster
            cluster_count += 1
            continue

        if len(children) != 2:
            cluster_count += 1
            continue

        left, right = children

        # Gate 1+2: significance
        left_sig = local_significant.get(left, False)
        right_sig = local_significant.get(right, False)
        gate_12 = left_sig or right_sig

        # Optional edge-effect gate
        if gate_12 and edge_min_effect is not None:
            eff_l = compute_edge_effect(tree, left, node_id)
            eff_r = compute_edge_effect(tree, right, node_id)
            max_edge_eff = max(eff_l, eff_r)
            if max_edge_eff < edge_min_effect:
                gate_12 = False

        if not gate_12:
            cluster_count += 1
            continue

        # Gate 3: sibling significance
        is_diff = sibling_different.get(node_id, False)
        is_skip = sibling_skipped.get(node_id, False)
        gate_3 = is_diff and not is_skip

        # Sibling effect gate
        if gate_3 and sibling_min_effect > 0:
            sib_eff = compute_sibling_effect(tree, left, right)
            if np.isnan(sib_eff) or sib_eff < sibling_min_effect:
                gate_3 = False

        if gate_3:
            nodes_to_visit.append(right)
            nodes_to_visit.append(left)
        else:
            cluster_count += 1

    return cluster_count


def labels_from_simulation(
    tree: PosetTree,
    annotations_df: pd.DataFrame,
    sample_names: list[str],
    sibling_min_effect: float,
    edge_min_effect: float | None = None,
) -> np.ndarray:
    """Full label assignment under simulated effect-size thresholds."""
    root = next(n for n, deg in tree.in_degree() if deg == 0)
    children_map = {n: list(tree.successors(n)) for n in tree.nodes()}
    leaf_partition = tree.compute_descendant_sets(use_labels=True)

    from kl_clustering_analysis.core_utils.data_utils import extract_bool_column_dict

    local_significant = extract_bool_column_dict(annotations_df, "Child_Parent_Divergence_Significant")
    sibling_different = extract_bool_column_dict(annotations_df, "Sibling_BH_Different")
    sibling_skipped = extract_bool_column_dict(annotations_df, "Sibling_Divergence_Skipped")

    nodes_to_visit = [root]
    cluster_leaf_sets: List[set] = []
    processed = set()

    while nodes_to_visit:
        node_id = nodes_to_visit.pop()
        if node_id in processed:
            continue
        processed.add(node_id)

        children = children_map[node_id]
        if len(children) == 0:
            cluster_leaf_sets.append(set(leaf_partition.get(node_id, {node_id})))
            continue

        if len(children) != 2:
            cluster_leaf_sets.append(set(leaf_partition.get(node_id, set())))
            continue

        left, right = children

        left_sig = local_significant.get(left, False)
        right_sig = local_significant.get(right, False)
        gate_12 = left_sig or right_sig

        if gate_12 and edge_min_effect is not None:
            eff_l = compute_edge_effect(tree, left, node_id)
            eff_r = compute_edge_effect(tree, right, node_id)
            if max(eff_l, eff_r) < edge_min_effect:
                gate_12 = False

        if not gate_12:
            cluster_leaf_sets.append(set(leaf_partition.get(node_id, set())))
            continue

        is_diff = sibling_different.get(node_id, False)
        is_skip = sibling_skipped.get(node_id, False)
        gate_3 = is_diff and not is_skip

        if gate_3 and sibling_min_effect > 0:
            sib_eff = compute_sibling_effect(tree, left, right)
            if np.isnan(sib_eff) or sib_eff < sibling_min_effect:
                gate_3 = False

        if gate_3:
            nodes_to_visit.append(right)
            nodes_to_visit.append(left)
        else:
            cluster_leaf_sets.append(set(leaf_partition.get(node_id, set())))

    # Build label array
    name_to_label = {}
    for cid, leaf_set in enumerate(cluster_leaf_sets):
        for name in leaf_set:
            name_to_label[name] = cid

    return np.array([name_to_label.get(s, -1) for s in sample_names])


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------


def run_case(case: dict) -> Tuple[Optional[pd.DataFrame], Optional[dict]]:
    """Run one benchmark case and return (node_records_df, case_summary)."""
    case_name = case.get("name", "unknown")

    try:
        data_df, y_true, x_original, metadata = generate_case_data(case)
    except Exception as e:
        print(f"  SKIP {case_name}: generator failed — {e}")
        return None, None

    if data_df is None or len(data_df) < 4:
        print(f"  SKIP {case_name}: too few samples")
        return None, None

    # Build tree + decompose
    try:
        distance_condensed = pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC)
        Z = linkage(distance_condensed, method=config.TREE_LINKAGE_METHOD)
        tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())
        decomp = tree.decompose(
            leaf_data=data_df,
            alpha_local=config.EDGE_ALPHA,
            sibling_alpha=config.SIBLING_ALPHA,
        )
    except Exception as e:
        print(f"  SKIP {case_name}: decomposition failed — {e}")
        return None, None

    annotations_df = tree.annotations_df
    n_clust = case.get("n_clusters")
    if n_clust is None:
        # Real-data cases may not have a ground-truth K
        unique_labels = set(y_true) - {None, -1}
        true_k = len(unique_labels) if unique_labels else 1
    else:
        true_k = int(n_clust)
    found_k = decomp["num_clusters"]

    # Labels from current pipeline
    from benchmarks.shared.util.decomposition import _labels_from_decomposition

    labels_current = np.asarray(_labels_from_decomposition(decomp, data_df.index.tolist()))

    # Check if ground-truth labels are usable
    has_gt = y_true is not None and not np.any(pd.isna(y_true))
    ari_current = adjusted_rand_score(y_true, labels_current) if has_gt else np.nan

    # Collect node records
    records = collect_node_records(tree, annotations_df, case_name)
    records_df = pd.DataFrame([r.__dict__ for r in records])

    # Simulate thresholds
    sibling_thresholds = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.10]
    threshold_results = []

    for thr in sibling_thresholds:
        sim_labels = labels_from_simulation(
            tree,
            annotations_df,
            data_df.index.tolist(),
            sibling_min_effect=thr,
            edge_min_effect=None,
        )
        sim_k = len(set(sim_labels) - {-1})
        if has_gt and sim_k > 0:
            sim_ari = adjusted_rand_score(y_true, sim_labels)
        else:
            sim_ari = np.nan
        threshold_results.append(
            {
                "sibling_threshold": thr,
                "k_found": sim_k,
                "ari": sim_ari,
            }
        )

    case_summary = {
        "case_name": case_name,
        "category": case.get("category", ""),
        "n_samples": len(data_df),
        "n_features": len(data_df.columns),
        "true_k": true_k,
        "found_k": found_k,
        "ari_baseline": ari_current,
        "threshold_results": threshold_results,
    }

    return records_df, case_summary


def main():
    """Run effect-size diagnosis across all benchmark cases."""
    print("=" * 80)
    print("EFFECT-SIZE GATE IMPACT DIAGNOSIS")
    print("=" * 80)

    cases = get_default_test_cases()

    # Filter out SBM cases (known broken — NaN distances)
    cases = [c for c in cases if c.get("generator") != "sbm"]

    print(f"\nRunning {len(cases)} cases (excluding SBM)...\n")

    all_node_records = []
    all_case_summaries = []

    for i, case in enumerate(cases):
        case_name = case.get("name", f"case_{i}")
        print(f"[{i+1}/{len(cases)}] {case_name}...", end=" ", flush=True)

        records_df, summary = run_case(case)

        if records_df is not None and summary is not None:
            all_node_records.append(records_df)
            all_case_summaries.append(summary)
            print(f"K={summary['found_k']}/{summary['true_k']}, ARI={summary['ari_baseline']:.3f}")
        else:
            print()

    if not all_node_records:
        print("\nNo cases completed successfully.")
        return

    # ---------------------------------------------------------------
    # ANALYSIS 1: Effect-size distribution at SPLIT vs MERGE nodes
    # ---------------------------------------------------------------
    all_nodes_df = pd.concat(all_node_records, ignore_index=True)

    print("\n" + "=" * 80)
    print("ANALYSIS 1: EFFECT-SIZE DISTRIBUTIONS BY DECISION")
    print("=" * 80)

    # Filter to binary-children nodes
    binary_mask = all_nodes_df["decision"].isin(["SPLIT", "MERGE"])
    binary_df = all_nodes_df[binary_mask].copy()

    if len(binary_df) > 0:
        for dec in ["SPLIT", "MERGE"]:
            subset = binary_df[binary_df["decision"] == dec]
            if len(subset) == 0:
                continue
            print(f"\n  {dec} nodes (n={len(subset)}):")
            for col, label in [
                ("sibling_effect_mean", "Sibling effect (mean |Δp|)"),
                ("sibling_effect_max", "Sibling effect (max |Δp|)"),
                ("edge_effect_left", "Edge effect L (mean |Δp|)"),
                ("edge_effect_right", "Edge effect R (mean |Δp|)"),
            ]:
                vals = subset[col].dropna()
                if len(vals) == 0:
                    continue
                print(f"    {label}:")
                print(
                    f"      min={vals.min():.4f}  p25={vals.quantile(0.25):.4f}  "
                    f"median={vals.median():.4f}  p75={vals.quantile(0.75):.4f}  "
                    f"max={vals.max():.4f}"
                )

        # Overlap analysis
        split_sibling = binary_df.loc[
            binary_df["decision"] == "SPLIT", "sibling_effect_mean"
        ].dropna()
        merge_sibling = binary_df.loc[
            binary_df["decision"] == "MERGE", "sibling_effect_mean"
        ].dropna()

        if len(split_sibling) > 0 and len(merge_sibling) > 0:
            print("\n  Separation:")
            print(
                f"    SPLIT sibling effects: [{split_sibling.min():.4f}, {split_sibling.max():.4f}]"
            )
            print(
                f"    MERGE sibling effects: [{merge_sibling.min():.4f}, {merge_sibling.max():.4f}]"
            )
            overlap_low = max(split_sibling.min(), merge_sibling.min())
            overlap_high = min(split_sibling.max(), merge_sibling.max())
            if overlap_low < overlap_high:
                n_overlap_split = (
                    (split_sibling >= overlap_low) & (split_sibling <= overlap_high)
                ).sum()
                n_overlap_merge = (
                    (merge_sibling >= overlap_low) & (merge_sibling <= overlap_high)
                ).sum()
                print(f"    Overlap zone: [{overlap_low:.4f}, {overlap_high:.4f}]")
                print(f"    SPLIT nodes in overlap: {n_overlap_split}/{len(split_sibling)}")
                print(f"    MERGE nodes in overlap: {n_overlap_merge}/{len(merge_sibling)}")
            else:
                print(f"    No overlap — clean separation at ~{overlap_low:.4f}")

    # ---------------------------------------------------------------
    # ANALYSIS 2: Threshold sweep — aggregate impact on K and ARI
    # ---------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ANALYSIS 2: SIBLING EFFECT-SIZE THRESHOLD SWEEP")
    print("=" * 80)

    sibling_thresholds = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.10]

    print(
        f"\n  {'Threshold':>10}  {'Mean ARI':>10}  {'Med ARI':>10}  {'Mean ΔK':>10}  "
        f"{'Exact K':>8}  {'K=1':>5}  {'Over-split':>12}  {'Cases':>6}"
    )
    print("  " + "-" * 85)

    for thr in sibling_thresholds:
        aris = []
        delta_ks = []
        exact_k = 0
        k1_count = 0
        over_split = 0

        for summary in all_case_summaries:
            true_k = summary["true_k"]
            thr_result = None
            for tr in summary["threshold_results"]:
                if abs(tr["sibling_threshold"] - thr) < 1e-9:
                    thr_result = tr
                    break
            if thr_result is None:
                continue

            k_found = thr_result["k_found"]
            ari = thr_result["ari"]
            aris.append(ari)
            delta_ks.append(k_found - true_k)

            if k_found == true_k:
                exact_k += 1
            if k_found == 1:
                k1_count += 1
            if k_found > true_k:
                over_split += 1

        if aris:
            mean_ari = np.mean(aris)
            med_ari = np.median(aris)
            mean_dk = np.mean(delta_ks)
            n_cases = len(aris)
            print(
                f"  {thr:>10.3f}  {mean_ari:>10.3f}  {med_ari:>10.3f}  "
                f"{mean_dk:>+10.1f}  {exact_k:>8d}  {k1_count:>5d}  "
                f"{over_split:>12d}  {n_cases:>6d}"
            )

    # ---------------------------------------------------------------
    # ANALYSIS 3: Per-case winners — which threshold is best per case?
    # ---------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ANALYSIS 3: PER-CASE BEST THRESHOLD")
    print("=" * 80)

    improvements = []
    degradations = []

    for summary in all_case_summaries:
        baseline_ari = summary["ari_baseline"]
        baseline_k = summary["found_k"]
        true_k = summary["true_k"]
        case_name = summary["case_name"]

        best_ari = baseline_ari
        best_thr = 0.0
        best_k = baseline_k

        for tr in summary["threshold_results"]:
            if tr["ari"] > best_ari + 1e-6:
                best_ari = tr["ari"]
                best_thr = tr["sibling_threshold"]
                best_k = tr["k_found"]

        if best_thr > 0 and best_ari > baseline_ari + 0.01:
            delta = best_ari - baseline_ari
            improvements.append(
                {
                    "case": case_name,
                    "category": summary["category"],
                    "true_k": true_k,
                    "baseline_k": baseline_k,
                    "best_k": best_k,
                    "baseline_ari": baseline_ari,
                    "best_ari": best_ari,
                    "delta_ari": delta,
                    "best_threshold": best_thr,
                }
            )

        # Check for degradation at moderate threshold (0.03)
        thr_03 = [
            tr for tr in summary["threshold_results"] if abs(tr["sibling_threshold"] - 0.03) < 1e-9
        ]
        if thr_03 and thr_03[0]["ari"] < baseline_ari - 0.01:
            degradations.append(
                {
                    "case": case_name,
                    "true_k": true_k,
                    "baseline_k": baseline_k,
                    "k_at_03": thr_03[0]["k_found"],
                    "baseline_ari": baseline_ari,
                    "ari_at_03": thr_03[0]["ari"],
                    "delta_ari": thr_03[0]["ari"] - baseline_ari,
                }
            )

    if improvements:
        print(f"\n  Cases IMPROVED by effect-size gate ({len(improvements)}):")
        imp_df = pd.DataFrame(improvements).sort_values("delta_ari", ascending=False)
        for _, row in imp_df.iterrows():
            print(
                f"    {row['case']:40s}  K: {row['baseline_k']:>2d}→{row['best_k']:>2d} "
                f"(true={row['true_k']:>2d})  ARI: {row['baseline_ari']:.3f}→{row['best_ari']:.3f} "
                f"(+{row['delta_ari']:.3f})  @ thr={row['best_threshold']:.3f}"
            )
    else:
        print("\n  No cases improved by > 0.01 ARI with any threshold.")

    if degradations:
        print(f"\n  Cases DEGRADED at threshold=0.03 ({len(degradations)}):")
        deg_df = pd.DataFrame(degradations).sort_values("delta_ari")
        for _, row in deg_df.iterrows():
            print(
                f"    {row['case']:40s}  K: {row['baseline_k']:>2d}→{row['k_at_03']:>2d} "
                f"(true={row['true_k']:>2d})  ARI: {row['baseline_ari']:.3f}→{row['ari_at_03']:.3f} "
                f"({row['delta_ari']:+.3f})"
            )
    else:
        print("\n  No cases degraded by > 0.01 ARI at threshold=0.03.")

    # ---------------------------------------------------------------
    # ANALYSIS 4: Node-level effect-size histogram (text)
    # ---------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ANALYSIS 4: EFFECT-SIZE HISTOGRAM (SIBLING MEAN |ΔP|)")
    print("=" * 80)

    if len(binary_df) > 0:
        bins = [0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 1.0]
        for dec in ["SPLIT", "MERGE"]:
            vals = binary_df.loc[binary_df["decision"] == dec, "sibling_effect_mean"].dropna()
            if len(vals) == 0:
                continue
            hist, _ = np.histogram(vals, bins=bins)
            print(f"\n  {dec} nodes (n={len(vals)}):")
            for i in range(len(bins) - 1):
                bar = "█" * hist[i]
                pct = 100 * hist[i] / len(vals) if len(vals) > 0 else 0
                print(f"    [{bins[i]:.3f}, {bins[i+1]:.3f}) : {hist[i]:>4d} ({pct:5.1f}%) {bar}")

    # ---------------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------------
    outdir = Path(__file__).parent / "results"
    outdir.mkdir(parents=True, exist_ok=True)

    if len(all_node_records) > 0:
        all_nodes_df.to_csv(outdir / "effect_size_node_records.csv", index=False)
        print(f"\n  Node records saved to {outdir / 'effect_size_node_records.csv'}")

    # Save threshold sweep summary
    sweep_rows = []
    for summary in all_case_summaries:
        for tr in summary["threshold_results"]:
            sweep_rows.append(
                {
                    "case_name": summary["case_name"],
                    "category": summary["category"],
                    "true_k": summary["true_k"],
                    "baseline_k": summary["found_k"],
                    "baseline_ari": summary["ari_baseline"],
                    "threshold": tr["sibling_threshold"],
                    "k_found": tr["k_found"],
                    "ari": tr["ari"],
                }
            )
    if sweep_rows:
        pd.DataFrame(sweep_rows).to_csv(outdir / "effect_size_threshold_sweep.csv", index=False)
        print(f"  Threshold sweep saved to {outdir / 'effect_size_threshold_sweep.csv'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
