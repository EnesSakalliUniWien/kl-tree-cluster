#!/usr/bin/env python3
"""Trace missing-sibling-space geometry for selected case/parent pairs.

This is a benchmark/debug helper. It compares:

- the live JL fallback path (variant A)
- the experimental parent Gate 2 k + random basis path (variant B)

for concrete parent nodes inside one benchmark case.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.compare_missing_space_policies import (  # noqa: E402
    _prepare_case,
    _run_variant,
)
from kl_clustering_analysis import config  # noqa: E402
from kl_clustering_analysis.hierarchy_analysis.decomposition.backends.random_projection.dimension import (  # noqa: E402
    resolve_minimum_projection_dimension,
)
from kl_clustering_analysis.hierarchy_analysis.decomposition.backends.random_projection.seed import (  # noqa: E402
    derive_projection_seed,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.projected_wald.projected_wald_projection_basis import (  # noqa: E402
    build_projection_basis_with_padding,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.wald_statistic.sibling_z_scores import (  # noqa: E402
    _compute_sibling_z_scores,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.projection.pair_testing.projection_dimension import (  # noqa: E402
    resolve_sibling_projection_dimension,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace sibling projection geometry for one benchmark case."
    )
    parser.add_argument("--case-num", type=int, required=True, help="1-based benchmark case id.")
    parser.add_argument(
        "--parents",
        type=str,
        required=True,
        help="Comma-separated parent node ids, e.g. N30,N58",
    )
    return parser.parse_args()


def _format_float_list(values: np.ndarray, limit: int = 10) -> str:
    if values.size == 0:
        return "[]"
    trimmed = values[:limit]
    rendered = ", ".join(f"{value:.6g}" for value in trimmed)
    suffix = ", ..." if values.size > limit else ""
    return f"[{rendered}{suffix}]"


def _trace_parent(case_num: int, parent: str) -> None:
    prepared = _prepare_case(case_num)
    tree = prepared.tree
    if parent not in tree:
        raise KeyError(f"Unknown parent {parent!r} for case {case_num}.")

    children = list(tree.successors(parent))
    if len(children) != 2:
        raise ValueError(f"Parent {parent!r} is not binary; children={children!r}")

    resolve_minimum_projection_dimension(
        config.PROJECTION_MINIMUM_DIMENSION,
        leaf_data=prepared.data_df,
    )

    left, right = children
    left_dist = np.asarray(tree.nodes[left]["distribution"], dtype=np.float64)
    right_dist = np.asarray(tree.nodes[right]["distribution"], dtype=np.float64)
    left_n = int(tree.nodes[left]["leaf_count"])
    right_n = int(tree.nodes[right]["leaf_count"])
    z = _compute_sibling_z_scores(
        left_dist,
        right_dist,
        left_n,
        right_n,
        branch_length_sum=None,
        mean_branch_length=None,
    ).astype(np.float64, copy=False)
    z_norm_sq = float(np.dot(z, z))
    z_norm = float(np.sqrt(z_norm_sq))
    z_unit = z / z_norm if z_norm > 0 else z
    n_features = int(z.shape[0])

    spectral_context = prepared.spectral_context
    edge_dims = spectral_context.spectral_projection_dimensions_by_node or {}
    edge_projs = spectral_context.principal_component_projections_by_node or {}
    edge_eigs = spectral_context.principal_component_eigenvalues_by_node or {}

    gate2_k = int(edge_dims.get(parent, 0))
    gate2_projection = edge_projs.get(parent)
    gate2_eigenvalues = edge_eigs.get(parent)
    gate2_projection = (
        np.asarray(gate2_projection, dtype=np.float64) if gate2_projection is not None else None
    )
    gate2_eigenvalues = (
        np.asarray(gate2_eigenvalues, dtype=np.float64) if gate2_eigenvalues is not None else None
    )

    seed = derive_projection_seed(config.PROJECTION_RANDOM_SEED, f"sibling:{parent}")
    jl_k, _ = resolve_sibling_projection_dimension(
        projection_dimension_from_edge_comparisons=None,
        left_sample_size=float(left_n),
        right_sample_size=float(right_n),
        n_features=n_features,
    )
    jl_basis, _ = build_projection_basis_with_padding(
        n_features=n_features,
        k=jl_k,
        pca_projection=None,
        pca_eigenvalues=None,
        random_state=seed,
    )
    parent_random_basis, _ = build_projection_basis_with_padding(
        n_features=n_features,
        k=max(gate2_k, 1),
        pca_projection=None,
        pca_eigenvalues=None,
        random_state=seed,
    )

    jl_projected = jl_basis @ z
    parent_random_projected = parent_random_basis @ z

    print()
    print("=" * 100)
    print(f"Case {prepared.case_num}: {prepared.case_name} | Parent {parent}")
    print("=" * 100)
    print(f"Children: {children}")
    print(f"Child leaf flags: {[bool(tree.nodes[ch].get('is_leaf', False)) for ch in children]}")
    print(f"Child leaf counts: {[int(tree.nodes[ch].get('leaf_count', 0)) for ch in children]}")
    print(f"n_features={n_features}, ||z||^2={z_norm_sq:.6f}")
    print(f"Gate 2 parent k={gate2_k}")
    print(f"Gate 2 eigenvalues={_format_float_list(gate2_eigenvalues if gate2_eigenvalues is not None else np.array([]))}")

    if gate2_projection is not None and gate2_projection.shape[0] > 0 and z_norm > 0:
        parent_pc1 = gate2_projection[0]
        pc1_norm = float(np.linalg.norm(parent_pc1))
        if pc1_norm > 0:
            cos_pc1 = float(np.dot(parent_pc1 / pc1_norm, z_unit))
        else:
            cos_pc1 = float("nan")
        pc1_capture = float(np.dot(parent_pc1, z) ** 2)
        print(f"Gate 2 projection shape={gate2_projection.shape}")
        print(f"cos(z, parent_pc1)={cos_pc1:.6f}")
        print(f"|<pc1, z>|^2={pc1_capture:.6f}")
        if gate2_projection.shape[0] > 1:
            pc2_capture = float(np.dot(gate2_projection[1], z) ** 2)
            print(f"|<pc2, z>|^2={pc2_capture:.6f}")

    print()
    print("Neutral random subspaces")
    print(f"JL fallback k={jl_k}, ||R_jl z||^2={float(np.dot(jl_projected, jl_projected)):.6f}")
    print(
        f"Parent-derived random k={max(gate2_k, 1)}, "
        f"||R_parent_k z||^2={float(np.dot(parent_random_projected, parent_random_projected)):.6f}"
    )
    print(
        f"JL first row cosine with z={float(np.dot(jl_basis[0], z_unit)):.6f}"
        if z_norm > 0
        else "JL first row cosine with z=nan"
    )
    print(
        f"Parent-k first row cosine with z={float(np.dot(parent_random_basis[0], z_unit)):.6f}"
        if z_norm > 0
        else "Parent-k first row cosine with z=nan"
    )

    for variant in ["A_current_jl", "B_parent_gate2_random"]:
        annotations_df, metrics, injected_parent_k = _run_variant(prepared, variant)
        audit = annotations_df.attrs.get("sibling_divergence_audit", {}) or {}
        row = annotations_df.loc[
            parent,
            [
                "Sibling_Test_Statistic",
                "Sibling_Degrees_of_Freedom",
                "Sibling_Divergence_P_Value",
                "Sibling_Divergence_P_Value_Corrected",
                "Sibling_BH_Different",
                "Sibling_Divergence_Skipped",
                "Sibling_Projection_Dimension_Source",
                "Sibling_Resolved_Projection_Dimension",
            ],
        ]
        print()
        print(f"{variant}")
        print(f"Case found_clusters={int(metrics['found_clusters'])}, ari={float(metrics['ari']):.6f}")
        print(
            "Calibration "
            f"center={audit.get('local_adjuster_center')}, "
            f"spread={audit.get('local_adjuster_spread')}, "
            f"global={audit.get('global_inflation_factor')}"
        )
        print(f"Injected parent random pairs={len(injected_parent_k)}")
        print(row.to_string())


def main() -> None:
    args = _parse_args()
    parent_ids = [part.strip() for part in args.parents.split(",") if part.strip()]
    for parent in parent_ids:
        _trace_parent(args.case_num, parent)


if __name__ == "__main__":
    main()
