from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from benchmarks.diagnostics.calibration import (
    selected_neighborhood_internal_spectral_flow_panel as panel,
)


def _edge_row(
    *,
    data_role: str,
    child_id: str,
    variant_id: str,
    supported: bool,
    parent_raw: int,
    child_raw: int,
    common: int,
    barrier: float,
    affinity: float,
) -> dict[str, object]:
    return {
        "case_id": "case",
        "data_role": data_role,
        "method_id": "method",
        "replicate": 0,
        "parent_id": "root",
        "child_id": child_id,
        "spectral_variant_id": variant_id,
        "flow_status": (
            "mp_supported_subspace_compared"
            if supported
            else "floor_only_no_mp_certified_mode"
        ),
        "mp_pair_supported": supported,
        "parent_raw_mp_signal_count": parent_raw,
        "child_raw_mp_signal_count": child_raw,
        "mp_common_dimension": common,
        "mp_subspace_chordal_distance": 0.8 if barrier > 0.5 else 0.1,
        "mp_log_eigenvalue_delta": 0.6 if barrier > 0.5 else 0.1,
        "spectral_barrier": barrier,
        "spectral_flow_affinity": affinity,
    }


def _mode_row(
    *,
    data_role: str,
    child_id: str,
    variant_id: str,
    matched: int,
    cost: float,
    affinity: float,
) -> dict[str, object]:
    return {
        "case_id": "case",
        "data_role": data_role,
        "method_id": "method",
        "replicate": 0,
        "parent_id": "root",
        "child_id": child_id,
        "spectral_variant_id": variant_id,
        "mode_flow_status": (
            "mp_blocks_compared"
            if matched > 0
            else "floor_only_no_mp_certified_block"
        ),
        "matched_mp_block_count": matched,
        "unmatched_mp_block_count": 0 if matched > 0 else 1,
        "mode_transport_cost": cost,
        "mode_transport_affinity": affinity,
        "connection_laplacian_residual": cost,
        "mean_block_projector_chordal_distance": cost,
    }


def _edge_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            _edge_row(
                data_role="signal",
                child_id="signal_child",
                variant_id=panel.LEAF_VARIANT,
                supported=True,
                parent_raw=1,
                child_raw=1,
                common=1,
                barrier=1.0,
                affinity=0.36,
            ),
            _edge_row(
                data_role="signal",
                child_id="signal_child",
                variant_id=panel.INTERNAL_VARIANT,
                supported=True,
                parent_raw=2,
                child_raw=2,
                common=2,
                barrier=0.2,
                affinity=0.82,
            ),
            _edge_row(
                data_role="selected_null",
                child_id="null_child",
                variant_id=panel.LEAF_VARIANT,
                supported=False,
                parent_raw=0,
                child_raw=0,
                common=0,
                barrier=float("nan"),
                affinity=float("nan"),
            ),
            _edge_row(
                data_role="selected_null",
                child_id="null_child",
                variant_id=panel.INTERNAL_VARIANT,
                supported=True,
                parent_raw=1,
                child_raw=1,
                common=1,
                barrier=0.1,
                affinity=0.90,
            ),
            _edge_row(
                data_role="signal",
                child_id="signal_child",
                variant_id=panel.BRANCH_LENGTH_INTERNAL_VARIANT,
                supported=True,
                parent_raw=2,
                child_raw=2,
                common=2,
                barrier=0.4,
                affinity=0.70,
            ),
            _edge_row(
                data_role="selected_null",
                child_id="null_child",
                variant_id=panel.BRANCH_LENGTH_INTERNAL_VARIANT,
                supported=False,
                parent_raw=0,
                child_raw=0,
                common=0,
                barrier=float("nan"),
                affinity=float("nan"),
            ),
        ]
    )


def _mode_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            _mode_row(
                data_role="selected_null",
                child_id="null_child",
                variant_id=panel.LEAF_VARIANT,
                matched=0,
                cost=1.0,
                affinity=0.36,
            ),
            _mode_row(
                data_role="selected_null",
                child_id="null_child",
                variant_id=panel.INTERNAL_VARIANT,
                matched=1,
                cost=0.2,
                affinity=0.82,
            ),
            _mode_row(
                data_role="selected_null",
                child_id="null_child",
                variant_id=panel.BRANCH_LENGTH_INTERNAL_VARIANT,
                matched=0,
                cost=1.0,
                affinity=0.36,
            ),
        ]
    )


def test_edge_pairwise_marks_internal_support_creation_and_stability_gain() -> None:
    pairwise = panel.build_edge_pairwise_rows(_edge_rows())
    by_key = pairwise.set_index(["spectral_variant_comparison_id", "data_role"])
    by_role = pairwise[
        pairwise["spectral_variant_comparison_id"].eq(panel.EMPIRICAL_INTERNAL_COMPARISON)
    ].set_index("data_role")

    assert (
        by_role.loc["selected_null", "edge_gain_status"]
        == "mp_support_created_by_internal_barycenter"
    )
    assert (
        by_role.loc["signal", "edge_gain_status"]
        == "both_supported_barrier_improved"
    )
    assert by_role.loc["signal", "delta_mp_common_dimension"] == 1
    assert by_role.loc["signal", "delta_spectral_barrier"] == -0.8
    assert (
        by_key.loc[
            (panel.BRANCH_LENGTH_INTERNAL_COMPARISON, "signal"),
            "edge_gain_status",
        ]
        == "both_supported_barrier_improved"
    )
    assert (
        by_key.loc[
            (panel.BRANCH_LENGTH_INTERNAL_COMPARISON, "selected_null"),
            "edge_gain_status",
        ]
        == "floor_only_both_variants"
    )


def test_mode_pairwise_marks_internal_block_support_creation() -> None:
    pairwise = panel.build_mode_pairwise_rows(_mode_rows())
    row = pairwise.iloc[0]

    assert row["mode_gain_status"] == "mp_block_support_created_by_internal_barycenter"
    assert row["delta_matched_mp_block_count"] == 1
    assert row["delta_mode_transport_cost"] == -0.8


def test_summary_warns_on_selected_null_gain_and_reports_signal_gain() -> None:
    edge_pairwise = panel.build_edge_pairwise_rows(_edge_rows())
    mode_pairwise = panel.build_mode_pairwise_rows(_mode_rows())

    summary = panel.summarize_internal_spectral_flow(edge_pairwise, mode_pairwise)
    by_role = summary[
        summary["spectral_variant_comparison_id"].eq(
            panel.EMPIRICAL_INTERNAL_COMPARISON
        )
    ].set_index("data_role")

    assert (
        by_role.loc["selected_null", "summary_status"]
        == "diagnostic_warn_selected_null_internal_support_gain"
    )
    assert (
        by_role.loc["signal", "summary_status"]
        == "diagnostic_signal_internal_flow_stability_gain_observed"
    )
    assert by_role.loc["selected_null", "mode_support_created_count"] == 1


def test_neighborhood_energy_separates_strict_and_internal_only_support() -> None:
    edge_pairwise = panel.build_edge_pairwise_rows(_edge_rows())
    energy = panel.build_neighborhood_energy_rows(edge_pairwise)
    by_role = energy[
        energy["spectral_variant_comparison_id"].eq(
            panel.EMPIRICAL_INTERNAL_COMPARISON
        )
    ].set_index("data_role")

    assert by_role.loc["signal", "strict_shared_mp_supported_edge_count"] == 1
    assert by_role.loc["signal", "internal_only_mp_supported_edge_count"] == 0
    assert np.isclose(
        by_role.loc["signal", "strict_shared_leaf_mp_joint_transport_energy"],
        1.0,
    )
    assert np.isclose(
        by_role.loc["signal", "strict_shared_internal_mp_joint_transport_energy"],
        0.02,
    )
    assert np.isclose(
        by_role.loc["signal", "delta_strict_shared_mp_joint_transport_energy"],
        -0.98,
    )
    assert (
        by_role.loc["signal", "neighborhood_energy_status"]
        == "diagnostic_internal_smooths_strict_shared_transport"
    )

    assert by_role.loc["selected_null", "strict_shared_mp_supported_edge_count"] == 0
    assert by_role.loc["selected_null", "internal_only_mp_supported_edge_count"] == 1
    assert np.isclose(
        by_role.loc["selected_null", "internal_only_mp_joint_transport_energy"],
        0.02,
    )
    assert (
        by_role.loc["selected_null", "neighborhood_energy_status"]
        == "diagnostic_warn_selected_null_internal_only_transport_energy"
    )


def test_node_pairwise_separates_angle_and_radius_transport() -> None:
    leaf_context = SimpleNamespace(
        principal_component_projections_by_node={
            "stable": np.asarray([[1.0, 0.0], [0.0, 1.0]]),
            "rotated": np.asarray(
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
            ),
            "created": np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        },
        principal_component_eigenvalues_by_node={
            "stable": np.asarray([4.0, 2.0]),
            "rotated": np.asarray([4.0, 2.0]),
            "created": np.asarray([1.0, 0.5]),
        },
        raw_mp_signal_counts_by_node={"stable": 2, "rotated": 2, "created": 0},
        test_projection_dimensions_by_node={"stable": 2, "rotated": 2, "created": 2},
    )
    internal_context = SimpleNamespace(
        principal_component_projections_by_node={
            "stable": np.asarray([[1.0, 0.0], [0.0, 1.0]]),
            "rotated": np.asarray(
                [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
            ),
            "created": np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        },
        principal_component_eigenvalues_by_node={
            "stable": np.asarray([8.0, 4.0]),
            "rotated": np.asarray([4.0, 2.0]),
            "created": np.asarray([5.0, 2.0]),
        },
        raw_mp_signal_counts_by_node={"stable": 2, "rotated": 2, "created": 1},
        test_projection_dimensions_by_node={"stable": 2, "rotated": 2, "created": 2},
    )

    rows = panel.build_node_pairwise_rows(
        case_id="case",
        data_role="signal",
        method_id="method",
        replicate=0,
        leaf_context=leaf_context,
        internal_context=internal_context,
    )
    by_node = rows.set_index("node_id")

    assert by_node.loc["stable", "object_transport_status"] == "stable_angle_power_gain"
    assert by_node.loc["stable", "mp_mean_sin_angle"] == 0.0
    assert np.isclose(by_node.loc["stable", "mp_mean_log_radius_delta"], np.log(2.0))

    assert by_node.loc["rotated", "object_transport_status"] == "whole_object_rotation"
    assert by_node.loc["rotated", "mp_mean_sin_angle"] == 1.0
    assert by_node.loc["rotated", "mp_rms_log_radius_delta"] == 0.0

    assert by_node.loc["created", "object_transport_status"] == "internal_only_spike_created"


def test_panel_writes_outputs_with_monkeypatched_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_rows(
        _config: panel.InternalSpectralFlowConfig,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return (
            pd.DataFrame(),
            _edge_rows(),
            pd.DataFrame(),
            _mode_rows(),
            pd.DataFrame(),
        )

    monkeypatch.setattr(panel, "build_internal_spectral_flow_rows", fake_rows)

    outputs = panel.run_selected_neighborhood_internal_spectral_flow_panel(
        panel.InternalSpectralFlowConfig(output_dir=tmp_path)
    )

    assert set(outputs) == {
        "nodes",
        "edges",
        "blocks",
        "mode_edges",
        "node_pairwise",
        "edge_pairwise",
        "mode_pairwise",
        "neighborhood_energy",
        "summary",
        "manifest",
    }
    assert all(path.exists() for path in outputs.values())

    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["generated_by"] == panel.GENERATED_BY
    assert set(manifest["variants"]) == {
        panel.LEAF_VARIANT,
        panel.INTERNAL_VARIANT,
        panel.BRANCH_LENGTH_INTERNAL_VARIANT,
    }
    assert {
        comparison["spectral_variant_comparison_id"]
        for comparison in manifest["comparisons"]
    } == {
        panel.EMPIRICAL_INTERNAL_COMPARISON,
        panel.BRANCH_LENGTH_INTERNAL_COMPARISON,
    }
