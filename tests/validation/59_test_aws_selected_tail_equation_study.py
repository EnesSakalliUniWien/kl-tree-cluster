from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from benchmarks.cloud.aws_selected_tail_equation_study import (
    SHARD_MANIFEST_NAME,
    SHARD_RECORDS_NAME,
    AwsSelectedTailStudyConfig,
    iter_expected_shard_dirs,
    load_combined_shard_records,
    load_shard_records,
    make_shard_spec,
    parse_case_names,
    recompute_outputs_from_records,
    resolve_shard_index,
    shard_seed,
    validate_shard_contract,
    write_combined_outputs,
)


def _geometry_records(case_id: str, *, replicate_offset: int = 0) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(20):
        replicate_index = replicate_offset + index // 2
        ratio = 1.0 + 0.2 * index
        cos2 = 0.05 + 0.04 * index
        rows.append(
            {
                "case_id": case_id,
                "case_category": "synthetic_family",
                "source_family": "source_family",
                "feature_representation": "bernoulli",
                "feature_family": "bernoulli",
                "replicate_index": replicate_index,
                "selected_hierarchy_simulation_id": f"{case_id}:{replicate_index}",
                "feature_dimension": 40,
                "parent_sample_size": 20,
                "left_child_sample_size": 10,
                "right_child_sample_size": 10,
                "selected_hierarchy_ratio": ratio,
                "log_selected_hierarchy_ratio": float(np.log(ratio)),
                "parent_depth": index % 4,
                "parent_size_fraction": 1.0 - index / 40.0,
                "parent_size_bin": "root_0.75_1",
                "child_balance": 0.5 - index / 100.0,
                "child_size_ratio": 1.0 + index / 10.0,
                "branch_length_sum": 1.0 + index / 20.0,
                "branch_length_asymmetry": index / 50.0,
                "left_edge_raw_p_value": 0.001 + index / 1000.0,
                "right_edge_raw_p_value": 0.002 + index / 1000.0,
                "min_child_edge_raw_p_value": 0.001 + index / 1000.0,
                "max_child_edge_raw_p_value": 0.002 + index / 1000.0,
                "left_edge_bh_p_value": 0.01 + index / 1000.0,
                "right_edge_bh_p_value": 0.02 + index / 1000.0,
                "min_child_edge_bh_p_value": 0.01 + index / 1000.0,
                "max_child_edge_bh_p_value": 0.02 + index / 1000.0,
                "negative_log10_min_child_edge_bh_p_value": float(-np.log10(0.01 + index / 1000.0)),
                "raw_mp_signal_count": 2 + index % 3,
                "parent_test_projection_dimension": 2 + index % 3,
                "sibling_projection_dimension": 2 + index % 3,
                "effective_independent_rows": 20 + index,
                "mp_threshold_rows": 20 + index,
                "eigenvalue_effective_rank": 1.5 + index / 30.0,
                "top_eigenvalue_share": 0.8 - index / 100.0,
                "selected_eigenvalue_mass_fraction": 0.5 + index / 80.0,
                "eigengap_at_sibling_projection_dimension": 1.1 + index / 20.0,
                "selected_eigenvalue_over_mp_upper_bound": 0.7 + index / 50.0,
                "selected_subspace_cos2": cos2,
                "selected_subspace_sin2": 1.0 - cos2,
                "selected_subspace_tan2": (1.0 - cos2) / cos2,
                "top_component_cos2": 0.01 + (index % 5) / 30.0,
                "max_component_cos2": 0.02 + (index % 7) / 25.0,
                "component_cos2_entropy": 0.2 + ((index * 7) % 11) / 40.0,
            }
        )
    return pd.DataFrame.from_records(rows)


def _write_shard(shard_dir: Path, *, shard_index: int, records: pd.DataFrame) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    records.to_csv(shard_dir / SHARD_RECORDS_NAME, index=False)
    (shard_dir / SHARD_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "shard_index": shard_index,
                "shard_seed": shard_seed(base_seed=20260603, shard_index=shard_index),
            }
        )
        + "\n"
    )


def test_parse_case_names_rejects_empty_contract() -> None:
    assert parse_case_names("a,b") == ("a", "b")
    with pytest.raises(ValueError, match="At least one case name"):
        parse_case_names(" , ")


def test_resolve_shard_index_uses_cli_or_aws_batch_environment() -> None:
    assert resolve_shard_index(3, {"AWS_BATCH_JOB_ARRAY_INDEX": "7"}) == 3
    assert resolve_shard_index(None, {"AWS_BATCH_JOB_ARRAY_INDEX": "7"}) == 7
    with pytest.raises(ValueError, match="Shard index is required"):
        resolve_shard_index(None, {})


def test_make_shard_spec_validates_bounds_and_names_path() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        config = AwsSelectedTailStudyConfig(
            case_names=("case",),
            output_dir=Path(tmpdir),
            base_seed=20260603,
            shard_count=4,
            replicates_per_shard=50,
        )

        spec = make_shard_spec(config, 2)

        assert spec.seed == 20260603 + 2_000_000_000
        assert spec.output_dir == Path(tmpdir) / "shards" / "shard_0002"
        with pytest.raises(ValueError, match="0 <= index < shard_count"):
            validate_shard_contract(
                shard_index=4,
                shard_count=4,
                replicates_per_shard=50,
            )


def test_load_shard_records_prefixes_independent_simulation_ids() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        shard_dir = Path(tmpdir) / "shard_0001"
        _write_shard(
            shard_dir,
            shard_index=1,
            records=_geometry_records("case").head(4),
        )

        loaded = load_shard_records(shard_dir)

    assert loaded["aws_shard_id"].eq("shard_0001").all()
    assert loaded["aws_shard_index"].eq(1).all()
    assert set(loaded["selected_hierarchy_simulation_id"]) == {
        "shard_0001:case:0",
        "shard_0001:case:1",
    }


def test_load_combined_records_rejects_duplicate_shard_inputs() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        shard_dir = Path(tmpdir) / "shard_0000"
        _write_shard(shard_dir, shard_index=0, records=_geometry_records("case").head(4))

        with pytest.raises(ValueError, match="Duplicate shard id"):
            load_combined_shard_records((shard_dir, shard_dir))


def test_merge_recomputes_outputs_from_namespaced_records() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        shard_dirs = iter_expected_shard_dirs(root, 2)
        _write_shard(shard_dirs[0], shard_index=0, records=_geometry_records("case_a"))
        _write_shard(shard_dirs[1], shard_index=1, records=_geometry_records("case_b"))

        combined = load_combined_shard_records(shard_dirs)
        outputs = recompute_outputs_from_records(combined)

    assert outputs["selected_geometry_records"].shape[0] == 40
    assert outputs["geometry_summary_by_case"].shape[0] == 2
    tail_law = outputs["selected_ratio_tail_law"]
    assert "n_matching_simulations" in tail_law.columns
    assert tail_law["tail_law_role"].notna().all()


def test_write_combined_outputs_records_cloud_manifest() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        shard_dirs = iter_expected_shard_dirs(root / "run", 1)
        _write_shard(shard_dirs[0], shard_index=0, records=_geometry_records("case"))
        combined = load_combined_shard_records(shard_dirs)
        config = AwsSelectedTailStudyConfig(
            case_names=("case",),
            output_dir=root / "run",
            base_seed=20260603,
            shard_count=1,
            replicates_per_shard=20,
        )

        outputs = write_combined_outputs(
            records=combined,
            output_dir=root / "run" / "merged",
            shard_dirs=shard_dirs,
            configured=config,
        )

        manifest_path = root / "run" / "merged" / "aws_selected_tail_equation_study_manifest.json"
        manifest = json.loads(manifest_path.read_text())
    assert "selected_ratio_tail_law" in outputs
    assert manifest["requested_total_replicates_per_case"] == 20
    assert manifest["n_independent_simulations"] == 10
