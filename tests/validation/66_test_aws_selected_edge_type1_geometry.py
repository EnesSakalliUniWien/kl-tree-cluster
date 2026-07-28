from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from benchmarks.cloud.aws_selected_edge_type1_geometry import (
    AwsSelectedEdgeGeometryConfig,
    make_shard_spec,
    merge_shards,
)


def _config(output_dir: Path, *, replicates: int = 10, shard_count: int = 3):
    return AwsSelectedEdgeGeometryConfig(
        output_dir=output_dir,
        suite="binary",
        case_names=("binary_2clusters",),
        modes=("selected_tree",),
        edge_alphas=(0.001,),
        sibling_alpha=0.01,
        replicates=replicates,
        base_seed=20260604,
        shard_count=shard_count,
        s3_uri=None,
        resume=True,
    )


def test_selected_edge_shards_distribute_replicates() -> None:
    config = _config(Path("/tmp/out"))

    assert make_shard_spec(config, 0).replicate_indices == (0, 3, 6, 9)
    assert make_shard_spec(config, 1).replicate_indices == (1, 4, 7)
    assert make_shard_spec(config, 2).replicate_indices == (2, 5, 8)


def _write_fake_shard(root: Path, shard_index: int, replicate_indices: tuple[int, ...]) -> None:
    shard_dir = root / "shards" / f"shard_{shard_index:04d}"
    shard_dir.mkdir(parents=True)
    (shard_dir / "aws_selected_edge_geometry_shard_manifest.json").write_text(
        json.dumps(
            {
                "shard_index": shard_index,
                "shard_count": 2,
                "replicate_indices": list(replicate_indices),
            }
        )
        + "\n"
    )
    pd.DataFrame(
        {"replicate": list(replicate_indices), "edge_rejected": [False] * len(replicate_indices)}
    ).to_csv(
        shard_dir / "selected_edge_geometry_edges.csv",
        index=False,
    )
    pd.DataFrame(
        {"replicate": list(replicate_indices), "sibling_rejected": [False] * len(replicate_indices)}
    ).to_csv(
        shard_dir / "selected_edge_geometry_siblings.csv",
        index=False,
    )
    pd.DataFrame(
        {"replicate": list(replicate_indices), "false_split": [False] * len(replicate_indices)}
    ).to_csv(
        shard_dir / "selected_edge_geometry_final.csv",
        index=False,
    )


def test_merge_shards_combines_expected_replicates(tmp_path: Path) -> None:
    config = _config(tmp_path, replicates=4, shard_count=2)
    _write_fake_shard(tmp_path, 0, (0, 2))
    _write_fake_shard(tmp_path, 1, (1, 3))

    outputs = merge_shards(config)

    assert outputs["edges"].shape[0] == 4
    assert outputs["final"].shape[0] == 4
    assert (tmp_path / "merged" / "selected_edge_geometry_edges.csv").exists()
    assert (tmp_path / "merged" / "aws_selected_edge_geometry_manifest.json").exists()


def test_merge_shards_rejects_missing_replicate(tmp_path: Path) -> None:
    config = _config(tmp_path, replicates=4, shard_count=2)
    _write_fake_shard(tmp_path, 0, (0, 2))
    _write_fake_shard(tmp_path, 1, (1,))

    with pytest.raises(ValueError, match="Missing replicate"):
        merge_shards(config)
