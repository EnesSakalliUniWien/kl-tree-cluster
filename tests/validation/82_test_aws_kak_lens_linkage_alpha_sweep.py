from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
from benchmarks.cloud.aws_kak_lens_linkage_alpha_sweep import (
    SHARD_MANIFEST_NAME,
    SUMMARY_NAME,
    AwsKakLensSweepConfig,
    iter_expected_shard_dirs,
    make_shard_spec,
    merge_shards,
    parse_lenses,
    resolve_shard_index,
)


def _config(root: Path) -> AwsKakLensSweepConfig:
    return AwsKakLensSweepConfig(
        input_path=Path("matrix.tsv"),
        main_assignments=Path("main.csv"),
        output_dir=root,
        edge_alphas=(0.001,),
        sibling_alphas=(0.01,),
        lenses=parse_lenses(
            "raw_kak:tfidf:adaptive_modes_14_30,raw_kak:binary:adaptive_modes_10_15"
        ),
        tree_linkage_methods=("average", "complete", "ward"),
        shard_count=3,
    )


def _write_shard(shard_dir: Path, *, shard_index: int, rows: list[dict[str, object]]) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(rows).to_csv(shard_dir / SUMMARY_NAME, index=False)
    (shard_dir / SHARD_MANIFEST_NAME).write_text(
        json.dumps({"shard_index": shard_index}) + "\n",
        encoding="utf-8",
    )


def test_resolve_shard_index_uses_explicit_or_aws_array_environment() -> None:
    assert resolve_shard_index(1, {"AWS_BATCH_JOB_ARRAY_INDEX": "0"}) == 1
    assert resolve_shard_index(None, {"AWS_BATCH_JOB_ARRAY_INDEX": "2"}) == 2


def test_make_shard_spec_distributes_lens_linkage_groups_by_modulo() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _config(Path(tmpdir))

        first = make_shard_spec(config, 0)
        second = make_shard_spec(config, 1)

    third = make_shard_spec(config, 2)

    assert [group.group_index for group in first.groups] == [0, 3]
    assert [group.tree_linkage_method for group in first.groups] == ["average", "average"]
    assert [group.group_index for group in second.groups] == [1, 4]
    assert [group.tree_linkage_method for group in second.groups] == ["complete", "complete"]
    assert [group.group_index for group in third.groups] == [2, 5]
    assert [group.tree_linkage_method for group in third.groups] == ["ward", "ward"]


def test_merge_shards_combines_all_lens_linkage_groups() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        config = _config(root)
        shard_dirs = iter_expected_shard_dirs(root, 3)
        first = make_shard_spec(config, 0)
        second = make_shard_spec(config, 1)
        third = make_shard_spec(config, 2)

        _write_shard(
            shard_dirs[0],
            shard_index=0,
            rows=[
                {
                    "aws_group_index": group.group_index,
                    "aws_group_id": group.group_id,
                    "edge_alpha": 0.001,
                    "sibling_alpha": 0.01,
                    "status": "ok",
                }
                for group in first.groups
            ],
        )
        _write_shard(
            shard_dirs[1],
            shard_index=1,
            rows=[
                {
                    "aws_group_index": group.group_index,
                    "aws_group_id": group.group_id,
                    "edge_alpha": 0.001,
                    "sibling_alpha": 0.01,
                    "status": "ok",
                }
                for group in second.groups
            ],
        )
        _write_shard(
            shard_dirs[2],
            shard_index=2,
            rows=[
                {
                    "aws_group_index": group.group_index,
                    "aws_group_id": group.group_id,
                    "edge_alpha": 0.001,
                    "sibling_alpha": 0.01,
                    "status": "ok",
                }
                for group in third.groups
            ],
        )

        summary = merge_shards(config)

        assert summary.shape[0] == 6
        assert (root / "merged" / SUMMARY_NAME).exists()
        assert (root / "merged" / "aws_kak_lens_linkage_alpha_sweep_manifest.json").exists()
