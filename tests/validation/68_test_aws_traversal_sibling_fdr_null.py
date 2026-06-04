from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
from benchmarks.cloud.aws_traversal_sibling_fdr_null import (
    SHARD_MANIFEST_NAME,
    AwsTraversalSiblingFdrConfig,
    iter_expected_shard_dirs,
    make_shard_spec,
    merge_shards,
    resolve_shard_index,
)
from benchmarks.validation.traversal_sibling_fdr_null import FdrLayer


def _write_shard(
    shard_dir: Path,
    *,
    shard_index: int,
    replicate_indices: tuple[int, ...],
) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(
        [
            {
                "schema_version": "traversal_sibling_fdr_null/v1",
                "layer": FdrLayer.SYNTHETIC_VALID_P.value,
                "case_id": "synthetic_balanced_binary_tree",
                "replicate_index": replicate_index,
                "status": "ok",
                "n_rejections": int(replicate_index == 0),
                "n_false_rejections": int(replicate_index == 0),
            }
            for replicate_index in replicate_indices
        ]
    ).to_csv(shard_dir / "traversal_sibling_fdr_simulations.csv", index=False)
    pd.DataFrame.from_records(
        [
            {
                "layer": FdrLayer.SYNTHETIC_VALID_P.value,
                "mean_fdp": 0.0,
            }
        ]
    ).to_csv(shard_dir / "traversal_sibling_fdr_summary.csv", index=False)
    (shard_dir / SHARD_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "shard_index": shard_index,
                "replicate_indices": list(replicate_indices),
            }
        )
        + "\n"
    )


def test_make_shard_spec_distributes_replicates_by_modulo() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        config = AwsTraversalSiblingFdrConfig(
            layer=FdrLayer.SYNTHETIC_VALID_P,
            case_names=("synthetic_balanced_binary_tree",),
            output_dir=Path(tmpdir),
            replicates=10,
            alpha=0.01,
            base_seed=1,
            shard_count=3,
        )
        first = make_shard_spec(config, 0)
        second = make_shard_spec(config, 1)

    assert first.replicate_indices == (0, 3, 6, 9)
    assert second.replicate_indices == (1, 4, 7)


def test_resolve_shard_index_prefers_explicit_value() -> None:
    assert resolve_shard_index(2, {"AWS_BATCH_JOB_ARRAY_INDEX": "7"}) == 2
    assert resolve_shard_index(None, {"AWS_BATCH_JOB_ARRAY_INDEX": "7"}) == 7


def test_merge_shards_combines_and_recomputes_summary() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        config = AwsTraversalSiblingFdrConfig(
            layer=FdrLayer.SYNTHETIC_VALID_P,
            case_names=("synthetic_balanced_binary_tree",),
            output_dir=root,
            replicates=4,
            alpha=0.01,
            base_seed=1,
            shard_count=2,
        )
        shard_dirs = iter_expected_shard_dirs(root, 2)
        _write_shard(shard_dirs[0], shard_index=0, replicate_indices=(0, 2))
        _write_shard(shard_dirs[1], shard_index=1, replicate_indices=(1, 3))

        outputs = merge_shards(config)

        assert outputs["simulations"].shape[0] == 4
        assert outputs["summary"].iloc[0]["n_simulations"] == 4.0
        assert (root / "merged" / "traversal_sibling_fdr_simulations.csv").exists()
        assert (root / "merged" / "traversal_sibling_fdr_summary.csv").exists()
        assert (root / "merged" / "aws_traversal_sibling_fdr_manifest.json").exists()
