from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from benchmarks.cloud.aws_alpha_grid_search import (
    SHARD_MANIFEST_NAME,
    AwsAlphaGridConfig,
    iter_expected_shard_dirs,
    make_shard_spec,
    merge_shards,
    parse_case_names,
    resolve_shard_index,
)
from benchmarks.validation.alpha_grid_search import alpha_pair_id


def _write_alpha_shard(
    shard_dir: Path,
    *,
    shard_index: int,
    pairs: tuple[tuple[float, float], ...],
) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for edge_alpha, sibling_alpha in pairs:
        rows.append(
            {
                "edge_alpha": edge_alpha,
                "sibling_alpha": sibling_alpha,
                "n_cases": 1,
                "n_ok": 1,
                "n_skip": 0,
                "exact_k": 1,
                "under_split": 0,
                "over_split": 0,
                "mean_ari": 1.0,
                "median_ari": 1.0,
                "mean_abs_cluster_count_error": 0.0,
                "skip_reasons": "{}",
                "elapsed_sec": 0.1,
            }
        )
        result_path = shard_dir / f"{alpha_pair_id(edge_alpha, sibling_alpha)}.csv"
        pd.DataFrame.from_records(
            [
                {
                    "grid_edge_alpha": edge_alpha,
                    "grid_sibling_alpha": sibling_alpha,
                    "case_id": "case",
                    "method": "tbs",
                    "status": "ok",
                }
            ]
        ).to_csv(result_path, index=False)
    pd.DataFrame.from_records(rows).to_csv(shard_dir / "alpha_grid_summary.csv", index=False)
    (shard_dir / SHARD_MANIFEST_NAME).write_text(
        json.dumps({"shard_index": shard_index}) + "\n"
    )


def test_parse_case_names_and_resolve_shard_index_contract() -> None:
    assert parse_case_names("a,b") == ("a", "b")
    assert parse_case_names(None) == ()
    assert resolve_shard_index(2, {"AWS_BATCH_JOB_ARRAY_INDEX": "7"}) == 2
    assert resolve_shard_index(None, {"AWS_BATCH_JOB_ARRAY_INDEX": "7"}) == 7
    with pytest.raises(ValueError, match="Shard index is required"):
        resolve_shard_index(None, {})


def test_make_shard_spec_distributes_alpha_pairs_by_index_modulo() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        config = AwsAlphaGridConfig(
            suite="full",
            case_names=(),
            output_dir=Path(tmpdir),
            edge_alphas=(0.001, 0.003),
            sibling_alphas=(0.01, 0.03),
            shard_count=2,
        )

        first = make_shard_spec(config, 0)
        second = make_shard_spec(config, 1)

    assert first.alpha_pair_indices == (0, 2)
    assert second.alpha_pair_indices == (1, 3)
    assert first.alpha_pairs == ((0.001, 0.01), (0.003, 0.01))
    assert second.alpha_pairs == ((0.001, 0.03), (0.003, 0.03))


def test_merge_shards_combines_all_alpha_pairs_and_results() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        config = AwsAlphaGridConfig(
            suite="full",
            case_names=("case",),
            output_dir=root,
            edge_alphas=(0.001, 0.003),
            sibling_alphas=(0.01, 0.03),
            shard_count=2,
        )
        shard_dirs = iter_expected_shard_dirs(root, 2)
        _write_alpha_shard(shard_dirs[0], shard_index=0, pairs=((0.001, 0.01), (0.003, 0.01)))
        _write_alpha_shard(shard_dirs[1], shard_index=1, pairs=((0.001, 0.03), (0.003, 0.03)))

        outputs = merge_shards(config)

        summary_path = root / "merged" / "alpha_grid_summary.csv"
        results_path = root / "merged" / "alpha_grid_results.csv"
        manifest_path = root / "merged" / "aws_alpha_grid_manifest.json"

        assert outputs["summary"].shape[0] == 4
        assert outputs["results"].shape[0] == 4
        assert summary_path.exists()
        assert results_path.exists()
        assert manifest_path.exists()
