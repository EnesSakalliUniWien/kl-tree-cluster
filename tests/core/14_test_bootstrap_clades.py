from __future__ import annotations

import pandas as pd
import pytest
from kl_clustering_analysis.hierarchy_analysis.bootstrap_consensus import (
    _extract_clades,
    bootstrap_consensus,
)


class _DescendantSetTree:
    def nodes(self) -> list[str]:
        return ["root", "left", "right", "a", "b", "c"]

    def out_degree(self, node: str) -> int:
        return {"root": 2, "left": 2, "right": 1, "a": 0, "b": 0, "c": 0}[node]

    def compute_descendant_sets(self, *, use_labels: bool) -> dict[str, frozenset[str]]:
        assert use_labels is True
        return {
            "root": frozenset({"A", "B", "C"}),
            "left": frozenset({"A", "B"}),
            "right": frozenset({"C"}),
            "a": frozenset({"A"}),
            "b": frozenset({"B"}),
            "c": frozenset({"C"}),
        }


def test_extract_clades_uses_precomputed_descendant_sets() -> None:
    assert _extract_clades(_DescendantSetTree()) == {
        frozenset({"A", "B", "C"}),
        frozenset({"A", "B"}),
    }


def test_bootstrap_consensus_does_not_rewrite_zero_alpha_to_default() -> None:
    data = pd.DataFrame(
        [[0, 0], [0, 1], [1, 0], [1, 1]],
        index=["A", "B", "C", "D"],
    )

    with pytest.raises(ValueError, match="alpha"):
        bootstrap_consensus(data, n_boot=1, alpha_local=0.0, random_seed=0)
