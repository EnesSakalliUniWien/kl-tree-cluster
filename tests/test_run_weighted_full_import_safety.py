"""Import-safety tests for benchmarks/run_weighted_full.py."""

from __future__ import annotations

import importlib
import sys


def test_import_run_weighted_full_has_no_side_effects(monkeypatch):
    monkeypatch.delenv("KL_TE_UMAP_ISOLATE_CASES", raising=False)
    monkeypatch.delenv("KL_TE_EMBEDDING_CACHE_DIR", raising=False)
    monkeypatch.delenv("KL_TE_SIBLING_TEST_METHOD", raising=False)

    # Force a fresh import.
    sys.modules.pop("benchmarks.run_weighted_full", None)
    module = importlib.import_module("benchmarks.run_weighted_full")

    assert "KL_TE_UMAP_ISOLATE_CASES" not in module.os.environ
    assert "KL_TE_EMBEDDING_CACHE_DIR" not in module.os.environ
    assert "KL_TE_SIBLING_TEST_METHOD" not in module.os.environ
    assert hasattr(module, "main")

