"""
Purpose: Verify UMAP embedding disk cache works end-to-end.
Inputs: Temporary cache directory, synthetic matrix, and embedding module helpers.
Outputs: Console pass/fail checks for cache keying, save/load, and disable behavior.
Expected runtime: ~5-30 seconds.
How to run: python debug_scripts/smoke/q_embedding_disk_cache_verification__embedding__smoke.py
"""

from __future__ import annotations

import os
import sys
import shutil
import tempfile
from pathlib import Path

# Ensure repo root is importable.
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np

# --- Setup cache dir ---
cache_dir = Path(tempfile.mkdtemp(prefix="emb_cache_test_"))
os.environ["KL_TE_EMBEDDING_CACHE_DIR"] = str(cache_dir)
# Force deterministic lightweight backends for smoke reliability in CI/local envs
# where UMAP/numba cache initialization may fail.
os.environ.setdefault("KL_TE_EMBEDDING_BACKEND", "pca")
os.environ.setdefault("KL_TE_EMBEDDING_BACKEND_3D", "pca")
print(f"Cache dir: {cache_dir}")

# --- Import after env var is set ---
# Bypass the circular import in benchmarks.shared.plots.__init__.py
# by loading the embedding module directly via its file path.
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "embedding",
    repo_root / "benchmarks" / "shared" / "plots" / "embedding.py",
    submodule_search_locations=[],
)
_mod = _ilu.module_from_spec(_spec)

# Stub out the two imports that would trigger the circular chain.
# The smoke test only needs the caching helpers, not the plotting ones.
import types as _types

_stub_color = _types.ModuleType("kl_clustering_analysis.plot.cluster_color_mapping")
_stub_color.build_cluster_color_spec = lambda *a, **k: None  # type: ignore[attr-defined]
_stub_color.present_cluster_ids = lambda *a, **k: []  # type: ignore[attr-defined]
sys.modules["kl_clustering_analysis.plot.cluster_color_mapping"] = _stub_color

_stub_pdf = _types.ModuleType("benchmarks.shared.util.pdf.layout")
_stub_pdf.PDF_PAGE_SIZE_INCHES = (11.0, 8.5)  # type: ignore[attr-defined]
sys.modules["benchmarks.shared.util.pdf.layout"] = _stub_pdf

_spec.loader.exec_module(_mod)

_cache_key_for_array = _mod._cache_key_for_array
_embedding_cache_dir = _mod._embedding_cache_dir
_fit_embedding_2d = _mod._fit_embedding_2d
_fit_embedding_3d = _mod._fit_embedding_3d
_load_cached_embedding = _mod._load_cached_embedding
_save_cached_embedding = _mod._save_cached_embedding


def main() -> None:
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 5))

    # 1) Verify cache dir discovery
    resolved = _embedding_cache_dir()
    assert resolved is not None, "Cache dir should be set"
    assert str(resolved) == str(cache_dir), f"Expected {cache_dir}, got {resolved}"
    print("[PASS] _embedding_cache_dir() found the env var")

    # 2) Verify key generation
    key = _cache_key_for_array(X, 2, "test_case_42")
    assert key == "embedding_test_case_42_2d", f"Unexpected key: {key}"
    key_hash = _cache_key_for_array(X, 2, None)
    assert key_hash.startswith("embedding_") and key_hash.endswith("_2d"), (
        f"Unexpected hash key: {key_hash}"
    )
    print("[PASS] _cache_key_for_array() works")

    # 3) Compute 2D embedding (will save to cache)
    emb_2d = _fit_embedding_2d(X, cache_key="smoke_2d")
    assert emb_2d.shape == (30, 2), f"Bad shape: {emb_2d.shape}"
    npy_path = cache_dir / "embedding_smoke_2d_2d.npy"
    assert npy_path.exists(), f"Cache file missing: {npy_path}"
    print(f"[PASS] 2D embedding computed and cached ({npy_path.name})")

    # 4) Reload from cache (should not recompute)
    emb_2d_cached = _fit_embedding_2d(X, cache_key="smoke_2d")
    assert np.array_equal(emb_2d, emb_2d_cached), "Cache miss: arrays differ"
    print("[PASS] 2D cache hit — arrays identical")

    # 5) Compute 3D embedding
    emb_3d = _fit_embedding_3d(X, cache_key="smoke_3d")
    assert emb_3d.shape == (30, 3), f"Bad 3D shape: {emb_3d.shape}"
    npy_3d = cache_dir / "embedding_smoke_3d_3d.npy"
    assert npy_3d.exists(), f"Cache file missing: {npy_3d}"
    print(f"[PASS] 3D embedding computed and cached ({npy_3d.name})")

    # 6) Without cache_key — content-addressable hash
    emb_hash = _fit_embedding_2d(X)
    npy_files = list(cache_dir.glob("embedding_*_2d.npy"))
    assert len(npy_files) >= 2, f"Expected ≥2 cache files, found {len(npy_files)}"
    print(f"[PASS] Content-addressable cache key works ({len(npy_files)} files)")

    # 7) Without env var — caching disabled
    del os.environ["KL_TE_EMBEDDING_CACHE_DIR"]
    assert _embedding_cache_dir() is None, "Should be None when env is unset"
    print("[PASS] Caching disabled when env var unset")

    # Cleanup
    shutil.rmtree(cache_dir, ignore_errors=True)
    print("\n=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    main()
