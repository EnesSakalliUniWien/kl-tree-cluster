"""Run the full benchmark suite with cousin_weighted_wald as the sibling method.

Delegates to benchmarks/full/run.py after toggling config.SIBLING_TEST_METHOD.
Uses subprocess isolation (KL_TE_UMAP_ISOLATE_CASES=1) so each case's UMAP
runs in a fresh process, preventing cumulative numba/llvmlite memory leaks.
Embeddings are cached to disk so re-runs skip recomputation.

Usage:
    python benchmarks/run_weighted_full.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Load shared path bootstrap helper from benchmarks root.
_script_path = Path(__file__).resolve()
_benchmarks_root = (
    _script_path.parent if _script_path.parent.name == "benchmarks" else _script_path.parents[1]
)
if str(_benchmarks_root) not in sys.path:
    sys.path.insert(0, str(_benchmarks_root))
from _bootstrap import ensure_repo_root_on_path


def main() -> None:
    repo_root = ensure_repo_root_on_path(__file__)

    # Enable UMAP case isolation — each case in its own subprocess.
    os.environ["KL_TE_UMAP_ISOLATE_CASES"] = "1"

    # Enable embedding disk cache to avoid keeping UMAP arrays in RAM.
    # Cache directory is created under the repo's benchmarks/results/.
    cache_dir = repo_root / "benchmarks" / "results" / ".embedding_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["KL_TE_EMBEDDING_CACHE_DIR"] = str(cache_dir)

    # Propagate sibling method to spawned subprocesses (spawn context reimports modules
    # from scratch, losing Python-level config changes).
    os.environ["KL_TE_SIBLING_TEST_METHOD"] = "cousin_weighted_wald"

    from kl_clustering_analysis import config

    config.SIBLING_TEST_METHOD = "cousin_weighted_wald"
    print(f">>> Sibling method overridden to: {config.SIBLING_TEST_METHOD}")
    print(f">>> Embedding cache dir: {cache_dir}")
    print(">>> Subprocess isolation: ON")

    from benchmarks.full.run import run_benchmarks

    run_benchmarks()


if __name__ == "__main__":
    main()
