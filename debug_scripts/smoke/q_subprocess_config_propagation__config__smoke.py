"""
Purpose: Verify config propagation to spawned subprocess workers.
Inputs: Environment variable KL_TE_SIBLING_TEST_METHOD and spawn worker process.
Outputs: Console pass/fail checks for default and overridden subprocess config.
Expected runtime: ~5-20 seconds.
How to run: python debug_scripts/smoke/q_subprocess_config_propagation__config__smoke.py
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def _worker(queue: "mp.Queue") -> None:
    """Simulate what _run_case_worker does: read env var and apply to config."""
    from kl_clustering_analysis import config as _cfg

    _sibling_override = os.environ.get("KL_TE_SIBLING_TEST_METHOD", "")
    if _sibling_override:
        _cfg.SIBLING_TEST_METHOD = _sibling_override

    queue.put(
        {
            "method": _cfg.SIBLING_TEST_METHOD,
            "env": _sibling_override,
        }
    )


def main() -> None:
    # 1) Without env var → should get the default
    ctx = mp.get_context("spawn")
    q1: "mp.Queue" = ctx.Queue()
    p1 = ctx.Process(target=_worker, args=(q1,))
    p1.start()
    p1.join(timeout=10)
    result1 = q1.get()
    assert (
        result1["method"] == "cousin_adjusted_wald"
    ), f"Expected default, got: {result1['method']}"
    print(f"[PASS] Default config in subprocess: {result1['method']}")

    # 2) With env var → should override
    os.environ["KL_TE_SIBLING_TEST_METHOD"] = "cousin_weighted_wald"
    q2: "mp.Queue" = ctx.Queue()
    p2 = ctx.Process(target=_worker, args=(q2,))
    p2.start()
    p2.join(timeout=10)
    result2 = q2.get()
    assert (
        result2["method"] == "cousin_weighted_wald"
    ), f"Expected cousin_weighted_wald, got: {result2['method']}"
    print(f"[PASS] Overridden config in subprocess: {result2['method']}")

    # Cleanup
    del os.environ["KL_TE_SIBLING_TEST_METHOD"]

    print("\n=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    main()
