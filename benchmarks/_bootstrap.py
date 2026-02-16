"""Path bootstrap helpers for direct benchmark script execution."""

from __future__ import annotations

import sys
from pathlib import Path


def find_repo_root(from_file: str | Path) -> Path:
    """Find the repository root using stable package-directory markers."""
    file_path = Path(from_file).resolve()
    for parent in (file_path.parent, *file_path.parents):
        if (parent / "benchmarks").is_dir() and (parent / "kl_clustering_analysis").is_dir():
            return parent
    raise RuntimeError(f"Could not locate repository root from: {from_file}")


def ensure_repo_root_on_path(from_file: str | Path) -> Path:
    """Ensure the repository root is importable and return it."""
    repo_root = find_repo_root(from_file)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


__all__ = ["find_repo_root", "ensure_repo_root_on_path"]
