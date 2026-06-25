from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_wiki_lint_contract_passes() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/wiki/lint.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "wiki-lint passed:" in result.stdout


def test_mex_bootstrap_delegates_to_wiki_memory_contract() -> None:
    root_agents = read("AGENTS.md")
    agents = read(".mex/AGENTS.md")
    router = read(".mex/ROUTER.md")

    assert ".mex/" in root_agents
    assert "compatibility" in root_agents.lower()
    assert "wiki/log.md" in root_agents

    combined = agents + "\n" + router
    assert "[Project Name]" not in combined
    assert "[YYYY-MM-DD]" not in combined
    assert "wiki/index.md" in combined
    assert "wiki/log.md" in combined
    assert "make wiki-lint" in combined
    assert "mex log" not in combined


def test_mex_pattern_index_has_no_placeholder_links() -> None:
    index = read(".mex/patterns/INDEX.md")

    assert "filename.md" not in index
    assert "add-api-client.md" not in index
    assert "debug-pipeline.md" not in index
    assert "crud-operations.md" not in index


def _wiki_title(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"^title:\s*(.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else path.stem


def _lookup(term: str) -> list[str]:
    tokens = [token.lower() for token in re.findall(r"[a-zA-Z0-9]+", term)]
    hits: list[tuple[int, str]] = []
    for path in sorted((ROOT / "wiki").rglob("*.md")):
        if "templates" in path.relative_to(ROOT / "wiki").parts:
            continue
        rel = path.relative_to(ROOT).as_posix()
        haystack = f"{path.stem} {_wiki_title(path)} {path.read_text(encoding='utf-8')[:4000]}".lower()
        if all(token in haystack for token in tokens):
            score = sum(path.stem.lower().count(token) + _wiki_title(path).lower().count(token) for token in tokens)
            hits.append((score, rel))
    return [rel for score, rel in sorted(hits, key=lambda item: (-item[0], item[1]))]


def test_lookup_finds_core_method_elements() -> None:
    expected = {
        "PosetTree": "wiki/entities/poset-tree.md",
        "projected Wald": "wiki/concepts/projected-wald-statistic.md",
        "top down traversal": "wiki/concepts/top-down-traversal.md",
        "guarded continuous covariance": "wiki/sources/guarded-continuous-covariance-implementation-20260623.md",
        "GO annotation feature matrix": "wiki/tools/go-annotation-feature-matrix-pipeline.md",
    }

    for query, expected_path in expected.items():
        hits = _lookup(query)
        assert expected_path in hits[:5], f"{query!r} returned {hits[:10]}"
