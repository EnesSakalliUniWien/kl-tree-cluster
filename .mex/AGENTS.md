---
name: agents
description: Mex bridge for Codex. The root AGENTS.md and wiki remain authoritative.
last_updated: 2026-06-23
---

# Tree-Break Selection

## What This Is
Tree-Break Selection is a Python research codebase for hierarchy decomposition, projected-Wald gate testing, benchmark diagnostics, manuscript work, and a docs-as-code project memory wiki.

## Non-Negotiables
- Treat `wiki/` as the authoritative durable memory layer; do not duplicate facts into `.mex/` unless they are routing instructions.
- Read `wiki/index.md` before answering project questions, then follow the relevant wiki links and source citations.
- Preserve raw evidence. Repair citations to existing local evidence rather than fabricating raw files.
- After wiki edits, run `make wiki-lint`.
- Record durable memory changes in `wiki/log.md`, not the mex event log.

## Commands
- Wiki lint: `make wiki-lint`
- Memory tests: `pytest tests/wiki/test_memory_contract.py -q`
- Tests: `pytest`
- Lint: `ruff check benchmarks tree_break_selection scripts tests`
- Build/install check: `python -m pip install -e '.[dev]'`

## Navigation
Read `.mex/ROUTER.md` only as a mex compatibility bridge. For real project memory, start with `wiki/index.md`.
