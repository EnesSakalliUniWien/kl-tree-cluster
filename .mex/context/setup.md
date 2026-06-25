---
name: setup
description: Setup pointers for mex; root project files remain authoritative.
triggers:
  - "setup"
  - "install"
  - "run"
  - "environment"
edges:
  - target: ../pyproject.toml
    condition: dependencies, optional groups, pytest, and ruff config
  - target: ../Makefile
    condition: wiki lint command
  - target: ../README.md
    condition: project overview and implementation map
last_updated: 2026-06-23
---

# Setup

## Prerequisites

- Python 3.11 or newer.
- Project dependencies from `pyproject.toml`.
- `pytest` and `ruff` from the `dev` optional dependency group.

## Common Commands

- `python -m pip install -e '.[dev]'` - editable install with development tools.
- `pytest` - full test suite.
- `pytest tests/wiki/test_memory_contract.py -q` - memory and lookup contract tests.
- `make wiki-lint` - structural wiki validation.
- `ruff check benchmarks tree_break_selection scripts tests` - configured lint target.

## Common Issues

- Mex scanner output currently omits Python dependency details; read `pyproject.toml`.
- Wiki lint failures usually mean a source citation no longer points to existing local evidence.
