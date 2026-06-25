---
name: stack
description: Tooling and dependency notes for mex users; pyproject.toml remains authoritative.
triggers:
  - "dependency"
  - "tooling"
  - "mex init"
  - "scanner"
edges:
  - target: ../pyproject.toml
    condition: authoritative dependency and pytest/ruff configuration
  - target: ../wiki/tools/wiki-search.md
    condition: authoritative local search commands
last_updated: 2026-06-23
---

# Stack

## Core Technologies

- Python `>=3.11`.
- Package metadata and dependency groups live in `pyproject.toml`.
- Tests use `pytest`; wiki validation uses `make wiki-lint`.
- Lint configuration is `ruff` in `pyproject.toml`.

## Key Libraries

The project uses scientific Python libraries including NumPy, pandas, SciPy,
scikit-learn, networkx, matplotlib, plotly, seaborn, pydiffmap, numba, and
statsmodels. Optional benchmark and visualization groups add graph clustering,
UMAP, scikit-bio, Kaleido, and notebook tooling.

## Mex Scanner Limitation

`npx mex-agent init --json` currently recognizes `pyproject.toml` but reports
empty dependency/script maps for this repository. Treat mex scanner output as a
coarse folder/entry-point brief only; read `pyproject.toml` directly before
making setup, dependency, or command decisions.
