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

Read `pyproject.toml` for the authoritative runtime, dependency groups, and
tool configuration. Mex scanning may under-report PEP 621 dependencies, so this
compatibility file intentionally does not copy the dependency list.
