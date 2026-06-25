---
name: conventions
description: Pointers to the repository conventions that are enforced outside mex.
triggers:
  - "convention"
  - "style"
  - "wiki edit"
  - "review"
edges:
  - target: ../AGENTS.md
    condition: authoritative operating guide
  - target: ../wiki/schema.md
    condition: wiki page schema and source citation rules
  - target: ../wiki/maintenance.md
    condition: validation and cleanup rules
last_updated: 2026-06-23
---

# Conventions

## Naming

- Wiki files use lowercase kebab-case stems except `wiki/README.md`.
- Wiki links use `[[page-stem]]` and stems must be unique across `wiki/`.
- Python code follows existing package/test naming and `ruff` configuration.

## Structure

- Primary evidence stays in source files, raw captures, data notes, manuscripts, tests, benchmarks, and reports.
- Reusable synthesis lives in `wiki/`.
- New external documents enter under `raw/inbox/` before wiki summarization.
- `.mex/` stores routing compatibility only.

## Verify Checklist

- Read `wiki/index.md` before project-memory work.
- Check raw/source evidence before updating synthesis.
- After wiki edits, run `make wiki-lint`.
- For memory routing changes, run `pytest tests/wiki/test_memory_contract.py -q`.
- Record durable memory changes in `wiki/log.md`.
