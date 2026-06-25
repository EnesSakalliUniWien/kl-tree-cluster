---
name: router
description: Mex compatibility router. The docs-as-code wiki is the authoritative memory system.
edges:
  - target: ../wiki/index.md
    condition: first read target for all project questions
  - target: ../wiki/schema.md
    condition: when creating or editing wiki pages
  - target: ../wiki/maintenance.md
    condition: when validating or repairing memory
  - target: ../wiki/log.md
    condition: when recording durable project-memory events
  - target: context/stack.md
    condition: when mex needs local notes about scanner limitations
last_updated: 2026-06-23
---

# Session Bootstrap

This repository already has a mature docs-as-code memory layer. Mex is present
only as a compatibility shell around that system.

## Current Project State

**Working:**
- Root `AGENTS.md` defines the project memory contract.
- `wiki/index.md` maps durable project memory.
- `wiki/schema.md` and `scripts/wiki/lint.py` define structural validation.
- `wiki/log.md` is the real chronological memory log.

**Not built here:**
- `.mex/` is not a second source of project facts.
- the mex event log is not the durable chronology for this repository.
- Mex-generated setup prompts are not authoritative without manual correction.

**Known issues:**
- Mex `init --json` currently under-reports PEP 621 dependencies from `pyproject.toml`; consult `pyproject.toml` directly.
- The root Codex bootstrap reads root `AGENTS.md`, so mex routing must defer to the wiki.
- `patterns/` is intentionally empty until a recurring task pattern is worth adding.

## Routing Table

| Task type | Load |
|-----------|------|
| Any project question | `wiki/index.md` |
| Wiki edit or ingest | `AGENTS.md`, `wiki/schema.md`, `wiki/maintenance.md` |
| Search or lookup | `wiki/tools/wiki-search.md`, then `rg` across wiki and source surfaces |
| Durable event logging | `wiki/log.md` |
| Project architecture | `wiki/project-overview.md`, then relevant concept/entity pages |
| Tooling or dependencies | `pyproject.toml`, `Makefile`, `wiki/tools/wiki-search.md` |

## Behavioural Contract

For every task:

1. **CONTEXT** - Read `wiki/index.md` and the relevant linked wiki pages.
2. **SOURCE** - Return to source files, raw captures, tests, benchmarks, or manuscript files when the wiki is stale, incomplete, or contradicted.
3. **BUILD** - Make the scoped change without overwriting unrelated worktree changes.
4. **VERIFY** - Run the relevant tests. After wiki edits, run `make wiki-lint`.
5. **RECORD** - If durable knowledge changed, update the relevant wiki page and append a dated entry to `wiki/log.md`.
