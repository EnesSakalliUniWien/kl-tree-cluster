---
name: patterns-readme
description: Pattern policy for this repository's mex bridge.
triggers:
  - "pattern"
  - "runbook"
edges:
  - target: ../wiki/tools/wiki-search.md
    condition: default search and lookup workflow
  - target: ../wiki/maintenance.md
    condition: wiki maintenance and validation workflow
last_updated: 2026-06-23
---

# Patterns

This directory is intentionally sparse. The repository already uses `wiki/` for
durable operating knowledge, source summaries, tools, questions, and analyses.

Create a `.mex/patterns/` file only when a recurring agent workflow needs a
short mex-specific runbook that cannot live cleanly in `wiki/tools/` or another
wiki page. After adding one, update `INDEX.md` and run:

```bash
npx mex-agent check --json
pytest tests/wiki/test_memory_contract.py -q
```
