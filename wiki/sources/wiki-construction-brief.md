---
title: Wiki Construction Brief
type: source
status: reviewed
updated: 2026-05-24
sources:
  - raw/inbox/wiki-construction-brief.md
tags:
  - source
  - wiki
---

# Wiki Construction Brief

## Summary

The brief defines a three-layer project wiki: immutable source material in
`raw/` and project files, reusable synthesis in `wiki/`, and a control layer
with operating guidance, schema, maintenance instructions, index, log,
templates, and lint.

## Key Points

- The wiki should be a docs-as-code LLM memory layer, not a disposable summary.
- The active tooling should stay small: `rg`, local file readers,
  `git status --short`, `make wiki-lint`, and a dependency-free Python linter.
- Ingest work should update source summaries, synthesis pages, index coverage,
  the chronological log, and lint status.
- Query work should start with [[index]] and escalate to raw or project sources
  only when necessary.
- The two initial open questions are [[generated-index]] and
  [[markdown-validation]].

## Evidence

- `raw/inbox/wiki-construction-brief.md` is the captured primary brief.

## Links

- [[wiki-construction]]
- [[llm-wiki-pattern]]
- [[schema]]
- [[maintenance]]
