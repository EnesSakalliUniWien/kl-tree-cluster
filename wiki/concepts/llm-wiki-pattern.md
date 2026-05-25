---
title: LLM Wiki Pattern
type: concept
status: reviewed
updated: 2026-05-24
sources:
  - raw/inbox/wiki-construction-brief.md
  - AGENTS.md
  - wiki/schema.md
tags:
  - wiki
  - memory
---

# LLM Wiki Pattern

## Summary

The LLM Wiki pattern separates evidence from synthesis so future agents can
answer project questions by reading a stable index and cited Markdown pages
before reopening primary sources.

## Details

The pattern has three layers. The source layer preserves raw captures and
project files. The synthesis layer stores reusable pages for sources,
concepts, entities, analyses, questions, and tools. The control layer defines
how pages are named, cited, linked, indexed, logged, and linted.

The key operating rule is evidence-first synthesis: wiki pages summarize and
connect sources, but they do not replace the source files listed in
frontmatter.

## Evidence

- `raw/inbox/wiki-construction-brief.md` defines the three-layer construction.
- `AGENTS.md` defines the agent workflow.
- `wiki/schema.md` defines the page contract.

## Links

- [[wiki-construction]]
- [[wiki-construction-brief]]
- [[schema]]
- [[maintenance]]

## Open Questions

- When should the project add semantic search instead of relying on exact
  search plus index coverage?
