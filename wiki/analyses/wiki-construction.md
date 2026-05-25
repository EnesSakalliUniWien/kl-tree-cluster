---
title: Wiki Construction
type: analysis
status: reviewed
updated: 2026-05-24
sources:
  - raw/inbox/wiki-construction-brief.md
  - AGENTS.md
  - wiki/schema.md
  - scripts/wiki/lint.py
tags:
  - wiki
  - analysis
---

# Wiki Construction

## Summary

The wiki construction creates a small, local, docs-as-code memory layer with
separate source, synthesis, and control layers. It favors cited Markdown,
manual index coverage, chronological logging, exact search, and structural
lint before adding heavier documentation tooling.

## Details

The source layer preserves raw evidence and project files. The synthesis layer
turns reusable knowledge into pages under `wiki/`. The control layer teaches
future agents how to query, ingest, maintain, and validate the wiki.

The practical quality gate is `make wiki-lint`, which checks parseable
frontmatter, allowed type and status values, ISO dates, local sources, H1s,
required sections, lowercase kebab-case filenames, non-dangling wikilinks, and
coverage in [[index]].

This construction intentionally defers `qmd`, markdownlint, and remark link
validation until the wiki shows a real need for broader search or additional
Markdown checks.

## Evidence

- `raw/inbox/wiki-construction-brief.md` states the requested construction and
  workflows.
- `AGENTS.md` records the operating guide.
- `wiki/schema.md` records the page contract.
- `scripts/wiki/lint.py` implements the local structural validation.

## Links

- [[llm-wiki-pattern]]
- [[wiki-construction-brief]]
- [[github-wiki-structure-research]]
- [[generated-index]]
- [[markdown-validation]]

## Open Questions

- Should index generation be introduced before or after semantic search?
- Which external Markdown checks add enough value to justify new dependencies?
