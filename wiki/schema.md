---
title: Wiki Schema
type: control
status: reviewed
updated: 2026-05-24
sources:
  - raw/inbox/wiki-construction-brief.md
tags:
  - wiki
  - schema
---

# Wiki Schema

## Summary

This page defines the required metadata, sections, naming, and links for
non-template pages under `wiki/`.

## Details

### Frontmatter

Every non-template wiki page must begin with YAML frontmatter:

```yaml
---
title: Human Title
type: concept
status: draft
updated: 2026-05-24
sources:
  - README.md
tags:
  - method
---
```

Required keys:

- `title`: human-readable page title.
- `type`: one of `project`, `source`, `concept`, `entity`, `analysis`,
  `question`, `tool`, or `control`.
- `status`: one of `draft`, `reviewed`, `stable`, or `deprecated`.
- `updated`: ISO date in `YYYY-MM-DD` form.
- `sources`: YAML list of existing local source paths.
- `tags`: YAML list of short lowercase terms.

### Body Sections

Use exactly one H1. Page types require these section headings:

- `project`, `concept`, `entity`, and `analysis`: `Summary`, `Details`,
  `Evidence`, `Links`, and `Open Questions`.
- `source`: `Summary`, `Key Points`, `Evidence`, and `Links`.
- `question`: `Question`, `Current State`, `Evidence`, and `Links`.
- `tool`: `Summary`, `Usage`, `Evidence`, and `Links`.
- `control`: `Summary`, `Details`, `Evidence`, and `Links`.

### Linking

Use Obsidian-style links to connect wiki pages, for example
`[[project-overview]]`. Link targets resolve by file stem, so every wiki page
stem must be unique.

Use normal Markdown links for external URLs only when needed. Source evidence
should still appear as local paths in frontmatter.

### Naming

Use lowercase kebab-case filenames, such as `projected-wald-statistic.md`.
`README.md` is the only uppercase wiki filename exception.

## Evidence

- `raw/inbox/wiki-construction-brief.md` requires YAML frontmatter, stable
  names, local citations, Obsidian-style links, index coverage, and lint.

## Links

- [[maintenance]]
- [[wiki-search]]
- [[wiki-construction]]
