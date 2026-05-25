---
title: Wiki Maintenance
type: control
status: reviewed
updated: 2026-05-24
sources:
  - raw/inbox/wiki-construction-brief.md
  - scripts/wiki/lint.py
tags:
  - wiki
  - maintenance
---

# Wiki Maintenance

## Summary

Maintain the wiki by keeping citations local, links non-dangling, index
coverage current, and the chronology append-only.

## Details

### Routine Checks

- Run `make wiki-lint` after wiki edits.
- Update [[index]] whenever adding, renaming, or promoting a durable page.
- Append [[log]] for scaffold, ingest, analysis, implementation, review,
  verification, and maintenance events.
- Preserve frontmatter `sources` when revising a page.
- Repair broken wikilinks immediately.
- Promote useful pages from `wiki/candidates/` once they become reusable.

### Cleanup Cadence

During a health check, look for contradictions, orphan pages, repeated concepts
without pages, stale summaries, missing citations, and open questions that now
have enough evidence to answer.

### Search Threshold

Use `rg` and [[index]] first. Defer `qmd` until page count, repeated failed
recall, or manual index drift makes keyword-only search unreliable.

### Future qmd Upgrade

`qmd` is not part of the current runtime contract and is not assumed to be
installed. Before adopting it, verify the local CLI and record exact project
commands here:

```bash
command -v qmd
qmd --help
```

Only after confirming the installed syntax, add the concrete keyword, vector,
and hybrid search commands to this section. Keep `rg` as the fallback search
path.

## Evidence

- `scripts/wiki/lint.py` implements the local structural checks.
- `raw/inbox/wiki-construction-brief.md` defines ingest, query, and health
  check workflows.

## Links

- [[schema]]
- [[wiki-search]]
- [[generated-index]]
- [[markdown-validation]]
