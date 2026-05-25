---
title: Wiki Search
type: tool
status: reviewed
updated: 2026-05-24
sources:
  - AGENTS.md
  - wiki/index.md
  - wiki/maintenance.md
tags:
  - wiki
  - search
---

# Wiki Search

## Summary

Use [[index]] first, then exact local search, then raw or project source files
when the wiki is missing, stale, or contradicted.

## Usage

Start with the content map:

```bash
sed -n '1,220p' wiki/index.md
```

Search wiki and source surfaces:

```bash
rg "term" wiki raw README.md manuscript kl_clustering_analysis benchmarks tests
```

Inspect matches without editing:

```bash
sed -n '1,220p' path/to/file.md
nl -ba path/to/file.py | sed -n '1,160p'
find wiki -maxdepth 3 -type f | sort
```

Before edits, check for unrelated worktree changes:

```bash
git status --short
```

After wiki edits, run:

```bash
make wiki-lint
```

## Evidence

- `AGENTS.md` defines the query, search, and lint workflows.
- `wiki/maintenance.md` defines the threshold for future `qmd` adoption.

## Links

- [[index]]
- [[maintenance]]
- [[schema]]
