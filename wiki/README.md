---
title: Wiki README
type: control
status: reviewed
updated: 2026-05-24
sources:
  - AGENTS.md
  - wiki/index.md
  - wiki/schema.md
  - wiki/maintenance.md
tags:
  - wiki
  - control
---

# Wiki README

## Summary

This directory is the project memory layer for reusable synthesis. Start with
[[project-overview]] for the project, [[index]] for the content map, and
[[wiki-search]] for operating instructions.

## Details

Pages under `wiki/` should summarize evidence without replacing it. Use local
source paths in frontmatter, preserve stable page names, and update
[[log]] when ingesting or changing durable knowledge.

The active structural contract is documented in [[schema]] and maintained with
[[maintenance]].

## Evidence

- `AGENTS.md` defines the repository-level wiki operating contract.
- `wiki/schema.md` defines page metadata, body sections, and link rules.
- `wiki/maintenance.md` defines routine validation and cleanup.

## Links

- [[index]]
- [[schema]]
- [[maintenance]]
- [[wiki-search]]
- [[log]]
