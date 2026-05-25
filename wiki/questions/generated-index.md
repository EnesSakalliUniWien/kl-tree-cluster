---
title: Generated Index
type: question
status: draft
updated: 2026-05-24
sources:
  - raw/inbox/wiki-construction-brief.md
  - wiki/index.md
  - scripts/wiki/lint.py
tags:
  - wiki
  - question
---

# Generated Index

## Question

Should `wiki/index.md` become generated when page count or manual index drift
makes hand maintenance unreliable?

## Current State

The index is currently hand-maintained. The linter checks whether each
non-template, non-README page appears in [[index]], which catches missing
coverage but does not produce descriptions automatically.

A generated index should wait until manual maintenance becomes visibly noisy:
many repeated index-only edits, frequent missing coverage failures, or enough
pages that hand-written summaries stop being useful.

## Evidence

- `raw/inbox/wiki-construction-brief.md` raises generated index maintenance as
  an open question.
- `scripts/wiki/lint.py` currently validates index coverage.

## Links

- [[wiki-construction]]
- [[maintenance]]
- [[schema]]
