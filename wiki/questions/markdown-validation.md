---
title: Markdown Validation
type: question
status: draft
updated: 2026-05-24
sources:
  - raw/inbox/wiki-construction-brief.md
  - wiki/maintenance.md
  - scripts/wiki/lint.py
tags:
  - wiki
  - question
---

# Markdown Validation

## Question

Should the repository add markdownlint or remark validation in addition to the
current dependency-free structural lint?

## Current State

The current linter validates the local wiki contract but does not enforce
general Markdown style or all Markdown links. That is deliberate: the initial
tooling should be dependency-free and easy to run in a local project checkout.

Add markdownlint or remark validation only when the wiki begins to accumulate
style drift, broken non-wiki links, or Markdown rendering issues that the
structural linter cannot catch.

## Evidence

- `raw/inbox/wiki-construction-brief.md` raises Markdown validation as an open
  question.
- `raw/inbox/github-wiki-structure-research.md` records markdownlint and remark
  validation as future design references.
- `scripts/wiki/lint.py` implements only local structural checks.

## Links

- [[github-wiki-structure-research]]
- [[wiki-construction]]
- [[maintenance]]
