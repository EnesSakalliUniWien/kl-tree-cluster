# GitHub Wiki Structure Research Notes

Captured on 2026-05-24 as design context for the local wiki scaffold.

## Summary

The wiki design borrows from established docs-as-code conventions without
adding runtime dependencies. Foam-style wikilinks provide lightweight local
navigation. GitHub Docs-style frontmatter provides stable page metadata.
Markdown linting and link validation projects show useful future checks, but
the initial project validator should stay dependency-free and local.

## References Considered

- Foam-style Markdown wikilinks for graph navigation.
- GitHub Docs frontmatter patterns for metadata regularity.
- `markdownlint-github` as a possible future prose and Markdown style check.
- `remark-validate-links` as a possible future link validation check.

## Local Design Decision

The current scaffold records these as design references only. The active
runtime contract is `make wiki-lint`, backed by `scripts/wiki/lint.py`.
