# Wiki Construction Brief

Captured on 2026-05-24 from the project wiki construction request.

## Summary

The project wiki is a docs-as-code LLM memory layer. Immutable or primary
material stays in `raw/` and the project source tree, while reusable synthesis
lives under `wiki/` as cited Markdown pages with stable names, YAML
frontmatter, Obsidian-style links, index coverage, a chronological log, and a
local structural lint.

## Required Layers

The source layer contains `raw/`, `raw/inbox/`, `raw/assets/`, and primary
project files in manuscript, supplement, software, verification, assets,
references, and notes areas.

The synthesis layer contains `wiki/`, with subdirectories for sources,
concepts, entities, analyses, questions, candidates, tools, and templates.

The control layer contains `AGENTS.md`, `wiki/schema.md`,
`wiki/maintenance.md`, `wiki/index.md`, `wiki/log.md`, templates, and
`make wiki-lint`.

## Required Tooling

The toolchain should remain small: `rg`, `sed`, `nl`, `find`,
`git status --short`, `make wiki-lint`, and `scripts/wiki/lint.py`. The
documented future upgrade path is `qmd` for local keyword, vector, and hybrid
search when exact search and manual index coverage are no longer enough.

## Required Scaffold

Create `raw/`, `raw/inbox/`, `raw/assets/`, `wiki/`, wiki control pages,
wiki subdirectories, templates, root `AGENTS.md`, a structural linter, a
`wiki-lint` target, seed project and method pages, index entries, and log
entries.

## Required Workflow

For queries, read `wiki/index.md`, then relevant wiki pages, then raw or
project sources if the wiki is missing, stale, or contradicted.

For ingests, read the source, summarize it in `wiki/sources/`, update affected
synthesis pages, update `wiki/index.md`, append `wiki/log.md`, and run
`make wiki-lint`.

For health checks, look for contradictions, orphan pages, repeated concepts
without pages, stale summaries, missing citations, and open questions that now
have enough evidence to answer.

## Open Questions

- Should `wiki/index.md` become generated when page count or manual index drift
  makes hand maintenance unreliable?
- Should the repository eventually add markdownlint or remark validation in
  addition to the current dependency-free structural lint?
