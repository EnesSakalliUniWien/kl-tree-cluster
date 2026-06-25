# LLM Wiki Operating Guide

## Project Context

This repository develops the Tree-Break Selection clustering method and software. The durable
memory layer is a docs-as-code wiki: primary evidence remains in source files,
raw captures, data notes, manuscripts, tests, benchmarks, and reports, while
reusable synthesis lives in `wiki/`.

The wiki is not a replacement for the project source tree. Treat it as a cited
index of what is known, where the evidence lives, and which questions remain
open.

## Mex Compatibility

The `.mex/` directory is a compatibility bridge for the `mex-agent` CLI, not a
second memory system. Root `AGENTS.md`, `wiki/index.md`, `wiki/schema.md`,
`wiki/maintenance.md`, and `wiki/log.md` remain authoritative.

When using mex, keep `.mex/` as a thin router into the wiki. Do not record
durable project events in mex event files; append them to `wiki/log.md`.
Mex scanner output may under-report this Python project's `pyproject.toml`
dependencies, so inspect `pyproject.toml` directly for setup and dependency
decisions.

## Source Surfaces

Primary evidence is read from these surfaces before creating or updating wiki
synthesis:

- `raw/`, `raw/inbox/`, and `raw/assets/` for captured source material.
- `README.md`, `CHANGELOG.md`, and root configuration files for current project
  intent.
- `manuscript/` for the paper, derivations, terminology, and submission gaps.
- `tree_break_selection/`, `benchmarks/`, `tests/`, `scripts/`, and
  `notebooks/` for implementation and validation behavior.
- `data/`, `reports/`, `docs/`, and `local_data/` for datasets, outputs,
  audits, and local notes.
- root `analysis/` is ignored local workspace if present; do not treat it as
  shared source unless a tracked file explicitly documents it.

Do not overwrite or normalize raw evidence while synthesizing it. If a new
external document enters the project, place it under `raw/inbox/` first, then
summarize it into the wiki.

## Wiki Layout

- `wiki/index.md` is the first read target for project questions.
- `wiki/log.md` is an append-only chronology of scaffold, ingest, query,
  analysis, implementation, review, verification, and maintenance events.
- `wiki/schema.md` defines page metadata, section, link, and naming rules.
- `wiki/maintenance.md` defines routine checks and search escalation.
- `wiki/sources/` holds summaries of individual primary sources.
- `wiki/concepts/` holds reusable ideas and method concepts.
- `wiki/entities/` holds named models, classes, artifacts, and datasets.
- `wiki/analyses/` holds reusable arguments and cross-source synthesis.
- `wiki/questions/` holds open questions and decision records.
- `wiki/tools/` holds operating instructions.
- `wiki/templates/` holds starter pages and is excluded from lint.

## Page Conventions

Every non-template wiki page must have YAML frontmatter, one H1, stable
lowercase kebab-case file naming, local source citations in `sources`, and
Obsidian-style links for wiki pages. Prefer stable conceptual names over dates
unless the page is explicitly an event log.

Use `[[page-name]]` for wiki links. File stems must be unique across `wiki/`
so links can resolve without directory prefixes.

## Ingest Workflow

1. Save new primary material in `raw/inbox/` or identify the existing project
   source path.
2. Read the source directly.
3. Create or update a `wiki/sources/` summary for the individual source.
4. Update affected concept, entity, analysis, or question pages with citations.
5. Add or revise one-line coverage in `wiki/index.md`.
6. Append a dated entry to `wiki/log.md`.
7. Run `make wiki-lint` and fix structural failures before stopping.

## Query Workflow

1. Read `wiki/index.md`.
2. Open the most relevant wiki pages.
3. Use `rg` for exact terms, symbols, paths, and wikilinks.
4. Return to raw or project source files when the wiki is missing, stale, or
   contradicted.
5. If the query reveals durable knowledge, update the wiki and log the change.

## Search Workflow

Start with:

```bash
rg "term" wiki raw README.md manuscript tree_break_selection benchmarks tests
```

Use `sed`, `nl`, and `find` for local inspection. Use `git status --short`
before editing so unrelated worktree changes are not overwritten.

Defer `qmd` until the hand-maintained index plus exact search stops being
enough for recall.

## Lint Workflow

Run:

```bash
make wiki-lint
```

The command validates frontmatter, page type and status values, ISO dates,
source paths, required sections, filename stability, wikilinks, and index
coverage.

## Seed Topic

The initial wiki seed covers the Tree-Break Selection method, `PosetTree`,
`TreeDecomposition`, projected-Wald testing, top-down traversal, and the wiki
construction pattern itself.
