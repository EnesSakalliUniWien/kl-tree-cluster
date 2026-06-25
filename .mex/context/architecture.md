---
name: architecture
description: Pointer to the authoritative architecture memory in wiki/.
triggers:
  - "architecture"
  - "system design"
  - "project overview"
edges:
  - target: ../wiki/project-overview.md
    condition: authoritative repository map
  - target: ../wiki/index.md
    condition: route to method, entity, source, and analysis pages
last_updated: 2026-06-23
---

# Architecture

## System Overview

The authoritative overview is `wiki/project-overview.md`. In short:
`tree_break_selection/` contains package code, `benchmarks/` contains benchmark
and diagnostic runners, `tests/` validates behavior, `manuscript/` holds paper
material, `raw/` and `reports/` hold evidence, and `wiki/` stores cited durable
synthesis.

## Key Components

- `tree_break_selection/tree/` - core `PosetTree` structures and distributions.
- `tree_break_selection/hierarchy_analysis/` - decomposition, gates, traversal, and projected-Wald statistics.
- `benchmarks/` - method comparisons, diagnostic panels, generated result artifacts, and report tooling.
- `wiki/` - project memory index, source summaries, concepts, analyses, questions, and operating instructions.

## External Dependencies

Use `pyproject.toml` as the dependency source of truth. Mex scanner output is
not sufficient for dependency review in this Python project.

## What Does NOT Exist Here

- `.mex/` does not replace the wiki.
- the mex event log does not replace `wiki/log.md`.
