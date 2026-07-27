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

The authoritative repository map is `wiki/project-overview.md`; use
`wiki/index.md` to reach the cited component, concept, source, and analysis
pages. This compatibility file intentionally carries no duplicate architecture
description.
