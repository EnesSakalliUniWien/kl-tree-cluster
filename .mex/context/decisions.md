---
name: decisions
description: Decision routing for mex; wiki pages and wiki/log.md preserve actual decisions.
triggers:
  - "decision"
  - "why"
  - "rationale"
edges:
  - target: ../wiki/log.md
    condition: chronological decisions and durable memory events
  - target: ../wiki/questions/open-mathematical-questions.md
    condition: method-level open questions and decision state
last_updated: 2026-06-23
---

# Decisions

Project decisions live in cited wiki pages and `wiki/log.md`. Do not create a
parallel `.mex` decision log. When rationale matters, append a dated entry to
`wiki/log.md` and update the relevant wiki analysis, question, concept, source,
or tool page.
