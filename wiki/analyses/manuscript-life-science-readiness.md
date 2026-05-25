---
title: Manuscript Life-Science Readiness
type: analysis
status: reviewed
updated: 2026-05-25
sources:
  - manuscript/main.tex
  - manuscript/sections/introduction/section.tex
  - manuscript/sections/experiments/section.tex
  - manuscript/guides/full_method_logic_map.md
  - data/feature_matrices/README.md
  - data/feature_matrices/README_HC_feature_matrix_GO_CC.md
  - data/reference/README.md
tags:
  - manuscript
  - life-science
  - validation
---

# Manuscript Life-Science Readiness

## Summary

The manuscript is currently coherent as a statistical methods draft, but it is
not yet ready to make a life-science application claim. The repo contains
biological feature matrices for CMS, hypertrophic cardiomyopathy, GO terms, and
Reactome pathways, plus an endotype reference table. Those files provide a
credible application surface, but the manuscript evaluation section still marks
results as prospective and does not lock a biological run, manifest, figure, or
interpretation table. The next scientific-writing step should therefore be a
gap-preserving application plan or a completed real-data section backed by
tracked evidence, not a polished biological claim.

## Details

The scientific-writing constraint is that the final manuscript should use
flowing prose and avoid bullet lists in the paper body except where method
formatting requires them. The present evaluation section still contains a
prospective plan and explicit TODO markers, so it should not be rewritten into
results prose until empirical artifacts exist. A safe revision can improve the
section order and claim language, but it must keep the missing-result markers.

The life-science routing check separates three possible application tracks.
The CMS matrices can support a colorectal-cancer pathway/subtype application if
the run is linked to a CMS reference and a stable output manifest. The
hypertrophic-cardiomyopathy Reactome matrix can support a gene-pathway module
application if cluster assignments, enrichment summaries, and pathway
interpretation are promoted into tracked evidence. The Julia GO matrices and
endotype table can support an endotype-comparison application if the sample or
gene identifiers, labels, and evaluation endpoint are made explicit.

None of these tracks should be described as validated biological discovery in
the current manuscript. The accurate claim is narrower: the repository has
biological input surfaces suitable for real-data demonstration once a locked
analysis is run and cited. A submission-oriented draft needs one selected
application, a manifest tying commit, configuration, seeds, input matrix,
output files, figures, and interpretation tables together, and a short
limitations paragraph explaining that pathway annotations are structured,
overlapping, and not independent Bernoulli features.

## Evidence

- `manuscript/main.tex` presents KL-TE as a methods draft and does not include
  a completed real-data result section.
- `manuscript/sections/experiments/section.tex` explicitly says the evaluation
  section is prospective and requires locked outputs before claims enter the
  paper.
- `manuscript/guides/full_method_logic_map.md` lists real-data
  interpretability and method-constant validation as missing before
  submission.
- `data/feature_matrices/README.md` records the available CMS, GO, Reactome,
  HC, and Julia feature matrices.
- `data/feature_matrices/README_HC_feature_matrix_GO_CC.md` documents the HC
  Reactome/pathway matrix and the older exploratory biological analysis
  surface.
- `data/reference/README.md` records the endotype reference table available for
  interpretation.

## Links

- [[project-overview]]
- [[kl-te-method]]
- [[oracle-gate-path-diagnostic]]

## Open Questions

- Which one biological application should be promoted into the first
  manuscript-ready real-data result?
- Should the first application emphasize CMS subtype recovery, HC pathway
  modules, or Julia GO/endotype agreement?
- Which generated outputs should be promoted into a locked
  manuscript-results manifest?
