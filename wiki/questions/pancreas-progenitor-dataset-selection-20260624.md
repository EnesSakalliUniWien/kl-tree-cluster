---
title: Pancreas Progenitor Dataset Selection 2026-06-24
type: question
status: reviewed
updated: 2026-06-24
sources:
  - wiki/sources/pancreas-tbs-inner-node-progenitor-comparison-20260624.md
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/pancreas_progenitor_signature_dataset_summary.csv
tags:
  - benchmark
  - pancreas
  - scrna
  - progenitor
---

# Pancreas Progenitor Dataset Selection 2026-06-24

## Question

Which pancreas scRNA dataset should replace the adult pancreas benchmark for
testing whether Tree-Break Selection internal nodes correspond to progenitor or
developmental precursor states?

## Current State

The adult pancreas benchmark remains useful for clustering mechanics, but it is
a poor biological validation dataset for progenitor-like internal nodes. The
full AnnData object contains only `5/14,693` `NEUROG3+` cells, and the
`2,500`-cell benchmark subset contains `0/2,500` `NEUROG3+` cells, so a
progenitor claim cannot be tested from sampled cells in the current subset.

The preferred next dataset is the Goncalves et al. human fetal pancreas
processed UCSC Cell Browser matrix because it is primary human fetal pancreas
from 7-10 post-conception weeks, is explicitly progenitor-rich, and has direct
matrix and metadata downloads for a fast rerun:

- Dataset browser: <https://cells.ucsc.edu/?ds=human-pancreas-dev>
- Expression matrix:
  <https://cells.ucsc.edu/human-pancreas-dev/fetal-pancreas/exprMatrix.tsv.gz>
- Metadata:
  <https://cells.ucsc.edu/human-pancreas-dev/fetal-pancreas/meta.tsv>

The preferred broader follow-up is Olaniru et al. `GSE197064`, a primary human
fetal pancreas time-course from 12, 13, 14, 15, 18, 19, and 20
post-conception weeks with scRNA-seq plus spatial transcriptomics. This is the
better main biological benchmark after the quick Goncalves rerun because it
adds temporal and spatial context.

Krentz et al. `GSE120522` should be used as a positive-control benchmark rather
than the main biological dataset. It includes a NEUROG3-2A-eGFP hESC reporter
line and thousands of GFP-positive endocrine progenitor cells, which makes it
valuable for checking whether the method can recover a progenitor-enriched
hierarchy when the signal is deliberately present.

The Ma et al. early human pancreas dataset is biologically attractive because
it covers PCW 4-11 and reports endocrine progenitor subclusters, but it is a
heavier accession and processing target. It should follow after the directly
downloadable fetal matrix and GEO reruns unless access and preprocessing are
already solved.

## Evidence

- `wiki/sources/pancreas-tbs-inner-node-progenitor-comparison-20260624.md`
  records that the current TBS inner-node analysis finds lineage-coherent
  mature endocrine hierarchy ancestors, not marker-supported progenitor nodes.
- `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/pancreas_progenitor_signature_dataset_summary.csv`
  records the local `NEUROG3+` scarcity that makes the current adult subset a
  weak progenitor benchmark.
- Goncalves et al. report a human fetal pancreas single-cell atlas at 7-10
  post-conception weeks and identify trunk, proliferating progenitor, and tip
  progenitor populations.
- Olaniru et al. `GSE197064` is public on GEO and provides processed scRNA-seq
  and spatial-transcriptomics data across fetal pancreas time points.
- Krentz et al. `GSE120522` explicitly enriches mouse and hESC-derived
  endocrine progenitor cells, including a NEUROG3 reporter-positive hESC
  sample.

## Links

- [[pancreas-tbs-inner-node-progenitor-comparison-20260624]]
- [[pancreas-scrna-clustering-benchmark-20260623]]
