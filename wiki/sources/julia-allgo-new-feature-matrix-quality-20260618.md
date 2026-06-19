---
title: Julia allGO New Feature Matrix Quality 2026-06-18
type: source
status: reviewed
updated: 2026-06-18
sources:
  - data/feature_matrices/feature_matrix_julia_allGO_new.tsv
  - scripts/analysis/feature_matrix_quality_analysis.py
  - raw/assets/benchmark-results/julia_allGO_new_feature_matrix_quality_20260618/README.md
  - raw/assets/benchmark-results/julia_allGO_new_feature_matrix_quality_20260618/matrix_quality_summary.csv
  - raw/assets/benchmark-results/julia_allGO_new_feature_matrix_quality_20260618/go_term_supports.csv
  - raw/assets/benchmark-results/julia_allGO_new_feature_matrix_quality_20260618/gene_annotation_burden.csv
  - raw/assets/benchmark-results/julia_allGO_new_feature_matrix_quality_20260618/entropy_summary.csv
  - raw/assets/benchmark-results/julia_allGO_new_feature_matrix_quality_20260618/go_term_entropy_bins.csv
  - raw/assets/benchmark-results/julia_allGO_new_feature_matrix_quality_20260618/gene_row_entropy_bins.csv
  - raw/assets/benchmark-results/julia_allGO_new_feature_matrix_quality_20260618/duplicate_go_term_patterns.csv
  - raw/assets/benchmark-results/julia_allGO_new_feature_matrix_quality_20260618/feature_filter_sensitivity.csv
  - raw/assets/benchmark-results/julia_allGO_new_feature_matrix_quality_20260618/feature_matrix_quality_report.pdf
tags:
  - source
  - julia
  - allgo
  - feature-matrix
  - quality
---

# Julia allGO New Feature Matrix Quality 2026-06-18

## Summary

The `feature_matrix_julia_allGO_new.tsv` input matrix is technically valid for
the allGO-new runs: it has `602` genes, `6368` GO-term columns, binary values
only, no missing values, no all-zero genes, and no all-zero GO terms. The main
quality concerns are biological-annotation structure rather than file validity:
the matrix is sparse, rare-term-heavy, has many exact duplicate GO-term
patterns, and has a large gene annotation-burden gradient.

## Key Points

- Matrix density is `0.024926856`, with `95558` active gene-term entries.
- GO-term support is highly skewed: `679` terms occur in exactly one gene,
  `1280` terms occur in at most two genes, and `2583` terms occur in at most
  five genes.
- Marginal GO-term entropy is low: mean Bernoulli entropy is `0.144802` bits,
  median is `0.101883` bits, and `3180/6368` GO terms are at or below `0.10`
  bits.
- Only `223/6368` GO terms have entropy at least `0.50` bits, and only
  `15/6368` reach at least `0.80` bits. The most entropic GO term has
  `0.931521` bits.
- Total marginal GO-term entropy is `922.099352` bits across `6368` terms,
  which is low relative to a balanced binary feature matrix of the same width.
- No GO term occurs in at least half of genes, so the broadest terms do not
  dominate by prevalence alone.
- Gene annotation burden is uneven: the median gene has `112.5` active terms
  while the mean is `158.734219`; the highest-burden genes include `PIK3R1`
  with `782` terms and `CBL` with `771` terms.
- Gene row entropy is also low because every gene has a small active-term
  fraction relative to `6368` GO terms: median row entropy is `0.128130` bits
  and the maximum is `0.537360` bits.
- Active annotation entries are moderately concentrated across genes: the
  entropy of active-entry mass across genes is `8.692633` bits, equivalent to
  about `413.755077` evenly annotated genes out of `602`.
- Active annotation entries are also concentrated across GO terms: the
  entropy of active-entry mass across terms is `11.731995` bits, equivalent to
  about `3401.592331` evenly supported GO terms out of `6368`.
- Exact duplicate gene patterns are rare: only one duplicate gene-pattern group
  was found, `CFHR1` and `CFHR2`.
- Exact duplicate GO-term patterns are common: `459` duplicate GO-term pattern
  groups were found. These repeated columns can overweight identical annotation
  evidence unless collapsed or downweighted.
- Pairwise gene similarity is mostly weak. Median cosine similarity is
  `0.027254939`, median Jaccard similarity is `0.010752688`, and the 95th
  percentile cosine similarity is `0.213200716`.
- The SVD spectrum is distributed rather than low rank: the first `44`
  components explain about half of the variance, and the `80`-component run
  estimates that slightly more than `80` components are needed to clearly pass
  the `80%` cumulative-variance threshold.
- Filtering singleton GO terms keeps `5689/6368` terms but creates one
  zero-active gene row. Filtering terms with support below three keeps
  `5088/6368` terms and still leaves one zero-active gene row.

## Interpretation

The matrix is usable, but raw GO annotation information criteria can reward
overfragmented trees because many GO terms are both rare and low-entropy, and
many GO columns encode identical gene sets. Quality-aware tree ranking should
therefore keep raw GO-IC as an audit field while also accounting for singleton
clusters, coherent cluster fraction, largest-cluster fraction, and
within-cluster embedding quality.

The gene annotation-burden gradient is also a geometry risk. High-burden genes
can drive cosine and TF-IDF structure, while low-burden genes are vulnerable to
becoming zero rows after rare-term filtering. Any filtered rerun should record
which genes become zero-active rows and whether exact duplicate GO columns were
collapsed.

## Evidence

- `matrix_quality_summary.csv` stores validity checks, support summaries,
  duplicate-pattern counts, pairwise gene similarity percentiles, and SVD
  cumulative-variance thresholds.
- `entropy_summary.csv`, `go_term_entropy_bins.csv`, and
  `gene_row_entropy_bins.csv` store the marginal and concentration entropy
  diagnostics.
- `feature_matrix_quality_report.pdf` stores the seven-page visual quality report.
- `go_term_supports.csv` and `gene_annotation_burden.csv` store row and column
  support tables for follow-up filtering.
- `duplicate_go_term_patterns.csv` stores exact duplicate GO-term pattern
  groups.
- `feature_filter_sensitivity.csv` stores min-support and max-prevalence
  filtering sensitivity rows.

## Links

- [[julia-allgo-new-go-ic-tree-summary-plots-20260618]]
- [[julia-allgo-new-method-version-tree-matrix-20260618]]
- [[julia-allgo-new-c2ef-cosine-subspace-validation-20260618]]
