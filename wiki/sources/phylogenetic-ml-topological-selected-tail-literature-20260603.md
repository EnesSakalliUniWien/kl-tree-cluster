---
title: Phylogenetic ML Topological Selected Tail Literature 2026-06-03
type: source
status: reviewed
updated: 2026-06-03
sources:
  - raw/inbox/phylogenetic-ml-topological-selected-tail-literature-20260603.md
tags:
  - source
  - literature
  - selection
  - phylogenetics
  - topology
---

# Phylogenetic ML Topological Selected Tail Literature 2026-06-03

## Summary

The literature scan supports a stricter interpretation of the remaining
undefined selected-tail contexts. Root and medium/large parent contexts should
not be treated as simple parent-size extrapolations from the admissible
small-parent, high-edge-action cases. Selective-inference work says the
clustering event must be conditioned on. Phylogenetic comparative methods say a
tree or precomputed distance becomes a statistical null only after its
covariance, branch-length, evolutionary, kernel, or permutation model is
specified. Topological graph analysis says subtree shape, merge persistence,
and tree balance can be part of the selected object.

## Key Points

- Hierarchical-clustering selective inference shows that classical
  mean-comparison tests are invalid after data-selected clustering unless the
  selected clustering event is accounted for.
- The pvclust and Shimodaira literature is useful for selected-region geometry:
  selected hypotheses are regions with signed-distance, curvature, and
  sample-size behavior. This does not imply that bootstrap should become a
  Tree-Break Selection production fallback.
- Phylogenetic comparative methods make branch-length covariance explicit.
  Brownian or OU-like assumptions produce a covariance model; a precomputed
  distance matrix alone is not a null law.
- Distance-based ecology and ML tests, including dbRDA/PERMANOVA, MMD, and
  distance covariance, support distance-based inference only after choosing a
  precise inferential object: a permutation design, kernel discrepancy,
  Euclidean/PCoA embedding, or covariance model.
- Tree topology matters. Tree balance, merge persistence, topology distances,
  and graph/TDA summaries suggest that medium/large parents can mix many
  selected-region shapes inside one parent-size bin.
- For Tree-Break Selection, medium/large high-edge contexts that pass simulation support but
  fail held-out precision likely lack tail homogeneity, not only replicate
  count.
- Candidate diagnostic variables are local subtree balance, child-size
  balance, subtree height, number of internal descendant nodes, merge
  persistence, nearest merge competitor margin, cumulative ancestor edge
  action, tree covariance condition number, eigenvalue concentration, and
  contrast/eigenvector angular alignment.

## Evidence

- `raw/inbox/phylogenetic-ml-topological-selected-tail-literature-20260603.md`
  records the checked papers and project-specific conclusions.
- [Gao, Bien, and Witten](https://arxiv.org/abs/2012.02936) show that
  inference after hierarchical clustering must account for the selected
  hypothesis.
- [Suzuki and Shimodaira](https://academic.oup.com/bioinformatics/article/22/12/1540/207339)
  connect hierarchical-cluster uncertainty to multiscale bootstrap p-values.
- [Shimodaira](https://arxiv.org/abs/math/0508602) frames approximately
  unbiased region tests through multiscale behavior.
- [Felsenstein](https://ichthyology.usm.edu/courses/multivariate/Felsenstein_1985.pdf)
  gives the Brownian phylogenetic contrast model that turns branch lengths
  into variance-standardized independent contrasts.
- [Jhwueng and O'Meara](https://pmc.ncbi.nlm.nih.gov/articles/PMC7019399/)
  analyze conditioning problems in phylogenetic covariance matrices.
- [Legendre and Anderson](https://www.numericalecology.com/Reprints/db-RDA.pdf)
  use PCoA to make distance-based linear-model inference explicit.
- [Gretton et al.](https://jmlr.csail.mit.edu/papers/v13/gretton12a.html)
  formalize kernel two-sample testing through MMD.
- [Szekely, Rizzo, and Bakirov](https://arxiv.org/abs/0803.4101) formalize
  distance covariance as a distance-based dependence object.
- [The mergegram literature](https://arxiv.org/abs/2007.11278) and
  [persistent homology of networks](https://appliednetsci.springeropen.com/articles/10.1007/s41109-019-0179-3)
  support treating multiscale graph topology as a stable object rather than
  collapsing it to a single parent-size scalar.

## Links

- [[selected-hierarchy-null-support-contract]]
- [[selected-hierarchy-geometric-law-map]]
- [[selected-tail-admissibility-domain-20260603]]
- [[open-mathematical-questions]]
