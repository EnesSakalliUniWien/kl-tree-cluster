# Phylogenetic, ML, and Topological Literature for Selected-Tail Contexts

Date: 2026-06-03

Purpose: literature capture for KL-TE selected-tail contexts that remain
undefined, especially root, medium/large parent, lower-edge-action,
continuous, precomputed-distance, and phylogenetic-like contexts.

## Sources Checked

- Gao and Witten / Gao, Bien, and Witten, "Selective Inference for
  Hierarchical Clustering":
  https://arxiv.org/abs/2012.02936
- Suzuki and Shimodaira, "Pvclust: an R package for assessing the uncertainty
  in hierarchical cluster analysis":
  https://academic.oup.com/bioinformatics/article/22/12/1540/207339
- Shimodaira, "Approximately unbiased tests of regions using
  multistep-multiscale bootstrap resampling":
  https://arxiv.org/abs/math/0508602
- Felsenstein, "Phylogenies and the Comparative Method":
  https://ichthyology.usm.edu/courses/multivariate/Felsenstein_1985.pdf
- Jhwueng and O'Meara, "On the Matrix Condition of Phylogenetic Tree":
  https://pmc.ncbi.nlm.nih.gov/articles/PMC7019399/
- Lozupone and Knight, "UniFrac: a new phylogenetic method for comparing
  microbial communities":
  https://pubmed.ncbi.nlm.nih.gov/16332807/
- Legendre and Anderson, "Distance-based redundancy analysis":
  https://www.numericalecology.com/Reprints/db-RDA.pdf
- Gretton et al., "A Kernel Two-Sample Test":
  https://jmlr.csail.mit.edu/papers/v13/gretton12a.html
- Szekely, Rizzo, and Bakirov, "Measuring and testing dependence by correlation
  of distances":
  https://arxiv.org/abs/0803.4101
- Aldous/Blum-Francois beta-splitting tree-balance literature summary:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4892442/
- "The Mergegram of a Dendrogram and Its Stability":
  https://arxiv.org/abs/2007.11278
- Persistent homology of networks review:
  https://appliednetsci.springeropen.com/articles/10.1007/s41109-019-0179-3
- Robinson-Foulds / generalized tree-topology distance literature:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC8742253/

## Literature Takeaways for KL-TE

1. Selective inference for hierarchical clustering shows that a post-clustering
   mean-comparison test is not valid if it ignores the clustering event. For
   KL-TE, the selected sibling statistic needs a selected-region law, not an
   ordinary fixed-tree Wald reference, whenever the hierarchy is selected from
   the same data.

2. The pvclust / Shimodaira line is useful as geometry, not as a production
   bootstrap prescription. It frames selected hypotheses as regions and
   studies how probabilities change when sample size or boundary geometry
   changes. KL-TE should keep the selected-region, signed-distance, curvature,
   and support ideas, but should not install bootstrap as a silent fallback.

3. Phylogenetic comparative methods say that a tree or distance matrix becomes
   a statistical model only after an evolutionary covariance model is specified.
   Brownian, OU, and related models induce covariance through shared branch
   length. Patristic, UniFrac, Hamming, Euclidean, and other precomputed
   distances are not interchangeable null laws.

4. Phylogenetic covariance conditioning matters. Tree shape and short terminal
   branches can make the phylogenetic covariance matrix ill-conditioned. In
   KL-TE terms, branch geometry and covariance conditioning are potential
   selected-tail variables for phylogenetic-like or precomputed-distance
   contexts.

5. Distance-based ecological and ML tests support precomputed-distance
   analysis only when the distance object has a clear inferential target. dbRDA
   and PERMANOVA use PCoA/permutation logic; MMD and distance covariance use
   kernel or distance-based population discrepancy logic. They do not license a
   Wald calibration unless the distance is embedded into a valid covariance or
   kernel null.

6. Topological graph analysis suggests that root and medium/large parent
   contexts should not be binned only by parent size. Dendrograms, merge trees,
   tree-balance indices, merge persistence, and tree-topology distances all
   treat topology as part of the object. Medium/large parents contain many
   possible selected subtree shapes, so parent-size bins can mix distinct
   selected regions and destroy held-out tail precision.

7. Local candidate variables from this literature are: local subtree balance,
   number of internal descendant nodes, subtree height, merge persistence or
   branch-length gap, nearest merge competitor margin, cumulative ancestor edge
   action, child balance, tree covariance condition number, effective
   resistance or Laplacian-like graph summaries, eigenvalue concentration, and
   angular alignment of the sibling contrast with selected spectral modes.

## Consequence for Current Undefined Contexts

The literature supports treating medium/large parent failures as tail
heterogeneity rather than simple Monte Carlo shortage when simulation support
is present but held-out precision fails. A correct next diagnostic should
stratify or model the selected-tail law by topology and geometry variables,
then test whether held-out tail precision improves. None of these variables is
yet a production calibration rule.
