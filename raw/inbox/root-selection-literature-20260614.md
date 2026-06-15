---
title: Root Selection Literature Notes 2026-06-14
captured: 2026-06-14
source_urls:
  - https://arxiv.org/abs/2012.02936
  - https://academic.oup.com/bioinformatics/article/22/12/1540/207339
  - https://rdrr.io/github/pkimes/sigclust2/man/shc.html
  - https://www.rdocumentation.org/packages/sigclust/versions/1.1.0/topics/sigclust
  - https://statistics.stanford.edu/technical-reports/estimating-number-clusters-dataset-gap-statistic
  - https://gregoryschwartz.github.io/too-many-cells/
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC7439807/
  - https://scikit-learn.org/stable/modules/clustering.html
  - https://link.springer.com/content/pdf/10.1023/A%3A1020443310743.pdf
  - https://www.sc-best-practices.org/trajectories/pseudotemporal.html
  - https://www.ebi.ac.uk/training/online/courses/introduction-to-phylogenetics/what-is-a-phylogeny/aspects-of-phylogenies/root/
  - https://link.springer.com/article/10.1186/s12859-021-03956-5
---

# Root Selection Literature Notes 2026-06-14

The literature distinguishes three different meanings of "root":

1. In ordinary hierarchical clustering, the root is the all-sample cluster.
   The inferential problem is whether the first selected split of that root is
   real under the appropriate selected null.
2. In phylogenetics, root means an oriented ancestral location on a tree. That
   requires external direction information such as outgroups, clocks, or a
   non-reversible model.
3. In single-cell trajectory inference, root means the starting cell or state
   for pseudotime. It is an orientation choice, usually imposed from biology or
   marker knowledge, not discovered by generic clustering alone.

For KL-TE, the current root problem is the first meaning. The root is not a
biological ancestor and should not be optimized as a free orientation parameter
unless the method adds external temporal or lineage information. The root split
is instead a selected top-level partition of the whole sample, chosen by the
same geometry later tested by the sibling gate.

Relevant literature points:

- Gao, Bien, and Witten's selective inference paper directly matches the
  warning seen in KL-TE: classical mean-difference/Wald tests are
  anti-conservative after clusters are selected from the data. Their object is
  a conditional p-value given the clustering event.
- Pvclust and Shimodaira-style multiscale bootstrap provide uncertainty
  assessment for clusters in a dendrogram. They are useful conceptual tools for
  selected-region geometry and curvature, but they do not by themselves define
  a KL-TE production calibration rule.
- SHC/SigClust tests splits from the root downward under a single-Gaussian
  null, with simulation and FWER control. This is the closest existing
  "optimize/test the root split" family, but its Gaussian cluster null is not
  the KL-TE Bernoulli/categorical selected-topology null.
- The gap statistic compares observed within-cluster dispersion against a
  reference null for estimating the number of clusters. It is a global
  cluster-count/stopping tool, not a selected-root sibling law.
- TooManyCells starts with all cells in one node and recursively applies
  spectral clustering, stopping when Newman-Girvan modularity is non-positive.
  This is relationally useful because it avoids a Wald root p-value by using a
  graph/community stopping objective, but it is not a direct KL-TE comparator
  or a calibration proof.
- PDDP and bisecting k-means show practical root-split tools in high-dimensional
  spaces: split by leading principal directions or repeated two-means. These
  optimize tree construction, not inferential validity of the selected split.
- High-dimensional clustering literature warns that distances inflate and
  root-split directions can be unstable. This supports KL-TE's root-stability,
  selected-family, and topology-replay diagnostics.
- Phylogenetic rooting tools optimize root placement only after a tree and an
  evolutionary direction model are specified. Without such a model, a clustering
  root has no ancestral meaning.

Project conclusion:

KL-TE should not "optimize the root" as a standalone hyperparameter. It should
optimize or validate one of two explicit objects:

1. Tree construction quality: whether the selected top-level partition is a
   useful first split under the domain feature space.
2. Root split admissibility: whether opening the selected root is valid under a
   selected-topology null law.

For the current method, the second object is the production blocker. The right
tool family is selected-root inference, root-stability replay, selected-family
permutation, root-balance/topology diagnostics, and optionally divisive or
graph-modularity root objectives as candidate tree constructors.
