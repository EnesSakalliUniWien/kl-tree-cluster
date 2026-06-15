---
title: Selected Root Selected Family Traversal Literature 2026-06-14
captured: 2026-06-14
source_urls:
  - https://arxiv.org/pdf/2012.02936
  - https://academic.oup.com/jrsssb/article-abstract/76/1/297/7075946
  - https://www.math.tau.ac.il/~yekutiel/papers/JASA%20FDR%20trees.pdf
  - https://people.math.ethz.ch/~nicolai/hierarchical.pdf
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC4764101/
  - https://www.treescan.org/
  - https://www.treescan.org/cgi-bin/treescan/register.pl/treescan.v2.1.userguide.pdf?todo=process_userguide_download
  - https://www.fieldtriptoolbox.org/tutorial/stats/cluster_permutation_freq/
  - https://www.sentinelinitiative.org/methods-data-tools/signal-identification-sentinel-system/faqs
  - https://academic.oup.com/bioinformatics/article/22/12/1540/207339
---

# Selected Root Selected Family Traversal Literature 2026-06-14

No checked source gives an exact off-the-shelf KL-TE selected-root and
selected-family pass-through law. The relevant literature decomposes the
problem into four adjacent tool families.

1. Selective inference after clustering:
   Gao, Bien, and Witten condition on the event that the tested clusters were
   selected by hierarchical clustering. This is the closest root-level
   inferential analogue. It addresses selected clusters from agglomerative
   hierarchical clustering, not KL-TE's feature-family permutations or
   pass-through traversal frontier.

2. Selective inference on families and hierarchical multiple testing:
   Benjamini and Bogomolov study testing inside families that were themselves
   selected from the data. Yekutieli's hierarchical FDR and Meinshausen's
   hierarchical testing give tree-structured FDR/FWER control when hypotheses
   are arranged in a hierarchy. These papers clarify the error-control object,
   but usually assume the tree of hypotheses or family structure is
   predeclared, not rebuilt under the same data as the test statistic.

3. Tree scan and max-statistic methods:
   TreeScan and spatial/cluster scan statistics test the maximum statistic over
   many overlapping candidate regions or tree branches, using Monte Carlo
   calibration to adjust for the search. This is the closest shape to KL-TE's
   selected-family pass-through null, because the observed statistic is a
   minimum p-value or maximum evidence over an entire searched family. The key
   difference is that TreeScan expects a pre-specified hierarchy, whereas KL-TE
   rebuilds the tree and pass-through frontier under the selected feature
   geometry.

4. Cluster-based permutation testing:
   Neuroimaging cluster-permutation methods form suprathreshold connected
   components, score each cluster, and use the maximum cluster statistic under
   permutations. This supports the logic of calibrating the whole selected
   family or connected frontier instead of testing the chosen descendant alone.
   It also carries a warning: a global cluster statistic controls the existence
   of a signal in the searched field, but it may not justify fine localization
   claims for every selected node.

Project mapping:

- The root law is a conditional selected-cluster problem:
  condition on root/topology construction enough to make the root split
  p-value honest.
- The pass-through law is a selected-family scan problem:
  condition on, or resample through, the whole root-closed descendant search
  and compare the observed pass-through evidence to the null minimum/maximum
  over the same selected family.
- Hierarchical FDR/gatekeeping is useful if KL-TE predeclares a tree of
  hypotheses, but it is not by itself sufficient when the tree is rebuilt from
  the same data.
- The current `global_sibling_min_passthrough_descendant_refined` diagnostic is
  best understood as a TreeScan/maxT-style selected-family permutation object
  adapted to a data-selected hierarchy.
