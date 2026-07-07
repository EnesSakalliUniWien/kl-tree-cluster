# Adaptive Alpha Method Literature Capture 2026-06-28

This capture records primary-source guidance for designing an adaptive alpha
policy for Tree-Break Selection gates. The design problem is not simply choosing
a larger alpha. TBS builds and tests a data-selected hierarchy, so any adaptive
threshold must avoid using the same node p-value both to choose the threshold
and to pass the test.

## Sources Reviewed

- Benjamini and Hochberg, "Controlling the False Discovery Rate: A Practical and
  Powerful Approach to Multiple Testing", JRSS-B 1995.
  DOI: https://doi.org/10.1111/j.2517-6161.1995.tb02031.x
- Yekutieli, "Hierarchical False Discovery Rate-Controlling Methodology", JASA
  2008. DOI: https://doi.org/10.1198/016214507000001373
- Ignatiadis, Klaus, Zaugg, and Huber, "Data-driven hypothesis weighting
  increases detection power in genome-scale multiple testing", Nature Methods
  2016. DOI: https://doi.org/10.1038/nmeth.3885
- Lei and Fithian, "AdaPT: An interactive procedure for multiple testing with
  side information", JRSS-B 2018; arXiv 1609.06035.
  https://arxiv.org/abs/1609.06035
- Foster and Stine, "alpha-Investing: a Procedure for Sequential Control of
  Expected False Discoveries", JRSS-B 2008.
  DOI: https://doi.org/10.1111/j.1467-9868.2007.00643.x
- Javanmard and Montanari, "Online rules for control of false discovery rate and
  false discovery exceedance", Annals of Statistics 2018.
  DOI: https://doi.org/10.1214/17-AOS1559
- Meinshausen and Buhlmann, "Stability Selection", JRSS-B 2010.
  DOI: https://doi.org/10.1111/j.1467-9868.2010.00740.x
- Suzuki and Shimodaira, "Pvclust: an R package for assessing the uncertainty in
  hierarchical clustering", Bioinformatics 2006.
  DOI: https://doi.org/10.1093/bioinformatics/btl117
- Gao, Bien, and Witten, "Selective Inference for Hierarchical Clustering",
  arXiv 2012.02936. https://arxiv.org/abs/2012.02936

## Methodological Notes

- BH establishes the baseline logic for controlling expected false discovery
  proportion across multiple tests, but it assumes a fixed family of hypotheses
  and dependence conditions that cannot be taken for granted in a data-selected
  hierarchy.
- Hierarchical FDR procedures are relevant because TBS tests tree-structured
  families, but published guarantees assume a properly defined testing tree and
  do not automatically validate a hierarchy selected from the same data.
- IHW and AdaPT show how to gain power with side information. The side
  information must be handled so that it does not leak the same p-value evidence
  used for the rejection decision.
- Online FDR and alpha-investing procedures show one principled way to adapt
  thresholds along an ordered sequence: later thresholds may depend on past
  decisions, not on the hidden p-value of the current test.
- Stability selection and pvclust support the practical idea that structural
  claims can be checked by resampling stability. For the TBS adaptive-alpha
  design, however, resampling should remain a validation diagnostic, not the
  runtime adaptive trigger.
- Selective inference for hierarchical clustering is a warning signal for TBS:
  classical p-values can be badly miscalibrated when the tested groups were
  chosen by clustering. Any production claim of adaptive FDR control therefore
  needs a TBS-specific selected-hierarchy calibration argument.

## Design Consequences For TBS

- Do not adapt a node's alpha upward because that node's own p-value is close to
  alpha. That is circular and can create unreported Type-I error inflation.
- Treat edge alpha and sibling alpha as different contracts. Edge alpha controls
  upstream traversal admissibility; sibling alpha controls local resolution.
- Use a predeclared sibling-alpha ladder to report the alpha level required for
  each accepted split. The ladder can drive adaptive resolution labels without
  pretending to be formal adaptive FDR control.
- Do not use bootstrap, subsampling, diffusion-parameter perturbation, or other
  perturbation loops as the runtime adaptive trigger. The adaptive decision
  should be computed from the ordinary edge/sibling test trace, calibration
  support fields, and predeclared context bins in the same run.
- Keep edge alpha conservative by default until selected edge calibration is
  explicitly extended. Current benchmark evidence shows the key structure
  changes are mostly sibling-gate driven.
- Name the first implementable version "adaptive sibling-alpha ladder" or
  "adaptive resolution", not "adaptive FDR", unless calibration is added.
