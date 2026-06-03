# Selected-geometry and Marchenko-Pastur integral literature notes

Date: 2026-06-02

Purpose: capture verified external references for the no-bootstrap analytic
direction for KL-TE selected-hierarchy calibration and local spectral
dimension rules. These notes are primary captured material for the wiki; they
are not production method changes.

## Verified sources

1. Gao, Bien, and Witten, "Selective Inference for Hierarchical Clustering"
   - arXiv: https://arxiv.org/abs/2012.02936
   - JASA DOI: https://doi.org/10.1080/01621459.2022.2116331
   - Relevance: classical tests have inflated type I error when clusters are
     selected by clustering; exact selective p-values condition on the
     data-chosen clustering hypothesis.

2. Terada and Shimodaira, "Selective inference for the problem of regions via
   multiscale bootstrap"
   - arXiv: https://arxiv.org/abs/1711.00949
   - Relevance: selected hypotheses can be represented as regions in a
     multivariate-normal parameter space. The geometric quantities are signed
     distance and mean curvature of the null/selection regions. The paper uses
     multiscale bootstrap to estimate them, but the quantities themselves are
     differential-geometric selection objects.

3. Shimodaira and Terada, "Selective Inference for Testing Trees and Edges in
   Phylogenetics"
   - Frontiers DOI: https://doi.org/10.3389/fevo.2019.00174
   - arXiv: https://arxiv.org/abs/1902.04964
   - Relevance: tree and edge support are selected hypotheses. Selective
     inference controls type I error conditional on the selection event and
     uses signed distance and mean curvature in the space of probability
     distributions. KL-TE should borrow the geometric target, not the bootstrap
     estimator, unless resampling is explicitly chosen as a diagnostic.

4. Silverstein and Choi, "Analysis of the Limiting Spectral Distribution of
   Large Dimensional Random Matrices"
   - PDF: https://jack.math.ncsu.edu/den.pdf
   - Relevance: the limiting spectrum of sample covariance matrices with
     general population spectral distribution H is defined through a Stieltjes
     transform integral equation. This generalizes the identity-population
     Marchenko-Pastur edge used by KL-TE.

5. Ledoit and Wolf, "Numerical Implementation of the QuEST Function"
   - PDF: https://www.ledoit.net/Numerical_QuEST_2017.pdf
   - arXiv: https://arxiv.org/abs/1601.05870
   - Relevance: practical numerical discretization and inversion of the
     Marcenko-Pastur equation for high-dimensional covariance spectra. Useful
     as a computational model if KL-TE moves from an identity MP edge to a
     deformed local spectral law.

6. Johnstone and Paul, "PCA in High Dimensions: An Orientation"
   - PMC landing page: https://pmc.ncbi.nlm.nih.gov/articles/PMC6167023/
   - DOI: https://doi.org/10.1109/JPROC.2018.2846730
   - Relevance: high-dimensional PCA has eigenvalue bias, phase transitions,
     eigenvector inconsistency, and soft-edge behavior. This supports keeping
     selected-PCA and MP threshold questions separate from the fixed-subspace
     projected-Wald reference.

## Mathematical conclusion for KL-TE

No-bootstrap production direction:

1. Keep the projected-Wald statistic as a squared norm in a null-whitened
   tangent chart:

   z_u = Sigma_0(u)^(-1/2) delta_u,
   W_u(P) = ||P z_u||^2.

2. Treat hierarchy construction, edge opening, and focal sibling selection as
   defining a selected region S_u in the same tangent coordinate system. A
   differential-geometric account should characterize S_u by local constraints,
   tangent cones, signed distances, curvature, and large-deviation action. It
   should not estimate these objects by bootstrap unless the project explicitly
   chooses a diagnostic resampling study.

3. Treat the current MP rule as the identity-population case H = delta_1:

   lambda_+ = (1 + sqrt(d/m))^2.

4. The analytic generalization is the deformed/general-population MP law. With
   aspect ratio c and population spectral distribution H, the limiting
   Stieltjes transform can be written in the Silverstein-Choi convention as

   m(z) = -1 / (z - c * int t / (1 + t m(z)) dH(t)).

   The density is recovered through

   f(x) = pi^(-1) Im m(x + i0),

   and the support edges are determined through the inverse map

   z(m) = -1/m + c * int t / (1 + t m) dH(t).

5. Therefore the next analytic MP question is not "run a bootstrap edge
   threshold". It is whether KL-TE's local null-whitened tangent spectra have
   H close enough to delta_1. If not, the method needs a local deformed-MP
   edge or a finite-sample soft-edge correction derived from the local spectral
   law.

6. The selected-hierarchy sibling tail and the MP spectral edge are connected
   through the same local tangent geometry: selection changes which contrast
   directions and which local spectral modes are observed. Edge-selection
   action, eigenvalue excess over the MP edge, effective rank, and angular
   mass are therefore geometric coordinates for a derivation, not tuning
   fallbacks.

## Boundaries

- These sources do not validate an external production selected-hierarchy
  calibration model for KL-TE.
- These sources do not justify restoring scalar c_hat fallback behavior.
- These sources do not justify replacing the production MP edge with a
  deformed or finite-sample edge without a locked validation run.
- The bootstrap parts of Shimodaira/Terada are not the chosen production path;
  only the region geometry is being retained as mathematical scaffold.
