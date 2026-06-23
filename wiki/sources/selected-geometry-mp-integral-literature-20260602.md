---
title: Selected Geometry and MP Integral Literature 2026-06-02
type: source
status: reviewed
updated: 2026-06-02
sources:
  - raw/inbox/selected-geometry-mp-integral-literature-20260602.md
tags:
  - source
  - geometry
  - selection
  - spectral
  - literature
---

# Selected Geometry and MP Integral Literature 2026-06-02

## Summary

This literature capture redirects the selected-hierarchy calibration work away
from bootstrap as a production mechanism and toward an analytic geometric
object. The useful part of the Shimodaira/Terada line is not the resampling
estimator; it is the representation of selected hypotheses as regions with
signed distance, curvature, and conditional selective error. The useful part
of the Marchenko--Pastur literature is not only the closed-form white-noise
edge. Silverstein--Choi and Ledoit--Wolf give the Stieltjes-transform integral
route for a general population spectrum, which is the correct analytic
candidate if Tree-Break Selection's local null-whitened spectra are not identity-like.

## Key Points

- Gao, Bien, and Witten show that ordinary mean-comparison tests can have
  inflated type I error when the tested clusters are selected by hierarchical
  clustering. This supports Tree-Break Selection's separation between fixed-tree projected
  Wald validity and selected-hierarchy calibration.
- Terada and Shimodaira represent selected hypotheses as regions in a
  multivariate-normal parameter space and use signed distance and mean
  curvature to adjust selection bias. Their papers estimate these quantities
  with multiscale bootstrap, but Tree-Break Selection can retain the differential-geometric
  target without adopting bootstrap as a production estimator.
- For Tree-Break Selection, the local fixed-subspace statistic remains
  \[
  W_u(P)=\lVert P\Sigma_0(u)^{-1/2}\delta_u\rVert^2.
  \]
  The selected-hierarchy problem is that hierarchy construction, edge
  openings, and focal sibling selection define a selected region in this same
  tangent coordinate system.
- The current local MP rule is the identity-population case
  \(H=\delta_1\), giving
  \[
  \lambda_+=(1+\sqrt{d/m})^2.
  \]
- The analytic generalization is a deformed MP law. In the
  Silverstein--Choi convention, the limiting Stieltjes transform \(m(z)\)
  solves
  \[
  m(z)=-
  \left(
  z-c\int \frac{t}{1+t\,m(z)}\,dH(t)
  \right)^{-1},
  \]
  with density recovered from \(f(x)=\pi^{-1}\operatorname{Im}m(x+i0)\).
  Support edges can be studied through
  \[
  z(m)=-\frac{1}{m}+c\int\frac{t}{1+t\,m}\,dH(t).
  \]
- Ledoit--Wolf's QuEST work is the practical numerical route if Tree-Break Selection needs to
  discretize or invert this MP equation for local covariance spectra.
- Johnstone--Paul's high-dimensional PCA review supports keeping eigenvalue
  bias, eigenvector inconsistency, selected PCA, and soft-edge behavior as
  explicit validation issues rather than treating PCA directions as harmless
  preprocessing.

## Evidence

- `raw/inbox/selected-geometry-mp-integral-literature-20260602.md` records the
  verified source list and the no-bootstrap mathematical conclusion.
- [Gao, Bien, and Witten, 2024](https://arxiv.org/abs/2012.02936) state that
  tests after clustering need selective inference because the null hypothesis
  was chosen based on the data.
- [Terada and Shimodaira, 2018](https://arxiv.org/abs/1711.00949) define the
  selected-region geometry using signed distance and mean curvature.
- [Shimodaira and Terada, 2019](https://doi.org/10.3389/fevo.2019.00174)
  apply selective inference to trees and edges and emphasize conditional
  selected-hypothesis error.
- [Silverstein and Choi, 1995](https://jack.math.ncsu.edu/den.pdf) analyze the
  limiting spectral distribution through a Stieltjes-transform integral
  equation and the inverse support map.
- [Ledoit and Wolf, 2017](https://www.ledoit.net/Numerical_QuEST_2017.pdf)
  describe numerical discretization of the Marcenko--Pastur equation for
  high-dimensional covariance spectra.
- [Johnstone and Paul, 2018](https://doi.org/10.1109/JPROC.2018.2846730)
  review high-dimensional PCA phase-transition and eigenvalue/eigenvector
  effects relevant to selected PCA validation.

## Links

- [[selected-hierarchy-selection-geometry]]
- [[selected-hierarchy-geometric-law-map]]
- [[local-marchenko-pastur-rule]]
- [[open-mathematical-questions]]
