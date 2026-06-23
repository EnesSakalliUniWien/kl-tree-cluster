---
title: Root Conditional Kernel Spectral Law
type: analysis
status: draft
updated: 2026-06-17
sources:
  - wiki/sources/legacy-c2ef9a69-root-tail-overlap-comparison-20260617.md
  - wiki/sources/legacy-c2ef9a69-edge-alpha-comparison-20260617.md
  - wiki/sources/spectral-vs-bandwidth-tradeoff-panel-20260616.md
  - wiki/sources/root-selected-same-geometry-external-support-attempt-20260617.md
  - wiki/sources/root-selected-spectral-tail-law-with-legacy-overlay-20260617.md
  - wiki/sources/root-selected-kernel-spectral-tail-law-20260617.md
  - wiki/sources/root-selected-validity-replay-panel-20260617.md
  - wiki/sources/root-tree-geometry-hard-negative-replay-20260617.md
  - wiki/sources/selected-neighborhood-signal-flow-literature-20260617.md
  - tree_break_selection/legacy_methods/commit_c2ef9a69/tree_break_selection/hierarchy_analysis/statistics/sibling_divergence/pair_testing/sibling_null_prior_interpolation/adaptive_kernel_bandwidths.py
  - tree_break_selection/legacy_methods/commit_c2ef9a69/tree_break_selection/hierarchy_analysis/statistics/sibling_divergence/pair_testing/sibling_null_prior_interpolation/kernel_interpolation.py
  - tree_break_selection/legacy_methods/commit_c2ef9a69/tree_break_selection/hierarchy_analysis/statistics/sibling_divergence/pair_testing/sibling_null_prior_interpolation/child_prior_estimation.py
tags:
  - analysis
  - root
  - kernel
  - spectral
  - calibration
---

# Root Conditional Kernel Spectral Law

## Summary

The old kernel smoother should not be restored as an unconditional sibling
p-value rescue. The useful part is the local averaging measure: it encodes
which nearby tree states look stable, which nearby states look signal-like, and
how far the target is from its stopping-edge ancestry and structural scale.

The root law to learn is therefore a selected conditional spectral-tail law
with kernel-smoothed admissible support, not a lower p-value threshold.

The literature split is the same as the method split. Selective-inference and
multiscale-bootstrap work supports the fail-closed rule for selected roots:
the tested object was chosen by the algorithm. Diffusion maps, tree wavelets,
and treelets support the neighborhood as a multiscale signal-flow/locality
operator. They do not justify using the smoothed value as an unconditional
rescue p-value.

## Details

For a blocked child or sibling side \(u\), the old smoother used four adaptive
scales:

\[
\tau_b,\quad \tau_t,\quad \tau_s,\quad h_k .
\]

These measure stopping-edge distance scale, nearest stable-neighbor tree
distance, nearest signal-neighbor tree distance, and stable log-neighborhood
scale spread. In simplified form, the old stable-neighbor weight is

\[
w_0(u,v)
=
\exp\{-d_T(u,v)/\tau_t\}
\exp\left\{-\frac{(\log k_v-\log k_u)^2}{2h_k^2}\right\},
\]

with exact log-scale matching when \(h_k=0\). The ancestor weight is

\[
w_b(u)=\exp\{-\max(d_b(u)-1,0)/\tau_b\}.
\]

The smoothed null-like value is a convex average of ancestor and stable
neighbor p-values,

\[
\tilde p(u)
=
\frac{w_b(u)p_b(u)+\sum_{v\in\mathcal N_0}w_0(u,v)p_v}
{w_b(u)+\sum_{v\in\mathcal N_0}w_0(u,v)}.
\]

Nearby signal rows then suppress this value:

\[
s(u)
=
\max_{v\in\mathcal N_1}
(1-p_v)\exp\{-d_T(u,v)/\tau_s\},
\]

\[
p_{\mathrm{old}}(u)
=
\operatorname{clip}\{\tilde p(u)(1-s(u)),0,1\}.
\]

For a sibling pair, the old code writes the smaller child-side estimate back
into the sibling null prior. This explains why the old method can reopen rows
that the current strict method skips. It also explains the failure mode:
selected non-null neighborhoods can leak into the estimate, so the result is
not a calibrated selected-root null law.

The refined target is:

\[
\Pr\left(S_{H_u}\ge s\mid
R_{\mathrm{root}},G_u,T,A,E,B,H_u,\mathcal N_\tau(u)\right).
\]

Here \(R_{\mathrm{root}}\) is the selected root event, \(G_u\) is the
canonical unordered bifurcation signature of the root, \(T\) is selected
tie-rank geometry, \(A\) is selected-ratio action, \(E\) is edge action, \(B\)
is measured topology/bandwidth status, \(H_u\) is the local population
spectral law, and \(S_{H_u}\) is the deformed-MP spectral excess. The
neighborhood \(\mathcal N_\tau(u)\) should be kernel weighted but
support-gated:

\[
w_\tau(u,v)
=
\mathbf 1\{v\ \mathrm{is\ admissible\ null/external\ support}\}
\mathbf 1\{G_v\ \mathrm{matches\ or\ coarsens\ }G_u\}
K_\tau(u,v).
\]

The first executable topology-conditioned version uses

\[
G_u =
\left(
\mathrm{component}_u,
\mathrm{balance\ bin}_u,
\mathrm{root\ merge\ count}_u,
\mathrm{tie\ density\ bin}_u,
\mathrm{tie\ cell\ count\ bin}_u,
\mathrm{smooth\ count\ bin}_u
\right),
\]

with a coarsened version that groups balance, merge size, tie density, and
tie/smooth presence. Exact topology matches get full weight; coarsened matches
get a smaller weight before the scalar kernel is applied. The scalar kernel
then compares

\[
x(u)=
\left(
T_u,\log(1+A_u),\log(1+E_u),
\log K_u,\log n_u,\log H_u
\right).
\]

A conservative weighted tail can then be reported only when support is
sufficient:

\[
\hat p_+(s\mid u)
=
\frac{1+\sum_v w_\tau(u,v)\mathbf 1\{S_{H_v}\ge s\}}
{1+\sum_v w_\tau(u,v)}.
\]

The support check must include effective sample size,

\[
n_{\mathrm{eff}}
=
\frac{(\sum_v w_\tau(u,v))^2}{\sum_v w_\tau(u,v)^2},
\]

maximum weight share, and nonzero same-stratum spectral support. If those fail,
the method fails closed.

Plain English: kernel smoothing should tell us which calibrated neighborhoods
are relevant. It should not itself declare a split significant. The split
decision still needs a selected-root tail law over the eigenvalue excess after
conditioning on the selected topology and the local population spectrum.

The signal-flow analogy makes the scale interpretation explicit. In diffusion
maps, \(\tau\) controls local diffusion over a kernel graph and the slow modes
describe coherent geometry. In tree wavelets and treelets, coarse tree scales
retain low-frequency structure while fine-scale details carry localized
artifacts. For Tree-Break Selection, \(\tau_b,\tau_t,\tau_s,h_k\) should therefore be read as
support-locality coordinates for a selected conditional law, not as knobs that
make an unsupported root significant.

## Evidence

- The seven-case root-tail legacy comparison shows the old method improves one
  signal row but creates two selected-null false splits.
- The edge-alpha grid comparison closes the simpler explanation that the
  legacy gain is just an alpha-setting issue. Across edge alpha
  `0.0001, 0.0003, 0.001, 0.003, 0.01`, every alpha keeps one legacy signal
  gain but also has at least one legacy extra selected-null false split. The
  strictest alpha still leaks selected null on `overlap_mod_6c_med`, so edge
  alpha alone does not make the old rule admissible.
- The bandwidth tradeoff panel shows default interpolation is conservative but
  misses direct signal positives, while widening `tau_s` reopens selected-null
  positives faster than it recovers signal.
- The same-geometry external-support attempt shows the current generator/replay
  stack can add support rows, but the hard roots still lack same-stratum
  nonzero \(S_{H_u}\) support.
- The current root-tail panel already conditions on \(T,A,E,B,H_u\) and fails
  closed when support is absent. The missing improvement is the kernel-weighted
  admissible support law, not an unconditional rescue rule.
- The first scalar kernel-spectral candidate panel measures this law on the
  seven overlap roots. It adds scalar kernel support for `2/7` roots, including
  one strict fail-closed root, but `kernel_nonzero_support_target_count = 0`,
  so it is not promotable as a signal rescue.
- The topology-conditioned extension reports
  `topology_kernel_available_count = 0`,
  `topology_support_missing_count = 6`, and
  `topology_degenerate_support_count = 1`. This confirms that scalar smoothing
  was borrowing from the wrong root topology class.
- The root-validity replay panel separates the deeper selection problem from
  the tail law: a p-value calibrated inside the observed \(G_{\hat r}\) is not
  evidence that \(\hat r\) is the correct first bifurcation. A selected root is
  usable only when root validity replay and selected-root tail support both
  pass. The all-seven signal replay reports `2/7` root-validity-supported
  targets, but only `overlap_unbal_6c_med` is also tail-calibrated.
  `overlap_extreme_4c` is the opposite case: tail-calibrated inside the
  observed root event, but root-validity replay fails.
- The tree-geometry hard-negative replay strengthens that interpretation for
  `overlap_extreme_4c`: across linkage/Hamming, linkage/Jaccard,
  linkage/Rogers-Tanimoto, and neighbor-joining variants, `0/6` geometries
  become root-validity supported and `0/6` leak. The maximum root-stability
  mean ARI is only `0.012785`, far below the `0.24` guard threshold. The same
  replay now removes the privileged root direction and scans all undirected edge
  bipartitions in the selected tree. That rootless geometry check finds `0/6`
  current geometries and `0/4` legacy linkage geometries with a truth-aligned
  edge cut; the maximum edge-cut ARI is only `0.004682`. Thus alternative tree
  construction and root removal do not turn this case into a valid root-tail
  rescue; it remains a tree-geometry hard negative.
- The old commit replay on the same hard-negative benchmark supports the same
  conclusion from the opposite direction. The copied legacy method only
  supports linkage trees; on the four supported linkage geometries it either
  under-splits to one cluster or fragments on Rogers-Tanimoto average linkage
  with `5` found clusters while the root partition truth ARI remains
  `0.000663`. This is old-method fragmentation without a selected-root
  validity guard, not evidence that bandwidth interpolation should rescue the
  root.
- [[selected-neighborhood-signal-flow-literature-20260617]] records the
  literature bridge: selected clustering needs conditional inference, while
  diffusion/tree multiscale methods justify neighborhood smoothing only as
  locality and support evidence.

## Links

- [[legacy-c2ef9a69-root-tail-overlap-comparison-20260617]]
- [[legacy-c2ef9a69-edge-alpha-comparison-20260617]]
- [[spectral-vs-bandwidth-tradeoff-panel-20260616]]
- [[root-selected-same-geometry-external-support-attempt-20260617]]
- [[root-selected-spectral-tail-law-with-legacy-overlay-20260617]]
- [[root-selected-kernel-spectral-tail-law-20260617]]
- [[root-selected-validity-replay-panel-20260617]]
- [[root-tree-geometry-hard-negative-replay-20260617]]
- [[selected-neighborhood-signal-flow-literature-20260617]]
- [[selected-neighborhood-measurability-law]]

## Open Questions

- Which spectral distance should enter \(K_\tau\): deformed-MP edge ratio,
  empirical spectral distribution distance, eigengap profile, eigenspace
  projector distance, or a product of these?
- What is the minimum effective support and maximum weight share for a
  production-admissible weighted selected-root tail?
- What canonical root bifurcation signature is stable enough for production:
  the current balance/merge/tie proxy, a branch-isomorphism signature, or an
  eigenspace-labeled topology signature?
- Can action-dominance or edge-dominance be proven monotone after conditioning
  on \(H_u\), or must those remain diagnostic relaxations only?
