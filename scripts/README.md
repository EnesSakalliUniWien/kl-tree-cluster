# Scripts

This directory contains repository maintenance and external-request helper
commands. It is not a home for reusable method code, dataset applications, or
manuscript figure composition.

- Wiki maintenance lives in `scripts/wiki/`.
- Dataset-specific analysis and figure generators live beside their
  application, including `applications/scrna/analysis/` and
  `applications/scrna/plots/`.
- Methodological separation lives in
  `tree_break_selection/space_separation/`.
- Reusable plotting engines live in `tree_break_selection/plot/`.
- Discoverable endotype, scRNA, and MNIST commands live in `applications/`.
- Explanatory paper figure generators live in `manuscript/tools/figures/`.

One-off diagnostics that become reusable should move to the package; commands
that become a maintained dataset workflow should move to the corresponding
application directory.
