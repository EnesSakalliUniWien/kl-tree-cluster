---
title: Alpha Structure Sweep 2026-06-27
type: source
status: reviewed
updated: 2026-06-27
sources:
  - benchmarks/validation/sweeps/alpha_structure_sweep.py
  - benchmarks/results/alpha_structure_sweep_focus_20260627/alpha_structure_manifest.json
  - benchmarks/results/alpha_structure_sweep_focus_20260627/alpha_structure_summary.csv
  - benchmarks/results/alpha_structure_sweep_focus_20260627/alpha_partition_transitions.csv
  - benchmarks/results/alpha_structure_sweep_cartesian_20260627/alpha_structure_manifest.json
  - benchmarks/results/alpha_structure_sweep_cartesian_20260627/alpha_structure_summary.csv
  - benchmarks/results/alpha_structure_sweep_cartesian_20260627/alpha_partition_transitions.csv
  - benchmarks/results/alpha_structure_sweep_hamming_nn_q4_q5_20260627/alpha_structure_summary.csv
  - benchmarks/results/alpha_structure_sweep_hamming_nn_q4_q5_20260627/alpha_partition_transitions.csv
  - benchmarks/results/alpha_structure_sweep_smoke_20260627/edge_0p001__sibling_0p01/partition_assignments.csv
tags:
  - benchmark
  - alpha
  - p-values
  - clustering
---

# Alpha Structure Sweep 2026-06-27

## Summary

`benchmarks/validation/sweeps/alpha_structure_sweep.py` adds a methodological alpha
sweep for TBS-family benchmark runs. It treats edge alpha and sibling alpha as
separate axes, reruns selected benchmark cases at each alpha pair, and writes
benchmark metrics, node p-value margins, traversal p-value margins, partition
assignments, structural summaries, and partition transitions against a baseline
alpha pair. The sweep is diagnostic evidence only; it does not change
production defaults or claim selected-tree Type-I error calibration.

## Key Points

- The runner records p-value boundary distances as
  `log10(alpha / p)`. Positive values mean the p-value is inside the current
  alpha boundary; negative values mean the gate is still closed.
- The focused diagonal sweep used four cases:
  `gauss_overlap_3c_small_q4`, `gauss_overlap_3c_small_q5`,
  `phylo_large_32taxa`, and `phylo_large_64taxa`.
- The focused sweep used `tbs` and `tbs_diffusion_adaptive` over alpha pairs
  `0.0003:0.003`, `0.001:0.01`, `0.003:0.03`, and `0.01:0.1`.
- In the focused run, adaptive diffusion recovers
  `gauss_overlap_3c_small_q4` and `gauss_overlap_3c_small_q5` exactly across
  all tested alpha pairs: `3` clusters and ARI `1.0`.
- In `phylo_large_32taxa`, adaptive diffusion changes from `41` clusters at
  `0.0003:0.003` to `43` at the baseline `0.001:0.01`, `47` at
  `0.003:0.03`, and `48` at `0.01:0.1`; ARI decreases from `0.983674` to
  `0.959953` as the sweep becomes more permissive.
- In `phylo_large_64taxa`, adaptive diffusion changes from `46` clusters at
  `0.0003:0.003` to `61` at baseline, `70` at `0.003:0.03`, and `104` at
  `0.01:0.1`; ARI rises from `0.542375` to `0.669707`, showing that more
  permissive sibling splitting reveals additional taxon structure in this
  case.
- The Cartesian sweep over `gauss_overlap_3c_small_q4` and
  `phylo_large_64taxa` separates edge alpha from sibling alpha. For
  `phylo_large_64taxa`, found clusters depend on sibling alpha (`46`, `61`,
  `70` for sibling alpha `0.003`, `0.01`, `0.03`) and are invariant across
  edge alpha `0.0003`, `0.001`, and `0.003`.
- In the same Cartesian sweep, `n_trace_edge_open` is constant at `395` for
  `phylo_large_64taxa`, while `n_trace_sibling_open` changes from `19` to
  `26` to `31`; this shows edge gates are already saturated in that range and
  the structural refinement is controlled by sibling p-values.
- The smoke run confirms that the current runner writes
  `partition_assignments.csv` along with p-value margin and transition files.
- A targeted Hamming NN diffusion sweep on q4/q5 reproduces the earlier
  baseline under-split: at `edge=0.001`, `sibling=0.01`, both q4 and q5 return
  `2` clusters with ARI `0.569784`.
- In the same Hamming NN diffusion q4/q5 sweep, relaxing sibling alpha to
  `0.03` opens one additional sibling gate and recovers `3` clusters:
  q4 reaches ARI `1.0`, and q5 reaches ARI `0.989983`.

## Evidence

- `uv run --all-extras ruff check benchmarks/validation/sweeps/alpha_structure_sweep.py`
  passed.
- `uv run --all-extras python -m py_compile
  benchmarks/validation/sweeps/alpha_structure_sweep.py` passed.
- `benchmarks/results/alpha_structure_sweep_focus_20260627/alpha_structure_summary.csv`
  has `24` structural rows from four cases, two methods, and four alpha pairs
  with successful computed-result records.
- `benchmarks/results/alpha_structure_sweep_focus_20260627/alpha_partition_transitions.csv`
  records each partition's ARI/NMI against the baseline
  `edge_0p001__sibling_0p01`.
- `benchmarks/results/alpha_structure_sweep_cartesian_20260627/alpha_structure_summary.csv`
  has `18` structural rows from two cases, one method, and a `3 x 3`
  edge/sibling alpha grid.
- `benchmarks/results/alpha_structure_sweep_cartesian_20260627/alpha_partition_transitions.csv`
  shows that the `phylo_large_64taxa` adaptive partition changes with sibling
  alpha but not edge alpha over the compact Cartesian grid.
- `benchmarks/results/alpha_structure_sweep_hamming_nn_q4_q5_20260627/alpha_structure_summary.csv`
  records the Hamming NN diffusion q4/q5 transition from `2` clusters at
  baseline to `3` clusters when sibling alpha is relaxed to `0.03`.
- `benchmarks/results/alpha_structure_sweep_smoke_20260627/edge_0p001__sibling_0p01/partition_assignments.csv`
  verifies the partition-assignment output contract added after the focused
  runs.

## Links

- [[categorical-adaptive-diffusion-focus-audit-20260626]]
- [[full-adaptive-pydiffmap-benchmark-run-20260627]]
- [[phylo-large-adaptive-pydiffmap-focus-audit-20260627]]
