# Single-Cell RNA Applications

The adult pancreas and Goncalves fetal-pancreas adapters live here. Dataset
loading, labels, manifests, and application reports stay local to this folder;
reusable space separation and plotting primitives come from
`tree_break_selection/`.

- `analysis/` contains dataset-specific biological comparisons and audit
  commands.
- `plots/` contains dataset-specific Python and R figure/report composition.
- The four top-level Python files below are the primary application entry
  points and orchestration commands.

```bash
python applications/scrna/pancreas_benchmark.py --help
python applications/scrna/goncalves_benchmark.py --help
python applications/scrna/analyze_space_decomposition.py --help
python applications/scrna/plot_pipeline.py --help
```

Specialized scRNA figure generators used by `plot_pipeline.py` live in
`applications/scrna/plots/`; dataset-specific follow-up analyses live in
`applications/scrna/analysis/`. They are application code, not reusable
method or plotting-engine code.
