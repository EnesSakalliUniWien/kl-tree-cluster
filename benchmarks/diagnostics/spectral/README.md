# Spectral Diagnostics

This directory contains diagnostics for projection-dimension and
Marchenko-Pastur threshold behavior.

- `mp/` owns MP-law checks, dimension sweeps, and backend profiling.
- `adaptive_cosine/` owns KAK, lens, and adaptive-cosine probes.
- `stability/` owns covariance-axis and tree-strategy stability panels.

Entrypoint:

```bash
uv run python -m benchmarks.diagnostics.spectral.mp.compare_mp_dimension_contracts
```

Backend profiling:

```bash
uv run python -m benchmarks.diagnostics.spectral.mp.profile_spectral_backends
```

Local identity-MP law screen:

```bash
uv run python -m benchmarks.diagnostics.spectral.mp.local_mp_identity_law_diagnostic
```

The local identity-MP screen is descriptive only. It writes case-level MP
departure summaries, node-level spectra, selected-tree spectral-law covariate
relationships, and a separate categorical extreme-node table. It must not be
used as a production threshold or fallback calibration path.
