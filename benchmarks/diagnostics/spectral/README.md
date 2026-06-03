# Spectral Diagnostics

This directory contains diagnostics for projection-dimension and
Marchenko-Pastur threshold behavior.

Entrypoint:

```bash
uv run python -m benchmarks.diagnostics.spectral.compare_mp_dimension_contracts
```

Backend profiling:

```bash
uv run python -m benchmarks.diagnostics.spectral.profile_spectral_backends
```

Local identity-MP law screen:

```bash
uv run python -m benchmarks.diagnostics.spectral.local_mp_identity_law_diagnostic
```

The local identity-MP screen is descriptive only. It writes case-level MP
departure summaries, node-level spectra, selected-tree spectral-law covariate
relationships, and a separate categorical extreme-node table. It must not be
used as a production threshold or fallback calibration path.
