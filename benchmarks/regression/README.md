# Benchmark Regression Gates

Regression gates run historically sensitive benchmark subsets with explicit
pass/fail expectations. They are development checks, not comprehensive
validation.

Entrypoint:

```bash
uv run python -m benchmarks.regression.run_gate
```
