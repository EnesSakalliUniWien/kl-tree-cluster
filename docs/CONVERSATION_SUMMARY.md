# Conversation Summary and Fix Log

Date: 2026-02-13  
Project: `kl-te-cluster`

## Scope We Discussed

- Publication readiness framing (methodology + benchmark rigor emphasis).
- Benchmark structure and reproducibility flow (single output path, concatenated report).
- PDF and figure readability (UMAP + tree pages were overlapping/squashed).
- Duplicate code/module cleanup and migration toward shared benchmark modules.
- Statistical-method robustness in KL tests (seeds, z-score handling, branch-length guards).
- Runtime stability (intermittent subprocess `-11` crashes on heavy overlap cases).

## Confirmed Fixes (Implemented)

| ID | Topic | Request | Status | What was changed | Evidence |
| --- | --- | --- | --- | --- | --- |
| F-001 | UMAP readability | Make UMAP pages readable; avoid squeezed multi-method grids | Fixed | Added paginated UMAP comparison generation (`max_panels_per_page`) and page-aware export path. | `benchmarks/shared/plots/embedding.py:149`, `benchmarks/shared/plots/export.py:82` |
| F-002 | Tree visualization layout | Each tree visualization on its own row/page, not squeezed | Fixed | Tree plotting exports one figure per method/page and labels pages `(i/n)`. | `benchmarks/shared/plots/export.py:149` |
| F-003 | No interactive plotting side effects | Benchmark outputs should go to report, not interactive windows | Fixed in benchmark path | Benchmark plot flow writes to PDF pages; no `plt.show()` in shared benchmark plot pipeline. | `benchmarks/shared/plots/export.py:42`, `benchmarks/shared/pipeline.py` |
| F-004 | Metrics gate behavior | Remove hard dependency on `num_clusters > 0`; compute when labels align | Fixed | Metrics function now validates alignment and computes from labels/report, returning NaN only when invalid/uncomputable. | `benchmarks/shared/metrics.py:13` |
| F-005 | Per-test projection seeding (M-001) | Avoid shared random matrix across all tests | Fixed | Edge and sibling tests derive deterministic seed per hypothesis via `derive_projection_seed(...)`. | `kl_clustering_analysis/hierarchy_analysis/statistics/kl_tests/edge_significance.py:279`, `kl_clustering_analysis/hierarchy_analysis/statistics/sibling_divergence/sibling_divergence_test.py:179` |
| F-006 | Z-score clamp concern | Remove hard clipping to ±100 | Fixed | Hard clamp removed; finite values preserved; only non-finite values are repaired. | `kl_clustering_analysis/hierarchy_analysis/statistics/kl_tests/edge_significance.py:188`, `kl_clustering_analysis/hierarchy_analysis/statistics/sibling_divergence/sibling_divergence_test.py:155` |
| F-007 | Benchmark runner crash resilience | Handle intermittent case subprocess `-11` failures | Fixed (mitigation) | Added per-case retries on `-11` and fallback to in-process execution after retry budget. | `benchmarks/full/run.py:293` |
| F-008 | Native thread pressure | Reduce instability in spawned workers | Fixed (mitigation) | Set worker-level thread caps (`OMP/OPENBLAS/MKL/VECLIB/NUMEXPR=1`). | `benchmarks/full/run.py:99` |
| F-009 | UMAP default behavior | Always apply UMAP in full benchmark when plots enabled | Fixed | Full runner defaults to UMAP and enforces UMAP backend env if unset. | `benchmarks/full/run.py:196` |
| F-010 | Numerical hard-fail on invalid edge p-values | Do not abort whole case on rare non-finite p-values | Fixed (conservative fallback) | Non-finite edge p-values now converted to conservative defaults (`p=1.0`, stat reset) with warning. | `kl_clustering_analysis/hierarchy_analysis/statistics/kl_tests/edge_significance.py:344` |
| F-011 | Numerical hard-fail on zero sibling branch lengths | Avoid exception for non-positive branch-length sum | Fixed | Now logs warning and disables branch-length adjustment for that sibling test instead of raising. | `kl_clustering_analysis/hierarchy_analysis/statistics/sibling_divergence/sibling_divergence_test.py:120` |
| F-012 | Projection generation stability | Investigate RNG/QR-heavy crash path | Fixed (mitigation) | Projection path switched away from `default_rng` hot loop and adds structured orthonormal projection when `k≈d` to avoid heavy QR. | `kl_clustering_analysis/hierarchy_analysis/statistics/random_projection.py:33`, `kl_clustering_analysis/hierarchy_analysis/statistics/random_projection.py:134` |

## Confirmed Runtime Findings

- Full benchmark run completed at:
  - `benchmarks/results/run_20260212_205730Z/full_benchmark_report.pdf`
  - `benchmarks/results/run_20260212_205730Z/full_benchmark_comparison.csv`
- That run had one missing case:
  - `overlap_part_10c_highd` (subprocess exited `-11`)
  - Case definition: `benchmarks/shared/cases/overlapping.py:138`
- During crash analysis, `-11` was reproduced in isolated mode and traced to native-level failures in spawned workers (macOS diagnostic reports), then mitigated with retry/fallback logic.

## Partially Fixed / Still Open

| ID | Item | Status | Notes |
| --- | --- | --- | --- |
| O-001 | Intermittent native crash in isolated subprocesses | Partially fixed | Retry + fallback prevents case loss, but root native crash origin (Python/NumPy/pandas stack-level instability) is not fully eliminated. |
| O-002 | Statistical calibration quality (type-I control/power) | Open | Still requires formal null/power simulation evidence for publication-level inferential claims. |
| O-003 | Over/under-splitting behavior on difficult families | Open | Failure diagnosis still flags multiple over-splitting and under-splitting cases in prior run report. See `benchmarks/results/run_20260212_205730Z/failure_report.md:6`. |

## Requests Discussed Earlier (Unverified in this final pass)

These were discussed in the thread but not fully re-audited in this final check:

- Full duplicate-module cleanup across all benchmark subpackages.
- Complete old-run folder pruning.
- Full PDF formatting uniformity across all previously generated reports.

Use `git log`/`git diff` plus a fresh full run to re-confirm each of those end-to-end.

## Current Recommended Validation Step

Run a fresh full benchmark and verify all 95/95 cases appear in the final CSV:

```bash
python benchmarks/full/run.py
```

Then verify:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("benchmarks/results/<latest_run>/full_benchmark_comparison.csv")
print("rows:", len(df))
print("cases:", df["test_case"].nunique())
print("methods/case min-max:", df.groupby("test_case")["method"].nunique().min(), df.groupby("test_case")["method"].nunique().max())
PY
```

