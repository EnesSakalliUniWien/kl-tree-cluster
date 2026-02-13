# Cross-Fit + Permutation Calibration Implementation Plan

## Goal
Integrate decomposition-respecting, selection-aware diagnostics into the calibration workflow and compare before/after behavior with reproducible benchmark artifacts.

## Scope
- Diagnostic-only integration into benchmark calibration.
- No production inference behavior changes in core clustering pipeline.
- Preserve current `run.py` default behavior unless explicit flags are enabled.

## Stage 0: Baseline Snapshot
1. Run current calibration suite (`benchmarks/calibration/run.py`) with standard settings.
2. Save baseline artifact paths and key metrics:
   - `null_type1_summary.csv`
   - `treebh_summary.csv`
   - `binary_covariance_type1_summary.csv`
3. Record baseline pass/fail status (Type-I and TreeBH) in a short summary note.

## Stage 1: Integrate New Diagnostics into Calibration Runner
1. Add optional execution hooks in `benchmarks/calibration/run.py` for:
   - `crossfit_permutation_diagnostic.py` (null-focused).
   - `crossfit_permutation_benchmark_probe.py` (real benchmark cases).
2. Add configurable controls (defaults OFF):
   - `run_crossfit_perm_diag` (bool)
   - `crossfit_diag_reps` (int)
   - `crossfit_diag_perms` (int)
   - `run_crossfit_benchmark_probe` (bool)
   - `crossfit_probe_perms` (int)
3. Include resulting artifact paths in runner outputs dictionary.
4. Extend markdown report with an explicit section:
   - Cross-fit + permutation diagnostic summary table.
   - Before/after comparison against in-sample baseline.

## Stage 2: Validation and Benchmark Re-Run
1. Re-run calibration with new diagnostic options enabled.
2. Verify generated artifacts exist and are loadable.
3. Compare key metrics against Stage 0 baseline:
   - In-sample null Type-I (expected inflated).
   - Cross-fit+permutation null Type-I (expected controlled/conservative).
4. Confirm no regressions in existing calibration outputs.

## Stage 3: Decision Gate
1. Mark status:
   - PASS if cross-fit permutation diagnostics reduce null inflation materially.
   - CHECK if mixed results with actionable follow-up.
   - FAIL if no improvement.
2. Propose next integration step:
   - Optional promotion of cross-fit path to a selectable inference mode (future phase).

## Acceptance Criteria
- Baseline and post-change runs both complete successfully.
- New diagnostic artifacts are emitted deterministically.
- Report includes explicit before/after metrics.
- No changes to production clustering decisions unless explicitly requested in future phase.

## Notes
- Because the repository has unrelated in-flight changes, commits should include only files directly touched for this stage.
