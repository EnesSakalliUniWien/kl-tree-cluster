# Debug Scripts Structure

This folder is being migrated from ad-hoc one-off scripts to a searchable and enforceable structure.

## Naming Contract

All active diagnostic scripts should use:

`q_<question>__<method>__<scope>.py`

Examples:

- `q_k1_overdeflation_diagnostic__cousin_adjusted_wald__compact.py`
- `q_case18_power_data_extraction__power__case18.py`

## Header Contract

Each active script must include a module-level header with:

- `Purpose:`
- `Inputs:`
- `Outputs:`
- `Expected runtime:`
- `How to run:`

## Directory Layout

- `_shared/`: shared tooling and validators.
- `smoke/`: quick sanity checks.
- `pipeline_gates/`: edge -> sibling -> split/merge gate diagnostics.
- `sibling_calibration/`: sibling test calibration and inflation diagnostics.
- `branch_length/methods/`: branch-length method derivation and method-focused diagnostics.
- `tree_construction/`: tree-building and topology diagnostics.
- `sbm/`: stochastic block model (SBM) specific diagnostics and transformations.
- `projection_power/`: projection/statistical power diagnostics.
- `case_studies/`: targeted benchmark case analyses.
- `diagnostics/`: cross-cutting behavior and failure diagnostics.
- `reports/`: generated report artifacts.
- `archive/`: legacy scripts kept for traceability (not required to satisfy active naming/header contract).

## Validation

Validate active structured scripts:

```bash
python debug_scripts/_shared/validate_debug_scripts.py
```

Also include archive scripts:

```bash
python debug_scripts/_shared/validate_debug_scripts.py --include-archive
```

Also include legacy root-level scripts during migration:

```bash
python debug_scripts/_shared/validate_debug_scripts.py --include-root-legacy
```

## Current Migration Scope

This migration pass standardized and relocated previously identified ad-hoc scripts into:

- `smoke/`
- `case_studies/`
- `sibling_calibration/`
- `branch_length/methods/`
- `projection_power/`
- `archive/` (legacy/non-runnable scripts)
