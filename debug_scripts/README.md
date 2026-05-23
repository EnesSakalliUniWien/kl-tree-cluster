# Debug Scripts Structure

This folder contains searchable diagnostic scripts with an enforceable structure.

## Naming Contract

All active diagnostic scripts should use:

`q_<question>__<method>__<scope>.py`

Examples:

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
- `sibling_calibration/`: sibling test calibration and scale diagnostics.
- `branch_length/methods/`: branch-length method derivation and method-focused diagnostics.
- `tree_construction/`: tree-building and topology diagnostics.
- `sbm/`: stochastic block model (SBM) specific diagnostics and transformations.
- `projection_power/`: projection/statistical power diagnostics.
- `case_studies/`: targeted benchmark case analyses.
- `diagnostics/`: cross-cutting behavior and failure diagnostics.
- `reports/`: generated report artifacts.
- `archive/`: non-runnable historical scripts kept outside the active validation target.

## Validation

Validate active structured scripts:

```bash
python debug_scripts/_shared/validate_debug_scripts.py
```

Also include archive scripts:

```bash
python debug_scripts/_shared/validate_debug_scripts.py --include-archive
```

## Current Migration Scope

Active scripts live in:

- `smoke/`
- `case_studies/`
- `sibling_calibration/`
- `branch_length/methods/`
- `projection_power/`
- `archive/` (non-runnable historical scripts)
