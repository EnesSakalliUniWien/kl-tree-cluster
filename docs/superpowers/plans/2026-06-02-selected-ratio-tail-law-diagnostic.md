# Selected Ratio Tail Law Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and commit a diagnostic-only selected-ratio tail-law evaluation for selected-hierarchy null records, with explicit production-admissibility failures and wiki evidence.

**Architecture:** Extend the existing selected-hierarchy geometry diagnostic rather than creating a production calibration path. The diagnostic computes within-context held-out tail exceedance for \(R_u=W_u/(a_u\nu_u)\), writes a `selected_ratio_tail_law.csv` evidence table, and documents that zero production-admissible contexts means no external fallback is available.

**Tech Stack:** Python 3.11, pandas, NumPy, pytest, ruff, docs-as-code wiki lint via `make wiki-lint`, CSV/JSON evidence under `raw/assets/benchmark-results/`.

Use `DEFAULT_SIBLING_ALPHA` from
`tree_break_selection.hierarchy_analysis.statistics.alpha_contract` in code
snippets; statistical thresholds are not read from `config.py`.

---

## File Structure

- Modify `benchmarks/diagnostics/calibration/selected_hierarchy_geometry_covariates.py`
  - Owns the selected-hierarchy geometry diagnostic.
  - Add diagnostic-only tail-law constants, context binning, held-out fold evaluation, production-admissibility checks, output writing, and manifest metadata.
- Modify `tests/validation/53_test_selected_hierarchy_geometry_covariates.py`
  - Owns focused validation for selected-hierarchy geometry diagnostics.
  - Add synthetic context fields, explicit independent simulation ids, and tests for tail-law support reporting and admissibility decisions.
- Create `raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_broad_200/`
  - Stores the locked broad diagnostic evidence: `manifest.json`, case summaries, geometry summaries, candidate-equation tables, and `selected_ratio_tail_law.csv`.
- Create `wiki/sources/selected-ratio-tail-law-diagnostic-20260602.md`
  - Source summary for the broad selected-ratio tail-law diagnostic.
- Modify `wiki/index.md`
  - Adds index coverage for the new source page.
- Modify `wiki/log.md`
  - Adds a dated chronology entry.
- Modify `wiki/analyses/selected-hierarchy-geometric-law-map.md`
  - Records that selected-ratio tail ranking and production tail calibration are different targets.
- Modify `wiki/analyses/selected-hierarchy-null-support-contract.md`
  - Records the exact tail-law context and the no-production-admissible result.
- Modify `wiki/analyses/selected-hierarchy-selection-geometry.md`
  - Places the result in the geometric explanation of same-data hierarchy selection.
- Modify `wiki/questions/open-mathematical-questions.md`
  - Narrows the open problem to admissible support and context design.

### Task 1: Confirm The Worktree And Evidence Boundary

**Files:**
- Inspect: `/Users/berksakalli/Projects/tree-break-selection`

- [ ] **Step 1: Inspect current git status**

Run:

```bash
git status --short
```

Expected: only selected-ratio tail-law diagnostic files, wiki files, and the new result directory are dirty.

- [ ] **Step 2: Inspect current diff size**

Run:

```bash
git diff --stat
```

Expected: changes are concentrated in:

```text
benchmarks/diagnostics/calibration/selected_hierarchy_geometry_covariates.py
tests/validation/53_test_selected_hierarchy_geometry_covariates.py
wiki/analyses/selected-hierarchy-geometric-law-map.md
wiki/analyses/selected-hierarchy-null-support-contract.md
wiki/analyses/selected-hierarchy-selection-geometry.md
wiki/index.md
wiki/log.md
wiki/questions/open-mathematical-questions.md
```

- [ ] **Step 3: Check generated evidence is small enough to commit**

Run:

```bash
du -sh raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_broad_200
wc -l raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_broad_200/*.csv raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_broad_200/manifest.json
```

Expected: the directory is about `100K`, and the result tables are compact CSV/JSON evidence, not raw selected-record dumps.

### Task 2: Add Tail-Law Tests First

**Files:**
- Modify: `tests/validation/53_test_selected_hierarchy_geometry_covariates.py`
- Test: `tests/validation/53_test_selected_hierarchy_geometry_covariates.py`

- [ ] **Step 1: Import the new evaluator in the validation test**

Add this imported name to the existing import block:

```python
from benchmarks.diagnostics.calibration.selected_hierarchy_geometry_covariates import (
    evaluate_candidate_equation_holdout,
    evaluate_candidate_equations,
    evaluate_covariate_block_models,
    evaluate_covariate_relationships,
    evaluate_selected_ratio_tail_law,
    run_selected_hierarchy_geometry_covariate_study,
    summarize_selected_geometry_by_case,
)
```

- [ ] **Step 2: Extend synthetic geometry records with canonical tail-law context fields**

Inside `_geometry_records()`, include these keys in every synthetic row:

```python
"feature_family": "bernoulli",
"parent_size_bin": "root_0.75_1",
```

- [ ] **Step 3: Add support-reporting test**

Append this test after the candidate-equation holdout test:

```python
def test_selected_ratio_tail_law_reports_context_support_and_holdout_error() -> None:
    records = _multi_case_geometry_records()

    tail_law = evaluate_selected_ratio_tail_law(
        records,
        n_folds=2,
        min_train_simulations=2,
        min_train_records=2,
        required_min_matching_simulations=100,
        required_min_matched_records=100,
    )

    assert "edge_action_bin" in tail_law.columns
    first = tail_law.iloc[0]
    assert first["tail_law_status"] == "descriptive_holdout_tail_law"
    assert not bool(first["production_tail_law_admissible"])
    assert "matching_simulations_below_tail_resolution_contract" in str(
        first["tail_law_admissibility_failure_reasons"]
    )
    assert float(first["heldout_exceedance_absolute_error"]) >= 0.0
```

- [ ] **Step 4: Add admissible-context test**

Append this test after the support-reporting test:

```python
def test_selected_ratio_tail_law_can_mark_supported_context_admissible() -> None:
    records = pd.concat([_geometry_records()] * 8, ignore_index=True)
    records["replicate_index"] = np.arange(records.shape[0])
    records["selected_hierarchy_simulation_id"] = [
        f"case:{replicate_index}" for replicate_index in records["replicate_index"]
    ]

    tail_law = evaluate_selected_ratio_tail_law(
        records,
        n_folds=4,
        min_train_simulations=10,
        min_train_records=10,
        required_min_matching_simulations=40,
        required_min_matched_records=40,
        max_exceedance_standard_error=0.2,
    )

    assert bool(tail_law.iloc[0]["production_tail_law_admissible"])
    assert tail_law.iloc[0]["tail_law_admissibility_failure_reasons"] == ""
```

- [ ] **Step 5: Extend smoke-output assertion**

Inside `test_smoke_run_writes_geometry_outputs()`, add:

```python
assert (output_dir / "selected_ratio_tail_law.csv").exists()
```

- [ ] **Step 6: Run the focused test and confirm failure before implementation**

Run:

```bash
uv run pytest tests/validation/53_test_selected_hierarchy_geometry_covariates.py::test_selected_ratio_tail_law_reports_context_support_and_holdout_error -q
```

Expected before implementation: import failure for `evaluate_selected_ratio_tail_law` or missing `edge_action_bin`.

### Task 3: Implement Diagnostic-Only Tail-Law Evaluation

**Files:**
- Modify: `benchmarks/diagnostics/calibration/selected_hierarchy_geometry_covariates.py`
- Test: `tests/validation/53_test_selected_hierarchy_geometry_covariates.py`

- [ ] **Step 1: Add canonical diagnostic constants**

Place these definitions after `RESPONSE_COLUMN`:

```python
TAIL_LAW_ROLE = "descriptive_selected_ratio_tail_law_not_calibration"
TAIL_LAW_CONTEXT_COLUMNS = (
    "source_family",
    "feature_family",
    "parent_size_bin",
    "sibling_projection_dimension",
    "edge_action_bin",
)
EDGE_ACTION_BINS = (0.0, 2.0, 4.0, 6.0, 8.0, np.inf)
EDGE_ACTION_BIN_LABELS = (
    "edge_action_0_2",
    "edge_action_2_4",
    "edge_action_4_6",
    "edge_action_6_8",
    "edge_action_ge8",
)
```

- [ ] **Step 2: Add strict edge-action binning**

Place this function after `_candidate_equation_table()`:

```python
def _edge_action_bin(edge_action: float) -> str:
    if not np.isfinite(edge_action) or edge_action < 0.0:
        raise ValueError(f"edge_action must be finite and non-negative; got {edge_action!r}.")
    for lower, upper, label in zip(
        EDGE_ACTION_BINS,
        EDGE_ACTION_BINS[1:],
        EDGE_ACTION_BIN_LABELS,
    ):
        if lower <= edge_action < upper:
            return label
    raise ValueError(f"edge_action did not match a bin: {edge_action!r}.")
```

- [ ] **Step 3: Add the tail-law table builder**

Place this function after `_edge_action_bin()`:

```python
def _selected_ratio_tail_law_table(records: pd.DataFrame) -> pd.DataFrame:
    table = _candidate_equation_table(records)
    table["edge_action_bin"] = [
        _edge_action_bin(float(value)) for value in table["edge_action"]
    ]
    return table
```

- [ ] **Step 4: Add fold-level held-out tail evaluation**

Place this function after `_selected_ratio_tail_law_table()`:

```python
def _fold_tail_law_evaluation(
    group: pd.DataFrame,
    *,
    alpha: float,
    n_folds: int,
    min_train_simulations: int,
    min_train_records: int,
) -> dict[str, object]:
    predictions: list[np.ndarray] = []
    observed: list[np.ndarray] = []
    thresholds: list[float] = []
    failures: list[str] = []
    n_train_rows = 0
    n_test_rows = 0
    used_folds = 0
    fold_ids = _simulation_fold_ids(group, n_folds=n_folds)

    for fold_id in sorted(int(value) for value in fold_ids.unique()):
        train = group.loc[fold_ids != fold_id]
        test = group.loc[fold_ids == fold_id]
        n_train_rows += int(train.shape[0])
        n_test_rows += int(test.shape[0])
        if test.empty:
            failures.append(f"fold_{fold_id}:empty_test")
            continue
        train_simulations = int(train["selected_hierarchy_simulation_id"].nunique())
        if train_simulations < min_train_simulations:
            failures.append(f"fold_{fold_id}:insufficient_train_simulations")
            continue
        if train.shape[0] < min_train_records:
            failures.append(f"fold_{fold_id}:insufficient_train_records")
            continue

        train_ratios = train["selected_hierarchy_ratio"].to_numpy(dtype=float)
        threshold = float(np.quantile(train_ratios, 1.0 - alpha))
        test_ratios = test["selected_hierarchy_ratio"].to_numpy(dtype=float)
        predictions.append(np.full(test_ratios.shape[0], threshold, dtype=float))
        observed.append(test_ratios)
        thresholds.append(threshold)
        used_folds += 1

    if not predictions:
        return {
            "n_train_rows_across_folds": int(n_train_rows),
            "n_test_rows_across_folds": int(n_test_rows),
            "n_used_folds": 0,
            "tail_threshold_mean": np.nan,
            "tail_threshold_median": np.nan,
            "heldout_exceedance_rate": np.nan,
            "heldout_exceedance_absolute_error": np.nan,
            "heldout_exceedance_standard_error": np.nan,
            "tail_law_status": "no_valid_tail_law_folds",
            "tail_law_failure_reasons": ";".join(failures),
        }

    predicted_thresholds = np.concatenate(predictions)
    observed_ratios = np.concatenate(observed)
    exceedances = observed_ratios > predicted_thresholds
    exceedance_rate = float(np.mean(exceedances))
    exceedance_se = float(
        np.sqrt(exceedance_rate * (1.0 - exceedance_rate) / exceedances.shape[0])
    )
    return {
        "n_train_rows_across_folds": int(n_train_rows),
        "n_test_rows_across_folds": int(n_test_rows),
        "n_used_folds": int(used_folds),
        "tail_threshold_mean": float(np.mean(thresholds)),
        "tail_threshold_median": float(np.median(thresholds)),
        "heldout_exceedance_rate": exceedance_rate,
        "heldout_exceedance_absolute_error": float(abs(exceedance_rate - alpha)),
        "heldout_exceedance_standard_error": exceedance_se,
        "tail_law_status": "descriptive_holdout_tail_law",
        "tail_law_failure_reasons": ";".join(failures),
    }
```

- [ ] **Step 5: Add production-admissibility failure names**

Place this function after `_fold_tail_law_evaluation()`:

```python
def _tail_law_admissibility_failures(
    *,
    n_matching_simulations: int,
    n_records: int,
    heldout_exceedance_se: float,
    required_min_matching_simulations: int,
    required_min_matched_records: int,
    max_exceedance_standard_error: float,
) -> list[str]:
    failures: list[str] = []
    if n_matching_simulations < required_min_matching_simulations:
        failures.append("matching_simulations_below_tail_resolution_contract")
    if n_records < required_min_matched_records:
        failures.append("matched_records_below_tail_resolution_contract")
    if (
        not np.isfinite(heldout_exceedance_se)
        or heldout_exceedance_se > max_exceedance_standard_error
    ):
        failures.append("heldout_exceedance_se_above_contract")
    return failures
```

- [ ] **Step 6: Add public diagnostic evaluator**

Place this function after `_tail_law_admissibility_failures()`:

```python
def evaluate_selected_ratio_tail_law(
    records: pd.DataFrame,
    *,
    alpha: float = float(DEFAULT_SIBLING_ALPHA),
    n_folds: int = 5,
    min_train_simulations: int = 20,
    min_train_records: int = 20,
    required_min_matching_simulations: int = 499,
    required_min_matched_records: int = 499,
    max_exceedance_standard_error: float = 0.002,
    context_columns: Sequence[str] = TAIL_LAW_CONTEXT_COLUMNS,
) -> pd.DataFrame:
    r"""Evaluate selected-ratio tail quantiles inside explicit contexts.

    This diagnostic estimates \(P(R_u > r \mid M_u)\) with held-out replicate
    folds. It reports support and precision; it does not create an external
    calibration fallback.
    """
    if records.empty:
        return pd.DataFrame()
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1).")
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")
    if min_train_simulations <= 0 or min_train_records <= 0:
        raise ValueError("minimum training support values must be positive.")
    if required_min_matching_simulations <= 0 or required_min_matched_records <= 0:
        raise ValueError("required production support values must be positive.")
    if max_exceedance_standard_error <= 0.0:
        raise ValueError("max_exceedance_standard_error must be positive.")

    table = _selected_ratio_tail_law_table(records)
    missing_columns = [column for column in context_columns if column not in table.columns]
    if missing_columns:
        raise KeyError(f"Missing tail-law context column(s): {missing_columns!r}.")

    rows: list[dict[str, object]] = []
    for group_values, group in table.groupby(list(context_columns), dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        ratios = group["selected_hierarchy_ratio"].to_numpy(dtype=float)
        n_matching_simulations = int(group["selected_hierarchy_simulation_id"].nunique())
        n_records = int(group.shape[0])
        fold_summary = _fold_tail_law_evaluation(
            group,
            alpha=alpha,
            n_folds=n_folds,
            min_train_simulations=min_train_simulations,
            min_train_records=min_train_records,
        )
        admissibility_failures = _tail_law_admissibility_failures(
            n_matching_simulations=n_matching_simulations,
            n_records=n_records,
            heldout_exceedance_se=float(
                fold_summary["heldout_exceedance_standard_error"]
            ),
            required_min_matching_simulations=required_min_matching_simulations,
            required_min_matched_records=required_min_matched_records,
            max_exceedance_standard_error=max_exceedance_standard_error,
        )
        row = {
            column: value for column, value in zip(context_columns, group_values)
        }
        row.update(
            {
                "tail_law_role": TAIL_LAW_ROLE,
                "alpha": float(alpha),
                "n_records": n_records,
                "n_matching_simulations": n_matching_simulations,
                "record_tail_resolution": float(1.0 / (n_records + 1)),
                "matching_simulation_tail_resolution": float(
                    1.0 / (n_matching_simulations + 1)
                ),
                "required_min_matching_simulations": int(
                    required_min_matching_simulations
                ),
                "required_min_matched_records": int(required_min_matched_records),
                "selected_ratio_mean": float(np.mean(ratios)),
                "selected_ratio_median": float(np.quantile(ratios, 0.5)),
                "selected_ratio_trainless_q90": float(np.quantile(ratios, 0.9)),
                "selected_ratio_trainless_q95": float(np.quantile(ratios, 0.95)),
                "selected_ratio_trainless_q99": float(np.quantile(ratios, 0.99)),
                "production_tail_law_admissible": not admissibility_failures,
                "tail_law_admissibility_failure_reasons": ";".join(
                    admissibility_failures
                ),
            }
        )
        row.update(fold_summary)
        rows.append(row)

    return (
        pd.DataFrame.from_records(rows)
        .sort_values(list(context_columns))
        .reset_index(drop=True)
    )
```

- [ ] **Step 7: Include evaluator in outputs and manifest**

Inside `run_selected_hierarchy_geometry_covariate_study()`, add:

```python
selected_ratio_tail_law = evaluate_selected_ratio_tail_law(selected_records)
```

Add this entry to `outputs`:

```python
"selected_ratio_tail_law": selected_ratio_tail_law,
```

Add this entry to `manifest`:

```python
"selected_ratio_tail_law": {
    "role": TAIL_LAW_ROLE,
    "context_columns": list(TAIL_LAW_CONTEXT_COLUMNS),
    "edge_action_bins": [
        {
            "label": label,
            "lower": float(lower),
            "upper": None if np.isinf(upper) else float(upper),
        }
        for lower, upper, label in zip(
            EDGE_ACTION_BINS,
            EDGE_ACTION_BINS[1:],
            EDGE_ACTION_BIN_LABELS,
        )
    ],
    "alpha": float(DEFAULT_SIBLING_ALPHA),
    "independent_simulation_id_column": "selected_hierarchy_simulation_id",
    "production_min_matching_simulations": 499,
    "production_min_matched_records": 499,
    "production_max_exceedance_standard_error": 0.002,
},
```

Update the manifest note to this exact text:

```python
"Diagnostic-only selected-hierarchy geometry study. Relationship "
"tables are descriptive and unadjusted; holdout tables are "
"descriptive transfer checks, not production calibration. The "
"selected-ratio tail-law table reports context support and held-out "
"tail exceedance only. These outputs do not define external "
"calibration borrowing, scalar inflation, or production fallback."
```

- [ ] **Step 8: Export the public evaluator**

Add this name to `__all__`:

```python
"evaluate_selected_ratio_tail_law",
```

- [ ] **Step 9: Run focused validation**

Run:

```bash
ruff check benchmarks/diagnostics/calibration/selected_hierarchy_geometry_covariates.py tests/validation/53_test_selected_hierarchy_geometry_covariates.py
uv run pytest tests/validation/53_test_selected_hierarchy_geometry_covariates.py
```

Expected:

```text
All checks passed!
10 passed
```

### Task 4: Run The Broad Selected-Ratio Tail-Law Diagnostic

**Files:**
- Create: `raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_broad_200/manifest.json`
- Create: `raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_broad_200/case_summary.csv`
- Create: `raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_broad_200/geometry_summary_by_case.csv`
- Create: `raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_broad_200/selected_ratio_tail_law.csv`
- Create: `raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_broad_200/candidate_equations.csv`
- Create: `raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_broad_200/candidate_equation_holdout.csv`
- Create: `raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_broad_200/covariate_relationships.csv`
- Create: `raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_broad_200/covariate_block_models.csv`

- [ ] **Step 1: Run the diagnostic**

Run:

```bash
uv run python -m benchmarks.diagnostics.calibration.selected_hierarchy_geometry_covariates \
  --case-names gauss_null_large,gauss_clear_medium,dim_diffuse_6c_136f,binary_low_noise_4c,cat_highcard_20cat_4c,cat_highd_3cat_500feat,overlap_heavy_4c_med_feat,phylo_dna_8taxa_med_mut,sbm_moderate \
  --n-replicates 200 \
  --seed 20260602 \
  --output-dir raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_broad_200
```

Expected: eight cases complete, and `sbm_moderate` is an explicit skip because the selected-hierarchy null generator does not support precomputed TBS tree distances.

- [ ] **Step 2: Summarize tail-law support**

Run:

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path
base = Path("raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_broad_200")
t = pd.read_csv(base / "selected_ratio_tail_law.csv")
print(t["tail_law_status"].value_counts(dropna=False).to_string())
print(t["production_tail_law_admissible"].value_counts(dropna=False).to_string())
h = t[t["tail_law_status"].eq("descriptive_holdout_tail_law")]
print("descriptive_contexts", len(h))
print("median_abs_error", h["heldout_exceedance_absolute_error"].median())
print("near_alpha_low_se", int(((h["heldout_exceedance_absolute_error"] <= 0.001) & (h["heldout_exceedance_standard_error"] <= 0.002)).sum()))
PY
```

Expected:

```text
no_valid_tail_law_folds         65
descriptive_holdout_tail_law    39
False    104
descriptive_contexts 39
median_abs_error 0.004388489208633
near_alpha_low_se 10
```

### Task 5: Write Wiki Evidence

**Files:**
- Create: `wiki/sources/selected-ratio-tail-law-diagnostic-20260602.md`
- Modify: `wiki/index.md`
- Modify: `wiki/log.md`
- Modify: `wiki/analyses/selected-hierarchy-geometric-law-map.md`
- Modify: `wiki/analyses/selected-hierarchy-null-support-contract.md`
- Modify: `wiki/analyses/selected-hierarchy-selection-geometry.md`
- Modify: `wiki/questions/open-mathematical-questions.md`

- [ ] **Step 1: Create the source page**

Create `wiki/sources/selected-ratio-tail-law-diagnostic-20260602.md` with:

```markdown
---
title: Selected Ratio Tail Law Diagnostic 2026-06-02
type: source
status: reviewed
updated: 2026-06-02
sources:
  - benchmarks/diagnostics/calibration/selected_hierarchy_geometry_covariates.py
  - tests/validation/53_test_selected_hierarchy_geometry_covariates.py
  - raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_broad_200/manifest.json
  - raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_broad_200/case_summary.csv
  - raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_broad_200/geometry_summary_by_case.csv
  - raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_broad_200/selected_ratio_tail_law.csv
tags:
  - source
  - calibration
  - selection
  - tail-law
---

# Selected Ratio Tail Law Diagnostic 2026-06-02

## Summary

This diagnostic tests whether a selected-ratio tail law can be estimated
inside explicit selected-hierarchy contexts. The object is
\[
R_u=\frac{W_u}{a_u\nu_u},
\]
evaluated only after same-data hierarchy construction, child-parent edge
opening, and focal sibling selection. The context is `source_family`,
`feature_family`, `parent_size_bin`, `sibling_projection_dimension`, and
`edge_action_bin`.

The result is diagnostic-only. It does not define a production external
calibration model, scalar inflation fallback, context-borrowing rule, or
application default. No context in the 200-replicate broad run is
production-admissible under the predeclared support contract.

## Key Points

- The production tail-law contract uses \(\alpha_{\mathrm{sib}}=0.01\) and
  requires at least `499` independent matching simulations, at least `499`
  matched selected records, and held-out exceedance standard error at most
  `0.002`.
- The broad run used `200` replicates over nine requested cases. Eight cases
  completed. `sbm_moderate` was skipped because the selected-hierarchy null
  generator does not own the precomputed TBS tree-distance contract.
- The selected-ratio tail table contains `104` contexts. `39` contexts have
  descriptive held-out tail-law folds; `65` have no valid tail-law folds.
- `0` of `104` contexts are production-admissible. Every context fails the
  independent matching-simulation requirement, and most also fail matched
  record count and held-out tail standard-error requirements.
- The independent simulation unit is `selected_hierarchy_simulation_id`, the
  pair of case id and replicate index. The largest context reaches `376`
  matching simulations, still below the `499` production threshold.
- High-support small-node, high-edge-action contexts often have held-out
  exceedance near the target `0.01`. Ten descriptive contexts have absolute
  exceedance error at most `0.001` and held-out standard error at most
  `0.002`.
- Sparse root or low-edge-action contexts are unstable. The median absolute
  held-out exceedance error among descriptive contexts is about `0.0052`, and
  the worst sparse context has absolute error about `0.323`.
- The case-level selected-ratio scale remains large and family-dependent:
  mean \(R\) is about `30.9` for `gauss_clear_medium`, `62.9` for
  `gauss_null_large`, `109.3` for `dim_diffuse_6c_136f`, `210.1` for
  `phylo_dna_8taxa_med_mut`, and `579.8` for
  `cat_highd_3cat_500feat`.
- The result sharpens the mathematical target. A high global tail-ranking AUC
  is not enough; a production external law would need admissible within-context
  support and calibrated absolute tail probabilities.

## Evidence

- `benchmarks/diagnostics/calibration/selected_hierarchy_geometry_covariates.py`
  implements the descriptive selected-ratio tail-law table and explicit
  production-admissibility checks.
- `tests/validation/53_test_selected_hierarchy_geometry_covariates.py`
  validates context support reporting, held-out exceedance estimation,
  admissibility failures, and output creation.
- `raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_broad_200/manifest.json`
  records the run seed, cases, context columns, edge-action bins, alpha, and
  production support thresholds.
- `raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_broad_200/case_summary.csv`
  records case completion status, selected-record counts, and the explicit SBM
  skip reason.
- `raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_broad_200/geometry_summary_by_case.csv`
  records selected-ratio and angular/eigenvalue summaries by completed case.
- `raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_broad_200/selected_ratio_tail_law.csv`
  records context-level support, held-out exceedance rates, admissibility
  failures, and diagnostic status.

## Links

- [[selected-hierarchy-null-support-contract]]
- [[selected-hierarchy-selection-geometry]]
- [[selected-hierarchy-geometric-law-map]]
- [[selected-hierarchy-geometry-covariates-20260602]]
- [[open-mathematical-questions]]
```

- [ ] **Step 2: Add index coverage**

In `wiki/index.md`, add this under source summaries:

```markdown
- [[selected-ratio-tail-law-diagnostic-20260602]] - within-context
  selected-ratio tail-law diagnostic showing descriptive tail behavior but no
  production-admissible context in the broad 200-replicate selected-hierarchy
  run.
```

- [ ] **Step 3: Add log entry**

In `wiki/log.md`, append this under `### 2026-06-02`:

```markdown
- Added a within-context selected-ratio tail-law diagnostic to the same
  selected-hierarchy geometry tool. The context uses source family, feature
  family, parent-size bin, sibling projection dimension, and binned edge
  action. A broad 200-replicate run produced descriptive held-out tail-law
  rows but no production-admissible context under the strict `499` matching
  simulation, `499` matched record, and `0.002` held-out SE contract. Added
  [[selected-ratio-tail-law-diagnostic-20260602]].
```

- [ ] **Step 4: Update analysis and question pages**

Add a short paragraph to each of these pages stating the same contract result:

```text
In the 200-replicate broad run, no context is production-admissible under the current support contract. Some high-support small-parent, high-edge-action contexts have held-out exceedance near alpha=0.01, but sparse root and low-edge-action contexts are unstable. This describes a possible conditional tail law; it does not license a production external calibration model.
```

Files:

```text
wiki/analyses/selected-hierarchy-geometric-law-map.md
wiki/analyses/selected-hierarchy-null-support-contract.md
wiki/analyses/selected-hierarchy-selection-geometry.md
wiki/questions/open-mathematical-questions.md
```

- [ ] **Step 5: Run wiki lint**

Run:

```bash
make wiki-lint
```

Expected:

```text
wiki-lint passed
```

### Task 6: Final Verification

**Files:**
- Verify: all files changed by Tasks 2 through 5.

- [ ] **Step 1: Run lint and focused tests**

Run:

```bash
ruff check benchmarks/diagnostics/calibration/selected_hierarchy_geometry_covariates.py tests/validation/53_test_selected_hierarchy_geometry_covariates.py
uv run pytest tests/validation/53_test_selected_hierarchy_geometry_covariates.py
make wiki-lint
git diff --check
```

Expected:

```text
All checks passed!
10 passed
wiki-lint passed
```

`git diff --check` should produce no output.

- [ ] **Step 2: Run full test suite**

Run:

```bash
uv run pytest
```

Expected:

```text
331 passed, 2 warnings
```

The two warnings are the existing sklearn spectral-embedding connectivity warnings in `tests/pipeline/53_test_runner_contract_alignment.py`.

- [ ] **Step 3: Review final diff**

Run:

```bash
git status --short
git diff --stat
git diff --check
```

Expected: status contains only the diagnostic code, focused tests, wiki updates, new source page, and compact result directory. `git diff --check` produces no output.

### Task 7: Commit The Diagnostic Package

**Files:**
- Stage: `benchmarks/diagnostics/calibration/selected_hierarchy_geometry_covariates.py`
- Stage: `tests/validation/53_test_selected_hierarchy_geometry_covariates.py`
- Stage: `raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_broad_200/`
- Stage: `wiki/sources/selected-ratio-tail-law-diagnostic-20260602.md`
- Stage: `wiki/index.md`
- Stage: `wiki/log.md`
- Stage: `wiki/analyses/selected-hierarchy-geometric-law-map.md`
- Stage: `wiki/analyses/selected-hierarchy-null-support-contract.md`
- Stage: `wiki/analyses/selected-hierarchy-selection-geometry.md`
- Stage: `wiki/questions/open-mathematical-questions.md`

- [ ] **Step 1: Stage the package**

Run:

```bash
git add \
  benchmarks/diagnostics/calibration/selected_hierarchy_geometry_covariates.py \
  tests/validation/53_test_selected_hierarchy_geometry_covariates.py \
  raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_broad_200 \
  wiki/sources/selected-ratio-tail-law-diagnostic-20260602.md \
  wiki/index.md \
  wiki/log.md \
  wiki/analyses/selected-hierarchy-geometric-law-map.md \
  wiki/analyses/selected-hierarchy-null-support-contract.md \
  wiki/analyses/selected-hierarchy-selection-geometry.md \
  wiki/questions/open-mathematical-questions.md
```

Expected: `git status --short` shows these paths staged.

- [ ] **Step 2: Inspect staged diff**

Run:

```bash
git diff --cached --stat
git diff --cached --check
```

Expected: staged diff contains one coherent diagnostic package; `git diff --cached --check` produces no output.

- [ ] **Step 3: Commit**

Run:

```bash
git commit -m "Add selected ratio tail-law diagnostic"
```

Expected: commit succeeds on branch `dev`.

## Self-Review

**Spec coverage:** The plan covers the diagnostic evaluator, tests, broad run, wiki source page, adjacent wiki pages, verification commands, and commit packaging. It explicitly preserves the no-production-fallback contract.

**Placeholder scan:** The plan contains no unresolved placeholder tokens, vague fill-in instructions, or references to missing functions. Every code-changing step includes concrete code or exact text.

**Type consistency:** The function name `evaluate_selected_ratio_tail_law`, field names `tail_law_status`, `production_tail_law_admissible`, `tail_law_admissibility_failure_reasons`, and context columns match across implementation, tests, manifest, CSV evidence, and wiki text.
