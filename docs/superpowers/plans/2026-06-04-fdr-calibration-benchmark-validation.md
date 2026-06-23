# FDR Calibration Benchmark Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether traversal-aligned sibling BH controls the relevant error family after edge gating, empirical-null inflation, and traversal, while separately validating the open calibration objects and benchmark consequences.

**Architecture:** Keep this diagnostic-only. Do not change production defaults, add external-null fallbacks, or tune alpha from incomplete evidence. Build layered validation: first test FDR machinery with valid synthetic p-values, then test fixed-tree projected-Wald calibration, then selected-tree/selected-hierarchy calibration, then benchmark impact.

**Tech Stack:** Python, NumPy, pandas, scipy/statsmodels BH routines, pytest, existing Tree-Break Selection benchmark and wiki docs-as-code.

---

## Mathematical Contract

Let \(T(X)\) be the hierarchy selected from data \(X\). Let \(E_u\) be the child-parent edge event opening node \(u\), and let \(S_u\) be the sibling null hypothesis at internal node \(u\):

\[
H_{0,u}^{\mathrm{sib}}:\theta_{L(u)}=\theta_{R(u)}.
\]

The active sibling FDR procedure does not test all internal nodes. It tests the traversal-reachable family

\[
\mathcal F_{\mathrm{trav}}(X)
=
\{u: u \text{ has binary children and edge prerequisites are open along the traversal path}\}.
\]

The open question is not whether ordinary BH works on valid fixed p-values. The question is whether

\[
\operatorname{FDR}
=
\mathbb E\left[
\frac{
\#\{u\in\mathcal F_{\mathrm{trav}}(X): H_{0,u}^{\mathrm{sib}}\text{ true and rejected}\}
}{
\max(1,\#\{u\in\mathcal F_{\mathrm{trav}}(X): S_u\text{ rejected}\})
}
\right]
\le \alpha_{\mathrm{sib}}
\]

holds when \(\mathcal F_{\mathrm{trav}}\), the projection, and the inflation estimator are all data-dependent.

Expected outcome categories:

- `algorithmic_fdr_control`: traversal-aligned BH controls FDR when supplied valid p-values.
- `fixed_tree_calibration_failure`: sibling p-values are invalid even with fixed hierarchy.
- `edge_selected_family_failure`: edge gating makes the tested sibling family anti-conservative.
- `inflation_support_failure`: empirical-null inflation has no admissible null support.
- `selected_projection_failure`: selected PCA/projection changes the null law.
- `benchmark_tradeoff_only`: benchmark improves but Type-I/FDR remains unproved.

## File Structure

- Create `benchmarks/validation/traversal_sibling_fdr_null.py`
  - Layered simulation runner for sibling FDR under controlled null structures.
  - Emits one row per simulation and one row per tested sibling node.
- Create `benchmarks/cloud/aws_traversal_sibling_fdr_null.py`
  - AWS Batch shard/merge wrapper for the runner.
- Create `tests/validation/67_test_traversal_sibling_fdr_null.py`
  - Unit tests for simulation contracts and FDR summaries.
- Create `tests/validation/68_test_aws_traversal_sibling_fdr_null.py`
  - Shard/merge contract tests.
- Modify `wiki/questions/open-mathematical-questions.md`
  - Add the validation ladder and current status.
- Modify `wiki/index.md` and `wiki/log.md`
  - Add source coverage after results exist.
- Create `wiki/sources/traversal-sibling-fdr-null-20260604.md`
  - Only after a real run writes raw evidence.

## Task 1: Define The FDR Simulation Contract

**Files:**
- Create: `benchmarks/validation/traversal_sibling_fdr_null.py`
- Test: `tests/validation/67_test_traversal_sibling_fdr_null.py`

- [ ] **Step 1: Write failing contract tests**

```python
from __future__ import annotations

from benchmarks.validation.traversal_sibling_fdr_null import (
    FdrLayer,
    TraversalSiblingFdrConfig,
    classify_fdr_outcome,
    estimate_fdr,
    parse_fdr_layers,
)


def test_parse_fdr_layers_accepts_named_layers() -> None:
    assert parse_fdr_layers("synthetic_valid_p,fixed_tree_wald") == (
        FdrLayer.SYNTHETIC_VALID_P,
        FdrLayer.FIXED_TREE_WALD,
    )


def test_estimate_fdr_uses_false_discoveries_over_rejections() -> None:
    summary = estimate_fdr(
        [
            {"n_false_rejections": 0, "n_rejections": 0},
            {"n_false_rejections": 1, "n_rejections": 2},
        ]
    )
    assert summary["mean_fdp"] == 0.25
    assert summary["false_rejection_rate"] == 0.5


def test_classify_fdr_outcome_separates_algorithm_from_calibration() -> None:
    assert (
        classify_fdr_outcome(layer=FdrLayer.SYNTHETIC_VALID_P, mean_fdp=0.008, alpha=0.01)
        == "algorithmic_fdr_control"
    )
    assert (
        classify_fdr_outcome(layer=FdrLayer.FIXED_TREE_WALD, mean_fdp=0.08, alpha=0.01)
        == "fixed_tree_calibration_failure"
    )


def test_config_rejects_invalid_alpha() -> None:
    try:
        TraversalSiblingFdrConfig(
            layer=FdrLayer.SYNTHETIC_VALID_P,
            case_names=("binary_2clusters",),
            replicates=10,
            alpha=0.0,
            base_seed=1,
        )
    except ValueError as exc:
        assert "alpha" in str(exc)
    else:
        raise AssertionError("invalid alpha accepted")
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
pytest -q tests/validation/67_test_traversal_sibling_fdr_null.py
```

Expected: import failure because `benchmarks.validation.traversal_sibling_fdr_null` does not exist.

- [ ] **Step 3: Implement the contract skeleton**

```python
"""Traversal-aligned sibling FDR null diagnostics.

Diagnostic only. This module does not change production FDR, alpha defaults, or
calibration behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable

import numpy as np


class FdrLayer(StrEnum):
    SYNTHETIC_VALID_P = "synthetic_valid_p"
    FIXED_TREE_WALD = "fixed_tree_wald"
    SELECTED_TREE_WALD = "selected_tree_wald"
    SELECTED_TREE_INFLATED = "selected_tree_inflated"


@dataclass(frozen=True)
class TraversalSiblingFdrConfig:
    layer: FdrLayer
    case_names: tuple[str, ...]
    replicates: int
    alpha: float
    base_seed: int

    def __post_init__(self) -> None:
        if self.replicates <= 0:
            raise ValueError(f"replicates must be positive; got {self.replicates!r}.")
        if not np.isfinite(self.alpha) or not (0.0 < float(self.alpha) <= 1.0):
            raise ValueError(f"alpha must be finite and in (0, 1]; got {self.alpha!r}.")
        if not self.case_names:
            raise ValueError("case_names must contain at least one case.")


def parse_fdr_layers(raw: str) -> tuple[FdrLayer, ...]:
    layers = tuple(FdrLayer(item.strip()) for item in raw.split(",") if item.strip())
    if not layers:
        raise ValueError("At least one FDR layer is required.")
    return layers


def estimate_fdr(rows: Iterable[dict[str, object]]) -> dict[str, float]:
    false_rejections: list[float] = []
    fdps: list[float] = []
    for row in rows:
        n_false = int(row["n_false_rejections"])
        n_rejections = int(row["n_rejections"])
        false_rejections.append(float(n_false > 0))
        fdps.append(float(n_false) / float(max(1, n_rejections)))
    if not fdps:
        raise ValueError("Cannot estimate FDR from zero simulation rows.")
    return {
        "mean_fdp": float(np.mean(fdps)),
        "false_rejection_rate": float(np.mean(false_rejections)),
        "n_simulations": float(len(fdps)),
    }


def classify_fdr_outcome(*, layer: FdrLayer, mean_fdp: float, alpha: float) -> str:
    if mean_fdp <= alpha:
        return "algorithmic_fdr_control" if layer == FdrLayer.SYNTHETIC_VALID_P else "empirical_control_observed"
    if layer == FdrLayer.SYNTHETIC_VALID_P:
        return "algorithmic_fdr_failure"
    if layer == FdrLayer.FIXED_TREE_WALD:
        return "fixed_tree_calibration_failure"
    if layer == FdrLayer.SELECTED_TREE_WALD:
        return "edge_selected_family_failure"
    if layer == FdrLayer.SELECTED_TREE_INFLATED:
        return "inflation_or_selected_family_failure"
    raise ValueError(f"Unhandled FDR layer: {layer!r}.")
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
pytest -q tests/validation/67_test_traversal_sibling_fdr_null.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/validation/traversal_sibling_fdr_null.py tests/validation/67_test_traversal_sibling_fdr_null.py
git commit -m "Add traversal sibling FDR diagnostic contract"
```

## Task 2: Implement Synthetic Valid-P FDR Layer

**Files:**
- Modify: `benchmarks/validation/traversal_sibling_fdr_null.py`
- Test: `tests/validation/67_test_traversal_sibling_fdr_null.py`

- [ ] **Step 1: Add failing synthetic-layer test**

```python
def test_synthetic_valid_p_layer_controls_fdr_with_valid_null_p_values() -> None:
    config = TraversalSiblingFdrConfig(
        layer=FdrLayer.SYNTHETIC_VALID_P,
        case_names=("synthetic_balanced_binary_tree",),
        replicates=200,
        alpha=0.05,
        base_seed=123,
    )
    outputs = run_traversal_sibling_fdr_layer(config)
    assert outputs["summary"]["n_simulations"] == 200.0
    assert outputs["summary"]["mean_fdp"] <= 0.08
    assert outputs["summary"]["outcome"] == "algorithmic_fdr_control"
```

- [ ] **Step 2: Implement deterministic synthetic tree rows**

Add:

```python
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.inflated_projected_wald_annotation.fdr_annotation import (
    apply_traversal_aligned_sibling_bh_results,
    init_sibling_annotation_df,
)
import networkx as nx
import pandas as pd


def _balanced_three_level_tree() -> nx.DiGraph:
    tree = nx.DiGraph()
    tree.add_edge("root", "A")
    tree.add_edge("root", "B")
    tree.add_edge("A", "A1")
    tree.add_edge("A", "A2")
    tree.add_edge("B", "B1")
    tree.add_edge("B", "B2")
    return tree


def _open_edge_annotations(tree: nx.DiGraph) -> pd.DataFrame:
    return init_sibling_annotation_df(
        pd.DataFrame(
            {
                "Child_Parent_Divergence_Significant": pd.Series(
                    {node: node != "root" for node in tree.nodes},
                    dtype=bool,
                )
            }
        )
    )


def _run_synthetic_valid_p_replicate(*, replicate_index: int, alpha: float, seed: int) -> dict[str, object]:
    tree = _balanced_three_level_tree()
    rng = np.random.default_rng(seed + replicate_index)
    parents = ["root", "A", "B"]
    p_values = rng.uniform(0.0, 1.0, size=len(parents))
    results = [(0.0, 1.0, float(value)) for value in p_values]
    annotated = apply_traversal_aligned_sibling_bh_results(
        tree,
        _open_edge_annotations(tree),
        parents,
        results,
        alpha,
    )
    rejected = annotated.loc[parents, "Sibling_BH_Different"].astype(bool).to_numpy()
    return {
        "replicate_index": int(replicate_index),
        "n_rejections": int(rejected.sum()),
        "n_false_rejections": int(rejected.sum()),
    }


def run_traversal_sibling_fdr_layer(config: TraversalSiblingFdrConfig) -> dict[str, object]:
    if config.layer != FdrLayer.SYNTHETIC_VALID_P:
        raise NotImplementedError(f"Layer {config.layer} is not implemented yet.")
    rows = [
        _run_synthetic_valid_p_replicate(
            replicate_index=index,
            alpha=float(config.alpha),
            seed=int(config.base_seed),
        )
        for index in range(config.replicates)
    ]
    summary = estimate_fdr(rows)
    summary["outcome"] = classify_fdr_outcome(
        layer=config.layer,
        mean_fdp=float(summary["mean_fdp"]),
        alpha=float(config.alpha),
    )
    return {"simulation_rows": rows, "summary": summary}
```

- [ ] **Step 3: Run synthetic-layer test**

Run:

```bash
pytest -q tests/validation/67_test_traversal_sibling_fdr_null.py
```

Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add benchmarks/validation/traversal_sibling_fdr_null.py tests/validation/67_test_traversal_sibling_fdr_null.py
git commit -m "Validate sibling BH on valid null p values"
```

## Task 3: Fixed-Tree Sibling Wald Calibration Layer

**Files:**
- Modify: `benchmarks/validation/traversal_sibling_fdr_null.py`
- Test: `tests/validation/67_test_traversal_sibling_fdr_null.py`

- [ ] **Step 1: Add test that fixed-tree layer records calibration status**

```python
def test_fixed_tree_wald_layer_reports_calibration_status() -> None:
    config = TraversalSiblingFdrConfig(
        layer=FdrLayer.FIXED_TREE_WALD,
        case_names=("binary_2clusters",),
        replicates=3,
        alpha=0.01,
        base_seed=20260604,
    )
    outputs = run_traversal_sibling_fdr_layer(config)
    assert outputs["summary"]["n_simulations"] == 3.0
    assert "outcome" in outputs["summary"]
    assert all("n_rejections" in row for row in outputs["simulation_rows"])
```

- [ ] **Step 2: Implement fixed-tree layer with existing benchmark context**

Implementation rule:

- Build one observed benchmark tree per case.
- Regenerate null data on the same fixed tree.
- Run child-parent and sibling annotations on the fixed tree.
- Treat every sibling null as true in the null replicate.
- Count false sibling rejections after traversal-aligned sibling BH.
- If empirical inflation support fails, record `inflation_support_failure` and do not convert to neutral p-values.

- [ ] **Step 3: Run focused test**

```bash
pytest -q tests/validation/67_test_traversal_sibling_fdr_null.py::test_fixed_tree_wald_layer_reports_calibration_status
```

Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add benchmarks/validation/traversal_sibling_fdr_null.py tests/validation/67_test_traversal_sibling_fdr_null.py
git commit -m "Add fixed tree sibling FDR calibration layer"
```

## Task 4: Selected-Tree And Inflated Selected-Tree Layers

**Files:**
- Modify: `benchmarks/validation/traversal_sibling_fdr_null.py`
- Test: `tests/validation/67_test_traversal_sibling_fdr_null.py`

- [ ] **Step 1: Add tests for explicit unsupported states**

```python
def test_selected_tree_layers_report_strict_support_failures() -> None:
    for layer in (FdrLayer.SELECTED_TREE_WALD, FdrLayer.SELECTED_TREE_INFLATED):
        config = TraversalSiblingFdrConfig(
            layer=layer,
            case_names=("binary_2clusters",),
            replicates=2,
            alpha=0.01,
            base_seed=20260604,
        )
        outputs = run_traversal_sibling_fdr_layer(config)
        assert outputs["summary"]["n_simulations"] == 2.0
        assert "n_support_failures" in outputs["summary"]
```

- [ ] **Step 2: Implement selected-tree layers**

Implementation rule:

- `selected_tree_wald`: regenerate null data, rebuild hierarchy, run raw sibling projected-Wald and traversal-aligned BH without empirical inflation.
- `selected_tree_inflated`: same as above, but include active empirical-null inflation.
- If active empirical-null support is invalid, count the replicate as support failure.
- Do not insert external calibration, scalar rescue factors, or neutral p-values.

- [ ] **Step 3: Run selected-tree tests**

```bash
pytest -q tests/validation/67_test_traversal_sibling_fdr_null.py::test_selected_tree_layers_report_strict_support_failures
```

Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add benchmarks/validation/traversal_sibling_fdr_null.py tests/validation/67_test_traversal_sibling_fdr_null.py
git commit -m "Add selected tree sibling FDR calibration layers"
```

## Task 5: AWS Sharding

**Files:**
- Create: `benchmarks/cloud/aws_traversal_sibling_fdr_null.py`
- Test: `tests/validation/68_test_aws_traversal_sibling_fdr_null.py`
- Modify: `benchmarks/cloud/aws/README.md`

- [ ] **Step 1: Write shard distribution tests**

```python
from pathlib import Path
import tempfile

from benchmarks.cloud.aws_traversal_sibling_fdr_null import (
    AwsTraversalSiblingFdrConfig,
    make_shard_spec,
)
from benchmarks.validation.traversal_sibling_fdr_null import FdrLayer


def test_make_shard_spec_distributes_replicates_by_modulo() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        config = AwsTraversalSiblingFdrConfig(
            layer=FdrLayer.SYNTHETIC_VALID_P,
            case_names=("binary_2clusters",),
            output_dir=Path(tmpdir),
            replicates=10,
            alpha=0.01,
            base_seed=1,
            shard_count=3,
        )
        first = make_shard_spec(config, 0)
        second = make_shard_spec(config, 1)
    assert first.replicate_indices == (0, 3, 6, 9)
    assert second.replicate_indices == (1, 4, 7)
```

- [ ] **Step 2: Implement AWS wrapper**

Use the pattern from `benchmarks/cloud/aws_selected_edge_type1_geometry.py`:

- `run-shard`
- `merge`
- S3 sync functions
- strict duplicate/missing replicate validation
- merged files:
  - `traversal_sibling_fdr_simulations.csv`
  - `traversal_sibling_fdr_summary.csv`
  - `aws_traversal_sibling_fdr_manifest.json`

- [ ] **Step 3: Run AWS wrapper tests**

```bash
pytest -q tests/validation/68_test_aws_traversal_sibling_fdr_null.py
```

Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add benchmarks/cloud/aws_traversal_sibling_fdr_null.py tests/validation/68_test_aws_traversal_sibling_fdr_null.py benchmarks/cloud/aws/README.md
git commit -m "Add AWS traversal sibling FDR diagnostic"
```

## Task 6: Local Smoke Runs

**Files:**
- Output only under `benchmarks/results/traversal_sibling_fdr_smoke/`

- [ ] **Step 1: Run algorithm-only smoke**

```bash
python -m benchmarks.validation.traversal_sibling_fdr_null \
  --layers synthetic_valid_p \
  --case-names synthetic_balanced_binary_tree \
  --replicates 200 \
  --alpha 0.01 \
  --base-seed 20260604 \
  --output-dir benchmarks/results/traversal_sibling_fdr_smoke
```

Expected:

- summary has `outcome=algorithmic_fdr_control`
- mean FDP near or below `0.01` with finite Monte Carlo noise

- [ ] **Step 2: Run binary fixed/selected smoke**

```bash
python -m benchmarks.validation.traversal_sibling_fdr_null \
  --layers fixed_tree_wald,selected_tree_wald,selected_tree_inflated \
  --case-names binary_2clusters \
  --replicates 20 \
  --alpha 0.01 \
  --base-seed 20260604 \
  --output-dir benchmarks/results/traversal_sibling_fdr_smoke
```

Expected:

- fixed-tree results are separated from selected-tree results
- selected-tree inflated rows record support failures instead of neutral decisions

- [ ] **Step 3: Commit smoke outputs only if they are small and cited**

If outputs are not durable evidence, leave them untracked. If they are durable evidence:

```bash
mkdir -p raw/assets/benchmark-results/traversal_sibling_fdr_smoke_20260604
cp benchmarks/results/traversal_sibling_fdr_smoke/*.csv raw/assets/benchmark-results/traversal_sibling_fdr_smoke_20260604/
git add raw/assets/benchmark-results/traversal_sibling_fdr_smoke_20260604
git commit -m "Record traversal sibling FDR smoke evidence"
```

## Task 7: AWS Focused Calibration Runs

**Files:**
- Output: `raw/assets/benchmark-results/traversal_sibling_fdr_focused_20260604/`
- Create: `wiki/sources/traversal-sibling-fdr-focused-20260604.md`

- [ ] **Step 1: Run binary focused study**

Run layers:

- `synthetic_valid_p`
- `fixed_tree_wald`
- `selected_tree_wald`
- `selected_tree_inflated`

Cases:

- `binary_2clusters`
- `binary_low_noise_4c`
- `overlap_heavy_4c_small_feat`

Replicates:

- at least `1000` for synthetic layer
- at least `300` per real null layer

- [ ] **Step 2: Run categorical focused study**

Cases:

- `cat_clear_3cat_4c`
- `cat_highcard_20cat_4c`

Replicates:

- at least `300` per layer

Interpretation rule:

- If `cat_highcard_20cat_4c` fails in `fixed_tree_wald`, classify it as finite-sample categorical calibration failure before blaming traversal FDR.

- [ ] **Step 3: Run continuous only after covariance null generator is validated**

Do not run continuous selected-tree FDR as production evidence until continuous null regeneration and covariance contracts are validated.

- [ ] **Step 4: Commit raw evidence and wiki source**

```bash
git add raw/assets/benchmark-results/traversal_sibling_fdr_focused_20260604 wiki/sources/traversal-sibling-fdr-focused-20260604.md wiki/index.md wiki/log.md wiki/questions/open-mathematical-questions.md
git commit -m "Record traversal sibling FDR calibration evidence"
```

## Task 8: Benchmark Reaction Runs

**Files:**
- Existing benchmark runners
- Output: `raw/assets/benchmark-results/fdr_calibration_benchmark_reaction_20260604/`

- [ ] **Step 1: Run small local benchmark**

```bash
python -m benchmarks.run_benchmark --suite binary --methods tbs --no-plots
```

Expected:

- no new hidden skips
- TBS rows record `edge_alpha` and `sibling_alpha`

- [ ] **Step 2: Run full benchmark only after calibration diagnostics complete**

Run the same benchmark command used for the last full suite. Save the exact command, commit hash, dirty status, and output path.

- [ ] **Step 3: Compare benchmark output against alpha-grid evidence**

Compare:

- mean ARI
- exact cluster-count rate
- skip count
- skip reasons
- over/under split counts
- stage timings

- [ ] **Step 4: Commit benchmark reaction evidence**

```bash
git add raw/assets/benchmark-results/fdr_calibration_benchmark_reaction_20260604 wiki/sources/fdr-calibration-benchmark-reaction-20260604.md wiki/index.md wiki/log.md
git commit -m "Record FDR calibration benchmark reaction"
```

## Stop Criteria

Do not change production FDR or alpha defaults unless all are true:

1. Synthetic valid-p layer confirms traversal-aligned sibling BH itself behaves as expected.
2. Fixed-tree sibling projected-Wald calibration is acceptable for binary cases.
3. Categorical fixed-tree calibration status is known separately.
4. Selected-tree layers quantify how much selection changes sibling FDR.
5. Empirical inflation support failures remain explicit, not converted to neutral p-values.
6. Full benchmark reaction does not hide failures by skip accounting.
7. Wiki and manuscript state exactly what is proved, simulated, benchmark-supported, or still open.

## Self-Review

- Spec coverage: covers FDR control, calibration layers, AWS execution, and benchmark reaction.
- Placeholder scan: no implementation step asks for unnamed error handling; unsupported continuous production evidence is explicitly blocked.
- Type consistency: uses `FdrLayer`, `TraversalSiblingFdrConfig`, `run_traversal_sibling_fdr_layer`, and output names consistently.
