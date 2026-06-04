# Selected Edge Type-I Geometry AWS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a strict validation and geometry-analysis pipeline that explains why edge alpha must be conservative, estimates selected-tree Type-I behavior, and identifies candidate selected-law variables before any production default change.

**Architecture:** Add one validation runner for selected-edge/null calibration, one geometry feature extractor for edge/sibling selected contexts, one AWS shard/merge wrapper, and one analysis summarizer. Keep production inference unchanged. All outputs are diagnostic evidence with explicit manifests, local smoke tests, AWS sharding, and wiki ingestion.

**Tech Stack:** Python 3.11+, NumPy, pandas, SciPy, statsmodels, scikit-learn, NetworkX, AWS Batch, S3, Docker, pytest, ruff, project wiki lint.

## Current Checkpoint

2026-06-04:

- Implemented the binary-null selected-edge validation runner.
- Implemented the AWS shard/merge wrapper.
- Implemented the selected-edge geometry analysis CLI.
- Added focused unit and CLI tests for the new runner, wrapper, and analysis.
- Ran local direct smoke and local shard/merge smoke on `binary_2clusters`.
- Confirmed the local smoke reproduces the selected-tree edge-selection effect:
  same-data selected-tree rows reject essentially all tested child-parent
  edges in the tiny null smoke, unlike the fixed-tree control.
- Ran Docker container shard/merge smoke.
- Ran AWS pilot array `0050e601-3a53-4554-8d93-713b4dea3dd1` and merge job
  `1c27b59e-7e3f-4f0c-ad20-2edd7064baca`.
- Synced merged AWS pilot outputs to
  `raw/assets/benchmark-results/selected-edge-type1-pilot-20260604/merged/`
  and added wiki source page
  `wiki/sources/selected-edge-type1-geometry-pilot-20260604.md`.
- Not launched yet: AWS full run across broader feature families.
- Strictness correction after review: full-space `edge_z_norm` and
  `edge_projection_energy_ratio` are not materialized by this runner and must
  remain `NaN`; only the projected norm is computed from the projected-Wald
  statistic.

---

## Assumptions And Non-Assumptions

- We stay on branch `dev`.
- We do not introduce cross-fitting as a production method.
- We do not change `DEFAULT_EDGE_ALPHA` or `DEFAULT_SIBLING_ALPHA` from this plan.
- We do not add a fallback calibration path.
- We treat current alpha-grid evidence as benchmark-performance evidence, not Type-I control evidence.
- We treat selected-tree inference as the mathematical problem:
  \[
  \mathcal L(T_{u\to c}^{\mathrm{edge}}
  \mid
  \text{tree selected},\ \text{ancestor path reached})
  \]
  and, jointly,
  \[
  \mathcal L(T_u^{\mathrm{sib}}
  \mid
  \text{tree selected},\ \text{edge path opened},\ u\text{ tested}).
  \]
- We include fixed-tree and fixed-subspace controls only as baselines. They cannot be promoted to production calibration unless they match the selected-tree behavior.
- We use existing dependencies first. New dependencies are not needed for the first implementation because SciPy, statsmodels, scikit-learn, and NetworkX already cover the required geometry, regression, and graph summaries.

## Interpretations To Keep Separate

1. **Local fixed-object Type-I:** fixed tree, fixed projection, fixed covariance. This is the clean proof baseline.
2. **Selected-edge Type-I:** same-data tree selected first, then child-parent edge tested. This is the main edge-alpha problem.
3. **Traversal Type-I:** edge decisions open or block later tests. This is why edge alpha behaves like a traversal opener.
4. **Sibling selected-tail law:** edge path and focal sibling context selected before sibling testing. This is the internal inflation/support problem.
5. **Benchmark ARI/power:** useful but not a Type-I proof. The alpha grid belongs here.

## Existing Hypothesis Check

The hypothesis to test is:

```text
The edge p-values are anti-conservative because the tree is selected from the
same data and then tested on that selected tree.
```

The existing source `wiki/sources/edge-selection-null-audit-20260601.md` already
supports this for pure Bernoulli null data. The raw evidence is in
`raw/assets/benchmark-results/edge_selection_null_audit_20260601/`.

Observed aggregate behavior at `edge_alpha = 0.001`:

```text
mode                    rows   tested   reject_rate   median_bh_p
fixed_tree_permutation  11670     102      0.000514   1.789456e-01
in_sample                3890    3488      0.894087   2.621178e-14
```

By scenario:

```text
null64x32   fixed_tree_permutation reject_rate=0.001058
null64x32   in_sample              reject_rate=0.882540
null128x64  fixed_tree_permutation reject_rate=0.001050
null128x64  in_sample              reject_rate=0.885039
null200x80  fixed_tree_permutation reject_rate=0.000000
null200x80  in_sample              reject_rate=0.903518
```

Interpretation:

- Fixed-tree/permuted null behavior is close to the nominal `0.001` edge level.
- Same-data selected-tree behavior rejects almost every reached child-parent
  edge under a global Bernoulli null.
- Therefore the old evidence strongly supports the selection hypothesis.
- The remaining work is not to check whether selection matters at all. It is to
  quantify the selected law across feature families, topology, projection
  dimensions, edge action, merge geometry, and traversal consequences.
- The existing audit is too narrow for manuscript-level claims because it has
  only three Bernoulli scenarios, five in-sample replicates, and three
  fixed-tree permutations per replicate. The AWS plan below turns this into a
  locked, broader validation study.

## File Structure

- Create `benchmarks/validation/selected_edge_type1_geometry.py`
  - Runs null simulations for fixed-tree, selected-tree, and selected-tree-plus-traversal modes.
  - Emits edge rows, sibling rows, final split rows, and run manifest.
- Create `benchmarks/cloud/aws_selected_edge_type1_geometry.py`
  - Shards the selected-edge validation runner across AWS Batch.
  - Merges shard outputs and validates complete shard coverage.
- Create `benchmarks/diagnostics/analysis/selected_edge_geometry_analysis.py`
  - Fits descriptive models for selected edge/sibling tail behavior.
  - Tests whether scalar alpha adjustment, edge action, merge margin, eigen concentration, angular capture, topology, or combined equations explain false openings.
- Create `tests/validation/65_test_selected_edge_type1_geometry.py`
  - Unit tests for simulation contracts and deterministic smoke runs.
- Create `tests/validation/66_test_aws_selected_edge_type1_geometry.py`
  - Unit tests for sharding, merge validation, duplicate detection, and missing shard failure.
- Create `tests/pipeline/64_test_selected_edge_geometry_analysis_cli.py`
  - Unit tests for analysis schema and model outputs.
- Modify `benchmarks/validation/README.md`
  - Document local and AWS commands.
- Modify `benchmarks/cloud/aws/README.md`
  - Add selected-edge Type-I AWS workflow.
- Create `wiki/sources/selected-edge-type1-geometry-YYYYMMDD.md` after the first real run.
- Update `wiki/questions/open-mathematical-questions.md` after real evidence exists.
- Append `wiki/log.md` after real evidence exists.

## Diagnostic Output Contracts

### Edge Row Contract

Each edge-level row must include:

```text
schema_version
run_id
mode
source_family
feature_representation
case_id
replicate
tree_seed
data_seed
node_id
parent_id
child_id
node_depth
n_parent
n_child
sample_ratio
edge_alpha
edge_raw_stat
edge_df
edge_raw_p
edge_bh_p
edge_tested
edge_rejected
ancestor_reached
selected_tree
fixed_tree
projection_dimension
mp_upper_edge
raw_mp_signal_count
effective_independent_rows
leading_eigenvalue
selected_eigenvalue_over_mp
effective_rank
spectral_entropy
edge_z_norm
edge_projected_norm
edge_projection_energy_ratio
edge_statistic_margin
edge_bh_action
merge_margin
merge_margin_rank
merge_persistence
first_order_signed_distance
null_whitened_signed_distance
tie_cell_status
tree_balance
subtree_leaf_count
descendant_leaf_count
path_length_from_root
```

### Sibling Row Contract

Each sibling-level row must include:

```text
schema_version
run_id
mode
source_family
feature_representation
case_id
replicate
tree_seed
data_seed
parent_id
node_depth
n_parent
n_left
n_right
child_balance
sibling_alpha
edge_alpha
left_edge_raw_p
right_edge_raw_p
left_edge_bh_p
right_edge_bh_p
edge_path_open
sibling_raw_stat
sibling_df
sibling_raw_p
inflation_factor
sibling_adjusted_p
sibling_bh_p
sibling_rejected
calibration_support_status
sibling_projection_dimension
edge_to_sibling_cosine
edge_to_sibling_cosine_squared
sibling_subspace_capture
principal_angle_min
principal_angle_max
selected_ratio
selected_ratio_log
```

### Final Split Row Contract

Each replicate-level final row must include:

```text
schema_version
run_id
mode
source_family
feature_representation
case_id
replicate
tree_seed
data_seed
edge_alpha
sibling_alpha
n_samples
n_features
true_null
found_clusters
false_split
max_depth_opened
n_edge_tested
n_edge_rejected
n_sibling_tested
n_sibling_rejected
n_calibration_support_failures
```

## Task 1: Add Deterministic Selected-Edge Simulation Config

**Files:**
- Create: `benchmarks/validation/selected_edge_type1_geometry.py`
- Test: `tests/validation/65_test_selected_edge_type1_geometry.py`

- [ ] **Step 1: Write config/dataclass tests**

Add this test:

```python
from pathlib import Path

from benchmarks.validation.selected_edge_type1_geometry import (
    SelectedEdgeGeometryConfig,
    build_run_id,
    parse_alpha_grid,
)


def test_parse_alpha_grid_rejects_invalid_values() -> None:
    assert parse_alpha_grid("0.0001,0.001") == (0.0001, 0.001)
    for raw in ("", "0", "1", "-0.1", "0.01,1.5"):
        try:
            parse_alpha_grid(raw)
        except ValueError:
            continue
        raise AssertionError(f"expected invalid grid to fail: {raw!r}")


def test_selected_edge_config_records_output_paths(tmp_path: Path) -> None:
    config = SelectedEdgeGeometryConfig(
        output_dir=tmp_path,
        suite="binary",
        case_names=("binary_2clusters",),
        modes=("selected_tree", "fixed_tree"),
        edge_alphas=(0.0001, 0.001),
        sibling_alpha=0.01,
        replicates=3,
        base_seed=20260604,
    )

    assert build_run_id(config).startswith("selected_edge_type1_geometry__")
    assert config.edge_rows_path.name == "selected_edge_geometry_edges.csv"
    assert config.sibling_rows_path.name == "selected_edge_geometry_siblings.csv"
    assert config.final_rows_path.name == "selected_edge_geometry_final.csv"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest -q tests/validation/65_test_selected_edge_type1_geometry.py
```

Expected: import failure because the new module does not exist.

- [ ] **Step 3: Implement config skeleton**

Create `benchmarks/validation/selected_edge_type1_geometry.py` with:

```python
#!/usr/bin/env python3
"""Selected-edge Type-I and geometry diagnostic.

This module generates validation evidence only. It does not change production
alpha defaults and does not add fallback calibration.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "selected_edge_type1_geometry/v1"
GENERATED_BY = "benchmarks.validation.selected_edge_type1_geometry"
DEFAULT_EDGE_ALPHA_GRID = (0.0001, 0.0003, 0.001, 0.003, 0.01)
DEFAULT_MODES = ("fixed_tree", "selected_tree", "selected_tree_traversal")


@dataclass(frozen=True)
class SelectedEdgeGeometryConfig:
    output_dir: Path
    suite: str
    case_names: tuple[str, ...]
    modes: tuple[str, ...]
    edge_alphas: tuple[float, ...]
    sibling_alpha: float
    replicates: int
    base_seed: int

    @property
    def edge_rows_path(self) -> Path:
        return self.output_dir / "selected_edge_geometry_edges.csv"

    @property
    def sibling_rows_path(self) -> Path:
        return self.output_dir / "selected_edge_geometry_siblings.csv"

    @property
    def final_rows_path(self) -> Path:
        return self.output_dir / "selected_edge_geometry_final.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "selected_edge_geometry_manifest.json"


def parse_alpha_grid(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("Alpha grid must contain at least one value.")
    invalid = [value for value in values if not 0.0 < value < 1.0]
    if invalid:
        raise ValueError(f"Alpha values must lie in (0, 1): {invalid!r}")
    return values


def parse_names(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def validate_modes(modes: Sequence[str]) -> tuple[str, ...]:
    allowed = set(DEFAULT_MODES)
    result = tuple(str(mode) for mode in modes)
    invalid = sorted(set(result) - allowed)
    if invalid:
        raise ValueError(f"Unknown selected-edge mode(s): {invalid!r}; allowed={sorted(allowed)!r}")
    if not result:
        raise ValueError("At least one selected-edge mode is required.")
    return result


def build_run_id(config: SelectedEdgeGeometryConfig) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    mode_id = "-".join(config.modes)
    return f"selected_edge_type1_geometry__{config.suite}__{mode_id}__{stamp}"


def write_manifest(config: SelectedEdgeGeometryConfig, *, run_id: str) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "suite": config.suite,
        "case_names": list(config.case_names),
        "modes": list(config.modes),
        "edge_alphas": list(config.edge_alphas),
        "sibling_alpha": float(config.sibling_alpha),
        "replicates": int(config.replicates),
        "base_seed": int(config.base_seed),
        "outputs": {
            "edges": str(config.edge_rows_path),
            "siblings": str(config.sibling_rows_path),
            "final": str(config.final_rows_path),
        },
        "note": (
            "Diagnostic selected-edge Type-I and geometry evidence only. "
            "Does not change production defaults or add a fallback calibration law."
        ),
    }
    config.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", default="binary")
    parser.add_argument("--case-names")
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--edge-alphas", default=",".join(str(value) for value in DEFAULT_EDGE_ALPHA_GRID))
    parser.add_argument("--sibling-alpha", type=float, default=0.01)
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=20260604)
    args = parser.parse_args(argv)

    if args.run != "run":
        raise ValueError("Only the 'run' command is supported.")
    if args.replicates <= 0:
        raise ValueError("replicates must be positive.")
    if not 0.0 < float(args.sibling_alpha) < 1.0:
        raise ValueError("sibling_alpha must lie in (0, 1).")

    config = SelectedEdgeGeometryConfig(
        output_dir=args.output_dir,
        suite=str(args.suite),
        case_names=parse_names(args.case_names),
        modes=validate_modes(parse_names(args.modes)),
        edge_alphas=parse_alpha_grid(str(args.edge_alphas)),
        sibling_alpha=float(args.sibling_alpha),
        replicates=int(args.replicates),
        base_seed=int(args.base_seed),
    )
    run_id = build_run_id(config)
    write_manifest(config, run_id=run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
pytest -q tests/validation/65_test_selected_edge_type1_geometry.py
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/validation/selected_edge_type1_geometry.py tests/validation/65_test_selected_edge_type1_geometry.py
git commit -m "validation: scaffold selected-edge geometry diagnostic"
```

## Task 2: Implement Null Case Regeneration Without Production Changes

**Files:**
- Modify: `benchmarks/validation/selected_edge_type1_geometry.py`
- Test: `tests/validation/65_test_selected_edge_type1_geometry.py`

- [ ] **Step 1: Add a deterministic null-regeneration test**

Append:

```python
from benchmarks.validation.selected_edge_type1_geometry import regenerate_null_case


def test_regenerate_null_case_preserves_shape_and_contract() -> None:
    data, metadata = regenerate_null_case(
        case_id="binary_2clusters",
        source_family="binary_template",
        feature_representation="binary",
        n_samples=12,
        n_features=5,
        seed=7,
    )

    assert data.shape == (12, 5)
    assert metadata["true_null"] is True
    assert metadata["source_family"] == "binary_template"
    assert metadata["feature_representation"] == "binary"
    assert set(data.to_numpy().ravel()).issubset({0, 1})
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest -q tests/validation/65_test_selected_edge_type1_geometry.py::test_regenerate_null_case_preserves_shape_and_contract
```

Expected: import failure for `regenerate_null_case`.

- [ ] **Step 3: Add null-regeneration helper**

Implement only binary first:

```python
import numpy as np
import pandas as pd


def regenerate_null_case(
    *,
    case_id: str,
    source_family: str,
    feature_representation: str,
    n_samples: int,
    n_features: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if source_family != "binary_template" or feature_representation != "binary":
        raise ValueError(
            "Initial selected-edge Type-I diagnostic supports only binary_template/binary "
            f"null regeneration; got {source_family!r}/{feature_representation!r} for {case_id!r}."
        )
    rng = np.random.default_rng(int(seed))
    matrix = rng.binomial(1, 0.5, size=(int(n_samples), int(n_features))).astype(int)
    data = pd.DataFrame(
        matrix,
        index=[f"S{i}" for i in range(int(n_samples))],
        columns=[f"F{j}" for j in range(int(n_features))],
    )
    metadata = {
        "case_id": str(case_id),
        "source_family": str(source_family),
        "feature_representation": str(feature_representation),
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "true_null": True,
        "seed": int(seed),
    }
    return data, metadata
```

- [ ] **Step 4: Run test**

Run:

```bash
pytest -q tests/validation/65_test_selected_edge_type1_geometry.py
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/validation/selected_edge_type1_geometry.py tests/validation/65_test_selected_edge_type1_geometry.py
git commit -m "validation: add strict binary null regeneration"
```

## Task 3: Extract Selected-Tree Edge Rows

**Files:**
- Modify: `benchmarks/validation/selected_edge_type1_geometry.py`
- Test: `tests/validation/65_test_selected_edge_type1_geometry.py`

- [ ] **Step 1: Add smoke test for selected-tree edge rows**

Append:

```python
from benchmarks.validation.selected_edge_type1_geometry import run_selected_edge_replicate


def test_run_selected_edge_replicate_emits_edge_rows() -> None:
    edge_rows, sibling_rows, final_rows = run_selected_edge_replicate(
        case_id="binary_2clusters",
        source_family="binary_template",
        feature_representation="binary",
        n_samples=12,
        n_features=5,
        replicate=0,
        data_seed=11,
        tree_seed=11,
        mode="selected_tree",
        edge_alpha=0.001,
        sibling_alpha=0.01,
        run_id="smoke",
    )

    assert edge_rows
    assert final_rows
    assert all(row["mode"] == "selected_tree" for row in edge_rows)
    assert all("edge_raw_p" in row for row in edge_rows)
    assert all("edge_rejected" in row for row in edge_rows)
    assert sibling_rows is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest -q tests/validation/65_test_selected_edge_type1_geometry.py::test_run_selected_edge_replicate_emits_edge_rows
```

Expected: import failure for `run_selected_edge_replicate`.

- [ ] **Step 3: Implement selected-tree extraction**

Use existing production APIs without adding fallbacks:

```python
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from benchmarks.shared.kl_tree_context import build_kl_tree_context


def run_selected_edge_replicate(
    *,
    case_id: str,
    source_family: str,
    feature_representation: str,
    n_samples: int,
    n_features: int,
    replicate: int,
    data_seed: int,
    tree_seed: int,
    mode: str,
    edge_alpha: float,
    sibling_alpha: float,
    run_id: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    if mode != "selected_tree":
        raise ValueError(f"Task 3 implements only selected_tree mode; got {mode!r}.")
    data, _metadata = regenerate_null_case(
        case_id=case_id,
        source_family=source_family,
        feature_representation=feature_representation,
        n_samples=n_samples,
        n_features=n_features,
        seed=data_seed,
    )
    distances = pdist(data.to_numpy(dtype=float), metric="hamming")
    linkage_matrix = linkage(distances, method="average")
    context = build_kl_tree_context(
        case_id=case_id,
        data=data,
        distance_condensed=distances,
        linkage_matrix=linkage_matrix,
    )
    tree = context.tree
    tree.populate_node_divergences(data)
    result = tree.decompose(
        annotations_df=tree.annotations_df,
        leaf_data=data,
        edge_alpha=float(edge_alpha),
        sibling_alpha=float(sibling_alpha),
    )
    annotations = tree.annotations_df
    if annotations is None:
        raise ValueError("Expected populated annotations after tree decomposition.")

    edge_rows = extract_edge_geometry_rows(
        tree=tree,
        annotations_df=annotations,
        run_id=run_id,
        mode=mode,
        case_id=case_id,
        source_family=source_family,
        feature_representation=feature_representation,
        replicate=replicate,
        tree_seed=tree_seed,
        data_seed=data_seed,
        edge_alpha=edge_alpha,
    )
    sibling_rows = extract_sibling_geometry_rows(
        tree=tree,
        annotations_df=annotations,
        run_id=run_id,
        mode=mode,
        case_id=case_id,
        source_family=source_family,
        feature_representation=feature_representation,
        replicate=replicate,
        tree_seed=tree_seed,
        data_seed=data_seed,
        edge_alpha=edge_alpha,
        sibling_alpha=sibling_alpha,
    )
    labels = result["labels"]
    final_rows = [
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "mode": mode,
            "source_family": source_family,
            "feature_representation": feature_representation,
            "case_id": case_id,
            "replicate": int(replicate),
            "tree_seed": int(tree_seed),
            "data_seed": int(data_seed),
            "edge_alpha": float(edge_alpha),
            "sibling_alpha": float(sibling_alpha),
            "n_samples": int(n_samples),
            "n_features": int(n_features),
            "true_null": True,
            "found_clusters": int(len(set(labels))),
            "false_split": bool(len(set(labels)) > 1),
            "max_depth_opened": int(max((row["node_depth"] for row in edge_rows if row["edge_rejected"]), default=0)),
            "n_edge_tested": int(sum(bool(row["edge_tested"]) for row in edge_rows)),
            "n_edge_rejected": int(sum(bool(row["edge_rejected"]) for row in edge_rows)),
            "n_sibling_tested": int(sum(np.isfinite(float(row["sibling_raw_p"])) for row in sibling_rows)),
            "n_sibling_rejected": int(sum(bool(row["sibling_rejected"]) for row in sibling_rows)),
            "n_calibration_support_failures": int(
                sum(row["calibration_support_status"] == "unsupported" for row in sibling_rows)
            ),
        }
    ]
    return edge_rows, sibling_rows, final_rows
```

Also add strict extractor functions. If a required annotation column is missing, raise `ValueError`; do not return empty rows.

- [ ] **Step 4: Run test**

Run:

```bash
pytest -q tests/validation/65_test_selected_edge_type1_geometry.py
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/validation/selected_edge_type1_geometry.py tests/validation/65_test_selected_edge_type1_geometry.py
git commit -m "validation: emit selected-tree edge type-I rows"
```

## Task 4: Add Geometry Variables

**Files:**
- Modify: `benchmarks/validation/selected_edge_type1_geometry.py`
- Test: `tests/validation/65_test_selected_edge_type1_geometry.py`

- [ ] **Step 1: Add geometry schema test**

Append:

```python
def test_edge_rows_include_geometry_variables() -> None:
    edge_rows, _sibling_rows, _final_rows = run_selected_edge_replicate(
        case_id="binary_2clusters",
        source_family="binary_template",
        feature_representation="binary",
        n_samples=12,
        n_features=5,
        replicate=0,
        data_seed=13,
        tree_seed=13,
        mode="selected_tree",
        edge_alpha=0.001,
        sibling_alpha=0.01,
        run_id="geometry",
    )
    required = {
        "sample_ratio",
        "edge_statistic_margin",
        "edge_bh_action",
        "tree_balance",
        "path_length_from_root",
    }
    for row in edge_rows:
        assert required.issubset(row)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest -q tests/validation/65_test_selected_edge_type1_geometry.py::test_edge_rows_include_geometry_variables
```

Expected: failure for missing variables.

- [ ] **Step 3: Implement geometry variables**

Use exact formulas:

```python
from scipy.stats import chi2


def edge_statistic_margin(statistic: float, degrees_of_freedom: float, edge_alpha: float) -> float:
    if degrees_of_freedom <= 0:
        raise ValueError(f"degrees_of_freedom must be positive; got {degrees_of_freedom!r}.")
    return float(np.sqrt(statistic) - np.sqrt(chi2.isf(float(edge_alpha), int(degrees_of_freedom))))


def negative_log10_action(p_value: float) -> float:
    if not np.isfinite(p_value) or p_value <= 0.0 or p_value > 1.0:
        raise ValueError(f"p-value must be finite in (0, 1]; got {p_value!r}.")
    return float(-np.log10(p_value))


def node_depth_map(tree) -> dict[str, int]:
    roots = [node for node in tree.nodes if tree.in_degree(node) == 0]
    if len(roots) != 1:
        raise ValueError(f"Expected exactly one root; got {roots!r}.")
    root = roots[0]
    lengths = nx.single_source_shortest_path_length(tree, root)
    return {str(node): int(depth) for node, depth in lengths.items()}
```

For angles and trigonometric variables, use:

```python
def cosine_squared(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float).ravel()
    right = np.asarray(right, dtype=float).ravel()
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        raise ValueError("cosine_squared requires non-zero vectors.")
    cosine = float(np.dot(left, right) / (left_norm * right_norm))
    return float(np.clip(cosine * cosine, 0.0, 1.0))
```

- [ ] **Step 4: Run test**

Run:

```bash
pytest -q tests/validation/65_test_selected_edge_type1_geometry.py
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/validation/selected_edge_type1_geometry.py tests/validation/65_test_selected_edge_type1_geometry.py
git commit -m "validation: add selected-edge geometry covariates"
```

## Task 5: Implement Fixed-Tree Control Mode

**Files:**
- Modify: `benchmarks/validation/selected_edge_type1_geometry.py`
- Test: `tests/validation/65_test_selected_edge_type1_geometry.py`

- [ ] **Step 1: Add fixed-tree contrast test**

Append:

```python
def test_fixed_tree_and_selected_tree_modes_are_distinct() -> None:
    selected_edges, _selected_siblings, selected_final = run_selected_edge_replicate(
        case_id="binary_2clusters",
        source_family="binary_template",
        feature_representation="binary",
        n_samples=12,
        n_features=5,
        replicate=0,
        data_seed=17,
        tree_seed=17,
        mode="selected_tree",
        edge_alpha=0.001,
        sibling_alpha=0.01,
        run_id="selected",
    )
    fixed_edges, _fixed_siblings, fixed_final = run_selected_edge_replicate(
        case_id="binary_2clusters",
        source_family="binary_template",
        feature_representation="binary",
        n_samples=12,
        n_features=5,
        replicate=0,
        data_seed=19,
        tree_seed=17,
        mode="fixed_tree",
        edge_alpha=0.001,
        sibling_alpha=0.01,
        run_id="fixed",
    )

    assert selected_edges
    assert fixed_edges
    assert selected_final[0]["mode"] == "selected_tree"
    assert fixed_final[0]["mode"] == "fixed_tree"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest -q tests/validation/65_test_selected_edge_type1_geometry.py::test_fixed_tree_and_selected_tree_modes_are_distinct
```

Expected: `fixed_tree` unsupported.

- [ ] **Step 3: Implement fixed-tree mode**

Build tree on `tree_seed` null data and test on independent `data_seed` null data using the same leaf labels. Keep this as a diagnostic baseline, not production cross-fitting.

- [ ] **Step 4: Run tests**

Run:

```bash
pytest -q tests/validation/65_test_selected_edge_type1_geometry.py
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/validation/selected_edge_type1_geometry.py tests/validation/65_test_selected_edge_type1_geometry.py
git commit -m "validation: add fixed-tree edge type-I baseline"
```

## Task 6: Add Analysis CLI For Candidate Laws

**Files:**
- Create: `benchmarks/diagnostics/analysis/selected_edge_geometry_analysis.py`
- Test: `tests/pipeline/64_test_selected_edge_geometry_analysis_cli.py`

- [ ] **Step 1: Add analysis test**

Create:

```python
from pathlib import Path

import pandas as pd

from benchmarks.diagnostics.analysis.selected_edge_geometry_analysis import analyze_selected_edge_geometry


def test_analyze_selected_edge_geometry_writes_ranked_models(tmp_path: Path) -> None:
    edge_rows = pd.DataFrame(
        {
            "mode": ["selected_tree", "selected_tree", "fixed_tree", "fixed_tree"],
            "edge_rejected": [1, 0, 0, 0],
            "edge_raw_p": [0.0001, 0.2, 0.4, 0.8],
            "edge_bh_action": [4.0, 0.7, 0.4, 0.1],
            "edge_statistic_margin": [3.0, -0.5, -1.0, -2.0],
            "selected_eigenvalue_over_mp": [2.0, 1.1, 0.9, 0.8],
            "tree_balance": [0.5, 0.6, 0.5, 0.7],
            "sample_ratio": [0.5, 0.5, 0.5, 0.5],
            "path_length_from_root": [1, 1, 1, 1],
        }
    )
    edge_path = tmp_path / "edges.csv"
    edge_rows.to_csv(edge_path, index=False)
    output_path = tmp_path / "analysis.csv"

    result = analyze_selected_edge_geometry(edge_path=edge_path, output_path=output_path)

    assert output_path.exists()
    assert "model_name" in result.columns
    assert "score" in result.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest -q tests/pipeline/64_test_selected_edge_geometry_analysis_cli.py
```

Expected: import failure.

- [ ] **Step 3: Implement analysis**

Use installed libraries only:

- `statsmodels.api.Logit` for interpretable logistic false-opening model.
- `sklearn.ensemble.GradientBoostingClassifier` for nonlinear benchmark.
- `sklearn.inspection.permutation_importance` for variable importance.
- `scipy.stats.spearmanr` for monotone relationship table.
- `sklearn.metrics.roc_auc_score`, `average_precision_score`, and Brier score.

Candidate feature groups:

```python
LINEAR_EDGE_ACTION = ("edge_bh_action",)
GEOMETRY = ("edge_statistic_margin", "sample_ratio", "tree_balance", "path_length_from_root")
SPECTRAL = ("selected_eigenvalue_over_mp",)
COMBINED = LINEAR_EDGE_ACTION + GEOMETRY + SPECTRAL
```

Do not choose a production equation. Rank descriptive candidates and mark the output role as `diagnostic_candidate_law_search`.

- [ ] **Step 4: Run test**

Run:

```bash
pytest -q tests/pipeline/64_test_selected_edge_geometry_analysis_cli.py
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/diagnostics/analysis/selected_edge_geometry_analysis.py tests/pipeline/64_test_selected_edge_geometry_analysis_cli.py
git commit -m "diagnostics: rank selected-edge geometry candidate laws"
```

## Task 7: Add AWS Wrapper

**Files:**
- Create: `benchmarks/cloud/aws_selected_edge_type1_geometry.py`
- Test: `tests/validation/66_test_aws_selected_edge_type1_geometry.py`
- Modify: `benchmarks/cloud/aws/README.md`

- [ ] **Step 1: Add shard/merge tests**

Mirror the contract style in `tests/validation/64_test_aws_alpha_grid_search.py`:

```python
from pathlib import Path

import pandas as pd

from benchmarks.cloud.aws_selected_edge_type1_geometry import (
    AwsSelectedEdgeGeometryConfig,
    make_shard_spec,
    merge_shards,
)


def test_selected_edge_shards_distribute_replicates() -> None:
    config = AwsSelectedEdgeGeometryConfig(
        output_dir=Path("/tmp/out"),
        suite="binary",
        case_names=("binary_2clusters",),
        modes=("selected_tree",),
        edge_alphas=(0.001,),
        sibling_alpha=0.01,
        replicates=10,
        base_seed=20260604,
        shard_count=3,
        s3_uri=None,
        resume=True,
    )

    assert make_shard_spec(config, 0).replicate_indices == (0, 3, 6, 9)
    assert make_shard_spec(config, 1).replicate_indices == (1, 4, 7)
    assert make_shard_spec(config, 2).replicate_indices == (2, 5, 8)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest -q tests/validation/66_test_aws_selected_edge_type1_geometry.py
```

Expected: import failure.

- [ ] **Step 3: Implement AWS wrapper**

Follow `benchmarks/cloud/aws_alpha_grid_search.py` structure:

- `run-shard` selects replicate indices by `index % shard_count`.
- `merge` requires every shard manifest and every expected replicate index.
- S3 sync is explicit.
- Missing shard raises `FileNotFoundError`.
- Duplicate replicate raises `ValueError`.

- [ ] **Step 4: Run AWS wrapper tests**

Run:

```bash
pytest -q tests/validation/66_test_aws_selected_edge_type1_geometry.py
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/cloud/aws_selected_edge_type1_geometry.py tests/validation/66_test_aws_selected_edge_type1_geometry.py benchmarks/cloud/aws/README.md
git commit -m "cloud: shard selected-edge type-I geometry diagnostic"
```

## Task 8: Local Smoke Run

**Files:**
- Modify: `benchmarks/validation/README.md`

- [ ] **Step 1: Add smoke command to README**

Add:

```bash
python -m benchmarks.validation.selected_edge_type1_geometry run \
  --suite binary \
  --case-names binary_2clusters \
  --modes fixed_tree,selected_tree \
  --edge-alphas 0.0001,0.001 \
  --sibling-alpha 0.01 \
  --replicates 5 \
  --base-seed 20260604 \
  --output-dir benchmarks/results/selected_edge_type1_geometry_smoke
```

- [ ] **Step 2: Run local smoke**

Run the command above.

Expected outputs:

```text
benchmarks/results/selected_edge_type1_geometry_smoke/selected_edge_geometry_edges.csv
benchmarks/results/selected_edge_type1_geometry_smoke/selected_edge_geometry_siblings.csv
benchmarks/results/selected_edge_type1_geometry_smoke/selected_edge_geometry_final.csv
benchmarks/results/selected_edge_type1_geometry_smoke/selected_edge_geometry_manifest.json
```

- [ ] **Step 3: Run analysis on smoke**

Run:

```bash
python -m benchmarks.diagnostics.analysis.selected_edge_geometry_analysis \
  --edge-rows benchmarks/results/selected_edge_type1_geometry_smoke/selected_edge_geometry_edges.csv \
  --output benchmarks/results/selected_edge_type1_geometry_smoke/selected_edge_geometry_models.csv
```

Expected: writes model ranking CSV.

- [ ] **Step 4: Run focused tests**

Run:

```bash
ruff check benchmarks/validation/selected_edge_type1_geometry.py benchmarks/cloud/aws_selected_edge_type1_geometry.py benchmarks/diagnostics/analysis/selected_edge_geometry_analysis.py tests/validation/65_test_selected_edge_type1_geometry.py tests/validation/66_test_aws_selected_edge_type1_geometry.py tests/pipeline/64_test_selected_edge_geometry_analysis_cli.py
pytest -q tests/validation/65_test_selected_edge_type1_geometry.py tests/validation/66_test_aws_selected_edge_type1_geometry.py tests/pipeline/64_test_selected_edge_geometry_analysis_cli.py
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/validation/README.md
git commit -m "docs: document selected-edge type-I smoke workflow"
```

## Task 9: AWS Pilot Run

**Files:**
- No source changes unless pilot reveals a contract bug.

- [ ] **Step 1: Build image locally**

Run:

```bash
docker build -f benchmarks/cloud/aws/Dockerfile -t kl-te-benchmark-diagnostics:local .
```

Expected: build succeeds.

- [ ] **Step 2: Run container smoke**

Run:

```bash
docker run --rm kl-te-benchmark-diagnostics:local \
  benchmarks.cloud.aws_selected_edge_type1_geometry run-shard \
  --suite binary \
  --case-names binary_2clusters \
  --modes fixed_tree,selected_tree \
  --edge-alphas 0.0001,0.001 \
  --sibling-alpha 0.01 \
  --replicates 6 \
  --base-seed 20260604 \
  --output-dir /tmp/selected-edge-type1-pilot \
  --shard-count 2 \
  --shard-index 0
```

Expected: writes shard outputs under `/tmp/selected-edge-type1-pilot/shards/shard_0000`.

- [ ] **Step 3: Push image and submit pilot AWS array**

Use the existing ECR repository and Batch stack. Submit a 4-shard pilot:

```bash
aws batch submit-job \
  --job-name kl-te-selected-edge-type1-pilot \
  --job-queue kl-te-benchmark-diagnostics \
  --job-definition kl-te-benchmark-diagnostics \
  --array-properties size=4 \
  --container-overrides '{
    "command": [
      "benchmarks.cloud.aws_selected_edge_type1_geometry",
      "run-shard",
      "--suite", "binary",
      "--case-names", "binary_2clusters,binary_low_noise_4c",
      "--modes", "fixed_tree,selected_tree",
      "--edge-alphas", "0.0001,0.001",
      "--sibling-alpha", "0.01",
      "--replicates", "40",
      "--base-seed", "20260604",
      "--output-dir", "/tmp/selected-edge-type1-pilot",
      "--shard-count", "4",
      "--s3-uri", "s3://kl-te-benchmark-diagnostics-067744548702-us-east-1/selected-edge-type1-pilot-20260604"
    ]
  }'
```

- [ ] **Step 4: Submit merge job**

Run:

```bash
aws batch submit-job \
  --job-name kl-te-selected-edge-type1-pilot-merge \
  --job-queue kl-te-benchmark-diagnostics \
  --job-definition kl-te-benchmark-diagnostics \
  --container-overrides '{
    "command": [
      "benchmarks.cloud.aws_selected_edge_type1_geometry",
      "merge",
      "--suite", "binary",
      "--case-names", "binary_2clusters,binary_low_noise_4c",
      "--modes", "fixed_tree,selected_tree",
      "--edge-alphas", "0.0001,0.001",
      "--sibling-alpha", "0.01",
      "--replicates", "40",
      "--base-seed", "20260604",
      "--output-dir", "/tmp/selected-edge-type1-pilot",
      "--shard-count", "4",
      "--s3-uri", "s3://kl-te-benchmark-diagnostics-067744548702-us-east-1/selected-edge-type1-pilot-20260604"
    ]
  }'
```

- [ ] **Step 5: Pull merged outputs locally**

Run:

```bash
aws s3 sync \
  s3://kl-te-benchmark-diagnostics-067744548702-us-east-1/selected-edge-type1-pilot-20260604/merged \
  raw/assets/benchmark-results/selected_edge_type1_pilot_20260604
```

Expected: merged edge, sibling, final, manifest, and analysis CSVs are present.

## Task 10: AWS Full Run

**Files:**
- No source changes unless pilot output shows a contract bug.

- [ ] **Step 1: Choose full design**

Use this design:

```text
suites: binary,categorical,continuous
case_names: binary_2clusters,binary_low_noise_4c,binary_many_features,cat_highcard_20cat_4c,cat_overlap_3cat_4c,gauss_moderate_3c_continuous,dim_consolidated_4c_24f_continuous
modes: fixed_tree,selected_tree,selected_tree_traversal
edge_alphas: 0.0001,0.0003,0.001,0.003,0.01
sibling_alpha: 0.01
replicates: 1000
shards: 50
```

This is intentionally not the full benchmark suite. It targets the edge-alpha/type-I question with null and near-null stress representatives.

- [ ] **Step 2: Estimate cost before launch**

Use the previous alpha-grid observed cost as the baseline. If pilot shard duration is `d` seconds for `r` replicate-units, estimate:

```text
full_task_hours = 50 * pilot_duration_seconds * (1000 / 40) / 3600
fargate_cost = full_task_hours * 0.23304
```

Do not launch the full run if the estimate exceeds the user-approved budget.

- [ ] **Step 3: Submit full AWS array**

Submit with `--array-properties size=50` and S3 prefix:

```text
s3://kl-te-benchmark-diagnostics-067744548702-us-east-1/selected-edge-type1-full-20260604
```

- [ ] **Step 4: Merge full outputs**

Submit merge job after all shards succeed.

- [ ] **Step 5: Pull merged outputs**

Sync to:

```text
raw/assets/benchmark-results/selected_edge_type1_full_20260604
```

- [ ] **Step 6: Run local analysis**

Run:

```bash
python -m benchmarks.diagnostics.analysis.selected_edge_geometry_analysis \
  --edge-rows raw/assets/benchmark-results/selected_edge_type1_full_20260604/selected_edge_geometry_edges.csv \
  --output raw/assets/benchmark-results/selected_edge_type1_full_20260604/selected_edge_geometry_models.csv
```

Expected: ranked model table with fixed-tree vs selected-tree separation.

## Task 11: Mathematical Interpretation Criteria

**Files:**
- Create after full run: `wiki/sources/selected-edge-type1-geometry-20260604.md`
- Modify after full run: `wiki/questions/open-mathematical-questions.md`
- Modify after full run: `wiki/log.md`

- [ ] **Step 1: Decide using predeclared criteria**

Use these criteria:

```text
If fixed_tree p-values are uniform but selected_tree p-values are not:
  tree selection is the dominant Type-I distortion.

If edge_bh_action alone predicts false openings:
  alpha is compensating for selected-edge action.

If edge_bh_action + spectral variables improves held-out tail prediction:
  selected spectral law is needed.

If topology/balance variables improve held-out prediction:
  edge alpha cannot be one global scalar without context or conditioning.

If selected_tree_traversal inflates false splits relative to selected_tree edge openings:
  traversal, not only edge testing, is part of the Type-I target.

If no compact variables predict held-out false openings:
  current geometry variables are incomplete; do not change method.
```

- [ ] **Step 2: Write wiki source page**

Create a source page with:

```text
Summary
Key Points
Evidence
Links
```

Include:

- run manifest path;
- rows and replicate counts;
- fixed-tree rejection rates;
- selected-tree rejection rates;
- final false-split rates;
- best descriptive variables;
- explicit statement that this is diagnostic evidence only.

- [ ] **Step 3: Run wiki lint**

Run:

```bash
make wiki-lint
```

Expected: pass.

- [ ] **Step 4: Commit evidence and wiki**

```bash
git add raw/assets/benchmark-results/selected_edge_type1_full_20260604 wiki/sources/selected-edge-type1-geometry-20260604.md wiki/questions/open-mathematical-questions.md wiki/log.md
git commit -m "wiki: record selected-edge type-I geometry evidence"
```

## Task 12: Verification Before Completion

**Files:**
- All modified files.

- [ ] **Step 1: Run focused tests**

```bash
pytest -q \
  tests/validation/65_test_selected_edge_type1_geometry.py \
  tests/validation/66_test_aws_selected_edge_type1_geometry.py \
  tests/pipeline/64_test_selected_edge_geometry_analysis_cli.py
```

- [ ] **Step 2: Run related existing tests**

```bash
pytest -q \
  tests/statistics/41_test_multiple_testing_contracts.py \
  tests/statistics/43_test_traversal_aligned_sibling_fdr.py \
  tests/statistics/39_test_projected_wald_statistics.py \
  tests/validation/51_test_selected_pca_projected_wald_calibration.py \
  tests/validation/64_test_aws_alpha_grid_search.py
```

- [ ] **Step 3: Run lint/checks**

```bash
ruff check benchmarks/validation benchmarks/cloud benchmarks/diagnostics tests/validation tests/pipeline
make wiki-lint
git diff --check
```

- [ ] **Step 4: Report final state**

Report:

```text
implemented files
tests run
AWS jobs run
cost estimate and actual cost
fixed-tree Type-I result
selected-tree Type-I result
selected-tree traversal false-split result
best descriptive geometry variables
whether any production change is justified
remaining mathematical gaps
```

## Libraries And Why They Are Enough

- `numpy`: vectorized simulation, norms, dot products, cosine variables.
- `pandas`: row-level contracts, merged CSVs, group summaries.
- `scipy.stats`: chi-square margins, KS tests, binomial intervals, Spearman correlations.
- `scipy.linalg`: subspace angles and spectral matrix operations.
- `statsmodels`: interpretable logistic and quantile regressions.
- `scikit-learn`: nonlinear descriptive models, held-out scoring, permutation importance.
- `networkx`: tree topology, root paths, depths, balance, descendant counts.
- `numba`: only if profiling shows row extraction is slow; do not add first.

Do not add `shap`, `gudhi`, `ripser`, `jax`, PyTorch, or CuPy in the first pass. They add dependency weight before we know the current geometry variables are insufficient.

## Self-Review

- Spec coverage: the plan covers strict assumptions, no hidden fallback, local smoke, AWS pilot, AWS full run, geometry variables, candidate-law analysis, wiki evidence, and verification.
- Placeholder scan: the plan does not use open-ended placeholders. The only date-containing page name is fixed to `20260604` because the current date is 2026-06-04.
- Type consistency: all planned contracts use stable names: `selected_edge_geometry_edges.csv`, `selected_edge_geometry_siblings.csv`, `selected_edge_geometry_final.csv`, and `selected_edge_geometry_manifest.json`.
- Main tradeoff surfaced: this plan can identify selected-law variables and Type-I distortion sources, but it does not itself prove Type-I control. A proof still requires either conditioning on the selected-tree event or proving a valid selected p-value transformation.
