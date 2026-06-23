"""Regression tests for audit-env handling in run_single_case."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from benchmarks.shared.util import case_run
from benchmarks.shared.util.case_inputs import PreparedCaseInputs


@pytest.mark.parametrize("initial_value", [None, "/tmp/original-audit-root"])
def test_run_single_case_restores_matrix_audit_env(monkeypatch, initial_value):
    env_name = "TBS_MATRIX_AUDIT_ROOT"
    if initial_value is None:
        monkeypatch.delenv(env_name, raising=False)
    else:
        monkeypatch.setenv(env_name, initial_value)

    def _fake_prepare_case_inputs(_tc, _selected_methods):
        data_t = pd.DataFrame(
            [[0, 1], [1, 0]],
            index=["S0", "S1"],
            columns=["F0", "F1"],
        )
        y_t = np.array([0, 1], dtype=int)
        x_original = np.array([[0, 1], [1, 0]], dtype=float)
        meta = {
            "n_clusters": 2,
            "n_samples": 2,
            "n_features": 2,
            "noise": 0.0,
            "category": "test",
        }
        distance_condensed = np.array([0.5], dtype=float)
        distance_matrix = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=float)
        return PreparedCaseInputs(
            data=data_t,
            labels=y_t,
            original_features=x_original,
            metadata=meta,
            distance_condensed=distance_condensed,
            distance_matrix=distance_matrix,
        )

    def _fake_run_single_method_once(**_kwargs):
        return object(), None, None

    monkeypatch.setattr(case_run, "prepare_case_inputs", _fake_prepare_case_inputs)
    monkeypatch.setattr(case_run, "run_single_method_once", _fake_run_single_method_once)
    monkeypatch.setattr(case_run, "export_case_and_method_matrix_audits", lambda **_kwargs: None)
    monkeypatch.setattr(case_run, "export_decomposition_audit", lambda **_kwargs: None)

    selected_methods = ["tbs"]
    param_sets = {"tbs": [{}]}
    case_run.run_single_case(
        tc={"name": "env_restore_case", "seed": 7, "test_case_num": 1},
        total_cases=1,
        selected_methods=selected_methods,
        param_sets=param_sets,
        significance_level=0.05,
        edge_alpha=0.001,
        output_pdf=None,
        plots_root=Path("benchmarks/results/plots"),
        matrix_audit=True,
        verbose=False,
    )

    if initial_value is None:
        assert env_name not in os.environ
    else:
        assert os.environ.get(env_name) == initial_value
