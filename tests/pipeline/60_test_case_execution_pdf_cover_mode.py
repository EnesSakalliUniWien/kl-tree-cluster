from __future__ import annotations

import pandas as pd

from benchmarks.shared.util import case_execution


def test_run_case_with_optional_isolation_disables_cover_pages(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_benchmark_fn(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame(), None

    monkeypatch.setattr(case_execution, "_get_benchmark_fn", lambda: _fake_benchmark_fn)

    case_execution.run_case_with_optional_isolation(
        case={"name": "tiny_case"},
        case_id="tiny_case",
        methods_to_test=["kl"],
        case_plot_umap=False,
        case_plot_manifold=False,
        enable_plots=True,
        pdf_path="/tmp/tiny_case.pdf",
        isolate_umap_cases=True,
        timeout_sec=1,
        retry_count=0,
    )

    assert captured.get("include_cover_pages") is False


def test_run_case_worker_disables_cover_pages(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_benchmark_fn(**kwargs):
        captured.update(kwargs)
        df = pd.DataFrame([{"method": "kl", "ari": 1.0}])
        return df, None

    monkeypatch.setattr(case_execution, "_get_benchmark_fn", lambda: _fake_benchmark_fn)

    class _Queue:
        def __init__(self):
            self.payloads: list[dict[str, object]] = []

        def put(self, payload):
            self.payloads.append(payload)

    q = _Queue()
    case_execution._run_case_worker(
        q,
        case={"name": "tiny_case"},
        methods_to_test=["kl"],
        case_plot_umap=True,
        case_plot_manifold=False,
        enable_plots=True,
        pdf_path="/tmp/tiny_case.pdf",
    )

    assert captured.get("include_cover_pages") is False
    assert q.payloads and q.payloads[0].get("ok") is True
