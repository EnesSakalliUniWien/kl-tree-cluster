"""Tests for Goncalves progenitor analysis plot timestamps."""

import importlib
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure


def _load_analysis_module():
    return importlib.import_module("applications.scrna.analysis.analyze_goncalves_progenitors")


def test_goncalves_meeting_plot_records_timestamp_before_save(monkeypatch, tmp_path):
    module = _load_analysis_module()
    generated_at = "2026-06-24T19:50:00+02:00"
    meetings = pd.DataFrame(
        [
            {
                "label": "N1",
                "n_cells": 30,
                "n_tbs_clusters": 2,
                "progenitor_population_fraction": 0.8,
                "trunk_progenitor_score": 0.5,
                "tip_progenitor_score": 0.25,
            }
        ]
    )
    saved = []

    def savefig_spy(self, path, *args, **kwargs):
        saved.append(
            {
                "path": Path(path),
                "texts": [text.get_text() for text in self.texts],
                "metadata": kwargs.get("metadata", {}),
            }
        )

    monkeypatch.setattr(Figure, "savefig", savefig_spy)

    module.write_meeting_plot(meetings, tmp_path / "meeting.png", generated_at)

    assert [item["path"].suffix for item in saved] == [".png", ".pdf"]
    assert all(f"Generated at: {generated_at}" in item["texts"] for item in saved)
    assert saved[1]["metadata"]["Subject"] == f"Generated at: {generated_at}"
