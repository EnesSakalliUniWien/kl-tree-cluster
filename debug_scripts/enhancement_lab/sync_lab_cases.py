#!/usr/bin/env python3
"""Sync lab case lists from the latest benchmark results.

Reads the most recent full benchmark CSV and updates FAILURE_CASES,
REGRESSION_GUARD_CASES, and INTERMEDIATE_CASES in lab_helpers.py.

Usage:
    python debug_scripts/enhancement_lab/sync_lab_cases.py                    # auto-find latest
    python debug_scripts/enhancement_lab/sync_lab_cases.py --csv path/to.csv  # explicit CSV
    python debug_scripts/enhancement_lab/sync_lab_cases.py --dry-run           # preview only
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

_LAB_HELPERS = Path(__file__).resolve().parent / "lab_helpers.py"
_RESULTS_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "results"

# Thresholds
FAILURE_ARI_THRESHOLD = 0.3
PERFECT_ARI_THRESHOLD = 1.0


def find_latest_csv() -> Path:
    """Return the most recent full_benchmark_comparison.csv."""
    candidates = sorted(_RESULTS_DIR.glob("run_*/full_benchmark_comparison.csv"))
    if not candidates:
        raise FileNotFoundError(f"No benchmark CSVs found in {_RESULTS_DIR}")
    return candidates[-1]


def derive_case_lists(csv_path: Path) -> dict[str, list[tuple[str, str]]]:
    """Derive case lists from benchmark CSV.

    Returns dict with 'failures', 'guards', 'intermediates' — each a list
    of (case_name, comment_string) tuples.
    """
    df = pd.read_csv(csv_path)
    kl = df[df["method"] == "kl"].copy()

    if kl.empty:
        raise ValueError("No 'kl' method rows found in CSV")

    kl["true_clusters"] = kl["true_clusters"].astype(int)
    kl["found_clusters"] = kl["found_clusters"].astype(int)

    failures = []
    guards = []
    intermediates = []

    for _, r in kl.sort_values("case_id").iterrows():
        name = r["case_id"]
        ari = r["ari"]
        k_found = r["found_clusters"]
        k_true = r["true_clusters"]
        comment = f"K={k_found}/{k_true}, ARI={ari:.3f}"

        if ari < FAILURE_ARI_THRESHOLD:
            failures.append((name, comment))
        elif ari >= PERFECT_ARI_THRESHOLD and k_found == k_true:
            guards.append((name, comment))
        elif FAILURE_ARI_THRESHOLD <= ari < 0.8:
            intermediates.append((name, comment))

    # Sort failures by ARI ascending, guards/intermediates by name
    failures.sort(key=lambda x: float(x[1].split("ARI=")[1]))
    intermediates.sort(key=lambda x: float(x[1].split("ARI=")[1]))

    return {"failures": failures, "guards": guards, "intermediates": intermediates}


def format_list(var_name: str, items: list[tuple[str, str]], doc: str) -> str:
    """Format a Python list assignment with per-item comments."""
    lines = [doc, f"{var_name} = ["]
    for name, comment in items:
        lines.append(f'    "{name}",  # {comment}')
    lines.append("]")
    return "\n".join(lines)


def update_lab_helpers(
    lists: dict[str, list[tuple[str, str]]],
    *,
    dry_run: bool = False,
    csv_path: Path | None = None,
) -> str:
    """Rewrite the case-list block in lab_helpers.py. Returns the new block."""
    source = _LAB_HELPERS.read_text()

    # Build the replacement block
    header = f"# --- Auto-generated from {csv_path.name if csv_path else 'benchmark CSV'} ---"
    blocks = [
        header,
        format_list(
            "FAILURE_CASES",
            lists["failures"],
            "# Cases with ARI < 0.3 — active failures to investigate",
        ),
        "",
        format_list(
            "REGRESSION_GUARD_CASES",
            lists["guards"],
            "# Cases that must NOT regress (ARI=1.0, exact K)",
        ),
        "",
        format_list(
            "INTERMEDIATE_CASES",
            lists["intermediates"],
            "# Cases with 0.3 <= ARI < 0.8 — partial successes",
        ),
        "# --- End auto-generated ---",
    ]
    new_block = "\n".join(blocks)

    # Find the region to replace: from FAILURE_CASES = [ ... to end of REGRESSION_GUARD_CASES ]
    # We match from `FAILURE_CASES = [` up to the closing `]` of REGRESSION_GUARD_CASES
    # Account for optional INTERMEDIATE_CASES too
    pattern = re.compile(
        r"^# ---.*?auto-generated.*?$\n"  # optional existing header
        r".*?"
        r"^# ---.*?End auto-generated.*?$",
        re.MULTILINE | re.DOTALL,
    )
    if pattern.search(source):
        new_source = pattern.sub(new_block, source)
    else:
        # Fallback: replace from first FAILURE_CASES to end of last known list
        # Find FAILURE_CASES = [ and the REGRESSION_GUARD_CASES closing ]
        start_pat = re.compile(r"^(?:# Cases with.*\n)?FAILURE_CASES\s*=\s*\[", re.MULTILINE)
        m_start = start_pat.search(source)
        if not m_start:
            raise ValueError("Cannot find FAILURE_CASES in lab_helpers.py")

        # Find the end: after REGRESSION_GUARD_CASES closing bracket
        # Look for the ] that closes REGRESSION_GUARD_CASES
        end_candidates = [
            ("REGRESSION_GUARD_CASES", r"REGRESSION_GUARD_CASES\s*=\s*\[.*?\]"),
            ("INTERMEDIATE_CASES", r"INTERMEDIATE_CASES\s*=\s*\[.*?\]"),
        ]
        end_pos = m_start.start()
        for _label, ep in end_candidates:
            m_end = re.search(ep, source, re.DOTALL)
            if m_end and m_end.end() > end_pos:
                end_pos = m_end.end()

        if end_pos == m_start.start():
            # Only FAILURE_CASES found, find its closing bracket
            m_fc = re.search(r"FAILURE_CASES\s*=\s*\[.*?\]", source, re.DOTALL)
            if m_fc:
                end_pos = m_fc.end()
            else:
                raise ValueError("Cannot find closing bracket of FAILURE_CASES")

        new_source = source[: m_start.start()] + new_block + "\n" + source[end_pos + 1 :]

    if dry_run:
        print("=== DRY RUN — would write to lab_helpers.py ===")
        print(new_block)
        print(f"\nFailures: {len(lists['failures'])}")
        print(f"Guards:   {len(lists['guards'])}")
        print(f"Intermediate: {len(lists['intermediates'])}")
        return new_block

    _LAB_HELPERS.write_text(new_source)
    print(f"Updated {_LAB_HELPERS}")
    print(f"  Failures:     {len(lists['failures'])}")
    print(f"  Guards:       {len(lists['guards'])}")
    print(f"  Intermediate: {len(lists['intermediates'])}")
    return new_block


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync lab case lists from benchmark results")
    parser.add_argument("--csv", type=Path, help="Path to benchmark CSV (default: latest)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    csv_path = args.csv or find_latest_csv()
    print(f"Reading: {csv_path}")

    lists = derive_case_lists(csv_path)
    update_lab_helpers(lists, dry_run=args.dry_run, csv_path=csv_path)


if __name__ == "__main__":
    main()
