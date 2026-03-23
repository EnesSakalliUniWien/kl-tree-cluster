"""Experiment 31B — Oracle policy equations under the one-active 1D guard.

This is a companion runner for ``exp31_oracle_policy_equations.py`` that keeps
the original script untouched and simply reruns the same oracle-policy analysis
with ``config.ONE_ACTIVE_1D_MODE = "per_tree_load_guard"``.

Outputs are written to separate files so the baseline exp31 artifacts are not
overwritten.

Usage:
    python debug_scripts/enhancement_lab/exp31b_oracle_policy_equations_1d.py
    python debug_scripts/enhancement_lab/exp31b_oracle_policy_equations_1d.py --n-permutations 199
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_LAB = Path(__file__).resolve().parent
_ROOT = _LAB.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_LAB) not in sys.path:
    sys.path.insert(0, str(_LAB))

import exp31_oracle_policy_equations as exp31  # noqa: E402
from lab_helpers import temporary_attr, temporary_config  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k-low-ess", type=int, default=5)
    parser.add_argument("--n-permutations", type=int, default=49)
    parser.add_argument(
        "--one-active-1d-mode",
        type=str,
        default="per_tree_load_guard",
        choices=["off", "per_tree_load_guard"],
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="debug_scripts/enhancement_lab/_oracle_policy_rows_one_active_1d.csv",
    )
    parser.add_argument(
        "--disagreement-output-csv",
        type=str,
        default="debug_scripts/enhancement_lab/_oracle_policy_disagreements_one_active_1d.csv",
    )
    parser.add_argument(
        "--family-portability-output-csv",
        type=str,
        default="debug_scripts/enhancement_lab/_oracle_policy_family_portability_one_active_1d.csv",
    )
    parser.add_argument(
        "--gaussian-subfamily-output-csv",
        type=str,
        default="debug_scripts/enhancement_lab/_oracle_policy_gaussian_subfamily_portability_one_active_1d.csv",
    )
    parser.add_argument(
        "--ranked-false-global-output-csv",
        type=str,
        default="debug_scripts/enhancement_lab/_oracle_policy_ranked_false_global_splits_one_active_1d.csv",
    )
    parser.add_argument(
        "--ranked-false-global-markdown",
        type=str,
        default="debug_scripts/enhancement_lab/_oracle_policy_ranked_false_global_splits_one_active_1d.md",
    )
    parser.add_argument(
        "--fourier-output-csv",
        type=str,
        default="debug_scripts/enhancement_lab/_oracle_policy_fourier_summary_one_active_1d.csv",
    )
    parser.add_argument(
        "--fourier-plot-dir",
        type=str,
        default="debug_scripts/enhancement_lab/_oracle_policy_fourier_plots_one_active_1d",
    )
    parser.add_argument(
        "--extreme-noise-output-csv",
        type=str,
        default="debug_scripts/enhancement_lab/_oracle_policy_gaussian_extreme_noise_slice_one_active_1d.csv",
    )
    parser.add_argument(
        "--extreme-noise-markdown",
        type=str,
        default="debug_scripts/enhancement_lab/_oracle_policy_gaussian_extreme_noise_slice_one_active_1d.md",
    )
    return parser


def _to_exp31_argv(args: argparse.Namespace) -> list[str]:
    return [
        "exp31_oracle_policy_equations.py",
        "--top-k-low-ess",
        str(args.top_k_low_ess),
        "--n-permutations",
        str(args.n_permutations),
        "--output-csv",
        str(args.output_csv),
        "--disagreement-output-csv",
        str(args.disagreement_output_csv),
        "--family-portability-output-csv",
        str(args.family_portability_output_csv),
        "--gaussian-subfamily-output-csv",
        str(args.gaussian_subfamily_output_csv),
        "--ranked-false-global-output-csv",
        str(args.ranked_false_global_output_csv),
        "--ranked-false-global-markdown",
        str(args.ranked_false_global_markdown),
        "--fourier-output-csv",
        str(args.fourier_output_csv),
        "--fourier-plot-dir",
        str(args.fourier_plot_dir),
        "--extreme-noise-output-csv",
        str(args.extreme_noise_output_csv),
        "--extreme-noise-markdown",
        str(args.extreme_noise_markdown),
    ]


def main() -> None:
    args = build_parser().parse_args()
    forwarded_argv = _to_exp31_argv(args)

    print("=" * 116)
    print("Experiment 31B: oracle policy equations under one-active 1D mode")
    print("=" * 116)
    print(f"Using ONE_ACTIVE_1D_MODE={args.one_active_1d_mode!r}")
    print()

    with temporary_config(ONE_ACTIVE_1D_MODE=args.one_active_1d_mode):
        with temporary_attr(sys, "argv", forwarded_argv):
            exp31.main()


if __name__ == "__main__":
    main()
