"""Command line: `otpf data | run | figures | report` (add `--quick` for a smoke run)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from otpf.experiments import Config, run_all, simulate_data


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="otpf", description=__doc__)
    parser.add_argument("command", choices=["data", "run", "figures", "report"])
    parser.add_argument(
        "--quick",
        action="store_true",
        help="tiny configuration, seconds long; writes under quick/ unless directories are given",
    )
    parser.add_argument("--threads", type=int, default=3, help="torch CPU threads")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--figures-dir", type=Path)
    parser.add_argument("--site-dir", type=Path)
    args = parser.parse_args(argv)
    # A smoke run must never overwrite the committed results the README quotes.
    root = Path("quick") if args.quick else Path()
    args.data_dir = args.data_dir or root / "data/simulated"
    args.results_dir = args.results_dir or root / "results"
    args.figures_dir = args.figures_dir or root / "docs/figures"
    args.site_dir = args.site_dir or root / "site"
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    cfg = Config().quick() if args.quick else Config()
    if args.command == "data":
        simulate_data(cfg, args.data_dir)
    elif args.command == "run":
        run_all(cfg, args.data_dir, args.results_dir)
    elif args.command == "figures":
        from otpf.figures import make_figures

        make_figures(args.results_dir, args.figures_dir)
    else:
        from otpf.report import make_report

        make_report(args.results_dir, args.site_dir)


if __name__ == "__main__":
    main()
