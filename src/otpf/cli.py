"""Command line: `otpf data | run | figures | report` (add `--quick` for a smoke run)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from otpf.experiments import Config, run_all, simulate_data


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="otpf", description=__doc__)
    parser.add_argument("command", choices=["data", "run", "figures", "report"])
    parser.add_argument("--quick", action="store_true", help="tiny configuration, seconds long")
    parser.add_argument("--threads", type=int, default=3, help="torch CPU threads")
    parser.add_argument("--data-dir", type=Path, default=Path("data/simulated"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--figures-dir", type=Path, default=Path("docs/figures"))
    parser.add_argument("--site-dir", type=Path, default=Path("site"))
    args = parser.parse_args(argv)

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
