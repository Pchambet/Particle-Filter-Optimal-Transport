from pathlib import Path

from otpf.cli import parse_args


def test_full_run_writes_to_the_committed_directories() -> None:
    args = parse_args(["run"])
    assert args.results_dir == Path("results")
    assert args.figures_dir == Path("docs/figures")


def test_quick_run_never_targets_the_committed_results() -> None:
    args = parse_args(["run", "--quick"])
    assert args.results_dir == Path("quick/results")
    assert args.figures_dir == Path("quick/docs/figures")
    assert args.site_dir == Path("quick/site")


def test_explicit_directory_wins_over_the_quick_default() -> None:
    assert parse_args(["run", "--quick", "--results-dir", "x"]).results_dir == Path("x")
