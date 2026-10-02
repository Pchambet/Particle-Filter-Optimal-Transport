import json
from pathlib import Path

import numpy as np
import pandas as pd

from otpf.summary import eps_pairs, gradient_summary

SCORE = [1.0, -2.0]


def _write(tmp_path: Path, shifts: dict[tuple[str, float | None], float], spread: dict) -> Path:
    """Synthetic benchmark: per-seed scores = exact score + shift + spread * noise."""
    rng = np.random.default_rng(1)
    noise = rng.standard_normal((400, 2))
    rows = []
    for (method, eps), shift in shifts.items():
        for seed, z in enumerate(noise):
            g = np.asarray(SCORE) + shift + spread[(method, eps)] * z
            rows.append([method, eps, 50, seed, -1.0, g[0], g[1], 0.0])
    columns = ["method", "eps", "n_particles", "seed", "log_likelihood", "grad_1", "grad_2"]
    pd.DataFrame(rows, columns=[*columns, "seconds_per_run"]).to_csv(
        tmp_path / "gradient_benchmark.csv", index=False
    )
    meta = {"kalman_at_truth": {"score": SCORE, "log_likelihood": -1.0}}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    return tmp_path


def test_zero_bias_test_separates_biased_from_unbiased_estimators(tmp_path: Path) -> None:
    shifts = {("multinomial", None): 3.0, ("ot", 0.5): 0.0}
    out = gradient_summary(_write(tmp_path, shifts, dict.fromkeys(shifts, 1.0)))
    p = out.set_index("method")["score_bias_p"]
    assert p["multinomial"] < 1e-6
    assert p["ot"] > 0.01
    se = out.set_index("method")["score_bias_norm_se"]
    assert 0.03 < se["multinomial"] < 0.1  # about 1 / sqrt(400)


def test_eps_pairs_measure_the_spread_difference_on_shared_seeds(tmp_path: Path) -> None:
    spread = {("ot", 0.25): 2.0, ("ot", 1.0): 1.0}
    pairs = eps_pairs(_write(tmp_path, dict.fromkeys(spread, 0.0), spread))
    row = pairs.iloc[0]
    assert (row["eps_small"], row["eps_large"]) == (0.25, 1.0)
    assert row["sd_diff"] > 0 and row["sd_diff_z"] > 5
    assert abs(row["bias_norm_diff_z"]) < 3
