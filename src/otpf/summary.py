"""Summary statistics shared by the figures, the report page and the README."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def label(method: str, eps: float | None) -> str:
    if method == "ot":
        return f"OT, eps={eps:g}"
    return {
        "multinomial": "Multinomial",
        "soft": "Soft (alpha=0.5)",
        "kalman": "Kalman (exact)",
    }.get(method, method.capitalize())


def load_meta(results_dir: Path) -> dict:
    return json.loads((results_dir / "meta.json").read_text())


def gradient_summary(results_dir: Path) -> pd.DataFrame:
    """Bias, spread and RMSE of the log-likelihood and score estimates, per method and N."""
    frame = pd.read_csv(results_dir / "gradient_benchmark.csv")
    ref = load_meta(results_dir)["kalman_at_truth"]
    score = np.asarray(ref["score"])
    rows = []
    for (method, eps, n), group in frame.groupby(["method", "eps", "n_particles"], dropna=False):
        g = group[["grad_1", "grad_2"]].to_numpy()
        err = g - score
        mean_err = err.mean(0)
        se = g.std(0, ddof=1) / np.sqrt(len(g))
        rows.append(
            {
                "method": method,
                "eps": None if pd.isna(eps) else float(eps),
                "label": label(method, None if pd.isna(eps) else float(eps)),
                "n_particles": int(n),
                "seeds": len(g),
                "loglik_bias": group["log_likelihood"].mean() - ref["log_likelihood"],
                "loglik_sd": group["log_likelihood"].std(ddof=1),
                "score_bias_1": mean_err[0],
                "score_bias_2": mean_err[1],
                "score_bias_se_1": se[0],
                "score_bias_se_2": se[1],
                "score_bias_norm": float(np.linalg.norm(mean_err)),
                "score_sd": float(np.sqrt(g.var(0, ddof=1).sum())),
                "score_rmse": float(np.sqrt((err**2).sum(1).mean())),
                "seconds_per_run": group["seconds_per_run"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["n_particles", "method", "eps"], ignore_index=True)


def learning_summary(results_dir: Path, tail: int = 20) -> pd.DataFrame:
    """Final parameters (mean of the last `tail` iterates, to average out SGD noise)."""
    frame = pd.read_csv(results_dir / "learning_paths.csv")
    mle = np.asarray(load_meta(results_dir)["kalman_mle"])
    last = frame[frame["step"] > frame["step"].max() - tail]
    finals = last.groupby(["method", "run"])[["theta_1", "theta_2"]].mean()
    finals["dist_to_mle"] = np.linalg.norm(finals[["theta_1", "theta_2"]].to_numpy() - mle, axis=1)
    out = finals.groupby("method").agg(
        theta_1=("theta_1", "mean"),
        theta_2=("theta_2", "mean"),
        theta_1_sd=("theta_1", "std"),
        theta_2_sd=("theta_2", "std"),
        dist_to_mle=("dist_to_mle", "mean"),
        dist_to_mle_max=("dist_to_mle", "max"),
        runs=("dist_to_mle", "size"),
    )
    order = ["kalman", "multinomial", "soft", "ot"]
    return out.reindex([m for m in order if m in out.index]).reset_index()


def kitagawa_summary(results_dir: Path) -> pd.DataFrame:
    """Mean RMSE per method, and the paired difference to multinomial with a 95% CI."""
    frame = pd.read_csv(results_dir / "kitagawa_tracking.csv")
    n_ref = load_meta(results_dir)["config"]["kitagawa_reference_particles"]
    base = frame[frame["method"] == "multinomial"].set_index("sequence")["rmse_truth"]
    rows = []
    for (method, eps), group in frame.groupby(["method", "eps"], dropna=False, sort=False):
        eps_value = None if pd.isna(eps) else float(eps)
        diff = group.set_index("sequence")["rmse_truth"] - base
        n = len(group)
        rows.append(
            {
                "method": method,
                "eps": eps_value,
                "label": f"Reference (multinomial, N={n_ref})"
                if method == "reference"
                else label(method, eps_value),
                "sequences": n,
                "rmse_truth": group["rmse_truth"].mean(),
                "rmse_truth_se": group["rmse_truth"].std(ddof=1) / np.sqrt(n),
                "rmse_reference": group["rmse_reference"].mean(),
                "diff_vs_multinomial": diff.mean(),
                "diff_ci95": 1.96 * diff.std(ddof=1) / np.sqrt(n),
            }
        )
    return pd.DataFrame(rows)
