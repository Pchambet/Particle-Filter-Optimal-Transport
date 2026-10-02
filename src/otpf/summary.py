"""Summary statistics shared by the figures, the report page and the README."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BOOTSTRAP = 2000


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


def _bias_and_sd(g: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Norm of the mean error and total standard deviation, over the seed axis (-2)."""
    bias = np.linalg.norm(g.mean(-2) - score, axis=-1)
    sd = np.sqrt(g.var(-2, ddof=1).sum(-1))
    return bias, sd


def _resample(n: int) -> np.ndarray:
    """Bootstrap seed indices, fixed so that every summary is reproducible."""
    return np.random.default_rng(0).integers(0, n, size=(BOOTSTRAP, n))


def _bias_p_value(err: np.ndarray) -> float:
    """Wald test of a zero mean error: chi2 with 2 degrees of freedom, exp(-w / 2)."""
    mean = err.mean(0)
    cov = np.cov(err.T) / len(err)
    return float(np.exp(-0.5 * mean @ np.linalg.solve(cov, mean)))


def _scores(frame: pd.DataFrame) -> np.ndarray:
    return frame.sort_values("seed")[["grad_1", "grad_2"]].to_numpy()


def gradient_summary(results_dir: Path) -> pd.DataFrame:
    """Bias, spread and RMSE of the log-likelihood and score estimates, per method and N.

    The standard errors of the bias norm and of the spread are bootstrapped over seeds;
    `score_bias_p` tests whether the bias vector could be zero.
    """
    frame = pd.read_csv(results_dir / "gradient_benchmark.csv")
    ref = load_meta(results_dir)["kalman_at_truth"]
    score = np.asarray(ref["score"])
    rows = []
    for (method, eps, n), group in frame.groupby(["method", "eps", "n_particles"], dropna=False):
        g = _scores(group)
        err = g - score
        mean_err = err.mean(0)
        se = g.std(0, ddof=1) / np.sqrt(len(g))
        boot_bias, boot_sd = _bias_and_sd(g[_resample(len(g))], score)
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
                "score_bias_norm_se": float(boot_bias.std(ddof=1)),
                "score_bias_p": _bias_p_value(err),
                "score_sd": float(np.sqrt(g.var(0, ddof=1).sum())),
                "score_sd_se": float(boot_sd.std(ddof=1)),
                "score_rmse": float(np.sqrt((err**2).sum(1).mean())),
                "seconds_per_run": group["seconds_per_run"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["n_particles", "method", "eps"], ignore_index=True)


def eps_pairs(results_dir: Path) -> pd.DataFrame:
    """Paired comparison of OT resampling between eps values (same seeds, bootstrapped).

    A positive difference means the smaller eps has the larger bias norm (or spread).
    """
    frame = pd.read_csv(results_dir / "gradient_benchmark.csv")
    frame = frame[frame["method"] == "ot"]
    score = np.asarray(load_meta(results_dir)["kalman_at_truth"]["score"])
    rows = []
    for n, at_n in frame.groupby("n_particles"):
        values = sorted(at_n["eps"].unique())
        by_eps = {e: at_n[at_n["eps"] == e] for e in values}
        for i, small in enumerate(values):
            for large in values[i + 1 :]:
                a, b = by_eps[small], by_eps[large]
                if not np.array_equal(np.sort(a["seed"]), np.sort(b["seed"])):
                    raise ValueError(f"eps={small} and eps={large} do not share seeds at N={n}")
                ga, gb = _scores(a), _scores(b)
                idx = _resample(len(ga))
                bias_a, sd_a = _bias_and_sd(ga, score)
                bias_b, sd_b = _bias_and_sd(gb, score)
                boot_bias_a, boot_sd_a = _bias_and_sd(ga[idx], score)
                boot_bias_b, boot_sd_b = _bias_and_sd(gb[idx], score)
                rows.append(
                    {
                        "n_particles": int(n),
                        "eps_small": float(small),
                        "eps_large": float(large),
                        "bias_norm_diff": float(bias_a - bias_b),
                        "bias_norm_diff_se": float((boot_bias_a - boot_bias_b).std(ddof=1)),
                        "sd_diff": float(sd_a - sd_b),
                        "sd_diff_se": float((boot_sd_a - boot_sd_b).std(ddof=1)),
                    }
                )
    columns = ["n_particles", "eps_small", "eps_large", "bias_norm_diff", "bias_norm_diff_se"]
    out = pd.DataFrame(rows, columns=[*columns, "sd_diff", "sd_diff_se"])
    out["bias_norm_diff_z"] = out["bias_norm_diff"] / out["bias_norm_diff_se"]
    out["sd_diff_z"] = out["sd_diff"] / out["sd_diff_se"]
    return out


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
        dist_to_mle_sd=("dist_to_mle", "std"),
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
