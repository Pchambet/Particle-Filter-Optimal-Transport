"""Static figures for the README (matplotlib, PNG, 200 dpi)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from otpf.summary import (
    eps_pairs,
    gradient_summary,
    kitagawa_summary,
    learning_summary,
    load_meta,
)

INK = "#0f172a"
TEAL = "#0d9488"
AMBER = "#d97706"
SLATE = "#64748b"
GRID = "#e2e8f0"
# OT variants in teal shades (darker = smaller eps), baselines in amber / slate.
OT_SHADES = {0.25: "#115e59", 0.5: TEAL, 1.0: "#14b8a6"}
COLORS = {"multinomial": AMBER, "soft": SLATE, "kalman": INK}


def color(method: str, eps: float | None = None) -> str:
    if method == "ot":
        return OT_SHADES.get(eps if eps is not None else 0.5, TEAL)
    return COLORS[method]


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": SLATE,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": SLATE,
            "ytick.color": SLATE,
            "text.color": INK,
            "font.size": 10,
            "legend.frameon": False,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _contour(ax: plt.Axes, results_dir: Path) -> None:
    surface = pd.read_csv(results_dir / "likelihood_surface.csv")
    n = int(np.sqrt(len(surface)))
    t1 = surface["theta_1"].to_numpy().reshape(n, n)
    t2 = surface["theta_2"].to_numpy().reshape(n, n)
    ll = surface["log_likelihood"].to_numpy().reshape(n, n)
    drop = ll.max() - ll
    levels = [1, 3, 10, 30, 100, 300]
    cs = ax.contour(t1, t2, drop, levels=levels, colors=GRID, linewidths=1.0)
    ax.clabel(cs, fmt=lambda v: f"-{v:g}", fontsize=7, colors=SLATE)


def hero(results_dir: Path, path: Path) -> None:
    """Left: score error vs particle count. Right: learning on the exact likelihood."""
    _style()
    meta = load_meta(results_dir)
    summary = gradient_summary(results_dir)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"wspace": 0.32})

    for (method, eps), group in summary.groupby(["method", "eps"], dropna=False):
        eps_v = None if pd.isna(eps) else float(eps)
        if method == "ot" and eps_v != 0.5:
            continue
        g = group.sort_values("n_particles")
        c = color(method, eps_v)
        ax1.plot(g["n_particles"], g["score_rmse"], "o-", color=c, lw=2, ms=5)
        last = g.iloc[-1]
        ax1.annotate(
            f"{last['label']}  {last['score_rmse']:.1f}",
            (last["n_particles"], last["score_rmse"]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            color=c,
            fontsize=9,
            fontweight="bold",
        )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    counts = sorted(summary["n_particles"].unique())
    ax1.set_xticks(counts, [str(c) for c in counts])
    ax1.minorticks_off()
    ax1.set_xlim(counts[0] * 0.85, counts[-1] * 2.6)
    ticks = [10, 20, 40, 80]
    ax1.set_yticks(ticks, [str(t) for t in ticks])
    ax1.set_ylim(
        min(ticks[0], summary["score_rmse"].min() * 0.8), summary["score_rmse"].max() * 1.15
    )
    seeds = int(summary["seeds"].min())
    ax1.set_xlabel(f"particles N ({seeds} seeds per point)")
    ax1.set_ylabel("RMSE of the score vs exact Kalman score")
    n_max = counts[-1]
    at_n = summary[summary["n_particles"] == n_max].set_index("label")["score_rmse"]
    ratio = at_n["Multinomial"] / at_n["OT, eps=0.5"]
    ax1.set_title(f"OT resampling cuts the score error {ratio:.1f}x at N={n_max}")

    _contour(ax2, results_dir)
    paths = pd.read_csv(results_dir / "learning_paths.csv")
    for method in ["multinomial", "soft", "ot", "kalman"]:
        sub = paths[paths["method"] == method]
        eps = meta["config"]["learn_eps"] if method == "ot" else None
        for _, run in sub.groupby("run"):
            ax2.plot(run["theta_1"], run["theta_2"], color=color(method, eps), lw=1.4, alpha=0.85)
    mle = meta["kalman_mle"]
    ax2.plot(*mle, marker="*", ms=14, color=INK, zorder=5)
    ax2.annotate("exact MLE", mle, xytext=(10, -16), textcoords="offset points", fontsize=9)
    start = meta["config"]["theta_init"]
    ax2.plot(*start, "o", ms=6, color=INK)
    ax2.annotate("start", start, xytext=(6, 4), textcoords="offset points", fontsize=9)
    finals = learning_summary(results_dir).set_index("method")
    names = {
        "multinomial": "Multinomial",
        "soft": "Soft",
        "ot": "OT, eps=0.5",
        "kalman": "Exact score",
    }
    ax2.text(0.97, 0.97, "distance of the learned theta to the\nexact MLE (last 20 iterates, mean of runs)",
             transform=ax2.transAxes, fontsize=9, color=SLATE, va="top", ha="right")  # fmt: skip
    for i, method in enumerate(["multinomial", "soft", "ot", "kalman"]):
        if method in finals.index:
            ax2.text(
                0.97, 0.85 - 0.06 * i, f"{names[method]}  {finals.loc[method, 'dist_to_mle']:.3f}",
                transform=ax2.transAxes, fontsize=9, fontweight="bold", va="top", ha="right",
                color=color(method, meta["config"]["learn_eps"]),
            )  # fmt: skip
    worst = finals.drop(index="kalman", errors="ignore")["dist_to_mle_max"].max()
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel(r"$\theta_1$ (transition coefficient, dim 1)")
    ax2.set_ylabel(r"$\theta_2$ (transition coefficient, dim 2)")
    ax2.set_title(f"Yet every run lands within {worst:.3f} of the MLE")
    _save(fig, path)


def score_scatter(results_dir: Path, path: Path, n_particles: int = 100) -> None:
    """Per-seed score estimates around the exact score."""
    _style()
    frame = pd.read_csv(results_dir / "gradient_benchmark.csv")
    frame = frame[frame["n_particles"] == n_particles]
    score = load_meta(results_dir)["kalman_at_truth"]["score"]
    fig, ax = plt.subplots(figsize=(6.4, 5))
    lines = []
    for method, eps in [("multinomial", None), ("soft", None), ("ot", 0.5)]:
        sub = frame[(frame["method"] == method) & (frame["eps"].fillna(-1) == (eps or -1))]
        c = color(method, eps)
        ax.scatter(sub["grad_1"], sub["grad_2"], s=12, color=c, alpha=0.45, lw=0)
        mx, my = sub["grad_1"].mean(), sub["grad_2"].mean()
        ax.plot(mx, my, "o", ms=10, color=c, mec="white", mew=1.5, zorder=4)
        name = {"multinomial": "Multinomial", "soft": "Soft", "ot": "OT, eps=0.5"}[method]
        lines.append((f"{name} mean ({mx:.1f}, {my:.1f})", c))
    ax.plot(*score, marker="*", ms=18, color=INK, mec="white", mew=1, zorder=5)
    lines.append((f"exact score ({score[0]:.1f}, {score[1]:.1f})", INK))
    for i, (text, c) in enumerate(lines):
        ax.text(0.02, 0.97 - 0.06 * i, text, transform=ax.transAxes, color=c, fontsize=9,
                fontweight="bold", va="top")  # fmt: skip
    ax.set_xlabel(r"$\partial \log p / \partial \theta_1$")
    ax.set_ylabel(r"$\partial \log p / \partial \theta_2$")
    ax.set_title("OT score estimates centre near the exact score")
    ax.text(0.0, -0.16, f"Each dot: one seed, N={n_particles}, at the true parameter.",
            transform=ax.transAxes, fontsize=9, color=SLATE)  # fmt: skip
    _save(fig, path)


def eps_tradeoff(results_dir: Path, path: Path) -> None:
    """Bias and spread of the OT score as eps varies, against the baselines."""
    _style()
    summary = gradient_summary(results_dir)
    n = summary["n_particles"].max()
    s = summary[summary["n_particles"] == n].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    y = np.arange(len(s))
    bar = {"height": 0.34, "error_kw": {"ecolor": INK, "elinewidth": 1, "capsize": 2}}
    for i, row in s.iterrows():
        c = color(row["method"], row["eps"])
        bias, bias_se = row["score_bias_norm"], row["score_bias_norm_se"]
        sd, sd_se = row["score_sd"], row["score_sd_se"]
        ax.barh(i - 0.18, bias, xerr=bias_se, color=c, **bar)
        ax.barh(i + 0.18, sd, xerr=sd_se, color=c, alpha=0.4, **bar)
        ax.text(bias + bias_se, i - 0.18, f" bias {bias:.1f} ± {bias_se:.1f}", va="center",
                fontsize=8, color=INK)  # fmt: skip
        ax.text(sd + sd_se, i + 0.18, f" sd {sd:.1f} ± {sd_se:.1f}", va="center", fontsize=8,
                color=SLATE)  # fmt: skip
    ax.set_yticks(y, s["label"])
    ax.invert_yaxis()
    ax.set_xlabel("norm of the score bias (solid) and standard deviation (light), ± 1 SE")
    ax.set_title("Every OT eps cuts the bias to a few units; eps differences are within noise")
    ax.text(0.0, -0.2, f"Score estimate at the true parameter, N={n}, {int(s['seeds'].min())} seeds;"
            " SE bootstrapped over seeds.", transform=ax.transAxes, fontsize=9, color=SLATE)  # fmt: skip
    top = (
        s[["score_bias_norm", "score_sd"]].to_numpy()
        + s[["score_bias_norm_se", "score_sd_se"]].to_numpy()
    ).max()
    ax.set_xlim(0, top * 1.35)
    _save(fig, path)


def kitagawa(results_dir: Path, path: Path) -> None:
    """Filtering RMSE on the bimodal nonlinear model, paired against multinomial."""
    _style()
    s = kitagawa_summary(results_dir)
    s = s[s["method"] != "multinomial"].reset_index(drop=True)
    base = kitagawa_summary(results_dir).set_index("method").loc["multinomial", "rmse_truth"]
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ax.axvline(0, color=AMBER, lw=1.5)
    ax.text(0, -0.75, f" multinomial, N=100 (RMSE {base:.2f})", color=AMBER, fontsize=9)
    for i, row in s.iterrows():
        c = INK if row["method"] == "reference" else color(row["method"], row["eps"])
        ax.errorbar(row["diff_vs_multinomial"], i, xerr=row["diff_ci95"], fmt="o", color=c,
                    ms=7, capsize=3, lw=1.5)  # fmt: skip
        ax.text(row["diff_vs_multinomial"] + row["diff_ci95"], i,
                f"  {row['diff_vs_multinomial']:+.2f} ± {row['diff_ci95']:.2f}", va="center",
                fontsize=9, color=c)  # fmt: skip
    ax.set_yticks(range(len(s)), s["label"])
    ax.set_ylim(len(s) - 0.5, -1)
    ax.set_xlim(None, s["diff_vs_multinomial"].max() + s["diff_ci95"].max() + 0.45)
    n_seq = int(s["sequences"].min())
    ax.set_xlabel(f"change in tracking RMSE vs multinomial (paired, 95% CI, {n_seq} sequences)")
    ax.set_title("Nonlinear model: no significant gain from OT; eps=1 borderline worse")
    _save(fig, path)


def make_figures(results_dir: Path, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    hero(results_dir, figures_dir / "hero.png")
    score_scatter(results_dir, figures_dir / "score_scatter.png")
    eps_tradeoff(results_dir, figures_dir / "eps_tradeoff.png")
    kitagawa(results_dir, figures_dir / "kitagawa.png")
    gradient_summary(results_dir).to_csv(results_dir / "summary_gradients.csv", index=False)
    learning_summary(results_dir).to_csv(results_dir / "summary_learning.csv", index=False)
    kitagawa_summary(results_dir).to_csv(results_dir / "summary_kitagawa.csv", index=False)
    eps_pairs(results_dir).to_csv(results_dir / "summary_eps_pairs.csv", index=False)
