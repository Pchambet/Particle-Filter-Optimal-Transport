"""The report page: one self-contained `site/index.html` with interactive Plotly charts."""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from otpf.figures import AMBER, INK, SLATE, color
from otpf.summary import gradient_summary, kitagawa_summary, learning_summary, load_meta

PLOTLY_CDN = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@4.1.1/plotly.min.js"
# Ink is invisible on a dark background: exact values use a mid slate that reads in both themes.
REF = "#94a3b8"
AXIS = {"gridcolor": "rgba(100,116,139,0.2)", "zeroline": False, "linecolor": SLATE}


def _layout(fig: go.Figure, **kwargs: object) -> go.Figure:
    settings: dict[str, object] = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Inter, system-ui, sans-serif", "color": SLATE, "size": 13},
        "margin": {"l": 60, "r": 20, "t": 20, "b": 50},
        "legend": {"orientation": "h", "y": -0.22},
        "hoverlabel": {"font": {"family": "Inter, system-ui, sans-serif"}},
    }
    fig.update_layout(**(settings | kwargs))
    fig.update_xaxes(**AXIS)
    fig.update_yaxes(**AXIS)
    return fig


def _score_error_chart(summary: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for (method, eps), g in summary.groupby(["method", "eps"], dropna=False):
        eps_v = None if pd.isna(eps) else float(eps)
        g = g.sort_values("n_particles")
        fig.add_trace(
            go.Scatter(
                x=g["n_particles"],
                y=g["score_rmse"],
                mode="lines+markers",
                name=g["label"].iloc[0],
                line={"color": color(method, eps_v), "width": 2.5},
                customdata=np.stack([g["score_bias_norm"], g["score_sd"]], axis=-1),
                hovertemplate="N=%{x}<br>RMSE %{y:.1f}<br>bias %{customdata[0]:.1f}"
                "<br>sd %{customdata[1]:.1f}<extra>%{fullData.name}</extra>",
            )
        )
    _layout(fig, height=380)
    fig.update_xaxes(
        type="log", title="particles N", tickvals=sorted(summary["n_particles"].unique())
    )
    fig.update_yaxes(type="log", title="RMSE of the score")
    return fig


def _scatter_chart(results_dir: Path, score: list[float], n: int) -> go.Figure:
    frame = pd.read_csv(results_dir / "gradient_benchmark.csv")
    frame = frame[frame["n_particles"] == n]
    fig = go.Figure()
    for method, eps, name in [
        ("multinomial", None, "Multinomial"),
        ("soft", None, "Soft (alpha=0.5)"),
        ("ot", 0.5, "OT, eps=0.5"),
    ]:
        sub = frame[(frame["method"] == method) & (frame["eps"].fillna(-1) == (eps or -1))]
        fig.add_trace(
            go.Scatter(
                x=sub["grad_1"],
                y=sub["grad_2"],
                mode="markers",
                name=name,
                marker={"color": color(method, eps), "size": 6, "opacity": 0.55},
                hovertemplate="seed %{text}<br>(%{x:.1f}, %{y:.1f})<extra>" + name + "</extra>",
                text=sub["seed"],
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[score[0]],
            y=[score[1]],
            mode="markers",
            name="Exact score (Kalman)",
            marker={"color": REF, "size": 18, "symbol": "star", "line": {"color": INK, "width": 1}},
        )
    )
    _layout(fig, height=420)
    fig.update_xaxes(title="d log p / d theta_1")
    fig.update_yaxes(title="d log p / d theta_2")
    return fig


def _learning_chart(results_dir: Path, meta: dict) -> go.Figure:
    surface = pd.read_csv(results_dir / "likelihood_surface.csv")
    n = int(np.sqrt(len(surface)))
    axis = surface["theta_1"].to_numpy().reshape(n, n)[:, 0]
    ll = surface["log_likelihood"].to_numpy().reshape(n, n)
    fig = go.Figure(
        go.Contour(
            x=axis,
            y=axis,
            z=np.log10(1 + ll.max() - ll).T,
            colorscale=[[0, "rgba(100,116,139,0.6)"], [1, "rgba(100,116,139,0.6)"]],
            showscale=False,
            contours={"coloring": "lines"},
            line={"width": 1},
            hoverinfo="skip",
            ncontours=12,
        )
    )
    paths = pd.read_csv(results_dir / "learning_paths.csv")
    names = {"kalman": "Kalman (exact)", "multinomial": "Multinomial", "soft": "Soft", "ot": "OT"}
    for method, name in names.items():
        eps = meta["config"]["learn_eps"] if method == "ot" else None
        for i, (_, run) in enumerate(paths[paths["method"] == method].groupby("run")):
            fig.add_trace(
                go.Scatter(
                    x=run["theta_1"],
                    y=run["theta_2"],
                    mode="lines",
                    name=name,
                    legendgroup=method,
                    showlegend=i == 0,
                    line={"color": REF if method == "kalman" else color(method, eps), "width": 2},
                    hovertemplate="step %{text}<br>(%{x:.3f}, %{y:.3f})<extra>" + name + "</extra>",
                    text=run["step"],
                )
            )
    mle = meta["kalman_mle"]
    fig.add_trace(
        go.Scatter(
            x=[mle[0]],
            y=[mle[1]],
            mode="markers",
            name="Exact MLE",
            marker={"color": REF, "size": 18, "symbol": "star", "line": {"color": INK, "width": 1}},
        )
    )
    _layout(fig, height=460)
    fig.update_xaxes(title="theta_1", range=[0, 1])
    fig.update_yaxes(title="theta_2", range=[0, 1])
    return fig


def _kitagawa_chart(summary: pd.DataFrame) -> go.Figure:
    s = summary[summary["method"] != "multinomial"]
    colors = [
        REF if m == "reference" else color(m, e) for m, e in zip(s["method"], s["eps"], strict=True)
    ]
    fig = go.Figure(
        go.Scatter(
            x=s["diff_vs_multinomial"],
            y=s["label"],
            mode="markers",
            marker={"color": colors, "size": 12},
            error_x={"type": "data", "array": s["diff_ci95"], "color": SLATE, "thickness": 1.5},
            hovertemplate="%{y}<br>%{x:+.2f} RMSE vs multinomial<extra></extra>",
        )
    )
    fig.add_vline(x=0, line={"color": AMBER, "width": 2})
    _layout(fig, height=320, margin={"l": 230, "r": 20, "t": 20, "b": 50})
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="change in RMSE vs multinomial (paired, 95% CI)")
    return fig


def _table(frame: pd.DataFrame, columns: dict[str, str], digits: int = 2) -> str:
    head = "".join(f"<th>{html.escape(v)}</th>" for v in columns.values())
    rows = []
    for _, r in frame.iterrows():
        cells = []
        for key in columns:
            value = r[key]
            text = f"{value:.{digits}f}" if isinstance(value, float) else str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='table'><table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def _chart(div_id: str, fig: go.Figure) -> str:
    spec = fig.to_json()
    return (
        f"<div id='{div_id}' class='chart'></div>"
        f"<script>(function(){{const f={spec};"
        f"Plotly.newPlot('{div_id}',f.data,f.layout,{{responsive:true,displaylogo:false}});}})();</script>"
    )


def build_numbers(results_dir: Path) -> dict[str, float]:
    """The handful of numbers quoted in the narrative (and in the README)."""
    g = gradient_summary(results_dir)
    n = g["n_particles"].max()
    at_n = g[g["n_particles"] == n].set_index("label")
    learn = learning_summary(results_dir).set_index("method")
    kit = kitagawa_summary(results_dir).set_index("label")
    meta = load_meta(results_dir)
    return {
        "n": int(n),
        "rmse_multinomial": at_n.loc["Multinomial", "score_rmse"],
        "rmse_soft": at_n.loc["Soft (alpha=0.5)", "score_rmse"],
        "rmse_ot": at_n.loc["OT, eps=0.5", "score_rmse"],
        "bias_multinomial": at_n.loc["Multinomial", "score_bias_norm"],
        "bias_ot": at_n.loc["OT, eps=0.5", "score_bias_norm"],
        "score_norm": float(np.linalg.norm(meta["kalman_at_truth"]["score"])),
        "seconds_multinomial": at_n.loc["Multinomial", "seconds_per_run"],
        "seconds_ot": at_n.loc["OT, eps=0.5", "seconds_per_run"],
        "dist_multinomial": learn.loc["multinomial", "dist_to_mle"],
        "dist_soft": learn.loc["soft", "dist_to_mle"],
        "dist_ot": learn.loc["ot", "dist_to_mle"],
        "kit_multinomial": kit.loc["Multinomial", "rmse_truth"],
        "kit_ot": kit.loc["OT, eps=0.5", "rmse_truth"],
        "kit_diff": kit.loc["OT, eps=0.5", "diff_vs_multinomial"],
        "kit_ci": kit.loc["OT, eps=0.5", "diff_ci95"],
        "kit_diff_eps1": kit.loc["OT, eps=1", "diff_vs_multinomial"],
        "kit_ci_eps1": kit.loc["OT, eps=1", "diff_ci95"],
        "worst_learning_run": learn.drop(index="kalman")["dist_to_mle_max"].max(),
    }


CSS = """
:root{--bg:#ffffff;--ink:#0f172a;--muted:#64748b;--line:#e2e8f0;--accent:#0d9488;--card:#f8fafc}
@media (prefers-color-scheme:dark){:root{--bg:#0b1120;--ink:#e2e8f0;--muted:#94a3b8;--line:#1e293b;--accent:#2dd4bf;--card:#111827}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.65 Inter,system-ui,-apple-system,sans-serif}
main{max-width:860px;margin:0 auto;padding:48px 16px 80px}
h1{font-size:2rem;line-height:1.2;margin:0 0 8px}
h2{font-size:1.3rem;margin:48px 0 8px;padding-top:8px;border-top:1px solid var(--line)}
p,li{color:var(--ink)}
.lede{color:var(--muted);font-size:1.1rem;margin:0 0 24px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:24px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.kpi b{display:block;font-size:1.5rem;color:var(--accent)}
.kpi span{color:var(--muted);font-size:.9rem}
.chart{width:100%;margin:8px 0 4px}
.takeaway{color:var(--muted);font-size:.95rem;margin-top:0}
.table{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:.9rem;font-variant-numeric:tabular-nums}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:right}
th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-weight:600}
code{font-size:.9em;background:var(--card);padding:1px 5px;border-radius:4px}
a{color:var(--accent)}
footer{margin-top:56px;color:var(--muted);font-size:.9rem}
"""


def make_report(results_dir: Path, site_dir: Path) -> None:
    meta = load_meta(results_dir)
    cfg = meta["config"]
    g = gradient_summary(results_dir)
    k = kitagawa_summary(results_dir)
    learn = learning_summary(results_dir)
    num = build_numbers(results_dir)
    n = num["n"]
    at_n = g[g["n_particles"] == n]

    kpis = [
        (
            f"{num['bias_multinomial'] / num['bias_ot']:.0f}x",
            f"smaller score bias than multinomial resampling (N={n})",
        ),
        (
            f"{num['rmse_multinomial'] / num['rmse_ot']:.1f}x",
            f"lower score RMSE than multinomial resampling (N={n})",
        ),
        (
            f"{num['worst_learning_run']:.3f}",
            "worst distance to the exact MLE after learning, any scheme",
        ),
        (
            f"{num['seconds_ot'] / num['seconds_multinomial']:.0f}x",
            "slower per filter run than multinomial (CPU, indicative)",
        ),
    ]
    kpi_html = "".join(
        f"<div class='kpi'><b>{html.escape(v)}</b><span>{html.escape(t)}</span></div>"
        for v, t in kpis
    )

    body = f"""
<h1>Differentiable particle filtering with optimal transport</h1>
<p class='lede'>Can a particle filter give gradients good enough to learn a model's parameters?
Benchmarked against the exact Kalman score on a linear-Gaussian model, over {cfg["n_seeds"]} seeds.</p>
<div class='kpis'>{kpi_html}</div>

<h2>The question</h2>
<p>Particle filters estimate the likelihood of state-space models that have no closed form. To fit
their parameters by gradient descent, we need the gradient of that estimate, but the resampling
step draws discrete ancestor indices: automatic differentiation silently treats them as constants
and returns a biased gradient. Corenflos et al. (ICML 2021) replace the random draw with an
entropy-regularised optimal transport map, which is smooth. This page measures what that buys, on a
model where the exact answer is known.</p>

<h2>Setup</h2>
<ul>
<li>Model: x<sub>t</sub> = diag(&theta;) x<sub>t-1</sub> + N(0, {cfg["sigma_x"]:g}<sup>2</sup> I),
y<sub>t</sub> = x<sub>t</sub> + N(0, {cfg["sigma_y"]:g}<sup>2</sup> I), in 2-D, T = {cfg["T"]} observations,
true &theta; = {tuple(cfg["theta_true"])}.</li>
<li>Ground truth: the Kalman filter gives the exact log-likelihood; autograd through it gives the exact score.</li>
<li>Resampling at every step: multinomial (gradient ignores resampling), soft resampling
(Karkus et al., 2018, &alpha; = {cfg["soft_alpha"]}), and OT resampling with &epsilon; in {list(cfg["eps_grid"])}
(squared distance between particles standardised per dimension).</li>
<li>OT plans by log-domain Sinkhorn; gradients by implicit differentiation at the fixed point, checked against finite differences.</li>
</ul>

<h2>1. Gradient accuracy</h2>
{_chart("score-error", _score_error_chart(g))}
<p class='takeaway'>At N = {n}, the score RMSE is {num["rmse_multinomial"]:.1f} for multinomial resampling,
{num["rmse_soft"]:.1f} for soft resampling and {num["rmse_ot"]:.1f} for OT (&epsilon; = 0.5); the exact
score has norm {num["score_norm"]:.1f}. Hover a point for the bias / spread split.</p>
{_chart("score-scatter", _scatter_chart(results_dir, meta["kalman_at_truth"]["score"], n))}
<p class='takeaway'>Each dot is one seed. The multinomial cloud is centred away from the exact score
(bias {num["bias_multinomial"]:.1f}); the OT cloud is centred near it (bias {num["bias_ot"]:.1f}).</p>
{_table(at_n, {"label": "Method", "loglik_bias": "log-lik bias", "loglik_sd": "log-lik sd", "score_bias_norm": "score bias", "score_sd": "score sd", "score_rmse": "score RMSE", "seconds_per_run": "s / run"})}

<h2>2. Learning the parameters</h2>
{_chart("learning", _learning_chart(results_dir, meta))}
<p class='takeaway'>Adam ({cfg["learn_steps"]} steps, learning rate {cfg["learn_lr"]}, N = {cfg["learn_particles"]},
{cfg["learn_runs"]} runs each) from &theta; = {tuple(cfg["theta_init"])}. Contours: exact log-likelihood.
Mean distance of the final iterate (average of the last 20) to the exact MLE: multinomial
{num["dist_multinomial"]:.3f}, soft {num["dist_soft"]:.3f}, OT {num["dist_ot"]:.3f}. On this model the
gradient bias measured at the true parameter does not translate into worse parameters: every run of every
scheme ends within {num["worst_learning_run"]:.3f} of the MLE. Better gradients did not buy
better estimates here; they matter where the biased gradient points somewhere else, which this benchmark does
not show.</p>
{_table(learn, {"method": "Gradient source", "theta_1": "theta_1", "theta_2": "theta_2", "dist_to_mle": "mean dist. to MLE", "dist_to_mle_max": "worst run"}, digits=3)}

<h2>3. A harder, nonlinear model</h2>
{_chart("kitagawa", _kitagawa_chart(k))}
<p class='takeaway'>The univariate growth model of the original student project (y = x<sup>2</sup>/20 + noise,
bimodal posterior), {cfg["kitagawa_sequences"]} sequences of length {cfg["kitagawa_T"]}, N = {cfg["kitagawa_particles"]}.
Paired change in RMSE against multinomial resampling: OT (&epsilon; = 0.5) {num["kit_diff"]:+.2f} &plusmn; {num["kit_ci"]:.2f},
not significant; OT (&epsilon; = 1) {num["kit_diff_eps1"]:+.2f} &plusmn; {num["kit_ci_eps1"]:.2f}, worse, consistent with a large &epsilon;
averaging particles across the two modes. The 2024 claim that OT resampling tracks more accurately came from a
single unseeded run and does not hold.</p>

<h2>Limitations</h2>
<ul>
<li>One small model (2 parameters, d = 2) where the Kalman filter is exact; the point is a ground truth, not scale.</li>
<li>OT resampling costs O(N<sup>2</sup>) per step and many Sinkhorn iterations at small &epsilon;; here it is
about {num["seconds_ot"] / num["seconds_multinomial"]:.0f}x slower than multinomial resampling at N = {n} (CPU, shared machine, indicative only).</li>
<li>The OT estimator is biased for any fixed &epsilon; &gt; 0: the barycentric map shrinks the particle cloud.</li>
<li>Learning runs use a fixed learning rate and few repeats; distances to the MLE are indicative.</li>
</ul>
<footer>Code, tests and data generation: <a href='https://github.com/Pchambet/Particle-Filter-Optimal-Transport'>github.com/Pchambet/Particle-Filter-Optimal-Transport</a>.
Built by <a href='https://github.com/Pchambet'>Pierre Chambet</a> — decision science for operations under uncertainty.</footer>
"""
    page = f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Differentiable Particle Filtering</title>
<meta name='description' content='Optimal-transport resampling vs multinomial and soft resampling, benchmarked against exact Kalman gradients.'>
<link rel='preconnect' href='https://fonts.googleapis.com'><link href='https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap' rel='stylesheet'>
<script src='{PLOTLY_CDN}'></script>
<style>{CSS}</style></head>
<body><main>{body}</main></body></html>
"""
    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "index.html").write_text(page)
    (results_dir / "headline_numbers.json").write_text(json.dumps(num, indent=2) + "\n")
