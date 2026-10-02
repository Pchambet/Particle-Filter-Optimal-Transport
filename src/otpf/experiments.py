"""The four experiments behind the README, each writing a tidy CSV to `results/`.

1. Gradient benchmark: log-likelihood and score estimates at the true
   parameter, many seeds, against the exact Kalman values.
2. Likelihood surface: the exact Kalman log-likelihood on a parameter grid.
3. Learning: gradient ascent from a distant start, per resampling scheme.
4. Nonlinear tracking: filtering accuracy on the legacy (Kitagawa) model.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor

from otpf.filter import particle_filter
from otpf.kalman import kalman_filter
from otpf.learn import gradient_ascent, kalman_mle
from otpf.models import KitagawaSSM, LinearGaussianSSM
from otpf.resampling import Multinomial, OptimalTransport, Resampler, Soft

DTYPE = torch.float64


@dataclass(frozen=True)
class Config:
    # Linear-Gaussian benchmark model (theta = diagonal of the transition matrix).
    theta_true: tuple[float, float] = (0.8, 0.5)
    theta_init: tuple[float, float] = (0.1, 0.95)
    sigma_x: float = 1.0
    sigma_y: float = 0.5
    T: int = 100
    data_seed: int = 2026
    # Experiment 1.
    n_seeds: int = 100
    seed_batch: int = 50
    particle_counts: tuple[int, ...] = (25, 50, 100)
    eps_grid: tuple[float, ...] = (0.25, 0.5, 1.0)
    soft_alpha: float = 0.5
    # Experiment 2.
    grid_size: int = 81
    # Experiment 3.
    learn_steps: int = 150
    learn_lr: float = 0.02
    learn_runs: int = 5
    learn_particles: int = 50
    learn_eps: float = 0.5
    # Experiment 4.
    kitagawa_T: int = 50
    kitagawa_sequences: int = 100
    kitagawa_particles: int = 100
    kitagawa_reference_particles: int = 5000

    def quick(self) -> Config:
        """A seconds-long version of every experiment, to smoke-test the pipeline."""
        return replace(
            self,
            T=30,
            n_seeds=8,
            seed_batch=8,
            particle_counts=(25,),
            eps_grid=(0.5,),
            grid_size=11,
            learn_steps=5,
            learn_runs=2,
            learn_particles=25,
            kitagawa_T=20,
            kitagawa_sequences=8,
            kitagawa_particles=25,
            kitagawa_reference_particles=500,
        )

    def model_kwargs(self) -> dict[str, float]:
        return {"sigma_x": self.sigma_x, "sigma_y": self.sigma_y}


def _theta(values: tuple[float, ...]) -> Tensor:
    return torch.tensor(values, dtype=DTYPE)


def simulate_data(cfg: Config, data_dir: Path) -> None:
    """Write the observation sequences every experiment reads."""
    data_dir.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(cfg.data_seed)
    x, y = LinearGaussianSSM.diagonal(_theta(cfg.theta_true), **cfg.model_kwargs()).simulate(
        cfg.T, g
    )
    frame = pd.DataFrame({"t": range(cfg.T)})
    for i in range(x.shape[1]):
        frame[f"x{i + 1}"] = x[:, i].numpy()
        frame[f"y{i + 1}"] = y[:, i].numpy()
    frame.to_csv(data_dir / "lgssm.csv", index=False)

    g = torch.Generator().manual_seed(cfg.data_seed + 1)
    rows = []
    for seq in range(cfg.kitagawa_sequences):
        xs, ys = KitagawaSSM().simulate(cfg.kitagawa_T, g)
        rows += [
            {"sequence": seq, "t": t, "x": xs[t, 0].item(), "y": ys[t, 0].item()}
            for t in range(cfg.kitagawa_T)
        ]
    pd.DataFrame(rows).to_csv(data_dir / "kitagawa.csv", index=False)


def load_lgssm(data_dir: Path) -> Tensor:
    frame = pd.read_csv(data_dir / "lgssm.csv")
    return torch.tensor(frame[["y1", "y2"]].to_numpy(), dtype=DTYPE)


def load_kitagawa(data_dir: Path) -> tuple[Tensor, Tensor]:
    frame = pd.read_csv(data_dir / "kitagawa.csv").sort_values(["sequence", "t"])
    n_seq = frame["sequence"].nunique()
    x = torch.tensor(frame["x"].to_numpy(), dtype=DTYPE).reshape(n_seq, -1, 1)
    y = torch.tensor(frame["y"].to_numpy(), dtype=DTYPE).reshape(n_seq, -1, 1)
    return x, y


def resamplers(cfg: Config) -> Iterator[tuple[str, float | None, Resampler]]:
    yield "multinomial", None, Multinomial()
    yield "soft", None, Soft(cfg.soft_alpha)
    for eps in cfg.eps_grid:
        yield "ot", eps, OptimalTransport(eps)


def gradient_benchmark(cfg: Config, y: Tensor) -> tuple[pd.DataFrame, dict[str, object]]:
    """Experiment 1: per-seed log-likelihood and score estimates at theta_true."""
    theta = _theta(cfg.theta_true).requires_grad_()
    kalman = kalman_filter(LinearGaussianSSM.diagonal(theta, **cfg.model_kwargs()), y)
    (score,) = torch.autograd.grad(kalman.log_likelihood, theta)
    reference = {"log_likelihood": kalman.log_likelihood.item(), "score": score.tolist()}

    rows = []
    for n in cfg.particle_counts:
        for name, eps, resampler in resamplers(cfg):
            for start in range(0, cfg.n_seeds, cfg.seed_batch):
                b = min(cfg.seed_batch, cfg.n_seeds - start)
                theta_b = _theta(cfg.theta_true).repeat(b, 1).requires_grad_()
                generator = torch.Generator().manual_seed(10_000 * n + start)
                tic = time.perf_counter()
                out = particle_filter(
                    LinearGaussianSSM.diagonal(theta_b, **cfg.model_kwargs()),
                    y,
                    resampler,
                    n_particles=n,
                    batch=b,
                    generator=generator,
                )
                (grad,) = torch.autograd.grad(out.log_likelihood.sum(), theta_b)
                seconds = (time.perf_counter() - tic) / b
                rows += [
                    {
                        "method": name,
                        "eps": eps,
                        "n_particles": n,
                        "seed": start + i,
                        "log_likelihood": out.log_likelihood[i].item(),
                        "grad_1": grad[i, 0].item(),
                        "grad_2": grad[i, 1].item(),
                        "seconds_per_run": seconds,
                    }
                    for i in range(b)
                ]
            print(f"  gradients: N={n} {name} eps={eps} done", flush=True)
    return pd.DataFrame(rows), reference


def likelihood_surface(cfg: Config, y: Tensor) -> pd.DataFrame:
    """Experiment 2: exact log-likelihood on a grid over (theta_1, theta_2)."""
    axis = torch.linspace(0.0, 1.0, cfg.grid_size, dtype=DTYPE)
    t1, t2 = torch.meshgrid(axis, axis, indexing="ij")
    grid = torch.stack([t1.reshape(-1), t2.reshape(-1)], dim=-1)
    with torch.no_grad():
        ll = kalman_filter(LinearGaussianSSM.diagonal(grid, **cfg.model_kwargs()), y).log_likelihood
    return pd.DataFrame(
        {"theta_1": grid[:, 0].numpy(), "theta_2": grid[:, 1].numpy(), "log_likelihood": ll.numpy()}
    )


def learning(cfg: Config, y: Tensor) -> tuple[pd.DataFrame, list[float]]:
    """Experiment 3: Adam from theta_init with each gradient source."""
    theta0 = _theta(cfg.theta_init)
    mle = kalman_mle(y, theta0, **cfg.model_kwargs())
    methods: list[tuple[str, Resampler | None, int]] = [
        ("kalman", None, 1),
        ("multinomial", Multinomial(), cfg.learn_runs),
        ("soft", Soft(cfg.soft_alpha), cfg.learn_runs),
        ("ot", OptimalTransport(cfg.learn_eps), cfg.learn_runs),
    ]
    frames = []
    for name, resampler, runs in methods:
        path = gradient_ascent(
            y,
            theta0,
            resampler,
            steps=cfg.learn_steps,
            lr=cfg.learn_lr,
            n_runs=runs,
            n_particles=cfg.learn_particles,
            seed=1,
            **cfg.model_kwargs(),
        )
        steps, n_runs, _ = path.shape
        frames.append(
            pd.DataFrame(
                {
                    "method": name,
                    "run": torch.arange(n_runs).repeat(steps).numpy(),
                    "step": torch.arange(steps).repeat_interleave(n_runs).numpy(),
                    "theta_1": path[..., 0].reshape(-1).numpy(),
                    "theta_2": path[..., 1].reshape(-1).numpy(),
                }
            )
        )
        print(f"  learning: {name} done", flush=True)
    return pd.concat(frames, ignore_index=True), mle.tolist()


def kitagawa_tracking(cfg: Config, x: Tensor, y: Tensor) -> pd.DataFrame:
    """Experiment 4: filtering RMSE on the bimodal nonlinear model, one filter per sequence."""
    model = KitagawaSSM()
    n_seq = y.shape[0]
    with torch.no_grad():
        reference = particle_filter(
            model,
            y,
            Multinomial(),
            n_particles=cfg.kitagawa_reference_particles,
            batch=n_seq,
            generator=torch.Generator().manual_seed(7),
        ).filtered_means
        rows = []
        for name, eps, resampler in resamplers(cfg):
            out = particle_filter(
                model,
                y,
                resampler,
                n_particles=cfg.kitagawa_particles,
                batch=n_seq,
                generator=torch.Generator().manual_seed(8),
            ).filtered_means
            rmse_truth = ((out - x) ** 2).mean(dim=(1, 2)).sqrt()
            rmse_reference = ((out - reference) ** 2).mean(dim=(1, 2)).sqrt()
            rows += [
                {
                    "method": name,
                    "eps": eps,
                    "sequence": s,
                    "rmse_truth": rmse_truth[s].item(),
                    "rmse_reference": rmse_reference[s].item(),
                }
                for s in range(n_seq)
            ]
            print(f"  kitagawa: {name} eps={eps} done", flush=True)
        ref_rmse = ((reference - x) ** 2).mean(dim=(1, 2)).sqrt()
        rows += [
            {
                "method": "reference",
                "eps": None,
                "sequence": s,
                "rmse_truth": ref_rmse[s].item(),
                "rmse_reference": 0.0,
            }
            for s in range(n_seq)
        ]
    return pd.DataFrame(rows)


def run_all(cfg: Config, data_dir: Path, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    y = load_lgssm(data_dir)
    meta: dict[str, object] = {"config": asdict(cfg)}

    print("experiment 1: gradient benchmark", flush=True)
    frame, reference = gradient_benchmark(cfg, y)
    frame.to_csv(results_dir / "gradient_benchmark.csv", index=False)
    meta["kalman_at_truth"] = reference

    print("experiment 2: likelihood surface", flush=True)
    likelihood_surface(cfg, y).to_csv(results_dir / "likelihood_surface.csv", index=False)

    print("experiment 3: learning", flush=True)
    frame, mle = learning(cfg, y)
    frame.to_csv(results_dir / "learning_paths.csv", index=False)
    meta["kalman_mle"] = mle

    print("experiment 4: nonlinear tracking", flush=True)
    xk, yk = load_kitagawa(data_dir)
    kitagawa_tracking(cfg, xk, yk).to_csv(results_dir / "kitagawa_tracking.csv", index=False)

    (results_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
