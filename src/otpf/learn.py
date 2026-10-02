"""Maximum-likelihood parameter learning, exact (Kalman) or by particle-filter gradients."""

from __future__ import annotations

import torch
from torch import Tensor

from otpf.filter import particle_filter
from otpf.kalman import kalman_filter
from otpf.models import LinearGaussianSSM
from otpf.resampling import Resampler


def kalman_mle(y: Tensor, theta0: Tensor, **model_kwargs: float) -> Tensor:
    """Exact MLE of theta in `LinearGaussianSSM.diagonal`, by L-BFGS on the Kalman likelihood."""
    theta = theta0.clone().requires_grad_()
    opt = torch.optim.LBFGS(
        [theta], lr=1.0, max_iter=200, tolerance_grad=1e-10, line_search_fn="strong_wolfe"
    )

    def closure() -> Tensor:
        opt.zero_grad()
        loss = -kalman_filter(LinearGaussianSSM.diagonal(theta, **model_kwargs), y).log_likelihood
        loss.backward()
        return loss

    opt.step(closure)
    return theta.detach()


def gradient_ascent(
    y: Tensor,
    theta0: Tensor,
    resampler: Resampler | None,
    *,
    steps: int,
    lr: float,
    n_runs: int = 1,
    n_particles: int = 100,
    seed: int = 0,
    **model_kwargs: float,
) -> Tensor:
    """Adam on the (estimated) average log-likelihood per time step.

    `resampler=None` uses the exact Kalman score. Otherwise `n_runs` independent
    runs share one batched filter: Adam updates each coordinate independently,
    so stacking runs along the batch axis is the same as running them one by one.

    Returns the parameter trajectory, shape (steps + 1, n_runs, d).
    """
    T = y.shape[0]
    theta = theta0.expand(n_runs, -1).clone().requires_grad_()
    opt = torch.optim.Adam([theta], lr=lr)
    generator = torch.Generator().manual_seed(seed)
    path = [theta.detach().clone()]
    for _ in range(steps):
        opt.zero_grad()
        model = LinearGaussianSSM.diagonal(theta, **model_kwargs)
        if resampler is None:
            ll = kalman_filter(model, y).log_likelihood
        else:
            ll = particle_filter(
                model, y, resampler, n_particles=n_particles, batch=n_runs, generator=generator
            ).log_likelihood
        (-ll.sum() / T).backward()
        opt.step()
        path.append(theta.detach().clone())
    return torch.stack(path)
