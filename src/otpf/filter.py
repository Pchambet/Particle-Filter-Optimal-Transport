"""Bootstrap particle filter with a pluggable resampling step."""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import Tensor

from otpf.models import StateSpaceModel
from otpf.resampling import Resampler


class FilterOutput(NamedTuple):
    log_likelihood: Tensor  # (B,) estimate of log p(y_{1:T})
    filtered_means: Tensor  # (B, T, d) weighted particle means


def particle_filter(
    model: StateSpaceModel,
    y: Tensor,
    resampler: Resampler,
    *,
    n_particles: int,
    batch: int,
    generator: torch.Generator,
) -> FilterOutput:
    """Run `batch` independent filters on observations y.

    `y` is (T, dy), shared by every filter, or (batch, T, dy), one sequence each.

    Resampling happens at every step, as in the differentiable filtering
    literature, so that every method is compared on the same footing. The
    log-likelihood estimate is the sum of log-mean incremental weights; with
    multinomial resampling its exponential is unbiased for p(y_{1:T}).
    """
    d, dtype = model.state_dim, y.dtype
    shape = (batch, n_particles, d)
    x = model.initial(torch.randn(shape, generator=generator, dtype=dtype))
    log_w = torch.full((batch, n_particles), -math.log(n_particles), dtype=dtype)
    ll = torch.zeros(batch, dtype=dtype)
    means = []
    per_filter = y.dim() == 3
    for t in range(y.shape[-2]):
        y_t = y[:, t, None, :] if per_filter else y[t]
        if t > 0:
            x, log_w = resampler(x, log_w, generator)
            x = model.transition(x, t, torch.randn(shape, generator=generator, dtype=dtype))
        unnorm = log_w + model.log_obs(x, y_t)
        increment = torch.logsumexp(unnorm, dim=1)
        ll = ll + increment
        log_w = unnorm - increment[:, None]
        means.append((log_w.exp().unsqueeze(-1) * x).sum(1))
    return FilterOutput(ll, torch.stack(means, dim=1))
