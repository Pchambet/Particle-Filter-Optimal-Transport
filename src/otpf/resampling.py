"""Resampling schemes, from non-differentiable to fully differentiable.

All take particles x (B, N, d), normalised log-weights (B, N) and a generator,
and return new (x, log_weights).

- `multinomial`: the textbook scheme. Ancestor indices are discrete draws, so
  autograd silently treats them as constants: the resulting gradient ignores how
  the parameters move the weights through resampling, and is biased.
- `soft` (Karkus et al., 2018): draw ancestors from a mixture of the weights and
  the uniform distribution, and correct with importance weights that keep a
  gradient path to the original weights. alpha = 1 is multinomial, alpha = 0
  is unbiased but never focuses the particles.
- `optimal_transport` (Corenflos et al., 2021): replace the random draw by the
  entropy-regularised OT plan P between the weighted cloud and the uniform one,
  and move each particle to its barycentric projection N * sum_i P_ij x_i. The
  map is deterministic and smooth in (x, w), so gradients flow end to end. The
  price is a bias that vanishes as eps -> 0 and an O(N^2) cost per step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from otpf.sinkhorn import sinkhorn_plan


class Resampler(Protocol):
    name: str

    def __call__(
        self, x: Tensor, log_w: Tensor, generator: torch.Generator
    ) -> tuple[Tensor, Tensor]: ...


def _gather(x: Tensor, idx: Tensor) -> Tensor:
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


@dataclass(frozen=True)
class Multinomial:
    name: str = "multinomial"

    def __call__(
        self, x: Tensor, log_w: Tensor, generator: torch.Generator
    ) -> tuple[Tensor, Tensor]:
        n = x.shape[1]
        idx = torch.multinomial(log_w.detach().exp(), n, replacement=True, generator=generator)
        return _gather(x, idx), torch.full_like(log_w, -math.log(n))


@dataclass(frozen=True)
class Soft:
    alpha: float = 0.5
    name: str = "soft"

    def __call__(
        self, x: Tensor, log_w: Tensor, generator: torch.Generator
    ) -> tuple[Tensor, Tensor]:
        n = x.shape[1]
        log_uniform = math.log((1 - self.alpha) / n) if self.alpha < 1 else -math.inf
        log_q = torch.logaddexp(math.log(self.alpha) + log_w, torch.full_like(log_w, log_uniform))
        idx = torch.multinomial(log_q.detach().exp(), n, replacement=True, generator=generator)
        new_log_w = torch.gather(log_w - log_q, 1, idx)
        return _gather(x, idx), new_log_w - torch.logsumexp(new_log_w, dim=1, keepdim=True)


@dataclass(frozen=True)
class OptimalTransport:
    """Entropic OT resampling.

    The cost is the squared distance between particles standardised by their
    (detached) per-dimension spread, so `eps` is dimensionless and comparable
    across models and time steps.
    """

    eps: float = 0.5
    max_iter: int = 2000
    tol: float = 1e-6
    name: str = "ot"

    def __call__(
        self, x: Tensor, log_w: Tensor, generator: torch.Generator
    ) -> tuple[Tensor, Tensor]:
        n = x.shape[1]
        scale = x.detach().std(dim=1, keepdim=True).clamp_min(1e-12)
        z = x / scale
        cost = torch.cdist(z, z) ** 2
        log_b = torch.full_like(log_w, -math.log(n))
        plan = sinkhorn_plan(log_w, log_b, cost, self.eps, max_iter=self.max_iter, tol=self.tol)
        new_x = n * plan.transpose(1, 2) @ x
        return new_x, log_b
