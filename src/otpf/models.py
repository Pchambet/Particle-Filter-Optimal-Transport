"""State-space models with reparameterised noise.

Every random draw is passed in as a standard-normal tensor and transformed by
the model. This is what lets gradients reach the noise parameters: the legacy
code drew the process noise with a fixed variance before calling the
transition, so the gradient with respect to that variance was identically zero.

Tensors carry a leading batch axis B (independent filters, typically one per
seed) and a particle axis N: states are (B, N, d). Parameters may carry the
batch axis too, so that a single backward pass returns one gradient per seed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor


class StateSpaceModel(Protocol):
    state_dim: int

    def initial(self, noise: Tensor) -> Tensor:
        """Draw x_1 from standard-normal noise of shape (B, N, d)."""
        ...

    def transition(self, x: Tensor, t: int, noise: Tensor) -> Tensor:
        """Draw x_t given x_{t-1} (both (B, N, d))."""
        ...

    def log_obs(self, x: Tensor, y_t: Tensor) -> Tensor:
        """log p(y_t | x_t) for each particle, shape (B, N)."""
        ...


def gaussian_logpdf(diff: Tensor, chol: Tensor) -> Tensor:
    """log N(diff; 0, L L^T) for diff (..., k) and lower-triangular L (..., k, k)."""
    k = diff.shape[-1]
    z = torch.linalg.solve_triangular(chol, diff.unsqueeze(-1), upper=False).squeeze(-1)
    log_det = torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
    return -0.5 * (z**2).sum(-1) - log_det - 0.5 * k * math.log(2 * math.pi)


@dataclass
class LinearGaussianSSM:
    """x_1 ~ N(m0, P0); x_t = A x_{t-1} + q_t, q_t ~ N(0, Q); y_t = C x_t + r_t, r_t ~ N(0, R).

    Matrices are (d, d) or batched (B, d, d); `chol_*` are lower Cholesky factors.
    """

    A: Tensor
    chol_Q: Tensor
    C: Tensor
    chol_R: Tensor
    m0: Tensor
    chol_P0: Tensor

    @property
    def state_dim(self) -> int:
        return self.A.shape[-1]

    def initial(self, noise: Tensor) -> Tensor:
        return self.m0 + noise @ self.chol_P0.transpose(-1, -2)

    def transition(self, x: Tensor, t: int, noise: Tensor) -> Tensor:
        return x @ self.A.transpose(-1, -2) + noise @ self.chol_Q.transpose(-1, -2)

    def log_obs(self, x: Tensor, y_t: Tensor) -> Tensor:
        diff = y_t - x @ self.C.transpose(-1, -2)
        # (B, 1, k, k) so the factor broadcasts over the particle axis.
        return gaussian_logpdf(diff, self.chol_R.unsqueeze(-3))

    @classmethod
    def diagonal(
        cls,
        theta: Tensor,
        sigma_x: float = 1.0,
        sigma_y: float = 0.5,
        p0: float = 1.0,
    ) -> LinearGaussianSSM:
        """The benchmark model: A = diag(theta), Q = sigma_x^2 I, C = I, R = sigma_y^2 I.

        `theta` is (d,) or (B, d); the noise scales are fixed and known.
        """
        d = theta.shape[-1]
        eye = torch.eye(d, dtype=theta.dtype)
        return cls(
            A=torch.diag_embed(theta),
            chol_Q=sigma_x * eye,
            C=eye,
            chol_R=sigma_y * eye,
            m0=torch.zeros(d, dtype=theta.dtype),
            chol_P0=math.sqrt(p0) * eye,
        )

    def simulate(self, T: int, generator: torch.Generator) -> tuple[Tensor, Tensor]:
        """One trajectory (x, y), each (T, d), for an unbatched model."""
        d, dtype = self.state_dim, self.A.dtype
        xs, ys = [], []
        x = self.initial(torch.randn(1, 1, d, generator=generator, dtype=dtype))
        for t in range(T):
            if t > 0:
                x = self.transition(x, t, torch.randn(1, 1, d, generator=generator, dtype=dtype))
            r = torch.randn(1, 1, d, generator=generator, dtype=dtype)
            y = x @ self.C.T + r @ self.chol_R.T
            xs.append(x[0, 0])
            ys.append(y[0, 0])
        return torch.stack(xs), torch.stack(ys)


@dataclass
class KitagawaSSM:
    """The univariate nonlinear growth model used by the legacy project.

    x_1 ~ N(0, p0);  x_t = x/2 + 25 x / (1 + x^2) + 8 cos(1.2 t) + sigma_x v_t;
    y_t = x_t^2 / 20 + sigma_y w_t.  The x^2 observation makes the filtering
    distribution bimodal (the sign of x is ambiguous), a hard case for any
    resampling scheme that averages particles.
    """

    sigma_x: float = math.sqrt(10.0)
    sigma_y: float = 1.0
    p0: float = 5.0
    state_dim: int = 1

    def initial(self, noise: Tensor) -> Tensor:
        return math.sqrt(self.p0) * noise

    def transition(self, x: Tensor, t: int, noise: Tensor) -> Tensor:
        return 0.5 * x + 25 * x / (1 + x**2) + 8 * math.cos(1.2 * t) + self.sigma_x * noise

    def log_obs(self, x: Tensor, y_t: Tensor) -> Tensor:
        diff = (y_t - x**2 / 20).squeeze(-1)
        return -0.5 * (diff / self.sigma_y) ** 2 - math.log(self.sigma_y * math.sqrt(2 * math.pi))

    def simulate(self, T: int, generator: torch.Generator) -> tuple[Tensor, Tensor]:
        dtype = torch.float64
        xs, ys = [], []
        x = self.initial(torch.randn(1, 1, 1, generator=generator, dtype=dtype))
        for t in range(T):
            if t > 0:
                x = self.transition(x, t, torch.randn(1, 1, 1, generator=generator, dtype=dtype))
            y = x**2 / 20 + self.sigma_y * torch.randn(1, 1, 1, generator=generator, dtype=dtype)
            xs.append(x[0, 0])
            ys.append(y[0, 0])
        return torch.stack(xs), torch.stack(ys)
