"""Kalman filter: the exact log-likelihood of a linear-Gaussian model.

It is written in PyTorch so that autograd gives the exact score
d log p(y_{1:T} | theta) / d theta. That pair (value, gradient) is the ground
truth every particle filter in this repository is measured against.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

from otpf.models import LinearGaussianSSM, gaussian_logpdf


class KalmanOutput(NamedTuple):
    log_likelihood: Tensor  # batch shape of the parameters, () if unbatched
    filtered_means: Tensor  # (..., T, d)


def kalman_filter(model: LinearGaussianSSM, y: Tensor) -> KalmanOutput:
    """Run the filter on observations y (T, dy); parameters may be batched."""
    A, C = model.A, model.C
    Q = model.chol_Q @ model.chol_Q.transpose(-1, -2)
    R = model.chol_R @ model.chol_R.transpose(-1, -2)
    m = model.m0
    P = model.chol_P0 @ model.chol_P0.transpose(-1, -2)
    batch = torch.broadcast_shapes(A.shape[:-2], C.shape[:-2], Q.shape[:-2], R.shape[:-2])
    m = m.expand(*batch, m.shape[-1])
    ll = torch.zeros(batch, dtype=y.dtype)
    means = []
    for t in range(y.shape[0]):
        if t > 0:
            m = (A @ m.unsqueeze(-1)).squeeze(-1)
            P = A @ P @ A.transpose(-1, -2) + Q
        S = C @ P @ C.transpose(-1, -2) + R
        chol_S = torch.linalg.cholesky(S)
        innovation = y[t] - (C @ m.unsqueeze(-1)).squeeze(-1)
        ll = ll + gaussian_logpdf(innovation, chol_S)
        # K = P C^T S^{-1}, computed with the Cholesky factor of S.
        PCt = P @ C.transpose(-1, -2)
        K = torch.cholesky_solve(PCt.transpose(-1, -2), chol_S).transpose(-1, -2)
        m = m + (K @ innovation.unsqueeze(-1)).squeeze(-1)
        P = P - K @ S @ K.transpose(-1, -2)
        P = 0.5 * (P + P.transpose(-1, -2))
        means.append(m)
    return KalmanOutput(ll, torch.stack(means, dim=-2))
