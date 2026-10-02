"""Batched log-domain Sinkhorn with exact implicit gradients.

Particle weights can span hundreds of orders of magnitude, so the scaling form
of Sinkhorn (`u = a / Kv`) underflows; every update is written with `logsumexp`
on the dual potentials instead.

Gradients. Unrolling hundreds of iterations at every filter step would keep an
N x N tensor per iteration on the autograd tape. We instead run the
iterations to convergence without recording, and differentiate the fixed point
with the implicit function theorem: one (N + M - 1) linear solve per plan,
memory O(N M), exact up to the solver tolerance (a RuntimeWarning is raised if
`max_iter` is reached first).
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import torch
from torch import Tensor


def _log_plan(
    log_a: Tensor, log_b: Tensor, cost: Tensor, f: Tensor, g: Tensor, eps: float
) -> Tensor:
    """log P_ij = log a_i + log b_j + (f_i + g_j - C_ij) / eps."""
    return (
        log_a[..., :, None] + log_b[..., None, :] + (f[..., :, None] + g[..., None, :] - cost) / eps
    )


def _iterate(
    log_a: Tensor, log_b: Tensor, cost: Tensor, eps: float, max_iter: int, tol: float
) -> tuple[Tensor, Tensor]:
    """Sinkhorn iterations until the source marginal L1 error is below `tol`.

    Works on the scaled potentials f/eps and g/eps with -C/eps precomputed, so
    an iteration costs two fused add-logsumexp passes over the N x M kernel.
    eps-scaling (a few iterations at geometrically decreasing eps, starting from
    the largest cost) warm-starts the potentials. After the final g-update the
    target marginal holds to machine precision.
    """
    fs = torch.zeros_like(log_a)
    gs = torch.zeros_like(log_b)
    level = max(float(cost.detach().max()), eps)
    while True:
        log_k = -cost / level
        if level == eps:
            break
        fs = -torch.logsumexp(log_k + (log_b + gs)[..., None, :], dim=-1)
        gs = -torch.logsumexp(log_k + (log_a + fs)[..., :, None], dim=-2)
        new = max(0.5 * level, eps)
        fs, gs, level = fs * level / new, gs * level / new, new
    target = log_a.exp()
    error = float("inf")
    for it in range(max_iter):
        fs = -torch.logsumexp(log_k + (log_b + gs)[..., None, :], dim=-1)
        gs = -torch.logsumexp(log_k + (log_a + fs)[..., :, None], dim=-2)
        if it % 5 == 4 or it == max_iter - 1:
            log_p = log_k + (log_a + fs)[..., :, None] + (log_b + gs)[..., None, :]
            error = float(
                (torch.logsumexp(log_p, dim=-1).exp() - target).abs().sum(-1).max().detach()
            )
            if error < tol:
                break
    else:
        warnings.warn(
            f"Sinkhorn reached max_iter={max_iter} with marginal L1 error {error:.1e} "
            f"above tol={tol:.0e}; the plan and its implicit gradient are approximate.",
            RuntimeWarning,
            stacklevel=2,
        )
    return eps * fs, eps * gs


class _ImplicitPlan(torch.autograd.Function):
    """P(log_a, log_b, C) at the Sinkhorn fixed point, differentiated implicitly.

    Write P_ij = exp((alpha_i + beta_j - C_ij) / eps) with the marginal
    constraints h1 = P 1 - a = 0 and h2 = P^T 1 - b = 0. The system is invariant
    to (alpha + c, beta - c) and one constraint is redundant, so we fix beta_M
    and drop h2_M. Eliminating alpha (Schur complement) leaves an (M-1) system
    in which every 1/a_i appears only inside K = diag(a)^-1 P, a row-stochastic
    matrix: particles with weights of 1e-300 cause no overflow.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any, log_a: Tensor, log_b: Tensor, cost: Tensor, eps: float, max_iter: int, tol: float
    ) -> Tensor:
        f, g = _iterate(log_a, log_b, cost, eps, max_iter, tol)
        log_p = _log_plan(log_a, log_b, cost, f, g, eps)
        ctx.save_for_backward(log_p, log_a, log_b)
        ctx.eps = eps
        return log_p.exp()

    @staticmethod
    def backward(ctx: Any, grad_p: Tensor) -> tuple[Tensor | None, ...]:  # type: ignore[override]
        log_p, log_a, log_b = ctx.saved_tensors
        eps = ctx.eps
        P = log_p.exp()
        K = (log_p - log_a[..., :, None]).exp()
        GP, GK = grad_p * P, grad_p * K
        u = GK.sum(-1)  # eps * diag(a)^-1 * (d loss / d alpha)
        rhs = GP.sum(-2)[..., :-1] - (P[..., :-1].transpose(-1, -2) @ u.unsqueeze(-1)).squeeze(-1)
        schur = (
            torch.diag_embed(log_b[..., :-1].exp()) - P[..., :-1].transpose(-1, -2) @ K[..., :-1]
        )
        lam2 = torch.linalg.solve(schur, rhs)
        lam1 = u - (K[..., :-1] @ lam2.unsqueeze(-1)).squeeze(-1)
        lam2 = torch.cat([lam2, torch.zeros_like(lam2[..., :1])], dim=-1)
        grad_cost = P * (lam1[..., :, None] + lam2[..., None, :] - grad_p) / eps
        grad_log_a = lam1 * log_a.exp()
        grad_log_b = lam2 * log_b.exp()
        return grad_log_a, grad_log_b, grad_cost, None, None, None


def sinkhorn_plan(
    log_a: Tensor,
    log_b: Tensor,
    cost: Tensor,
    eps: float,
    *,
    max_iter: int = 2000,
    tol: float = 1e-6,
    gradient: Literal["implicit", "unroll"] = "implicit",
) -> Tensor:
    """Entropic OT coupling between weights a (..., N) and b (..., M).

    Args:
        log_a, log_b: log weights, each summing to one along the last axis.
        cost: (..., N, M) ground cost.
        eps: entropic regularisation, in the units of `cost`.
        max_iter, tol: stopping rule on the source-marginal L1 error.
        gradient: "implicit" (default, exact at convergence, O(NM) memory) or
            "unroll" (autograd through every iteration; reference for tests).

    Returns:
        P (..., N, M) with row sums a (up to `tol`) and column sums b.
        Gradients are exact for perturbations that keep sum(a) = sum(b), which is
        the case whenever `log_a` comes from normalised weights.
    """
    if gradient == "unroll":
        f, g = _iterate(log_a, log_b, cost, eps, max_iter, tol)
        return _log_plan(log_a, log_b, cost, f, g, eps).exp()
    return _ImplicitPlan.apply(log_a, log_b, cost, eps, max_iter, tol)
