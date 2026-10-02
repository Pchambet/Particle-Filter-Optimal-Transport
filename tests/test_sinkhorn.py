import math

import pytest
import torch

from otpf.sinkhorn import sinkhorn_plan

DT = torch.float64


def _problem(n: int = 12, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(2, n, 1, generator=g, dtype=DT)
    log_a = torch.log_softmax(torch.randn(2, n, generator=g, dtype=DT), dim=1)
    log_b = torch.full((2, n), -math.log(n), dtype=DT)
    return x, log_a, log_b


def test_plan_has_requested_marginals() -> None:
    x, log_a, log_b = _problem()
    plan = sinkhorn_plan(log_a, log_b, torch.cdist(x, x) ** 2, eps=0.1, tol=1e-10)
    torch.testing.assert_close(plan.sum(-1), log_a.exp(), atol=1e-9, rtol=0)
    torch.testing.assert_close(plan.sum(-2), log_b.exp(), atol=1e-12, rtol=0)


def test_two_point_problem_matches_closed_form() -> None:
    # Two atoms at 0 and 1, uniform marginals: by symmetry P = [[p, q], [q, p]] / 2
    # with p + q = 1 and p / q = exp(1 / eps) (the off-diagonal cost is 1).
    eps = 0.5
    z = torch.tensor([[[0.0], [1.0]]], dtype=DT)
    log_u = torch.full((1, 2), -math.log(2), dtype=DT)
    plan = sinkhorn_plan(log_u, log_u, torch.cdist(z, z) ** 2, eps=eps, tol=1e-12)[0]
    p = 1 / (1 + math.exp(-1 / eps))
    expected = torch.tensor([[p, 1 - p], [1 - p, p]], dtype=DT) / 2
    torch.testing.assert_close(plan, expected, atol=1e-10, rtol=0)


def test_large_eps_tends_to_independent_coupling() -> None:
    x, log_a, log_b = _problem()
    plan = sinkhorn_plan(log_a, log_b, torch.cdist(x, x) ** 2, eps=1e4, tol=1e-12)
    outer = log_a.exp()[..., :, None] * log_b.exp()[..., None, :]
    torch.testing.assert_close(plan, outer, atol=1e-5, rtol=0)


def test_survives_extreme_weights() -> None:
    # Weights spanning 1e-300 would underflow a non-log implementation.
    x, _, log_b = _problem()
    log_a = torch.log_softmax(torch.linspace(0, 700, x.shape[1], dtype=DT).repeat(2, 1), dim=1)
    plan = sinkhorn_plan(log_a, log_b, torch.cdist(x, x) ** 2, eps=0.05, max_iter=2000)
    assert torch.isfinite(plan).all()
    torch.testing.assert_close(plan.sum(-2), log_b.exp(), atol=1e-12, rtol=0)


def _projection_loss(x: torch.Tensor, log_a: torch.Tensor, gradient: str) -> torch.Tensor:
    n = x.shape[1]
    log_b = torch.full_like(log_a, -math.log(n))
    cost = torch.cdist(x, x) ** 2
    plan = sinkhorn_plan(log_a, log_b, cost, eps=0.3, tol=1e-13, max_iter=5000, gradient=gradient)
    new_x = n * plan.transpose(1, 2) @ x
    return (new_x**3).sum()


@pytest.mark.parametrize("gradient", ["implicit", "unroll"])
def test_gradient_matches_finite_differences(gradient: str) -> None:
    x, log_a, _ = _problem(n=6, seed=3)
    logits = log_a.clone().requires_grad_()
    x = x.clone().requires_grad_()
    loss = _projection_loss(x, torch.log_softmax(logits, 1), gradient)
    gx, gl = torch.autograd.grad(loss, (x, logits))

    h = 1e-6
    direction_x = torch.randn(x.shape, generator=torch.Generator().manual_seed(7), dtype=DT)
    direction_l = torch.randn(logits.shape, generator=torch.Generator().manual_seed(8), dtype=DT)

    def f(s: float) -> float:
        with torch.no_grad():
            lx = x + s * direction_x
            ll = torch.log_softmax(logits + s * direction_l, 1)
            return _projection_loss(lx, ll, "unroll").item()

    fd = (f(h) - f(-h)) / (2 * h)
    analytic = (gx * direction_x).sum() + (gl * direction_l).sum()
    assert analytic.item() == pytest.approx(fd, rel=1e-6)


def test_implicit_gradient_survives_negligible_weights() -> None:
    # One particle carries almost all the mass: 1 / a_i reaches 1e250 for the others.
    x, _, _ = _problem(n=10, seed=5)
    logits = torch.zeros(2, 10, dtype=DT)
    logits[:, 0] = 600.0
    logits.requires_grad_()
    x = x.clone().requires_grad_()
    loss = _projection_loss(x, torch.log_softmax(logits, 1), "implicit")
    gx, gl = torch.autograd.grad(loss, (x, logits))
    assert torch.isfinite(gx).all() and torch.isfinite(gl).all()
