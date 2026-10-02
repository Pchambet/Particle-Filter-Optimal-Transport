import math

import pytest
import torch

from otpf.kalman import kalman_filter
from otpf.models import LinearGaussianSSM

DT = torch.float64


def _dense_loglik(model: LinearGaussianSSM, y: torch.Tensor) -> torch.Tensor:
    """log N(vec(y); 0, Sigma) with Sigma built from the joint Gaussian of (x, y)."""
    T, d = y.shape
    A, C = model.A, model.C
    Q = model.chol_Q @ model.chol_Q.T
    R = model.chol_R @ model.chol_R.T
    P0 = model.chol_P0 @ model.chol_P0.T
    # Cov(x_s, x_t) = A^(t-s) Var(x_s) for t >= s.
    var = [P0]
    for _ in range(1, T):
        var.append(A @ var[-1] @ A.T + Q)
    cov = torch.zeros(T * d, T * d, dtype=DT)
    for s in range(T):
        for t in range(s, T):
            block = torch.linalg.matrix_power(A, t - s) @ var[s]
            cov[t * d : (t + 1) * d, s * d : (s + 1) * d] = C @ block @ C.T
            cov[s * d : (s + 1) * d, t * d : (t + 1) * d] = (C @ block @ C.T).T
    cov += torch.block_diag(*[R] * T)
    dist = torch.distributions.MultivariateNormal(torch.zeros(T * d, dtype=DT), cov)
    return dist.log_prob(y.reshape(-1))


def test_matches_dense_gaussian_likelihood() -> None:
    model = LinearGaussianSSM.diagonal(torch.tensor([0.9, -0.4], dtype=DT))
    _, y = model.simulate(12, torch.Generator().manual_seed(0))
    torch.testing.assert_close(kalman_filter(model, y).log_likelihood, _dense_loglik(model, y))


def test_scalar_first_step_by_hand() -> None:
    # One observation: y_1 ~ N(0, p0 + sigma_y^2).
    model = LinearGaussianSSM.diagonal(torch.tensor([0.5], dtype=DT), sigma_y=0.5, p0=1.0)
    y = torch.tensor([[0.7]], dtype=DT)
    var = 1.0 + 0.25
    expected = -0.5 * 0.7**2 / var - 0.5 * math.log(2 * math.pi * var)
    assert kalman_filter(model, y).log_likelihood.item() == pytest.approx(expected, rel=1e-12)


def test_batched_parameters_match_loop() -> None:
    thetas = torch.tensor([[0.2, 0.3], [0.8, -0.5], [0.95, 0.1]], dtype=DT)
    _, y = LinearGaussianSSM.diagonal(thetas[0]).simulate(30, torch.Generator().manual_seed(1))
    batched = kalman_filter(LinearGaussianSSM.diagonal(thetas), y).log_likelihood
    looped = torch.stack(
        [kalman_filter(LinearGaussianSSM.diagonal(t), y).log_likelihood for t in thetas]
    )
    torch.testing.assert_close(batched, looped)


def test_score_matches_finite_differences() -> None:
    theta = torch.tensor([0.6, 0.3], dtype=DT, requires_grad=True)
    _, y = LinearGaussianSSM.diagonal(theta.detach()).simulate(40, torch.Generator().manual_seed(2))
    ll = kalman_filter(LinearGaussianSSM.diagonal(theta), y).log_likelihood
    (grad,) = torch.autograd.grad(ll, theta)
    h = 1e-6
    for i in range(2):
        e = torch.zeros(2, dtype=DT)
        e[i] = h
        with torch.no_grad():
            up = kalman_filter(LinearGaussianSSM.diagonal(theta + e), y).log_likelihood
            down = kalman_filter(LinearGaussianSSM.diagonal(theta - e), y).log_likelihood
        assert grad[i].item() == pytest.approx(((up - down) / (2 * h)).item(), rel=1e-6)
