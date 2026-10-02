import math

import pytest
import torch

from otpf.filter import particle_filter
from otpf.kalman import kalman_filter
from otpf.learn import gradient_ascent, kalman_mle
from otpf.models import KitagawaSSM, LinearGaussianSSM
from otpf.resampling import Multinomial, OptimalTransport, Soft

DT = torch.float64
THETA = torch.tensor([0.8, 0.5], dtype=DT)


def _data(T: int = 20, seed: int = 0) -> torch.Tensor:
    _, y = LinearGaussianSSM.diagonal(THETA).simulate(T, torch.Generator().manual_seed(seed))
    return y


def _run(resampler, y, n=200, batch=4, seed=1, theta=THETA):  # type: ignore[no-untyped-def]
    return particle_filter(
        LinearGaussianSSM.diagonal(theta),
        y,
        resampler,
        n_particles=n,
        batch=batch,
        generator=torch.Generator().manual_seed(seed),
    )


@pytest.mark.parametrize(
    ("resampler", "n"), [(Multinomial(), 500), (Soft(0.5), 500), (OptimalTransport(0.5), 200)]
)
def test_filter_tracks_kalman(resampler, n: int) -> None:  # type: ignore[no-untyped-def]
    y = _data()
    exact = kalman_filter(LinearGaussianSSM.diagonal(THETA), y)
    out = _run(resampler, y, n=n)
    assert (out.log_likelihood.mean() - exact.log_likelihood).abs() < 1.5
    assert (out.filtered_means - exact.filtered_means).abs().mean() < 0.1


def test_same_seed_same_output_and_new_seed_differs() -> None:
    y = _data()
    a = _run(OptimalTransport(0.5), y, n=50, seed=3)
    b = _run(OptimalTransport(0.5), y, n=50, seed=3)
    c = _run(OptimalTransport(0.5), y, n=50, seed=4)
    torch.testing.assert_close(a.log_likelihood, b.log_likelihood, rtol=0, atol=0)
    assert not torch.allclose(a.log_likelihood, c.log_likelihood)


def test_soft_with_alpha_one_is_multinomial() -> None:
    y = _data()
    a = _run(Multinomial(), y, n=50)
    b = _run(Soft(1.0), y, n=50)
    torch.testing.assert_close(a.log_likelihood, b.log_likelihood)


def test_ot_resampling_preserves_weighted_mean() -> None:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(3, 40, 2, generator=g, dtype=DT)
    log_w = torch.log_softmax(torch.randn(3, 40, generator=g, dtype=DT), dim=1)
    new_x, new_log_w = OptimalTransport(0.3, tol=1e-12)(x, log_w, g)
    before = (log_w.exp().unsqueeze(-1) * x).sum(1)
    torch.testing.assert_close(new_x.mean(1), before, atol=1e-9, rtol=0)
    torch.testing.assert_close(new_log_w, torch.full_like(log_w, -math.log(40)))


def test_ot_score_error_is_well_below_multinomial() -> None:
    # The headline effect on a small problem (T=30, N=50, 32 seeds): the
    # multinomial score ignores resampling and is biased; the OT score is not
    # exact either (eps-bias), but its error is several times smaller.
    y = _data(T=30, seed=0)
    theta = THETA.clone().requires_grad_()
    exact = kalman_filter(LinearGaussianSSM.diagonal(theta), y).log_likelihood
    (score,) = torch.autograd.grad(exact, theta)
    errors = {}
    for resampler in (Multinomial(), OptimalTransport(0.5)):
        theta_b = THETA.repeat(32, 1).requires_grad_()
        out = _run(resampler, y, n=50, batch=32, theta=theta_b)
        (grad,) = torch.autograd.grad(out.log_likelihood.sum(), theta_b)
        errors[resampler.name] = (grad.mean(0) - score).norm().item()
    assert errors["ot"] < 0.5 * errors["multinomial"]


def test_gradients_reach_noise_parameters() -> None:
    # The legacy bug: a pre-drawn noise made d loglik / d sigma_x identically zero.
    y = _data()
    sigma = torch.tensor(1.0, dtype=DT, requires_grad=True)
    model = LinearGaussianSSM.diagonal(THETA)
    model.chol_Q = sigma * torch.eye(2, dtype=DT)
    out = particle_filter(
        model,
        y,
        OptimalTransport(0.5),
        n_particles=50,
        batch=2,
        generator=torch.Generator().manual_seed(0),
    )
    (grad,) = torch.autograd.grad(out.log_likelihood.sum(), sigma)
    assert torch.isfinite(grad) and grad.abs() > 1e-3


def test_kalman_mle_recovers_true_parameters() -> None:
    # Ground truth on synthetic data: with a long series the MLE is consistent.
    _, y = LinearGaussianSSM.diagonal(THETA).simulate(500, torch.Generator().manual_seed(11))
    mle = kalman_mle(y, torch.tensor([0.3, 0.2], dtype=DT))
    torch.testing.assert_close(mle, THETA, atol=0.1, rtol=0)


def test_gradient_ascent_reaches_the_mle_and_ot_points_to_it() -> None:
    y = _data(T=40, seed=11)
    theta0 = torch.tensor([0.3, 0.2], dtype=DT)
    mle = kalman_mle(y, theta0)
    exact_path = gradient_ascent(y, theta0, None, steps=80, lr=0.05)
    torch.testing.assert_close(exact_path[-1, 0], mle, atol=2e-2, rtol=0)
    # One Adam step uses the sign of the gradient: every OT run must move from
    # theta0 towards the MLE in both coordinates.
    ot_step = gradient_ascent(
        y, theta0, OptimalTransport(0.5), steps=1, lr=0.05, n_runs=4, n_particles=32
    )
    assert ((ot_step[1] - theta0) * (mle - theta0) > 0).all()


def test_kitagawa_filter_runs_on_per_filter_sequences() -> None:
    model = KitagawaSSM()
    g = torch.Generator().manual_seed(0)
    _, ys = zip(*(model.simulate(15, g) for _ in range(3)), strict=True)
    out = particle_filter(
        model,
        torch.stack(ys),
        OptimalTransport(0.5),
        n_particles=64,
        batch=3,
        generator=g,
    )
    assert out.filtered_means.shape == (3, 15, 1)
    assert torch.isfinite(out.log_likelihood).all()
