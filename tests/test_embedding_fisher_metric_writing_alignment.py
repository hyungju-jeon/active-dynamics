from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from actdyn.metrics.information import EmbeddingFisherMetric
from actdyn.models.decoder import Decoder, LogLinearMapping, PoissonNoise


def _fe_identity(z: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    if z.ndim == 2:
        z = z.unsqueeze(0)
    batch, time, _ = z.shape
    out = torch.zeros(batch, time, 2, 2, device=z.device)
    out[..., 0, 0] = 1.0
    out[..., 1, 1] = 1.0
    return out


def _fz_zero(z: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    if z.ndim == 2:
        z = z.unsqueeze(0)
    batch, time, _ = z.shape
    return torch.zeros(batch, time, 2, 2, device=z.device)


def _fz_shear(z: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    if z.ndim == 2:
        z = z.unsqueeze(0)
    batch, time, _ = z.shape
    out = torch.zeros(batch, time, 2, 2, device=z.device)
    out[..., 0, 1] = 1.0
    return out


class _DummyPoissonDecoder:
    def __init__(self, d_obs: int = 2, d_latent: int = 2):
        self.d_obs = d_obs
        self.d_latent = d_latent
        self.noise = PoissonNoise(device="cpu")
        self.forward_calls = 0
        self.jacobian_calls = 0
        self.var_calls = 0

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return torch.ones(*z.shape[:-1], self.d_obs, device=z.device)

    def jacobian(self, z: torch.Tensor) -> torch.Tensor:
        self.jacobian_calls += 1
        batch, time, _, = z.shape
        eye = torch.eye(self.d_obs, self.d_latent, device=z.device).view(
            1, 1, self.d_obs, self.d_latent
        )
        return eye.expand(batch, time, -1, -1)

    def var(self, z: torch.Tensor) -> torch.Tensor:
        self.var_calls += 1
        return torch.ones(*z.shape[:-1], self.d_obs, device=z.device)


class _WeightedPoissonDecoder:
    def __init__(self):
        self.noise = PoissonNoise(device="cpu")
        self.forward_calls = 0
        self.jacobian_calls = 0
        self.var_calls = 0
        self.rate = torch.tensor([2.0, 4.0, 5.0], dtype=torch.float32)
        self.H = torch.tensor(
            [[1.0, 0.0], [0.0, 2.0], [1.0, -1.0]], dtype=torch.float32
        )

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return self.rate.to(z.device).view(*([1] * (z.ndim - 1)), -1).expand(*z.shape[:-1], -1)

    def jacobian(self, z: torch.Tensor) -> torch.Tensor:
        self.jacobian_calls += 1
        return self.H.to(z.device).view(1, 1, 3, 2).expand(z.shape[0], z.shape[1], -1, -1)

    def var(self, z: torch.Tensor) -> torch.Tensor:
        self.var_calls += 1
        return self.__call__(z)


class _CountingLogLinearMapping(LogLinearMapping):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.jacobian_calls = 0

    @property
    def jacobian(self):
        self.jacobian_calls += 1
        return super().jacobian


def _build_metric(gamma: float) -> EmbeddingFisherMetric:
    model = SimpleNamespace(
        e={"m": torch.zeros(1, 2), "P": torch.eye(2).unsqueeze(0)},
        z={"m": torch.zeros(1, 1, 2), "P": torch.eye(2).unsqueeze(0).unsqueeze(0)},
        dynamics=SimpleNamespace(logvar=torch.nn.Parameter(torch.zeros(1, 2))),
        decoder=_DummyPoissonDecoder(d_obs=2, d_latent=2),
    )
    metric = EmbeddingFisherMetric(
        model=model,
        Fe_net=_fe_identity,
        Fz_net=_fz_zero,
        gamma=gamma,
        device="cpu",
    )
    # These tests isolate the discount/logdet algebra, not catalog-level
    # validation of mutually exclusive approximation flags.
    metric.no_sensitivity_propagation = True
    metric.fully_observed = True
    return metric


def test_discounted_eig_matches_closed_form_gamma_one() -> None:
    metric = _build_metric(gamma=1.0)
    rollout = {"model_state": torch.zeros(1, 3, 2), "next_model_state": torch.zeros(1, 3, 2)}
    value = float(metric.compute_stepwise(rollout).item())
    # With J_t = I and P_theta = I, EIG = 0.5 * 2 * log(1 + 3) = log(4).
    expected = -math.log(4.0)
    assert abs(value - expected) < 1e-6


def test_discounted_eig_uses_gamma_and_half_logdet_factor() -> None:
    metric = _build_metric(gamma=0.5)
    rollout = {"model_state": torch.zeros(1, 3, 2), "next_model_state": torch.zeros(1, 3, 2)}
    value = float(metric.compute_stepwise(rollout).item())
    # Sum_k gamma^(k-1) = 1 + 0.5 + 0.25 = 1.75, so EIG = 0.5 * 2 * log(1 + 1.75).
    expected = -math.log(2.75)
    assert abs(value - expected) < 1e-6

    metric_no_discount = _build_metric(gamma=1.0)
    no_discount_value = float(metric_no_discount.compute_stepwise(rollout).item())
    assert no_discount_value < value


def test_discounted_eig_reuses_decoder_terms_across_horizon() -> None:
    metric = _build_metric(gamma=1.0)
    rollout = {"model_state": torch.zeros(4, 3, 2), "next_model_state": torch.zeros(4, 3, 2)}

    metric.compute_stepwise(rollout)

    decoder = metric.model.decoder
    assert decoder.jacobian_calls == 1
    assert decoder.forward_calls == 1
    assert decoder.var_calls == 0

def test_eig_uses_exact_diagonal_observation_fisher() -> None:
    decoder = _WeightedPoissonDecoder()
    model = SimpleNamespace(
        e={"m": torch.zeros(1, 2), "P": torch.eye(2).unsqueeze(0)},
        z={"m": torch.zeros(1, 1, 2), "P": torch.eye(2).unsqueeze(0).unsqueeze(0)},
        dynamics=SimpleNamespace(logvar=torch.nn.Parameter(torch.zeros(1, 2))),
        decoder=decoder,
    )
    metric = EmbeddingFisherMetric(
        model=model,
        Fe_net=_fe_identity,
        Fz_net=_fz_zero,
        gamma=1.0,
        device="cpu",
    )
    metric.no_sensitivity_propagation = True
    metric.fully_observed = True

    value = metric.compute_stepwise({"model_state": torch.zeros(1, 1, 2), "next_model_state": torch.zeros(1, 1, 2)})

    I_z = decoder.H.T @ torch.diag(decoder.rate.reciprocal()) @ decoder.H
    expected = -0.5 * torch.logdet(torch.eye(2) + I_z)
    assert torch.allclose(value.reshape(()), expected, atol=1e-6, rtol=1e-6)
    assert decoder.jacobian_calls == 1
    assert decoder.forward_calls == 1
    assert decoder.var_calls == 0


def test_loglinear_poisson_eig_reuses_rate_without_jacobian() -> None:
    mapping = _CountingLogLinearMapping(latent_dim=2, obs_dim=3, dt=1.0, device="cpu")
    with torch.no_grad():
        mapping.network[0].weight.copy_(
            torch.tensor([[0.3, -0.2], [0.1, 0.4], [-0.5, 0.2]], dtype=torch.float32)
        )
        mapping.network[0].bias.copy_(torch.tensor([0.2, -0.1, 0.05], dtype=torch.float32))
    decoder = Decoder(mapping=mapping, noise=PoissonNoise(device="cpu"), device="cpu")
    model = SimpleNamespace(
        e={"m": torch.zeros(1, 2), "P": torch.eye(2).unsqueeze(0)},
        z={"m": torch.zeros(1, 1, 2), "P": torch.eye(2).unsqueeze(0).unsqueeze(0)},
        dynamics=SimpleNamespace(logvar=torch.nn.Parameter(torch.zeros(1, 2))),
        decoder=decoder,
    )
    metric = EmbeddingFisherMetric(
        model=model,
        Fe_net=_fe_identity,
        Fz_net=_fz_zero,
        gamma=1.0,
        device="cpu",
    )
    metric.no_sensitivity_propagation = True
    metric.fully_observed = True
    z = torch.tensor([[[0.25, -0.5]]], dtype=torch.float32)

    value = metric.compute_stepwise({"model_state": z, "next_model_state": z})

    rate = decoder(z).reshape(-1)
    W = mapping.network[0].weight.detach()
    I_z = W.T @ torch.diag(rate) @ W
    expected = -0.5 * torch.logdet(torch.eye(2) + I_z)
    assert torch.allclose(value.reshape(()), expected, atol=1e-6, rtol=1e-6)
    assert mapping.jacobian_calls == 0


@pytest.mark.parametrize("planning_rollout", ["prediction_only", "measurement_conditioned"])
def test_diagonal_covariance_ablation_is_finite_and_distinct(planning_rollout) -> None:
    def build(*, diagonal_covariance: bool) -> EmbeddingFisherMetric:
        model = SimpleNamespace(
            e={"m": torch.zeros(1, 2), "P": torch.eye(2).unsqueeze(0)},
            z={"m": torch.zeros(1, 1, 2), "P": torch.eye(2).unsqueeze(0).unsqueeze(0)},
            dynamics=SimpleNamespace(logvar=torch.nn.Parameter(torch.zeros(1, 2))),
            decoder=_DummyPoissonDecoder(d_obs=2, d_latent=2),
        )
        return EmbeddingFisherMetric(
            model=model,
            Fe_net=_fe_identity,
            Fz_net=_fz_shear,
            gamma=1.0,
            diagonal_covariance=diagonal_covariance,
            planning_rollout=planning_rollout,
            device="cpu",
        )

    rollout = {"model_state": torch.zeros(1, 4, 2), "next_model_state": torch.zeros(1, 4, 2)}
    full_value = build(diagonal_covariance=False).compute_stepwise(rollout)
    diagonal_value = build(diagonal_covariance=True).compute_stepwise(rollout)

    assert torch.isfinite(diagonal_value).all()
    assert not torch.allclose(diagonal_value, full_value)


@pytest.mark.parametrize("planning_rollout", ["prediction_only", "measurement_conditioned"])
def test_next_observation_information_matches_scalar_closed_form(planning_rollout) -> None:
    """A one-step score uses the destination rate and includes transition noise."""
    from actdyn.metrics.objectives import EOptimalityMetric

    mapping = LogLinearMapping(latent_dim=1, obs_dim=1, dt=1.0, device="cpu")
    with torch.no_grad():
        mapping.network[0].weight.fill_(1.0)
        mapping.network[0].bias.zero_()
    model = SimpleNamespace(
        e={"m": torch.zeros(1, 1), "P": torch.tensor([[[0.7]]])},
        z={"P": torch.tensor([[[0.4]]])},
        dt=1.0,
        dynamics=SimpleNamespace(logvar=torch.tensor([[math.log(math.expm1(0.3))]])),
        decoder=Decoder(mapping=mapping, noise=PoissonNoise(device="cpu"), device="cpu"),
    )

    def fe(z, e):
        return (2.0 + z).unsqueeze(-1)

    def fz(z, e):
        return (0.5 + z).unsqueeze(-1)

    # Both candidates start at zero, but end at states with rates 1 and 4.
    rollout = {
        "model_state": torch.zeros(2, 1, 1),
        "next_model_state": torch.tensor([[[0.0]], [[math.log(4.0)]]]),
    }
    rates = torch.tensor([1.0, 4.0])
    # Fz = 1.5, Ftheta = 2 at the source; P_next = 1.5^2 * 0.4 + 0.3.
    predicted_variance = 1.2
    for flag in (None, "diagonal_covariance", "freeze_covariance", "fully_observed"):
        kwargs = {} if flag is None else {flag: True}
        metric = EmbeddingFisherMetric(
            model=model, Fe_net=fe, Fz_net=fz, gamma=1.0, device="cpu",
            planning_rollout=planning_rollout, **kwargs
        )
        variance = 0.4 if flag == "freeze_covariance" else predicted_variance
        info = 4.0 * rates if flag == "fully_observed" else 4.0 * rates / (1.0 + variance * rates)
        expected = -0.5 * torch.log1p(0.7 * info)
        torch.testing.assert_close(metric.compute_stepwise(rollout).flatten(), expected)

    e_opt = EOptimalityMetric(
        model=model, Fe_net=fe, Fz_net=fz, gamma=1.0, device="cpu",
        planning_rollout=planning_rollout,
    )
    expected_e_opt = -0.7 * 4.0 * rates / (1.0 + predicted_variance * rates)
    torch.testing.assert_close(e_opt.compute_stepwise(rollout).flatten(), expected_e_opt)


@pytest.mark.parametrize(
    "planning_rollout,second_sensitivity,second_variance",
    [("prediction_only", 2.0, 2.0), ("measurement_conditioned", 1.4, 1.1)],
)
def test_rollout_covariance_and_sensitivity_match_two_step_closed_form(
    planning_rollout, second_sensitivity, second_variance,
) -> None:
    """Measurement conditioning changes both P and S before the second prediction."""
    model = SimpleNamespace(
        e={"m": torch.zeros(1, 2), "P": torch.eye(2).unsqueeze(0)},
        z={"P": torch.eye(2).unsqueeze(0)},
        dynamics=SimpleNamespace(logvar=torch.full((1, 2), math.log(math.expm1(0.5)))),
        decoder=_DummyPoissonDecoder(),
        dt=1.0,
    )
    metric = EmbeddingFisherMetric(
        model=model, Fe_net=_fe_identity, Fz_net=_fz_zero, gamma=0.5, device="cpu",
        planning_rollout=planning_rollout,
    )
    rollout = {"model_state": torch.zeros(1, 2, 2), "next_model_state": torch.zeros(1, 2, 2)}
    # S1-=I, P1-=1.5I. Conditioning gives S1+=0.4I, P1+=0.6I.
    expected = torch.tensor(-math.log1p(
        1.0 / 2.5 + 0.5 * second_sensitivity**2 / (1.0 + second_variance)
    ))
    torch.testing.assert_close(metric.compute_stepwise(rollout).reshape(()), expected)


def test_contraction_matches_noncommuting_gaussian_update() -> None:
    from actdyn.utils.torch_utils import posterior_state_covariance
    P = torch.tensor([[2., .4], [.4, .7]], dtype=torch.float64)
    H = torch.tensor([[1., .3], [-.2, 2.]], dtype=torch.float64)
    R = torch.diag(torch.tensor([.6, 1.2], dtype=torch.float64))
    I_z = H.T @ torch.linalg.solve(R, H)
    expected = P - P @ H.T @ torch.linalg.solve(H @ P @ H.T + R, H @ P)
    actual = posterior_state_covariance(P, I_z)
    torch.testing.assert_close(actual, expected)
    assert torch.linalg.eigvalsh(P-actual).min() >= -1e-10
    torch.testing.assert_close(posterior_state_covariance(P, torch.zeros_like(P)), P)


@pytest.mark.parametrize(
    "planning_rollout,second_sensitivity,second_variance",
    [("prediction_only", 2.0, 2.0), ("measurement_conditioned", 1.4, 1.1)],
)
def test_eoptimality_matches_two_step_closed_form(
    planning_rollout, second_sensitivity, second_variance,
) -> None:
    from actdyn.metrics.objectives import EOptimalityMetric
    model = SimpleNamespace(
        e={"m": torch.zeros(1, 2), "P": torch.eye(2).unsqueeze(0)},
        z={"P": torch.eye(2).unsqueeze(0)},
        dynamics=SimpleNamespace(logvar=torch.full((1, 2), math.log(math.expm1(.5)))),
        decoder=_DummyPoissonDecoder(), dt=1.,
    )
    metric = EOptimalityMetric(
        model=model, Fe_net=_fe_identity, Fz_net=_fz_zero, gamma=.5, device="cpu",
        planning_rollout=planning_rollout,
    )
    rollout = {"model_state":torch.zeros(1,2,2), "next_model_state":torch.zeros(1,2,2)}
    before = model.z['P'].clone()
    expected = torch.tensor(-(1 / 2.5 + .5 * second_sensitivity**2 / (1 + second_variance)))
    torch.testing.assert_close(metric.compute_stepwise(rollout).reshape(()), expected)
    torch.testing.assert_close(model.z['P'], before)
