from __future__ import annotations

import math
from types import SimpleNamespace

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
    rollout = {"model_state": torch.zeros(1, 3, 2)}
    value = float(metric.compute_stepwise(rollout).item())
    # With J_t = I and P_theta = I, EIG = 0.5 * 2 * log(1 + 3) = log(4).
    expected = -math.log(4.0)
    assert abs(value - expected) < 1e-6


def test_discounted_eig_uses_gamma_and_half_logdet_factor() -> None:
    metric = _build_metric(gamma=0.5)
    rollout = {"model_state": torch.zeros(1, 3, 2)}
    value = float(metric.compute_stepwise(rollout).item())
    # Sum_k gamma^(k-1) = 1 + 0.5 + 0.25 = 1.75, so EIG = 0.5 * 2 * log(1 + 1.75).
    expected = -math.log(2.75)
    assert abs(value - expected) < 1e-6

    metric_no_discount = _build_metric(gamma=1.0)
    no_discount_value = float(metric_no_discount.compute_stepwise(rollout).item())
    assert no_discount_value < value


def test_discounted_eig_reuses_decoder_terms_across_horizon() -> None:
    metric = _build_metric(gamma=1.0)
    rollout = {"model_state": torch.zeros(4, 3, 2)}

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

    value = metric.compute_stepwise({"model_state": torch.zeros(1, 1, 2)})

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

    value = metric.compute_stepwise({"model_state": z})

    rate = decoder(z).reshape(-1)
    W = mapping.network[0].weight.detach()
    I_z = W.T @ torch.diag(rate) @ W
    expected = -0.5 * torch.logdet(torch.eye(2) + I_z)
    assert torch.allclose(value.reshape(()), expected, atol=1e-6, rtol=1e-6)
    assert mapping.jacobian_calls == 0


def test_diagonal_covariance_ablation_is_finite_and_distinct() -> None:
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
            device="cpu",
        )

    rollout = {"model_state": torch.zeros(1, 4, 2)}
    full_value = build(diagonal_covariance=False).compute_stepwise(rollout)
    diagonal_value = build(diagonal_covariance=True).compute_stepwise(rollout)

    assert torch.isfinite(diagonal_value).all()
    assert not torch.allclose(diagonal_value, full_value)
