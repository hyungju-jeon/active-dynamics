from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from actdyn.metrics.information import EmbeddingFisherMetric
from actdyn.models.decoder import PoissonNoise


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


class _DummyPoissonDecoder:
    def __init__(self, d_obs: int = 2, d_latent: int = 2):
        self.d_obs = d_obs
        self.d_latent = d_latent
        self.noise = PoissonNoise(device="cpu")

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        return torch.ones(*z.shape[:-1], self.d_obs, device=z.device)

    def jacobian(self, z: torch.Tensor) -> torch.Tensor:
        batch, time, _, = z.shape
        eye = torch.eye(self.d_obs, self.d_latent, device=z.device).view(
            1, 1, self.d_obs, self.d_latent
        )
        return eye.expand(batch, time, -1, -1)

    def var(self, z: torch.Tensor) -> torch.Tensor:
        return torch.ones(*z.shape[:-1], self.d_obs, device=z.device)


def _build_metric(gamma: float) -> EmbeddingFisherMetric:
    model = SimpleNamespace(
        e={"m": torch.zeros(1, 2), "P": torch.eye(2).unsqueeze(0)},
        z={"m": torch.zeros(1, 1, 2), "P": torch.eye(2).unsqueeze(0).unsqueeze(0)},
        dynamics=SimpleNamespace(logvar=torch.nn.Parameter(torch.zeros(1, 2))),
        decoder=_DummyPoissonDecoder(d_obs=2, d_latent=2),
    )
    return EmbeddingFisherMetric(
        model=model,
        Fe_net=_fe_identity,
        Fz_net=_fz_zero,
        gamma=gamma,
        device="cpu",
    )


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
