from __future__ import annotations

import torch

from actdyn.models.decoder import Decoder, GaussianNoise, LinearMapping
from actdyn.models.dynamics import FunctionDynamics
from actdyn.models.model import FilteringEmbedding


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


def _zero_dynamics(z: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(z)


def _build_model(
    q_theta: float = 1e-4,
    k_theta: int = 2,
    *,
    adaptive_update: bool = False,
    adaptive_update_min_interval: int = 1,
    adaptive_update_eig_threshold: float | None = None,
) -> FilteringEmbedding:
    dynamics = FunctionDynamics(state_dim=2, dynamics_fn=_zero_dynamics, dt=0.1, device="cpu")
    dynamics.logvar = torch.nn.Parameter(torch.log(torch.ones(1, 2) * 0.01))

    mapping = LinearMapping(latent_dim=2, obs_dim=2, device="cpu")
    with torch.no_grad():
        mapping.network.weight.copy_(torch.eye(2))
        mapping.network.bias.zero_()
    decoder = Decoder(mapping=mapping, noise=GaussianNoise(obs_dim=2, sigma=0.1, device="cpu"))

    sigma0 = 0.1
    e_bel = {
        "m": torch.zeros(1, 2),
        "P": sigma0 * torch.eye(2).unsqueeze(0),
        "L": (1.0 / sigma0) * torch.eye(2).unsqueeze(0),
    }
    model = FilteringEmbedding(
        dynamics=dynamics,
        decoder=decoder,
        e=e_bel,
        Fe=_fe_identity,
        Fz=_fz_zero,
        q_theta=q_theta,
        k_theta=k_theta,
        adaptive_update=adaptive_update,
        adaptive_update_min_interval=adaptive_update_min_interval,
        adaptive_update_eig_threshold=adaptive_update_eig_threshold,
        device="cpu",
    )
    # Keep prediction deterministic so only filtering updates influence parameter belief.
    model.predict = lambda action=None: model.z["m"]
    model.z = {
        "m": torch.zeros(1, 1, 2),
        "P": torch.eye(2).unsqueeze(0).unsqueeze(0),
    }
    return model


def test_block_update_applies_only_every_k_theta_steps() -> None:
    model = _build_model(q_theta=1e-4, k_theta=3)
    y = torch.tensor([[[1.0, -0.5]]], dtype=torch.float32)
    u = torch.zeros(1, 1, 2)
    before = model.e["m"].clone()

    model.update_posterior_embedding(y, u)
    model.update_posterior_embedding(y, u)
    assert torch.allclose(model.e["m"], before)

    model.update_posterior_embedding(y, u)
    assert not torch.allclose(model.e["m"], before)


def test_adaptive_block_update_can_apply_before_k_theta() -> None:
    model = _build_model(
        q_theta=1e-4,
        k_theta=5,
        adaptive_update=True,
        adaptive_update_min_interval=2,
        adaptive_update_eig_threshold=0.0,
    )
    y = torch.tensor([[[1.0, -0.5]]], dtype=torch.float32)
    u = torch.zeros(1, 1, 2)
    before = model.e["m"].clone()

    model.update_posterior_embedding(y, u)
    assert torch.allclose(model.e["m"], before)

    model.update_posterior_embedding(y, u)
    assert not torch.allclose(model.e["m"], before)
    assert model.last_information["parameter_update_reason"] == "block_eig"
    assert model._last_theta_block_steps_applied == 2


def test_q_theta_drift_increases_predictive_parameter_covariance() -> None:
    model = _build_model(q_theta=0.05, k_theta=1)
    p_before = model.e["P"].clone()
    m_before = model.e["m"].clone()
    model._theta_score_block.zero_()
    model._theta_info_block.zero_()

    model._apply_embedding_block_update()

    p_after = model.e["P"]
    expected_diag = torch.full((2,), 0.05)
    observed_diag = torch.diagonal((p_after - p_before)[0])
    assert torch.allclose(observed_diag, expected_diag, atol=1e-6)
    assert torch.allclose(model.e["m"], m_before)


def test_embedding_belief_remains_finite_and_spd_after_updates() -> None:
    model = _build_model(q_theta=1e-4, k_theta=2)
    u = torch.zeros(1, 1, 2)
    for _ in range(6):
        y = torch.randn(1, 1, 2) * 0.2
        model.update_posterior_embedding(y, u)
        assert torch.isfinite(model.e["m"]).all()
        assert torch.isfinite(model.e["P"]).all()
        assert torch.isfinite(model.e["L"]).all()
        eigvals = torch.linalg.eigvalsh(model.e["P"][0])
        assert torch.all(eigvals > 0.0)
