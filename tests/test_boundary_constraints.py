from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from actdyn.environment.boundary import boundary_visibility, project_to_boundary
from actdyn.environment.vectorfield import VectorFieldEnv
from actdyn.metrics.information import EmbeddingFisherMetric
from actdyn.models.dynamics import FunctionDynamics


def test_radial_boundary_visibility_decays_near_boundary():
    center = torch.zeros(1, 2)
    boundary = torch.tensor([[3.95, 0.0]])

    v_center = boundary_visibility(
        center,
        boundary_type="radial",
        radius=4.0,
        margin=1.0,
        temperature=0.15,
    )
    v_boundary = boundary_visibility(
        boundary,
        boundary_type="radial",
        radius=4.0,
        margin=1.0,
        temperature=0.15,
    )

    assert float(v_center.item()) > 0.99
    assert float(v_boundary.item()) < 0.01


def test_project_to_radial_boundary_clamps_norm():
    z = torch.tensor([[3.0, 4.0], [0.5, 0.0]])
    projected = project_to_boundary(z, boundary_type="radial", radius=2.0)

    assert torch.linalg.norm(projected, dim=-1).max().item() <= 2.0 + 1e-5
    assert torch.allclose(projected[1], z[1])


def test_vectorfield_env_step_cannot_leave_radial_boundary():
    env = VectorFieldEnv(
        "double_integrator",
        dt=1.0,
        Q=0.0,
        dyn_params=[0.0],
        initial_state=[0.0, 0.0],
        boundary_enabled=True,
        boundary_type="radial",
        boundary_radius=1.0,
        boundary_projection_enabled=True,
    )

    state, *_ = env.step(torch.tensor([100.0, 100.0]))

    assert torch.linalg.norm(state.float()).item() <= 1.0 + 1e-5


def test_vectorfield_env_separates_action_and_state_space_dimensions():
    env = VectorFieldEnv(
        "confounded_gate",
        d_state=3,
        d_action=2,
        Q=0.0,
        dyn_params=[0.5],
        action_bounds=[[-1.0, -0.5], [1.0, 0.5]],
        initial_state=[-0.5, 0.0, 0.25],
    )

    assert env.action_space.shape == (2,)
    assert env.observation_space.shape == (3,)

    state, *_ = env.step(torch.tensor([0.0, 0.0, 0.0]))
    assert state.shape == (3,)


def test_vectorfield_env_accepts_batched_single_parameter_values():
    env = VectorFieldEnv("confounded_gate", d_state=3, dyn_params=[0.5])
    dynamics = FunctionDynamics(
        state_dim=3,
        dynamics_fn=env,
        param_formatter=lambda params: params,
    )
    dynamics.set_params(torch.tensor([[0.25], [0.75]]))

    states = torch.tensor([[-0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]])
    drift = env.compute_dynamics(states)

    assert drift.shape == (2, 3)
    assert drift[0, 1] == pytest.approx(5.0, abs=1e-3)
    assert drift[1, 1] == pytest.approx(15.0, abs=1e-3)


class _LinearDecoder:
    noise = object()

    def jacobian(self, z: torch.Tensor) -> torch.Tensor:
        batch = z.shape[0]
        eye = torch.eye(2, dtype=z.dtype, device=z.device)
        return eye.reshape(1, 1, 2, 2).expand(batch, 1, 2, 2)

    def var(self, z: torch.Tensor) -> torch.Tensor:
        return torch.ones(z.shape[0], 1, 2, dtype=z.dtype, device=z.device)


def test_embedding_fisher_metric_visibility_reduces_boundary_eig():
    model = SimpleNamespace(
        e={
            "m": torch.zeros(1, 1),
            "P": torch.eye(1).unsqueeze(0),
        },
        z={"P": torch.eye(2).unsqueeze(0)},
        dynamics=SimpleNamespace(logvar=torch.log(torch.full((1, 2), 1e-4))),
        decoder=_LinearDecoder(),
        dt=1.0,
    )

    def fe_net(z: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return torch.ones(*z.shape[:-1], 2, 1, dtype=z.dtype, device=z.device)

    def fz_net(z: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(2, dtype=z.dtype, device=z.device)
        return eye.reshape(1, 1, 2, 2).expand(z.shape[0], z.shape[1], 2, 2)

    metric = EmbeddingFisherMetric(
        model=model,
        Fe_net=fe_net,
        Fz_net=fz_net,
        gamma=1.0,
        device="cpu",
        boundary_visibility_enabled=True,
        boundary_type="radial",
        boundary_radius=4.0,
        boundary_margin=1.0,
        boundary_temperature=0.15,
    )

    interior_eig = -metric.compute_stepwise({"model_state": torch.zeros(1, 3, 2)})
    boundary_eig = -metric.compute_stepwise({"model_state": torch.full((1, 3, 2), 3.95)})

    assert float(boundary_eig.item()) < float(interior_eig.item())
