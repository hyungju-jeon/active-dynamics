from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, rel_path: str):
    module_path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _IdentityAction:
    action_dim = 2
    d_action = 2

    def __call__(self, action, state=None):
        return action


class _DummyDecoder:
    obs_dim = 2
    latent_dim = 2

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        return torch.ones(z.shape[0], z.shape[1], 2, dtype=z.dtype, device=z.device)

    def jacobian(self, z: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(2, dtype=z.dtype, device=z.device).reshape(1, 1, 2, 2)
        return eye.expand(z.shape[0], z.shape[1], -1, -1)

    def var(self, z: torch.Tensor) -> torch.Tensor:
        return torch.ones(z.shape[0], z.shape[1], 2, dtype=z.dtype, device=z.device)


def _make_model(*, support_radius: int = 1):
    module = _load_module("cosyne_rbf_filtering_test_module", "experiments/cosyne/rbf_filtering.py")
    axis = torch.linspace(-1.0, 1.0, 3)
    gx, gy = torch.meshgrid(axis, axis, indexing="ij")
    centers = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    dynamics = module.SparseRbfDynamics(
        state_dim=2,
        centers=centers,
        axis=axis,
        width=float(axis[1] - axis[0]),
        support_radius=support_radius,
        dt=0.1,
        is_residual=True,
        device="cpu",
    )
    d_embed = int(centers.shape[0] * 2)
    e_bel = {
        "m": torch.zeros(1, d_embed),
        "P": torch.diag_embed(torch.ones(1, d_embed)),
    }
    model = module.SparseRbfFilteringModel(
        dynamics=dynamics,
        decoder=_DummyDecoder(),
        action_encoder=_IdentityAction(),
        e=e_bel,
        q_theta=1e-4,
        k_theta=1,
        state_init_uncertainty=2.0,
        device="cpu",
    )
    return module, model


def test_sparse_rbf_metric_returns_finite_costs():
    module, model = _make_model()
    metric = module.StructuredLocalRbfParameterMetric(model=model, gamma=1.0, device="cpu")
    rollout = {"model_state": torch.zeros(24, 5, 2)}
    costs = metric(rollout)
    assert costs.shape == (24,)
    assert torch.isfinite(costs).all()


def test_sparse_rbf_filter_update_is_finite_and_updates_beliefs():
    module, model = _make_model()
    observation = torch.zeros(1, 1, 2)
    model.reset(observation)
    prev_var = torch.diagonal(model.e["P"], dim1=-2, dim2=-1).clone()
    posterior = model.update_posterior_embedding(
        y=torch.ones(1, 1, 2),
        u=torch.zeros(1, 1, 2),
    )
    assert posterior.shape == (1, 1, 2)
    assert torch.isfinite(model.z["P"]).all()
    assert torch.isfinite(model.e["m"]).all()
    assert torch.isfinite(model.e["P"]).all()
    assert model._theta_block_steps == 0
    assert torch.any(torch.diagonal(model.e["P"], dim1=-2, dim2=-1) < prev_var)
    assert torch.isfinite(torch.as_tensor(model.last_information["I_theta_t"]))
    off_structure = model.e["P"] * (~model.dynamics.parameter_structure_mask).to(model.e["P"].dtype)
    assert torch.allclose(off_structure, torch.zeros_like(off_structure), atol=1e-6)


def test_sparse_rbf_metric_uses_rollout_belief_location_for_local_covariance():
    module, model = _make_model(support_radius=0)
    metric = module.StructuredLocalRbfParameterMetric(model=model, gamma=1.0, device="cpu")
    diag = torch.full((model.e["m"].shape[-1],), 1e-4, dtype=torch.float32)

    def _block_for_state(state_xy: tuple[float, float]) -> torch.Tensor:
        state = torch.tensor(state_xy, dtype=torch.float32).reshape(1, 1, 2)
        _, _, active_centers = model.dynamics.local_jacobians(state)
        active_idx = model.dynamics.parameter_indices_for_centers(active_centers.reshape(-1))
        return model.dynamics.expand_parameter_indices(active_idx)

    low_var_block = _block_for_state((-1.0, -1.0))
    high_var_block = _block_for_state((1.0, 1.0))
    diag[low_var_block] = 1e-3
    diag[high_var_block] = 10.0
    model.e["P"] = torch.diag_embed(diag.unsqueeze(0))
    model._normalize_embedding_belief()
    model.set_state(torch.tensor([[[-1.0, -1.0]]], dtype=torch.float32))

    low_rollout = {"model_state": torch.tensor([[[-1.0, -1.0]]], dtype=torch.float32)}
    high_rollout = {"model_state": torch.tensor([[[1.0, 1.0]]], dtype=torch.float32)}

    low_cost = float(metric(low_rollout).item())
    high_cost = float(metric(high_rollout).item())

    assert high_cost < low_cost
