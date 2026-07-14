"""FLEX policy wrapper for the original FLEX implementation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from .base import BasePolicy

REPO_ROOT = Path(__file__).resolve().parents[2]
FLEX_ROOT = REPO_ROOT / "external" / "FLEX"


def _load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name!r} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_flex_policy():
    official_root = str(FLEX_ROOT)
    inserted_root = False
    if official_root not in sys.path:
        sys.path.insert(0, official_root)
        inserted_root = True
    saved_modules = {name: sys.modules.get(name) for name in ("agent", "computations")}
    try:
        for name in saved_modules:
            sys.modules.pop(name, None)
        official_policies = _load_module_from_path(
            "_flex_policies",
            FLEX_ROOT / "policies.py",
        )
    finally:
        for name in ("agent", "computations"):
            sys.modules.pop(name, None)
        for name, module in saved_modules.items():
            if module is not None:
                sys.modules[name] = module
        if inserted_root:
            sys.path.remove(official_root)
    return official_policies.Flex


_FLEX_POLICY_CLASS = None


def _flex_policy_class():
    global _FLEX_POLICY_CLASS
    if _FLEX_POLICY_CLASS is None:
        _FLEX_POLICY_CLASS = _load_flex_policy()
    return _FLEX_POLICY_CLASS


class _FlexModelBase(nn.Module):
    def __init__(
        self,
        *,
        dt: float,
        latent_dim: int,
        action_dim: int,
    ) -> None:
        super().__init__()
        self.t_period = 1.0 / float(dt)
        self.period = 1.0
        self.B_star = torch.tensor(
            np.eye(int(latent_dim), int(action_dim), dtype=np.float64), dtype=torch.float
        )
        self.d = int(latent_dim)
        self.m = int(action_dim)
        self.evaluation = None


class _FlexVectorFieldModel(_FlexModelBase):
    def __init__(
        self,
        *,
        dt: float,
        dynamics_alpha: float,
        latent_dim: int,
        action_dim: int,
        initial_embedding: np.ndarray,
        fixed_tail: np.ndarray,
        lr: float | None = None,
    ):
        super().__init__(dt=dt, latent_dim=latent_dim, action_dim=action_dim)
        self.alpha = float(dynamics_alpha)
        init = np.asarray(initial_embedding, dtype=np.float32).reshape(-1)
        self.learnable = nn.ParameterList(
            [nn.Parameter(torch.tensor(float(v), dtype=torch.float32)) for v in init.tolist()]
        )
        tail = np.asarray(fixed_tail, dtype=np.float32).reshape(-1)
        self.register_buffer("_fixed_tail", torch.as_tensor(tail, dtype=torch.float32))
        self.linear = False
        if lr is not None:
            # Official FLEX switches from OLS to Adam when model.lr is present.
            self.lr = float(lr)

    def parameter_vector(self) -> np.ndarray:
        if len(self.learnable) == 0:
            return np.zeros(0, dtype=np.float32)
        return np.asarray([float(p.detach().item()) for p in self.learnable], dtype=np.float32)

    def get_B(self, x):
        return np.eye(self.d, self.m, dtype=np.float64)

    def _full_params(self, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if len(self.learnable) > 0:
            learnable = torch.stack([p for p in self.learnable]).to(device=device, dtype=dtype)
        else:
            learnable = torch.zeros(0, device=device, dtype=dtype)
        if int(self._fixed_tail.numel()) == 0:
            return learnable
        return torch.cat((learnable, self._fixed_tail.to(device=device, dtype=dtype)), dim=0)


class FlexDampedPendulumModel(_FlexVectorFieldModel):
    def forward(self, z):
        params = self._full_params(dtype=z.dtype, device=z.device)
        damping, gravity = params[0], params[1]
        theta = z[:, 0]
        omega = z[:, 1]
        u = z[:, 2:4]
        drift = torch.stack((omega, damping * omega - gravity * torch.sin(theta)), dim=1)
        return self.alpha * drift + u


class FlexDuffingModel(_FlexVectorFieldModel):
    def forward(self, z):
        params = self._full_params(dtype=z.dtype, device=z.device)
        a, b, c = params[0], params[1], params[2]
        x0 = z[:, 0]
        x1 = z[:, 1]
        u = z[:, 2:4]
        drift = torch.stack((x1, a * x1 - x0 * (b + c * x0.square())), dim=1)
        return self.alpha * drift + u


class FlexAsymmetricBasinModel(_FlexVectorFieldModel):
    def __init__(self, *args, gate_sharpness: float = 3.0, cubic: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate_sharpness = float(gate_sharpness)
        self.cubic = float(cubic)

    def forward(self, z):
        params = self._full_params(dtype=z.dtype, device=z.device)
        a_left, b_left, a_right, b_right = params[0], params[1], params[2], params[3]
        x0 = z[:, 0]
        x1 = z[:, 1]
        u = z[:, 2:4]
        gate = torch.sigmoid(self.gate_sharpness * x0)
        a_eff = (1.0 - gate) * a_left + gate * a_right
        b_eff = (1.0 - gate) * b_left + gate * b_right
        drift = torch.stack((x1, a_eff * x1 - b_eff * x0 - self.cubic * x0.pow(3)), dim=1)
        return self.alpha * drift + u


class FlexMultiStableModel(_FlexVectorFieldModel):
    def __init__(self, *args, sigma: float = 1.0, center_scale: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma = float(sigma)
        self.register_buffer(
            "_centers",
            torch.tensor(
                [
                    [-center_scale, center_scale],
                    [center_scale, center_scale],
                    [-center_scale, -center_scale],
                    [center_scale, -center_scale],
                ],
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "_rotations",
            torch.tensor([1.55, -0.2, -0.4, -2.0], dtype=torch.float32),
        )

    def forward(self, z):
        params = self._full_params(dtype=z.dtype, device=z.device)
        x = z[:, :2]
        u = z[:, 2:4]
        centers = self._centers.to(device=z.device, dtype=z.dtype)
        disp = x.unsqueeze(-2) - centers
        r2 = torch.sum(disp * disp, dim=-1)
        sigma2 = max(self.sigma**2, 1e-6)
        envelope = torch.exp(-0.5 * r2 / sigma2)
        amplitudes = params.unsqueeze(0).expand(x.shape[0], -1)
        rotations = (
            self._rotations.to(device=z.device, dtype=z.dtype).unsqueeze(0).expand(x.shape[0], -1)
        )
        tangent = torch.stack((-disp[..., 1], disp[..., 0]), dim=-1)
        local_field = -amplitudes.unsqueeze(-1) * disp + rotations.unsqueeze(-1) * tangent
        field = torch.sum(envelope.unsqueeze(-1) * local_field, dim=-2)
        return self.alpha * field + u


def build_flex_model(
    *,
    env_preset: Any,
    initial_embedding: torch.Tensor | np.ndarray,
    lr: float | None = None,
):
    embedding = torch.as_tensor(initial_embedding, dtype=torch.float32).reshape(-1).cpu().numpy()
    full_params = np.asarray(env_preset.resolved_true_params(estimator=True), dtype=np.float32)
    fixed_tail = full_params[embedding.shape[0] :]
    kwargs = dict(
        dt=float(env_preset.dt),
        dynamics_alpha=float(env_preset.dynamics_alpha),
        latent_dim=int(env_preset.latent_dim),
        action_dim=int(env_preset.action_dim),
        initial_embedding=embedding,
        fixed_tail=fixed_tail,
        lr=lr,
    )
    dynamics_type = str(env_preset.resolved_dynamics_type(estimator=True))
    if dynamics_type == "damped_pendulum":
        return FlexDampedPendulumModel(**kwargs)
    if dynamics_type == "duffing":
        return FlexDuffingModel(**kwargs)
    if dynamics_type == "gated_duffing":
        return FlexAsymmetricBasinModel(**kwargs)
    if dynamics_type == "multi_stable":
        return FlexMultiStableModel(**kwargs)
    raise ValueError(f"Official FLEX wrapper does not support dynamics_type={dynamics_type!r}")


class FLEXPolicy(BasePolicy):
    owns_parameter_estimate = True

    def __init__(
        self,
        *,
        action_space: gym.Space,
        model: Any,
        env_preset: Any,
        initial_parameter_mean: torch.Tensor | None = None,
        use_observed_state: bool = False,
        regularization: float = 1e-2,
        parameter_step_clip: float | None = 0.5,
        parameter_min: float | None = -5.0,
        parameter_max: float | None = 5.0,
        lr: float | None = None,
        rollback_unstable_update: bool = False,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(action_space=action_space, chunk=1, device=device)
        self.model = model
        self.env_preset = env_preset
        self.rollback_unstable_update = bool(rollback_unstable_update)
        self.use_observed_state = bool(use_observed_state)
        self.regularization = float(regularization)
        self.parameter_step_clip = (
            None if parameter_step_clip is None else float(parameter_step_clip)
        )
        self.parameter_min = None if parameter_min is None else float(parameter_min)
        self.parameter_max = None if parameter_max is None else float(parameter_max)
        self.lr = None if lr is None else float(lr)
        if initial_parameter_mean is None:
            model_e = getattr(model, "e", {})
            initial_parameter_mean = model_e.get("m")
        if initial_parameter_mean is None:
            initial_parameter_mean = torch.as_tensor(
                env_preset.true_embedding_vector(
                    embedding_dim=int(env_preset.embedding_dim), estimator=True
                ),
                dtype=torch.float32,
            ).unsqueeze(0)
        init_mean = torch.as_tensor(initial_parameter_mean, dtype=torch.float32, device=self.device)
        if init_mean.dim() == 1:
            init_mean = init_mean.unsqueeze(0)
        self._initial_parameter_mean = init_mean[:1].detach().clone()
        self._flex_model = None
        self._flex_agent = None
        self.last_update_info = {
            "parameter_posterior_updated": False,
            "flex_residual_norm": 0.0,
            "flex_update_norm": 0.0,
            "flex_update_rejected": False,
            "flex_gram_trace": 0.0,
        }
        self.reset_policy_state(seed=None)

    def _state_for_control(self, filtered_state: Any, true_state: Any) -> Any:
        return true_state if self.use_observed_state and true_state is not None else filtered_state

    def _state_keys_for_update(self) -> tuple[str, str]:
        if self.use_observed_state:
            return "env_state", "next_env_state"
        return "model_state", "next_model_state"

    def _build_agent(self, *, seed: int | None) -> None:
        if seed is not None:
            np.random.seed(int(seed))
        self._flex_model = build_flex_model(
            env_preset=self.env_preset,
            initial_embedding=self._initial_parameter_mean.reshape(-1),
            lr=self.lr,
        )
        flex_cls = _flex_policy_class()
        self._flex_agent = flex_cls(
            self._flex_model,
            int(self.env_preset.latent_dim),
            int(self.env_preset.action_dim),
            float(self.env_preset.action_max),
            dt=float(self.env_preset.dt),
            regularization=self.regularization,
        )

    def reset_policy_state(self, seed: int | None = None) -> None:
        self.count = 0
        self.action_list = []
        self.cost = 0.0
        self._build_agent(seed=seed)
        self.last_update_info = {
            "parameter_posterior_updated": False,
            "flex_residual_norm": 0.0,
            "flex_update_norm": 0.0,
            "flex_update_rejected": False,
            "flex_gram_trace": float(np.trace(self._flex_agent.M)),
        }

    def get_parameter_mean(self) -> torch.Tensor:
        assert self._flex_model is not None
        params = torch.as_tensor(
            self._flex_model.parameter_vector(), dtype=torch.float32, device=self.device
        )
        return params.unsqueeze(0)

    def get_parameter_covariance(self) -> torch.Tensor:
        assert self._flex_agent is not None
        return torch.as_tensor(
            self._flex_agent.M_inv, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

    def get_parameter_precision(self) -> torch.Tensor:
        assert self._flex_agent is not None
        return torch.as_tensor(
            self._flex_agent.M, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

    def _flex_parameter_vector(self) -> torch.Tensor:
        assert self._flex_model is not None
        return torch.as_tensor(
            self._flex_model.parameter_vector(), dtype=torch.float32, device=self.device
        )

    def _extract_last_tensor(self, rollout: Any, key: str) -> torch.Tensor | None:
        value = rollout.get(key, None)
        if value is None:
            return None
        tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if tensor.dim() == 1:
            tensor = tensor.view(1, 1, -1)
        elif tensor.dim() == 2:
            tensor = tensor.unsqueeze(1)
        return tensor[:, -1:, :]

    def get_action(self, state: torch.Tensor, **kwargs):
        assert self._flex_agent is not None
        observed_state = kwargs.get("observed_state")
        state_for_control = self._state_for_control(state, observed_state)
        x = (
            torch.as_tensor(state_for_control, dtype=torch.float32, device=self.device)
            .reshape(-1)
            .detach()
            .cpu()
            .numpy()
        )
        t = int(self.count)
        self.count += 1
        u = np.asarray(self._flex_agent.policy(x, t), dtype=np.float32).reshape(1, 1, -1)
        action = torch.as_tensor(u, dtype=torch.float32, device=self.device)
        return action, torch.zeros((), dtype=torch.float32, device=self.device)

    def update(self, rollout: Any):
        assert self._flex_agent is not None
        prev_mean = self._flex_parameter_vector()
        x_key, x_next_key = self._state_keys_for_update()
        x_t = self._extract_last_tensor(rollout, x_key)
        x_next = self._extract_last_tensor(rollout, x_next_key)
        if x_t is None or x_next is None:
            x_t = self._extract_last_tensor(rollout, "env_state")
            x_next = self._extract_last_tensor(rollout, "next_env_state")
        played_action = self._extract_last_tensor(rollout, "env_action")
        if x_t is None or x_next is None or played_action is None:
            info = {
                "parameter_posterior_updated": False,
                "flex_residual_norm": 0.0,
                "flex_update_norm": 0.0,
                "flex_update_rejected": False,
                "flex_gram_trace": float(np.trace(self._flex_agent.M)),
            }
            self.last_update_info = info
            return info
        x = x_t.reshape(-1).detach().cpu().numpy().astype(np.float64, copy=False)
        u = played_action.reshape(-1).detach().cpu().numpy().astype(np.float64, copy=False)
        dx_dt = (
            ((x_next - x_t) / float(self.env_preset.dt))
            .reshape(-1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )
        old_parameters = [param.detach().clone() for param in self._flex_model.parameters()]
        old_m = np.asarray(self._flex_agent.M).copy()
        old_m_inv = np.asarray(self._flex_agent.M_inv).copy()
        self._flex_agent.learning_step(x, u, dx_dt)
        rejected = self.rollback_unstable_update and self._unstable_update(prev_mean)
        if rejected:
            self._restore_flex_state(old_parameters, old_m, old_m_inv)
        else:
            self._stabilize_parameter_update(prev_mean)
        new_mean = self._flex_parameter_vector()
        update_norm = float(torch.linalg.norm(new_mean - prev_mean).item())
        info = {
            "parameter_posterior_updated": not rejected,
            "flex_residual_norm": 0.0,
            "flex_update_norm": update_norm,
            "flex_update_rejected": rejected,
            "flex_gram_trace": float(np.trace(self._flex_agent.M)),
        }
        self.last_update_info = info
        return info

    def _unstable_update(self, previous_mean: torch.Tensor) -> bool:
        current = self._flex_parameter_vector().reshape(-1)
        previous = previous_mean.detach().to(current.device).reshape(-1)
        delta = current - previous
        norm = torch.linalg.norm(delta)
        if not torch.isfinite(norm) or not torch.all(torch.isfinite(current)):
            return True
        if self.parameter_step_clip is not None and float(norm.item()) > self.parameter_step_clip:
            return True
        if self.parameter_min is not None and bool(torch.any(current < self.parameter_min)):
            return True
        if self.parameter_max is not None and bool(torch.any(current > self.parameter_max)):
            return True
        return False

    def _restore_flex_state(
        self,
        parameters: list[torch.Tensor],
        gram: np.ndarray,
        gram_inv: np.ndarray,
    ) -> None:
        assert self._flex_model is not None
        assert self._flex_agent is not None
        for param, old_value in zip(self._flex_model.parameters(), parameters):
            param.data.copy_(old_value.to(device=param.device, dtype=param.dtype))
        self._flex_agent.M = gram
        self._flex_agent.M_inv = gram_inv

    def _stabilize_parameter_update(self, previous_mean: torch.Tensor) -> None:
        assert self._flex_model is not None
        current = self.get_parameter_mean().reshape(-1)
        target = current.detach().clone()
        if self.parameter_step_clip is not None and self.parameter_step_clip > 0:
            delta = target - previous_mean.detach().to(target.device)
            norm = torch.linalg.norm(delta)
            if torch.isfinite(norm) and float(norm.item()) > self.parameter_step_clip:
                target = previous_mean.detach().to(target.device) + delta * (
                    self.parameter_step_clip / float(norm.item())
                )
        if self.parameter_min is not None or self.parameter_max is not None:
            lo = -torch.inf if self.parameter_min is None else self.parameter_min
            hi = torch.inf if self.parameter_max is None else self.parameter_max
            target = torch.clamp(target, min=lo, max=hi)
        for param, value in zip(self._flex_model.learnable, target.detach().cpu().tolist()):
            param.data.fill_(float(value))


class FLEXUpstreamPolicy(FLEXPolicy):
    """Run the vendored FLEX learning step without local clipping or rollback.

    Inputs use the same rollout tensors as :class:`FLEXPolicy`: filtered or true
    states with shape ``(batch, time, latent_dim)`` and float dtype, plus actions
    with shape ``(batch, time, action_dim)``. The update delegates directly to
    ``external/FLEX/agent.py::Agent.learning_step`` and returns scalar diagnostics.
    """

    update_mode = "upstream"

    def update(self, rollout: Any):
        """Apply one unmodified upstream FLEX parameter and Gram-matrix update."""
        assert self._flex_agent is not None
        prev_mean = self._flex_parameter_vector()
        x_key, x_next_key = self._state_keys_for_update()
        x_t = self._extract_last_tensor(rollout, x_key)
        x_next = self._extract_last_tensor(rollout, x_next_key)
        if x_t is None or x_next is None:
            x_t = self._extract_last_tensor(rollout, "env_state")
            x_next = self._extract_last_tensor(rollout, "next_env_state")
        played_action = self._extract_last_tensor(rollout, "env_action")
        if x_t is None or x_next is None or played_action is None:
            info = {
                "parameter_posterior_updated": False,
                "flex_residual_norm": 0.0,
                "flex_update_norm": 0.0,
                "flex_update_rejected": False,
                "flex_gram_trace": float(np.trace(self._flex_agent.M)),
            }
            self.last_update_info = info
            return info

        x = x_t.reshape(-1).detach().cpu().numpy().astype(np.float64, copy=False)
        u = played_action.reshape(-1).detach().cpu().numpy().astype(np.float64, copy=False)
        dx_dt = (
            ((x_next - x_t) / float(self.env_preset.dt))
            .reshape(-1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )
        self._flex_agent.learning_step(x, u, dx_dt)

        new_mean = self._flex_parameter_vector()
        info = {
            "parameter_posterior_updated": True,
            "flex_residual_norm": 0.0,
            "flex_update_norm": float(torch.linalg.norm(new_mean - prev_mean).item()),
            "flex_update_rejected": False,
            "flex_gram_trace": float(np.trace(self._flex_agent.M)),
        }
        self.last_update_info = info
        return info
