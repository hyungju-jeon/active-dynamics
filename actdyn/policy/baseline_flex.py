"""FLEX policy with point parameter estimate and Gram matrix.

FLEX in Blanke & Lelarge (ICML 2023) does not maintain a Bayesian parameter
posterior. In this repo we therefore keep shared latent-state filtering in the
model, but parameter estimation lives inside the policy as a point estimate plus
an online Gram / information matrix. The inverse Gram is exposed only as a
diagnostic proxy so the existing trace/export pipeline can remain unchanged.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from .base import BasePolicy


class FLEXPolicy(BasePolicy):
    """FLEX with policy-local point estimate and online Gram matrix."""

    owns_parameter_estimate = True

    def __init__(
        self,
        *,
        action_space: gym.Space,
        model: Any,
        initial_parameter_mean: torch.Tensor | None = None,
        gram_init_scale: float = 1e-3,
        gram_ridge: float = 1e-8,
        parameter_step_clip: float = 0.25,
        eps: float = 1e-8,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(action_space=action_space, chunk=1, device=device, **kwargs)
        self.model = model
        self.gram_init_scale = float(max(gram_init_scale, 1e-12))
        self.gram_ridge = float(max(gram_ridge, 1e-12))
        self.parameter_step_clip = float(max(parameter_step_clip, 1e-8))
        self.eps = float(max(eps, 1e-12))
        self.parameter_clip = float(max(getattr(model, "e_clip", 100.0), 1.0))
        lr = getattr(model, "lr", None)
        self.learning_rate = None if lr is None else float(lr)
        self.batch_size = int(max(1, getattr(model, "batch_size", 100)))

        if initial_parameter_mean is None:
            model_e = getattr(model, "e", {})
            initial_parameter_mean = model_e.get("m")
        if initial_parameter_mean is None:
            fallback_dim = int(getattr(model, "embedding_dim", getattr(model, "d_embed", 2)))
            initial_parameter_mean = torch.zeros(1, fallback_dim, dtype=torch.float32)

        init_mean = torch.as_tensor(initial_parameter_mean, dtype=torch.float32, device=self.device)
        if init_mean.dim() == 1:
            init_mean = init_mean.unsqueeze(0)
        self._initial_parameter_mean = init_mean[:1].detach().clone()
        self._parameter_mean = self._initial_parameter_mean.detach().clone()
        self._parameter_dim = int(self._initial_parameter_mean.shape[-1])
        self._gram = self.gram_init_scale * torch.eye(
            self._parameter_dim, dtype=torch.float32, device=self.device
        )
        self._gram_inv = (1.0 / self.gram_init_scale) * torch.eye(
            self._parameter_dim, dtype=torch.float32, device=self.device
        )
        self._rng = np.random.default_rng()
        self._theta_param: nn.Parameter | None = None
        self._theta_optimizer: torch.optim.Optimizer | None = None
        self._state_history: deque[torch.Tensor] = deque(maxlen=self.batch_size)
        self._action_history: deque[torch.Tensor] = deque(maxlen=self.batch_size)
        self._dxdt_history: deque[torch.Tensor] = deque(maxlen=self.batch_size)
        self.last_update_info: dict[str, float | bool] = {
            "parameter_posterior_updated": False,
            "flex_residual_norm": 0.0,
            "flex_update_norm": 0.0,
            "flex_gram_trace": float(torch.trace(self._gram).item()),
        }

        action_low = np.asarray(getattr(self.action_space, "low", -1.0), dtype=np.float32).reshape(-1)
        action_high = np.asarray(getattr(self.action_space, "high", 1.0), dtype=np.float32).reshape(-1)
        self._action_low = torch.as_tensor(action_low, dtype=torch.float32, device=self.device)
        self._action_high = torch.as_tensor(action_high, dtype=torch.float32, device=self.device)
        self._gamma = float(
            torch.min(torch.minimum(self._action_high.abs(), self._action_low.abs())).item()
        )
        if not math.isfinite(self._gamma) or self._gamma <= 0.0:
            self._gamma = float(
                torch.max(torch.maximum(self._action_high.abs(), self._action_low.abs())).item()
            )
        self._gamma = max(self._gamma, 1e-6)

    def reset_policy_state(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(None if seed is None else int(seed))
        self.count = 0
        self.action_list = []
        self.cost = 0.0
        self._parameter_mean = self._initial_parameter_mean.detach().clone()
        self._gram = self.gram_init_scale * torch.eye(
            self._parameter_dim, dtype=torch.float32, device=self.device
        )
        self._gram_inv = (1.0 / self.gram_init_scale) * torch.eye(
            self._parameter_dim, dtype=torch.float32, device=self.device
        )
        self._theta_param = None
        self._theta_optimizer = None
        self._state_history.clear()
        self._action_history.clear()
        self._dxdt_history.clear()
        self.last_update_info = {
            "parameter_posterior_updated": False,
            "flex_residual_norm": 0.0,
            "flex_update_norm": 0.0,
            "flex_gram_trace": float(torch.trace(self._gram).item()),
        }

    def get_parameter_mean(self) -> torch.Tensor:
        return self._parameter_mean.detach().clone()

    def get_parameter_covariance(self) -> torch.Tensor:
        return self._gram_inv.detach().clone().unsqueeze(0)

    def get_parameter_precision(self) -> torch.Tensor:
        return self._gram.detach().clone().unsqueeze(0)

    def _current_embedding(self) -> torch.Tensor:
        return self._parameter_mean[:1]

    def _current_state(self, state: torch.Tensor) -> torch.Tensor:
        z = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if z.dim() == 1:
            z = z.view(1, 1, -1)
        elif z.dim() == 2:
            z = z.unsqueeze(1)
        return z[:1]

    def _parameter_jacobian(self, z: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        fe = getattr(self.model, "Fe", None)
        if fe is None:
            raise ValueError("FLEX requires model.Fe(z, e)")
        V = fe(z, e.unsqueeze(1))
        if V.dim() == 4:
            V = V.squeeze(0).squeeze(0)
        elif V.dim() == 3:
            V = V.squeeze(0)
        return torch.as_tensor(V, dtype=torch.float32, device=self.device)

    def _state_jacobian_of_feature(self, z_t: torch.Tensor, e: torch.Tensor, row_idx: int) -> torch.Tensor:
        z_req = z_t.detach().clone().requires_grad_(True)
        V = self._parameter_jacobian(z_req, e)
        v = V[row_idx]
        d_embed = int(v.shape[0])
        d_state = int(z_req.shape[-1])
        rows = []
        for j in range(d_embed):
            if not v[j].requires_grad:
                rows.append(torch.zeros(d_state, dtype=z_req.dtype, device=self.device))
                continue
            grad = torch.autograd.grad(
                v[j], z_req, retain_graph=j < d_embed - 1, allow_unused=True
            )[0]
            if grad is None:
                grad = torch.zeros_like(z_req)
            rows.append(grad.reshape(d_state))
        return torch.stack(rows, dim=0)

    def _action_jacobian(self, z_t: torch.Tensor) -> torch.Tensor:
        d_action = int(self._action_low.numel())
        action = torch.zeros((1, 1, d_action), device=self.device, dtype=torch.float32, requires_grad=True)
        if getattr(self.model, "action_encoder", None) is not None:
            encoded = self.model.action_encoder(action, z_t)
        else:
            encoded = action
        encoded = torch.as_tensor(encoded, dtype=torch.float32, device=self.device)
        d_latent = int(encoded.shape[-1])
        rows = []
        for i in range(d_latent):
            grad = torch.autograd.grad(encoded[0, 0, i], action, retain_graph=i < d_latent - 1)[0]
            rows.append(grad.reshape(d_action))
        J = torch.stack(rows, dim=0)
        return float(getattr(self.model, "dt", 1.0)) * J

    def _encode_action(self, z: torch.Tensor, action: torch.Tensor | None) -> torch.Tensor | None:
        if action is None:
            return None
        if getattr(self.model, "action_encoder", None) is not None:
            return self.model.action_encoder(action, z)
        return action

    def _predict_derivative(
        self, z: torch.Tensor, *, action: torch.Tensor | None, e: torch.Tensor
    ) -> torch.Tensor:
        dynamics = getattr(self.model, "dynamics", None)
        if dynamics is None:
            raise ValueError("FLEX requires model.dynamics")
        try:
            drift, _ = dynamics.compute_param(z, e=e)
        except (AttributeError, TypeError):
            if hasattr(dynamics, "set_params"):
                dynamics.set_params(e)
            drift, _ = dynamics.compute_param(z)
        pred = drift
        u_enc = self._encode_action(z, action)
        if u_enc is not None and u_enc.shape[-1] > 0:
            pred = pred + u_enc
        return pred

    def _solve_quadratic(self, Q: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        Q = 0.5 * (Q + Q.transpose(-1, -2))
        gamma = float(self._gamma)
        d_action = int(b.shape[0])
        eigvals, eigvecs = torch.linalg.eigh(Q)
        if torch.linalg.norm(b) <= self.eps:
            vec = eigvecs[:, 0]
            if torch.linalg.norm(vec) <= self.eps:
                return torch.zeros(d_action, device=self.device)
            return gamma * vec / torch.linalg.norm(vec)

        beta = eigvecs.transpose(-1, -2) @ b
        mu_lo = float(-eigvals[0].item() + 0.9 * (1.0 / gamma) * abs(beta[0].item()))
        mu_hi = float(-eigvals[0].item() + 1.1 * (1.0 / gamma) * torch.linalg.norm(b).item())
        mu_lo = max(mu_lo, self.eps)
        mu_hi = max(mu_hi, mu_lo + self.eps)

        def _norm_sq(mu: float) -> float:
            denom = eigvals + torch.as_tensor(mu, device=self.device, dtype=torch.float32)
            return float(torch.sum((beta / denom) ** 2).item())

        while _norm_sq(mu_hi) > gamma * gamma:
            mu_hi *= 2.0
            if mu_hi > 1e8:
                break

        for _ in range(80):
            mu = 0.5 * (mu_lo + mu_hi)
            if _norm_sq(mu) > gamma * gamma:
                mu_lo = mu
            else:
                mu_hi = mu

        mu = mu_hi
        denom = eigvals + torch.as_tensor(mu, device=self.device, dtype=torch.float32)
        u = eigvecs @ (beta / denom)
        norm_u = float(torch.linalg.norm(u).item())
        if norm_u > self.eps:
            u = u * (gamma / norm_u)
        return u

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

    def get_action(self, state: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        del kwargs
        z_t = self._current_state(state)
        e_t = self._current_embedding()
        V_t = self._parameter_jacobian(z_t, e_t)
        row_idx = int(self._rng.integers(0, max(1, V_t.shape[0])))
        v = V_t[row_idx]
        D = self._state_jacobian_of_feature(z_t, e_t, row_idx=row_idx)
        B_dyn = self._action_jacobian(z_t)
        B = D @ B_dyn
        M_inv = self._gram_inv

        Q = -(B.transpose(-1, -2) @ M_inv @ B)
        b = B.transpose(-1, -2) @ M_inv @ v

        action_vec = self._solve_quadratic(Q, b)
        action_vec = torch.maximum(torch.minimum(action_vec, self._action_high), self._action_low)
        objective = action_vec @ (Q @ action_vec) - 2.0 * (b @ action_vec)
        return action_vec.view(1, 1, -1), torch.tensor(float((-objective).item()), device=self.device)

    def _rls_update(
        self,
        feature_matrix: torch.Tensor,
        observation: torch.Tensor,
        *,
        action: torch.Tensor | None,
        state: torch.Tensor,
    ) -> tuple[float, float]:
        theta = self._parameter_mean.reshape(-1)
        feature_matrix = torch.as_tensor(feature_matrix, dtype=torch.float32, device=self.device)
        observation = torch.as_tensor(observation, dtype=torch.float32, device=self.device).reshape(-1)
        action_effect = self._encode_action(state, action)
        if action_effect is not None and action_effect.shape[-1] == observation.shape[0]:
            observation = observation - action_effect.reshape(-1)
        prediction = feature_matrix @ theta
        residual_norm = float(torch.linalg.norm(observation - prediction).item())
        update_norm = 0.0
        for row_idx in range(feature_matrix.shape[0]):
            v = feature_matrix[row_idx]
            posterior_gram = self._gram + torch.outer(v, v)
            combination = self._gram @ theta + observation[row_idx] * v
            theta_new = torch.linalg.solve(
                posterior_gram + self.gram_ridge * torch.eye(self._parameter_dim, device=self.device),
                combination.unsqueeze(-1),
            ).squeeze(-1)
            update_norm += float(torch.linalg.norm(theta_new - theta).item())
            denom = 1.0 + (v @ self._gram_inv @ v)
            correction = (self._gram_inv @ v.unsqueeze(-1)) @ (v.unsqueeze(0) @ self._gram_inv)
            self._gram_inv = self._gram_inv - correction / denom.clamp_min(self.eps)
            self._gram = posterior_gram
            theta = theta_new
        self._parameter_mean = torch.nan_to_num(
            theta.unsqueeze(0), nan=0.0, posinf=self.parameter_clip, neginf=-self.parameter_clip
        ).clamp(-self.parameter_clip, self.parameter_clip).detach()
        self._gram = 0.5 * (self._gram + self._gram.transpose(-1, -2))
        self._gram_inv = 0.5 * (self._gram_inv + self._gram_inv.transpose(-1, -2))
        return residual_norm, update_norm

    def _ensure_theta_optimizer(self) -> None:
        if self.learning_rate is None:
            return
        if self._theta_param is None:
            self._theta_param = nn.Parameter(self._parameter_mean.reshape(-1).detach().clone())
            self._theta_optimizer = torch.optim.Adam([self._theta_param], lr=self.learning_rate)

    def _gradient_update(
        self, state: torch.Tensor, action: torch.Tensor | None, dx_dt: torch.Tensor
    ) -> tuple[float, float]:
        default_action = torch.zeros(1, 1, self._action_low.numel(), dtype=torch.float32, device=self.device)
        self._state_history.append(state.detach().clone())
        self._action_history.append((default_action if action is None else action.detach().clone()))
        self._dxdt_history.append(dx_dt.detach().clone())
        self._ensure_theta_optimizer()
        assert self._theta_param is not None
        assert self._theta_optimizer is not None

        states = torch.cat(list(self._state_history), dim=0)
        actions = torch.cat(list(self._action_history), dim=0)
        targets = torch.cat(list(self._dxdt_history), dim=0)
        theta_batch = self._theta_param.view(1, -1).expand(states.shape[0], -1)
        pred = self._predict_derivative(states, action=actions, e=theta_batch)
        loss = torch.mean((pred - targets) ** 2)
        self._theta_optimizer.zero_grad()
        loss.backward()
        self._theta_optimizer.step()

        theta_new = torch.nan_to_num(
            self._theta_param.detach().view(1, -1),
            nan=0.0,
            posinf=self.parameter_clip,
            neginf=-self.parameter_clip,
        ).clamp(-self.parameter_clip, self.parameter_clip)
        update_norm = float(torch.linalg.norm(theta_new - self._parameter_mean).item())
        self._parameter_mean = theta_new
        feature_matrix = self._parameter_jacobian(state, self._parameter_mean)
        self._gram = self._gram + feature_matrix.transpose(-1, -2) @ feature_matrix
        self._gram_inv = torch.linalg.pinv(
            self._gram + self.gram_ridge * torch.eye(self._parameter_dim, device=self.device)
        )
        return float(torch.sqrt(loss).item()), update_norm

    def update(self, rollout: Any) -> dict[str, float | bool]:
        x_t = self._extract_last_tensor(rollout, "env_state")
        x_next = self._extract_last_tensor(rollout, "next_env_state")
        played_action = self._extract_last_tensor(rollout, "env_action")
        if x_t is None or x_next is None:
            info = {
                "parameter_posterior_updated": False,
                "flex_residual_norm": 0.0,
                "flex_update_norm": 0.0,
                "flex_gram_trace": float(torch.trace(self._gram).item()),
            }
            self.last_update_info = info
            return info

        dx_dt = (x_next - x_t) / float(getattr(self.model, "dt", 1.0))
        if self.learning_rate is None:
            feature_matrix = self._parameter_jacobian(x_t, self._current_embedding())
            residual_norm, update_norm = self._rls_update(
                feature_matrix,
                dx_dt.reshape(-1),
                action=played_action,
                state=x_t,
            )
        else:
            residual_norm, update_norm = self._gradient_update(x_t, played_action, dx_dt)

        info_dict = {
            "parameter_posterior_updated": True,
            "flex_residual_norm": float(residual_norm),
            "flex_update_norm": float(update_norm),
            "flex_gram_trace": float(torch.trace(self._gram).item()),
        }
        self.last_update_info = info_dict
        return info_dict


FLEXExactPolicy = FLEXPolicy
FlexExactPolicy = FLEXPolicy
