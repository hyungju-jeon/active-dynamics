"""FLEX policy with point parameter estimate and Gram matrix.

FLEX in Blanke & Lelarge (ICML 2023) does not maintain a Bayesian parameter
posterior. In this repo we therefore keep shared latent-state filtering in the
model, but parameter estimation lives inside the policy as a point estimate plus
an online Gram / information matrix. The inverse Gram is exposed only as a
diagnostic proxy so the existing trace/export pipeline can remain unchanged.
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
import torch

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
        self.last_update_info: dict[str, float | bool] = {
            "parameter_posterior_updated": False,
            "flex_residual_norm": 0.0,
            "flex_update_norm": 0.0,
            "flex_gram_trace": float(torch.trace(self._gram).item()),
        }

        action_low = np.asarray(getattr(self.action_space, "low", -1.0), dtype=np.float32).reshape(
            -1
        )
        action_high = np.asarray(getattr(self.action_space, "high", 1.0), dtype=np.float32).reshape(
            -1
        )
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
        del seed
        self.count = 0
        self.action_list = []
        self.cost = 0.0
        self._parameter_mean = self._initial_parameter_mean.detach().clone()
        self._gram = self.gram_init_scale * torch.eye(
            self._parameter_dim, dtype=torch.float32, device=self.device
        )
        self.last_update_info = {
            "parameter_posterior_updated": False,
            "flex_residual_norm": 0.0,
            "flex_update_norm": 0.0,
            "flex_gram_trace": float(torch.trace(self._gram).item()),
        }

    def get_parameter_mean(self) -> torch.Tensor:
        return self._parameter_mean.detach().clone()

    def get_parameter_covariance(self) -> torch.Tensor:
        gram = self._gram + self.gram_ridge * torch.eye(
            self._parameter_dim, device=self.device, dtype=torch.float32
        )
        cov = torch.linalg.pinv(gram)
        return cov.unsqueeze(0)

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

    def _choose_feature_row(self, V: torch.Tensor, M_inv: torch.Tensor) -> tuple[int, torch.Tensor]:
        leverage = torch.einsum("id,de,ie->i", V, M_inv, V)
        row_idx = int(torch.argmax(leverage).item())
        return row_idx, V[row_idx]

    def _state_jacobian_of_feature(
        self, z_bar: torch.Tensor, e: torch.Tensor, row_idx: int
    ) -> torch.Tensor:
        z_req = z_bar.detach().clone().requires_grad_(True)
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

    def _action_jacobian(self, z_bar: torch.Tensor) -> torch.Tensor:
        d_action = int(self._action_low.numel())
        action = torch.zeros(
            (1, 1, d_action), device=self.device, dtype=torch.float32, requires_grad=True
        )
        if getattr(self.model, "action_encoder", None) is not None:
            encoded = self.model.action_encoder(action, z_bar)
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

    def _predict_next_state(
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
        pred = z + float(getattr(self.model, "dt", 1.0)) * drift
        if action is not None and action.shape[-1] > 0:
            pred = pred + float(getattr(self.model, "dt", 1.0)) * action
        return pred

    def _zero_action_prediction(self, z: torch.Tensor) -> torch.Tensor:
        return self._predict_next_state(z, action=None, e=self._current_embedding())

    def _solve_quadratic(self, Q: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        Q = 0.5 * (Q + Q.transpose(-1, -2))
        gamma = float(self._gamma)
        d_action = int(b.shape[0])
        if torch.linalg.norm(b) <= self.eps:
            eigvals, eigvecs = torch.linalg.eigh(Q)
            vec = eigvecs[:, -1]
            if torch.linalg.norm(vec) <= self.eps:
                return torch.zeros(d_action, device=self.device)
            return gamma * vec / torch.linalg.norm(vec)

        eigvals, eigvecs = torch.linalg.eigh(Q)
        coeff = eigvecs.transpose(-1, -2) @ b
        lambda_lo = float(eigvals[-1].item()) + self.eps

        def _norm_sq(lam: float) -> float:
            denom = torch.as_tensor(lam, device=self.device, dtype=torch.float32) - eigvals
            return float(torch.sum((coeff / denom) ** 2).item())

        lambda_hi = max(lambda_lo + 1.0, 1.0)
        while _norm_sq(lambda_hi) > gamma * gamma:
            lambda_hi *= 2.0
            if lambda_hi > 1e8:
                break

        for _ in range(80):
            lam = 0.5 * (lambda_lo + lambda_hi)
            if _norm_sq(lam) > gamma * gamma:
                lambda_lo = lam
            else:
                lambda_hi = lam

        lam = lambda_hi
        denom = torch.as_tensor(lam, device=self.device, dtype=torch.float32) - eigvals
        u = -(eigvecs @ (coeff / denom))
        norm_u = float(torch.linalg.norm(u).item())
        if norm_u > gamma + 1e-6:
            u = u * (gamma / max(norm_u, self.eps))
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
        z_bar = self._zero_action_prediction(z_t)

        M_inv = torch.linalg.pinv(
            self._gram + self.gram_ridge * torch.eye(self._parameter_dim, device=self.device)
        )
        V_bar = self._parameter_jacobian(z_bar, e_t)
        row_idx, v = self._choose_feature_row(V_bar, M_inv)
        D = self._state_jacobian_of_feature(z_bar, e_t, row_idx=row_idx)
        B = self._action_jacobian(z_bar)

        Q = B.transpose(-1, -2) @ D.transpose(-1, -2) @ M_inv @ D @ B
        Q = 0.5 * (Q + Q.transpose(-1, -2))
        b = -(B.transpose(-1, -2) @ D.transpose(-1, -2) @ M_inv @ v)

        action_vec = self._solve_quadratic(Q, b)
        action_vec = torch.maximum(torch.minimum(action_vec, self._action_high), self._action_low)
        objective = action_vec @ (Q @ action_vec) - 2.0 * (b @ action_vec)
        return action_vec.view(1, 1, -1), torch.tensor(
            float((-objective).item()), device=self.device
        )

    def update(self, rollout: Any) -> dict[str, float | bool]:
        z_t = self._extract_last_tensor(rollout, "model_state")
        z_next = self._extract_last_tensor(rollout, "next_model_state")
        model_action = self._extract_last_tensor(rollout, "model_action")
        if z_t is None or z_next is None:
            info = {
                "parameter_posterior_updated": False,
                "flex_residual_norm": 0.0,
                "flex_update_norm": 0.0,
                "flex_gram_trace": float(torch.trace(self._gram).item()),
            }
            self.last_update_info = info
            return info

        e_t = self._current_embedding()
        pred = self._predict_next_state(z_t, action=model_action, e=e_t)
        residual = (z_next - pred).reshape(-1)

        V = self._parameter_jacobian(z_t, e_t)
        Phi = float(getattr(self.model, "dt", 1.0)) * V.reshape(-1, V.shape[-1])
        grad = Phi.transpose(-1, -2) @ residual
        info = Phi.transpose(-1, -2) @ Phi
        self._gram = 0.5 * (self._gram + self._gram.transpose(-1, -2)) + info.detach()

        ridge_eye = self.gram_ridge * torch.eye(self._parameter_dim, device=self.device)
        delta = torch.linalg.solve(self._gram + ridge_eye, grad.unsqueeze(-1)).squeeze(-1)
        delta_norm = float(torch.linalg.norm(delta).item())
        if delta_norm > self.parameter_step_clip:
            delta = delta * (self.parameter_step_clip / max(delta_norm, self.eps))
            delta_norm = self.parameter_step_clip

        theta_new = torch.nan_to_num(
            self._parameter_mean + delta.unsqueeze(0),
            nan=0.0,
            posinf=self.parameter_clip,
            neginf=-self.parameter_clip,
        ).clamp(-self.parameter_clip, self.parameter_clip)
        self._parameter_mean = theta_new.detach()

        info_dict = {
            "parameter_posterior_updated": True,
            "flex_residual_norm": float(torch.linalg.norm(residual).item()),
            "flex_update_norm": float(delta_norm),
            "flex_gram_trace": float(torch.trace(self._gram).item()),
        }
        self.last_update_info = info_dict
        return info_dict
