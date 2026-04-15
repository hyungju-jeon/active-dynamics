"""Exact-style FLEX policy following Blanke & Lelarge (ICML 2023).

This policy intentionally does not reuse the generic MPC / CEM stack. FLEX is a
one-step adaptive D-optimal controller that solves the paper's local quadratic
program at each decision step using the current parameter estimate and an online
Gram matrix over parameter-Jacobian features.

Current scope / approximation notes:
- The policy is implemented for the repo's synthetic identification path where a
  differentiable state Jacobian `model.Fe(z, e)` is available.
- The action influence matrix B is computed through the action encoder at
  zero-action. This matches the synthetic path's additive action handling.
- The informative row `v^(k)` is chosen by leverage under the current online
  Gram inverse. This is still a local approximation because the current runner
  does not expose the full paper reference stack.
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
import torch

from .base import BasePolicy


class FLEXPolicy(BasePolicy):
    """Literature-style FLEX policy.

    Implements the paper's local action selection rule

        max_u  u^T Q u - 2 b^T u   s.t. ||u||_2 <= gamma

    with

        Q = B^T D^T M^{-1} D B
        b = -B^T D^T M^{-1} v

    where M is the online Gram matrix, v is one informative row of the
    parameter Jacobian, D=dv/dx, and B=dt * df/du at the zero-action
    prediction.
    """

    def __init__(
        self,
        *,
        action_space: gym.Space,
        model,
        gram_init_scale: float = 1e-3,
        eps: float = 1e-8,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(action_space=action_space, chunk=1, device=device, **kwargs)
        self.model = model
        self.gram_init_scale = float(max(gram_init_scale, 1e-12))
        self.eps = float(max(eps, 1e-12))

        action_low = np.asarray(getattr(self.action_space, "low", -1.0), dtype=np.float32).reshape(-1)
        action_high = np.asarray(getattr(self.action_space, "high", 1.0), dtype=np.float32).reshape(-1)
        self._action_low = torch.as_tensor(action_low, dtype=torch.float32, device=self.device)
        self._action_high = torch.as_tensor(action_high, dtype=torch.float32, device=self.device)
        self._gamma = float(torch.min(torch.minimum(self._action_high.abs(), self._action_low.abs())).item())
        if not math.isfinite(self._gamma) or self._gamma <= 0.0:
            self._gamma = float(torch.max(torch.maximum(self._action_high.abs(), self._action_low.abs())).item())
        self._gamma = max(self._gamma, 1e-6)

        self._gram: torch.Tensor | None = None
        self._pending_feature: torch.Tensor | None = None
        self._last_cost = 0.0

    def _current_embedding(self) -> torch.Tensor:
        e = getattr(self.model, "e", {}).get("m")
        if e is None:
            raise ValueError("FLEX exact requires model.e['m'] for the current parameter estimate")
        e = torch.as_tensor(e, dtype=torch.float32, device=self.device)
        if e.dim() == 1:
            e = e.unsqueeze(0)
        return e[:1]

    def _current_state(self, state: torch.Tensor) -> torch.Tensor:
        z = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if z.dim() == 1:
            z = z.view(1, 1, -1)
        elif z.dim() == 2:
            z = z.unsqueeze(1)
        if z.shape[0] != 1:
            z = z[:1]
        return z

    def _ensure_gram(self, d_embed: int) -> None:
        if self._gram is None or self._gram.shape[-1] != d_embed:
            self._gram = self.gram_init_scale * torch.eye(d_embed, device=self.device)
            self._pending_feature = None

    def _update_gram_from_pending(self) -> None:
        if self._pending_feature is None:
            return
        v = self._pending_feature
        self._gram = self._gram + v.unsqueeze(-1) @ v.unsqueeze(-2)
        self._pending_feature = None

    def _zero_action_prediction(self, z: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.model, "dynamics"):
            raise ValueError("FLEX exact requires model.dynamics")
        e = self._current_embedding()
        dynamics = self.model.dynamics
        try:
            drift, _var = dynamics.compute_param(z, e=e)
        except (AttributeError, TypeError):
            if hasattr(dynamics, "set_params"):
                dynamics.set_params(e)
            drift, _var = dynamics.compute_param(z)
        return z + float(getattr(self.model, "dt", 1.0)) * drift

    def _parameter_jacobian(self, z: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        fe = getattr(self.model, "Fe", None)
        if fe is None:
            raise ValueError("FLEX exact requires model.Fe(z, e) in the current model path")
        V = fe(z, e.unsqueeze(1))
        if V.dim() == 4:
            V = V.squeeze(0).squeeze(0)
        elif V.dim() == 3:
            V = V.squeeze(0)
        return torch.as_tensor(V, dtype=torch.float32, device=self.device)

    def _choose_feature_row(self, V: torch.Tensor, M_inv: torch.Tensor) -> tuple[int, torch.Tensor]:
        scores = torch.einsum("id,de,ie->i", V, M_inv, V)
        row_idx = int(torch.argmax(scores).item())
        return row_idx, V[row_idx]

    def _state_jacobian_of_feature(self, z_bar: torch.Tensor, e: torch.Tensor, row_idx: int) -> torch.Tensor:
        z_req = z_bar.detach().clone().requires_grad_(True)
        V = self._parameter_jacobian(z_req, e)
        v = V[row_idx]
        d_embed = int(v.shape[0])
        d_state = int(z_req.shape[-1])
        rows = []
        for j in range(d_embed):
            grad = torch.autograd.grad(v[j], z_req, retain_graph=j < d_embed - 1, allow_unused=False)[0]
            rows.append(grad.reshape(d_state))
        return torch.stack(rows, dim=0)

    def _action_jacobian(self, z_bar: torch.Tensor) -> torch.Tensor:
        d_action = int(self._action_low.numel())
        action = torch.zeros((1, 1, d_action), device=self.device, dtype=torch.float32, requires_grad=True)
        if getattr(self.model, "action_encoder", None) is not None:
            encoded = self.model.action_encoder(action, z_bar)
        else:
            encoded = action
        if encoded is None:
            encoded = action
        encoded = torch.as_tensor(encoded, dtype=torch.float32, device=self.device)
        d_latent = int(encoded.shape[-1])
        rows = []
        for i in range(d_latent):
            grad = torch.autograd.grad(encoded[0, 0, i], action, retain_graph=i < d_latent - 1)[0]
            rows.append(grad.reshape(d_action))
        J = torch.stack(rows, dim=0)
        return float(getattr(self.model, "dt", 1.0)) * J

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

    def get_action(self, state: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        del kwargs
        z_t = self._current_state(state)
        e_t = self._current_embedding()

        z_bar = self._zero_action_prediction(z_t)
        V_bar = self._parameter_jacobian(z_bar, e_t)
        d_embed = int(V_bar.shape[-1])
        self._ensure_gram(d_embed)
        self._update_gram_from_pending()

        M_inv = torch.linalg.pinv(self._gram)
        row_idx, v = self._choose_feature_row(V_bar, M_inv)
        D = self._state_jacobian_of_feature(z_bar, e_t, row_idx=row_idx)
        B = self._action_jacobian(z_bar)

        Q = B.transpose(-1, -2) @ D.transpose(-1, -2) @ M_inv @ D @ B
        Q = 0.5 * (Q + Q.transpose(-1, -2))
        b = -(B.transpose(-1, -2) @ D.transpose(-1, -2) @ M_inv @ v)

        action_vec = self._solve_quadratic(Q, b)
        action_vec = torch.maximum(torch.minimum(action_vec, self._action_high), self._action_low)

        objective = action_vec @ (Q @ action_vec) - 2.0 * (b @ action_vec)
        self._last_cost = float((-objective).item())

        # Update M on the next call so the current state's feature only enters after
        # the action selected at this state has been executed.
        self._pending_feature = v.detach()

        return action_vec.view(1, 1, -1), torch.tensor(self._last_cost, device=self.device)


FLEXExactPolicy = FLEXPolicy
FlexExactPolicy = FLEXPolicy
