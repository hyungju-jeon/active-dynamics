from __future__ import annotations

from typing import Any

import torch
from torch.nn.functional import softplus

from actdyn.metrics.base import BaseMetric
from actdyn.models.base import BaseModel
from actdyn.models.dynamics import BaseDynamics
from actdyn.utils.helper import eps, safe_cholesky, symmetrize


def _support_offsets(radius: int) -> tuple[list[int], list[int]]:
    off_i: list[int] = []
    off_j: list[int] = []
    for di in range(-int(radius), int(radius) + 1):
        for dj in range(-int(radius), int(radius) + 1):
            if abs(di) + abs(dj) <= int(radius):
                off_i.append(di)
                off_j.append(dj)
    return off_i, off_j


def _sorted_unique_indices(indices: torch.Tensor | list[int], *, device: torch.device) -> torch.Tensor:
    idx = torch.as_tensor(indices, dtype=torch.long, device=device).reshape(-1)
    if idx.numel() == 0:
        return idx
    idx = idx[idx >= 0]
    if idx.numel() == 0:
        return idx
    return torch.unique(idx, sorted=True)


def _expand_columns(values: torch.Tensor, old_idx: torch.Tensor, new_idx: torch.Tensor) -> torch.Tensor:
    out = values.new_zeros(values.shape[0], int(new_idx.numel()))
    if values.numel() == 0 or old_idx.numel() == 0:
        return out
    pos = torch.searchsorted(new_idx, old_idx)
    out[:, pos] = values
    return out


def _expand_square(values: torch.Tensor, old_idx: torch.Tensor, new_idx: torch.Tensor) -> torch.Tensor:
    out = values.new_zeros(int(new_idx.numel()), int(new_idx.numel()))
    if values.numel() == 0 or old_idx.numel() == 0:
        return out
    pos = torch.searchsorted(new_idx, old_idx)
    out[pos.unsqueeze(1), pos.unsqueeze(0)] = values
    return out


class SparseRbfDynamics(BaseDynamics):
    def __init__(
        self,
        *,
        state_dim: int,
        centers: torch.Tensor,
        axis: torch.Tensor,
        width: float,
        support_radius: int,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        super().__init__(
            state_dim=state_dim,
            dt=kwargs.get("dt", 1.0),
            is_residual=kwargs.get("is_residual", True),
            device=device,
        )
        centers = torch.as_tensor(centers, dtype=torch.float32, device=self.device)
        axis = torch.as_tensor(axis, dtype=torch.float32, device=self.device)
        self.register_buffer("centers", centers)
        self.register_buffer("axis", axis)
        self.width = float(max(width, 1e-6))
        self.support_radius = int(support_radius)
        off_i, off_j = _support_offsets(self.support_radius)
        self.register_buffer("offset_i", torch.as_tensor(off_i, dtype=torch.long, device=self.device))
        self.register_buffer("offset_j", torch.as_tensor(off_j, dtype=torch.long, device=self.device))
        self.n_axis = int(axis.numel())
        grid_i, grid_j = torch.meshgrid(
            torch.arange(self.n_axis, device=self.device),
            torch.arange(self.n_axis, device=self.device),
            indexing="ij",
        )
        center_i = grid_i.reshape(-1)
        center_j = grid_j.reshape(-1)
        self.register_buffer("center_i", center_i)
        self.register_buffer("center_j", center_j)
        center_dist = torch.abs(center_i.unsqueeze(1) - center_i.unsqueeze(0)) + torch.abs(
            center_j.unsqueeze(1) - center_j.unsqueeze(0)
        )
        center_overlap = center_dist <= (2 * self.support_radius)
        param_mask = center_overlap.repeat_interleave(self.state_dim, dim=0).repeat_interleave(
            self.state_dim, dim=1
        )
        eye = torch.eye(int(param_mask.shape[0]), dtype=torch.bool, device=self.device)
        self.register_buffer("parameter_structure_mask", param_mask | eye)
        self.weights = torch.zeros(
            int(self.centers.shape[0]), int(state_dim), dtype=torch.float32, device=self.device
        )
        self.network = None

    @property
    def d_embed(self) -> int:
        return int(self.centers.shape[0] * self.state_dim)

    def set_params(self, dyn_param: torch.Tensor | list[float] | dict[str, float]) -> None:
        if isinstance(dyn_param, dict):
            flat = torch.as_tensor(list(dyn_param.values()), dtype=torch.float32, device=self.device)
        else:
            flat = torch.as_tensor(dyn_param, dtype=torch.float32, device=self.device).reshape(-1)
        expected = int(self.centers.shape[0] * self.state_dim)
        if flat.numel() != expected:
            raise ValueError(f"Expected {expected} RBF weights, got {flat.numel()}")
        self.weights = flat.reshape(int(self.centers.shape[0]), int(self.state_dim)).detach().clone()

    def _nearest_axis_index(self, values: torch.Tensor) -> torch.Tensor:
        diffs = torch.abs(values.unsqueeze(-1) - self.axis.view(1, -1))
        return torch.argmin(diffs, dim=-1)

    def parameter_indices_for_centers(self, center_idx: torch.Tensor | list[int]) -> torch.Tensor:
        center_idx_t = _sorted_unique_indices(center_idx, device=self.device)
        if center_idx_t.numel() == 0:
            return center_idx_t
        base = center_idx_t.unsqueeze(-1) * self.state_dim
        dim_offset = torch.arange(self.state_dim, device=self.device, dtype=torch.long).view(1, -1)
        return (base + dim_offset).reshape(-1)

    def expand_parameter_indices(self, param_idx: torch.Tensor | list[int]) -> torch.Tensor:
        param_idx_t = _sorted_unique_indices(param_idx, device=self.device)
        if param_idx_t.numel() == 0:
            return param_idx_t
        overlap_mask = self.parameter_structure_mask.index_select(0, param_idx_t).any(dim=0)
        return torch.nonzero(overlap_mask, as_tuple=False).reshape(-1)

    def active_parameter_indices(self, state: torch.Tensor) -> torch.Tensor:
        flat_idx, _phi, _local_centers, _valid, _base_shape, _flat = self.local_features(state)
        return self.parameter_indices_for_centers(flat_idx.reshape(-1))

    def local_features(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Size, torch.Tensor]:
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        base_shape = state.shape[:-1]
        flat = state.reshape(-1, self.state_dim)
        x_idx = self._nearest_axis_index(flat[:, 0])
        y_idx = self._nearest_axis_index(flat[:, 1])
        ix = x_idx.unsqueeze(-1) + self.offset_i.view(1, -1)
        iy = y_idx.unsqueeze(-1) + self.offset_j.view(1, -1)
        valid = (ix >= 0) & (ix < self.n_axis) & (iy >= 0) & (iy < self.n_axis)
        safe_ix = ix.clamp(0, self.n_axis - 1)
        safe_iy = iy.clamp(0, self.n_axis - 1)
        flat_idx = safe_ix * self.n_axis + safe_iy
        local_centers = self.centers[flat_idx]
        scaled = (local_centers - flat.unsqueeze(1)) / self.width
        phi = torch.exp(-0.5 * torch.sum(scaled * scaled, dim=-1)) * valid.to(flat.dtype)
        return flat_idx, phi, local_centers, valid, base_shape, flat

    def local_jacobians(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_idx, phi, local_centers, _valid, base_shape, flat = self.local_features(state)
        local_weights = self.weights[flat_idx]
        grad_phi = phi.unsqueeze(-1) * (local_centers - flat.unsqueeze(1)) / (self.width**2)
        fz = torch.einsum("nko,nkd->nod", local_weights, grad_phi)
        n = int(flat.shape[0])
        d_embed = self.d_embed
        fe = torch.zeros((n, self.state_dim, d_embed), dtype=flat.dtype, device=self.device)
        base_cols = flat_idx * self.state_dim
        for dim in range(self.state_dim):
            target = fe[:, dim, :]
            target.scatter_add_(1, base_cols + dim, phi)
        return (
            fz.reshape(*base_shape, self.state_dim, self.state_dim),
            fe.reshape(*base_shape, self.state_dim, d_embed),
            flat_idx.reshape(*base_shape, -1),
        )

    def compute_param(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat_idx, phi, _local_centers, _valid, base_shape, _flat = self.local_features(state)
        local_weights = self.weights[flat_idx]
        mu = torch.sum(phi.unsqueeze(-1) * local_weights, dim=1)
        mu = mu.reshape(*base_shape, self.state_dim)
        var = softplus(self.logvar) + eps
        return mu, var


class SparseRbfFilteringModel(BaseModel):
    def __init__(
        self,
        *,
        dynamics: SparseRbfDynamics,
        decoder: Any,
        action_encoder: Any,
        e: dict[str, torch.Tensor],
        q_theta: float = 1e-4,
        k_theta: int = 1,
        e_clip: float = 100.0,
        state_init_uncertainty: float = 1.0,
        device: str = "cpu",
    ) -> None:
        super().__init__(encoder=None, action_encoder=action_encoder, dynamics=dynamics, decoder=decoder, device=device)
        self.beta = 0.0
        self.q_theta = float(q_theta)
        self.k_theta = max(1, int(k_theta))
        self.e_clip = max(float(e_clip), 1.0)
        self.state_init_uncertainty = max(float(state_init_uncertainty), 1e-9)
        self._parameter_structure_mask = dynamics.parameter_structure_mask.to(self.device)
        self._parameter_eye = torch.eye(dynamics.d_embed, dtype=torch.float32, device=self.device)
        e_cov = e.get("P")
        if e_cov is None:
            e_diag = torch.as_tensor(e["P_diag"], dtype=torch.float32, device=self.device)
            if e_diag.ndim == 1:
                e_diag = e_diag.unsqueeze(0)
            e_cov = torch.diag_embed(e_diag)
        self.e = {
            "m": torch.as_tensor(e["m"], dtype=torch.float32, device=self.device),
            "P": torch.as_tensor(e_cov, dtype=torch.float32, device=self.device),
        }
        self._normalize_embedding_belief()
        self._initial_e_cov = self.e["P"].detach().clone()
        batch = int(self.e["m"].shape[0])
        self.z = {
            "m": torch.zeros(batch, 1, self.latent_dim, dtype=torch.float32, device=self.device),
            "P": self._initial_state_covariance(batch_size=batch),
        }
        self._state = self.z["m"].detach().clone()
        self._reset_embedding_block_state(batch_size=batch)
        self.last_information = {
            "I_z_t": 0.0,
            "I_theta_t": 0.0,
            "Pz00": 0.0,
            "Pz01": 0.0,
            "Pz11": 0.0,
        }
        self.Fe = self._continuous_fe
        self.Fz = self._continuous_fz
        self.set_params(self.e["m"])

    def _normalize_embedding_belief(self) -> None:
        e_m = self.e["m"].to(self.device)
        if e_m.ndim == 1:
            e_m = e_m.unsqueeze(0)
        e_m = torch.nan_to_num(e_m, nan=0.0, posinf=self.e_clip, neginf=-self.e_clip).clamp(
            -self.e_clip, self.e_clip
        )
        batch = int(e_m.shape[0])
        p = self.e["P"].to(self.device)
        if p.ndim == 2:
            p = p.unsqueeze(0)
        if p.shape[0] == 1 and batch > 1:
            p = p.expand(batch, -1, -1).clone()
        p = self._mask_parameter_matrix(p)
        self.e = {"m": e_m, "P": p}

    def _ensure_state_belief_shapes(self, batch_size: int) -> None:
        z_m = self.z["m"]
        if z_m.dim() == 2:
            z_m = z_m.unsqueeze(1)
        z_m = z_m.to(self.device)
        if z_m.shape[0] == 1 and batch_size > 1:
            z_m = z_m.expand(batch_size, -1, -1).clone()
        z_p = self.z["P"]
        if z_p.dim() == 2:
            z_p = z_p.unsqueeze(0).unsqueeze(0)
        elif z_p.dim() == 3:
            z_p = z_p.unsqueeze(1)
        z_p = z_p.to(self.device)
        if z_p.shape[0] == 1 and batch_size > 1:
            z_p = z_p.expand(batch_size, -1, -1, -1).clone()
        self.z = {"m": z_m, "P": symmetrize(z_p)}

    def _reset_embedding_block_state(self, batch_size: int) -> None:
        d_embed = int(self.e["m"].shape[-1])
        self._theta_block_steps = 0
        self._theta_score_block = torch.zeros(batch_size, d_embed, dtype=torch.float32, device=self.device)
        self._theta_info_block = torch.zeros(
            batch_size, d_embed, d_embed, dtype=torch.float32, device=self.device
        )
        self._theta_active_mask_block = torch.zeros(
            batch_size, d_embed, dtype=torch.bool, device=self.device
        )
        self._theta_sensitivity = torch.zeros(
            batch_size, self.latent_dim, d_embed, dtype=torch.float32, device=self.device
        )

    def _initial_state_covariance(self, batch_size: int) -> torch.Tensor:
        eye = torch.eye(self.latent_dim, dtype=torch.float32, device=self.device)
        return (self.state_init_uncertainty * eye.view(1, 1, self.latent_dim, self.latent_dim)).expand(
            batch_size, -1, -1, -1
        ).clone()

    @staticmethod
    def _project_spd(M: torch.Tensor, min_eig: float = 1e-6) -> torch.Tensor:
        M = symmetrize(torch.nan_to_num(M.float(), nan=0.0, posinf=1e6, neginf=-1e6))
        floor = max(float(min_eig), 1e-8)
        eye = torch.eye(M.shape[-1], device=M.device, dtype=M.dtype).expand_as(M)
        work = M
        for _ in range(4):
            try:
                eigvals, eigvecs = torch.linalg.eigh((work + floor * eye).double())
                eigvals = eigvals.clamp_min(floor)
                proj = eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2)
                return symmetrize(torch.nan_to_num(proj, nan=0.0, posinf=1e6, neginf=-1e6).to(M.dtype))
            except RuntimeError:
                work = symmetrize(work + floor * eye)
                floor *= 10.0
        diag = torch.diagonal(M, dim1=-2, dim2=-1)
        diag = torch.nan_to_num(diag, nan=floor, posinf=1e6, neginf=floor).clamp_min(floor)
        return torch.diag_embed(diag)

    @staticmethod
    def _project_psd(M: torch.Tensor, min_eig: float = 0.0) -> torch.Tensor:
        M = symmetrize(torch.nan_to_num(M.float(), nan=0.0, posinf=1e6, neginf=-1e6))
        eigvals, eigvecs = torch.linalg.eigh(M.double())
        eigvals = eigvals.clamp_min(float(min_eig))
        proj = eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2)
        return symmetrize(torch.nan_to_num(proj, nan=0.0, posinf=1e6, neginf=-1e6).to(M.dtype))

    def _mask_parameter_matrix(self, matrix: torch.Tensor) -> torch.Tensor:
        matrix = torch.nan_to_num(matrix, nan=0.0, posinf=1e6, neginf=-1e6)
        matrix = symmetrize(matrix)
        mask = self._parameter_structure_mask.to(matrix.dtype)
        masked = matrix * mask
        diag = torch.diagonal(masked, dim1=-2, dim2=-1).clamp_min(1e-8)
        masked = masked - torch.diag_embed(torch.diagonal(masked, dim1=-2, dim2=-1)) + torch.diag_embed(diag)
        return symmetrize(masked)

    def _continuous_fz(self, z: torch.Tensor, e: torch.Tensor | None = None) -> torch.Tensor:
        fz, _fe, _active = self.dynamics.local_jacobians(z)
        return fz

    def _continuous_fe(self, z: torch.Tensor, e: torch.Tensor | None = None) -> torch.Tensor:
        _fz, fe, _active = self.dynamics.local_jacobians(z)
        return fe

    def _structured_local_info_gain(self, cov_local: torch.Tensor, info_local: torch.Tensor) -> float:
        if cov_local.numel() == 0:
            return 0.0
        info_local = self._project_psd(info_local)
        cov_local = self._project_spd(cov_local)
        eye = torch.eye(cov_local.shape[-1], dtype=cov_local.dtype, device=cov_local.device)
        gain_mat = self._project_spd(eye + cov_local @ info_local, min_eig=1e-8)
        chol = safe_cholesky(gain_mat)
        return float(torch.sum(torch.log(torch.diagonal(chol, dim1=-2, dim2=-1))).item())

    def _apply_embedding_block_update(self) -> None:
        score = torch.nan_to_num(self._theta_score_block, nan=0.0, posinf=1e6, neginf=-1e6)
        info_block = self._mask_parameter_matrix(
            torch.nan_to_num(self._theta_info_block, nan=0.0, posinf=1e6, neginf=0.0)
        )
        if score.shape[0] != self.e["m"].shape[0]:
            score = score.mean(dim=0, keepdim=True)
            info_block = info_block.mean(dim=0, keepdim=True)
        p_prior = self._mask_parameter_matrix(self.e["P"] + float(self.q_theta) * self._parameter_eye.unsqueeze(0))
        m_new = self.e["m"].detach().clone()
        p_new = p_prior.detach().clone()
        for batch_idx in range(int(score.shape[0])):
            score_b = score[batch_idx]
            info_b = info_block[batch_idx]
            active_mask = (
                self._theta_active_mask_block[batch_idx]
                | (torch.abs(score_b) > 1e-10)
                | (torch.diagonal(info_b, dim1=-2, dim2=-1) > 1e-10)
            )
            active_idx = torch.nonzero(active_mask, as_tuple=False).reshape(-1)
            if active_idx.numel() == 0:
                continue
            block_idx = self.dynamics.expand_parameter_indices(active_idx)
            if block_idx.numel() == 0:
                continue
            p_local = p_prior[batch_idx].index_select(0, block_idx).index_select(1, block_idx)
            p_local = self._project_spd(p_local, min_eig=1e-6)
            info_local = info_b.index_select(0, block_idx).index_select(1, block_idx)
            info_local = self._project_psd(info_local)
            chol_p = safe_cholesky(p_local)
            eye_local = torch.eye(int(block_idx.numel()), dtype=p_local.dtype, device=self.device)
            l_prior_local = torch.cholesky_inverse(chol_p)
            l_post_local = self._project_spd(l_prior_local + info_local, min_eig=1e-6)
            chol_l = safe_cholesky(l_post_local)
            p_post_local = torch.cholesky_solve(eye_local, chol_l)
            delta_local = p_post_local @ score_b.index_select(0, block_idx).unsqueeze(-1)
            m_new[batch_idx, block_idx] = (
                m_new[batch_idx, block_idx] + delta_local.squeeze(-1)
            ).clamp(-self.e_clip, self.e_clip)
            row_idx = block_idx.unsqueeze(1).expand(-1, block_idx.numel())
            col_idx = block_idx.unsqueeze(0).expand(block_idx.numel(), -1)
            p_new[batch_idx, row_idx, col_idx] = p_post_local
        self.e = {
            "m": torch.nan_to_num(m_new, nan=0.0, posinf=self.e_clip, neginf=-self.e_clip).detach(),
            "P": self._mask_parameter_matrix(p_new).detach(),
        }
        self.set_params(self.e["m"])
        self._reset_embedding_block_state(batch_size=int(self.e["m"].shape[0]))

    def set_params(self, e: torch.Tensor) -> None:
        e = torch.as_tensor(e, dtype=torch.float32, device=self.device)
        if e.ndim == 1:
            e = e.unsqueeze(0)
        e = torch.nan_to_num(e, nan=0.0, posinf=self.e_clip, neginf=-self.e_clip).clamp(
            -self.e_clip, self.e_clip
        )
        self.e["m"] = e
        self.dynamics.set_params(e[0])

    def reset(self, observation: torch.Tensor):
        observation, info = super().reset(observation)
        batch = int(self.e["m"].shape[0])
        self.e = {
            "m": self.e["m"].detach().clone(),
            "P": self._initial_e_cov.detach().clone(),
        }
        self._normalize_embedding_belief()
        self.z = {
            "m": self._state.detach().clone(),
            "P": self._initial_state_covariance(batch_size=batch),
        }
        self._ensure_state_belief_shapes(batch_size=batch)
        self.set_params(self.e["m"])
        self._reset_embedding_block_state(batch_size=batch)
        self.last_information = {
            "I_z_t": 0.0,
            "I_theta_t": 0.0,
            "Pz00": 0.0,
            "Pz01": 0.0,
            "Pz11": 0.0,
        }
        return observation, info

    def set_state(self, state: torch.Tensor):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state.ndim == 2:
            state = state.unsqueeze(1)
        self._state = state
        self.z["m"] = state

    def predict_state(self, u=None):
        self._normalize_embedding_belief()
        self._ensure_state_belief_shapes(batch_size=int(self.e["m"].shape[0]))
        q = softplus(self.dynamics.logvar).diag_embed().unsqueeze(0) * self.dt
        eye = torch.eye(self.latent_dim, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        if u is not None and self.action_encoder is not None:
            u_enc = self.action_encoder(u, self.z["m"])
        else:
            u_enc = u
        fz = self.Fz(self.z["m"], self.e["m"])
        dfdz = eye + fz * self.dt
        pred_m = self.predict(action=u_enc)
        pred_cov = self._project_spd(dfdz @ self.z["P"] @ dfdz.transpose(-1, -2) + q + 1e-6 * eye)
        self.z = {"m": pred_m.detach(), "P": pred_cov.detach()}
        self._state = pred_m.detach()
        self.last_information = {
            "I_z_t": 0.0,
            "I_theta_t": 0.0,
            "Pz00": float(pred_cov[..., 0, 0].mean().item()),
            "Pz01": float(pred_cov[..., 0, 1].mean().item()) if pred_cov.shape[-1] > 1 else 0.0,
            "Pz11": float(pred_cov[..., 1, 1].mean().item()) if pred_cov.shape[-1] > 1 else 0.0,
        }
        return {
            "env_action": u_enc[..., -1:, :] if u_enc is not None else None,
            "latent_state": self._state,
        }

    @torch.no_grad()
    def update_posterior_embedding(self, y, u=None, **kwargs):
        self._normalize_embedding_belief()
        self._ensure_state_belief_shapes(batch_size=int(self.e["m"].shape[0]))
        y = y[:, -1:, :]
        u = u[:, -1:, :] if u is not None else None
        q = softplus(self.dynamics.logvar).diag_embed().unsqueeze(0) * self.dt
        eye = torch.eye(self.latent_dim, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        batch_size = int(y.shape[0])
        if self._theta_score_block.shape[0] != batch_size:
            self._reset_embedding_block_state(batch_size=batch_size)

        z_prev = torch.nan_to_num(self.z["m"], nan=0.0, posinf=1e6, neginf=-1e6)
        fz, fe, active_centers = self.dynamics.local_jacobians(z_prev)
        dfdz = eye + fz * self.dt
        f_theta = fe * self.dt

        if u is not None and self.action_encoder is not None:
            u_enc = self.action_encoder(u, self.z["m"])
        else:
            u_enc = u

        pred_m = torch.nan_to_num(self.predict(action=u_enc), nan=0.0, posinf=1e6, neginf=-1e6)
        pred_cov = self._project_spd(dfdz @ self.z["P"] @ dfdz.transpose(-1, -2) + q + 1e-6 * eye)
        z_pred = {"m": pred_m, "P": pred_cov}

        z_obs = torch.nan_to_num(z_pred["m"], nan=0.0, posinf=1e6, neginf=-1e6)
        dhdz = torch.nan_to_num(self.decoder.jacobian(z_obs), nan=0.0, posinf=1e6, neginf=-1e6)
        r_diag = torch.nan_to_num(self.decoder.var(z_obs), nan=1e-6, posinf=1e6, neginf=1e-6).clamp_min(1e-6)
        r = self._project_spd(r_diag.diag_embed())
        eye_obs = torch.eye(r.shape[-1], device=self.device, dtype=torch.float32).view(
            1, 1, r.shape[-1], r.shape[-1]
        )
        s = self._project_spd(dhdz @ z_pred["P"] @ dhdz.transpose(-1, -2) + r + 1e-6 * eye_obs)
        try:
            chol_s = safe_cholesky(s)
        except Exception:
            s = self._project_spd(s + 1e-4 * eye_obs, min_eig=1e-4)
            chol_s = safe_cholesky(s)

        hpt = dhdz @ z_pred["P"]
        k_gain = torch.cholesky_solve(hpt, chol_s).transpose(-1, -2)
        kh = k_gain @ dhdz
        p_upd = (eye - kh) @ z_pred["P"] @ (eye - kh).transpose(-1, -2) + k_gain @ r @ k_gain.transpose(-1, -2)

        y_pred = torch.nan_to_num(self.decoder(z_obs), nan=0.0, posinf=1e6, neginf=-1e6)
        resid_y = y - y_pred
        z_post_m = z_pred["m"] + (k_gain @ resid_y.unsqueeze(-1)).squeeze(-1)
        z_post = {
            "m": torch.nan_to_num(z_post_m, nan=0.0, posinf=1e6, neginf=-1e6),
            "P": self._project_spd(torch.nan_to_num(p_upd, nan=0.0, posinf=1e6, neginf=-1e6) + 1e-8 * eye),
        }

        self.z = {"m": z_post["m"].detach(), "P": z_post["P"].detach()}
        self._state = z_post["m"].detach()

        s_prev = self._theta_sensitivity
        s_t = f_theta.squeeze(1) + dfdz.squeeze(1) @ s_prev

        chol_p_pred = safe_cholesky(self._project_spd(z_pred["P"]))
        delta_z = z_post["m"] - z_pred["m"]
        invp_delta = torch.cholesky_solve(delta_z.unsqueeze(-1), chol_p_pred).squeeze(-1).squeeze(1)
        score_t = torch.einsum("bze,bz->be", s_t, invp_delta)

        chol_r = safe_cholesky(self._project_spd(r))
        invr_h = torch.cholesky_solve(dhdz, chol_r)
        i_z = symmetrize(dhdz.transpose(-1, -2) @ invr_h)
        atten_mat = self._project_spd(eye + z_pred["P"] @ i_z)
        chol_atten = safe_cholesky(atten_mat)
        atten_i_z = torch.cholesky_solve(i_z, chol_atten).squeeze(1)

        info_scalar = 0.0
        info_block = self._theta_info_block
        active_block_mask = self._theta_active_mask_block
        support_mask = torch.sum(torch.abs(s_t), dim=1) > 1e-10
        for batch_idx in range(batch_size):
            support_idx = torch.nonzero(support_mask[batch_idx], as_tuple=False).reshape(-1)
            if support_idx.numel() == 0:
                support_idx = self.dynamics.parameter_indices_for_centers(active_centers[batch_idx].reshape(-1))
            block_idx = self.dynamics.expand_parameter_indices(support_idx)
            if block_idx.numel() == 0:
                continue
            s_block = s_t[batch_idx : batch_idx + 1].index_select(-1, block_idx)
            mid = torch.einsum("bij,bje->bie", atten_i_z[batch_idx : batch_idx + 1], s_block)
            info_local = torch.einsum("bze,bzf->bef", s_block, mid).squeeze(0)
            info_local = self._project_psd(info_local)
            row_idx = block_idx.unsqueeze(1).expand(-1, block_idx.numel())
            col_idx = block_idx.unsqueeze(0).expand(block_idx.numel(), -1)
            info_block[batch_idx, row_idx, col_idx] = info_block[batch_idx, row_idx, col_idx] + info_local
            active_block_mask[batch_idx, block_idx] = True
            cov_local = self.e["P"][min(batch_idx, self.e["P"].shape[0] - 1)].index_select(0, block_idx).index_select(
                1, block_idx
            )
            info_scalar += self._structured_local_info_gain(cov_local, info_local)

        i_z_scalar = torch.diagonal(i_z.squeeze(1), dim1=-2, dim2=-1).sum(dim=-1).mean()
        pz_eval = z_pred["P"].squeeze(1)
        self.last_information = {
            "I_z_t": float(torch.nan_to_num(i_z_scalar, nan=0.0, posinf=1e6, neginf=0.0).item()),
            "I_theta_t": float(info_scalar / max(1, batch_size)),
            "Pz00": float(torch.nan_to_num(pz_eval[:, 0, 0].mean(), nan=0.0, posinf=1e6, neginf=0.0).item()),
            "Pz01": float(torch.nan_to_num(pz_eval[:, 0, 1].mean(), nan=0.0, posinf=1e6, neginf=0.0).item()),
            "Pz11": float(torch.nan_to_num(pz_eval[:, 1, 1].mean(), nan=0.0, posinf=1e6, neginf=0.0).item()),
        }

        self._theta_score_block += score_t
        self._theta_info_block = info_block
        self._theta_active_mask_block = active_block_mask
        self._theta_sensitivity = s_t.detach()
        self._theta_block_steps += 1
        if self._theta_block_steps >= self.k_theta:
            self._apply_embedding_block_update()

        return self._state

    def weight_mean(self) -> torch.Tensor:
        return self.e["m"].reshape(-1, self.latent_dim)

    def weight_precision(self) -> torch.Tensor:
        diag_var = torch.diagonal(self.e["P"], dim1=-2, dim2=-1).clamp_min(1e-8)
        return (1.0 / diag_var).reshape(-1, self.latent_dim)

    def weight_covariance(self) -> torch.Tensor:
        return self.e["P"]


class StructuredLocalRbfParameterMetric(BaseMetric):
    def __init__(self, *, model: SparseRbfFilteringModel, gamma: float, device: str = "cpu") -> None:
        super().__init__(compute_type="sum", device=device)
        self.model = model
        self.gamma = float(gamma)

    def _to_batch_latent_cov(self, cov: torch.Tensor, batch: int) -> torch.Tensor:
        if cov.dim() == 4:
            cov = cov.squeeze(1)
        elif cov.dim() == 2:
            cov = cov.unsqueeze(0)
        if cov.shape[0] == 1 and batch > 1:
            cov = cov.expand(batch, -1, -1)
        return symmetrize(cov)

    def _parameter_covariance_for_indices(self, indices: torch.Tensor, *, batch_idx: int) -> torch.Tensor:
        if indices.numel() == 0:
            return torch.zeros(0, 0, dtype=torch.float32, device=self.device)
        cov = self.model.e["P"].to(self.device)
        if cov.dim() == 2:
            cov = cov.unsqueeze(0)
        cov_batch = cov[min(int(batch_idx), int(cov.shape[0] - 1))]
        cov_local = cov_batch.index_select(0, indices).index_select(1, indices)
        mask_local = self.model._parameter_structure_mask.index_select(0, indices).index_select(1, indices)
        cov_local = symmetrize(torch.nan_to_num(cov_local, nan=0.0, posinf=1e6, neginf=-1e6))
        cov_local = cov_local * mask_local.to(cov_local.dtype)
        diag = torch.diagonal(cov_local, dim1=-2, dim2=-1).clamp_min(1e-8)
        cov_local = cov_local - torch.diag_embed(torch.diagonal(cov_local, dim1=-2, dim2=-1)) + torch.diag_embed(diag)
        return self.model._project_spd(cov_local, min_eig=1e-6)

    def _posterior_parameter_covariance(
        self, cov_local: torch.Tensor, info_local: torch.Tensor
    ) -> torch.Tensor:
        if cov_local.numel() == 0:
            return cov_local
        cov_local = self.model._project_spd(cov_local, min_eig=1e-6)
        info_local = self.model._project_psd(info_local)
        eye = torch.eye(int(cov_local.shape[-1]), dtype=cov_local.dtype, device=self.device)
        chol_cov = safe_cholesky(cov_local)
        prec_prior = torch.cholesky_inverse(chol_cov)
        prec_post = self.model._project_spd(prec_prior + info_local, min_eig=1e-6)
        chol_prec = safe_cholesky(prec_post)
        return torch.cholesky_solve(eye, chol_prec)

    def compute_stepwise(self, rollout: dict) -> torch.Tensor:
        z = torch.as_tensor(rollout["model_state"], dtype=torch.float32, device=self.device)
        if z.ndim != 3:
            z = z.unsqueeze(0)
        batch, steps, latent_dim = z.shape
        dt = float(getattr(self.model, "dt", 1.0))
        eye_latent = torch.eye(latent_dim, dtype=torch.float32, device=self.device)
        p_pred_all = self._to_batch_latent_cov(self.model.z["P"].to(self.device), batch)
        q = softplus(self.model.dynamics.logvar).diag_embed().to(self.device) * dt
        if q.dim() == 2:
            q = q.unsqueeze(0)
        if q.shape[0] == 1 and batch > 1:
            q = q.expand(batch, -1, -1)
        q = symmetrize(q)
        step_costs: list[torch.Tensor] = []

        for batch_idx in range(batch):
            p_pred = p_pred_all[batch_idx]
            union_idx = torch.empty(0, dtype=torch.long, device=self.device)
            s_local = torch.zeros(latent_dim, 0, dtype=torch.float32, device=self.device)
            info_union = torch.zeros(0, 0, dtype=torch.float32, device=self.device)
            batch_step_cost = torch.zeros(steps, dtype=torch.float32, device=self.device)

            for step_idx in range(steps):
                z_i = z[batch_idx : batch_idx + 1, step_idx : step_idx + 1]
                fz, fe, active_centers = self.model.dynamics.local_jacobians(z_i)
                dfdz = eye_latent + fz.squeeze(0).squeeze(0) * dt
                active_idx = self.model.dynamics.parameter_indices_for_centers(active_centers.reshape(-1))
                block_idx = self.model.dynamics.expand_parameter_indices(active_idx)
                if block_idx.numel() > 0:
                    new_union = torch.unique(torch.cat((union_idx, block_idx)), sorted=True)
                    s_local = _expand_columns(s_local, union_idx, new_union)
                    info_union = _expand_square(info_union, union_idx, new_union)
                    fe_local = fe.squeeze(0).squeeze(0).index_select(-1, new_union)
                    s_local = dfdz @ s_local + fe_local * dt
                    union_idx = new_union

                    h_i = self.model.decoder.jacobian(z_i).squeeze(0).squeeze(0).to(self.device)
                    r_diag = torch.nan_to_num(
                        self.model.decoder.var(z_i).squeeze(0).squeeze(0),
                        nan=1e-6,
                        posinf=1e6,
                        neginf=1e-6,
                    ).clamp_min(1e-6)
                    r_i = self.model._project_spd(r_diag.diag_embed())
                    chol_r = safe_cholesky(r_i)
                    invr_h = torch.cholesky_solve(h_i, chol_r)
                    i_z = symmetrize(h_i.transpose(-1, -2) @ invr_h)
                    atten = self.model._project_spd(eye_latent + p_pred @ i_z)
                    chol_atten = safe_cholesky(atten)
                    atten_i_z = torch.cholesky_solve(i_z, chol_atten)
                    mid = atten_i_z @ s_local
                    info_step = self.model._project_psd(s_local.transpose(-1, -2) @ mid)
                    p_theta_local = self._posterior_parameter_covariance(
                        self._parameter_covariance_for_indices(union_idx, batch_idx=batch_idx),
                        info_union,
                    )
                    gain = self.model._structured_local_info_gain(p_theta_local, info_step)
                    batch_step_cost[step_idx] = -((self.gamma**step_idx) * gain)
                    info_union = self.model._project_psd(info_union + info_step)

                p_pred = symmetrize(dfdz @ p_pred @ dfdz.transpose(-1, -2) + q[batch_idx])

            step_costs.append(batch_step_cost)

        self.current_cost = torch.stack(step_costs, dim=0)
        return self.current_cost


LocalDiagonalRbfParameterMetric = StructuredLocalRbfParameterMetric
