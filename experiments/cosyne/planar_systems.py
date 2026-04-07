from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


_DUFFING_CUBIC = 0.1
_BISTABLE_CENTER_OFFSET = 1.6
_BISTABLE_GATE_SHARPNESS = 3.0
_EPS = 1e-6


@dataclass(frozen=True)
class PlanarSystemSpec:
    system_id: str
    label: str
    dynamics_type: str
    true_params: tuple[float, float]
    state_low: tuple[float, float]
    state_high: tuple[float, float]


PLANAR_SYSTEM_SPECS: dict[str, PlanarSystemSpec] = {
    "single_attractor": PlanarSystemSpec(
        system_id="single_attractor",
        label="single attractor",
        dynamics_type="duffing",
        true_params=(-0.55, 1.00),
        state_low=(-2.5, -2.5),
        state_high=(2.5, 2.5),
    ),
    "bistable_attractor": PlanarSystemSpec(
        system_id="bistable_attractor",
        label="bistable attractor",
        dynamics_type="duffing",
        true_params=(-1.1, -0.9),
        state_low=(-2.5, -2.5),
        state_high=(2.5, 2.5),
    ),
    "bistable_limitcycle": PlanarSystemSpec(
        system_id="bistable_limitcycle",
        label="bistable limit cycle",
        dynamics_type="bistable_limit_cycle",
        true_params=(1.00, 0.90),
        state_low=(-2.5, -2.5),
        state_high=(2.5, 2.5),
    ),
}


def get_planar_system_spec(system_id: str) -> PlanarSystemSpec:
    return PLANAR_SYSTEM_SPECS[system_id]


def sample_initial_state(system_id: str, seed: int) -> np.ndarray:
    spec = get_planar_system_spec(system_id)
    rng = np.random.default_rng(int(seed))
    low = np.asarray(spec.state_low, dtype=np.float64)
    high = np.asarray(spec.state_high, dtype=np.float64)
    return (low + (high - low) * rng.random(2)).astype(np.float32)


def true_embedding(system_id: str) -> np.ndarray:
    spec = get_planar_system_spec(system_id)
    return np.asarray(spec.true_params, dtype=np.float32)


def env_params_from_embedding(system_id: str, embedding: Any) -> Any:
    if torch.is_tensor(embedding):
        e = embedding.to(dtype=torch.float32)
        if system_id in {"single_attractor", "bistable_attractor"}:
            if e.ndim == 1:
                return torch.stack((e[0], e[1], e.new_tensor(_DUFFING_CUBIC)))
            cubic = torch.full(
                (*e.shape[:-1], 1),
                _DUFFING_CUBIC,
                dtype=e.dtype,
                device=e.device,
            )
            return torch.cat((e[..., :2], cubic), dim=-1)
        return e[..., :2]
    e_np = np.asarray(embedding, dtype=np.float32)
    if system_id in {"single_attractor", "bistable_attractor"}:
        if e_np.ndim == 1:
            return np.asarray([e_np[0], e_np[1], _DUFFING_CUBIC], dtype=np.float32)
        cubic = np.full((*e_np.shape[:-1], 1), _DUFFING_CUBIC, dtype=np.float32)
        return np.concatenate((e_np[..., :2], cubic), axis=-1)
    return e_np[..., :2]


def _align_embedding_torch(state: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
    e = torch.as_tensor(embedding, dtype=torch.float32, device=state.device)
    while e.ndim < state.ndim:
        e = e.unsqueeze(-2)
    return e


def _align_embedding_np(state: np.ndarray, embedding: np.ndarray) -> np.ndarray:
    e = np.asarray(embedding, dtype=np.float64)
    while e.ndim < state.ndim:
        e = np.expand_dims(e, axis=-2)
    return e


def residual_torch(
    system_id: str, state: torch.Tensor, embedding: torch.Tensor, *, dynamics_alpha: float
) -> torch.Tensor:
    state_t = torch.as_tensor(state, dtype=torch.float32, device=state.device)
    embed_t = _align_embedding_torch(state_t, embedding)
    if system_id in {"single_attractor", "bistable_attractor"}:
        return _duffing_residual_torch(state_t, embed_t, dynamics_alpha=float(dynamics_alpha))
    if system_id == "bistable_limitcycle":
        return _bistable_limitcycle_residual_torch(
            state_t, embed_t, dynamics_alpha=float(dynamics_alpha)
        )
    raise ValueError(f"Unknown system_id={system_id!r}")


def jacobian_state_torch(
    system_id: str,
    state: torch.Tensor,
    embedding: torch.Tensor,
    *,
    dynamics_alpha: float,
) -> torch.Tensor:
    state_t = torch.as_tensor(state, dtype=torch.float32, device=state.device)
    embed_t = _align_embedding_torch(state_t, embedding)
    if system_id in {"single_attractor", "bistable_attractor"}:
        return _duffing_jacobian_state_torch(state_t, embed_t, dynamics_alpha=float(dynamics_alpha))
    if system_id == "bistable_limitcycle":
        return _bistable_limitcycle_jacobian_state_torch(
            state_t, embed_t, dynamics_alpha=float(dynamics_alpha)
        )
    raise ValueError(f"Unknown system_id={system_id!r}")


def jacobian_param_torch(
    system_id: str,
    state: torch.Tensor,
    embedding: torch.Tensor,
    *,
    dynamics_alpha: float,
) -> torch.Tensor:
    state_t = torch.as_tensor(state, dtype=torch.float32, device=state.device)
    embed_t = _align_embedding_torch(state_t, embedding)
    if system_id in {"single_attractor", "bistable_attractor"}:
        return _duffing_jacobian_param_torch(state_t, embed_t, dynamics_alpha=float(dynamics_alpha))
    if system_id == "bistable_limitcycle":
        return _bistable_limitcycle_jacobian_param_torch(
            state_t, embed_t, dynamics_alpha=float(dynamics_alpha)
        )
    raise ValueError(f"Unknown system_id={system_id!r}")


def residual_np(
    system_id: str, state: np.ndarray, embedding: np.ndarray, *, dynamics_alpha: float
) -> np.ndarray:
    state_np = np.asarray(state, dtype=np.float64)
    embed_np = _align_embedding_np(state_np, embedding)
    if system_id in {"single_attractor", "bistable_attractor"}:
        return _duffing_residual_np(state_np, embed_np, dynamics_alpha=float(dynamics_alpha))
    if system_id == "bistable_limitcycle":
        return _bistable_limitcycle_residual_np(
            state_np, embed_np, dynamics_alpha=float(dynamics_alpha)
        )
    raise ValueError(f"Unknown system_id={system_id!r}")


def step_np(
    system_id: str,
    state: np.ndarray,
    action: np.ndarray,
    *,
    embedding: np.ndarray,
    dt: float,
    dynamics_alpha: float,
    clip_limit: float,
) -> np.ndarray:
    next_state = np.asarray(state, dtype=np.float64) + float(dt) * (
        residual_np(system_id, state, embedding, dynamics_alpha=float(dynamics_alpha))
        + np.asarray(action, dtype=np.float64)
    )
    return np.clip(next_state, -float(clip_limit), float(clip_limit))


def _duffing_residual_torch(
    state: torch.Tensor,
    embedding: torch.Tensor,
    *,
    dynamics_alpha: float,
) -> torch.Tensor:
    x = state[..., 0]
    v = state[..., 1]
    a = embedding[..., 0]
    b = embedding[..., 1]
    return torch.stack(
        (
            float(dynamics_alpha) * v,
            float(dynamics_alpha) * (a * v - b * x - _DUFFING_CUBIC * x**3),
        ),
        dim=-1,
    )


def _duffing_jacobian_state_torch(
    state: torch.Tensor,
    embedding: torch.Tensor,
    *,
    dynamics_alpha: float,
) -> torch.Tensor:
    x = state[..., 0]
    a = embedding[..., 0]
    out = torch.zeros(*state.shape[:-1], 2, 2, dtype=state.dtype, device=state.device)
    out[..., 0, 1] = 1.0
    out[..., 1, 0] = -embedding[..., 1] - (3.0 * _DUFFING_CUBIC) * x**2
    out[..., 1, 1] = a
    return float(dynamics_alpha) * out


def _duffing_jacobian_param_torch(
    state: torch.Tensor,
    embedding: torch.Tensor,
    *,
    dynamics_alpha: float,
) -> torch.Tensor:
    del embedding
    x = state[..., 0]
    v = state[..., 1]
    out = torch.zeros(*state.shape[:-1], 2, 2, dtype=state.dtype, device=state.device)
    out[..., 1, 0] = v
    out[..., 1, 1] = -x
    return float(dynamics_alpha) * out


def _duffing_residual_np(
    state: np.ndarray,
    embedding: np.ndarray,
    *,
    dynamics_alpha: float,
) -> np.ndarray:
    x = state[..., 0]
    v = state[..., 1]
    a = embedding[..., 0]
    b = embedding[..., 1]
    return np.stack(
        (
            float(dynamics_alpha) * v,
            float(dynamics_alpha) * (a * v - b * x - _DUFFING_CUBIC * x**3),
        ),
        axis=-1,
    )


def _bistable_limitcycle_local_torch(
    state: torch.Tensor,
    *,
    center_x: float,
    omega: torch.Tensor,
    radius: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    px = state[..., 0] - float(center_x)
    py = state[..., 1]
    r = torch.sqrt(px.square() + py.square()).clamp_min(_EPS)
    k = radius - r
    field = torch.stack((px * k - omega * py, py * k + omega * px), dim=-1)
    jac = torch.zeros(*state.shape[:-1], 2, 2, dtype=state.dtype, device=state.device)
    jac[..., 0, 0] = k - px.square() / r
    jac[..., 0, 1] = -(px * py) / r - omega
    jac[..., 1, 0] = -(px * py) / r + omega
    jac[..., 1, 1] = k - py.square() / r
    return field, jac


def _bistable_limitcycle_residual_torch(
    state: torch.Tensor,
    embedding: torch.Tensor,
    *,
    dynamics_alpha: float,
) -> torch.Tensor:
    omega = embedding[..., 0]
    radius = embedding[..., 1]
    x = state[..., 0]
    w_right = torch.sigmoid(float(_BISTABLE_GATE_SHARPNESS) * x)
    w_left = 1.0 - w_right
    left_field, _left_jac = _bistable_limitcycle_local_torch(
        state,
        center_x=-_BISTABLE_CENTER_OFFSET,
        omega=omega,
        radius=radius,
    )
    right_field, _right_jac = _bistable_limitcycle_local_torch(
        state,
        center_x=_BISTABLE_CENTER_OFFSET,
        omega=omega,
        radius=radius,
    )
    field = w_left.unsqueeze(-1) * left_field + w_right.unsqueeze(-1) * right_field
    return float(dynamics_alpha) * field


def _bistable_limitcycle_jacobian_state_torch(
    state: torch.Tensor,
    embedding: torch.Tensor,
    *,
    dynamics_alpha: float,
) -> torch.Tensor:
    omega = embedding[..., 0]
    radius = embedding[..., 1]
    x = state[..., 0]
    w_right = torch.sigmoid(float(_BISTABLE_GATE_SHARPNESS) * x)
    w_left = 1.0 - w_right
    dw_dx = float(_BISTABLE_GATE_SHARPNESS) * w_right * w_left
    left_field, left_jac = _bistable_limitcycle_local_torch(
        state,
        center_x=-_BISTABLE_CENTER_OFFSET,
        omega=omega,
        radius=radius,
    )
    right_field, right_jac = _bistable_limitcycle_local_torch(
        state,
        center_x=_BISTABLE_CENTER_OFFSET,
        omega=omega,
        radius=radius,
    )
    jac = (
        w_left.unsqueeze(-1).unsqueeze(-1) * left_jac
        + w_right.unsqueeze(-1).unsqueeze(-1) * right_jac
    )
    jac[..., 0, 0] = jac[..., 0, 0] + dw_dx * (right_field[..., 0] - left_field[..., 0])
    jac[..., 1, 0] = jac[..., 1, 0] + dw_dx * (right_field[..., 1] - left_field[..., 1])
    return float(dynamics_alpha) * jac


def _bistable_limitcycle_jacobian_param_torch(
    state: torch.Tensor,
    embedding: torch.Tensor,
    *,
    dynamics_alpha: float,
) -> torch.Tensor:
    omega = embedding[..., 0]
    radius = embedding[..., 1]
    x = state[..., 0]
    w_right = torch.sigmoid(float(_BISTABLE_GATE_SHARPNESS) * x)
    w_left = 1.0 - w_right
    left_px = state[..., 0] + float(_BISTABLE_CENTER_OFFSET)
    right_px = state[..., 0] - float(_BISTABLE_CENTER_OFFSET)
    py = state[..., 1]
    del omega, radius
    domega = torch.stack(
        (
            -(w_left * py + w_right * py),
            w_left * left_px + w_right * right_px,
        ),
        dim=-1,
    )
    dradius = torch.stack(
        (
            w_left * left_px + w_right * right_px,
            (w_left + w_right) * py,
        ),
        dim=-1,
    )
    out = torch.stack((domega, dradius), dim=-1)
    return float(dynamics_alpha) * out


def _bistable_limitcycle_local_np(
    state: np.ndarray,
    *,
    center_x: float,
    omega: np.ndarray,
    radius: np.ndarray,
) -> np.ndarray:
    px = state[..., 0] - float(center_x)
    py = state[..., 1]
    r = np.sqrt(px * px + py * py)
    r = np.maximum(r, _EPS)
    k = radius - r
    return np.stack((px * k - omega * py, py * k + omega * px), axis=-1)


def _bistable_limitcycle_residual_np(
    state: np.ndarray,
    embedding: np.ndarray,
    *,
    dynamics_alpha: float,
) -> np.ndarray:
    omega = embedding[..., 0]
    radius = embedding[..., 1]
    x = state[..., 0]
    w_right = 1.0 / (1.0 + np.exp(-float(_BISTABLE_GATE_SHARPNESS) * x))
    w_left = 1.0 - w_right
    left_field = _bistable_limitcycle_local_np(
        state,
        center_x=-_BISTABLE_CENTER_OFFSET,
        omega=omega,
        radius=radius,
    )
    right_field = _bistable_limitcycle_local_np(
        state,
        center_x=_BISTABLE_CENTER_OFFSET,
        omega=omega,
        radius=radius,
    )
    return float(dynamics_alpha) * (
        w_left[..., None] * left_field + w_right[..., None] * right_field
    )
