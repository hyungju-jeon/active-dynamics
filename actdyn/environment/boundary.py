from __future__ import annotations

from typing import Sequence

import torch


def _is_enabled(boundary_type: str | None) -> bool:
    return str(boundary_type or "none").lower() != "none"


def _radial_norm(z: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(z, dim=-1, keepdim=True)


def boundary_visibility(
    z: torch.Tensor,
    *,
    boundary_type: str = "none",
    radius: float | None = None,
    box_bounds: Sequence[Sequence[float]] | None = None,
    margin: float = 1.0,
    temperature: float = 0.15,
) -> torch.Tensor:
    """Return a smooth visibility gate that decays near the boundary."""
    if not _is_enabled(boundary_type):
        return torch.ones(*z.shape[:-1], 1, dtype=z.dtype, device=z.device)
    boundary_type = str(boundary_type).lower()
    temp = max(float(temperature), 1e-8)
    if boundary_type == "radial":
        if radius is None:
            raise ValueError("radial boundary requires radius")
        signed_distance = float(radius) - _radial_norm(z)
    elif boundary_type == "box":
        if box_bounds is None:
            raise ValueError("box boundary requires box_bounds")
        low, high = box_bounds
        low_t = torch.as_tensor(low, dtype=z.dtype, device=z.device)
        high_t = torch.as_tensor(high, dtype=z.dtype, device=z.device)
        signed_distance = torch.minimum(z - low_t, high_t - z).amin(dim=-1, keepdim=True)
    else:
        raise ValueError(f"Unsupported boundary_type={boundary_type!r}")
    return torch.sigmoid((signed_distance - float(margin)) / temp)


def boundary_barrier_drift(
    z: torch.Tensor,
    *,
    boundary_type: str = "none",
    radius: float | None = None,
    box_bounds: Sequence[Sequence[float]] | None = None,
    width: float = 0.5,
    strength: float = 5.0,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Return a smooth inward drift near the boundary."""
    if not _is_enabled(boundary_type):
        return torch.zeros_like(z)
    boundary_type = str(boundary_type).lower()
    temp = max(float(temperature), 1e-8)
    if boundary_type == "radial":
        if radius is None:
            raise ValueError("radial boundary requires radius")
        norm = _radial_norm(z).clamp_min(1e-8)
        gate = torch.sigmoid((norm - (float(radius) - float(width))) / temp)
        return -float(strength) * gate * z / norm
    if boundary_type == "box":
        if box_bounds is None:
            raise ValueError("box boundary requires box_bounds")
        low, high = box_bounds
        low_t = torch.as_tensor(low, dtype=z.dtype, device=z.device)
        high_t = torch.as_tensor(high, dtype=z.dtype, device=z.device)
        near_low = torch.sigmoid(((low_t + float(width)) - z) / temp)
        near_high = torch.sigmoid((z - (high_t - float(width))) / temp)
        return float(strength) * (near_low - near_high)
    raise ValueError(f"Unsupported boundary_type={boundary_type!r}")


def project_to_boundary(
    z: torch.Tensor,
    *,
    boundary_type: str = "none",
    radius: float | None = None,
    box_bounds: Sequence[Sequence[float]] | None = None,
) -> torch.Tensor:
    """Project states into the configured boundary set."""
    if not _is_enabled(boundary_type):
        return z
    boundary_type = str(boundary_type).lower()
    if boundary_type == "radial":
        if radius is None:
            raise ValueError("radial boundary requires radius")
        norm = _radial_norm(z).clamp_min(1e-8)
        scale = torch.clamp(float(radius) / norm, max=1.0)
        return z * scale
    if boundary_type == "box":
        if box_bounds is None:
            raise ValueError("box boundary requires box_bounds")
        low, high = box_bounds
        low_t = torch.as_tensor(low, dtype=z.dtype, device=z.device)
        high_t = torch.as_tensor(high, dtype=z.dtype, device=z.device)
        return torch.maximum(torch.minimum(z, high_t), low_t)
    raise ValueError(f"Unsupported boundary_type={boundary_type!r}")
