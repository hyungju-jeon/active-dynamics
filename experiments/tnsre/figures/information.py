"""State Fisher-information grids for Poisson log-linear observation models."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from actdyn.utils.experiment_runtime import safe_float

from ...experiment_io import reconstruct_loglinear_rate_model
from . import theme
from .records import RunRecord, state_bounds_from_metadata


def logdet_information(
    latent: np.ndarray,
    *,
    metadata: dict[str, Any],
) -> np.ndarray:
    """Compute log det of the state Fisher information for Poisson log-linear observations.

    For mean counts mu(z)=dt*exp(W z + b), H=dmu/dz=diag(mu)W and
    R=diag(mu), so I_z = H^T R^{-1} H = W^T diag(mu) W.
    """
    latent = np.asarray(latent, dtype=np.float64)
    if latent.ndim == 1:
        latent = latent.reshape(1, -1)
    weights, bias, dt = reconstruct_loglinear_rate_model(
        metadata,
        obs_dim=int(metadata.get("observation_dim", 20)),
        latent_dim=int(metadata.get("latent_dim", 2)),
    )
    weights = np.asarray(weights, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    log_rate_hz = latent @ weights.T + bias.reshape(1, -1)
    rate_hz = np.exp(np.clip(log_rate_hz, -20.0, 20.0))
    mean_counts = np.clip(rate_hz * float(dt), 1e-12, 1e12)
    info_mats = np.einsum("nd,di,dj->nij", mean_counts, weights, weights, optimize=True)
    info_mats = 0.5 * (info_mats + np.swapaxes(info_mats, -1, -2))
    info_mats = info_mats + 1e-9 * np.eye(weights.shape[1], dtype=np.float64)[None, :, :]
    sign, logabsdet = np.linalg.slogdet(info_mats)
    return np.where(sign > 0.0, logabsdet, np.nan)


def observation_model_key(metadata: dict[str, Any]) -> tuple[Any, ...]:
    loading_seed = metadata.get("observation_loading_seed")
    if loading_seed is None:
        loading_seed = metadata.get("seed", 0)
    return (
        int(loading_seed),
        int(metadata.get("loading_snr_trajectory_seed", 0)),
        str(metadata.get("env_preset_id", "")),
        int(metadata.get("observation_dim", 20)),
        int(metadata.get("latent_dim", 2)),
        float(metadata.get("dt", 0.01)),
        float(metadata.get("mean_firing_rate_target", 10.0)),
        float(metadata.get("max_firing_rate_target", 100.0)),
        safe_float(metadata.get("loading_target_snr_db")),
    )


def information_reference_records(records: Sequence[RunRecord]) -> list[RunRecord]:
    out: list[RunRecord] = []
    seen: set[tuple[Any, ...]] = set()
    for record in sorted(records, key=lambda item: (item.seed, theme.policy_sort_key(item.policy_id))):
        key = observation_model_key(record.metadata)
        if key in seen:
            continue
        seen.add(key)
        out.append(record)
    return out


def make_information_grid(
    metadata: dict[str, Any],
    *,
    n_grid: int = 121,
    axis_min: float | None = None,
    axis_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if axis_min is None or axis_max is None:
        state_min, state_max = state_bounds_from_metadata(metadata)
    else:
        state_min, state_max = float(axis_min), float(axis_max)
    axis = np.linspace(state_min, state_max, n_grid, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    latent = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    logdet = logdet_information(latent, metadata=metadata).reshape(n_grid, n_grid)
    return axis, axis, logdet


def make_mean_information_grid(
    records: Sequence[RunRecord],
    *,
    n_grid: int = 121,
    axis_min: float | None = None,
    axis_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not records:
        raise ValueError("At least one record is required to compute an information grid")
    x_axis, y_axis, first_grid = make_information_grid(
        records[0].metadata,
        n_grid=n_grid,
        axis_min=axis_min,
        axis_max=axis_max,
    )
    maps = [first_grid.astype(np.float64)]
    for record in records[1:]:
        _x, _y, grid = make_information_grid(
            record.metadata,
            n_grid=n_grid,
            axis_min=axis_min,
            axis_max=axis_max,
        )
        maps.append(grid.astype(np.float64))
    return x_axis, y_axis, np.nanmean(np.stack(maps, axis=0), axis=0)
