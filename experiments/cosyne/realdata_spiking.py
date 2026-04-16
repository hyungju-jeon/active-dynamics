from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ReplayDataset:
    dataset_id: str
    source_path: Path
    states: np.ndarray
    spikes: np.ndarray
    dt: float
    metadata: dict[str, Any]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_dataset_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    candidate = (_repo_root() / path).resolve()
    if candidate.exists():
        return candidate
    return path.resolve()


def _standardize_behavior(behavior: np.ndarray, target_dim: int) -> tuple[np.ndarray, dict[str, Any]]:
    arr = np.asarray(behavior, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected behavior to have shape (T, D), got {arr.shape}")
    centered = arr - np.mean(arr, axis=0, keepdims=True)
    if centered.shape[1] > int(target_dim):
        u, s, _vh = np.linalg.svd(centered, full_matrices=False)
        latent = u[:, : int(target_dim)] * s[: int(target_dim)]
        explained = (s[: int(target_dim)] ** 2) / np.maximum(np.sum(s**2), 1e-12)
        meta = {
            "behavior_projection": "pca",
            "behavior_input_dim": int(arr.shape[1]),
            "behavior_output_dim": int(target_dim),
            "pca_explained_variance": [float(x) for x in explained.tolist()],
        }
    else:
        latent = centered[:, : int(target_dim)]
        meta = {
            "behavior_projection": "truncate",
            "behavior_input_dim": int(arr.shape[1]),
            "behavior_output_dim": int(latent.shape[1]),
        }
        if latent.shape[1] < int(target_dim):
            pad = np.zeros((latent.shape[0], int(target_dim) - latent.shape[1]), dtype=np.float64)
            latent = np.concatenate([latent, pad], axis=1)
    scale = np.std(latent, axis=0, keepdims=True)
    scale = np.where(scale > 1e-8, scale, 1.0)
    latent = latent / scale
    meta["behavior_scale"] = [float(x) for x in scale.reshape(-1).tolist()]
    return latent.astype(np.float32), meta


def load_replay_dataset(
    *,
    dataset_id: str,
    dataset_path: str | Path,
    state_key: str = "behavior",
    observation_key: str = "spikes",
    latent_dim: int = 2,
    max_observation_dim: int | None = None,
    time_bin_ms: float = 20.0,
) -> ReplayDataset:
    path = resolve_dataset_path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Replay dataset not found at {path}. Prepare a standardized NPZ with "
            f"'{state_key}' and '{observation_key}' arrays before running this experiment."
        )
    if path.suffix.lower() != ".npz":
        raise ValueError(
            f"Unsupported replay dataset format {path.suffix!r}. "
            "Use a standardized .npz archive for the current TBME scaffold."
        )

    with np.load(path, allow_pickle=True) as data:
        if state_key not in data or observation_key not in data:
            keys = ", ".join(sorted(data.files))
            raise KeyError(
                f"Expected keys '{state_key}' and '{observation_key}' in {path}, found: {keys}"
            )
        behavior = np.asarray(data[state_key], dtype=np.float64)
        spikes = np.asarray(data[observation_key], dtype=np.float64)
        dt = float(np.asarray(data["dt"]).reshape(-1)[0]) if "dt" in data else float(time_bin_ms) / 1000.0
        metadata = {
            "available_keys": [str(key) for key in data.files],
            "dataset_id": str(dataset_id),
        }

    if behavior.shape[0] != spikes.shape[0]:
        raise ValueError(
            f"Behavior and spike arrays must share the time dimension, got {behavior.shape} vs {spikes.shape}"
        )
    if behavior.shape[0] < 8:
        raise ValueError("Replay dataset is too short for train/eval splitting")
    if spikes.ndim != 2:
        raise ValueError(f"Expected spikes to have shape (T, N), got {spikes.shape}")

    states, projection_meta = _standardize_behavior(behavior, target_dim=int(latent_dim))
    metadata.update(projection_meta)

    counts = np.maximum(np.asarray(spikes, dtype=np.float64), 0.0)
    if max_observation_dim is not None and counts.shape[1] > int(max_observation_dim):
        order = np.argsort(np.var(counts, axis=0))[::-1][: int(max_observation_dim)]
        counts = counts[:, order]
        metadata["observation_selection"] = "variance_topk"
        metadata["observation_indices"] = [int(i) for i in order.tolist()]
    metadata["num_timepoints"] = int(counts.shape[0])
    metadata["num_units"] = int(counts.shape[1])

    return ReplayDataset(
        dataset_id=str(dataset_id),
        source_path=path,
        states=states.astype(np.float32, copy=False),
        spikes=counts.astype(np.float32, copy=False),
        dt=float(dt),
        metadata=metadata,
    )


def split_replay_dataset(
    dataset: ReplayDataset,
    *,
    train_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_steps = int(dataset.states.shape[0] - 1)
    if n_steps <= 1:
        raise ValueError("Replay dataset must contain at least two transitions")
    split = int(np.clip(round(n_steps * float(train_fraction)), 1, n_steps - 1))
    train_idx = np.arange(0, split, dtype=np.int64)
    eval_idx = np.arange(split, n_steps, dtype=np.int64)
    return train_idx, eval_idx


def build_transition_matrices(dataset: ReplayDataset) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(dataset.states, dtype=np.float64)
    return states[:-1], states[1:]


def fit_linear_dynamics_ridge(
    x: np.ndarray,
    y: np.ndarray,
    *,
    ridge: float = 1e-3,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.ndim != 2 or y_arr.ndim != 2:
        raise ValueError(f"Expected rank-2 design/target arrays, got {x_arr.shape} and {y_arr.shape}")
    gram = x_arr.T @ x_arr + float(ridge) * np.eye(x_arr.shape[1], dtype=np.float64)
    coef = np.linalg.solve(gram, x_arr.T @ y_arr)
    return coef.astype(np.float64, copy=False)


def predict_linear_dynamics(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64) @ np.asarray(coef, dtype=np.float64)


def evaluate_prediction_mse(x: np.ndarray, y: np.ndarray, coef: np.ndarray) -> float:
    pred = predict_linear_dynamics(x, coef)
    err = pred - np.asarray(y, dtype=np.float64)
    return float(np.mean(err * err))


def evaluate_prediction_r2(x: np.ndarray, y: np.ndarray, coef: np.ndarray) -> float:
    target = np.asarray(y, dtype=np.float64)
    pred = predict_linear_dynamics(x, coef)
    ss_res = float(np.sum((pred - target) ** 2))
    ss_tot = float(np.sum((target - np.mean(target, axis=0, keepdims=True)) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)
