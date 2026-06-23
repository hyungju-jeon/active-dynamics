from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from actdyn.environment.vectorfield import ResidualDynamicsCallable
from actdyn.utils.experiment_runtime import read_trace_csv, safe_float
from experiments.experiment_io import get_environment_preset_from_metadata, resolve_artifact_path


def trace_path(run_dir: Path, metadata: dict[str, Any], key: str, fallback_name: str) -> Path:
    """Resolve a run artifact path from metadata, falling back to run_dir/fallback_name."""
    return resolve_artifact_path(run_dir, metadata, key=key, fallback_name=fallback_name)


def read_xy_trace(path: Path, *, max_step: int | None = None) -> np.ndarray:
    """Read true latent position columns as an array with shape (T, 2)."""
    points: list[tuple[float, float]] = []
    for row in read_trace_csv(path):
        row_step = safe_float(row.get("step"))
        x_val = safe_float(row.get("true_x"))
        v_val = safe_float(row.get("true_v"))
        if x_val is None or v_val is None:
            continue
        if max_step is not None and (row_step is None or row_step > int(max_step)):
            continue
        points.append((x_val, v_val))
    if not points:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def read_state_action_trace(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read saved rollout state/action traces.

    Returns steps with shape (T,), true states, model states, and actions with
    shape (T, 2).
    """
    rows = sorted(read_trace_csv(path), key=lambda row: int(float(row["step"])))
    steps = np.asarray([int(float(row["step"])) for row in rows], dtype=int)
    true_state = np.asarray(
        [[float(row["true_x"]), float(row["true_v"])] for row in rows],
        dtype=np.float32,
    ).reshape(-1, 2)
    model_state = np.asarray(
        [[float(row["model_x"]), float(row["model_v"])] for row in rows],
        dtype=np.float32,
    ).reshape(-1, 2)
    action = np.asarray(
        [[float(row["action_x"]), float(row["action_v"])] for row in rows],
        dtype=np.float32,
    ).reshape(-1, 2)
    return steps, true_state, model_state, action


def read_embedding_trace(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read embedding estimate columns e0, e1, ... as an array with shape (T, D)."""
    rows = sorted(read_trace_csv(path), key=lambda row: int(float(row["step"])))
    if not rows:
        return np.empty((0,), dtype=int), np.empty((0, 0), dtype=np.float32)
    e_cols = sorted(
        (key for key in rows[0] if key.startswith("e") and key[1:].isdigit()),
        key=lambda key: int(key[1:]),
    )
    steps = np.asarray([int(float(row["step"])) for row in rows], dtype=int)
    theta = np.asarray(
        [[float(row[col]) for col in e_cols] for row in rows],
        dtype=np.float32,
    )
    return steps, theta


def embedding_at_step(
    path: Path,
    step: int,
    *,
    embedding_dim: int | None = None,
    run_dir: Path | None = None,
) -> np.ndarray:
    """Return the latest embedding estimate at or before step, or the nearest fallback."""
    selected: dict[str, str] | None = None
    selected_step = -math.inf
    fallback: dict[str, str] | None = None
    fallback_step = math.inf
    for row in read_trace_csv(path):
        row_step = safe_float(row.get("step"))
        if row_step is None:
            continue
        if row_step <= step and row_step >= selected_step:
            selected = row
            selected_step = row_step
        if row_step >= step and row_step <= fallback_step:
            fallback = row
            fallback_step = row_step
    row = selected if selected is not None else fallback
    if row is None:
        where = run_dir if run_dir is not None else path
        raise RuntimeError(f"No embedding estimates found for {where}")
    if embedding_dim is None or int(embedding_dim) <= 0:
        cols = [key for key in row if key.startswith("e") and key[1:].isdigit()]
        embedding_dim = len(cols)
    values: list[float] = []
    for idx in range(int(embedding_dim)):
        value = safe_float(row.get(f"e{idx}"))
        if value is None:
            raise RuntimeError(f"Missing e{idx} in {path}")
        values.append(value)
    return np.asarray(values, dtype=np.float32)


def load_planned_trace(run_dir: Path, metadata: dict[str, Any]) -> tuple[np.ndarray, ...] | None:
    """Load planned trajectory trace arrays: steps (K,), paths (K, H, 2+), lengths (K,)."""
    if metadata.get("planned_trajectory_trace_path") is None and not (
        run_dir / "planned_trajectory_trace.npz"
    ).exists():
        return None
    path = trace_path(
        run_dir,
        metadata,
        key="planned_trajectory_trace_path",
        fallback_name="planned_trajectory_trace.npz",
    )
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as data:
        return (
            np.asarray(data["steps"], dtype=int),
            np.asarray(data["paths"], dtype=np.float32),
            np.asarray(data["lengths"], dtype=int),
        )


def planned_xy_cycle_for_step(
    trace: tuple[np.ndarray, ...] | None, step: int
) -> np.ndarray | None:
    """Return the full saved planning cycle active at step as shape (H, 2)."""
    if trace is None:
        return None
    steps, paths, lengths = trace
    steps = np.asarray(steps, dtype=int)
    if steps.size == 0:
        return None
    idx = int(np.searchsorted(steps, int(step), side="right") - 1)
    idx = int(np.clip(idx, 0, steps.size - 1))
    while (
        idx > 0
        and steps[idx - 1] == steps[idx] - 1
        and int(lengths[idx - 1]) == int(lengths[idx]) + 1
    ):
        idx -= 1
    n_points = int(lengths[idx])
    if n_points < 2:
        return None
    path = np.asarray(paths[idx, :n_points, :2], dtype=float)
    path = path[np.all(np.isfinite(path), axis=1)]
    return path if path.shape[0] >= 2 else None


def dynamics_from_metadata(
    metadata: dict[str, Any],
    theta: np.ndarray,
    *,
    estimator: bool,
) -> ResidualDynamicsCallable:
    """Construct TBME residual dynamics from metadata and embedding parameters."""
    env_preset = get_environment_preset_from_metadata(metadata)
    return ResidualDynamicsCallable(
        dynamics_type=env_preset.resolved_dynamics_type(estimator=estimator),
        dyn_params=env_preset.params_from_embedding(theta, estimator=estimator),
        dynamics_alpha=float(metadata.get("dynamics_alpha", 1.0)),
        device="cpu",
    )


def true_dynamics_from_metadata(metadata: dict[str, Any]) -> ResidualDynamicsCallable:
    """Construct the true TBME residual dynamics recorded in run metadata."""
    env_preset = get_environment_preset_from_metadata(metadata)
    theta_true = np.asarray(metadata.get("embedding_true", []), dtype=np.float32)
    if theta_true.size == 0:
        theta_true = np.asarray(metadata.get("true_params_full", []), dtype=np.float32)
    if theta_true.size == 0:
        theta_true = np.asarray(env_preset.true_embedding_vector(), dtype=np.float32)
    return dynamics_from_metadata(metadata, theta_true, estimator=False)
