#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from experiment_common import (
        load_json,
        parse_csv_ints,
        parse_csv_list,
        resolve_session_root,
        write_json,
    )
    from experiment_specs import (
        configure_catalogs,
        describe_catalogs,
        get_environment_preset,
        get_experiment_spec,
        get_policy_spec,
        get_schedule_spec,
        list_experiment_ids,
    )
    from cosyne.planar_systems import (
        env_params_from_embedding,
        get_planar_system_spec,
        has_planar_system_spec,
        jacobian_param_torch,
        jacobian_state_torch,
        residual_np,
        residual_torch,
        sample_initial_state,
        step_np,
        true_embedding,
    )
    from cosyne.realdata_spiking import (
        build_transition_matrices,
        evaluate_prediction_mse,
        evaluate_prediction_r2,
        fit_linear_dynamics_ridge,
        load_replay_dataset,
        split_replay_dataset,
    )
    from cosyne.rbf_filtering import (
        SparseRbfDynamics,
        SparseRbfFilteringModel,
        StructuredLocalRbfParameterMetric,
    )
else:
    from .experiment_common import (
        load_json,
        parse_csv_ints,
        parse_csv_list,
        resolve_session_root,
        write_json,
    )
    from .experiment_specs import (
        configure_catalogs,
        describe_catalogs,
        get_environment_preset,
        get_experiment_spec,
        get_policy_spec,
        get_schedule_spec,
        list_experiment_ids,
    )
    from .cosyne.planar_systems import (
        env_params_from_embedding,
        get_planar_system_spec,
        has_planar_system_spec,
        jacobian_param_torch,
        jacobian_state_torch,
        residual_np,
        residual_torch,
        sample_initial_state,
        step_np,
        true_embedding,
    )
    from .cosyne.realdata_spiking import (
        build_transition_matrices,
        evaluate_prediction_mse,
        evaluate_prediction_r2,
        fit_linear_dynamics_ridge,
        load_replay_dataset,
        split_replay_dataset,
    )
    from .cosyne.rbf_filtering import (
        SparseRbfDynamics,
        SparseRbfFilteringModel,
        StructuredLocalRbfParameterMetric,
    )

WRITING_REFERENCE = "docs/active-dynamics-writing/methods.tex"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _current_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_repo_root()),
            text=True,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolved_policy_type(policy_id: str, policy_spec: Any | None) -> str:
    configured = None if policy_spec is None else getattr(policy_spec, "policy_type", None)
    if isinstance(configured, str) and configured.strip():
        return configured.strip()
    if str(policy_id) == "random":
        return "random"
    if str(policy_id) == "off_policy":
        return "off-policy"
    return "mpc-icem"


def _resolved_estimator_system_id(env_preset: Any) -> str:
    configured = getattr(env_preset, "estimator_system_id", None)
    if isinstance(configured, str) and configured.strip():
        return configured.strip()
    return str(env_preset.system_id)


def _environment_summary(env_preset: Any) -> dict[str, Any]:
    system_id = str(env_preset.system_id)
    estimator_system_id = _resolved_estimator_system_id(env_preset)
    if has_planar_system_spec(system_id):
        system_spec = get_planar_system_spec(system_id)
        estimator_spec = get_planar_system_spec(estimator_system_id)
        return {
            "system_id": str(system_spec.system_id),
            "system_label": str(system_spec.label),
            "dynamics_type": str(system_spec.dynamics_type),
            "estimator_system_id": str(estimator_spec.system_id),
            "estimator_system_label": str(estimator_spec.label),
            "estimator_dynamics_type": str(estimator_spec.dynamics_type),
            "true_embedding": [float(x) for x in true_embedding(system_spec.system_id).tolist()],
        }
    return {
        "system_id": system_id,
        "system_label": str(getattr(env_preset, "system_label", None) or system_id),
        "dynamics_type": "replay_dataset" if bool(getattr(env_preset, "real_data", False)) else "unknown",
        "estimator_system_id": estimator_system_id,
        "estimator_system_label": None,
        "estimator_dynamics_type": None,
        "true_embedding": None,
    }


def _build_runtime_experiment_config(
    *,
    run_dir: Path,
    seed: int,
    total_steps: int,
    experiment_kind: str,
    policy_id: str,
    env_preset: Any,
    schedule_spec: Any,
    policy_spec: Any | None = None,
):
    from actdyn.config import ExperimentConfig

    exp_config = ExperimentConfig()
    exp_config.seed = int(seed)
    exp_config.device = "cpu"
    exp_config.results_dir = str(run_dir)
    exp_config.action_dim = int(env_preset.action_dim)
    exp_config.observation_dim = int(env_preset.observation_dim)
    exp_config.latent_dim = int(env_preset.latent_dim)
    exp_config.dt = float(env_preset.dt)
    exp_config.run_analysis = False
    exp_config.run_offline = False

    exp_config.environment.env_dt = float(env_preset.dt)
    exp_config.environment.env_noise_scale = float(env_preset.state_noise)
    exp_config.environment.env_action_bounds = [
        -float(env_preset.action_max),
        float(env_preset.action_max),
    ]
    exp_config.environment.env_x_range = float(env_preset.x_range)
    exp_config.environment.env_alpha = float(env_preset.dynamics_alpha)
    exp_config.environment.observation_type = "log-linear"
    exp_config.environment.obs_noise_type = str(env_preset.observation_noise_type)
    exp_config.environment.obs_noise_scale = float(env_preset.observation_noise_scale)
    exp_config.environment.action_type = "identity"

    exp_config.model.is_residual = True
    exp_config.model.dyn_dt = float(env_preset.dt)
    exp_config.model.dyn_alpha = float(env_preset.dynamics_alpha)
    exp_config.model.dynamics_type = "rbf" if str(experiment_kind) == "rbf" else "linear"
    exp_config.model.emb_k_theta = int(schedule_spec.update_interval)
    exp_config.model.emb_state_init_uncertainty = float(env_preset.state_init_uncertainty)

    policy_type = _resolved_policy_type(policy_id, policy_spec)
    exp_config.policy.policy_type = policy_type
    if policy_type == "mpc-icem":
        exp_config.policy.policy_type = "mpc-icem"
        exp_config.policy.mpc_horizon = int(schedule_spec.planning_horizon)

    exp_config.training.total_steps = int(total_steps)
    exp_config.training.train_every = int(total_steps) + 1
    return exp_config


def _write_trace_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _to_xy_pair(value: Any) -> tuple[float, float]:
    import torch

    flat = torch.as_tensor(value).detach().reshape(-1)
    if flat.numel() == 0:
        return 0.0, 0.0
    if flat.numel() == 1:
        return float(flat[0].item()), 0.0
    return float(flat[0].item()), float(flat[1].item())


def _as_bool(value: Any) -> bool:
    import torch

    if isinstance(value, bool):
        return value
    if isinstance(value, torch.Tensor):
        flat = value.detach().reshape(-1)
        return bool(flat[0].item()) if flat.numel() > 0 else False
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "t"}
    return bool(value)


def _current_plan_index(policy: Any) -> int:
    chunk = max(1, int(getattr(policy, "chunk", 1)))
    count = max(0, int(getattr(policy, "count", 0)))
    if count <= 0:
        return 0
    return (count - 1) % chunk


def _extract_remaining_plan_actions(policy: Any):
    import torch

    plan = None
    elite_actions = getattr(policy, "elite_actions", None)
    if elite_actions is not None:
        elite_actions = torch.as_tensor(elite_actions).detach()
        if elite_actions.ndim == 3 and elite_actions.shape[0] > 0:
            plan = elite_actions[0]
    if plan is None:
        mean_actions = getattr(policy, "mean", None)
        if mean_actions is not None:
            mean_actions = torch.as_tensor(mean_actions).detach()
            if mean_actions.ndim == 2:
                plan = mean_actions
            elif mean_actions.ndim == 3 and mean_actions.shape[0] > 0:
                plan = mean_actions[0]
    if plan is None or plan.ndim != 2 or plan.shape[0] == 0:
        return None
    start = min(_current_plan_index(policy), int(plan.shape[0] - 1))
    return plan[start:].unsqueeze(0)


def _clone_filter_belief_state(model: Any) -> dict[str, Any]:
    def _clone_value(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): _clone_value(v) for k, v in value.items()}
        if hasattr(value, "detach"):
            return value.detach().clone()
        return value

    return {
        "e": _clone_value(getattr(model, "e", None)),
        "z": _clone_value(getattr(model, "z", None)),
        "_state": _clone_value(getattr(model, "_state", None)),
        "last_information": dict(getattr(model, "last_information", {}) or {}),
        "_theta_score_block": _clone_value(getattr(model, "_theta_score_block", None)),
        "_theta_info_block": _clone_value(getattr(model, "_theta_info_block", None)),
        "_theta_info_diag_block": _clone_value(getattr(model, "_theta_info_diag_block", None)),
        "_theta_active_mask_block": _clone_value(getattr(model, "_theta_active_mask_block", None)),
        "_theta_sensitivity": _clone_value(getattr(model, "_theta_sensitivity", None)),
        "_theta_block_steps": int(getattr(model, "_theta_block_steps", 0)),
    }


def _restore_filter_belief_state(model: Any, snapshot: dict[str, Any]) -> None:
    if snapshot.get("e") is not None:
        model.e = snapshot["e"]
    if snapshot.get("z") is not None:
        model.z = snapshot["z"]
    if snapshot.get("_state") is not None:
        model._state = snapshot["_state"]
    model.last_information = dict(snapshot.get("last_information", {}) or {})
    if snapshot.get("_theta_score_block") is not None:
        model._theta_score_block = snapshot["_theta_score_block"]
    if snapshot.get("_theta_info_block") is not None:
        model._theta_info_block = snapshot["_theta_info_block"]
    if snapshot.get("_theta_info_diag_block") is not None:
        model._theta_info_diag_block = snapshot["_theta_info_diag_block"]
    if snapshot.get("_theta_active_mask_block") is not None:
        model._theta_active_mask_block = snapshot["_theta_active_mask_block"]
    if snapshot.get("_theta_sensitivity") is not None:
        model._theta_sensitivity = snapshot["_theta_sensitivity"]
    model._theta_block_steps = int(snapshot.get("_theta_block_steps", 0))
    if hasattr(model, "set_params") and getattr(model, "e", None) is not None:
        model.set_params(model.e["m"])


def _ensure_batch_time_tensor(value: Any, *, device: Any):
    import torch

    if value is None:
        return None
    tensor = torch.as_tensor(value, device=device)
    if tensor.ndim == 1:
        tensor = tensor.reshape(1, 1, -1)
    elif tensor.ndim == 2:
        tensor = tensor.unsqueeze(1) if tensor.shape[0] == 1 else tensor.unsqueeze(0)
    return tensor


def _predictive_only_embedding_step(model: Any, action: Any) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    model._normalize_embedding_belief()
    model._ensure_state_belief_shapes(batch_size=model.e["m"].shape[0])
    q = F.softplus(model.dynamics.logvar).diag_embed().unsqueeze(0) * model.dt
    eye = torch.eye(model.latent_dim, device=model.device).unsqueeze(0).unsqueeze(0)
    action_bt = _ensure_batch_time_tensor(action, device=model.device)
    if action_bt is not None and model.action_encoder is not None:
        u_enc = model.action_encoder(action_bt, model.z["m"])
    else:
        u_enc = action_bt
    fz = model.Fz(model.z["m"], model.e["m"])
    dfdz = fz * model.dt + eye
    pred_m = model.predict(action=u_enc)
    pred_cov = model._project_spd(dfdz @ model.z["P"] @ dfdz.transpose(-1, -2) + q + 1e-6 * eye)
    model.z = {"m": pred_m.detach(), "P": pred_cov.detach()}
    model._state = pred_m.detach()
    model.last_information = {
        "I_z_t": 0.0,
        "I_theta_t": 0.0,
        "Pz00": float(pred_cov[..., 0, 0].mean().item()),
        "Pz01": float(pred_cov[..., 0, 1].mean().item()) if pred_cov.shape[-1] > 1 else 0.0,
        "Pz11": float(pred_cov[..., 1, 1].mean().item()) if pred_cov.shape[-1] > 1 else 0.0,
    }
    return {
        "env_action": u_enc[..., -1:, :] if u_enc is not None else None,
        "latent_state": model._state,
    }


def _predict_planned_xy_trajectory(
    *, model: Any, policy: Any, transition: dict[str, Any]
) -> np.ndarray | None:
    import torch

    if getattr(policy, "metric", None) is None:
        return None
    planned_actions = _extract_remaining_plan_actions(policy)
    if planned_actions is None:
        return None
    model_state = transition.get("model_state")
    if model_state is None:
        return None
    state = torch.as_tensor(model_state).detach()
    if state.ndim == 1:
        state = state.reshape(1, 1, -1)
    elif state.ndim == 2:
        state = state.unsqueeze(0)
    if state.ndim != 3:
        return None
    device = getattr(model, "device", state.device)
    state = state.to(device)
    planned_actions = planned_actions.to(device)
    prev_state = None
    try:
        current_state = model.get_state()
        if current_state is not None:
            prev_state = current_state.detach().clone()
    except Exception:
        prev_state = None
    try:
        with torch.no_grad():
            model.set_state(state)
            encoded_actions = (
                model.action_encoder(planned_actions)
                if model.action_encoder is not None
                else planned_actions
            )
            predicted = model.predict(encoded_actions)
            trajectory = torch.cat([state, predicted], dim=-2)
    except Exception:
        return None
    finally:
        if prev_state is not None:
            model.set_state(prev_state)
    trajectory = trajectory.detach().cpu().reshape(-1, trajectory.shape[-1]).numpy()
    xy = np.zeros((trajectory.shape[0], 2), dtype=np.float32)
    if trajectory.shape[1] > 0:
        xy[:, 0] = trajectory[:, 0].astype(np.float32, copy=False)
    if trajectory.shape[1] > 1:
        xy[:, 1] = trajectory[:, 1].astype(np.float32, copy=False)
    return xy


def _extract_rollout_metrics(results_path: Path) -> dict[str, Any]:
    from actdyn.utils import save_load
    import torch

    rollouts_dir = results_path / "rollouts"
    if not rollouts_dir.exists():
        return {
            "rollout_steps": 0,
            "state_error_mean": None,
            "state_error_final": None,
            "nan_detected": False,
            "action_abs_mean": None,
            "action_abs_max": None,
        }
    try:
        rollout = save_load.load_and_concatenate_rollouts(str(rollouts_dir))
        env_state = rollout["env_state"][0]
        model_state = rollout["model_state"][0]
        diff = torch.linalg.norm(env_state - model_state, dim=-1)
        action = rollout["action"][0]
        nan_detected = bool(torch.isnan(diff).any().item() or torch.isnan(action).any().item())
        return {
            "rollout_steps": int(diff.shape[0]),
            "state_error_mean": float(diff.mean().item()),
            "state_error_final": float(diff[-1].item()),
            "nan_detected": nan_detected,
            "action_abs_mean": float(action.abs().mean().item()),
            "action_abs_max": float(action.abs().max().item()),
        }
    except Exception:
        return {
            "rollout_steps": 0,
            "state_error_mean": None,
            "state_error_final": None,
            "nan_detected": False,
            "action_abs_mean": None,
            "action_abs_max": None,
        }


def _build_system_jacobians(*, system_id: str, dynamics_alpha: float):
    import torch

    def _fe(z: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return jacobian_param_torch(
            system_id,
            z,
            e,
            dynamics_alpha=float(dynamics_alpha),
        )

    def _fz(z: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return jacobian_state_torch(
            system_id,
            z,
            e,
            dynamics_alpha=float(dynamics_alpha),
        )

    return _fe, _fz


def _rollout_no_input(
    z0,
    e,
    *,
    system_id: str,
    horizon: int,
    dt: float,
    dynamics_alpha: float,
):
    import torch

    z = z0.clone()
    traj = [z]
    for _ in range(horizon):
        drift = residual_torch(
            system_id,
            z,
            e,
            dynamics_alpha=float(dynamics_alpha),
        )
        z = z + float(dt) * drift
        traj.append(z)
    return torch.stack(traj, dim=1)


def _trajectory_r2(
    e_est,
    e_true,
    *,
    system_id: str,
    dt: float,
    dynamics_alpha: float,
    horizon: int,
    n_starts: int,
    rng: np.random.Generator,
    device,
) -> float:
    import torch

    starts = torch.as_tensor(
        rng.uniform(low=-3.0, high=3.0, size=(n_starts, 2)),
        dtype=torch.float32,
        device=device,
    )
    e_true_batch = e_true.reshape(1, 2).repeat(n_starts, 1)
    e_est_batch = e_est.reshape(1, 2).repeat(n_starts, 1)
    with torch.no_grad():
        traj_true = _rollout_no_input(
            starts,
            e_true_batch,
            system_id=system_id,
            horizon=horizon,
            dt=dt,
            dynamics_alpha=dynamics_alpha,
        )
        traj_est = _rollout_no_input(
            starts,
            e_est_batch,
            system_id=system_id,
            horizon=horizon,
            dt=dt,
            dynamics_alpha=dynamics_alpha,
        )
        y_true = traj_true.reshape(-1)
        y_est = traj_est.reshape(-1)
        sse = torch.sum((y_true - y_est) ** 2)
        sst = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 0.0 if float(sst.item()) <= 1e-12 else float((1.0 - sse / sst).item())


def _clip_state_np(state: np.ndarray, limit: float) -> np.ndarray:
    return np.clip(np.asarray(state, dtype=np.float64), -float(limit), float(limit))


def _system_step_np(
    *,
    system_id: str,
    state: np.ndarray,
    action: np.ndarray,
    embedding: np.ndarray,
    dt: float,
    dynamics_alpha: float,
    clip_limit: float,
) -> np.ndarray:
    return step_np(
        system_id,
        state,
        action,
        embedding=np.asarray(embedding, dtype=np.float64),
        dt=float(dt),
        dynamics_alpha=float(dynamics_alpha),
        clip_limit=float(clip_limit),
    )


def _rbf_grid(axis_limit: float, num_grid_pts: int) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(-float(axis_limit), float(axis_limit), int(num_grid_pts), dtype=np.float64)
    gx, gy = np.meshgrid(axis, axis, indexing="ij")
    centers = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=1)
    return axis, centers


def _nearest_axis_index(axis: np.ndarray, value: float) -> int:
    if axis.size <= 1:
        return 0
    idx = int(np.searchsorted(axis, float(value)))
    idx = int(np.clip(idx, 0, axis.size - 1))
    if idx > 0 and abs(float(axis[idx - 1]) - float(value)) <= abs(float(axis[idx]) - float(value)):
        idx -= 1
    return idx


def _rbf_active_indices(state: np.ndarray, axis: np.ndarray, support_radius: int) -> np.ndarray:
    n = int(axis.shape[0])
    x_idx = _nearest_axis_index(axis, float(state[0]))
    y_idx = _nearest_axis_index(axis, float(state[1]))
    idxs: list[int] = []
    for i in range(max(0, x_idx - support_radius), min(n, x_idx + support_radius + 1)):
        for j in range(max(0, y_idx - support_radius), min(n, y_idx + support_radius + 1)):
            if abs(i - x_idx) + abs(j - y_idx) <= int(support_radius):
                idxs.append(i * n + j)
    return np.asarray(idxs, dtype=np.int64)


def _rbf_local_features(
    state: np.ndarray,
    *,
    centers: np.ndarray,
    axis: np.ndarray,
    support_radius: int,
    width: float,
) -> tuple[np.ndarray, np.ndarray]:
    active = _rbf_active_indices(state, axis, support_radius)
    if active.size == 0:
        return active, np.zeros((0,), dtype=np.float64)
    local_centers = centers[active]
    scaled = (local_centers - np.asarray(state, dtype=np.float64)) / max(float(width), 1e-6)
    phi = np.exp(-0.5 * np.sum(scaled * scaled, axis=1))
    return active, phi.astype(np.float64, copy=False)


def _rbf_predict_drift(
    state: np.ndarray,
    *,
    weights: np.ndarray,
    centers: np.ndarray,
    axis: np.ndarray,
    support_radius: int,
    width: float,
) -> np.ndarray:
    active, phi = _rbf_local_features(
        state,
        centers=centers,
        axis=axis,
        support_radius=support_radius,
        width=width,
    )
    if active.size == 0:
        return np.zeros((2,), dtype=np.float64)
    return phi @ weights[active]


def _rbf_predict_batch(
    states: np.ndarray,
    *,
    weights: np.ndarray,
    centers: np.ndarray,
    axis: np.ndarray,
    support_radius: int,
    width: float,
) -> np.ndarray:
    states_arr = np.asarray(states, dtype=np.float64).reshape(-1, 2)
    preds = np.zeros((states_arr.shape[0], 2), dtype=np.float64)
    for idx, state in enumerate(states_arr):
        preds[idx] = _rbf_predict_drift(
            state,
            weights=weights,
            centers=centers,
            axis=axis,
            support_radius=support_radius,
            width=width,
        )
    return preds.reshape(np.asarray(states, dtype=np.float64).shape)


def _rbf_information_gain(
    phi: np.ndarray, precision: np.ndarray, obs_var: float, n_outputs: int = 2
) -> float:
    if phi.size == 0:
        return 0.0
    prior_var = 1.0 / np.maximum(np.asarray(precision, dtype=np.float64), 1e-8)
    scaled = (np.asarray(phi, dtype=np.float64) ** 2) * prior_var / max(float(obs_var), 1e-8)
    return float(n_outputs * np.sum(np.log1p(scaled)))


def _rbf_update_local_posterior(
    *,
    weights: np.ndarray,
    precision: np.ndarray,
    active: np.ndarray,
    phi: np.ndarray,
    target: np.ndarray,
    obs_var: float,
    n_outputs: int = 2,
) -> tuple[float, float, float]:
    if active.size == 0:
        return 0.0, 0.0, 0.0
    local_precision = np.maximum(precision[active], 1e-8)
    local_var = 1.0 / local_precision
    pred = phi @ weights[active]
    resid = np.asarray(target, dtype=np.float64) - pred
    gain = (local_var * phi) / max(float(obs_var), 1e-8)
    weights[active] = weights[active] + gain[:, None] * resid[None, :]
    precision[active] = local_precision + float(n_outputs) * (phi**2) / max(float(obs_var), 1e-8)
    updated_var = 1.0 / np.maximum(precision[active], 1e-8)
    return (
        float(np.mean(updated_var)),
        float(np.max(updated_var)),
        float(np.linalg.norm(resid)),
    )


def _rbf_evaluate_action_sequence(
    *,
    init_state: np.ndarray,
    action_seq: np.ndarray,
    weights: np.ndarray,
    precision: np.ndarray,
    centers: np.ndarray,
    axis: np.ndarray,
    support_radius: int,
    width: float,
    gamma: float,
    dt: float,
    obs_var: float,
    clip_limit: float,
) -> tuple[float, np.ndarray]:
    score = 0.0
    z = np.asarray(init_state, dtype=np.float64).reshape(2)
    traj = [z.copy()]
    plan_precision = np.asarray(precision, dtype=np.float64).copy()
    for t, action in enumerate(np.asarray(action_seq, dtype=np.float64)):
        active, phi = _rbf_local_features(
            z,
            centers=centers,
            axis=axis,
            support_radius=support_radius,
            width=width,
        )
        if active.size > 0:
            score += (float(gamma) ** t) * _rbf_information_gain(
                phi, plan_precision[active], obs_var
            )
            plan_precision[active] = np.maximum(plan_precision[active], 1e-8) + 2.0 * (
                phi**2
            ) / max(float(obs_var), 1e-8)
            drift = phi @ weights[active]
        else:
            drift = np.zeros((2,), dtype=np.float64)
        z = _clip_state_np(
            z + float(dt) * (drift + np.asarray(action, dtype=np.float64)), clip_limit
        )
        traj.append(z.copy())
    return float(score), np.asarray(traj, dtype=np.float32)


def _plan_rbf_actions(
    *,
    state: np.ndarray,
    weights: np.ndarray,
    precision: np.ndarray,
    centers: np.ndarray,
    axis: np.ndarray,
    support_radius: int,
    width: float,
    action_max: float,
    horizon: int,
    gamma: float,
    dt: float,
    obs_var: float,
    clip_limit: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    horizon = max(1, int(horizon))
    block_size = 4 if horizon > 1 else 1
    n_blocks = max(1, int(np.ceil(horizon / block_size)))
    mean = np.zeros((n_blocks, 2), dtype=np.float64)
    std = np.full((n_blocks, 2), max(float(action_max), 1e-3) * 0.75, dtype=np.float64)
    best_seq = np.zeros((horizon, 2), dtype=np.float64)
    best_traj = np.asarray(state, dtype=np.float32).reshape(1, 2)
    best_score = -np.inf
    num_samples = 48
    num_elite = 8
    for _ in range(3):
        block_samples = rng.normal(loc=mean, scale=std, size=(num_samples, n_blocks, 2))
        block_samples = np.clip(block_samples, -float(action_max), float(action_max))
        seq_samples = np.repeat(block_samples, block_size, axis=1)[:, :horizon, :]
        scores = np.full((num_samples,), -np.inf, dtype=np.float64)
        trajs: list[np.ndarray] = []
        for idx in range(num_samples):
            score, traj = _rbf_evaluate_action_sequence(
                init_state=state,
                action_seq=seq_samples[idx],
                weights=weights,
                precision=precision,
                centers=centers,
                axis=axis,
                support_radius=support_radius,
                width=width,
                gamma=gamma,
                dt=dt,
                obs_var=obs_var,
                clip_limit=clip_limit,
            )
            scores[idx] = score
            trajs.append(traj)
        elite_idx = np.argsort(scores)[-num_elite:]
        mean = np.mean(block_samples[elite_idx], axis=0)
        std = np.maximum(
            np.std(block_samples[elite_idx], axis=0), max(float(action_max), 1e-3) * 0.10
        )
        top_idx = int(np.argmax(scores))
        if float(scores[top_idx]) > float(best_score):
            best_score = float(scores[top_idx])
            best_seq = np.asarray(seq_samples[top_idx], dtype=np.float64)
            best_traj = np.asarray(trajs[top_idx], dtype=np.float32)
    return best_seq, best_traj, float(best_score)


def _rbf_dynamics_mse(
    *,
    system_id: str,
    weights: np.ndarray,
    centers: np.ndarray,
    axis: np.ndarray,
    support_radius: int,
    width: float,
    true_embedding_vec: np.ndarray,
    dynamics_alpha: float,
    eval_states: np.ndarray,
) -> float:
    true_drift = residual_np(
        system_id,
        eval_states,
        np.asarray(true_embedding_vec, dtype=np.float64),
        dynamics_alpha=float(dynamics_alpha),
    )
    est_drift = _rbf_predict_batch(
        eval_states,
        weights=weights,
        centers=centers,
        axis=axis,
        support_radius=support_radius,
        width=width,
    )
    return float(np.mean((true_drift - est_drift) ** 2))


def _trajectory_r2_rbf(
    *,
    system_id: str,
    weights: np.ndarray,
    centers: np.ndarray,
    axis: np.ndarray,
    support_radius: int,
    width: float,
    true_embedding_vec: np.ndarray,
    dt: float,
    dynamics_alpha: float,
    horizon: int,
    n_starts: int,
    rng: np.random.Generator,
    clip_limit: float,
) -> float:
    true_state = rng.uniform(low=-3.0, high=3.0, size=(int(n_starts), 2)).astype(np.float64)
    est_state = true_state.copy()
    true_traj = [true_state.copy()]
    est_traj = [est_state.copy()]
    zero_action = np.zeros((int(n_starts), 2), dtype=np.float64)
    for _ in range(int(horizon)):
        true_state = _system_step_np(
            system_id=system_id,
            state=true_state,
            action=zero_action,
            embedding=np.asarray(true_embedding_vec, dtype=np.float64),
            dt=dt,
            dynamics_alpha=dynamics_alpha,
            clip_limit=clip_limit,
        )
        est_drift = _rbf_predict_batch(
            est_state,
            weights=weights,
            centers=centers,
            axis=axis,
            support_radius=support_radius,
            width=width,
        )
        est_state = _clip_state_np(est_state + float(dt) * est_drift, clip_limit)
        true_traj.append(true_state.copy())
        est_traj.append(est_state.copy())
    y_true = np.asarray(true_traj, dtype=np.float64).reshape(-1)
    y_est = np.asarray(est_traj, dtype=np.float64).reshape(-1)
    sse = float(np.sum((y_true - y_est) ** 2))
    sst = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    return 0.0 if sst <= 1e-12 else float(1.0 - sse / sst)


def _rbf_acquisition_map(
    *,
    weights: np.ndarray,
    precision: np.ndarray,
    map_axis: np.ndarray,
    centers: np.ndarray,
    axis: np.ndarray,
    support_radius: int,
    width: float,
    obs_var: float,
) -> np.ndarray:
    out = np.zeros((map_axis.shape[0], map_axis.shape[0]), dtype=np.float32)
    for ix, x_val in enumerate(map_axis):
        for iy, y_val in enumerate(map_axis):
            state = np.asarray([x_val, y_val], dtype=np.float64)
            active, phi = _rbf_local_features(
                state,
                centers=centers,
                axis=axis,
                support_radius=support_radius,
                width=width,
            )
            if active.size > 0:
                out[iy, ix] = float(_rbf_information_gain(phi, precision[active], obs_var))
    return out


def _set_vectorfield_params(env: Any, params: Any) -> None:
    flat_vals = params.reshape(-1).tolist() if hasattr(params, "reshape") else list(params)
    set_ok = False
    if hasattr(env, "dynamics"):
        dyn = env.dynamics
        if hasattr(dyn, "_set_params"):
            dyn._set_params(*flat_vals)
            set_ok = True
        elif hasattr(dyn, "set_params"):
            try:
                dyn.set_params(params)
            except TypeError:
                dyn.set_params(*flat_vals)
            set_ok = True
    if not set_ok and hasattr(env, "set_params"):
        try:
            env.set_params(params)
        except TypeError:
            env.set_params(*flat_vals)
        set_ok = True
    if set_ok and hasattr(env, "dynamics"):
        import torch

        dyn_params = torch.as_tensor(
            flat_vals, device=getattr(env, "device", "cpu"), dtype=torch.float32
        )
        env.dynamics.dyn_params = dyn_params.unsqueeze(0)


class _VectorFieldDynamicsAdapter:
    def __init__(self, dynamics_obj: Any, *, param_formatter: Any | None = None):
        self.dynamics_obj = dynamics_obj
        self.param_formatter = param_formatter

    def __call__(self, state):
        return self.dynamics_obj(state)

    def set_params(self, *params) -> None:
        import torch

        def _as_param_tensor(value: Any) -> torch.Tensor:
            if hasattr(value, "detach"):
                tensor = value.detach().to(
                    getattr(self.dynamics_obj, "device", "cpu"),
                    dtype=torch.float32,
                )
            else:
                tensor = torch.as_tensor(
                    value,
                    device=getattr(self.dynamics_obj, "device", "cpu"),
                    dtype=torch.float32,
                )
            return tensor

        def _format_params(value: Any) -> Any:
            if self.param_formatter is None:
                return value
            return self.param_formatter(value)

        if len(params) == 1:
            value = params[0]
            value = _format_params(value)
            if hasattr(value, "detach"):
                value_t = value.detach()
                if value_t.ndim > 1:
                    self.dynamics_obj.set_params(value_t)
                    return
                flat_vals = value_t.reshape(-1).tolist()
            elif isinstance(value, (list, tuple)):
                flat_vals = list(value)
            else:
                flat_vals = [value]
        else:
            if self.param_formatter is not None:
                stacked = torch.stack([_as_param_tensor(p).reshape(-1) for p in params], dim=-1)
                if stacked.shape[0] == 1:
                    stacked = stacked.reshape(-1)
                self.set_params(_format_params(stacked))
                return
            if any(hasattr(p, "numel") and int(p.numel()) > 1 for p in params):
                tensor_params = [
                    (
                        p.detach().to(
                            getattr(self.dynamics_obj, "device", "cpu"), dtype=torch.float32
                        )
                        if hasattr(p, "detach")
                        else torch.as_tensor(
                            p,
                            device=getattr(self.dynamics_obj, "device", "cpu"),
                            dtype=torch.float32,
                        )
                    )
                    for p in params
                ]
                self.dynamics_obj.set_params(*tensor_params)
                return
            flat_vals = [
                float(p.detach().item()) if hasattr(p, "detach") else float(p) for p in params
            ]
        if hasattr(self.dynamics_obj, "_set_params"):
            self.dynamics_obj._set_params(*flat_vals)
            return
        if hasattr(self.dynamics_obj, "set_params"):
            try:
                self.dynamics_obj.set_params(flat_vals)
            except TypeError:
                self.dynamics_obj.set_params(*flat_vals)


def _build_metadata(
    *,
    exp_id: str,
    policy_id: str,
    seed: int,
    total_steps: int,
    run_dir: Path,
    status: str,
    start_time: str,
    end_time: str,
    runtime_sec: float,
    results_path: Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "commit": _current_commit(),
        "seed": seed,
        "exp_id": exp_id,
        "policy_id": policy_id,
        "total_steps": total_steps,
        "base_dir": str(run_dir),
        "status": status,
        "start_time": start_time,
        "end_time": end_time,
        "runtime_sec": runtime_sec,
        "results_path": str(results_path),
    }
    if extra:
        payload.update(extra)
    return payload


def _build_metric(
    *,
    objective_kind: str,
    model: Any,
    Fe_net: Any,
    Fz_net: Any,
    gamma: float,
    device: str,
    sampling_variance_samples: int,
    sampling_variance_seed: int | None,
):
    if __package__ in {None, ""}:
        from cosyne.objectives import (
            dynamics as build_dynamics_metric,
            e_optimality as build_e_optimality_metric,
            fully_observable_parameter_eig,
            parameter_eig,
            sampling_variance as build_sampling_variance_metric,
            state_information as build_state_information_metric,
        )
    else:
        from .cosyne.objectives import (
            dynamics as build_dynamics_metric,
            e_optimality as build_e_optimality_metric,
            fully_observable_parameter_eig,
            parameter_eig,
            sampling_variance as build_sampling_variance_metric,
            state_information as build_state_information_metric,
        )

    if objective_kind == "parameter_eig":
        return parameter_eig(model=model, Fe_net=Fe_net, Fz_net=Fz_net, gamma=gamma, device=device)
    if objective_kind == "e_optimality":
        return build_e_optimality_metric(
            model=model,
            Fe_net=Fe_net,
            Fz_net=Fz_net,
            gamma=gamma,
            device=device,
        )
    if objective_kind == "fully_observable_parameter_eig":
        return fully_observable_parameter_eig(
            model=model,
            Fe_net=Fe_net,
            Fz_net=Fz_net,
            gamma=gamma,
            device=device,
        )
    if objective_kind == "state_information":
        return build_state_information_metric(
            model=model,
            Fe_net=Fe_net,
            Fz_net=Fz_net,
            gamma=gamma,
            device=device,
        )
    if objective_kind == "dynamics":
        return build_dynamics_metric(
            model=model,
            Fe_net=Fe_net,
            Fz_net=Fz_net,
            gamma=gamma,
            device=device,
        )
    if objective_kind == "sampling_variance":
        return build_sampling_variance_metric(
            model=model,
            Fe_net=Fe_net,
            Fz_net=Fz_net,
            gamma=gamma,
            device=device,
            num_parameter_samples=int(sampling_variance_samples),
            sample_seed=sampling_variance_seed,
        )
    raise ValueError(f"Unsupported objective_kind={objective_kind}")


def _instantiate_synthetic_policy(
    *,
    actdyn_module: Any,
    env: Any,
    model: Any,
    metric: Any,
    device: str,
    policy_id: str,
    policy_spec: Any,
    schedule_spec: Any,
    seed: int,
    mpc_num_iterations: int = 10,
    mpc_num_samples: int = 40,
    mpc_num_elite: int = 10,
) -> Any:
    policy_type = _resolved_policy_type(policy_id, policy_spec)
    if policy_type == "random":
        return actdyn_module.policy.RandomPolicy(action_space=env.action_space, device=device)
    if policy_type == "baseline-random":
        return actdyn_module.policy.BaselineRandomPolicy(
            action_space=env.action_space,
            device=device,
            seed=int(seed),
        )
    if policy_type == "baseline-prbs":
        return actdyn_module.policy.BaselinePRBSPolicy(
            action_space=env.action_space,
            device=device,
            seed=int(seed),
            hold_steps=max(1, int(getattr(schedule_spec, "planning_chunk", 1))),
            amplitude=1.0,
        )
    if policy_type == "flex":
        return actdyn_module.policy.FLEXPolicy(
            action_space=env.action_space,
            model=model,
            device=device,
        )
    if policy_type == "rhc":
        return actdyn_module.policy.RecedingHorizonCuriosityPolicy(
            action_space=env.action_space,
            device=device,
            horizon=int(schedule_spec.planning_horizon),
            objective="rhc_mvr" if str(policy_id).endswith("_mvr") else "rhc_us",
            num_features=128,
            prior_precision=1.0,
            obs_noise_var=1e-3,
            lengthscale=1.0,
            optimize_hyperparams=True,
            planner_maxiter=500,
            seed=int(seed),
        )
    if policy_type == "off-policy":
        return actdyn_module.policy.OffPolicy(action_space=env.action_space, device=device)
    if policy_type != "mpc-icem":
        raise ValueError(f"Unsupported policy_type={policy_type!r} for synthetic experiments")
    return actdyn_module.policy.mpc.MpcICem(
        metric=metric,
        model=model,
        device=device,
        horizon=int(schedule_spec.planning_horizon),
        num_iterations=int(mpc_num_iterations),
        num_samples=int(mpc_num_samples),
        num_elite=int(mpc_num_elite),
        chunk=int(schedule_spec.planning_chunk),
        verbose=False,
    )


def _run_single_duffing_identification(
    *,
    exp_id: str,
    policy_id: str,
    seed: int,
    total_steps: int,
    run_dir: Path,
    eig_gamma: float,
    q_theta: float,
    q_theta_meas_coeff: float,
    q_theta_max_scale: float,
    traj_eval_interval: int,
    traj_eval_horizon: int,
    traj_eval_samples: int,
    acq_map_interval: int,
    acq_map_grid: int,
    acq_map_lim: float,
    sampling_variance_samples: int,
) -> dict[str, Any]:
    import torch
    import torch.nn as nn

    import actdyn
    import actdyn.core.experiment
    import actdyn.environment
    import actdyn.environment.action
    import actdyn.environment.observation
    import actdyn.metrics
    import actdyn.models
    import actdyn.models.dynamics
    import actdyn.policy
    import actdyn.policy.mpc
    from actdyn.utils.runtime import configure_runtime
    from actdyn.utils.visualize import set_matplotlib_style

    exp_spec = get_experiment_spec(exp_id)
    policy_spec = get_policy_spec(policy_id)
    schedule_spec = get_schedule_spec(policy_spec.schedule_id)
    env_preset = get_environment_preset(exp_spec.env_preset_id)
    system_spec = get_planar_system_spec(env_preset.system_id)
    estimator_system_spec = get_planar_system_spec(_resolved_estimator_system_id(env_preset))

    start_time = _utc_now()
    set_matplotlib_style()
    device = configure_runtime(seed=seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    init_state = sample_initial_state(system_spec.system_id, seed)
    e_true = torch.as_tensor(
        true_embedding(system_spec.system_id), dtype=torch.float32, device=device
    ).unsqueeze(0)
    dz = int(env_preset.latent_dim)
    de = int(env_preset.embedding_dim)
    du = int(env_preset.action_dim)
    dy = int(env_preset.observation_dim)
    dt = float(env_preset.dt)
    alpha = float(env_preset.dynamics_alpha)
    fe_true, fz_true = _build_system_jacobians(
        system_id=estimator_system_spec.system_id,
        dynamics_alpha=alpha,
    )
    noise_scale = max(1e-8, float(env_preset.state_noise))
    action_max = float(max(1e-6, env_preset.action_max))
    mean_firing = float(env_preset.mean_firing_rate_target)
    max_firing_rate = float(env_preset.max_firing_rate_target)

    action_model = actdyn.environment.action.IdentityActionEncoder(
        d_action=du,
        d_latent=dz,
        action_dim=du,
        latent_dim=dz,
        action_bounds=[-action_max, action_max],
        device=device,
    )
    obs_model = actdyn.environment.observation.LogLinearObservation(
        d_obs=dy,
        d_latent=dz,
        obs_dim=dy,
        latent_dim=dz,
        noise_scale=float(env_preset.observation_noise_scale),
        noise_type=str(env_preset.observation_noise_type),
        dt=dt,
        device=device,
    )
    c = obs_model.network[0].weight.detach()
    if env_preset.asymmetric_loading:
        c[:, 0] = torch.abs(c[:, 0])
        c[:, 1] = c[:, 1] * 2
    state_range_for_cap = 5.0
    mean_log_rate = torch.log(torch.full((dy,), mean_firing, device=device))
    max_log_rate = torch.log(torch.full((dy,), max_firing_rate, device=device))
    for _ in range(6):
        c_row_l1 = torch.sum(torch.abs(c), dim=1)
        c_row_l2_sq = torch.sum(c * c, dim=1)
        bias_from_mean = mean_log_rate - 0.5 * c_row_l2_sq
        capped_log_rate = state_range_for_cap * c_row_l1 + bias_from_mean
        if torch.all(capped_log_rate <= max_log_rate):
            break
        safe_den = torch.clamp(state_range_for_cap * c_row_l1, min=1e-8)
        row_scale = torch.clamp((max_log_rate - bias_from_mean) / safe_den, min=0.0, max=1.0)
        c = c * row_scale.unsqueeze(1)
    bias = mean_log_rate - 0.5 * torch.sum(c * c, dim=1)
    obs_model.network[0].bias = nn.Parameter(bias)
    obs_model.network[0].weight = nn.Parameter(c)

    duffing_env = actdyn.VectorFieldEnv(
        system_spec.dynamics_type,
        x_range=5,
        dyn_params=None,
        dt=dt,
        alpha=alpha,
        Q=noise_scale,
        action_bounds=[action_model.action_space.low, action_model.action_space.high],
        state_bounds=[-5.0, 5.0],
        initial_state=init_state.tolist(),
        device=device,
    )
    _set_vectorfield_params(
        duffing_env,
        torch.as_tensor(
            env_params_from_embedding(system_spec.system_id, e_true.reshape(-1)),
            device=device,
        ),
    )
    env = actdyn.environment.EnvWrapper(duffing_env, obs_model, action_model, dt=dt, device=device)

    mapping = actdyn.models.decoder.LogLinearMapping(
        latent_dim=dz, obs_dim=dy, dt=dt, device=device
    )
    noise = actdyn.models.decoder.PoissonNoise(obs_dim=dy, sigma=0.01, device=device)
    decoder = actdyn.models.Decoder(mapping=mapping, noise=noise, device=device)

    sim_vec_env = actdyn.VectorFieldEnv(
        estimator_system_spec.dynamics_type,
        x_range=5,
        dyn_params=None,
        dt=dt,
        alpha=alpha,
        Q=noise_scale,
        device=device,
    )
    _set_vectorfield_params(
        sim_vec_env,
        torch.as_tensor(
            env_params_from_embedding(
                estimator_system_spec.system_id, torch.zeros(2, device=device)
            ),
            device=device,
        ),
    )
    dynamics_fn = _VectorFieldDynamicsAdapter(
        sim_vec_env.dynamics,
        param_formatter=lambda params: env_params_from_embedding(
            estimator_system_spec.system_id, params
        ),
    )
    dynamics = actdyn.models.dynamics.FunctionDynamics(
        state_dim=dz, dt=dt, dynamics_fn=dynamics_fn, device=device
    )
    dynamics.logvar = nn.Parameter(torch.log(torch.ones(1, dz, device=device) * noise_scale))

    sigma_0 = 1e-2
    e_bel = {
        "m": torch.ones(1, de, device=device),
        "P": sigma_0 * torch.eye(de, device=device).unsqueeze(0),
        "L": (1.0 / sigma_0) * torch.eye(de, device=device).unsqueeze(0),
    }
    model_kwargs: dict[str, Any] = {
        "dynamics": dynamics,
        "decoder": decoder,
        "e": e_bel,
        "action_encoder": action_model,
        "Fe": fe_true,
        "Fz": fz_true,
        "device": device,
    }
    fe_init = inspect.signature(actdyn.models.FilteringEmbedding.__init__)
    if "q_theta" in fe_init.parameters:
        model_kwargs["q_theta"] = q_theta
    if "k_theta" in fe_init.parameters:
        model_kwargs["k_theta"] = int(schedule_spec.update_interval)
    if "q_theta_meas_coeff" in fe_init.parameters:
        model_kwargs["q_theta_meas_coeff"] = q_theta_meas_coeff
    if "q_theta_max_scale" in fe_init.parameters:
        model_kwargs["q_theta_max_scale"] = q_theta_max_scale
    if "state_init_uncertainty" in fe_init.parameters:
        model_kwargs["state_init_uncertainty"] = float(env_preset.state_init_uncertainty)
    model = actdyn.models.FilteringEmbedding(**model_kwargs)
    model.set_params(e_bel["m"])

    metric = None
    if not policy_spec.passive:
        base_metric = _build_metric(
            objective_kind=str(policy_spec.objective_kind),
            model=model,
            Fe_net=fe_true,
            Fz_net=fz_true,
            gamma=eig_gamma,
            device=device,
            sampling_variance_samples=int(sampling_variance_samples),
            sampling_variance_seed=int(seed),
        )
        metric = actdyn.metrics.CompositeMetric(
            metrics=[base_metric], compute_type="sum", weights=[1.0], device=device
        )

    policy = _instantiate_synthetic_policy(
        actdyn_module=actdyn,
        env=env,
        model=model,
        metric=metric,
        device=device,
        policy_id=policy_id,
        policy_spec=policy_spec,
        schedule_spec=schedule_spec,
        seed=seed,
        mpc_num_iterations=4,
        mpc_num_samples=24,
        mpc_num_elite=6,
    )

    class _CadencedAgent(actdyn.Agent):
        def __init__(
            self, *, env: Any, model: Any, policy: Any, buffer_length: int, device: str
        ) -> None:
            super().__init__(
                env=env, model=model, policy=policy, buffer_length=buffer_length, device=device
            )
            self.state_update_interval = max(1, int(schedule_spec.update_interval))
            self.predictive_only_window = bool(schedule_spec.predictive_only_window)
            self._window_buffer: list[dict[str, Any]] = []
            self._window_start_snapshot: dict[str, Any] | None = None

        def reset(self, seed: int | None = None):
            obs = super().reset(seed=seed)
            self._window_buffer = []
            self._window_start_snapshot = (
                _clone_filter_belief_state(self.model) if self.predictive_only_window else None
            )
            return obs

        def step(self, action: torch.Tensor | None = None):
            obs, reward, terminated, truncated, env_info = self.env.step(action)
            done = terminated or truncated
            env_transition = {
                "obs": self._observation,
                "next_obs": obs,
                "action": action,
                "env_action": env_info["env_action"],
                "reward": reward,
                "env_state": self._env_state,
                "next_env_state": env_info["latent_state"],
                "model_state": self._model_state,
            }
            self.recent.add(**env_transition)
            state_posterior_updated = False
            parameter_posterior_updated = False
            if self.predictive_only_window and self.state_update_interval > 1:
                model_info = _predictive_only_embedding_step(self.model, action)
                self._window_buffer.append(
                    {
                        "next_obs": _ensure_batch_time_tensor(obs, device=self.device),
                        "action": _ensure_batch_time_tensor(action, device=self.device),
                    }
                )
                if len(self._window_buffer) >= self.state_update_interval:
                    if self._window_start_snapshot is None:
                        self._window_start_snapshot = _clone_filter_belief_state(self.model)
                    _restore_filter_belief_state(self.model, self._window_start_snapshot)
                    for buffered in self._window_buffer:
                        prev_block_steps = int(getattr(self.model, "_theta_block_steps", 0))
                        self.model.update_posterior_embedding(
                            y=buffered["next_obs"], u=buffered["action"]
                        )
                        parameter_posterior_updated = parameter_posterior_updated or (
                            prev_block_steps + 1 >= max(1, int(getattr(self.model, "k_theta", 1)))
                            and int(getattr(self.model, "_theta_block_steps", 0)) == 0
                        )
                    model_info["latent_state"] = self.model.get_state()
                    state_posterior_updated = True
                    self._window_buffer = []
                    self._window_start_snapshot = _clone_filter_belief_state(self.model)
            else:
                prev_block_steps = int(getattr(self.model, "_theta_block_steps", 0))
                model_info = self.model.update(self.recent)
                state_posterior_updated = True
                parameter_posterior_updated = (
                    prev_block_steps + 1 >= max(1, int(getattr(self.model, "k_theta", 1)))
                    and int(getattr(self.model, "_theta_block_steps", 0)) == 0
                )
            model_transition = {
                "model_action": model_info["env_action"],
                "next_model_state": model_info["latent_state"],
            }
            self.recent.add(**model_transition)
            transition = {
                **env_transition,
                **model_transition,
                "policy_action": action,
                "state_posterior_updated": state_posterior_updated,
                "parameter_posterior_updated": parameter_posterior_updated,
                "window_buffer_length": len(self._window_buffer),
                "state_update_interval": self.state_update_interval,
            }
            self.update_policy(self.recent)
            self._observation = obs
            self._env_state = env_info["latent_state"]
            self._model_state = self.model.get_state()
            return transition, done

    exp_config = _build_runtime_experiment_config(
        run_dir=run_dir,
        seed=seed,
        total_steps=total_steps,
        experiment_kind=exp_spec.experiment_kind,
        policy_id=policy_id,
        env_preset=env_preset,
        schedule_spec=schedule_spec,
        policy_spec=policy_spec,
    )
    exp_config.device = str(device)
    agent = _CadencedAgent(env=env, model=model, buffer_length=10, policy=policy, device=device)
    experiment = actdyn.core.experiment.Experiment(agent=agent, config=exp_config)
    decoder.set_params(obs_model)

    param_rows: list[dict[str, Any]] = []
    emb_rows: list[dict[str, Any]] = []
    info_rows: list[dict[str, Any]] = []
    traj_rows: list[dict[str, Any]] = []
    state_action_rows: list[dict[str, Any]] = []
    acq_map_steps: list[int] = []
    acq_map_frames: list[np.ndarray] = []
    planned_traj_steps: list[int] = []
    planned_traj_frames: list[np.ndarray] = []
    perf_start = time.perf_counter()
    trace_rng = np.random.default_rng(seed + 137)
    e_true_flat = e_true.detach().reshape(-1)
    acq_axis = np.linspace(
        -float(acq_map_lim), float(acq_map_lim), max(25, int(acq_map_grid)), dtype=np.float32
    )
    acq_x, acq_v = np.meshgrid(acq_axis, acq_axis, indexing="xy")
    acq_points = torch.as_tensor(
        np.stack([acq_x.reshape(-1), acq_v.reshape(-1)], axis=1), dtype=torch.float32, device=device
    ).unsqueeze(1)

    def _on_step_end(transition: dict[str, Any]) -> None:
        step = int(experiment.env_step)
        cpu_time_sec = float(time.perf_counter() - perf_start)
        e_est = model.e["m"].detach().reshape(-1)
        param_err = float(torch.linalg.norm(e_est - e_true_flat).item())
        e_cov = model.e.get("P")
        cov_diag0 = cov_diag1 = cov_diag_mean = None
        if e_cov is not None:
            diag = torch.diagonal(e_cov.detach(), dim1=-2, dim2=-1).reshape(-1)
            if diag.numel() > 0:
                cov_diag0 = float(diag[0].item())
                cov_diag1 = float(diag[1].item()) if diag.numel() > 1 else None
                cov_diag_mean = float(diag.mean().item())
        param_rows.append(
            {"step": step, "cpu_time_sec": cpu_time_sec, "parameter_error": param_err}
        )
        emb_rows.append(
            {
                "step": step,
                "cpu_time_sec": cpu_time_sec,
                "e0": float(e_est[0].item()),
                "e1": float(e_est[1].item()),
                "cov_diag0": cov_diag0,
                "cov_diag1": cov_diag1,
                "cov_diag_mean": cov_diag_mean,
            }
        )
        info_diag = getattr(model, "last_information", {}) or {}
        info_rows.append(
            {
                "step": step,
                "cpu_time_sec": cpu_time_sec,
                "I_z_t": float(info_diag.get("I_z_t", 0.0)),
                "I_theta_t": float(info_diag.get("I_theta_t", 0.0)),
                "Pz00": float(info_diag.get("Pz00", 0.0)),
                "Pz01": float(info_diag.get("Pz01", 0.0)),
                "Pz11": float(info_diag.get("Pz11", 0.0)),
                "state_posterior_updated": _as_bool(
                    transition.get("state_posterior_updated", True)
                ),
                "parameter_posterior_updated": _as_bool(
                    transition.get("parameter_posterior_updated", True)
                ),
                "window_buffer_length": int(transition.get("window_buffer_length", 0)),
            }
        )
        env_x, env_v = _to_xy_pair(transition.get("env_state", torch.zeros(2, device=device)))
        model_x, model_v = _to_xy_pair(transition.get("model_state", torch.zeros(2, device=device)))
        next_model_x, next_model_v = _to_xy_pair(
            transition.get("next_model_state", torch.zeros(2, device=device))
        )
        action_x, action_v = _to_xy_pair(transition.get("action", torch.zeros(2, device=device)))
        policy_x, policy_v = _to_xy_pair(
            transition.get("policy_action", transition.get("action", torch.zeros(2, device=device)))
        )
        env_action_x, env_action_v = _to_xy_pair(
            transition.get(
                "env_action", transition.get("policy_action", torch.zeros(2, device=device))
            )
        )
        state_action_rows.append(
            {
                "step": step,
                "cpu_time_sec": cpu_time_sec,
                "true_x": env_x,
                "true_v": env_v,
                "model_x": model_x,
                "model_v": model_v,
                "next_model_x": next_model_x,
                "next_model_v": next_model_v,
                "action_x": action_x,
                "action_v": action_v,
                "action_norm": float(np.hypot(action_x, action_v)),
                "policy_action_x": policy_x,
                "policy_action_v": policy_v,
                "policy_action_norm": float(np.hypot(policy_x, policy_v)),
                "env_action_x": env_action_x,
                "env_action_v": env_action_v,
                "env_action_norm": float(np.hypot(env_action_x, env_action_v)),
                "policy_action_delta_norm": float(
                    np.hypot(policy_x - action_x, policy_v - action_v)
                ),
                "execution_delta_norm": float(
                    np.hypot(env_action_x - policy_x, env_action_v - policy_v)
                ),
                "action_total_delta_norm": float(
                    np.hypot(env_action_x - action_x, env_action_v - action_v)
                ),
                "action_clipped": _as_bool(transition.get("action_clipped", False)),
                "env_action_clipped": _as_bool(transition.get("env_action_clipped", False)),
                "planned_at_bound": bool(max(abs(action_x), abs(action_v)) >= action_max - 1e-6),
                "policy_at_bound": bool(max(abs(policy_x), abs(policy_v)) >= action_max - 1e-6),
                "env_action_at_bound": bool(
                    max(abs(env_action_x), abs(env_action_v)) >= action_max - 1e-6
                ),
                "policy_cost": (
                    float(getattr(policy, "cost", np.nan))
                    if getattr(policy, "cost", None) is not None
                    else None
                ),
                "state_posterior_updated": _as_bool(
                    transition.get("state_posterior_updated", True)
                ),
                "parameter_posterior_updated": _as_bool(
                    transition.get("parameter_posterior_updated", True)
                ),
                "window_buffer_length": int(transition.get("window_buffer_length", 0)),
            }
        )
        planned = _predict_planned_xy_trajectory(model=model, policy=policy, transition=transition)
        if planned is not None and planned.shape[0] >= 2:
            planned_traj_steps.append(step)
            planned_traj_frames.append(planned)
        if (
            policy_spec.save_acq_map
            and metric is not None
            and step % max(1, int(acq_map_interval)) == 0
        ):
            rollout = {"model_state": acq_points, "next_model_state": acq_points}
            acq_cost = metric(rollout).detach().reshape(-1)
            acq_map = (-acq_cost).cpu().numpy().reshape(acq_axis.shape[0], acq_axis.shape[0])
            acq_map_frames.append(
                np.nan_to_num(acq_map, nan=0.0, posinf=1e6, neginf=0.0).astype(np.float32)
            )
            acq_map_steps.append(step)
        if traj_eval_interval > 0 and step % traj_eval_interval == 0:
            traj_rows.append(
                {
                    "step": step,
                    "cpu_time_sec": cpu_time_sec,
                    "trajectory_r2": _trajectory_r2(
                        e_est=e_est,
                        e_true=e_true_flat,
                        system_id=system_spec.system_id,
                        dt=dt,
                        dynamics_alpha=alpha,
                        horizon=traj_eval_horizon,
                        n_starts=traj_eval_samples,
                        rng=trace_rng,
                        device=device,
                    ),
                    "traj_eval_horizon": int(traj_eval_horizon),
                    "traj_eval_samples": int(traj_eval_samples),
                }
            )

    experiment._run_online_loop(
        train_cfg=exp_config.training,
        pbar_desc="COSYNE",
        plot_fcn=None,
        reset=True,
        on_step_end=_on_step_end,
    )
    ended = datetime.now(timezone.utc)
    result_dir = Path(experiment.results_path)
    param_trace_path = run_dir / "parameter_error_trace.csv"
    traj_trace_path = run_dir / "trajectory_r2_trace.csv"
    emb_trace_path = run_dir / "embedding_estimate_trace.csv"
    info_trace_path = run_dir / "information_trace.csv"
    state_action_trace_path = run_dir / "state_action_trace.csv"
    planned_trace_path = run_dir / "planned_trajectory_trace.npz"
    acq_trace_path = run_dir / "acquisition_map_trace.npz"
    _write_trace_csv(param_trace_path, param_rows, ["step", "cpu_time_sec", "parameter_error"])
    _write_trace_csv(
        emb_trace_path,
        emb_rows,
        ["step", "cpu_time_sec", "e0", "e1", "cov_diag0", "cov_diag1", "cov_diag_mean"],
    )
    _write_trace_csv(
        traj_trace_path,
        traj_rows,
        ["step", "cpu_time_sec", "trajectory_r2", "traj_eval_horizon", "traj_eval_samples"],
    )
    _write_trace_csv(
        info_trace_path,
        info_rows,
        [
            "step",
            "cpu_time_sec",
            "I_z_t",
            "I_theta_t",
            "Pz00",
            "Pz01",
            "Pz11",
            "state_posterior_updated",
            "parameter_posterior_updated",
            "window_buffer_length",
        ],
    )
    _write_trace_csv(
        state_action_trace_path,
        state_action_rows,
        [
            "step",
            "cpu_time_sec",
            "true_x",
            "true_v",
            "model_x",
            "model_v",
            "next_model_x",
            "next_model_v",
            "action_x",
            "action_v",
            "action_norm",
            "policy_action_x",
            "policy_action_v",
            "policy_action_norm",
            "env_action_x",
            "env_action_v",
            "env_action_norm",
            "policy_action_delta_norm",
            "execution_delta_norm",
            "action_total_delta_norm",
            "action_clipped",
            "env_action_clipped",
            "planned_at_bound",
            "policy_at_bound",
            "env_action_at_bound",
            "policy_cost",
            "state_posterior_updated",
            "parameter_posterior_updated",
            "window_buffer_length",
        ],
    )
    if policy_spec.save_acq_map and acq_map_frames:
        np.savez_compressed(
            acq_trace_path,
            steps=np.asarray(acq_map_steps, dtype=np.int64),
            axis=acq_axis.astype(np.float32),
            maps=np.asarray(acq_map_frames, dtype=np.float32),
            exp_id=np.asarray([exp_id], dtype=object),
            policy_id=np.asarray([policy_id], dtype=object),
        )
    if planned_traj_frames:
        max_points = max(frame.shape[0] for frame in planned_traj_frames)
        paths = np.full((len(planned_traj_frames), max_points, 2), np.nan, dtype=np.float32)
        lengths = np.zeros((len(planned_traj_frames),), dtype=np.int64)
        for idx, frame in enumerate(planned_traj_frames):
            n_points = int(frame.shape[0])
            paths[idx, :n_points, :] = frame[:, :2]
            lengths[idx] = n_points
        np.savez_compressed(
            planned_trace_path,
            steps=np.asarray(planned_traj_steps, dtype=np.int64),
            paths=paths,
            lengths=lengths,
            exp_id=np.asarray([exp_id], dtype=object),
            policy_id=np.asarray([policy_id], dtype=object),
        )
    rollout_metrics = _extract_rollout_metrics(result_dir)
    payload = _build_metadata(
        exp_id=exp_id,
        policy_id=policy_id,
        seed=seed,
        total_steps=total_steps,
        run_dir=run_dir,
        status="completed",
        start_time=start_time,
        end_time=ended.isoformat().replace("+00:00", "Z"),
        runtime_sec=(
            ended - datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        ).total_seconds(),
        results_path=result_dir,
        extra={
            **rollout_metrics,
            "system_id": str(system_spec.system_id),
            "system_label": str(system_spec.label),
            "dynamics_type": str(system_spec.dynamics_type),
            "estimator_system_id": str(estimator_system_spec.system_id),
            "estimator_system_label": str(estimator_system_spec.label),
            "estimator_dynamics_type": str(estimator_system_spec.dynamics_type),
            "initial_state_true": [float(x) for x in init_state.tolist()],
            "embedding_true": [float(x) for x in e_true_flat.tolist()],
            "embedding_estimate": [float(x) for x in model.e["m"].detach().reshape(-1).tolist()],
            "embedding_error_final": (
                float(param_rows[-1]["parameter_error"]) if param_rows else None
            ),
            "embedding_error_mean": (
                float(np.mean([row["parameter_error"] for row in param_rows]))
                if param_rows
                else None
            ),
            "objective_variant": str(policy_spec.objective_kind),
            "schedule_id": str(schedule_spec.schedule_id),
            "update_interval": int(schedule_spec.update_interval),
            "replan_interval": int(schedule_spec.replan_interval),
            "planning_horizon": int(schedule_spec.planning_horizon),
            "planning_chunk": int(schedule_spec.planning_chunk),
            "predictive_only_window": bool(schedule_spec.predictive_only_window),
            "state_update_interval": int(schedule_spec.update_interval),
            "parameter_update_interval": int(schedule_spec.update_interval),
            "q_theta": float(q_theta),
            "q_theta_meas_coeff": float(q_theta_meas_coeff),
            "q_theta_max_scale": float(q_theta_max_scale),
            "eig_gamma": float(eig_gamma),
            "sampling_variance_samples": int(sampling_variance_samples),
            "state_noise": float(noise_scale),
            "action_max": float(action_max),
            "dynamics_alpha": float(alpha),
            "state_init_uncertainty": float(env_preset.state_init_uncertainty),
            "firing_rate_scale": float(env_preset.firing_rate_scale),
            "mean_firing_rate_target": float(mean_firing),
            "max_firing_rate_target": float(max_firing_rate),
            "hard_setup": env_preset.preset_id == "hard_duffing",
            "traj_eval_interval": int(traj_eval_interval),
            "traj_eval_horizon": int(traj_eval_horizon),
            "traj_eval_samples": int(traj_eval_samples),
            "parameter_error_trace_path": str(param_trace_path),
            "trajectory_r2_trace_path": str(traj_trace_path),
            "embedding_estimate_trace_path": str(emb_trace_path),
            "information_trace_path": str(info_trace_path),
            "state_action_trace_path": str(state_action_trace_path),
            "planned_trajectory_trace_path": (
                str(planned_trace_path) if planned_traj_frames else None
            ),
            "acquisition_map_trace_path": (
                str(acq_trace_path) if policy_spec.save_acq_map and acq_map_frames else None
            ),
            "writing_ref": WRITING_REFERENCE,
        },
    )
    return payload


def _run_single_rbf_identification(
    *,
    exp_id: str,
    policy_id: str,
    seed: int,
    total_steps: int,
    run_dir: Path,
    eig_gamma: float,
    q_theta: float,
    traj_eval_interval: int,
    traj_eval_horizon: int,
    traj_eval_samples: int,
    acq_map_interval: int,
    acq_map_grid: int,
    acq_map_lim: float,
) -> dict[str, Any]:
    import torch
    import torch.nn as nn

    import actdyn
    import actdyn.core.experiment
    import actdyn.environment
    import actdyn.environment.action
    import actdyn.environment.observation
    import actdyn.policy
    import actdyn.policy.mpc
    from actdyn.utils.runtime import configure_runtime
    from actdyn.utils.visualize import set_matplotlib_style

    exp_spec = get_experiment_spec(exp_id)
    policy_spec = get_policy_spec(policy_id)
    schedule_spec = get_schedule_spec(policy_spec.schedule_id)
    env_preset = get_environment_preset(exp_spec.env_preset_id)
    system_spec = get_planar_system_spec(env_preset.system_id)

    start_time = _utc_now()
    set_matplotlib_style()
    device = configure_runtime(seed=seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    init_state = sample_initial_state(system_spec.system_id, seed)
    e_true = torch.as_tensor(
        true_embedding(system_spec.system_id), dtype=torch.float32, device=device
    ).unsqueeze(0)
    dz = int(env_preset.latent_dim)
    du = int(env_preset.action_dim)
    dy = int(env_preset.observation_dim)
    dt = float(env_preset.dt)
    alpha = float(env_preset.dynamics_alpha)
    noise_scale = max(1e-8, float(env_preset.state_noise))
    action_max = float(max(1e-6, env_preset.action_max))
    mean_firing = float(env_preset.mean_firing_rate_target)
    max_firing_rate = float(env_preset.max_firing_rate_target)
    support_radius = 2
    grid_shape = (30, 30)
    grid_axis, centers = _rbf_grid(env_preset.x_range, grid_shape[0])
    width = float(grid_axis[1] - grid_axis[0]) if grid_axis.shape[0] > 1 else 1.0

    action_model = actdyn.environment.action.IdentityActionEncoder(
        d_action=du,
        d_latent=dz,
        action_dim=du,
        latent_dim=dz,
        action_bounds=[-action_max, action_max],
        device=device,
    )
    obs_model = actdyn.environment.observation.LogLinearObservation(
        d_obs=dy,
        d_latent=dz,
        obs_dim=dy,
        latent_dim=dz,
        noise_scale=float(env_preset.observation_noise_scale),
        noise_type=str(env_preset.observation_noise_type),
        dt=dt,
        device=device,
    )
    c = obs_model.network[0].weight.detach()
    if env_preset.asymmetric_loading:
        c[:, 0] = torch.abs(c[:, 0])
        c[:, 1] = c[:, 1] * 2
    state_range_for_cap = 5.0
    mean_log_rate = torch.log(torch.full((dy,), mean_firing, device=device))
    max_log_rate = torch.log(torch.full((dy,), max_firing_rate, device=device))
    for _ in range(6):
        c_row_l1 = torch.sum(torch.abs(c), dim=1)
        c_row_l2_sq = torch.sum(c * c, dim=1)
        bias_from_mean = mean_log_rate - 0.5 * c_row_l2_sq
        capped_log_rate = state_range_for_cap * c_row_l1 + bias_from_mean
        if torch.all(capped_log_rate <= max_log_rate):
            break
        safe_den = torch.clamp(state_range_for_cap * c_row_l1, min=1e-8)
        row_scale = torch.clamp((max_log_rate - bias_from_mean) / safe_den, min=0.0, max=1.0)
        c = c * row_scale.unsqueeze(1)
    bias = mean_log_rate - 0.5 * torch.sum(c * c, dim=1)
    obs_model.network[0].bias = nn.Parameter(bias)
    obs_model.network[0].weight = nn.Parameter(c)

    duffing_env = actdyn.VectorFieldEnv(
        system_spec.dynamics_type,
        x_range=5,
        dyn_params=None,
        dt=dt,
        alpha=alpha,
        Q=noise_scale,
        action_bounds=[action_model.action_space.low, action_model.action_space.high],
        state_bounds=[-5.0, 5.0],
        initial_state=init_state.tolist(),
        device=device,
    )
    _set_vectorfield_params(
        duffing_env,
        torch.as_tensor(
            env_params_from_embedding(system_spec.system_id, e_true.reshape(-1)),
            device=device,
        ),
    )
    env = actdyn.environment.EnvWrapper(duffing_env, obs_model, action_model, dt=dt, device=device)

    mapping = actdyn.models.decoder.LogLinearMapping(
        latent_dim=dz, obs_dim=dy, dt=dt, device=device
    )
    noise = actdyn.models.decoder.PoissonNoise(obs_dim=dy, sigma=0.01, device=device)
    decoder = actdyn.models.Decoder(mapping=mapping, noise=noise, device=device)

    centers_t = torch.as_tensor(centers, dtype=torch.float32, device=device)
    axis_t = torch.as_tensor(grid_axis, dtype=torch.float32, device=device)
    dynamics = SparseRbfDynamics(
        state_dim=dz,
        centers=centers_t,
        axis=axis_t,
        width=width,
        support_radius=support_radius,
        dt=dt,
        is_residual=True,
        device=device,
    )
    dynamics.logvar = nn.Parameter(torch.log(torch.ones(1, dz, device=device) * noise_scale))

    d_embed = int(centers.shape[0] * dz)
    init_param_var = torch.ones(1, d_embed, dtype=torch.float32, device=device)
    e_bel = {
        "m": torch.zeros(1, d_embed, dtype=torch.float32, device=device),
        "P": torch.diag_embed(init_param_var.clone()),
    }
    model = SparseRbfFilteringModel(
        dynamics=dynamics,
        decoder=decoder,
        action_encoder=action_model,
        e=e_bel,
        q_theta=q_theta,
        k_theta=int(schedule_spec.update_interval),
        state_init_uncertainty=float(env_preset.state_init_uncertainty),
        device=device,
    )

    metric = None
    if not policy_spec.passive:
        metric = StructuredLocalRbfParameterMetric(model=model, gamma=eig_gamma, device=device)

    policy = _instantiate_synthetic_policy(
        actdyn_module=actdyn,
        env=env,
        model=model,
        metric=metric,
        device=device,
        policy_id=policy_id,
        policy_spec=policy_spec,
        schedule_spec=schedule_spec,
        seed=seed,
    )

    class _CadencedAgent(actdyn.Agent):
        def __init__(
            self, *, env: Any, model: Any, policy: Any, buffer_length: int, device: str
        ) -> None:
            super().__init__(
                env=env, model=model, policy=policy, buffer_length=buffer_length, device=device
            )
            self.state_update_interval = max(1, int(schedule_spec.update_interval))
            self.predictive_only_window = bool(schedule_spec.predictive_only_window)
            self._window_buffer: list[dict[str, Any]] = []
            self._window_start_snapshot: dict[str, Any] | None = None

        def reset(self, seed: int | None = None):
            obs = super().reset(seed=seed)
            self._window_buffer = []
            self._window_start_snapshot = (
                _clone_filter_belief_state(self.model) if self.predictive_only_window else None
            )
            return obs

        def step(self, action: torch.Tensor | None = None):
            obs, reward, terminated, truncated, env_info = self.env.step(action)
            done = terminated or truncated
            env_transition = {
                "obs": self._observation,
                "next_obs": obs,
                "action": action,
                "env_action": env_info["env_action"],
                "reward": reward,
                "env_state": self._env_state,
                "next_env_state": env_info["latent_state"],
                "model_state": self._model_state,
            }
            self.recent.add(**env_transition)
            state_posterior_updated = False
            parameter_posterior_updated = False
            if self.predictive_only_window and self.state_update_interval > 1:
                model_info = _predictive_only_embedding_step(self.model, action)
                self._window_buffer.append(
                    {
                        "next_obs": _ensure_batch_time_tensor(obs, device=self.device),
                        "action": _ensure_batch_time_tensor(action, device=self.device),
                    }
                )
                if len(self._window_buffer) >= self.state_update_interval:
                    if self._window_start_snapshot is None:
                        self._window_start_snapshot = _clone_filter_belief_state(self.model)
                    _restore_filter_belief_state(self.model, self._window_start_snapshot)
                    for buffered in self._window_buffer:
                        prev_block_steps = int(getattr(self.model, "_theta_block_steps", 0))
                        self.model.update_posterior_embedding(
                            y=buffered["next_obs"], u=buffered["action"]
                        )
                        parameter_posterior_updated = parameter_posterior_updated or (
                            prev_block_steps + 1 >= max(1, int(getattr(self.model, "k_theta", 1)))
                            and int(getattr(self.model, "_theta_block_steps", 0)) == 0
                        )
                    model_info["latent_state"] = self.model.get_state()
                    state_posterior_updated = True
                    self._window_buffer = []
                    self._window_start_snapshot = _clone_filter_belief_state(self.model)
            else:
                prev_block_steps = int(getattr(self.model, "_theta_block_steps", 0))
                model_info = self.model.update(self.recent)
                state_posterior_updated = True
                parameter_posterior_updated = (
                    prev_block_steps + 1 >= max(1, int(getattr(self.model, "k_theta", 1)))
                    and int(getattr(self.model, "_theta_block_steps", 0)) == 0
                )
            model_transition = {
                "model_action": model_info["env_action"],
                "next_model_state": model_info["latent_state"],
            }
            self.recent.add(**model_transition)
            transition = {
                **env_transition,
                **model_transition,
                "policy_action": action,
                "state_posterior_updated": state_posterior_updated,
                "parameter_posterior_updated": parameter_posterior_updated,
                "window_buffer_length": len(self._window_buffer),
                "state_update_interval": self.state_update_interval,
            }
            self.update_policy(self.recent)
            self._observation = obs
            self._env_state = env_info["latent_state"]
            self._model_state = self.model.get_state()
            return transition, done

    exp_config = _build_runtime_experiment_config(
        run_dir=run_dir,
        seed=seed,
        total_steps=total_steps,
        experiment_kind=exp_spec.experiment_kind,
        policy_id=policy_id,
        env_preset=env_preset,
        schedule_spec=schedule_spec,
        policy_spec=policy_spec,
    )
    exp_config.device = str(device)
    agent = _CadencedAgent(env=env, model=model, buffer_length=10, policy=policy, device=device)
    experiment = actdyn.core.experiment.Experiment(agent=agent, config=exp_config)
    decoder.set_params(obs_model)

    info_rows: list[dict[str, Any]] = []
    traj_rows: list[dict[str, Any]] = []
    state_action_rows: list[dict[str, Any]] = []
    dynamics_rows: list[dict[str, Any]] = []
    acq_map_steps: list[int] = []
    acq_map_frames: list[np.ndarray] = []
    planned_traj_steps: list[int] = []
    planned_traj_frames: list[np.ndarray] = []
    model_trace_steps: list[int] = []
    model_weight_frames: list[np.ndarray] = []
    model_precision_frames: list[np.ndarray] = []
    model_covariance_diag_frames: list[np.ndarray] = []
    perf_start = time.perf_counter()
    trace_rng = np.random.default_rng(seed + 911)
    acq_axis = np.linspace(
        -float(acq_map_lim), float(acq_map_lim), max(25, int(acq_map_grid)), dtype=np.float32
    )
    acq_x, acq_v = np.meshgrid(acq_axis, acq_axis, indexing="xy")
    acq_points = torch.as_tensor(
        np.stack([acq_x.reshape(-1), acq_v.reshape(-1)], axis=1), dtype=torch.float32, device=device
    ).unsqueeze(1)
    eval_axis = np.linspace(
        -float(env_preset.x_range), float(env_preset.x_range), 25, dtype=np.float64
    )
    eval_x, eval_y = np.meshgrid(eval_axis, eval_axis, indexing="xy")
    eval_states = np.stack([eval_x.reshape(-1), eval_y.reshape(-1)], axis=1)
    last_weight_mean = [model.e["m"].detach().clone()]

    def _on_step_end(transition: dict[str, Any]) -> None:
        step = int(experiment.env_step)
        cpu_time_sec = float(time.perf_counter() - perf_start)
        weight_vec = model.e["m"].detach()
        weight_mat = (
            weight_vec.reshape(-1, dz).detach().cpu().numpy().astype(np.float64, copy=False)
        )
        precision_mat = (
            model.weight_precision().detach().cpu().numpy().astype(np.float64, copy=False)
        )
        covariance_diag = (
            torch.diagonal(model.e["P"], dim1=-2, dim2=-1)
            .detach()
            .reshape(-1, dz)
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )
        dynamics_mse = _rbf_dynamics_mse(
            system_id=system_spec.system_id,
            weights=weight_mat,
            centers=centers,
            axis=grid_axis,
            support_radius=support_radius,
            width=width,
            true_embedding_vec=e_true.reshape(-1).detach().cpu().numpy(),
            dynamics_alpha=alpha,
            eval_states=eval_states,
        )
        info_diag = getattr(model, "last_information", {}) or {}
        model_x, model_v = _to_xy_pair(transition.get("model_state", torch.zeros(2, device=device)))
        active_kernel_count = int(
            _rbf_active_indices(
                np.asarray([model_x, model_v], dtype=np.float64), grid_axis, support_radius
            ).size
        )
        param_update_norm = float(torch.linalg.norm(weight_vec - last_weight_mean[0]).item())
        last_weight_mean[0] = weight_vec.detach().clone()
        info_rows.append(
            {
                "step": step,
                "cpu_time_sec": cpu_time_sec,
                "I_z_t": float(info_diag.get("I_z_t", 0.0)),
                "I_theta_t": float(info_diag.get("I_theta_t", 0.0)),
                "Pz00": float(info_diag.get("Pz00", 0.0)),
                "Pz01": float(info_diag.get("Pz01", 0.0)),
                "Pz11": float(info_diag.get("Pz11", 0.0)),
                "active_kernel_count": active_kernel_count,
                "local_weight_residual_norm": param_update_norm,
                "state_posterior_updated": _as_bool(
                    transition.get("state_posterior_updated", True)
                ),
                "parameter_posterior_updated": _as_bool(
                    transition.get("parameter_posterior_updated", True)
                ),
                "window_buffer_length": int(transition.get("window_buffer_length", 0)),
            }
        )
        dynamics_rows.append(
            {"step": step, "cpu_time_sec": cpu_time_sec, "dynamics_mse": float(dynamics_mse)}
        )
        env_x, env_v = _to_xy_pair(transition.get("env_state", torch.zeros(2, device=device)))
        next_model_x, next_model_v = _to_xy_pair(
            transition.get("next_model_state", torch.zeros(2, device=device))
        )
        action_x, action_v = _to_xy_pair(transition.get("action", torch.zeros(2, device=device)))
        policy_x, policy_v = _to_xy_pair(
            transition.get("policy_action", transition.get("action", torch.zeros(2, device=device)))
        )
        env_action_x, env_action_v = _to_xy_pair(
            transition.get(
                "env_action", transition.get("policy_action", torch.zeros(2, device=device))
            )
        )
        state_action_rows.append(
            {
                "step": step,
                "cpu_time_sec": cpu_time_sec,
                "true_x": env_x,
                "true_v": env_v,
                "model_x": model_x,
                "model_v": model_v,
                "next_model_x": next_model_x,
                "next_model_v": next_model_v,
                "action_x": action_x,
                "action_v": action_v,
                "action_norm": float(np.hypot(action_x, action_v)),
                "policy_action_x": policy_x,
                "policy_action_v": policy_v,
                "policy_action_norm": float(np.hypot(policy_x, policy_v)),
                "env_action_x": env_action_x,
                "env_action_v": env_action_v,
                "env_action_norm": float(np.hypot(env_action_x, env_action_v)),
                "policy_action_delta_norm": float(
                    np.hypot(policy_x - action_x, policy_v - action_v)
                ),
                "execution_delta_norm": float(
                    np.hypot(env_action_x - policy_x, env_action_v - policy_v)
                ),
                "action_total_delta_norm": float(
                    np.hypot(env_action_x - action_x, env_action_v - action_v)
                ),
                "action_clipped": _as_bool(transition.get("action_clipped", False)),
                "env_action_clipped": _as_bool(transition.get("env_action_clipped", False)),
                "planned_at_bound": bool(max(abs(action_x), abs(action_v)) >= action_max - 1e-6),
                "policy_at_bound": bool(max(abs(policy_x), abs(policy_v)) >= action_max - 1e-6),
                "env_action_at_bound": bool(
                    max(abs(env_action_x), abs(env_action_v)) >= action_max - 1e-6
                ),
                "policy_cost": (
                    float(getattr(policy, "cost", np.nan))
                    if getattr(policy, "cost", None) is not None
                    else None
                ),
                "state_posterior_updated": _as_bool(
                    transition.get("state_posterior_updated", True)
                ),
                "parameter_posterior_updated": _as_bool(
                    transition.get("parameter_posterior_updated", True)
                ),
                "window_buffer_length": int(transition.get("window_buffer_length", 0)),
            }
        )
        planned = _predict_planned_xy_trajectory(model=model, policy=policy, transition=transition)
        if planned is not None and planned.shape[0] >= 2:
            planned_traj_steps.append(step)
            planned_traj_frames.append(planned)
        if (
            policy_spec.save_acq_map
            and metric is not None
            and step % max(1, int(acq_map_interval)) == 0
        ):
            rollout = {"model_state": acq_points, "next_model_state": acq_points}
            acq_cost = metric(rollout).detach().reshape(-1)
            acq_map = (-acq_cost).cpu().numpy().reshape(acq_axis.shape[0], acq_axis.shape[0])
            acq_map_frames.append(
                np.nan_to_num(acq_map, nan=0.0, posinf=1e6, neginf=0.0).astype(np.float32)
            )
            acq_map_steps.append(step)
        if traj_eval_interval > 0 and step % traj_eval_interval == 0:
            traj_rows.append(
                {
                    "step": step,
                    "cpu_time_sec": cpu_time_sec,
                    "trajectory_r2": _trajectory_r2_rbf(
                        system_id=system_spec.system_id,
                        weights=weight_mat,
                        centers=centers,
                        axis=grid_axis,
                        support_radius=support_radius,
                        width=width,
                        true_embedding_vec=e_true.reshape(-1).detach().cpu().numpy(),
                        dt=dt,
                        dynamics_alpha=alpha,
                        horizon=traj_eval_horizon,
                        n_starts=traj_eval_samples,
                        rng=trace_rng,
                        clip_limit=max(6.0, env_preset.x_range * 1.6),
                    ),
                    "traj_eval_horizon": int(traj_eval_horizon),
                    "traj_eval_samples": int(traj_eval_samples),
                }
            )
        model_trace_steps.append(step)
        model_weight_frames.append(weight_mat.astype(np.float32, copy=True))
        model_precision_frames.append(precision_mat.astype(np.float32, copy=True))
        model_covariance_diag_frames.append(covariance_diag.astype(np.float32, copy=True))

    experiment._run_online_loop(
        train_cfg=exp_config.training,
        pbar_desc="COSYNE",
        plot_fcn=None,
        reset=True,
        on_step_end=_on_step_end,
    )
    ended = datetime.now(timezone.utc)
    result_dir = Path(experiment.results_path)
    dynamics_trace_path = run_dir / "dynamics_mse_trace.csv"
    traj_trace_path = run_dir / "trajectory_r2_trace.csv"
    info_trace_path = run_dir / "information_trace.csv"
    state_action_trace_path = run_dir / "state_action_trace.csv"
    planned_trace_path = run_dir / "planned_trajectory_trace.npz"
    acq_trace_path = run_dir / "acquisition_map_trace.npz"
    rbf_trace_path = run_dir / "rbf_model_trace.npz"

    _write_trace_csv(dynamics_trace_path, dynamics_rows, ["step", "cpu_time_sec", "dynamics_mse"])
    _write_trace_csv(
        traj_trace_path,
        traj_rows,
        ["step", "cpu_time_sec", "trajectory_r2", "traj_eval_horizon", "traj_eval_samples"],
    )
    _write_trace_csv(
        info_trace_path,
        info_rows,
        [
            "step",
            "cpu_time_sec",
            "I_z_t",
            "I_theta_t",
            "Pz00",
            "Pz01",
            "Pz11",
            "active_kernel_count",
            "local_weight_residual_norm",
            "state_posterior_updated",
            "parameter_posterior_updated",
            "window_buffer_length",
        ],
    )
    _write_trace_csv(
        state_action_trace_path,
        state_action_rows,
        [
            "step",
            "cpu_time_sec",
            "true_x",
            "true_v",
            "model_x",
            "model_v",
            "next_model_x",
            "next_model_v",
            "action_x",
            "action_v",
            "action_norm",
            "policy_action_x",
            "policy_action_v",
            "policy_action_norm",
            "env_action_x",
            "env_action_v",
            "env_action_norm",
            "policy_action_delta_norm",
            "execution_delta_norm",
            "action_total_delta_norm",
            "action_clipped",
            "env_action_clipped",
            "planned_at_bound",
            "policy_at_bound",
            "env_action_at_bound",
            "policy_cost",
            "state_posterior_updated",
            "parameter_posterior_updated",
            "window_buffer_length",
        ],
    )
    if planned_traj_frames:
        max_points = max(frame.shape[0] for frame in planned_traj_frames)
        paths = np.full((len(planned_traj_frames), max_points, 2), np.nan, dtype=np.float32)
        lengths = np.zeros((len(planned_traj_frames),), dtype=np.int64)
        for idx, frame in enumerate(planned_traj_frames):
            n_points = int(frame.shape[0])
            paths[idx, :n_points, :] = frame[:, :2]
            lengths[idx] = n_points
        np.savez_compressed(
            planned_trace_path,
            steps=np.asarray(planned_traj_steps, dtype=np.int64),
            paths=paths,
            lengths=lengths,
            exp_id=np.asarray([exp_id], dtype=object),
            policy_id=np.asarray([policy_id], dtype=object),
        )
    if policy_spec.save_acq_map and acq_map_frames:
        np.savez_compressed(
            acq_trace_path,
            steps=np.asarray(acq_map_steps, dtype=np.int64),
            axis=acq_axis.astype(np.float32),
            maps=np.asarray(acq_map_frames, dtype=np.float32),
            exp_id=np.asarray([exp_id], dtype=object),
            policy_id=np.asarray([policy_id], dtype=object),
        )
    np.savez_compressed(
        rbf_trace_path,
        steps=np.asarray(model_trace_steps, dtype=np.int64),
        weights=np.asarray(model_weight_frames, dtype=np.float32),
        precision=np.asarray(model_precision_frames, dtype=np.float32),
        covariance_diag=np.asarray(model_covariance_diag_frames, dtype=np.float32),
        centers=centers.astype(np.float32),
        axis=grid_axis.astype(np.float32),
        width=np.asarray([width], dtype=np.float32),
        support_radius=np.asarray([support_radius], dtype=np.int64),
    )

    rollout_metrics = _extract_rollout_metrics(result_dir)
    payload = _build_metadata(
        exp_id=exp_id,
        policy_id=policy_id,
        seed=seed,
        total_steps=total_steps,
        run_dir=run_dir,
        status="completed",
        start_time=start_time,
        end_time=ended.isoformat().replace("+00:00", "Z"),
        runtime_sec=(
            ended - datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        ).total_seconds(),
        results_path=result_dir,
        extra={
            **rollout_metrics,
            "system_id": str(system_spec.system_id),
            "system_label": str(system_spec.label),
            "dynamics_type": str(system_spec.dynamics_type),
            "initial_state_true": [float(x) for x in init_state.tolist()],
            "embedding_true": [float(x) for x in e_true.reshape(-1).tolist()],
            "dynamics_mse_final": (
                float(dynamics_rows[-1]["dynamics_mse"]) if dynamics_rows else None
            ),
            "dynamics_mse_mean": (
                float(np.mean([row["dynamics_mse"] for row in dynamics_rows]))
                if dynamics_rows
                else None
            ),
            "trajectory_r2_final": float(traj_rows[-1]["trajectory_r2"]) if traj_rows else None,
            "objective_variant": (
                str(policy_spec.objective_kind) if not policy_spec.passive else None
            ),
            "parameter_belief_type": "structured_dense_overlap",
            "schedule_id": str(schedule_spec.schedule_id),
            "update_interval": int(schedule_spec.update_interval),
            "replan_interval": int(schedule_spec.replan_interval),
            "planning_horizon": int(schedule_spec.planning_horizon),
            "planning_chunk": int(schedule_spec.planning_chunk),
            "predictive_only_window": bool(schedule_spec.predictive_only_window),
            "state_update_interval": int(schedule_spec.update_interval),
            "parameter_update_interval": int(schedule_spec.update_interval),
            "q_theta": float(q_theta),
            "action_max": float(action_max),
            "state_noise": float(noise_scale),
            "dynamics_alpha": float(alpha),
            "state_init_uncertainty": float(env_preset.state_init_uncertainty),
            "firing_rate_scale": float(env_preset.firing_rate_scale),
            "mean_firing_rate_target": float(mean_firing),
            "max_firing_rate_target": float(max_firing_rate),
            "rbf_grid_shape": [int(grid_shape[0]), int(grid_shape[1])],
            "rbf_parameter_dim": int(d_embed),
            "rbf_support_manhattan_radius": int(support_radius),
            "rbf_width": float(width),
            "traj_eval_interval": int(traj_eval_interval),
            "traj_eval_horizon": int(traj_eval_horizon),
            "traj_eval_samples": int(traj_eval_samples),
            "mpc_num_iterations": 4 if not policy_spec.passive else None,
            "mpc_num_samples": 24 if not policy_spec.passive else None,
            "mpc_num_elite": 6 if not policy_spec.passive else None,
            "dynamics_mse_trace_path": str(dynamics_trace_path),
            "trajectory_r2_trace_path": str(traj_trace_path),
            "information_trace_path": str(info_trace_path),
            "state_action_trace_path": str(state_action_trace_path),
            "planned_trajectory_trace_path": (
                str(planned_trace_path) if planned_traj_frames else None
            ),
            "acquisition_map_trace_path": (
                str(acq_trace_path) if policy_spec.save_acq_map and acq_map_frames else None
            ),
            "rbf_model_trace_path": str(rbf_trace_path),
            "writing_ref": WRITING_REFERENCE,
        },
    )
    return payload


def _replay_parameter_info(state: np.ndarray, state_dim: int) -> np.ndarray:
    x = np.asarray(state, dtype=np.float64).reshape(-1)
    xx = np.outer(x, x)
    return np.kron(xx, np.eye(int(state_dim), dtype=np.float64))


def _replay_candidate_score(
    *,
    objective_kind: str | None,
    state: np.ndarray,
    next_state: np.ndarray,
    spikes: np.ndarray,
    info_accum: np.ndarray,
    planning_mode: bool,
) -> tuple[float, np.ndarray]:
    state_vec = np.asarray(state, dtype=np.float64).reshape(-1)
    next_vec = np.asarray(next_state, dtype=np.float64).reshape(-1)
    delta = next_vec - state_vec
    info_step = _replay_parameter_info(state_vec, state_vec.shape[0])
    reference = info_accum + info_step if planning_mode else info_step
    ref_eye = np.eye(reference.shape[0], dtype=np.float64)

    if objective_kind in {None, "parameter_eig", "fully_observable_parameter_eig"}:
        sign, logdet = np.linalg.slogdet(ref_eye + reference)
        score = float(logdet if sign > 0 else -1e12)
    elif objective_kind == "e_optimality":
        score = float(np.min(np.linalg.eigvalsh(reference + 1e-8 * ref_eye)))
    elif objective_kind == "state_information":
        score = float(np.dot(state_vec, state_vec))
    elif objective_kind == "dynamics":
        score = float(np.dot(delta, delta))
    elif objective_kind == "sampling_variance":
        score = float(np.var(np.asarray(spikes, dtype=np.float64)))
    else:
        raise ValueError(f"Unsupported replay objective_kind={objective_kind!r}")
    return score, info_step


def _prbs_selection_order(num_items: int, budget: int) -> np.ndarray:
    if num_items <= 0 or budget <= 0:
        return np.zeros((0,), dtype=np.int64)
    anchors = np.linspace(0.0, float(num_items - 1), int(budget))
    order: list[int] = []
    seen: set[int] = set()
    for idx in np.round(anchors).astype(int).tolist():
        idx_i = int(np.clip(idx, 0, num_items - 1))
        if idx_i not in seen:
            seen.add(idx_i)
            order.append(idx_i)
    for idx in range(num_items):
        if idx not in seen:
            order.append(idx)
    return np.asarray(order[: min(num_items, budget)], dtype=np.int64)


def _run_single_realdata_replay(
    *,
    exp_id: str,
    policy_id: str,
    seed: int,
    total_steps: int,
    run_dir: Path,
) -> dict[str, Any]:
    exp_spec = get_experiment_spec(exp_id)
    policy_spec = get_policy_spec(policy_id)
    schedule_spec = get_schedule_spec(policy_spec.schedule_id)
    env_preset = get_environment_preset(exp_spec.env_preset_id)

    start_time = _utc_now()
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    dataset = load_replay_dataset(
        dataset_id=str(env_preset.dataset_id or env_preset.system_id),
        dataset_path=str(env_preset.dataset_path or ""),
        state_key=str(env_preset.state_key),
        observation_key=str(env_preset.observation_key),
        latent_dim=int(env_preset.latent_dim),
        max_observation_dim=env_preset.max_observation_dim,
        time_bin_ms=float(env_preset.time_bin_ms),
    )
    x_all, y_all = build_transition_matrices(dataset)
    spike_all = np.asarray(dataset.spikes[:-1], dtype=np.float64)
    train_idx, eval_idx = split_replay_dataset(
        dataset,
        train_fraction=float(env_preset.train_fraction),
    )
    if eval_idx.size == 0:
        raise ValueError("Real-data replay split produced an empty evaluation set")

    budget = min(int(total_steps), int(train_idx.size))
    if budget <= 0:
        raise ValueError("No replay transitions available for selection")

    policy_type = _resolved_policy_type(policy_id, policy_spec)
    planning_mode = policy_type == "mpc-icem" and int(schedule_spec.planning_horizon) > 2
    prbs_order = _prbs_selection_order(int(train_idx.size), budget)

    state_dim = int(x_all.shape[1])
    info_dim = int(state_dim * state_dim)
    info_accum = np.zeros((info_dim, info_dim), dtype=np.float64)
    ridge = 1e-3
    selected_positions: list[int] = []
    remaining: list[int] = list(range(int(train_idx.size)))
    dynamics_rows: list[dict[str, Any]] = []
    info_rows: list[dict[str, Any]] = []
    traj_rows: list[dict[str, Any]] = []
    state_action_rows: list[dict[str, Any]] = []
    perf_start = time.perf_counter()
    coef = np.zeros((state_dim, state_dim), dtype=np.float64)

    def _pop_selected_position(step: int) -> tuple[int, float, np.ndarray]:
        nonlocal info_accum
        if policy_type == "off-policy":
            pos = remaining.pop(0)
            score, info_step = _replay_candidate_score(
                objective_kind="parameter_eig",
                state=x_all[train_idx[pos]],
                next_state=y_all[train_idx[pos]],
                spikes=spike_all[train_idx[pos]],
                info_accum=info_accum,
                planning_mode=False,
            )
            return pos, score, info_step
        if policy_type in {"random", "baseline-random"}:
            rem_idx = int(rng.integers(0, len(remaining)))
            pos = remaining.pop(rem_idx)
            score, info_step = _replay_candidate_score(
                objective_kind="parameter_eig",
                state=x_all[train_idx[pos]],
                next_state=y_all[train_idx[pos]],
                spikes=spike_all[train_idx[pos]],
                info_accum=info_accum,
                planning_mode=False,
            )
            return pos, score, info_step
        if policy_type == "baseline-prbs":
            for candidate in prbs_order.tolist():
                if candidate in remaining:
                    remaining.remove(candidate)
                    score, info_step = _replay_candidate_score(
                        objective_kind="parameter_eig",
                        state=x_all[train_idx[candidate]],
                        next_state=y_all[train_idx[candidate]],
                        spikes=spike_all[train_idx[candidate]],
                        info_accum=info_accum,
                        planning_mode=False,
                    )
                    return int(candidate), score, info_step
            pos = remaining.pop(0)
            score, info_step = _replay_candidate_score(
                objective_kind="parameter_eig",
                state=x_all[train_idx[pos]],
                next_state=y_all[train_idx[pos]],
                spikes=spike_all[train_idx[pos]],
                info_accum=info_accum,
                planning_mode=False,
            )
            return pos, score, info_step

        best_pos = remaining[0]
        best_score = -np.inf
        best_info = np.zeros_like(info_accum)
        for pos in remaining:
            score, info_step = _replay_candidate_score(
                objective_kind=policy_spec.objective_kind,
                state=x_all[train_idx[pos]],
                next_state=y_all[train_idx[pos]],
                spikes=spike_all[train_idx[pos]],
                info_accum=info_accum,
                planning_mode=planning_mode,
            )
            if score > best_score:
                best_pos = int(pos)
                best_score = float(score)
                best_info = info_step
        remaining.remove(best_pos)
        return best_pos, best_score, best_info

    for step in range(1, budget + 1):
        pos, score, info_step = _pop_selected_position(step)
        selected_positions.append(int(pos))
        selected_idx = train_idx[np.asarray(selected_positions, dtype=np.int64)]
        coef = fit_linear_dynamics_ridge(x_all[selected_idx], y_all[selected_idx], ridge=ridge)
        info_accum = info_accum + info_step

        cpu_time_sec = float(time.perf_counter() - perf_start)
        dynamics_mse = evaluate_prediction_mse(x_all[eval_idx], y_all[eval_idx], coef)
        trajectory_r2 = evaluate_prediction_r2(x_all[eval_idx], y_all[eval_idx], coef)

        cov_state = (
            np.cov(x_all[selected_idx].T).astype(np.float64, copy=False)
            if selected_idx.size > 1
            else np.eye(state_dim, dtype=np.float64)
        )
        cov_state = np.atleast_2d(cov_state)
        eigvals = np.linalg.eigvalsh(info_accum + 1e-8 * np.eye(info_dim, dtype=np.float64))
        sign, logdet = np.linalg.slogdet(np.eye(info_dim, dtype=np.float64) + info_accum)
        info_rows.append(
            {
                "step": step,
                "cpu_time_sec": cpu_time_sec,
                "I_z_t": float(score),
                "I_theta_t": float(logdet if sign > 0 else np.nan),
                "Pz00": float(cov_state[0, 0]),
                "Pz01": float(cov_state[0, 1]) if cov_state.shape[1] > 1 else 0.0,
                "Pz11": float(cov_state[1, 1]) if cov_state.shape[0] > 1 else float(cov_state[0, 0]),
                "state_posterior_updated": True,
                "parameter_posterior_updated": True,
                "window_buffer_length": 0,
                "eigmin_info": float(eigvals[0]),
            }
        )
        dynamics_rows.append(
            {
                "step": step,
                "cpu_time_sec": cpu_time_sec,
                "dynamics_mse": float(dynamics_mse),
            }
        )
        traj_rows.append(
            {
                "step": step,
                "cpu_time_sec": cpu_time_sec,
                "trajectory_r2": float(trajectory_r2),
                "traj_eval_horizon": 1,
                "traj_eval_samples": int(eval_idx.size),
            }
        )

        sample_idx = int(train_idx[pos])
        state = np.asarray(x_all[sample_idx], dtype=np.float64)
        target = np.asarray(y_all[sample_idx], dtype=np.float64)
        pred = np.asarray(state @ coef, dtype=np.float64)
        zero_action = np.zeros((1,), dtype=np.float64)
        state_action_rows.append(
            {
                "step": step,
                "cpu_time_sec": cpu_time_sec,
                "true_x": float(state[0]),
                "true_v": float(state[1]) if state.shape[0] > 1 else 0.0,
                "model_x": float(state[0]),
                "model_v": float(state[1]) if state.shape[0] > 1 else 0.0,
                "next_model_x": float(pred[0]),
                "next_model_v": float(pred[1]) if pred.shape[0] > 1 else 0.0,
                "action_x": float(zero_action[0]),
                "action_v": 0.0,
                "action_norm": 0.0,
                "policy_action_x": float(zero_action[0]),
                "policy_action_v": 0.0,
                "policy_action_norm": 0.0,
                "env_action_x": float(zero_action[0]),
                "env_action_v": 0.0,
                "env_action_norm": 0.0,
                "policy_action_delta_norm": 0.0,
                "execution_delta_norm": 0.0,
                "action_total_delta_norm": 0.0,
                "action_clipped": False,
                "env_action_clipped": False,
                "planned_at_bound": False,
                "policy_at_bound": False,
                "env_action_at_bound": False,
                "policy_cost": -float(score),
                "state_posterior_updated": True,
                "parameter_posterior_updated": True,
                "window_buffer_length": 0,
                "true_next_x": float(target[0]),
                "true_next_v": float(target[1]) if target.shape[0] > 1 else 0.0,
            }
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    dynamics_trace_path = run_dir / "dynamics_mse_trace.csv"
    traj_trace_path = run_dir / "trajectory_r2_trace.csv"
    info_trace_path = run_dir / "information_trace.csv"
    state_action_trace_path = run_dir / "state_action_trace.csv"
    _write_trace_csv(dynamics_trace_path, dynamics_rows, ["step", "cpu_time_sec", "dynamics_mse"])
    _write_trace_csv(
        traj_trace_path,
        traj_rows,
        ["step", "cpu_time_sec", "trajectory_r2", "traj_eval_horizon", "traj_eval_samples"],
    )
    _write_trace_csv(
        info_trace_path,
        info_rows,
        [
            "step",
            "cpu_time_sec",
            "I_z_t",
            "I_theta_t",
            "Pz00",
            "Pz01",
            "Pz11",
            "state_posterior_updated",
            "parameter_posterior_updated",
            "window_buffer_length",
            "eigmin_info",
        ],
    )
    _write_trace_csv(
        state_action_trace_path,
        state_action_rows,
        [
            "step",
            "cpu_time_sec",
            "true_x",
            "true_v",
            "model_x",
            "model_v",
            "next_model_x",
            "next_model_v",
            "action_x",
            "action_v",
            "action_norm",
            "policy_action_x",
            "policy_action_v",
            "policy_action_norm",
            "env_action_x",
            "env_action_v",
            "env_action_norm",
            "policy_action_delta_norm",
            "execution_delta_norm",
            "action_total_delta_norm",
            "action_clipped",
            "env_action_clipped",
            "planned_at_bound",
            "policy_at_bound",
            "env_action_at_bound",
            "policy_cost",
            "state_posterior_updated",
            "parameter_posterior_updated",
            "window_buffer_length",
            "true_next_x",
            "true_next_v",
        ],
    )

    ended = datetime.now(timezone.utc)
    result_dir = run_dir
    rollout_metrics = _extract_rollout_metrics(result_dir)
    final_coef = coef.reshape(-1)
    payload = _build_metadata(
        exp_id=exp_id,
        policy_id=policy_id,
        seed=seed,
        total_steps=budget,
        run_dir=run_dir,
        status="completed",
        start_time=start_time,
        end_time=ended.isoformat().replace("+00:00", "Z"),
        runtime_sec=(
            ended - datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        ).total_seconds(),
        results_path=result_dir,
        extra={
            **rollout_metrics,
            "system_id": str(env_preset.system_id),
            "system_label": str(env_preset.system_label or env_preset.system_id),
            "dynamics_type": "replay_dataset",
            "dataset_id": str(dataset.dataset_id),
            "dataset_path": str(dataset.source_path),
            "real_data": True,
            "replay_protocol": "offline_counterfactual_selection",
            "observation_key": str(env_preset.observation_key),
            "state_key": str(env_preset.state_key),
            "train_fraction": float(env_preset.train_fraction),
            "time_bin_ms": float(env_preset.time_bin_ms),
            "selection_budget": int(budget),
            "num_train_transitions": int(train_idx.size),
            "num_eval_transitions": int(eval_idx.size),
            "observation_dim": int(dataset.spikes.shape[1]),
            "latent_dim": int(dataset.states.shape[1]),
            "embedding_estimate": [float(x) for x in final_coef.tolist()],
            "dynamics_mse_final": (
                float(dynamics_rows[-1]["dynamics_mse"]) if dynamics_rows else None
            ),
            "dynamics_mse_mean": (
                float(np.mean([row["dynamics_mse"] for row in dynamics_rows]))
                if dynamics_rows
                else None
            ),
            "trajectory_r2_final": float(traj_rows[-1]["trajectory_r2"]) if traj_rows else None,
            "objective_variant": (
                str(policy_spec.objective_kind)
                if policy_spec.objective_kind is not None
                else "uniform_replay"
            ),
            "policy_type": str(policy_type),
            "schedule_id": str(schedule_spec.schedule_id),
            "update_interval": int(schedule_spec.update_interval),
            "replan_interval": int(schedule_spec.replan_interval),
            "planning_horizon": int(schedule_spec.planning_horizon),
            "planning_chunk": int(schedule_spec.planning_chunk),
            "predictive_only_window": False,
            "dynamics_mse_trace_path": str(dynamics_trace_path),
            "trajectory_r2_trace_path": str(traj_trace_path),
            "information_trace_path": str(info_trace_path),
            "state_action_trace_path": str(state_action_trace_path),
            "writing_ref": WRITING_REFERENCE,
            "dataset_metadata": dict(dataset.metadata),
        },
    )
    return payload


def _run_one(
    *, exp_id: str, policy_id: str, seed: int, repeat: int, base_dir: Path, args: argparse.Namespace
) -> dict[str, Any]:
    exp_spec = get_experiment_spec(exp_id)
    total_steps = int(args.total_steps or exp_spec.total_steps)
    run_dir = base_dir / exp_id / "track" / policy_id / f"seed_{seed}" / f"repeat_{repeat:02d}"
    _ensure_dir(run_dir)
    try:
        if exp_spec.experiment_kind == "duffing":
            payload = _run_single_duffing_identification(
                exp_id=exp_id,
                policy_id=policy_id,
                seed=seed,
                total_steps=total_steps,
                run_dir=run_dir,
                eig_gamma=float(args.eig_gamma),
                q_theta=float(args.q_theta),
                q_theta_meas_coeff=float(args.q_theta_meas_coeff),
                q_theta_max_scale=float(args.q_theta_max_scale),
                traj_eval_interval=int(exp_spec.trajectory_eval_interval),
                traj_eval_horizon=int(exp_spec.trajectory_eval_horizon),
                traj_eval_samples=int(exp_spec.trajectory_eval_samples),
                acq_map_interval=int(args.acq_map_interval),
                acq_map_grid=int(args.acq_map_grid),
                acq_map_lim=float(args.acq_map_lim),
                sampling_variance_samples=int(args.sampling_variance_samples),
            )
        elif exp_spec.experiment_kind == "realdata":
            payload = _run_single_realdata_replay(
                exp_id=exp_id,
                policy_id=policy_id,
                seed=seed,
                total_steps=total_steps,
                run_dir=run_dir,
            )
        else:
            payload = _run_single_rbf_identification(
                exp_id=exp_id,
                policy_id=policy_id,
                seed=seed,
                total_steps=total_steps,
                run_dir=run_dir,
                eig_gamma=float(args.eig_gamma),
                q_theta=float(args.q_theta),
                traj_eval_interval=int(exp_spec.trajectory_eval_interval),
                traj_eval_horizon=int(exp_spec.trajectory_eval_horizon),
                traj_eval_samples=int(exp_spec.trajectory_eval_samples),
                acq_map_interval=int(args.acq_map_interval),
                acq_map_grid=int(args.acq_map_grid),
                acq_map_lim=float(args.acq_map_lim),
            )
    except Exception as exc:
        payload = _build_metadata(
            exp_id=exp_id,
            policy_id=policy_id,
            seed=seed,
            total_steps=total_steps,
            run_dir=run_dir,
            status="failed",
            start_time=_utc_now(),
            end_time=_utc_now(),
            runtime_sec=0.0,
            results_path=run_dir,
            extra={"error": f"{type(exc).__name__}: {exc}"},
        )
    write_json(run_dir / "run_metadata.json", payload)
    return payload


def run_matrix(
    *, exp_ids: list[str], seeds: list[int], repeats: int, base_dir: Path, args: argparse.Namespace
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    policy_filter = set(parse_csv_list(getattr(args, "policy_ids", None))) or None
    for exp_id in exp_ids:
        exp_spec = get_experiment_spec(exp_id)
        selected_policy_ids = tuple(
            policy_id
            for policy_id in exp_spec.policy_ids
            if policy_filter is None or policy_id in policy_filter
        )
        for policy_id in selected_policy_ids:
            for seed in seeds:
                for repeat in range(1, repeats + 1):
                    records.append(
                        _run_one(
                            exp_id=exp_id,
                            policy_id=policy_id,
                            seed=seed,
                            repeat=repeat,
                            base_dir=base_dir,
                            args=args,
                        )
                    )
    return records


def _runner_model_name(experiment_kind: str) -> str:
    if experiment_kind == "duffing":
        return "FilteringEmbedding"
    if experiment_kind == "rbf":
        return "SparseRbfFilteringModel"
    if experiment_kind == "realdata":
        return "ReplayLinearDynamics"
    return "unknown"


def _dynamics_model_name(experiment_kind: str) -> str:
    if experiment_kind == "duffing":
        return "FunctionDynamics"
    if experiment_kind == "rbf":
        return "SparseRbfDynamics"
    if experiment_kind == "realdata":
        return "LinearReplayFit"
    return "unknown"


def _build_session_experiment_entry(
    *,
    exp_id: str,
    seeds: list[int],
    repeats: int,
    total_steps_override: int | None,
    policy_filter: set[str] | None = None,
) -> dict[str, Any]:
    exp_spec = get_experiment_spec(exp_id)
    env_preset = get_environment_preset(exp_spec.env_preset_id)
    env_summary = _environment_summary(env_preset)
    selected_policy_ids = tuple(
        policy_id
        for policy_id in exp_spec.policy_ids
        if policy_filter is None or policy_id in policy_filter
    )
    policies: list[dict[str, Any]] = []
    for policy_id in selected_policy_ids:
        policy_spec = get_policy_spec(policy_id)
        schedule_spec = get_schedule_spec(policy_spec.schedule_id)
        policies.append(
            {
                "policy_id": str(policy_id),
                "passive": bool(policy_spec.passive),
                "policy_type": _resolved_policy_type(policy_id, policy_spec),
                "objective_kind": (
                    str(policy_spec.objective_kind)
                    if policy_spec.objective_kind is not None
                    else None
                ),
                "save_acq_map": bool(policy_spec.save_acq_map),
                "schedule_id": str(schedule_spec.schedule_id),
                "schedule": {
                    "update_interval": int(schedule_spec.update_interval),
                    "replan_interval": int(schedule_spec.replan_interval),
                    "planning_horizon": int(schedule_spec.planning_horizon),
                    "planning_chunk": int(schedule_spec.planning_chunk),
                    "predictive_only_window": bool(schedule_spec.predictive_only_window),
                },
                "model": {
                    "runner": str(exp_spec.experiment_kind),
                    "filter_model": _runner_model_name(str(exp_spec.experiment_kind)),
                    "dynamics_model": _dynamics_model_name(str(exp_spec.experiment_kind)),
                    "residual_form": True,
                },
            }
        )
    return {
        "exp_id": str(exp_spec.exp_id),
        "experiment_kind": str(exp_spec.experiment_kind),
        "total_steps_default": int(exp_spec.total_steps),
        "total_steps_resolved": int(total_steps_override or exp_spec.total_steps),
        "summary_value_kind": str(exp_spec.summary_value_kind),
        "summary_value_label": str(exp_spec.summary_value_label),
        "trajectory_eval_interval": int(exp_spec.trajectory_eval_interval),
        "trajectory_eval_horizon": int(exp_spec.trajectory_eval_horizon),
        "trajectory_eval_samples": int(exp_spec.trajectory_eval_samples),
        "environment": {
            "preset_id": str(env_preset.preset_id),
            "system_id": str(env_summary["system_id"]),
            "system_label": str(env_summary["system_label"]),
            "dynamics_type": str(env_summary["dynamics_type"]),
            "estimator_system_id": env_summary.get("estimator_system_id"),
            "estimator_system_label": env_summary.get("estimator_system_label"),
            "estimator_dynamics_type": env_summary.get("estimator_dynamics_type"),
            "dt": float(env_preset.dt),
            "action_dim": int(env_preset.action_dim),
            "latent_dim": int(env_preset.latent_dim),
            "embedding_dim": int(env_preset.embedding_dim),
            "observation_dim": int(env_preset.observation_dim),
            "action_max": float(env_preset.action_max),
            "dynamics_alpha": float(env_preset.dynamics_alpha),
            "state_noise": float(env_preset.state_noise),
            "state_init_uncertainty": float(env_preset.state_init_uncertainty),
            "firing_rate_scale": float(env_preset.firing_rate_scale),
            "observation_noise_scale": float(env_preset.observation_noise_scale),
            "observation_noise_type": str(env_preset.observation_noise_type),
            "mean_firing_rate_target": float(env_preset.mean_firing_rate_target),
            "max_firing_rate_target": float(env_preset.max_firing_rate_target),
            "asymmetric_loading": bool(env_preset.asymmetric_loading),
            "x_range": float(env_preset.x_range),
            "real_data": bool(getattr(env_preset, "real_data", False)),
            "dataset_id": getattr(env_preset, "dataset_id", None),
            "dataset_path": getattr(env_preset, "dataset_path", None),
            "state_key": getattr(env_preset, "state_key", None),
            "observation_key": getattr(env_preset, "observation_key", None),
            "train_fraction": float(getattr(env_preset, "train_fraction", 0.7)),
            "time_bin_ms": float(getattr(env_preset, "time_bin_ms", 20.0)),
            "max_observation_dim": getattr(env_preset, "max_observation_dim", None),
            "true_embedding": env_summary["true_embedding"],
        },
        "policies": policies,
        "seeds": [int(seed) for seed in seeds],
        "repeats": int(repeats),
        "planned_runs": int(len(selected_policy_ids) * len(seeds) * repeats),
    }


def _build_session_metadata(
    *,
    session_root: Path,
    args: argparse.Namespace,
    raw_argv: list[str],
    exp_ids: list[str],
    seeds: list[int],
    repeats: int,
    records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    policy_filter = set(parse_csv_list(getattr(args, "policy_ids", None))) or None
    experiments = [
        _build_session_experiment_entry(
            exp_id=exp_id,
            seeds=seeds,
            repeats=repeats,
            total_steps_override=args.total_steps,
            policy_filter=policy_filter,
        )
        for exp_id in exp_ids
    ]
    requested_ops = {
        "run": args.mode in {"run", "all"},
        "summary": args.mode in {"summary", "all"},
        "video": args.mode in {"video", "all"},
    }
    payload: dict[str, Any] = {
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "commit": _current_commit(),
        "repo_root": str(_repo_root()),
        "cwd": str(Path.cwd()),
        "session_root": str(session_root),
        "catalogs": describe_catalogs(),
        "cli": {
            "argv": [str(x) for x in raw_argv],
            "command": shlex.join([str(sys.executable), *[str(x) for x in raw_argv]]),
            "python_executable": str(sys.executable),
            "script": str(raw_argv[0]) if raw_argv else None,
        },
        "parameters": {
            **{str(key): value for key, value in vars(args).items()},
            "exp_ids_resolved": [str(exp_id) for exp_id in exp_ids],
            "seeds_resolved": [int(seed) for seed in seeds],
            "repeats_resolved": int(repeats),
            "base_dir_resolved": str(session_root),
        },
        "requested_operations": requested_ops,
        "experiments": experiments,
    }
    if records is not None:
        failed_runs = [
            {
                "exp_id": str(record.get("exp_id")),
                "policy_id": str(record.get("policy_id")),
                "seed": int(record.get("seed", -1)),
                "status": str(record.get("status")),
                "error": record.get("error"),
                "results_path": record.get("results_path"),
            }
            for record in records
            if str(record.get("status")) != "completed"
        ]
        by_exp: dict[str, dict[str, Any]] = {}
        for record in records:
            exp_id = str(record.get("exp_id"))
            bucket = by_exp.setdefault(
                exp_id,
                {"exp_id": exp_id, "total_runs": 0, "completed_runs": 0, "failed_runs": 0},
            )
            bucket["total_runs"] += 1
            if str(record.get("status")) == "completed":
                bucket["completed_runs"] += 1
            else:
                bucket["failed_runs"] += 1
        payload["run_summary"] = {
            "total_runs": int(len(records)),
            "completed_runs": int(
                sum(1 for record in records if str(record.get("status")) == "completed")
            ),
            "failed_runs": int(
                sum(1 for record in records if str(record.get("status")) != "completed")
            ),
            "by_experiment": list(by_exp.values()),
            "failed_run_records": failed_runs,
        }
    return payload


def _write_session_metadata(
    *,
    session_root: Path,
    args: argparse.Namespace,
    raw_argv: list[str],
    exp_ids: list[str],
    seeds: list[int],
    repeats: int,
    records: list[dict[str, Any]] | None = None,
) -> None:
    created_at = None
    metadata_path = session_root / "session_metadata.json"
    if metadata_path.exists():
        try:
            created_at = load_json(metadata_path).get("created_at")
        except Exception:
            try:
                created_at = json.loads(metadata_path.read_text(encoding="utf-8")).get("created_at")
            except Exception:
                created_at = None
    payload = _build_session_metadata(
        session_root=session_root,
        args=args,
        raw_argv=raw_argv,
        exp_ids=exp_ids,
        seeds=seeds,
        repeats=repeats,
        records=records,
    )
    if isinstance(created_at, str) and created_at.strip():
        payload["created_at"] = created_at
    payload["updated_at"] = _utc_now()
    write_json(metadata_path, payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run experiments from configured catalogs",
        allow_abbrev=False,
    )
    parser.add_argument("--env-catalog", action="append", dest="env_catalogs")
    parser.add_argument("--model-catalog", action="append", dest="model_catalogs")
    parser.add_argument("--suite-catalog", action="append", dest="suite_catalogs")
    parser.add_argument("--exp-id", choices=[*list_experiment_ids(), "all"], default="all")
    parser.add_argument("--exp-ids", type=str, default=None)
    parser.add_argument("--policy-ids", type=str, default=None)
    parser.add_argument("--mode", choices=["run", "summary", "video", "all"], default="run")
    parser.add_argument("--seeds", type=str, default="0,10,20,30")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--base-dir", type=str, default="results/experiments")
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--q-theta", type=float, default=5e-4)
    parser.add_argument("--q-theta-meas-coeff", type=float, default=0.0)
    parser.add_argument("--q-theta-max-scale", type=float, default=10.0)
    parser.add_argument("--eig-gamma", type=float, default=1.0)
    parser.add_argument("--sampling-variance-samples", type=int, default=8)
    parser.add_argument("--acq-map-interval", type=int, default=5)
    parser.add_argument("--acq-map-grid", type=int, default=61)
    parser.add_argument("--acq-map-lim", type=float, default=10.0)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--grid-lim", type=float, default=10.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    catalog_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    catalog_parser.add_argument("--env-catalog", action="append", dest="env_catalogs")
    catalog_parser.add_argument("--model-catalog", action="append", dest="model_catalogs")
    catalog_parser.add_argument("--suite-catalog", action="append", dest="suite_catalogs")
    catalog_args, _unknown = catalog_parser.parse_known_args(argv_list)
    configure_catalogs(
        env_catalog_paths=catalog_args.env_catalogs,
        model_catalog_paths=catalog_args.model_catalogs,
        suite_catalog_paths=catalog_args.suite_catalogs,
    )
    catalog_desc = describe_catalogs()
    env_catalogs = catalog_desc.get("environment")
    model_catalogs = catalog_desc.get("model")
    suite_catalogs = catalog_desc.get("suite")
    os.environ["ACTDYN_ENV_CATALOGS"] = (
        str(env_catalogs)
        if isinstance(env_catalogs, str)
        else os.pathsep.join(str(item) for item in env_catalogs)
    )
    os.environ["ACTDYN_MODEL_CATALOGS"] = (
        str(model_catalogs)
        if isinstance(model_catalogs, str)
        else os.pathsep.join(str(item) for item in model_catalogs)
    )
    os.environ["ACTDYN_SUITE_CATALOGS"] = (
        str(suite_catalogs)
        if isinstance(suite_catalogs, str)
        else os.pathsep.join(str(item) for item in suite_catalogs)
    )
    raw_argv = [str(Path(__file__).resolve()), *argv_list]
    parser = build_parser()
    args = parser.parse_args(argv_list)
    if args.exp_ids is not None and str(args.exp_ids).strip():
        exp_ids = parse_csv_list(args.exp_ids)
        unknown = [exp_id for exp_id in exp_ids if exp_id not in set(list_experiment_ids())]
        if unknown:
            parser.error(f"Unknown experiment ids: {', '.join(unknown)}")
    else:
        exp_ids = list_experiment_ids() if args.exp_id == "all" else [str(args.exp_id)]
    policy_filter = set(parse_csv_list(args.policy_ids)) or None
    if policy_filter is not None:
        available_policy_ids = {
            policy_id
            for exp_id in exp_ids
            for policy_id in get_experiment_spec(exp_id).policy_ids
        }
        unknown_policy_ids = sorted(policy_filter - available_policy_ids)
        if unknown_policy_ids:
            parser.error(f"Unknown policy ids for selected experiments: {', '.join(unknown_policy_ids)}")
    base_dir = resolve_session_root(
        Path(args.base_dir),
        create=args.mode in {"run", "all"},
        exp_ids=exp_ids,
    )
    seeds = parse_csv_ints(args.seeds) or [0, 10, 20, 30]
    repeats = int(args.repeats)
    if args.mode in {"run", "all"}:
        _write_session_metadata(
            session_root=base_dir,
            args=args,
            raw_argv=raw_argv,
            exp_ids=exp_ids,
            seeds=seeds,
            repeats=repeats,
            records=None,
        )
    log_dir = base_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    redirect_output = not sys.stdout.isatty() or not sys.stderr.isatty()
    log_path = log_dir / "experiment_driver.log"
    with (
        log_path.open("a", encoding="utf-8") if redirect_output else contextlib.nullcontext()
    ) as log_stream:
        stdout_cm = (
            contextlib.redirect_stdout(log_stream) if redirect_output else contextlib.nullcontext()
        )
        stderr_cm = (
            contextlib.redirect_stderr(log_stream) if redirect_output else contextlib.nullcontext()
        )
        with stdout_cm, stderr_cm:
            records: list[dict[str, Any]] = []
            if args.mode in {"run", "all"}:
                records = run_matrix(
                    exp_ids=exp_ids,
                    seeds=seeds,
                    repeats=repeats,
                    base_dir=base_dir,
                    args=args,
                )
                _write_session_metadata(
                    session_root=base_dir,
                    args=args,
                    raw_argv=raw_argv,
                    exp_ids=exp_ids,
                    seeds=seeds,
                    repeats=repeats,
                    records=records,
                )
            if args.mode in {"summary", "all"}:
                if __package__ in {None, ""}:
                    from summarize_experiments import main as summarize_main
                else:
                    from .summarize_experiments import main as summarize_main

                for exp_id in exp_ids:
                    summarize_main(
                        ["--base-dir", str(base_dir), "--exp-id", exp_id, "--seeds", args.seeds]
                    )
            if args.mode in {"video", "all"}:
                if __package__ in {None, ""}:
                    from render_experiment_videos import main as render_main
                else:
                    from .render_experiment_videos import main as render_main

                for exp_id in exp_ids:
                    render_main(
                        [
                            "--base-dir",
                            str(base_dir),
                            "--exp-id",
                            exp_id,
                            "--seeds",
                            args.seeds,
                            "--stride",
                            str(args.stride),
                            "--fps",
                            str(args.fps),
                            "--grid-lim",
                            str(args.grid_lim),
                        ]
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
