#!/usr/bin/env python3
"""Cosyne helper to preflight and run parameter-identification experiments."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib
import inspect
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
from cosyne_common import (
    parse_csv_ints as _parse_csv_ints,
    parse_csv_list as _parse_csv_list,
    write_json as _write_json,
)

DEFAULT_EXP_IDS = ["active_short", "random", "no_policy"]
DEFAULT_SEEDS = [0, 10, 20]
APPROXIMATION_EXP_IDS = {"frozen_cov", "no_sensitivity", "fully_observed"}
FIXED_ACTIVE_EXP_IDS = {
    "active_long",
    "active_short",
    "active_chunk",
    "async_window",
    "async_chunk",
    "window_update",
}
LEGACY_EXP_ID_ALIASES = {
    "step_update": "active_long",
}
SUPPORTED_EXP_IDS = {
    *FIXED_ACTIVE_EXP_IDS,
    *LEGACY_EXP_ID_ALIASES,
    "RND",
    "random",
    "no_policy",
    *APPROXIMATION_EXP_IDS,
}
PASSIVE_EXP_IDS = {"random", "no_policy"}

REQUIRED_CONFIGS = [
    "experiments/ciss/conf/config.yaml",
    "experiments/ciss/conf/intro_video.yaml",
]
REQUIRED_IMPORTS = [
    "actdyn",
    "einops",
]
WRITING_REFERENCE = "docs/active-dynamics-writing/methods.tex"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


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


def _validate_exp_id(exp_id: str) -> None:
    if exp_id not in SUPPORTED_EXP_IDS:
        raise ValueError(f"Unsupported exp_id={exp_id}. Expected one of {sorted(SUPPORTED_EXP_IDS)}")


def _canonical_exp_id(exp_id: str) -> str:
    return LEGACY_EXP_ID_ALIASES.get(exp_id, exp_id)


def _resolve_schedule_from_request(
    *,
    exp_id: str,
    requested_k_theta: int,
    requested_planning_horizon: int | None,
) -> dict[str, Any]:
    requested_k_theta = max(1, int(requested_k_theta))
    effective_k_theta = 1 if exp_id == "step_update" else requested_k_theta
    state_update_interval = requested_k_theta if exp_id == "window_update" else 1
    predictive_only_window = exp_id == "window_update"
    effective_horizon = requested_planning_horizon
    canonical_exp_id = _canonical_exp_id(exp_id)
    if effective_horizon is None and exp_id not in PASSIVE_EXP_IDS:
        if canonical_exp_id in {"active_long", "active_chunk", "async_window", "async_chunk", "window_update"}:
            effective_horizon = 20
        elif canonical_exp_id == "active_short":
            effective_horizon = 1
        elif exp_id in APPROXIMATION_EXP_IDS:
            effective_horizon = 20
        else:
            effective_horizon = 5

    if exp_id in PASSIVE_EXP_IDS:
        planning_chunk = 1
    elif exp_id == "window_update":
        planning_chunk = max(1, int(requested_k_theta))
    elif exp_id in {"step_update", "async_window"}:
        planning_chunk = 1
    elif exp_id in {"active_chunk", "async_chunk"}:
        planning_chunk = 5
    else:
        planning_chunk = (
            1 if int(effective_horizon) <= 1 else (5 if int(effective_horizon) >= 10 else 3)
        )

    if effective_horizon is not None:
        planning_chunk = min(int(planning_chunk), int(effective_horizon))

    resolved_window_size = max(int(effective_k_theta), int(state_update_interval))
    return {
        "planning_horizon": int(effective_horizon) if effective_horizon is not None else None,
        "k_theta": int(effective_k_theta),
        "planning_chunk": int(planning_chunk),
        "state_update_interval": int(state_update_interval),
        "predictive_only_window": bool(predictive_only_window),
        "window_size": int(resolved_window_size),
        "update_scheme": exp_id,
        "metric_variant": exp_id if exp_id in APPROXIMATION_EXP_IDS else None,
    }


def _resolve_schedule_config(
    *,
    exp_id: str,
    requested_k_theta: int,
    requested_planning_horizon: int | None,
    use_explicit_schedule: bool = False,
) -> dict[str, Any]:
    if use_explicit_schedule:
        return _resolve_schedule_from_request(
            exp_id=exp_id,
            requested_k_theta=requested_k_theta,
            requested_planning_horizon=requested_planning_horizon,
        )

    canonical_exp_id = _canonical_exp_id(exp_id)
    if canonical_exp_id in PASSIVE_EXP_IDS:
        return {
            "planning_horizon": None,
            "k_theta": 1,
            "planning_chunk": 1,
            "state_update_interval": 1,
            "predictive_only_window": False,
            "window_size": 1,
            "update_scheme": canonical_exp_id,
            "metric_variant": None,
        }

    if canonical_exp_id == "active_long":
        config = {
            "planning_horizon": 20,
            "k_theta": 1,
            "planning_chunk": 1,
            "state_update_interval": 1,
            "predictive_only_window": False,
            "window_size": 1,
            "metric_variant": "exact",
        }
    elif canonical_exp_id == "active_short":
        config = {
            "planning_horizon": 1,
            "k_theta": 1,
            "planning_chunk": 1,
            "state_update_interval": 1,
            "predictive_only_window": False,
            "window_size": 1,
            "metric_variant": "exact",
        }
    elif canonical_exp_id == "active_chunk":
        config = {
            "planning_horizon": 20,
            "k_theta": 1,
            "planning_chunk": 5,
            "state_update_interval": 1,
            "predictive_only_window": False,
            "window_size": 1,
            "metric_variant": "exact",
        }
    elif canonical_exp_id == "async_window":
        config = {
            "planning_horizon": 20,
            "k_theta": 5,
            "planning_chunk": 1,
            "state_update_interval": 1,
            "predictive_only_window": False,
            "window_size": 5,
            "metric_variant": "exact",
        }
    elif canonical_exp_id == "async_chunk":
        config = {
            "planning_horizon": 20,
            "k_theta": 5,
            "planning_chunk": 5,
            "state_update_interval": 1,
            "predictive_only_window": False,
            "window_size": 5,
            "metric_variant": "exact",
        }
    elif canonical_exp_id == "window_update":
        config = {
            "planning_horizon": 20,
            "k_theta": 5,
            "planning_chunk": 5,
            "state_update_interval": 5,
            "predictive_only_window": True,
            "window_size": 5,
            "metric_variant": "exact",
        }
    elif exp_id in APPROXIMATION_EXP_IDS:
        config = {
            "planning_horizon": 20,
            "k_theta": 1,
            "planning_chunk": 1,
            "state_update_interval": 1,
            "predictive_only_window": False,
            "window_size": 1,
            "metric_variant": exp_id,
        }
    elif exp_id == "RND":
        config = {
            "planning_horizon": 20,
            "k_theta": 1,
            "planning_chunk": 1,
            "state_update_interval": 1,
            "predictive_only_window": False,
            "window_size": 1,
            "metric_variant": None,
        }
    else:
        config = _resolve_schedule_from_request(
            exp_id=exp_id,
            requested_k_theta=requested_k_theta,
            requested_planning_horizon=requested_planning_horizon,
        )

    if requested_planning_horizon is not None and config["planning_horizon"] is not None:
        config["planning_horizon"] = max(1, int(requested_planning_horizon))
        config["planning_chunk"] = min(int(config["planning_chunk"]), int(config["planning_horizon"]))

    config["update_scheme"] = exp_id
    return config


def _embedding_metric_variant(exp_id: str) -> dict[str, bool]:
    if exp_id == "frozen_cov":
        return {"freeze_covariance": True}
    if exp_id == "no_sensitivity":
        return {"no_sensitivity_propagation": True}
    if exp_id == "fully_observed":
        return {"fully_observed": True}
    return {}


def _parameter_update_applied(*, prev_block_steps: int, model: Any) -> bool:
    k_theta = max(1, int(getattr(model, "k_theta", 1)))
    next_block_steps = int(getattr(model, "_theta_block_steps", 0))
    return prev_block_steps + 1 >= k_theta and next_block_steps == 0


def _sample_true_embedding(seed: int) -> np.ndarray:
    """Sample true Duffing parameters deterministically from run seed only."""
    rng = np.random.default_rng(int(seed))
    low = np.asarray([-3.0, -2.0], dtype=np.float64)
    high = np.asarray([-0.1, 2.0], dtype=np.float64)
    return (low + (high - low) * rng.random(2)).astype(np.float32)


def _write_trace_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _to_xy_pair(value: Any) -> tuple[float, float]:
    """Convert arbitrary tensor-like values to an `(x, y)` pair."""
    import torch

    flat = torch.as_tensor(value).detach().reshape(-1)
    if flat.numel() == 0:
        return 0.0, 0.0
    if flat.numel() == 1:
        v = float(flat[0].item())
        return v, 0.0
    return float(flat[0].item()), float(flat[1].item())


def _as_bool(value: Any) -> bool:
    """Normalize bool-like values from tensors/scalars/strings."""
    import torch

    if isinstance(value, bool):
        return value
    if isinstance(value, torch.Tensor):
        flat = value.detach().reshape(-1)
        if flat.numel() == 0:
            return False
        return bool(flat[0].item())
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "t"}
    return bool(value)


def _current_plan_index(policy: Any) -> int:
    """Current action index within the most recent chunked MPC plan."""
    chunk = max(1, int(getattr(policy, "chunk", 1)))
    count = max(0, int(getattr(policy, "count", 0)))
    if count <= 0:
        return 0
    return (count - 1) % chunk


def _extract_remaining_plan_actions(policy: Any):
    """Return the remaining planned action sequence for the current step."""
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
        if tensor.shape[0] == 1:
            tensor = tensor.unsqueeze(1)
        else:
            tensor = tensor.unsqueeze(0)
    return tensor


def _predictive_only_embedding_step(model: Any, action: Any) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    if not hasattr(model, "z") or not hasattr(model, "Fz"):
        raise TypeError("Predictive-only window updates require a FilteringEmbedding-style model.")

    model._normalize_embedding_belief()
    model._ensure_state_belief_shapes(batch_size=model.e["m"].shape[0])

    Q = F.softplus(model.dynamics.logvar).diag_embed().unsqueeze(0) * model.dt
    I = torch.eye(model.latent_dim, device=model.device).unsqueeze(0).unsqueeze(0)

    action_bt = _ensure_batch_time_tensor(action, device=model.device)
    if action_bt is not None and model.action_encoder is not None:
        u_enc = model.action_encoder(action_bt, model.z["m"])
    else:
        u_enc = action_bt

    Fz = model.Fz(model.z["m"], model.e["m"])
    dfdz = Fz * model.dt + I
    pred_m = model.predict(action=u_enc)
    pred_cov = dfdz @ model.z["P"] @ dfdz.transpose(-1, -2) + Q + 1e-6 * I
    pred_cov = model._project_spd(pred_cov)

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
    *,
    model: Any,
    policy: Any,
    transition: dict[str, Any],
) -> np.ndarray | None:
    """Predict the remaining latent trajectory for MPC-based active policies."""
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
    if state.ndim != 3 or state.shape[-1] == 0:
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
            if model.action_encoder is not None:
                encoded_actions = model.action_encoder(planned_actions)
            else:
                encoded_actions = planned_actions
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


def _check_configs(repo_root: Path) -> None:
    missing = [rel for rel in REQUIRED_CONFIGS if not (repo_root / rel).exists()]
    if missing:
        missing_lines = "\n".join(f"- {rel}" for rel in missing)
        raise FileNotFoundError(f"Missing required config files:\n{missing_lines}")


def _check_imports() -> None:
    failed: list[str] = []
    for module_name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(module_name)
        except Exception:
            failed.append(module_name)
    if failed:
        failed_lines = "\n".join(f"- {name}" for name in failed)
        raise ImportError(f"Failed import sanity checks:\n{failed_lines}")


def run_preflight_checks() -> None:
    repo_root = _repo_root()
    _check_configs(repo_root=repo_root)
    _check_imports()


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


def _fe_true(z, e):
    import torch

    if z.ndim == 2:
        z = z.unsqueeze(0)
    B, T, _ = z.shape
    Fe = torch.zeros(B, T, 2, 2, device=z.device)
    Fe[..., 1, 0] = z[..., 1]
    Fe[..., 1, 1] = -z[..., 0]
    return Fe


def _fz_true(z, e):
    import torch

    if e.ndim == 2:
        e = e.unsqueeze(0)
    if z.ndim == 2:
        z = z.unsqueeze(0)
    B, T, _ = z.shape
    Fz = torch.zeros(B, T, 2, 2, device=z.device)
    Fz[..., 0, 1] = 1
    Fz[..., 1, 0] = -e[..., 1] - 0.3 * z[..., 0] ** 2
    Fz[..., 1, 1] = e[..., 0]
    return Fz


def _duffing_rollout_no_input(z0, e, horizon: int, dt: float, dynamics_alpha: float):
    import torch

    z = z0.clone()
    traj = [z]
    a = e[..., 0]
    b = e[..., 1]
    for _ in range(horizon):
        x = z[..., 0]
        v = z[..., 1]
        dx = dynamics_alpha * v
        dv = dynamics_alpha * (a * v - b * x - 0.1 * x**3)
        z = torch.stack((x + dt * dx, v + dt * dv), dim=-1)
        traj.append(z)
    return torch.stack(traj, dim=1)


def _trajectory_r2(
    e_est,
    e_true,
    *,
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
        traj_true = _duffing_rollout_no_input(
            starts, e_true_batch, horizon=horizon, dt=dt, dynamics_alpha=dynamics_alpha
        )
        traj_est = _duffing_rollout_no_input(
            starts, e_est_batch, horizon=horizon, dt=dt, dynamics_alpha=dynamics_alpha
        )

        y_true = traj_true.reshape(-1)
        y_est = traj_est.reshape(-1)
        sse = torch.sum((y_true - y_est) ** 2)
        sst = torch.sum((y_true - torch.mean(y_true)) ** 2)

    if float(sst.item()) <= 1e-12:
        return 0.0
    return float((1.0 - sse / sst).item())


def _set_vectorfield_params(env: Any, params: Any) -> None:
    """Set vector-field parameters across legacy/new VectorField APIs."""
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
    """Adapter exposing a robust set_params(*args) API for FunctionDynamics."""

    def __init__(self, dynamics_obj: Any):
        self.dynamics_obj = dynamics_obj

    def __call__(self, state):
        return self.dynamics_obj(state)

    def set_params(self, *params) -> None:
        if len(params) == 1:
            value = params[0]
            if hasattr(value, "detach"):
                flat_vals = value.detach().reshape(-1).tolist()
            elif isinstance(value, (list, tuple)):
                flat_vals = list(value)
            else:
                flat_vals = [value]
        else:
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
    ctx: dict[str, Any],
    status: str,
    end_time: str,
    runtime_sec: float,
    results_path: Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model_tag": ctx["model_tag"],
        "commit": ctx["commit"],
        "seed": ctx["seed"],
        "exp_id": ctx["exp_id"],
        "total_steps": ctx["total_steps"],
        "base_dir": ctx["base_dir"],
        "status": status,
        "start_time": ctx["start_time"],
        "end_time": end_time,
        "runtime_sec": runtime_sec,
        "results_path": str(results_path),
    }
    if extra:
        payload.update(extra)
    return payload


def _build_failure_metadata(
    *,
    model_tag: str,
    exp_id: str,
    seed: int,
    total_steps: int,
    run_dir: Path,
    error: Exception,
) -> dict[str, Any]:
    now = _utc_now()
    return {
        "model_tag": model_tag,
        "commit": _current_commit(),
        "seed": seed,
        "exp_id": exp_id,
        "total_steps": total_steps,
        "base_dir": str(run_dir),
        "status": "failed",
        "start_time": now,
        "end_time": now,
        "runtime_sec": 0.0,
        "results_path": str(run_dir),
        "error": f"{type(error).__name__}: {error}",
    }


def _run_single_parameter_identification(
    model_tag: str,
    exp_id: str,
    seed: int,
    total_steps: int,
    run_dir: Path,
    q_theta: float,
    k_theta: int,
    q_theta_meas_coeff: float,
    q_theta_max_scale: float,
    eig_gamma: float,
    state_noise: float,
    action_max: float,
    dynamics_alpha: float,
    state_init_uncertainty: float,
    firing_rate_scale: float,
    dry_run: bool,
    planning_horizon: int | None,
    traj_eval_interval: int,
    traj_eval_horizon: int,
    traj_eval_samples: int,
    save_acq_map: bool = False,
    acq_map_interval: int = 5,
    acq_map_grid: int = 61,
    acq_map_lim: float = 10.0,
    extra_metadata: dict[str, Any] | None = None,
    use_explicit_schedule: bool = False,
) -> dict[str, Any]:
    import torch
    import torch.nn as nn

    import actdyn
    import actdyn.core.experiment
    import actdyn.environment
    import actdyn.environment.action
    import actdyn.environment.observation
    import actdyn.metrics
    import actdyn.metrics.information
    import actdyn.metrics.uncertainty
    import actdyn.models
    import actdyn.models.dynamics
    import actdyn.policy
    import actdyn.policy.mpc
    from actdyn.config import ExperimentConfig
    from actdyn.utils.runtime import configure_runtime
    from actdyn.utils.visualize import set_matplotlib_style

    schedule_config = _resolve_schedule_config(
        exp_id=exp_id,
        requested_k_theta=k_theta,
        requested_planning_horizon=planning_horizon,
        use_explicit_schedule=use_explicit_schedule,
    )
    effective_k_theta = int(schedule_config["k_theta"])
    state_update_interval = int(schedule_config["state_update_interval"])
    predictive_only_window = bool(schedule_config["predictive_only_window"])
    effective_horizon = schedule_config["planning_horizon"]
    planning_chunk = int(schedule_config["planning_chunk"])
    resolved_window_size = int(schedule_config["window_size"])
    update_scheme = str(schedule_config["update_scheme"])
    metric_variant = schedule_config["metric_variant"]

    ctx = {
        "model_tag": model_tag,
        "commit": _current_commit(),
        "seed": seed,
        "exp_id": exp_id,
        "total_steps": total_steps,
        "base_dir": str(run_dir),
        "start_time": _utc_now(),
    }
    _validate_exp_id(exp_id)
    if dry_run:
        payload = _build_metadata(
            ctx=ctx,
            status="dry_run",
            end_time=ctx["start_time"],
            runtime_sec=0.0,
            results_path=run_dir,
            extra={
                "q_theta": float(q_theta),
                "k_theta": int(effective_k_theta),
                "requested_k_theta": int(k_theta),
                "window_size": int(resolved_window_size),
                "state_update_interval": int(state_update_interval),
                "parameter_update_interval": int(effective_k_theta),
                "predictive_only_window": bool(predictive_only_window),
                "q_theta_meas_coeff": float(q_theta_meas_coeff),
                "q_theta_max_scale": float(q_theta_max_scale),
                "eig_gamma": float(eig_gamma),
                "state_noise": float(state_noise),
                "action_max": float(action_max),
                "dynamics_alpha": float(dynamics_alpha),
                "state_init_uncertainty": float(state_init_uncertainty),
                "firing_rate_scale": float(firing_rate_scale),
                "mean_firing_rate_target": float(50.0 * firing_rate_scale),
                "max_firing_rate_target": float(100.0 * firing_rate_scale),
                "planning_horizon": int(effective_horizon) if effective_horizon is not None else None,
                "planning_chunk": int(planning_chunk),
                "metric_variant": metric_variant,
                "update_scheme": update_scheme,
                "traj_eval_interval": int(traj_eval_interval),
                "traj_eval_horizon": int(traj_eval_horizon),
                "traj_eval_samples": int(traj_eval_samples),
                "save_acq_map": bool(save_acq_map),
                "acq_map_interval": int(acq_map_interval),
                "acq_map_grid": int(acq_map_grid),
                "acq_map_lim": float(acq_map_lim),
                "parameter_error_trace_path": str(run_dir / "parameter_error_trace.csv"),
                "trajectory_r2_trace_path": str(run_dir / "trajectory_r2_trace.csv"),
                "embedding_estimate_trace_path": str(run_dir / "embedding_estimate_trace.csv"),
                "information_trace_path": str(run_dir / "information_trace.csv"),
                "state_action_trace_path": str(run_dir / "state_action_trace.csv"),
                "acquisition_map_trace_path": str(run_dir / "acquisition_map_trace.npz"),
                "writing_ref": WRITING_REFERENCE,
            },
        )
        if extra_metadata:
            payload.update(extra_metadata)
        return payload

    started = datetime.now(timezone.utc)
    set_matplotlib_style()
    device = configure_runtime(seed=seed)
    torch.manual_seed(seed)

    e_true = torch.as_tensor(_sample_true_embedding(seed), dtype=torch.float32, device=device).unsqueeze(0)
    a, b = e_true.reshape(-1)

    dz, de, du, dy = 2, 2, 2, 50
    dt = 0.01
    alpha = float(dynamics_alpha)
    noise_scale = max(1e-8, float(state_noise))
    action_max = float(max(1e-6, action_max))
    firing_rate_scale = max(1e-6, float(firing_rate_scale))

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
        noise_scale=0.1,
        noise_type="poisson",
        dt=dt,
        device=device,
    )
    C = obs_model.network[0].weight.detach()
    C[:, 0] = torch.abs(C[:, 0])
    C[:, 1] = C[:, 1] * 2
    mean_firing = 50.0 * firing_rate_scale
    max_firing_rate = 100.0 * firing_rate_scale
    state_range_for_cap = 5.0

    mean_log_rate = torch.log(torch.full((dy,), mean_firing, device=device))
    max_log_rate = torch.log(torch.full((dy,), max_firing_rate, device=device))
    for _ in range(6):
        c_row_l1 = torch.sum(torch.abs(C), dim=1)
        c_row_l2_sq = torch.sum(C * C, dim=1)
        bias_from_mean = mean_log_rate - 0.5 * c_row_l2_sq
        capped_log_rate = state_range_for_cap * c_row_l1 + bias_from_mean
        if torch.all(capped_log_rate <= max_log_rate):
            break
        safe_den = torch.clamp(state_range_for_cap * c_row_l1, min=1e-8)
        row_scale = torch.clamp((max_log_rate - bias_from_mean) / safe_den, min=0.0, max=1.0)
        C = C * row_scale.unsqueeze(1)
    bias = mean_log_rate - 0.5 * torch.sum(C * C, dim=1)

    obs_model.network[0].bias = nn.Parameter(bias)
    obs_model.network[0].weight = nn.Parameter(C)

    duffing_env = actdyn.VectorFieldEnv(
        "duffing",
        x_range=5,
        dyn_params=None,
        dt=dt,
        alpha=alpha,
        Q=noise_scale,
        action_bounds=[action_model.action_space.low, action_model.action_space.high],
        state_bounds=[-5.0, 5.0],
        initial_state=[2.5, 2.5],
        device=device,
    )
    _set_vectorfield_params(duffing_env, torch.tensor([a, b, 0.1], device=device))
    env = actdyn.environment.EnvWrapper(duffing_env, obs_model, action_model, dt=dt, device=device)

    mapping = actdyn.models.decoder.LogLinearMapping(
        latent_dim=dz, obs_dim=dy, dt=dt, device=device
    )
    noise = actdyn.models.decoder.PoissonNoise(obs_dim=dy, sigma=0.01, device=device)
    decoder = actdyn.models.Decoder(mapping=mapping, noise=noise, device=device)

    sim_vec_env = actdyn.VectorFieldEnv(
        "duffing",
        x_range=5,
        dyn_params=None,
        dt=dt,
        alpha=alpha,
        Q=noise_scale,
        device=device,
    )
    _set_vectorfield_params(sim_vec_env, torch.tensor([0.0, 0.0, 0.1], device=device))
    dynamics_fn = _VectorFieldDynamicsAdapter(sim_vec_env.dynamics)
    dynamics = actdyn.models.dynamics.FunctionDynamics(
        state_dim=dz,
        dt=env.dt,
        dynamics_fn=dynamics_fn,
        device=device,
    )
    dynamics.logvar = nn.Parameter(torch.log(torch.ones(1, dz, device=device) * noise_scale))

    sigma_0 = 1e-2
    e_bel = {
        "m": torch.ones(1, de, device=device),
        "P": sigma_0 * torch.eye(de, device=device).unsqueeze(0),
        "L": (1 / sigma_0) * torch.eye(de, device=device).unsqueeze(0),
    }

    model_kwargs: dict[str, Any] = {
        "dynamics": dynamics,
        "decoder": decoder,
        "e": e_bel,
        "action_encoder": action_model,
        "Fe": _fe_true,
        "Fz": _fz_true,
        "device": device,
    }
    fe_init = inspect.signature(actdyn.models.FilteringEmbedding.__init__)
    if "q_theta" in fe_init.parameters:
        model_kwargs["q_theta"] = q_theta
    if "k_theta" in fe_init.parameters:
        model_kwargs["k_theta"] = effective_k_theta
    if "q_theta_meas_coeff" in fe_init.parameters:
        model_kwargs["q_theta_meas_coeff"] = q_theta_meas_coeff
    if "q_theta_max_scale" in fe_init.parameters:
        model_kwargs["q_theta_max_scale"] = q_theta_max_scale
    if "state_init_uncertainty" in fe_init.parameters:
        model_kwargs["state_init_uncertainty"] = state_init_uncertainty
    model = actdyn.models.FilteringEmbedding(**model_kwargs)
    model.set_params(e_bel["m"])

    emb_metric = actdyn.metrics.information.EmbeddingFisherMetric(
        model=model,
        Fe_net=_fe_true,
        Fz_net=_fz_true,
        gamma=eig_gamma,
        **_embedding_metric_variant(exp_id),
        device=device,
    )
    rnd_metric = actdyn.metrics.uncertainty.RandomNetworkDistillation(device=device)

    if exp_id == "random":
        policy = actdyn.policy.RandomPolicy(action_space=env.action_space, device=device)
        metric = None
    elif exp_id == "no_policy":
        policy = actdyn.policy.OffPolicy(action_space=env.action_space, device=device)
        metric = None
    else:
        if exp_id == "RND":
            metric = rnd_metric
        else:
            metric = actdyn.metrics.CompositeMetric(
                metrics=[emb_metric],
                compute_type="sum",
                weights=[1.0],
                device=device,
            )
        policy = actdyn.policy.mpc.MpcICem(
            metric=metric,
            model=model,
            device=device,
            horizon=int(effective_horizon),
            num_iterations=10,
            num_samples=40,
            num_elite=10,
            chunk=int(planning_chunk),
            verbose=False,
        )

    class _CadencedCissAgent(actdyn.Agent):
        def __init__(
            self,
            *,
            env: Any,
            model: Any,
            policy: Any,
            buffer_length: int,
            state_update_interval: int,
            predictive_only_window: bool,
            device: str,
        ) -> None:
            super().__init__(
                env=env,
                model=model,
                policy=policy,
                buffer_length=buffer_length,
                device=device,
            )
            self.state_update_interval = max(1, int(state_update_interval))
            self.predictive_only_window = bool(predictive_only_window)
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
                            y=buffered["next_obs"],
                            u=buffered["action"],
                        )
                        parameter_posterior_updated = (
                            parameter_posterior_updated
                            or _parameter_update_applied(
                                prev_block_steps=prev_block_steps,
                                model=self.model,
                            )
                        )
                    model_info["latent_state"] = self.model.get_state()
                    state_posterior_updated = True
                    self._window_buffer = []
                    self._window_start_snapshot = _clone_filter_belief_state(self.model)
            else:
                prev_block_steps = int(getattr(self.model, "_theta_block_steps", 0))
                model_info = self.model.update(self.recent)
                state_posterior_updated = True
                parameter_posterior_updated = _parameter_update_applied(
                    prev_block_steps=prev_block_steps,
                    model=self.model,
                )

            model_transition = {
                "model_action": model_info["env_action"],
                "next_model_state": model_info["latent_state"],
            }
            self.recent.add(**model_transition)
            transition = {
                **env_transition,
                **model_transition,
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

    exp_config = ExperimentConfig.from_yaml(str(_repo_root() / "experiments/ciss/conf/config.yaml"))
    exp_config.results_dir = str(run_dir)
    exp_config.training.total_steps = total_steps
    exp_config.training.train_every = total_steps + 1
    exp_config.run_analysis = False

    agent = _CadencedCissAgent(
        env=env,
        model=model,
        buffer_length=10,
        policy=policy,
        state_update_interval=state_update_interval,
        predictive_only_window=predictive_only_window,
        device=device,
    )
    experiment = actdyn.core.experiment.Experiment(agent=agent, config=exp_config)

    decoder.set_params(obs_model)

    trace_rows: list[dict[str, Any]] = []
    embedding_rows: list[dict[str, Any]] = []
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
    acq_grid_n = max(25, int(acq_map_grid))
    acq_grid_lim = float(acq_map_lim)
    acq_interval = max(1, int(acq_map_interval))
    acq_axis = np.linspace(-acq_grid_lim, acq_grid_lim, acq_grid_n, dtype=np.float32)
    acq_X, acq_V = np.meshgrid(acq_axis, acq_axis, indexing="xy")
    acq_points = torch.as_tensor(
        np.stack([acq_X.reshape(-1), acq_V.reshape(-1)], axis=1),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(1)

    def _on_step_end(transition: dict[str, Any]) -> None:
        step = int(experiment.env_step)
        cpu_time_sec = float(time.perf_counter() - perf_start)
        e_est = model.e["m"].detach().reshape(-1)
        param_err = float(torch.linalg.norm(e_est - e_true_flat).item())
        e_cov = model.e.get("P")
        cov_diag0 = None
        cov_diag1 = None
        cov_diag_mean = None
        if e_cov is not None:
            e_cov = e_cov.detach()
            if e_cov.dim() >= 3:
                cov_diag = torch.diagonal(e_cov, dim1=-2, dim2=-1).reshape(-1)
                if cov_diag.numel() > 0:
                    cov_diag0 = float(cov_diag[0].item())
                    cov_diag1 = float(cov_diag[1].item()) if cov_diag.numel() > 1 else None
                    cov_diag_mean = float(cov_diag.mean().item())
        trace_rows.append(
            {
                "step": step,
                "cpu_time_sec": cpu_time_sec,
                "parameter_error": param_err,
            }
        )
        embedding_rows.append(
            {
                "step": step,
                "cpu_time_sec": cpu_time_sec,
                "e0": float(e_est[0].item()) if e_est.numel() > 0 else None,
                "e1": float(e_est[1].item()) if e_est.numel() > 1 else None,
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
                "state_posterior_updated": _as_bool(transition.get("state_posterior_updated", True)),
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

        planned_action_x, planned_action_v = _to_xy_pair(
            transition.get("action", torch.zeros(2, device=device))
        )
        policy_action_x, policy_action_v = _to_xy_pair(
            transition.get("policy_action", transition.get("action", torch.zeros(2, device=device)))
        )
        env_action_x, env_action_v = _to_xy_pair(
            transition.get(
                "env_action", transition.get("policy_action", torch.zeros(2, device=device))
            )
        )
        action_norm = float(np.sqrt(planned_action_x**2 + planned_action_v**2))
        policy_action_norm = float(np.sqrt(policy_action_x**2 + policy_action_v**2))
        env_action_norm = float(np.sqrt(env_action_x**2 + env_action_v**2))
        policy_action_delta = float(
            np.sqrt(
                (policy_action_x - planned_action_x) ** 2
                + (policy_action_v - planned_action_v) ** 2
            )
        )
        execution_delta = float(
            np.sqrt((env_action_x - policy_action_x) ** 2 + (env_action_v - policy_action_v) ** 2)
        )
        action_total_delta = float(
            np.sqrt((env_action_x - planned_action_x) ** 2 + (env_action_v - planned_action_v) ** 2)
        )
        planned_sat = bool(
            max(abs(planned_action_x), abs(planned_action_v)) >= float(action_max) - 1e-6
        )
        policy_sat = bool(
            max(abs(policy_action_x), abs(policy_action_v)) >= float(action_max) - 1e-6
        )
        env_sat = bool(max(abs(env_action_x), abs(env_action_v)) >= float(action_max) - 1e-6)
        action_clipped = _as_bool(transition.get("action_clipped", False))
        env_action_clipped = _as_bool(transition.get("env_action_clipped", False))
        policy_cost = getattr(policy, "cost", None)
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
                "action_x": planned_action_x,
                "action_v": planned_action_v,
                "action_norm": action_norm,
                "policy_action_x": policy_action_x,
                "policy_action_v": policy_action_v,
                "policy_action_norm": policy_action_norm,
                "env_action_x": env_action_x,
                "env_action_v": env_action_v,
                "env_action_norm": env_action_norm,
                "policy_action_delta_norm": policy_action_delta,
                "execution_delta_norm": execution_delta,
                "action_total_delta_norm": action_total_delta,
                "action_clipped": action_clipped,
                "env_action_clipped": env_action_clipped,
                "planned_at_bound": planned_sat,
                "policy_at_bound": policy_sat,
                "env_action_at_bound": env_sat,
                "policy_cost": float(policy_cost) if policy_cost is not None else None,
                "state_posterior_updated": _as_bool(transition.get("state_posterior_updated", True)),
                "parameter_posterior_updated": _as_bool(
                    transition.get("parameter_posterior_updated", True)
                ),
                "window_buffer_length": int(transition.get("window_buffer_length", 0)),
            }
        )

        planned_traj_xy = _predict_planned_xy_trajectory(
            model=model,
            policy=policy,
            transition=transition,
        )
        if planned_traj_xy is not None and planned_traj_xy.shape[0] >= 2:
            planned_traj_steps.append(step)
            planned_traj_frames.append(planned_traj_xy)

        if save_acq_map and metric is not None and step % acq_interval == 0:
            map_rollout = {
                "model_state": acq_points,
                "next_model_state": acq_points,
            }
            acq_cost = metric(map_rollout).detach().reshape(-1)
            acq_map = (-acq_cost).cpu().numpy().reshape(acq_grid_n, acq_grid_n)
            acq_map = np.nan_to_num(acq_map, nan=0.0, posinf=1e6, neginf=0.0).astype(np.float32)
            acq_map_frames.append(acq_map)
            acq_map_steps.append(step)

        if traj_eval_interval > 0 and step % traj_eval_interval == 0:
            r2 = _trajectory_r2(
                e_est=e_est,
                e_true=e_true_flat,
                dt=dt,
                dynamics_alpha=alpha,
                horizon=traj_eval_horizon,
                n_starts=traj_eval_samples,
                rng=trace_rng,
                device=device,
            )
            traj_rows.append(
                {
                    "step": step,
                    "cpu_time_sec": cpu_time_sec,
                    "trajectory_r2": r2,
                    "traj_eval_horizon": int(traj_eval_horizon),
                    "traj_eval_samples": int(traj_eval_samples),
                }
            )

    experiment._run_online_loop(
        train_cfg=exp_config.training,
        pbar_desc="Online",
        plot_fcn=None,
        reset=True,
        on_step_end=_on_step_end,
    )

    result_dir = Path(experiment.results_path)
    ended = datetime.now(timezone.utc)

    param_trace_path = run_dir / "parameter_error_trace.csv"
    traj_trace_path = run_dir / "trajectory_r2_trace.csv"
    emb_trace_path = run_dir / "embedding_estimate_trace.csv"
    info_trace_path = run_dir / "information_trace.csv"
    state_action_trace_path = run_dir / "state_action_trace.csv"
    acq_map_trace_path = run_dir / "acquisition_map_trace.npz"
    planned_traj_trace_path = run_dir / "planned_trajectory_trace.npz"
    _write_trace_csv(
        param_trace_path,
        trace_rows,
        fields=["step", "cpu_time_sec", "parameter_error"],
    )
    _write_trace_csv(
        emb_trace_path,
        embedding_rows,
        fields=["step", "cpu_time_sec", "e0", "e1", "cov_diag0", "cov_diag1", "cov_diag_mean"],
    )
    _write_trace_csv(
        traj_trace_path,
        traj_rows,
        fields=["step", "cpu_time_sec", "trajectory_r2", "traj_eval_horizon", "traj_eval_samples"],
    )
    _write_trace_csv(
        info_trace_path,
        info_rows,
        fields=[
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
        fields=[
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
    if save_acq_map and acq_map_frames:
        np.savez_compressed(
            acq_map_trace_path,
            steps=np.asarray(acq_map_steps, dtype=np.int64),
            axis=acq_axis.astype(np.float32),
            maps=np.asarray(acq_map_frames, dtype=np.float32),
            exp_id=np.asarray([exp_id], dtype=object),
            model_tag=np.asarray([model_tag], dtype=object),
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
            planned_traj_trace_path,
            steps=np.asarray(planned_traj_steps, dtype=np.int64),
            paths=paths,
            lengths=lengths,
            exp_id=np.asarray([exp_id], dtype=object),
            model_tag=np.asarray([model_tag], dtype=object),
        )

    rollout_metrics = _extract_rollout_metrics(result_dir)
    if trace_rows:
        param_series = np.asarray([row["parameter_error"] for row in trace_rows], dtype=np.float64)
        embedding_error_final = float(param_series[-1])
        embedding_error_mean = float(param_series.mean())
    else:
        embedding_error_final = float(torch.norm(model.e["m"].reshape(-1) - e_true_flat).item())
        embedding_error_mean = None

    if state_action_rows:
        policy_delta = np.asarray(
            [float(row["policy_action_delta_norm"]) for row in state_action_rows], dtype=np.float64
        )
        exec_delta = np.asarray(
            [float(row["execution_delta_norm"]) for row in state_action_rows], dtype=np.float64
        )
        total_delta = np.asarray(
            [float(row["action_total_delta_norm"]) for row in state_action_rows], dtype=np.float64
        )
        action_clipped_count = int(
            sum(1 for row in state_action_rows if _as_bool(row["action_clipped"]))
        )
        env_action_clipped_count = int(
            sum(1 for row in state_action_rows if _as_bool(row["env_action_clipped"]))
        )
    else:
        policy_delta = np.asarray([], dtype=np.float64)
        exec_delta = np.asarray([], dtype=np.float64)
        total_delta = np.asarray([], dtype=np.float64)
        action_clipped_count = 0
        env_action_clipped_count = 0

    payload = _build_metadata(
        ctx=ctx,
        status="completed",
        end_time=ended.isoformat().replace("+00:00", "Z"),
        runtime_sec=(ended - started).total_seconds(),
        results_path=result_dir,
        extra={
            **rollout_metrics,
            "embedding_true": [float(x) for x in e_true_flat.tolist()],
            "embedding_estimate": [float(x) for x in model.e["m"].detach().reshape(-1).tolist()],
            "embedding_error_mean": embedding_error_mean,
            "embedding_error_final": embedding_error_final,
            "q_theta": float(q_theta),
            "k_theta": int(effective_k_theta),
            "requested_k_theta": int(k_theta),
            "window_size": int(resolved_window_size),
            "state_update_interval": int(state_update_interval),
            "parameter_update_interval": int(effective_k_theta),
            "predictive_only_window": bool(predictive_only_window),
            "q_theta_meas_coeff": float(q_theta_meas_coeff),
            "q_theta_max_scale": float(q_theta_max_scale),
            "eig_gamma": float(eig_gamma),
            "state_noise": float(noise_scale),
            "action_max": float(action_max),
            "dynamics_alpha": float(alpha),
            "state_init_uncertainty": float(state_init_uncertainty),
            "firing_rate_scale": float(firing_rate_scale),
            "mean_firing_rate_target": float(mean_firing),
            "max_firing_rate_target": float(max_firing_rate),
            "planning_horizon": int(effective_horizon) if effective_horizon is not None else None,
            "planning_chunk": int(planning_chunk),
            "metric_variant": metric_variant,
            "update_scheme": update_scheme,
            "traj_eval_interval": int(traj_eval_interval),
            "traj_eval_horizon": int(traj_eval_horizon),
            "traj_eval_samples": int(traj_eval_samples),
            "save_acq_map": bool(save_acq_map),
            "acq_map_interval": int(acq_interval),
            "acq_map_grid": int(acq_grid_n),
            "acq_map_lim": float(acq_grid_lim),
            "parameter_error_trace_path": str(param_trace_path),
            "trajectory_r2_trace_path": str(traj_trace_path),
            "embedding_estimate_trace_path": str(emb_trace_path),
            "information_trace_path": str(info_trace_path),
            "state_action_trace_path": str(state_action_trace_path),
            "planned_trajectory_trace_path": (
                str(planned_traj_trace_path) if planned_traj_frames else None
            ),
            "policy_action_delta_mean": (
                float(policy_delta.mean()) if policy_delta.size > 0 else None
            ),
            "execution_delta_mean": float(exec_delta.mean()) if exec_delta.size > 0 else None,
            "action_total_delta_mean": float(total_delta.mean()) if total_delta.size > 0 else None,
            "action_clipped_count": int(action_clipped_count),
            "env_action_clipped_count": int(env_action_clipped_count),
            "acquisition_map_trace_path": (
                str(acq_map_trace_path) if save_acq_map and acq_map_frames else None
            ),
            "writing_ref": WRITING_REFERENCE,
        },
    )
    if extra_metadata:
        payload.update(extra_metadata)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Cosyne parameter-identification workflows")
    parser.add_argument(
        "--mode",
        choices=["preflight", "smoke", "tracks", "ablation", "all"],
        default="tracks",
        help="Execution scope",
    )
    parser.add_argument(
        "--exp-ids",
        type=str,
        default=",".join(DEFAULT_EXP_IDS),
        help="Comma-separated experiment IDs. Default comparison set is active_short,random,no_policy.",
    )
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--total-steps", type=int, default=500, help="Track run steps")
    parser.add_argument("--smoke-steps", type=int, default=500, help="Smoke run steps")
    parser.add_argument("--model-tag", type=str, default="updated")
    parser.add_argument("--base-dir", type=str, default="results/cosyne")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=1, help="Parallel workers for track mode")
    parser.add_argument("--q-theta", type=float, default=5e-4)
    parser.add_argument(
        "--k-theta",
        type=int,
        default=10,
        help="Requested parameter-update interval for explicit-schedule runs; fixed policy presets ignore this unless overridden internally.",
    )
    parser.add_argument("--q-theta-meas-coeff", type=float, default=0.0)
    parser.add_argument("--q-theta-max-scale", type=float, default=10.0)
    parser.add_argument("--eig-gamma", type=float, default=1.0)
    parser.add_argument("--state-noise", type=float, default=0.2)
    parser.add_argument("--action-max", type=float, default=2.0)
    parser.add_argument("--dynamics-alpha", type=float, default=1.0)
    parser.add_argument("--state-init-uncertainty", type=float, default=25.0)
    parser.add_argument(
        "--firing-rate-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to the default mean/max observation firing rates (50 Hz / 100 Hz).",
    )
    parser.add_argument(
        "--planning-horizon",
        type=int,
        default=None,
        help="Optional explicit planning horizon override for the selected experiment preset.",
    )
    parser.add_argument("--traj-eval-interval", type=int, default=100)
    parser.add_argument("--traj-eval-horizon", type=int, default=100)
    parser.add_argument("--traj-eval-samples", type=int, default=16)
    parser.add_argument("--save-acq-map", action="store_true")
    parser.add_argument("--acq-map-interval", type=int, default=5)
    parser.add_argument("--acq-map-grid", type=int, default=61)
    parser.add_argument("--acq-map-lim", type=float, default=10.0)

    parser.add_argument("--ablation-exp-id", type=str, default="active_short")
    parser.add_argument("--ablation-total-steps", type=int, default=500)
    parser.add_argument("--ablation-planning-windows", type=str, default="3,5,10,15")
    parser.add_argument("--ablation-k-thetas", type=str, default="1,5,10,20")
    parser.add_argument("--ablation-fixed-k-theta", type=int, default=10)
    parser.add_argument("--ablation-fixed-planning-window", type=int, default=5)

    parser.add_argument("--dry-run", action="store_true")
    return parser


def _run_smoke(
    model_tag: str,
    seeds: list[int],
    smoke_steps: int,
    base_dir: Path,
    q_theta: float,
    k_theta: int,
    q_theta_meas_coeff: float,
    q_theta_max_scale: float,
    eig_gamma: float,
    state_noise: float,
    action_max: float,
    dynamics_alpha: float,
    state_init_uncertainty: float,
    firing_rate_scale: float,
    dry_run: bool,
    planning_horizon: int | None,
    traj_eval_interval: int,
    traj_eval_horizon: int,
    traj_eval_samples: int,
    save_acq_map: bool,
    acq_map_interval: int,
    acq_map_grid: int,
    acq_map_lim: float,
) -> None:
    seed = seeds[0] if seeds else 0
    exp_id = "active_short"
    run_dir = _ensure_dir(base_dir / "smoke" / model_tag / exp_id / f"seed_{seed}" / "repeat_01")
    metadata = _run_single_parameter_identification(
        model_tag=model_tag,
        exp_id=exp_id,
        seed=seed,
        total_steps=smoke_steps,
        run_dir=run_dir,
        q_theta=q_theta,
        k_theta=k_theta,
        q_theta_meas_coeff=q_theta_meas_coeff,
        q_theta_max_scale=q_theta_max_scale,
        eig_gamma=eig_gamma,
        state_noise=state_noise,
        action_max=action_max,
        dynamics_alpha=dynamics_alpha,
        state_init_uncertainty=state_init_uncertainty,
        firing_rate_scale=firing_rate_scale,
        dry_run=dry_run,
        planning_horizon=planning_horizon,
        traj_eval_interval=traj_eval_interval,
        traj_eval_horizon=traj_eval_horizon,
        traj_eval_samples=traj_eval_samples,
        save_acq_map=save_acq_map,
        acq_map_interval=acq_map_interval,
        acq_map_grid=acq_map_grid,
        acq_map_lim=acq_map_lim,
    )
    _write_json(run_dir / "run_metadata.json", metadata)


def _run_track_matrix(
    model_tag: str,
    exp_ids: list[str],
    seeds: list[int],
    repeats: int,
    total_steps: int,
    jobs: int,
    base_dir: Path,
    q_theta: float,
    k_theta: int,
    q_theta_meas_coeff: float,
    q_theta_max_scale: float,
    eig_gamma: float,
    state_noise: float,
    action_max: float,
    dynamics_alpha: float,
    state_init_uncertainty: float,
    firing_rate_scale: float,
    dry_run: bool,
    planning_horizon: int | None,
    traj_eval_interval: int,
    traj_eval_horizon: int,
    traj_eval_samples: int,
    save_acq_map: bool,
    acq_map_interval: int,
    acq_map_grid: int,
    acq_map_lim: float,
) -> None:
    tasks: list[dict[str, Any]] = []
    for exp_id in exp_ids:
        for seed in seeds:
            for repeat in range(1, repeats + 1):
                run_dir = _ensure_dir(
                    base_dir
                    / "tracks"
                    / model_tag
                    / exp_id
                    / f"seed_{seed}"
                    / f"repeat_{repeat:02d}"
                )
                tasks.append(
                    {
                        "model_tag": model_tag,
                        "exp_id": exp_id,
                        "seed": seed,
                        "total_steps": total_steps,
                        "run_dir": str(run_dir),
                        "q_theta": q_theta,
                        "k_theta": k_theta,
                        "q_theta_meas_coeff": q_theta_meas_coeff,
                        "q_theta_max_scale": q_theta_max_scale,
                        "eig_gamma": eig_gamma,
                        "state_noise": state_noise,
                        "action_max": action_max,
                        "dynamics_alpha": dynamics_alpha,
                        "state_init_uncertainty": state_init_uncertainty,
                        "firing_rate_scale": firing_rate_scale,
                        "dry_run": dry_run,
                        "planning_horizon": planning_horizon,
                        "traj_eval_interval": traj_eval_interval,
                        "traj_eval_horizon": traj_eval_horizon,
                        "traj_eval_samples": traj_eval_samples,
                        "save_acq_map": save_acq_map,
                        "acq_map_interval": acq_map_interval,
                        "acq_map_grid": acq_map_grid,
                        "acq_map_lim": acq_map_lim,
                    }
                )

    if jobs <= 1:
        for task in tasks:
            run_dir = Path(task["run_dir"])
            metadata = _run_track_task(task)
            _write_json(run_dir / "run_metadata.json", metadata)
        return

    executor: ProcessPoolExecutor | ThreadPoolExecutor
    try:
        executor = ProcessPoolExecutor(max_workers=max(1, jobs))
    except Exception:
        executor = ThreadPoolExecutor(max_workers=max(1, jobs))

    with executor:
        future_to_task = {executor.submit(_run_track_task, task): task for task in tasks}
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            run_dir = Path(task["run_dir"])
            try:
                metadata = future.result()
            except Exception as exc:
                metadata = _build_failure_metadata(
                    model_tag=str(task["model_tag"]),
                    exp_id=str(task["exp_id"]),
                    seed=int(task["seed"]),
                    total_steps=int(task["total_steps"]),
                    run_dir=run_dir,
                    error=exc,
                )
            _write_json(run_dir / "run_metadata.json", metadata)


def _run_track_task(task: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(task["run_dir"])
    try:
        return _run_single_parameter_identification(
            model_tag=str(task["model_tag"]),
            exp_id=str(task["exp_id"]),
            seed=int(task["seed"]),
            total_steps=int(task["total_steps"]),
            run_dir=run_dir,
            q_theta=float(task["q_theta"]),
            k_theta=int(task["k_theta"]),
            q_theta_meas_coeff=float(task["q_theta_meas_coeff"]),
            q_theta_max_scale=float(task["q_theta_max_scale"]),
            eig_gamma=float(task["eig_gamma"]),
            state_noise=float(task["state_noise"]),
            action_max=float(task["action_max"]),
            dynamics_alpha=float(task["dynamics_alpha"]),
            state_init_uncertainty=float(task["state_init_uncertainty"]),
            firing_rate_scale=float(task["firing_rate_scale"]),
            dry_run=bool(task["dry_run"]),
            planning_horizon=(
                int(task["planning_horizon"]) if task["planning_horizon"] is not None else None
            ),
            traj_eval_interval=int(task["traj_eval_interval"]),
            traj_eval_horizon=int(task["traj_eval_horizon"]),
            traj_eval_samples=int(task["traj_eval_samples"]),
            save_acq_map=bool(task["save_acq_map"]),
            acq_map_interval=int(task["acq_map_interval"]),
            acq_map_grid=int(task["acq_map_grid"]),
            acq_map_lim=float(task["acq_map_lim"]),
        )
    except Exception as exc:
        return _build_failure_metadata(
            model_tag=str(task["model_tag"]),
            exp_id=str(task["exp_id"]),
            seed=int(task["seed"]),
            total_steps=int(task["total_steps"]),
            run_dir=run_dir,
            error=exc,
        )


def _run_ablation_suite(
    model_tag: str,
    seeds: list[int],
    repeats: int,
    total_steps: int,
    base_dir: Path,
    q_theta: float,
    q_theta_meas_coeff: float,
    q_theta_max_scale: float,
    eig_gamma: float,
    state_noise: float,
    action_max: float,
    dynamics_alpha: float,
    state_init_uncertainty: float,
    firing_rate_scale: float,
    dry_run: bool,
    exp_id: str,
    planning_windows: list[int],
    k_thetas: list[int],
    fixed_k_theta: int,
    fixed_planning_window: int,
    traj_eval_interval: int,
    traj_eval_horizon: int,
    traj_eval_samples: int,
    save_acq_map: bool,
    acq_map_interval: int,
    acq_map_grid: int,
    acq_map_lim: float,
) -> None:
    for planning_horizon in planning_windows:
        for seed in seeds:
            for repeat in range(1, repeats + 1):
                run_dir = _ensure_dir(
                    base_dir
                    / "ablation"
                    / "planning_window"
                    / model_tag
                    / f"horizon_{planning_horizon}"
                    / exp_id
                    / f"seed_{seed}"
                    / f"repeat_{repeat:02d}"
                )
                try:
                    metadata = _run_single_parameter_identification(
                        model_tag=model_tag,
                        exp_id=exp_id,
                        seed=seed,
                        total_steps=total_steps,
                        run_dir=run_dir,
                        q_theta=q_theta,
                        k_theta=fixed_k_theta,
                        q_theta_meas_coeff=q_theta_meas_coeff,
                        q_theta_max_scale=q_theta_max_scale,
                        eig_gamma=eig_gamma,
                        state_noise=state_noise,
                        action_max=action_max,
                        dynamics_alpha=dynamics_alpha,
                        state_init_uncertainty=state_init_uncertainty,
                        firing_rate_scale=firing_rate_scale,
                        dry_run=dry_run,
                        planning_horizon=planning_horizon,
                        traj_eval_interval=traj_eval_interval,
                        traj_eval_horizon=traj_eval_horizon,
                        traj_eval_samples=traj_eval_samples,
                        save_acq_map=save_acq_map,
                        acq_map_interval=acq_map_interval,
                        acq_map_grid=acq_map_grid,
                        acq_map_lim=acq_map_lim,
                        use_explicit_schedule=True,
                        extra_metadata={
                            "ablation_axis": "planning_window",
                            "ablation_value": int(planning_horizon),
                            "ablation_fixed_k_theta": int(fixed_k_theta),
                        },
                    )
                except Exception as exc:
                    metadata = _build_failure_metadata(
                        model_tag=model_tag,
                        exp_id=exp_id,
                        seed=seed,
                        total_steps=total_steps,
                        run_dir=run_dir,
                        error=exc,
                    )
                    metadata.update(
                        {
                            "ablation_axis": "planning_window",
                            "ablation_value": int(planning_horizon),
                            "ablation_fixed_k_theta": int(fixed_k_theta),
                        }
                    )
                _write_json(run_dir / "run_metadata.json", metadata)

    for k_theta in k_thetas:
        for seed in seeds:
            for repeat in range(1, repeats + 1):
                run_dir = _ensure_dir(
                    base_dir
                    / "ablation"
                    / "update_frequency"
                    / model_tag
                    / f"k_theta_{k_theta}"
                    / exp_id
                    / f"seed_{seed}"
                    / f"repeat_{repeat:02d}"
                )
                try:
                    metadata = _run_single_parameter_identification(
                        model_tag=model_tag,
                        exp_id=exp_id,
                        seed=seed,
                        total_steps=total_steps,
                        run_dir=run_dir,
                        q_theta=q_theta,
                        k_theta=k_theta,
                        q_theta_meas_coeff=q_theta_meas_coeff,
                        q_theta_max_scale=q_theta_max_scale,
                        eig_gamma=eig_gamma,
                        state_noise=state_noise,
                        action_max=action_max,
                        dynamics_alpha=dynamics_alpha,
                        state_init_uncertainty=state_init_uncertainty,
                        firing_rate_scale=firing_rate_scale,
                        dry_run=dry_run,
                        planning_horizon=fixed_planning_window,
                        traj_eval_interval=traj_eval_interval,
                        traj_eval_horizon=traj_eval_horizon,
                        traj_eval_samples=traj_eval_samples,
                        save_acq_map=save_acq_map,
                        acq_map_interval=acq_map_interval,
                        acq_map_grid=acq_map_grid,
                        acq_map_lim=acq_map_lim,
                        use_explicit_schedule=True,
                        extra_metadata={
                            "ablation_axis": "update_frequency",
                            "ablation_value": int(k_theta),
                            "ablation_fixed_planning_window": int(fixed_planning_window),
                        },
                    )
                except Exception as exc:
                    metadata = _build_failure_metadata(
                        model_tag=model_tag,
                        exp_id=exp_id,
                        seed=seed,
                        total_steps=total_steps,
                        run_dir=run_dir,
                        error=exc,
                    )
                    metadata.update(
                        {
                            "ablation_axis": "update_frequency",
                            "ablation_value": int(k_theta),
                            "ablation_fixed_planning_window": int(fixed_planning_window),
                        }
                    )
                _write_json(run_dir / "run_metadata.json", metadata)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    exp_ids = _parse_csv_list(args.exp_ids)
    seeds = _parse_csv_ints(args.seeds)
    if not exp_ids:
        exp_ids = list(DEFAULT_EXP_IDS)
    if not seeds:
        seeds = list(DEFAULT_SEEDS)

    base_dir = Path(args.base_dir)
    if not base_dir.is_absolute():
        base_dir = _repo_root() / base_dir
    base_dir = base_dir.resolve()

    run_preflight_checks()
    if args.mode == "preflight":
        print("Preflight checks passed.")
        return 0

    if args.mode in {"smoke", "all"}:
        _run_smoke(
            model_tag=args.model_tag,
            seeds=seeds,
            smoke_steps=args.smoke_steps,
            base_dir=base_dir,
            q_theta=args.q_theta,
            k_theta=args.k_theta,
            q_theta_meas_coeff=args.q_theta_meas_coeff,
            q_theta_max_scale=args.q_theta_max_scale,
            eig_gamma=args.eig_gamma,
            state_noise=args.state_noise,
            action_max=args.action_max,
            dynamics_alpha=args.dynamics_alpha,
            state_init_uncertainty=args.state_init_uncertainty,
            firing_rate_scale=args.firing_rate_scale,
            dry_run=args.dry_run,
            planning_horizon=args.planning_horizon,
            traj_eval_interval=args.traj_eval_interval,
            traj_eval_horizon=args.traj_eval_horizon,
            traj_eval_samples=args.traj_eval_samples,
            save_acq_map=args.save_acq_map,
            acq_map_interval=args.acq_map_interval,
            acq_map_grid=args.acq_map_grid,
            acq_map_lim=args.acq_map_lim,
        )

    if args.mode in {"tracks", "all"}:
        _run_track_matrix(
            model_tag=args.model_tag,
            exp_ids=exp_ids,
            seeds=seeds,
            repeats=args.repeats,
            total_steps=args.total_steps,
            jobs=max(1, args.jobs),
            base_dir=base_dir,
            q_theta=args.q_theta,
            k_theta=args.k_theta,
            q_theta_meas_coeff=args.q_theta_meas_coeff,
            q_theta_max_scale=args.q_theta_max_scale,
            eig_gamma=args.eig_gamma,
            state_noise=args.state_noise,
            action_max=args.action_max,
            dynamics_alpha=args.dynamics_alpha,
            state_init_uncertainty=args.state_init_uncertainty,
            firing_rate_scale=args.firing_rate_scale,
            dry_run=args.dry_run,
            planning_horizon=args.planning_horizon,
            traj_eval_interval=args.traj_eval_interval,
            traj_eval_horizon=args.traj_eval_horizon,
            traj_eval_samples=args.traj_eval_samples,
            save_acq_map=args.save_acq_map,
            acq_map_interval=args.acq_map_interval,
            acq_map_grid=args.acq_map_grid,
            acq_map_lim=args.acq_map_lim,
        )

    if args.mode == "ablation":
        planning_windows = _parse_csv_ints(args.ablation_planning_windows)
        k_thetas = _parse_csv_ints(args.ablation_k_thetas)
        _run_ablation_suite(
            model_tag=args.model_tag,
            seeds=seeds,
            repeats=args.repeats,
            total_steps=args.ablation_total_steps,
            base_dir=base_dir,
            q_theta=args.q_theta,
            q_theta_meas_coeff=args.q_theta_meas_coeff,
            q_theta_max_scale=args.q_theta_max_scale,
            eig_gamma=args.eig_gamma,
            state_noise=args.state_noise,
            action_max=args.action_max,
            dynamics_alpha=args.dynamics_alpha,
            state_init_uncertainty=args.state_init_uncertainty,
            firing_rate_scale=args.firing_rate_scale,
            dry_run=args.dry_run,
            exp_id=args.ablation_exp_id,
            planning_windows=planning_windows,
            k_thetas=k_thetas,
            fixed_k_theta=args.ablation_fixed_k_theta,
            fixed_planning_window=args.ablation_fixed_planning_window,
            traj_eval_interval=args.traj_eval_interval,
            traj_eval_horizon=args.traj_eval_horizon,
            traj_eval_samples=args.traj_eval_samples,
            save_acq_map=args.save_acq_map,
            acq_map_interval=args.acq_map_interval,
            acq_map_grid=args.acq_map_grid,
            acq_map_lim=args.acq_map_lim,
        )

    print(f"Finished mode={args.mode} model_tag={args.model_tag} base_dir={base_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
