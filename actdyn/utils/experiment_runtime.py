from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


def write_trace_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_trace_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def seed_range_csv(count: int) -> str:
    """Return the canonical comma-separated seed range `0,...,count-1`."""
    return ",".join(str(seed) for seed in range(int(count)))


def to_xy_pair(value: Any) -> tuple[float, float]:
    import torch

    flat = torch.as_tensor(value).detach().reshape(-1)
    if flat.numel() == 0:
        return 0.0, 0.0
    if flat.numel() == 1:
        return float(flat[0].item()), 0.0
    return float(flat[0].item()), float(flat[1].item())



def as_bool(value: Any) -> bool:
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



def current_plan_index(policy: Any) -> int:
    chunk = max(1, int(getattr(policy, "chunk", 1)))
    count = max(0, int(getattr(policy, "count", 0)))
    if count <= 0:
        return 0
    return (count - 1) % chunk



def extract_remaining_plan_actions(policy: Any):
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
    start = min(current_plan_index(policy), int(plan.shape[0] - 1))
    return plan[start:].unsqueeze(0)



def apply_loglinear_loading_asymmetry(weight: Any, env_preset: Any):
    import torch

    c = weight.detach().clone()
    if not bool(getattr(env_preset, "asymmetric_loading", False)):
        return c
    primary_scale = float(getattr(env_preset, "observation_primary_scale", 1.0))
    secondary_scale = float(getattr(env_preset, "observation_secondary_scale", 2.0))
    row_skew = float(getattr(env_preset, "observation_row_skew", 0.0))
    if c.shape[1] >= 1:
        c[:, 0] = torch.abs(c[:, 0]) * primary_scale
    if c.shape[1] >= 2:
        c[:, 1] = c[:, 1] * secondary_scale
        if abs(row_skew) > 1e-8 and c.shape[0] > 1:
            row_axis = torch.linspace(-1.0, 1.0, steps=c.shape[0], device=c.device)
            primary_gain = 1.0 + row_skew * torch.clamp(row_axis, min=0.0)
            secondary_gain = 1.0 + row_skew * torch.clamp(-row_axis, min=0.0)
            c[:, 0] = c[:, 0] * primary_gain
            c[:, 1] = c[:, 1] * secondary_gain
    return c



def apply_loglinear_loading_mismatch(weight: Any, *, variance: float, seed: int):
    """Add iid Gaussian mismatch to each log-linear loading entry.

    Args:
        weight: Loading matrix with shape (observation_dim, latent_dim).
        variance: Nonnegative perturbation variance.
        seed: PRNG seed for the perturbation.

    Returns:
        Perturbed loading matrix with the same shape and dtype as ``weight``.

    The variance is the observation-model mismatch stress:
    eps_ij ~ N(0, variance), C_model = C_true + eps.
    """
    import torch

    mismatch_variance = float(variance)
    if mismatch_variance < 0.0:
        raise ValueError(f"Loading mismatch variance must be nonnegative, got {variance}.")
    c = weight.detach().clone()
    if mismatch_variance == 0.0:
        return c
    generator = torch.Generator(device=c.device)
    generator.manual_seed(int(seed))
    eps = torch.randn(c.shape, dtype=c.dtype, device=c.device, generator=generator)
    return c + eps * float(np.sqrt(mismatch_variance))



def predict_planned_xy_trajectory(
    *, model: Any, policy: Any, transition: dict[str, Any]
) -> np.ndarray | None:
    import torch

    if getattr(policy, "metric", None) is None:
        return None
    planned_actions = extract_remaining_plan_actions(policy)
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



def extract_rollout_metrics(results_path: Path) -> dict[str, Any]:
    from actdyn.utils.persistence import load_and_concatenate_rollouts
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
        rollout = load_and_concatenate_rollouts(str(rollouts_dir))
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
