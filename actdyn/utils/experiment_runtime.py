from __future__ import annotations

import csv
from functools import lru_cache
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



def _ensure_neurofisher_path() -> None:
    import sys

    external_root = Path(__file__).resolve().parents[2] / "external" / "neurofisherSNR"
    if not (external_root / "neurofisherSNR" / "snr.py").exists():
        raise ImportError(
            "neurofisherSNR submodule is missing. Run "
            "`git submodule update --init external/neurofisherSNR`."
        )
    root_str = str(external_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)



def calibrate_loglinear_loading(
    weight: Any,
    env_preset: Any,
    *,
    mean_firing_rate: float | None = None,
    max_firing_rate: float | None = None,
    target_snr: float | None = None,
    snr_seed: int = 0,
    snr_num_trajectories: int = 100,
    snr_trajectory_length: int = 500,
    state_range_for_cap: float = 5.0,
):
    """Return the runtime log-linear loading matrix and bias.

    Args:
        weight: Initial loading matrix with shape ``(observation_dim, latent_dim)``.
        env_preset: Environment preset carrying firing-rate and asymmetry settings.
        mean_firing_rate: Optional override for the target rate in Hz.
        max_firing_rate: Optional override for the capped rate in Hz.
        target_snr: Optional target per-bin Fisher SNR in dB. When provided,
            the loading matrix is rescaled with ``neurofisherSNR.optimize_C``.
        snr_seed: Seed for target-SNR trajectory initial states.
        snr_num_trajectories: Number of trajectories used for SNR calibration.
        snr_trajectory_length: Number of states per SNR calibration trajectory.
        state_range_for_cap: Symmetric state range used to enforce the max-rate cap.

    Returns:
        ``(C, b)`` where ``C`` has shape ``(observation_dim, latent_dim)`` and
        ``b`` has shape ``(observation_dim,)`` for ``lambda(z)=exp(C z + b)``.

    The bias sets the expected rate under unit-variance Gaussian latents to
    ``mean_firing_rate_target`` before the observation model multiplies by ``dt``.
    """
    import torch

    c = apply_loglinear_loading_asymmetry(weight, env_preset)
    mean_firing = (
        float(getattr(env_preset, "mean_firing_rate_target"))
        if mean_firing_rate is None
        else float(mean_firing_rate)
    )
    capped_rate = (
        float(getattr(env_preset, "max_firing_rate_target"))
        if max_firing_rate is None
        else float(max_firing_rate)
    )
    if mean_firing <= 0.0:
        raise ValueError(f"mean_firing_rate_target must be positive, got {mean_firing}.")
    if capped_rate <= 0.0:
        raise ValueError(f"max_firing_rate_target must be positive, got {capped_rate}.")

    if target_snr is not None:
        dt = float(getattr(env_preset, "dt"))
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}.")
        latents = _normalized_zero_action_trajectories(
            env_preset,
            seed=int(snr_seed),
            num_trajectories=int(snr_num_trajectories),
            trajectory_length=int(snr_trajectory_length),
        )
        c_np = c.detach().cpu().numpy().astype(np.float64, copy=True)
        b0 = np.zeros((1, c_np.shape[0]), dtype=np.float64)
        _ensure_neurofisher_path()
        from neurofisherSNR.optimize import optimize_C

        c_opt, b_per_bin, _snr = optimize_C(
            x=latents,
            C=c_np,
            b=b0,
            tgt_rate_per_bin=mean_firing * dt,
            max_rate_per_bin=capped_rate * dt,
            tgt_snr=float(target_snr),
            priority="max",
            min_gain=0.01,
        )
        c_t = torch.as_tensor(c_opt, dtype=c.dtype, device=c.device)
        b_t = torch.as_tensor(b_per_bin.reshape(-1), dtype=c.dtype, device=c.device)
        return c_t, b_t - float(np.log(dt))

    mean_log_rate = torch.log(
        torch.full((c.shape[0],), mean_firing, dtype=c.dtype, device=c.device)
    )
    max_log_rate = torch.log(
        torch.full((c.shape[0],), capped_rate, dtype=c.dtype, device=c.device)
    )
    for _ in range(6):
        c_row_l1 = torch.sum(torch.abs(c), dim=1)
        c_row_l2_sq = torch.sum(c * c, dim=1)
        bias_from_mean = mean_log_rate - 0.5 * c_row_l2_sq
        capped_log_rate = float(state_range_for_cap) * c_row_l1 + bias_from_mean
        if torch.all(capped_log_rate <= max_log_rate):
            break
        safe_den = torch.clamp(float(state_range_for_cap) * c_row_l1, min=1e-8)
        row_scale = torch.clamp((max_log_rate - bias_from_mean) / safe_den, min=0.0, max=1.0)
        c = c * row_scale.unsqueeze(1)
    bias = mean_log_rate - 0.5 * torch.sum(c * c, dim=1)
    return c, bias



@lru_cache(maxsize=1)
def _neurofisher_snr_bound():
    _ensure_neurofisher_path()
    from neurofisherSNR.snr import SNR_bound_instantaneous

    return SNR_bound_instantaneous



def _normalized_zero_action_trajectories(
    env_preset: Any,
    *,
    seed: int,
    num_trajectories: int,
    trajectory_length: int,
) -> np.ndarray:
    """Sample normalized latent trajectories from unforced true dynamics.

    Args:
        env_preset: Environment preset defining dynamics, true parameters, bounds, and dt.
        seed: Seed for reproducible random initial states.
        num_trajectories: Number of independently sampled initial states.
        trajectory_length: Number of latent states retained per trajectory, including
            the initial state.

    Returns:
        Flattened latent trajectory array with shape
        ``(num_trajectories * trajectory_length, latent_dim)``. Columns are
        centered and scaled to unit variance for the NeuroFisher SNR convention.
    """
    from actdyn.environment.vectorfield import step_np

    n_traj = int(num_trajectories)
    horizon = int(trajectory_length)
    if n_traj <= 0:
        raise ValueError(f"num_trajectories must be positive, got {num_trajectories}.")
    if horizon <= 1:
        raise ValueError(f"trajectory_length must be greater than 1, got {trajectory_length}.")

    latent_dim = int(getattr(env_preset, "latent_dim"))
    starts = np.stack(
        [env_preset.sample_initial_state(int(seed) + idx) for idx in range(n_traj)],
        axis=0,
    ).astype(np.float64, copy=False)
    if starts.shape != (n_traj, latent_dim):
        raise ValueError(
            f"Initial states must have shape {(n_traj, latent_dim)}, got {starts.shape}."
        )

    trajectories = np.empty((n_traj, horizon, latent_dim), dtype=np.float64)
    trajectories[:, 0, :] = starts
    current = starts
    action = np.zeros_like(current)
    for step in range(1, horizon):
        current = step_np(
            str(env_preset.resolved_dynamics_type()),
            current,
            action,
            dyn_params=np.asarray(env_preset.resolved_true_params(), dtype=np.float64),
            dt=float(env_preset.dt),
            dynamics_alpha=float(env_preset.dynamics_alpha),
            clip_limit=float(env_preset.resolved_plot_limit()),
        )
        trajectories[:, step, :] = current

    latents = trajectories.reshape(n_traj * horizon, latent_dim)
    latents = latents - np.mean(latents, axis=0, keepdims=True)
    latent_std = np.std(latents, axis=0, keepdims=True)
    return latents / np.maximum(latent_std, 1e-12)



def compute_loglinear_loading_fisher_snr_db(
    env_preset: Any,
    *,
    seed: int = 0,
    num_trajectories: int = 100,
    trajectory_length: int = 500,
) -> float:
    """Compute NeuroFisher SNR for the resolved log-linear Poisson loading.

    Args:
        env_preset: Environment preset defining ``latent_dim``, ``observation_dim``,
            ``dt``, firing-rate targets, and loading asymmetry.
        seed: PRNG seed for the runtime loading initialization and initial states.
        num_trajectories: Number of random initial states used for the SNR trajectory.
        trajectory_length: Number of latent states per trajectory, including each
            initial state.

    Returns:
        Instantaneous Fisher SNR bound in dB from
        ``neurofisherSNR.snr.SNR_bound_instantaneous``.

    The external SNR function expects ``lambda(z)=exp(C z + b)``. The runtime
    observation model uses mean counts ``dt * exp(C z + b)``, so this passes
    ``b + log(dt)`` to measure per-bin observation SNR.
    """
    import torch

    if str(getattr(env_preset, "observation_noise_type", "poisson")).lower() != "poisson":
        raise ValueError("Fisher SNR is defined here for Poisson log-linear observations.")
    dt = float(getattr(env_preset, "dt"))
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}.")

    from actdyn.environment.observation import LogLinearObservation

    latent_dim = int(getattr(env_preset, "latent_dim"))
    observation_dim = int(getattr(env_preset, "observation_dim"))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        obs_model = LogLinearObservation(
            d_obs=observation_dim,
            d_latent=latent_dim,
            obs_dim=observation_dim,
            latent_dim=latent_dim,
            noise_scale=float(getattr(env_preset, "observation_noise_scale")),
            noise_type=str(getattr(env_preset, "observation_noise_type")),
            dt=dt,
            device="cpu",
        )
        target_snr = getattr(env_preset, "loading_target_snr_db", None)
        c, bias = calibrate_loglinear_loading(
            obs_model.network[0].weight,
            env_preset,
            target_snr=None if target_snr is None else float(target_snr),
            snr_seed=int(seed),
            snr_num_trajectories=int(num_trajectories),
            snr_trajectory_length=int(trajectory_length),
        )

    latents = _normalized_zero_action_trajectories(
        env_preset,
        seed=int(seed),
        num_trajectories=int(num_trajectories),
        trajectory_length=int(trajectory_length),
    )

    c_np = c.detach().cpu().numpy().astype(np.float64, copy=False)
    bias_per_bin = bias.detach().cpu().numpy().reshape(1, -1).astype(
        np.float64, copy=False
    ) + np.log(dt)
    snr_bound = _neurofisher_snr_bound()
    return float(snr_bound(latents, c_np.T, bias_per_bin))



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
