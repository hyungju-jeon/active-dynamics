#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from experiment_specs import (
        get_environment_preset,
        get_experiment_spec,
        get_policy_spec,
        get_schedule_spec,
    )
    from actdyn.environment import step_np
    from actdyn.utils.runtime import utc_now
    from actdyn.utils.experiment_runtime import (
        apply_loglinear_loading_asymmetry,
        as_bool,
        extract_rollout_metrics,
        predict_planned_xy_trajectory,
        to_xy_pair,
        write_trace_csv,
    )
    from cosyne.rbf_filtering import (
        SparseRbfDynamics,
        SparseRbfFilteringModel,
        StructuredLocalRbfParameterMetric,
    )
else:
    from .experiment_specs import (
        get_environment_preset,
        get_experiment_spec,
        get_policy_spec,
        get_schedule_spec,
    )
    from actdyn.environment import step_np
    from actdyn.utils.runtime import utc_now
    from actdyn.utils.experiment_runtime import (
        apply_loglinear_loading_asymmetry,
        as_bool,
        extract_rollout_metrics,
        predict_planned_xy_trajectory,
        to_xy_pair,
        write_trace_csv,
    )
    from .cosyne.rbf_filtering import (
        SparseRbfDynamics,
        SparseRbfFilteringModel,
        StructuredLocalRbfParameterMetric,
    )

WRITING_REFERENCE = "docs/active-dynamics-writing/methods.tex"


def _clip_state_np(state: np.ndarray, limit: float) -> np.ndarray:
    return np.clip(np.asarray(state, dtype=np.float64), -float(limit), float(limit))


def _system_step_np(
    *,
    dynamics_type: str,
    state: np.ndarray,
    action: np.ndarray,
    embedding: np.ndarray,
    env_preset: Any,
    dt: float,
    dynamics_alpha: float,
    clip_limit: float,
) -> np.ndarray:
    return step_np(
        dynamics_type,
        state,
        action,
        dyn_params=env_preset.params_from_embedding(np.asarray(embedding, dtype=np.float64)),
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
    env_preset: Any,
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
        env_preset.resolved_dynamics_type(),
        eval_states,
        env_preset.params_from_embedding(np.asarray(true_embedding_vec, dtype=np.float64)),
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
    env_preset: Any,
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
            dynamics_type=env_preset.resolved_dynamics_type(),
            state=true_state,
            action=zero_action,
            embedding=np.asarray(true_embedding_vec, dtype=np.float64),
            env_preset=env_preset,
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

def run_single_rbf_identification(
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
    build_runtime_experiment_config,
    build_metadata,
    instantiate_synthetic_policy,
    resolve_parameter_mean,
    resolve_parameter_precision,
    resolve_parameter_covariance,
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

    start_time = utc_now()
    set_matplotlib_style()
    device = configure_runtime(seed=seed)

    init_state = env_preset.sample_initial_state(seed)
    de = int(env_preset.embedding_dim)
    e_true = torch.as_tensor(
        env_preset.true_embedding_vector(embedding_dim=de), dtype=torch.float32, device=device
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
    grid_axis, centers = _rbf_grid(env_preset.resolved_plot_limit(), grid_shape[0])
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
    c = apply_loglinear_loading_asymmetry(obs_model.network[0].weight, env_preset)
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
        env_preset.resolved_dynamics_type(),
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
    duffing_env.set_params(
        torch.as_tensor(
            env_preset.params_from_embedding(e_true.reshape(-1)),
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

    policy = instantiate_synthetic_policy(
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

    exp_config = build_runtime_experiment_config(
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
    agent = actdyn.Agent(
        env=env,
        model=model,
        buffer_length=10,
        policy=policy,
        device=device,
        state_update_interval=int(schedule_spec.update_interval),
        predictive_only_window=bool(schedule_spec.predictive_only_window),
    )
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
        -float(env_preset.resolved_plot_limit()), float(env_preset.resolved_plot_limit()), 25, dtype=np.float64
    )
    eval_x, eval_y = np.meshgrid(eval_axis, eval_axis, indexing="xy")
    eval_states = np.stack([eval_x.reshape(-1), eval_y.reshape(-1)], axis=1)
    last_weight_mean = [resolve_parameter_mean(model=model, policy=policy).detach().clone()]

    def _on_step_end(transition: dict[str, Any]) -> None:
        step = int(experiment.env_step)
        cpu_time_sec = float(time.perf_counter() - perf_start)
        weight_vec = resolve_parameter_mean(model=model, policy=policy).detach()
        weight_mat = (
            weight_vec.reshape(-1, dz).detach().cpu().numpy().astype(np.float64, copy=False)
        )
        precision = resolve_parameter_precision(model=model, policy=policy)
        precision_mat = (
            precision.detach().cpu().numpy().astype(np.float64, copy=False)
            if precision is not None
            else np.zeros((1, weight_vec.numel(), weight_vec.numel()), dtype=np.float64)
        )
        covariance = resolve_parameter_covariance(model=model, policy=policy)
        covariance_diag = (
            torch.diagonal(covariance, dim1=-2, dim2=-1)
            .detach()
            .reshape(-1, dz)
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
            if covariance is not None
            else np.zeros_like(weight_mat)
        )
        dynamics_mse = _rbf_dynamics_mse(
            env_preset=env_preset,
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
        model_x, model_v = to_xy_pair(transition.get("model_state", torch.zeros(2, device=device)))
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
                "state_posterior_updated": as_bool(transition.get("state_posterior_updated", True)),
                "parameter_posterior_updated": as_bool(
                    transition.get("parameter_posterior_updated", True)
                ),
                "window_buffer_length": int(transition.get("window_buffer_length", 0)),
            }
        )
        dynamics_rows.append(
            {"step": step, "cpu_time_sec": cpu_time_sec, "dynamics_mse": float(dynamics_mse)}
        )
        env_x, env_v = to_xy_pair(transition.get("env_state", torch.zeros(2, device=device)))
        next_model_x, next_model_v = to_xy_pair(
            transition.get("next_model_state", torch.zeros(2, device=device))
        )
        action_x, action_v = to_xy_pair(transition.get("action", torch.zeros(2, device=device)))
        policy_x, policy_v = to_xy_pair(
            transition.get("policy_action", transition.get("action", torch.zeros(2, device=device)))
        )
        env_action_x, env_action_v = to_xy_pair(
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
                "action_clipped": as_bool(transition.get("action_clipped", False)),
                "env_action_clipped": as_bool(transition.get("env_action_clipped", False)),
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
                "state_posterior_updated": as_bool(transition.get("state_posterior_updated", True)),
                "parameter_posterior_updated": as_bool(
                    transition.get("parameter_posterior_updated", True)
                ),
                "window_buffer_length": int(transition.get("window_buffer_length", 0)),
            }
        )
        planned = predict_planned_xy_trajectory(model=model, policy=policy, transition=transition)
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
                        env_preset=env_preset,
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
                        clip_limit=max(6.0, env_preset.resolved_plot_limit() * 1.6),
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

    write_trace_csv(dynamics_trace_path, dynamics_rows, ["step", "cpu_time_sec", "dynamics_mse"])
    write_trace_csv(
        traj_trace_path,
        traj_rows,
        ["step", "cpu_time_sec", "trajectory_r2", "traj_eval_horizon", "traj_eval_samples"],
    )
    write_trace_csv(
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
    write_trace_csv(
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

    rollout_metrics = extract_rollout_metrics(result_dir)
    payload = build_metadata(
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
            "env_preset_id": str(env_preset.preset_id),
            "system_id": str(env_preset.system_id),
            "system_label": str(getattr(env_preset, "system_label", None) or env_preset.system_id),
            "dynamics_type": str(env_preset.resolved_dynamics_type()),
            "true_params_full": [float(x) for x in env_preset.resolved_true_params()],
            "state_low": [float(x) for x in env_preset.resolved_state_bounds()[0].tolist()],
            "state_high": [float(x) for x in env_preset.resolved_state_bounds()[1].tolist()],
            "min_embedding_dim": int(env_preset.resolved_min_embedding_dim()),
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
            "coarse_dt_factor": int(getattr(policy_spec, "coarse_dt_factor", 1)),
            "coarse_action_mapping": str(getattr(policy_spec, "coarse_action_mapping", "hold")),
            "coarse_mapping_opt_steps": int(getattr(policy_spec, "coarse_mapping_opt_steps", 25)),
            "coarse_mapping_opt_lr": float(getattr(policy_spec, "coarse_mapping_opt_lr", 0.05)),
            "async_planning": bool(getattr(policy_spec, "async_planning", False)),
            "async_stale_tolerance": float(getattr(policy_spec, "async_stale_tolerance", 0.5)),
            "async_stale_refine_iterations": int(
                getattr(policy_spec, "async_stale_refine_iterations", 2)
            ),
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
