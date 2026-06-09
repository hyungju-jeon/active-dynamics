#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
from functools import lru_cache
import inspect
import json
import os
from pathlib import Path
import shlex
import sys
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from experiment_io import (
        load_json,
        parse_csv_ints,
        parse_csv_list,
        resolve_session_root,
        write_json,
    )
    from experiment_definitions import (
        configure_catalogs,
        describe_catalogs,
        get_environment_preset,
        get_experiment_spec,
        get_policy_spec,
        get_schedule_spec,
        list_experiment_ids,
    )
    from actdyn.environment import jacobian_state_torch, residual_np, residual_torch, step_np
    from actdyn.utils.runtime import current_commit, ensure_dir, repo_root, utc_now
    from actdyn.utils.validation import trajectory_r2_vectorfield
    from actdyn.utils.experiment_runtime import (
        calibrate_loglinear_loading,
        apply_loglinear_loading_mismatch,
        as_bool,
        compute_loglinear_loading_fisher_snr_db,
        extract_rollout_metrics,
        predict_planned_xy_trajectory,
        to_xy_pair,
        write_trace_csv,
    )
else:
    from .experiment_io import (
        load_json,
        parse_csv_ints,
        parse_csv_list,
        resolve_session_root,
        write_json,
    )
    from .experiment_definitions import (
        configure_catalogs,
        describe_catalogs,
        get_environment_preset,
        get_experiment_spec,
        get_policy_spec,
        get_schedule_spec,
        list_experiment_ids,
    )
    from actdyn.environment import jacobian_state_torch, residual_np, residual_torch, step_np
    from actdyn.utils.runtime import current_commit, ensure_dir, repo_root, utc_now
    from actdyn.utils.validation import trajectory_r2_vectorfield
    from actdyn.utils.experiment_runtime import (
        calibrate_loglinear_loading,
        apply_loglinear_loading_mismatch,
        as_bool,
        compute_loglinear_loading_fisher_snr_db,
        extract_rollout_metrics,
        predict_planned_xy_trajectory,
        to_xy_pair,
        write_trace_csv,
    )

WRITING_REFERENCE = "docs/active-dynamics-writing/methods.tex"


def _resolved_policy_type(policy_id: str, policy_spec: Any | None) -> str:
    configured = None if policy_spec is None else getattr(policy_spec, "policy_type", None)
    if isinstance(configured, str) and configured.strip():
        return configured.strip()
    if bool(getattr(policy_spec, "async_planning", False)):
        return "async-mpc-icem"
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


def _batched_jacobian(output: Any, wrt: Any):
    with torch.enable_grad():
        flat_output = output.reshape(-1, output.shape[-1])
        rows = []
        for out_idx in range(flat_output.shape[-1]):
            grads = torch.autograd.grad(
                flat_output[:, out_idx].sum(),
                wrt,
                retain_graph=out_idx < flat_output.shape[-1] - 1,
                create_graph=True,
                allow_unused=False,
            )[0]
            rows.append(grads)
        jac = torch.stack(rows, dim=-2)
    return jac.reshape(*wrt.shape[:-1], output.shape[-1], wrt.shape[-1])


def _embedding_jacobian(env_preset: Any, z: Any, e: Any, *, estimator: bool, dynamics_alpha: float):
    with torch.enable_grad():
        z_t = torch.as_tensor(z, dtype=torch.float32, device=z.device)
        e_t = torch.as_tensor(e, dtype=torch.float32, device=z_t.device).detach().clone()
        while e_t.ndim < z_t.ndim:
            e_t = e_t.unsqueeze(-2)
        e_t.requires_grad_(True)
        dyn_params = env_preset.params_from_embedding(e_t, estimator=estimator)
        drift = residual_torch(
            str(env_preset.resolved_dynamics_type(estimator=estimator)),
            z_t,
            dyn_params,
            dynamics_alpha=float(dynamics_alpha),
        )
        return _batched_jacobian(drift, e_t)


def _build_env_jacobians(env_preset: Any, *, estimator: bool, dynamics_alpha: float):
    dynamics_type = str(env_preset.resolved_dynamics_type(estimator=estimator))

    def _fe(z: Any, e: Any):
        return _embedding_jacobian(
            env_preset,
            z,
            e,
            estimator=estimator,
            dynamics_alpha=float(dynamics_alpha),
        )

    def _fz(z: Any, e: Any):
        dyn_params = env_preset.params_from_embedding(e, estimator=estimator)
        return jacobian_state_torch(
            dynamics_type,
            z,
            dyn_params,
            dynamics_alpha=float(dynamics_alpha),
        )

    return _fe, _fz


@lru_cache(maxsize=128)
def _computed_loading_fisher_snr_db(env_preset: Any) -> float | None:
    if bool(getattr(env_preset, "real_data", False)):
        return None
    if str(getattr(env_preset, "observation_noise_type", "poisson")).lower() != "poisson":
        return None
    return compute_loglinear_loading_fisher_snr_db(env_preset)


def _loading_target_snr_db(env_preset: Any) -> float | None:
    configured = getattr(env_preset, "loading_target_snr_db", None)
    if configured is None:
        return None
    return float(configured)


def _loading_fisher_snr_db(env_preset: Any) -> float | None:
    configured = getattr(env_preset, "loading_fisher_snr_db", None)
    if configured is not None:
        return float(configured)
    return _computed_loading_fisher_snr_db(env_preset)


def _environment_summary(env_preset: Any) -> dict[str, Any]:
    system_id = str(env_preset.system_id)
    estimator_system_id = _resolved_estimator_system_id(env_preset)
    if not bool(getattr(env_preset, "real_data", False)):
        embedding_dim = int(env_preset.embedding_dim)
        system_label = str(getattr(env_preset, "system_label", None) or system_id)
        estimator_label = str(
            getattr(env_preset, "estimator_system_id", None) or estimator_system_id
        )
        return {
            "system_id": system_id,
            "system_label": system_label,
            "dynamics_type": str(env_preset.resolved_dynamics_type()),
            "estimator_system_id": estimator_system_id,
            "estimator_system_label": estimator_label,
            "estimator_dynamics_type": str(env_preset.resolved_dynamics_type(estimator=True)),
            "true_embedding": [
                float(x)
                for x in env_preset.true_embedding_vector(embedding_dim=embedding_dim).tolist()
            ],
        }
    return {
        "system_id": system_id,
        "system_label": str(getattr(env_preset, "system_label", None) or system_id),
        "dynamics_type": (
            "replay_dataset" if bool(getattr(env_preset, "real_data", False)) else "unknown"
        ),
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
    exp_config.results_dir = str(run_dir.resolve())
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
    exp_config.environment.env_x_range = float(env_preset.resolved_plot_limit())
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


def _clip_state_np(state: np.ndarray, limit: float) -> np.ndarray:
    return np.clip(np.asarray(state, dtype=np.float64), -float(limit), float(limit))


def _system_step_np(
    *,
    dynamics_type: str,
    state: np.ndarray,
    action: np.ndarray,
    embedding: np.ndarray,
    full_params: np.ndarray,
    min_embedding_dim: int,
    dt: float,
    dynamics_alpha: float,
    clip_limit: float,
) -> np.ndarray:
    full = np.asarray(full_params, dtype=np.float32)
    emb = np.asarray(embedding, dtype=np.float32)
    if emb.shape[-1] < int(min_embedding_dim):
        raise ValueError(
            f"Embedding must have at least {min_embedding_dim} coordinates, got shape {emb.shape}."
        )
    dyn_params = (
        emb[..., : full.shape[0]]
        if emb.shape[-1] >= full.shape[0]
        else np.concatenate((emb, full[emb.shape[-1] :]), axis=-1)
    )
    return step_np(
        dynamics_type,
        state,
        action,
        dyn_params=dyn_params,
        dt=float(dt),
        dynamics_alpha=float(dynamics_alpha),
        clip_limit=float(clip_limit),
    )


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
        "commit": current_commit(),
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


def _policy_owns_parameter_estimate(policy: Any) -> bool:
    return bool(getattr(policy, "owns_parameter_estimate", False))


def _resolve_parameter_mean(*, model: Any, policy: Any):
    if _policy_owns_parameter_estimate(policy) and hasattr(policy, "get_parameter_mean"):
        return policy.get_parameter_mean()
    model_e = getattr(model, "e", None)
    if isinstance(model_e, dict) and "m" in model_e:
        return model_e["m"]
    raise AttributeError("Unable to resolve parameter mean from policy or model")


def _resolve_parameter_covariance(*, model: Any, policy: Any):
    if _policy_owns_parameter_estimate(policy) and hasattr(policy, "get_parameter_covariance"):
        return policy.get_parameter_covariance()
    model_e = getattr(model, "e", None)
    if isinstance(model_e, dict):
        return model_e.get("P")
    return None


def _resolve_parameter_precision(*, model: Any, policy: Any):
    if _policy_owns_parameter_estimate(policy) and hasattr(policy, "get_parameter_precision"):
        return policy.get_parameter_precision()
    model_e = getattr(model, "e", None)
    if not isinstance(model_e, dict):
        return None
    precision = model_e.get("L")
    if precision is not None:
        return precision
    covariance = model_e.get("P")
    if covariance is None:
        return None
    import torch

    eye = torch.eye(covariance.shape[-1], dtype=covariance.dtype, device=covariance.device)
    while eye.ndim < covariance.ndim:
        eye = eye.unsqueeze(0)
    return torch.linalg.pinv(covariance + 1e-8 * eye)


def _make_shrinkage_map(
    *,
    shrinkage_kind: str | None,
    shrinkage_min: float | None,
) -> Any | None:
    if shrinkage_kind is None:
        return None
    if shrinkage_kind != "inverse_quadratic":
        raise ValueError(f"Unsupported shrinkage_kind={shrinkage_kind!r}")

    tau_min = 0.0 if shrinkage_min is None else float(shrinkage_min)

    def _map(delta):
        import torch

        delta_tensor = torch.as_tensor(delta)
        tau = torch.reciprocal(1.0 + delta_tensor)
        tau = torch.nan_to_num(tau, nan=1.0, posinf=1.0, neginf=tau_min)
        return tau.clamp(min=tau_min, max=1.0)

    return _map


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
    ambiguity_temperature: float | None = None,
    ensemble_kind: str | None = None,
):
    from actdyn.metrics.objectives import (
        ambiguity_aware_parameter_eig,
        dynamics as build_dynamics_metric,
        e_optimality as build_e_optimality_metric,
        fully_observable_parameter_eig,
        parameter_eig,
        corrected_sampling_variance as build_corrected_sampling_variance_metric,
        sampling_variance as build_sampling_variance_metric,
        state_variance as build_state_variance_metric,
        shrinkage_parameter_eig,
        state_information as build_state_information_metric,
    )

    if objective_kind == "parameter_eig":
        return parameter_eig(model=model, Fe_net=Fe_net, Fz_net=Fz_net, gamma=gamma, device=device)
    if objective_kind == "shrinkage_parameter_eig":
        return shrinkage_parameter_eig(
            model=model,
            Fe_net=Fe_net,
            Fz_net=Fz_net,
            gamma=gamma,
            device=device,
        )
    if objective_kind == "ambiguity_aware_parameter_eig":
        return ambiguity_aware_parameter_eig(
            model=model,
            Fe_net=Fe_net,
            Fz_net=Fz_net,
            gamma=gamma,
            device=device,
            ambiguity_temperature=(
                1.0 if ambiguity_temperature is None else float(ambiguity_temperature)
            ),
            ensemble_kind=ensemble_kind,
        )
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
    if objective_kind == "corrected_sampling_variance":
        return build_corrected_sampling_variance_metric(
            model=model,
            Fe_net=Fe_net,
            Fz_net=Fz_net,
            gamma=gamma,
            device=device,
            num_parameter_samples=int(sampling_variance_samples),
            sample_seed=sampling_variance_seed,
            correction_df=3.0,
            ess_gate_fraction=0.05,
        )
    if objective_kind == "state_variance":
        return build_state_variance_metric(
            model=model,
            Fe_net=Fe_net,
            Fz_net=Fz_net,
            gamma=gamma,
            device=device,
            num_parameter_samples=int(sampling_variance_samples),
            sample_seed=sampling_variance_seed,
        )
    raise ValueError(f"Unsupported objective_kind={objective_kind}")


def _boundary_env_kwargs(env_preset: Any) -> dict[str, Any]:
    return {
        "boundary_enabled": bool(getattr(env_preset, "boundary_enabled", False)),
        "boundary_type": str(getattr(env_preset, "boundary_type", "none")),
        "boundary_radius": getattr(env_preset, "boundary_radius", None),
        "boundary_barrier_enabled": bool(
            getattr(env_preset, "boundary_barrier_enabled", False)
        ),
        "boundary_projection_enabled": bool(
            getattr(env_preset, "boundary_projection_enabled", False)
        ),
        "boundary_barrier_width": float(getattr(env_preset, "boundary_barrier_width", 0.5)),
        "boundary_barrier_strength": float(
            getattr(env_preset, "boundary_barrier_strength", 5.0)
        ),
        "boundary_barrier_temperature": float(
            getattr(env_preset, "boundary_barrier_temperature", 0.1)
        ),
    }


def _apply_boundary_visibility_to_metric(metric: Any, env_preset: Any) -> None:
    if metric is None:
        return
    metric.boundary_visibility_enabled = bool(
        getattr(env_preset, "information_boundary_visibility_enabled", False)
    )
    metric.boundary_type = str(getattr(env_preset, "boundary_type", "none"))
    metric.boundary_radius = getattr(env_preset, "boundary_radius", None)
    metric.boundary_margin = float(getattr(env_preset, "information_boundary_margin", 1.0))
    metric.boundary_temperature = float(
        getattr(env_preset, "information_boundary_temperature", 0.15)
    )


def _boundary_visibility_mean(states: Any, env_preset: Any) -> float | None:
    if not bool(getattr(env_preset, "information_boundary_visibility_enabled", False)):
        return None
    try:
        from actdyn.environment.boundary import boundary_visibility

        z = torch.as_tensor(states, dtype=torch.float32)
        visibility = boundary_visibility(
            z,
            boundary_type=str(getattr(env_preset, "boundary_type", "none")),
            radius=getattr(env_preset, "boundary_radius", None),
            margin=float(getattr(env_preset, "information_boundary_margin", 1.0)),
            temperature=float(getattr(env_preset, "information_boundary_temperature", 0.15)),
        )
        return float(visibility.mean().item())
    except Exception:
        return None


def _instantiate_synthetic_policy(
    *,
    actdyn_module: Any,
    env: Any,
    env_preset: Any,
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
            hold_steps=max(1, int(getattr(schedule_spec, "replan_interval", 1))),
            amplitude=1.0,
        )
    if policy_type == "flex":
        initial_parameter_mean = None
        if getattr(model, "e", None) is not None and "m" in model.e:
            initial_parameter_mean = model.e["m"].detach().clone()
        use_observed_state = str(policy_id).endswith("_true_state")
        return actdyn_module.policy.FLEXPolicy(
            action_space=env.action_space,
            model=model,
            env_preset=env_preset,
            initial_parameter_mean=initial_parameter_mean,
            use_observed_state=use_observed_state,
            regularization=(
                1e-2
                if policy_spec.flex_regularization is None
                else float(policy_spec.flex_regularization)
            ),
            parameter_step_clip=policy_spec.flex_parameter_step_clip,
            parameter_min=policy_spec.flex_parameter_min,
            parameter_max=policy_spec.flex_parameter_max,
            lr=policy_spec.flex_lr,
            device=device,
        )
    if policy_type == "rhc":
        return actdyn_module.policy.RecedingHorizonCuriosityPolicy(
            action_space=env.action_space,
            device=device,
            horizon=int(schedule_spec.planning_horizon),
            objective="rhc_mvr" if str(policy_id).endswith("_mvr") else "rhc_us",
            num_features=128,
            prior_precision=1e-8,
            beta=1.0,
            bandwidth_init=1.0,
            optimize_hyperparams=False,
            planner_maxiter=500,
            warm_start=False,
            seed=int(seed),
        )
    if policy_type == "off-policy":
        return actdyn_module.policy.OffPolicy(action_space=env.action_space, device=device)
    if policy_type not in {"mpc-icem", "async-mpc-icem"}:
        raise ValueError(f"Unsupported policy_type={policy_type!r} for synthetic experiments")
    mpc_cls = (
        actdyn_module.policy.mpc.AsyncMpcICem
        if policy_type == "async-mpc-icem" or bool(getattr(policy_spec, "async_planning", False))
        else actdyn_module.policy.mpc.MpcICem
    )
    kwargs = {}
    if mpc_cls is actdyn_module.policy.mpc.AsyncMpcICem:
        kwargs.update(
            async_stale_tolerance=getattr(policy_spec, "async_stale_tolerance", 0.5),
            async_stale_refine_iterations=getattr(
                policy_spec, "async_stale_refine_iterations", 2
            ),
            async_worker_backend=getattr(policy_spec, "async_worker_backend", "thread"),
            async_start_after_first_plan=getattr(
                policy_spec, "async_start_after_first_plan", True
            ),
        )
    return mpc_cls(
        metric=metric,
        model=model,
        device=device,
        horizon=int(schedule_spec.planning_horizon),
        num_iterations=int(mpc_num_iterations),
        num_samples=int(mpc_num_samples),
        num_elite=int(mpc_num_elite),
        chunk=int(schedule_spec.replan_interval),
        action_constraint=getattr(policy_spec, "action_constraint", None) or "box",
        action_radius=getattr(policy_spec, "action_radius", None),
        coarse_dt_factor=getattr(policy_spec, "coarse_dt_factor", 1),
        coarse_action_mapping=getattr(policy_spec, "coarse_action_mapping", "hold"),
        coarse_mapping_opt_steps=getattr(policy_spec, "coarse_mapping_opt_steps", 25),
        coarse_mapping_opt_lr=getattr(policy_spec, "coarse_mapping_opt_lr", 0.05),
        verbose=False,
        **kwargs,
    )


def _run_single_parameter_identification(
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
    parameter_prior_covariance: float,
    traj_eval_interval: int,
    traj_eval_horizon: int,
    traj_eval_samples: int,
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
    from actdyn.utils.plotting import set_matplotlib_style

    exp_spec = get_experiment_spec(exp_id)
    policy_spec = get_policy_spec(policy_id)
    schedule_spec = get_schedule_spec(policy_spec.schedule_id)
    env_preset = get_environment_preset(exp_spec.env_preset_id)

    start_time = utc_now()
    set_matplotlib_style()
    device = configure_runtime(seed=seed)

    init_state = env_preset.sample_initial_state(seed)
    dz = int(env_preset.latent_dim)
    de = int(env_preset.embedding_dim)
    e_true = torch.as_tensor(
        env_preset.true_embedding_vector(embedding_dim=de), dtype=torch.float32, device=device
    ).unsqueeze(0)
    du = int(env_preset.action_dim)
    dy = int(env_preset.observation_dim)
    dt = float(env_preset.dt)
    alpha = float(env_preset.dynamics_alpha)
    fe_true, fz_true = _build_env_jacobians(
        env_preset,
        estimator=True,
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
    c, bias = calibrate_loglinear_loading(
        obs_model.network[0].weight,
        env_preset,
        target_snr=_loading_target_snr_db(env_preset),
    )
    obs_model.network[0].bias = nn.Parameter(bias)
    obs_model.network[0].weight = nn.Parameter(c)

    true_vec_env = actdyn.VectorFieldEnv(
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
        **_boundary_env_kwargs(env_preset),
    )
    true_vec_env.set_params(
        torch.as_tensor(
            env_preset.params_from_embedding(e_true.reshape(-1)),
            device=device,
        ),
    )
    env = actdyn.environment.EnvWrapper(true_vec_env, obs_model, action_model, dt=dt, device=device)

    mapping = actdyn.models.decoder.LogLinearMapping(
        latent_dim=dz, obs_dim=dy, dt=dt, device=device
    )
    noise = actdyn.models.decoder.PoissonNoise(obs_dim=dy, sigma=0.01, device=device)
    decoder = actdyn.models.Decoder(mapping=mapping, noise=noise, device=device)
    decoder.set_params(obs_model)
    loading_mismatch_variance = float(
        getattr(env_preset, "observation_loading_mismatch_variance", 0.0)
    )
    if loading_mismatch_variance < 0.0:
        raise ValueError(
            "observation_loading_mismatch_variance must be nonnegative, "
            f"got {loading_mismatch_variance}."
        )
    if loading_mismatch_variance > 0.0:
        decoder.mapping.set_weights(
            apply_loglinear_loading_mismatch(
                decoder.mapping.network[0].weight.data,
                variance=loading_mismatch_variance,
                seed=int(seed) + 11003,
            )
        )

    sim_vec_env = actdyn.VectorFieldEnv(
        env_preset.resolved_dynamics_type(estimator=True),
        x_range=5,
        dyn_params=None,
        dt=dt,
        alpha=alpha,
        Q=noise_scale,
        device=device,
        **_boundary_env_kwargs(env_preset),
    )
    sim_vec_env.set_params(
        torch.as_tensor(
            env_preset.params_from_embedding(torch.zeros(de, device=device), estimator=True),
            device=device,
        ),
    )
    dynamics = actdyn.models.dynamics.FunctionDynamics(
        state_dim=dz,
        dt=dt,
        dynamics_fn=sim_vec_env,
        param_formatter=lambda params: env_preset.params_from_embedding(params, estimator=True),
        device=device,
    )
    dynamics.logvar = nn.Parameter(torch.log(torch.ones(1, dz, device=device) * noise_scale))

    sigma_0 = max(float(parameter_prior_covariance), 1e-12)
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
    shrinkage_map = _make_shrinkage_map(
        shrinkage_kind=getattr(policy_spec, "shrinkage_kind", None),
        shrinkage_min=getattr(policy_spec, "shrinkage_min", None),
    )
    if "shrinkage_map" in fe_init.parameters and shrinkage_map is not None:
        model_kwargs["shrinkage_map"] = shrinkage_map
    if (
        "shrinkage_min" in fe_init.parameters
        and getattr(policy_spec, "shrinkage_min", None) is not None
    ):
        model_kwargs["shrinkage_min"] = float(policy_spec.shrinkage_min)
    model = actdyn.models.FilteringEmbedding(**model_kwargs)
    model.set_params(e_bel["m"])

    metric = None
    if not policy_spec.passive and policy_spec.objective_kind is not None:
        base_metric = _build_metric(
            objective_kind=str(policy_spec.objective_kind),
            model=model,
            Fe_net=fe_true,
            Fz_net=fz_true,
            gamma=eig_gamma,
            device=device,
            sampling_variance_samples=int(sampling_variance_samples),
            sampling_variance_seed=int(seed),
            ambiguity_temperature=getattr(policy_spec, "ambiguity_temperature", None),
            ensemble_kind=getattr(policy_spec, "ensemble_kind", None),
        )
        _apply_boundary_visibility_to_metric(base_metric, env_preset)
        action_cost_weight = float(getattr(policy_spec, "action_cost_weight", 0.01))
        metrics = [base_metric]
        weights = [1.0]
        if action_cost_weight > 0.0:
            metrics.append(
                actdyn.metrics.NormalizedActionCost.from_action_bounds(
                    (action_model.action_space.low, action_model.action_space.high),
                    compute_type="sum",
                    device=device,
                    normalize_horizon=True,
                )
            )
            weights.append(action_cost_weight)
        metric = actdyn.metrics.CompositeMetric(
            metrics=metrics, compute_type="sum", weights=weights, device=device
        )

    policy = _instantiate_synthetic_policy(
        actdyn_module=actdyn,
        env=env,
        env_preset=env_preset,
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

    param_rows: list[dict[str, Any]] = []
    emb_rows: list[dict[str, Any]] = []
    info_rows: list[dict[str, Any]] = []
    traj_rows: list[dict[str, Any]] = []
    state_action_rows: list[dict[str, Any]] = []
    planned_traj_steps: list[int] = []
    planned_traj_frames: list[np.ndarray] = []
    perf_start = time.perf_counter()
    trace_rng = np.random.default_rng(seed + 137)
    e_true_flat = e_true.detach().reshape(-1)

    def _on_step_end(transition: dict[str, Any]) -> None:
        step = int(experiment.env_step)
        cpu_time_sec = float(time.perf_counter() - perf_start)
        e_est = _resolve_parameter_mean(model=model, policy=policy).reshape(-1)
        param_err = float(torch.linalg.norm(e_est - e_true_flat).item())
        e_cov = _resolve_parameter_covariance(model=model, policy=policy)
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
        emb_row: dict[str, Any] = {
            "step": step,
            "cpu_time_sec": cpu_time_sec,
            "cov_diag_mean": cov_diag_mean,
        }
        e_vec = e_est.reshape(-1)
        embedding_dim_active = int(e_vec.numel())
        full_dyn_params = env_preset.params_from_embedding(e_vec, estimator=True)
        full_dyn_params = torch.as_tensor(
            full_dyn_params, dtype=torch.float32, device=e_vec.device
        ).reshape(-1)
        emb_row["embedding_dim"] = embedding_dim_active
        emb_row["full_param_dim"] = int(full_dyn_params.numel())
        for idx, value in enumerate(e_vec.tolist()):
            emb_row[f"e{idx}"] = float(value)
        for idx, value in enumerate(full_dyn_params.tolist()):
            emb_row[f"dyn_param{idx}"] = float(value)
            emb_row[f"dyn_param_learned{idx}"] = int(idx < embedding_dim_active)
        if e_cov is not None:
            diag_list = diag.tolist()
            for idx, value in enumerate(diag_list):
                emb_row[f"cov_diag{idx}"] = float(value)
        else:
            if cov_diag0 is not None:
                emb_row["cov_diag0"] = cov_diag0
            if cov_diag1 is not None:
                emb_row["cov_diag1"] = cov_diag1
        emb_rows.append(emb_row)
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
                "innovation_statistic": (
                    float(model._last_innovation_statistic.reshape(-1).mean().item())
                    if getattr(model, "_last_innovation_statistic", None) is not None
                    else None
                ),
                "parameter_shrinkage": (
                    float(model._last_parameter_shrinkage.reshape(-1).mean().item())
                    if getattr(model, "_last_parameter_shrinkage", None) is not None
                    else None
                ),
                "state_posterior_updated": as_bool(transition.get("state_posterior_updated", True)),
                "parameter_posterior_updated": as_bool(
                    transition.get("parameter_posterior_updated", True)
                ),
                "window_buffer_length": int(transition.get("window_buffer_length", 0)),
                "boundary_visibility_mean": _boundary_visibility_mean(
                    transition.get("model_state", transition.get("env_state")), env_preset
                ),
            }
        )
        env_x, env_v = to_xy_pair(transition.get("env_state", torch.zeros(2, device=device)))
        model_x, model_v = to_xy_pair(transition.get("model_state", torch.zeros(2, device=device)))
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
        if traj_eval_interval > 0 and step % traj_eval_interval == 0:
            traj_rows.append(
                {
                    "step": step,
                    "cpu_time_sec": cpu_time_sec,
                    "trajectory_r2": trajectory_r2_vectorfield(
                        e_est=e_est,
                        e_true=e_true_flat,
                        true_dynamics_type=str(env_preset.resolved_dynamics_type()),
                        true_full_params=np.asarray(
                            env_preset.resolved_true_params(), dtype=np.float32
                        ),
                        estimator_dynamics_type=str(
                            env_preset.resolved_dynamics_type(estimator=True)
                        ),
                        estimator_full_params=np.asarray(
                            env_preset.resolved_true_params(estimator=True), dtype=np.float32
                        ),
                        true_min_embedding_dim=int(env_preset.resolved_min_embedding_dim()),
                        estimator_min_embedding_dim=int(env_preset.resolved_min_embedding_dim()),
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
    write_trace_csv(param_trace_path, param_rows, ["step", "cpu_time_sec", "parameter_error"])
    emb_value_fields = sorted(
        {key for row in emb_rows for key in row if key.startswith("e") and key[1:].isdigit()},
        key=lambda key: int(key[1:]),
    )
    emb_cov_fields = sorted(
        {
            key
            for row in emb_rows
            for key in row
            if key.startswith("cov_diag") and key.removeprefix("cov_diag").isdigit()
        },
        key=lambda key: int(key.removeprefix("cov_diag")),
    )
    emb_dyn_param_fields = sorted(
        {
            key
            for row in emb_rows
            for key in row
            if key.startswith("dyn_param")
            and not key.startswith("dyn_param_learned")
            and key.removeprefix("dyn_param").isdigit()
        },
        key=lambda key: int(key.removeprefix("dyn_param")),
    )
    emb_dyn_param_learned_fields = sorted(
        {
            key
            for row in emb_rows
            for key in row
            if key.startswith("dyn_param_learned")
            and key.removeprefix("dyn_param_learned").isdigit()
        },
        key=lambda key: int(key.removeprefix("dyn_param_learned")),
    )
    write_trace_csv(
        emb_trace_path,
        emb_rows,
        [
            "step",
            "cpu_time_sec",
            "embedding_dim",
            "full_param_dim",
            *emb_value_fields,
            *emb_dyn_param_fields,
            *emb_dyn_param_learned_fields,
            *emb_cov_fields,
            "cov_diag_mean",
        ],
    )
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
            "innovation_statistic",
            "parameter_shrinkage",
            "state_posterior_updated",
            "parameter_posterior_updated",
            "window_buffer_length",
            "boundary_visibility_mean",
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
    rollout_metrics = extract_rollout_metrics(result_dir)
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
            "env_preset_id": str(env_preset.preset_id),
            "system_id": str(env_preset.system_id),
            "system_label": str(getattr(env_preset, "system_label", None) or env_preset.system_id),
            "dynamics_type": str(env_preset.resolved_dynamics_type()),
            "estimator_system_id": str(_resolved_estimator_system_id(env_preset)),
            "estimator_system_label": str(
                getattr(env_preset, "estimator_system_id", None)
                or _resolved_estimator_system_id(env_preset)
            ),
            "estimator_dynamics_type": str(env_preset.resolved_dynamics_type(estimator=True)),
            "true_params_full": [float(x) for x in env_preset.resolved_true_params()],
            "estimator_true_params_full": [
                float(x) for x in env_preset.resolved_true_params(estimator=True)
            ],
            "state_low": [float(x) for x in env_preset.resolved_state_bounds()[0].tolist()],
            "state_high": [float(x) for x in env_preset.resolved_state_bounds()[1].tolist()],
            "min_embedding_dim": int(env_preset.resolved_min_embedding_dim()),
            "boundary_enabled": bool(getattr(env_preset, "boundary_enabled", False)),
            "boundary_type": str(getattr(env_preset, "boundary_type", "none")),
            "boundary_radius": getattr(env_preset, "boundary_radius", None),
            "boundary_barrier_enabled": bool(
                getattr(env_preset, "boundary_barrier_enabled", False)
            ),
            "boundary_projection_enabled": bool(
                getattr(env_preset, "boundary_projection_enabled", False)
            ),
            "boundary_barrier_width": float(getattr(env_preset, "boundary_barrier_width", 0.5)),
            "boundary_barrier_strength": float(
                getattr(env_preset, "boundary_barrier_strength", 5.0)
            ),
            "boundary_barrier_temperature": float(
                getattr(env_preset, "boundary_barrier_temperature", 0.1)
            ),
            "loading_fisher_snr_db": _loading_fisher_snr_db(env_preset),
            "loading_target_snr_db": _loading_target_snr_db(env_preset),
            "observation_loading_mismatch_variance": float(
                getattr(env_preset, "observation_loading_mismatch_variance", 0.0)
            ),
            "information_boundary_visibility_enabled": bool(
                getattr(env_preset, "information_boundary_visibility_enabled", False)
            ),
            "information_boundary_margin": float(
                getattr(env_preset, "information_boundary_margin", 1.0)
            ),
            "information_boundary_temperature": float(
                getattr(env_preset, "information_boundary_temperature", 0.15)
            ),
            "initial_state_true": [float(x) for x in init_state.tolist()],
            "embedding_true": [float(x) for x in e_true_flat.tolist()],
            "embedding_estimate": [
                float(x)
                for x in _resolve_parameter_mean(model=model, policy=policy).reshape(-1).tolist()
            ],
            "embedding_error_final": (
                float(param_rows[-1]["parameter_error"]) if param_rows else None
            ),
            "embedding_error_mean": (
                float(np.mean([row["parameter_error"] for row in param_rows]))
                if param_rows
                else None
            ),
            "trajectory_r2_final": float(traj_rows[-1]["trajectory_r2"]) if traj_rows else None,
            "trajectory_r2_mean": (
                float(np.mean([row["trajectory_r2"] for row in traj_rows])) if traj_rows else None
            ),
            "trajectory_eval_interval": int(traj_eval_interval),
            "trajectory_eval_horizon": int(traj_eval_horizon),
            "trajectory_eval_samples": int(traj_eval_samples),
            "objective_variant": (
                None if policy_spec.objective_kind is None else str(policy_spec.objective_kind)
            ),
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
            "parameter_prior_covariance": float(parameter_prior_covariance),
            "q_theta_meas_coeff": float(q_theta_meas_coeff),
            "q_theta_max_scale": float(q_theta_max_scale),
            "eig_gamma": float(eig_gamma),
            "action_cost_weight": float(getattr(policy_spec, "action_cost_weight", 0.01)),
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
            "writing_ref": WRITING_REFERENCE,
        },
    )
    return payload


def _run_one(
    *, exp_id: str, policy_id: str, seed: int, repeat: int, base_dir: Path, args: argparse.Namespace
) -> dict[str, Any]:
    exp_spec = get_experiment_spec(exp_id)
    total_steps = int(args.total_steps or exp_spec.total_steps)
    run_dir = base_dir / exp_id / "track" / policy_id / f"seed_{seed}" / f"repeat_{repeat:02d}"
    metadata_path = run_dir / "run_metadata.json"
    if bool(getattr(args, "skip_existing", False)) and metadata_path.exists():
        existing_payload = load_json(metadata_path)
        if str(existing_payload.get("status")) == "completed":
            return existing_payload
    ensure_dir(run_dir)
    try:
        if exp_spec.experiment_kind == "parameter":
            payload = _run_single_parameter_identification(
                exp_id=exp_id,
                policy_id=policy_id,
                seed=seed,
                total_steps=total_steps,
                run_dir=run_dir,
                eig_gamma=float(args.eig_gamma),
                q_theta=float(args.q_theta),
                q_theta_meas_coeff=float(args.q_theta_meas_coeff),
                q_theta_max_scale=float(args.q_theta_max_scale),
                parameter_prior_covariance=float(args.parameter_prior_covariance),
                traj_eval_interval=int(exp_spec.trajectory_eval_interval),
                traj_eval_horizon=int(exp_spec.trajectory_eval_horizon),
                traj_eval_samples=int(exp_spec.trajectory_eval_samples),
                sampling_variance_samples=int(args.sampling_variance_samples),
            )
        else:
            if __package__ in {None, ""}:
                from run_rbf_experiment import run_single_rbf_identification
            else:
                from .run_rbf_experiment import run_single_rbf_identification

            payload = run_single_rbf_identification(
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
                build_runtime_experiment_config=_build_runtime_experiment_config,
                build_metadata=_build_metadata,
                instantiate_synthetic_policy=_instantiate_synthetic_policy,
                resolve_parameter_mean=_resolve_parameter_mean,
                resolve_parameter_precision=_resolve_parameter_precision,
                resolve_parameter_covariance=_resolve_parameter_covariance,
            )
    except Exception as exc:
        payload = _build_metadata(
            exp_id=exp_id,
            policy_id=policy_id,
            seed=seed,
            total_steps=total_steps,
            run_dir=run_dir,
            status="failed",
            start_time=utc_now(),
            end_time=utc_now(),
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


def _build_session_experiment_entry(
    *,
    exp_id: str,
    seeds: list[int],
    repeats: int,
    total_steps_override: int | None,
    policy_filter: set[str] | None = None,
) -> dict[str, Any]:
    exp_spec = get_experiment_spec(exp_id)
    experiment_kind = str(exp_spec.experiment_kind)
    model_name = {
        "parameter": "FilteringEmbedding",
        "rbf": "SparseRbfFilteringModel",
    }.get(experiment_kind, "unknown")
    dynamics_name = {
        "parameter": "FunctionDynamics",
        "rbf": "SparseRbfDynamics",
    }.get(experiment_kind, "unknown")
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
                "schedule_id": str(schedule_spec.schedule_id),
                "schedule": {
                    "update_interval": int(schedule_spec.update_interval),
                    "replan_interval": int(schedule_spec.replan_interval),
                    "planning_horizon": int(schedule_spec.planning_horizon),
                    "predictive_only_window": bool(schedule_spec.predictive_only_window),
                },
                "coarse_dt_factor": int(getattr(policy_spec, "coarse_dt_factor", 1)),
                "coarse_action_mapping": str(getattr(policy_spec, "coarse_action_mapping", "hold")),
                "async_planning": bool(getattr(policy_spec, "async_planning", False)),
                "async_stale_tolerance": float(getattr(policy_spec, "async_stale_tolerance", 0.5)),
                "model": {
                    "runner": experiment_kind,
                    "filter_model": model_name,
                    "dynamics_model": dynamics_name,
                    "residual_form": True,
                },
            }
        )
    return {
        "exp_id": str(exp_spec.exp_id),
        "experiment_kind": experiment_kind,
        "total_steps_default": int(exp_spec.total_steps),
        "total_steps_resolved": int(total_steps_override or exp_spec.total_steps),
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
            "loading_fisher_snr_db": _loading_fisher_snr_db(env_preset),
            "loading_target_snr_db": _loading_target_snr_db(env_preset),
            "observation_loading_mismatch_variance": float(
                getattr(env_preset, "observation_loading_mismatch_variance", 0.0)
            ),
            "x_range": float(env_preset.resolved_plot_limit()),
            "real_data": bool(getattr(env_preset, "real_data", False)),
            "dataset_id": getattr(env_preset, "dataset_id", None),
            "dataset_path": getattr(env_preset, "dataset_path", None),
            "state_key": getattr(env_preset, "state_key", None),
            "observation_key": getattr(env_preset, "observation_key", None),
            "train_fraction": float(getattr(env_preset, "train_fraction", 0.7)),
            "time_bin_ms": float(getattr(env_preset, "time_bin_ms", 20.0)),
            "max_observation_dim": getattr(env_preset, "max_observation_dim", None),
            "boundary_enabled": bool(getattr(env_preset, "boundary_enabled", False)),
            "boundary_type": str(getattr(env_preset, "boundary_type", "none")),
            "boundary_radius": getattr(env_preset, "boundary_radius", None),
            "boundary_barrier_enabled": bool(
                getattr(env_preset, "boundary_barrier_enabled", False)
            ),
            "boundary_projection_enabled": bool(
                getattr(env_preset, "boundary_projection_enabled", False)
            ),
            "information_boundary_visibility_enabled": bool(
                getattr(env_preset, "information_boundary_visibility_enabled", False)
            ),
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
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "commit": current_commit(),
        "repo_root": str(repo_root()),
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
    payload["updated_at"] = utc_now()
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
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--q-theta", type=float, default=1e-4)
    parser.add_argument("--parameter-prior-covariance", type=float, default=1.0)
    parser.add_argument("--q-theta-meas-coeff", type=float, default=0.0)
    parser.add_argument("--q-theta-max-scale", type=float, default=10.0)
    parser.add_argument("--eig-gamma", type=float, default=1.0)
    parser.add_argument("--sampling-variance-samples", type=int, default=8)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--grid-lim", type=float, default=10.0)
    return parser


def main(argv: list[str] | None = None, *, suite_entries: Mapping[str, Any] | None = None) -> int:
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
        suite_entries=suite_entries,
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
            policy_id for exp_id in exp_ids for policy_id in get_experiment_spec(exp_id).policy_ids
        }
        unknown_policy_ids = sorted(policy_filter - available_policy_ids)
        if unknown_policy_ids:
            parser.error(
                f"Unknown policy ids for selected experiments: {', '.join(unknown_policy_ids)}"
            )
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
                    from summarize import main as summarize_main
                else:
                    from .summarize import main as summarize_main

                for exp_id in exp_ids:
                    summary_args = [
                        "--base-dir",
                        str(base_dir),
                        "--exp-id",
                        exp_id,
                        "--seeds",
                        args.seeds,
                    ]
                    if args.policy_ids:
                        summary_args.extend(["--policy-ids", str(args.policy_ids)])
                    summarize_main(summary_args)
            if args.mode in {"video", "all"}:
                if __package__ in {None, ""}:
                    from render_videos import main as render_main
                else:
                    from .render_videos import main as render_main

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
