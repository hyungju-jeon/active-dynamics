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
    from actdyn.policy.baseline_prbs import _prbs_selection_order
    from actdyn.utils.experiment_runtime import extract_rollout_metrics, write_trace_csv
    from actdyn.utils.runtime import utc_now
    from cosyne.realdata_spiking import (
        build_transition_matrices,
        evaluate_prediction_mse,
        evaluate_prediction_r2,
        fit_linear_dynamics_ridge,
        load_replay_dataset,
        split_replay_dataset,
    )
else:
    from .experiment_specs import (
        get_environment_preset,
        get_experiment_spec,
        get_policy_spec,
        get_schedule_spec,
    )
    from actdyn.policy.baseline_prbs import _prbs_selection_order
    from actdyn.utils.experiment_runtime import extract_rollout_metrics, write_trace_csv
    from actdyn.utils.runtime import utc_now
    from .cosyne.realdata_spiking import (
        build_transition_matrices,
        evaluate_prediction_mse,
        evaluate_prediction_r2,
        fit_linear_dynamics_ridge,
        load_replay_dataset,
        split_replay_dataset,
    )

WRITING_REFERENCE = "docs/active-dynamics-writing/methods.tex"

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

    if objective_kind in {
        None,
        "parameter_eig",
        "shrinkage_parameter_eig",
        "ambiguity_aware_parameter_eig",
        "fully_observable_parameter_eig",
    }:
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


def run_single_realdata_replay(
    *,
    exp_id: str,
    policy_id: str,
    seed: int,
    total_steps: int,
    run_dir: Path,
    build_metadata,
    resolved_policy_type,
) -> dict[str, Any]:
    exp_spec = get_experiment_spec(exp_id)
    policy_spec = get_policy_spec(policy_id)
    schedule_spec = get_schedule_spec(policy_spec.schedule_id)
    env_preset = get_environment_preset(exp_spec.env_preset_id)

    start_time = utc_now()
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

    policy_type = resolved_policy_type(policy_id, policy_spec)
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
                "Pz11": (
                    float(cov_state[1, 1]) if cov_state.shape[0] > 1 else float(cov_state[0, 0])
                ),
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
            "state_posterior_updated",
            "parameter_posterior_updated",
            "window_buffer_length",
            "eigmin_info",
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
            "true_next_x",
            "true_next_v",
        ],
    )

    ended = datetime.now(timezone.utc)
    result_dir = run_dir
    rollout_metrics = extract_rollout_metrics(result_dir)
    final_coef = coef.reshape(-1)
    payload = build_metadata(
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
            "coarse_dt_factor": int(getattr(policy_spec, "coarse_dt_factor", 1)),
            "coarse_action_mapping": str(getattr(policy_spec, "coarse_action_mapping", "hold")),
            "async_planning": bool(getattr(policy_spec, "async_planning", False)),
            "async_stale_tolerance": float(getattr(policy_spec, "async_stale_tolerance", 0.5)),
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
