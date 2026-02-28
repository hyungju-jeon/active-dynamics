#!/usr/bin/env python3
"""Cosyne helper to preflight and run parameter-identification CISS experiments."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib
import inspect
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np

DEFAULT_EXP_IDS = ["active_short", "active_long", "RND", "random"]
DEFAULT_SEEDS = [0, 10, 20, 30, 40]

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


def _parse_csv_list(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",")]
    return [item for item in values if item]


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(item) for item in _parse_csv_list(raw)]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_trace_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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


def _duffing_rollout_no_input(z0, e, horizon: int, dt: float):
    import torch

    z = z0.clone()
    traj = [z]
    a = e[..., 0]
    b = e[..., 1]
    for _ in range(horizon):
        x = z[..., 0]
        v = z[..., 1]
        dx = v
        dv = a * v - b * x - 0.1 * x**3
        z = torch.stack((x + dt * dx, v + dt * dv), dim=-1)
        traj.append(z)
    return torch.stack(traj, dim=1)


def _trajectory_r2(
    e_est,
    e_true,
    *,
    dt: float,
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
        traj_true = _duffing_rollout_no_input(starts, e_true_batch, horizon=horizon, dt=dt)
        traj_est = _duffing_rollout_no_input(starts, e_est_batch, horizon=horizon, dt=dt)

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

        dyn_params = torch.as_tensor(flat_vals, device=getattr(env, "device", "cpu"), dtype=torch.float32)
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
    eig_gamma: float,
    dry_run: bool,
    planning_horizon: int | None,
    traj_eval_interval: int,
    traj_eval_horizon: int,
    traj_eval_samples: int,
    extra_metadata: dict[str, Any] | None = None,
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
    from actdyn.utils.helper import make_uniform_sampler
    from actdyn.utils.runtime import configure_runtime
    from actdyn.utils.visualize import set_matplotlib_style

    ctx = {
        "model_tag": model_tag,
        "commit": _current_commit(),
        "seed": seed,
        "exp_id": exp_id,
        "total_steps": total_steps,
        "base_dir": str(run_dir),
        "start_time": _utc_now(),
    }
    if dry_run:
        payload = _build_metadata(
            ctx=ctx,
            status="dry_run",
            end_time=ctx["start_time"],
            runtime_sec=0.0,
            results_path=run_dir,
            extra={
                "q_theta": float(q_theta),
                "k_theta": int(k_theta),
                "eig_gamma": float(eig_gamma),
                "planning_horizon": planning_horizon,
                "traj_eval_interval": int(traj_eval_interval),
                "traj_eval_horizon": int(traj_eval_horizon),
                "traj_eval_samples": int(traj_eval_samples),
                "parameter_error_trace_path": str(run_dir / "parameter_error_trace.csv"),
                "trajectory_r2_trace_path": str(run_dir / "trajectory_r2_trace.csv"),
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

    z_sampler = make_uniform_sampler(-5.0, 5.0, 2)
    e_sampler = make_uniform_sampler([-3.0, -2.0], [-0.1, 2.0], 2)
    _ = z_sampler(1)
    e_true = e_sampler(1)
    a, b = e_true.reshape(-1)

    dz, de, du, dy = 2, 2, 2, 50
    dt = 0.01
    alpha = 10.0
    action_strength = 0.1
    noise_scale = 0.1

    action_model = actdyn.environment.action.IdentityActionEncoder(
        d_action=du,
        d_latent=dz,
        action_dim=du,
        latent_dim=dz,
        action_bounds=[-action_strength * alpha, action_strength * alpha],
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
    mean_firing = 50
    bias = torch.log(mean_firing * torch.ones(dy, device=device)) - 0.5 * torch.diag(C @ C.T)
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
        device=device,
    )
    _set_vectorfield_params(duffing_env, torch.tensor([a, b, 0.1], device=device))
    env = actdyn.environment.EnvWrapper(duffing_env, obs_model, action_model, dt=dt, device=device)

    mapping = actdyn.models.decoder.LogLinearMapping(latent_dim=dz, obs_dim=dy, dt=dt, device=device)
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
        model_kwargs["k_theta"] = k_theta
    model = actdyn.models.FilteringEmbedding(**model_kwargs)
    model.set_params(e_bel["m"])

    emb_metric = actdyn.metrics.information.EmbeddingFisherMetric(
        model=model,
        Fe_net=_fe_true,
        Fz_net=_fz_true,
        gamma=eig_gamma,
        device=device,
    )
    rnd_metric = actdyn.metrics.uncertainty.RandomNetworkDistillation(device=device)

    effective_horizon = planning_horizon
    if effective_horizon is None and exp_id != "random":
        effective_horizon = 10 if exp_id == "active_long" else 5

    if exp_id == "random":
        policy = actdyn.policy.RandomPolicy(action_space=env.action_space, device=device)
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
            chunk=5 if int(effective_horizon) >= 10 else 3,
            verbose=False,
        )

    exp_config = ExperimentConfig.from_yaml(str(_repo_root() / "experiments/ciss/conf/config.yaml"))
    exp_config.results_dir = str(run_dir)
    exp_config.training.total_steps = total_steps
    exp_config.training.train_every = total_steps + 1
    exp_config.run_analysis = False

    agent = actdyn.Agent(env=env, model=model, buffer_length=10, policy=policy, device=device)
    experiment = actdyn.core.experiment.Experiment(agent=agent, config=exp_config)

    decoder.set_params(obs_model)

    trace_rows: list[dict[str, Any]] = []
    traj_rows: list[dict[str, Any]] = []
    perf_start = time.perf_counter()
    trace_rng = np.random.default_rng(seed + 137)
    e_true_flat = e_true.detach().reshape(-1)

    def _on_step_end(_: dict[str, Any]) -> None:
        step = int(experiment.env_step)
        cpu_time_sec = float(time.perf_counter() - perf_start)
        e_est = model.e["m"].detach().reshape(-1)
        param_err = float(torch.linalg.norm(e_est - e_true_flat).item())
        trace_rows.append(
            {
                "step": step,
                "cpu_time_sec": cpu_time_sec,
                "parameter_error": param_err,
            }
        )

        if traj_eval_interval > 0 and step % traj_eval_interval == 0:
            r2 = _trajectory_r2(
                e_est=e_est,
                e_true=e_true_flat,
                dt=dt,
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
    _write_trace_csv(
        param_trace_path,
        trace_rows,
        fields=["step", "cpu_time_sec", "parameter_error"],
    )
    _write_trace_csv(
        traj_trace_path,
        traj_rows,
        fields=["step", "cpu_time_sec", "trajectory_r2", "traj_eval_horizon", "traj_eval_samples"],
    )

    rollout_metrics = _extract_rollout_metrics(result_dir)
    if trace_rows:
        param_series = np.asarray([row["parameter_error"] for row in trace_rows], dtype=np.float64)
        embedding_error_final = float(param_series[-1])
        embedding_error_mean = float(param_series.mean())
    else:
        embedding_error_final = float(torch.norm(model.e["m"].reshape(-1) - e_true_flat).item())
        embedding_error_mean = None

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
            "k_theta": int(k_theta),
            "eig_gamma": float(eig_gamma),
            "planning_horizon": int(effective_horizon) if effective_horizon is not None else None,
            "traj_eval_interval": int(traj_eval_interval),
            "traj_eval_horizon": int(traj_eval_horizon),
            "traj_eval_samples": int(traj_eval_samples),
            "parameter_error_trace_path": str(param_trace_path),
            "trajectory_r2_trace_path": str(traj_trace_path),
            "writing_ref": WRITING_REFERENCE,
        },
    )
    if extra_metadata:
        payload.update(extra_metadata)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Cosyne CISS parameter-identification workflows")
    parser.add_argument(
        "--mode",
        choices=["preflight", "smoke", "tracks", "ablation", "all"],
        default="tracks",
        help="Execution scope",
    )
    parser.add_argument("--exp-ids", type=str, default=",".join(DEFAULT_EXP_IDS))
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--total-steps", type=int, default=1000, help="Track run steps")
    parser.add_argument("--smoke-steps", type=int, default=1000, help="Smoke run steps")
    parser.add_argument("--model-tag", type=str, default="updated")
    parser.add_argument("--base-dir", type=str, default="results/cosyne")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--q-theta", type=float, default=1e-4)
    parser.add_argument("--k-theta", type=int, default=10)
    parser.add_argument("--eig-gamma", type=float, default=1.0)
    parser.add_argument("--planning-horizon", type=int, default=None)
    parser.add_argument("--traj-eval-interval", type=int, default=100)
    parser.add_argument("--traj-eval-horizon", type=int, default=100)
    parser.add_argument("--traj-eval-samples", type=int, default=16)

    parser.add_argument("--ablation-exp-id", type=str, default="active_short")
    parser.add_argument("--ablation-total-steps", type=int, default=1000)
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
    eig_gamma: float,
    dry_run: bool,
    planning_horizon: int | None,
    traj_eval_interval: int,
    traj_eval_horizon: int,
    traj_eval_samples: int,
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
        eig_gamma=eig_gamma,
        dry_run=dry_run,
        planning_horizon=planning_horizon,
        traj_eval_interval=traj_eval_interval,
        traj_eval_horizon=traj_eval_horizon,
        traj_eval_samples=traj_eval_samples,
    )
    _write_json(run_dir / "run_metadata.json", metadata)


def _run_track_matrix(
    model_tag: str,
    exp_ids: list[str],
    seeds: list[int],
    repeats: int,
    total_steps: int,
    base_dir: Path,
    q_theta: float,
    k_theta: int,
    eig_gamma: float,
    dry_run: bool,
    planning_horizon: int | None,
    traj_eval_interval: int,
    traj_eval_horizon: int,
    traj_eval_samples: int,
) -> None:
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
                try:
                    metadata = _run_single_parameter_identification(
                        model_tag=model_tag,
                        exp_id=exp_id,
                        seed=seed,
                        total_steps=total_steps,
                        run_dir=run_dir,
                        q_theta=q_theta,
                        k_theta=k_theta,
                        eig_gamma=eig_gamma,
                        dry_run=dry_run,
                        planning_horizon=planning_horizon,
                        traj_eval_interval=traj_eval_interval,
                        traj_eval_horizon=traj_eval_horizon,
                        traj_eval_samples=traj_eval_samples,
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
                _write_json(run_dir / "run_metadata.json", metadata)


def _run_ablation_suite(
    model_tag: str,
    seeds: list[int],
    repeats: int,
    total_steps: int,
    base_dir: Path,
    q_theta: float,
    eig_gamma: float,
    dry_run: bool,
    exp_id: str,
    planning_windows: list[int],
    k_thetas: list[int],
    fixed_k_theta: int,
    fixed_planning_window: int,
    traj_eval_interval: int,
    traj_eval_horizon: int,
    traj_eval_samples: int,
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
                        eig_gamma=eig_gamma,
                        dry_run=dry_run,
                        planning_horizon=planning_horizon,
                        traj_eval_interval=traj_eval_interval,
                        traj_eval_horizon=traj_eval_horizon,
                        traj_eval_samples=traj_eval_samples,
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
                        eig_gamma=eig_gamma,
                        dry_run=dry_run,
                        planning_horizon=fixed_planning_window,
                        traj_eval_interval=traj_eval_interval,
                        traj_eval_horizon=traj_eval_horizon,
                        traj_eval_samples=traj_eval_samples,
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
            eig_gamma=args.eig_gamma,
            dry_run=args.dry_run,
            planning_horizon=args.planning_horizon,
            traj_eval_interval=args.traj_eval_interval,
            traj_eval_horizon=args.traj_eval_horizon,
            traj_eval_samples=args.traj_eval_samples,
        )

    if args.mode in {"tracks", "all"}:
        _run_track_matrix(
            model_tag=args.model_tag,
            exp_ids=exp_ids,
            seeds=seeds,
            repeats=args.repeats,
            total_steps=args.total_steps,
            base_dir=base_dir,
            q_theta=args.q_theta,
            k_theta=args.k_theta,
            eig_gamma=args.eig_gamma,
            dry_run=args.dry_run,
            planning_horizon=args.planning_horizon,
            traj_eval_interval=args.traj_eval_interval,
            traj_eval_horizon=args.traj_eval_horizon,
            traj_eval_samples=args.traj_eval_samples,
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
            eig_gamma=args.eig_gamma,
            dry_run=args.dry_run,
            exp_id=args.ablation_exp_id,
            planning_windows=planning_windows,
            k_thetas=k_thetas,
            fixed_k_theta=args.ablation_fixed_k_theta,
            fixed_planning_window=args.ablation_fixed_planning_window,
            traj_eval_interval=args.traj_eval_interval,
            traj_eval_horizon=args.traj_eval_horizon,
            traj_eval_samples=args.traj_eval_samples,
        )

    print(f"Finished mode={args.mode} model_tag={args.model_tag} base_dir={base_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
