# %% Import
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


# %%
duffing_env = actdyn.VectorFieldEnv(
    "duffing",
    x_range=5,
    dyn_params=None,
    dt=0.01,
    alpha=1.0,
    Q=0.2,
    action_bounds=[-1.0, 1.0],
    state_bounds=[-5.0, 5.0],
    initial_state=[2.5, 2.5],
    device="cpu",
)

# %% Helpers

# %%

started = datetime.now(timezone.utc)
set_matplotlib_style()
device = configure_runtime(seed=seed)
torch.manual_seed(seed)

e_true = torch.as_tensor(
    _sample_true_embedding(seed), dtype=torch.float32, device=device
).unsqueeze(0)
a, b = e_true.reshape(-1)

dz, de, du, dy = 2, 2, 2, 50
dt = 0.01
alpha = float(dynamics_alpha)
noise_scale = max(1e-8, float(state_noise))
action_max = float(max(1e-6, action_max))

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
mean_firing = 50
max_firing_rate = 100.0
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
    device=device,
)
rnd_metric = actdyn.metrics.uncertainty.RandomNetworkDistillation(device=device)

effective_horizon = planning_horizon
if effective_horizon is None and exp_id != "random":
    effective_horizon = 10 if exp_id == "active_long" else 5

if exp_id == "random":
    policy = actdyn.policy.RandomPolicy(action_space=env.action_space, device=device)
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
embedding_rows: list[dict[str, Any]] = []
info_rows: list[dict[str, Any]] = []
traj_rows: list[dict[str, Any]] = []
state_action_rows: list[dict[str, Any]] = []
acq_map_steps: list[int] = []
acq_map_frames: list[np.ndarray] = []
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
        transition.get("env_action", transition.get("policy_action", torch.zeros(2, device=device)))
    )
    action_norm = float(np.sqrt(planned_action_x**2 + planned_action_v**2))
    policy_action_norm = float(np.sqrt(policy_action_x**2 + policy_action_v**2))
    env_action_norm = float(np.sqrt(env_action_x**2 + env_action_v**2))
    policy_action_delta = float(
        np.sqrt(
            (policy_action_x - planned_action_x) ** 2 + (policy_action_v - planned_action_v) ** 2
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
    policy_sat = bool(max(abs(policy_action_x), abs(policy_action_v)) >= float(action_max) - 1e-6)
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
        }
    )

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
