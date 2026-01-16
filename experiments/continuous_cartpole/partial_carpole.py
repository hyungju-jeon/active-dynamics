# %%
import os
import numpy as np
import torch
from wandb import run
from actdyn.config import ExperimentConfig
from actdyn.utils import save_load, setup_experiment
from actdyn.utils.validation import compute_model_r2, compute_kstep_r2
from actdyn.utils.visualize import create_subplot
from actdyn.utils.rollout import Rollout, RolloutBuffer

import matplotlib.pyplot as plt
from actdyn.utils.helper import to_np
from torch.utils.tensorboard.writer import SummaryWriter

plt.rcParams["text.usetex"] = False
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
plt.rcParams["font.size"] = 16
plt.rcParams["pdf.fonttype"] = 42  # TrueType fonts


class ARX:
    def __init__(self, order=1):
        self.order = order
        self.A = None
        self.B = None

    def fit(self, y, u):
        # y: (B, T, D), u: (B, T, U)
        B, T, D = y.shape
        U = u.shape[-1]
        Y = []
        X = []
        for t in range(self.order, T):
            y_t = y[:, t, :].reshape(B, D)
            y_prev = y[:, t - self.order : t, :].reshape(B, D * self.order)
            u_prev = u[:, t - self.order : t, :].reshape(B, U * self.order)
            Y.append(y_t)
            X.append(torch.cat((y_prev, u_prev), dim=-1))
        Y = torch.cat(Y, dim=0)  # (B*(T-order), D)
        X = torch.cat(X, dim=0)  # (B*(T-order), D*order + U*order)
        # Solve for A and B using least squares
        X_pinv = torch.linalg.pinv(X)
        AB = X_pinv @ Y  # (D*order + U*order, D)
        self.A = AB[: D * self.order, :].reshape(self.order, D, D)  # (order, D, D)
        self.B = AB[D * self.order :, :].reshape(self.order, U, D)  # (order, U, D)

    def predict(self, y_init, u, k_step):
        # y_init: (B, order, D), u: (B, order+k_step, U)
        B, T, U = u.shape
        D = y_init.shape[-1]
        y_pred = [y for y in y_init.unbind(dim=1)]  # list of (B, D)
        for k in range(k_step):
            y_prev = y_pred[-self.order :]  # order * (B, D)
            y_prev = torch.stack(y_prev, dim=1).reshape(B, -1)  # (B, D*order)
            u_t = u[:, k : k + self.order, :].reshape(B, -1)  # (B, U*order)
            y_t = y_prev @ self.A.reshape(-1, D) + u_t @ self.B.reshape(-1, D)  # (B, D)
            y_pred.append(y_t)
        y_pred = torch.stack(y_pred[self.order - 1 :], dim=1)  # (B, k_step, D)
        return y_pred


def list_to_str(lst):
    return "x".join([str(x) for x in lst])


def run_experiment(exp_config: ExperimentConfig):
    data_dir = os.path.join(exp_config.results_dir)

    # Set random seeds
    if torch.cuda.is_available() and exp_config.device == "cuda":
        torch.set_default_device(exp_config.device)

    sample_ratio = exp_config.model.dyn_dt / exp_config.environment.env_dt
    # -------------------------------
    # Single Trajectory - Online
    # -------------------------------
    # Set up experiment
    torch.manual_seed(exp_config.seed)
    np.random.seed(exp_config.seed)
    experiment, _, _, _ = setup_experiment(exp_config)
    online_model = experiment.agent.model.model
    experiment.run()

    if not os.path.exists(data_dir + "/validation.pkl"):
        experiment.generate_rollout(100, 1000, data_dir)

    # -------------------------------
    # Single Trajectory - Offline
    # -------------------------------
    torch.manual_seed(exp_config.seed)
    np.random.seed(exp_config.seed)
    offline_experiment, _, _, _ = setup_experiment(exp_config)
    offline_experiment.offline_run()

    off_single_model = offline_experiment.agent.model.model
    offline_experiment.agent.model.save_model(
        os.path.join(offline_experiment.results_path, f"model/model_offline.pth")
    )

    # -------------------------------
    # Multiple Trajectories - Offline
    # -------------------------------
    torch.manual_seed(exp_config.seed)
    np.random.seed(exp_config.seed)
    multi_ro = save_load.load_rollout(data_dir + "/train.pkl").to(device=exp_config.device)

    multi_offline_experiment, _, _, _ = setup_experiment(exp_config)
    multi_cfg = multi_offline_experiment.cfg.training.get_offline_optim_cfg()
    multi_ro.downsample(n=int(sample_ratio))
    loss = multi_offline_experiment.agent.model.train_model(multi_ro, **multi_cfg)

    writer = SummaryWriter(log_dir=os.path.join(multi_offline_experiment.results_path, "logs"))
    elbo_list, loglike_list, kl_list = [], [], []
    for t in loss:
        elbo_list.append(float(-t[0]))
        loglike_list.append(float(t[1]))
        kl_list.append(float(t[2]))

    for i, (e, l, k) in enumerate(zip(elbo_list, loglike_list, kl_list), start=1):
        writer.add_scalar("offline_multi/train/ELBO", e, i)
        writer.add_scalar("offline_multi/train/log_like", l, i)
        writer.add_scalar("offline_multi/train/kl_d", k, i)

    writer.close()

    off_multi_model = multi_offline_experiment.agent.model.model
    multi_offline_experiment.agent.model.save_model(
        os.path.join(multi_offline_experiment.results_path, f"model/model_full.pth")
    )

    # -------------------------------
    # k-step Prediction R2
    # -------------------------------
    validation_ro = save_load.load_rollout(data_dir + "/validation.pkl").to(
        device=exp_config.device
    )
    # Adjust dt for dynamics model
    online_model.dynamics.dt = exp_config.environment.env_dt
    off_single_model.dynamics.dt = exp_config.environment.env_dt
    off_multi_model.dynamics.dt = exp_config.environment.env_dt

    k_max = 100

    fig_path = None
    fig_path = os.path.join(exp_config.results_dir, "online_kstep_r2_observation_validate.pdf")
    _, r2on_m, r2on_std = compute_model_r2(
        model=online_model, rollout=validation_ro, k_max=k_max, fig_path=fig_path
    )
    fig_path = os.path.join(exp_config.results_dir, "off_single_kstep_r2_observation_validate.pdf")
    _, r2off_s, r2off_std = compute_model_r2(
        model=off_single_model, rollout=validation_ro, k_max=k_max, fig_path=fig_path
    )
    fig_path = os.path.join(exp_config.results_dir, "off_multi_kstep_r2_observation_validate.pdf")
    _, r2full_m, r2full_std = compute_model_r2(
        model=off_multi_model, rollout=validation_ro, k_max=k_max, fig_path=fig_path
    )

    # -------------------------------
    # Baseline 2: Simple AR model
    # -------------------------------

    torch.manual_seed(exp_config.seed)
    np.random.seed(exp_config.seed)

    y = validation_ro["obs"]
    u = validation_ro["action"]
    arx_model = ARX(order=5)
    arx_model.fit(multi_ro["obs"], multi_ro["action"])
    n_idx = 100

    B, T, D = y.shape
    y_mean = y.mean(dim=(1), keepdim=True)

    start_idx = torch.randint(k_max, T - k_max - 1, (n_idx,))
    r2_list = []

    y_true_list = []
    y_pred_list = []
    for t_idx in start_idx:
        y_true_i = y[:, t_idx : t_idx + k_max + 1, :]  # (B, k, D)
        y_pred_i = arx_model.predict(
            y[:, t_idx - arx_model.order : t_idx, :],
            u[:, t_idx - arx_model.order : t_idx + k_max, :],
            k_max,
        )
        y_true_list.append(y_true_i)
        y_pred_list.append(y_pred_i)

    y_true = torch.stack(y_true_list, dim=0)  # (n_idx, B, k, D)
    y_pred = torch.stack(y_pred_list, dim=0)  # (n_idx, B, k, D)
    ss_res = ((y_true - y_pred) ** 2).sum(dim=0)  # (k, D)
    ss_tot = ((y_true - y_mean) ** 2).sum(dim=0)  # (k, D)

    r2 = 1 - ss_res / (ss_tot + 1e-6)  # (B, k, D)
    r2_mean = torch.mean(r2, dim=0)  # (k, D)
    r2_std = torch.std(r2, dim=0)  # (k, D)

    r2 = r2.cpu().numpy()
    r2_mean = r2_mean.cpu().numpy()
    r2_std = r2_std.cpu().numpy()

    #  Plot R2 against different methods
    fig, axs = create_subplot(r2on_m)
    for i in range(r2on_m.shape[1]):
        axs[i].plot(range(0, k_max + 1), r2on_m[:, i], label="Online")
        axs[i].fill_between(
            range(0, k_max + 1),
            r2on_m[:, i] - r2on_std[:, i],
            r2on_m[:, i] + r2on_std[:, i],
            alpha=0.3,
        )
        axs[i].plot(range(0, k_max + 1), r2off_s[:, i], label="Offline Single")
        axs[i].fill_between(
            range(0, k_max + 1),
            r2off_s[:, i] - r2off_std[:, i],
            r2off_s[:, i] + r2off_std[:, i],
            alpha=0.3,
        )
        axs[i].plot(range(0, k_max + 1), r2full_m[:, i], label="Offline Multi")
        axs[i].fill_between(
            range(0, k_max + 1),
            r2full_m[:, i] - r2full_std[:, i],
            r2full_m[:, i] + r2full_std[:, i],
            alpha=0.3,
        )
        axs[i].plot(range(0, k_max + 1), r2_mean[:, i], label="ARX")
        axs[i].fill_between(
            range(0, k_max + 1),
            r2_mean[:, i] - r2_std[:, i],
            r2_mean[:, i] + r2_std[:, i],
            alpha=0.3,
        )
        axs[i].set_title(f"Dimension {i+1}")
        axs[i].set_xlabel("Prediction Steps")
        axs[i].set_ylabel(r"$R^2$")
        y_min = max(-3, min(-0.1, np.min(r2_mean[:, i])))
        axs[i].set_ylim([y_min, 1.1])
        axs[i].grid(True)
        axs[i].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(exp_config.results_dir, "kstep_r2_comparison.pdf"))
    plt.close()

    # -------------------------------
    # Plot k-step Prediction
    # -------------------------------
    k_max = 100

    with torch.no_grad():
        t_start = 100
        batch_idx = 0

        y = validation_ro["next_obs"]
        u = validation_ro["action"]
        torch.manual_seed(0)
        B, T, D = y.shape
        z_post = off_multi_model.encoder(y, u, n_samples=10)[0]
        y_lik = off_multi_model.decoder(z_post)
        z_init = z_post[..., t_start, :].unsqueeze(-2)  # (S, B, 1, D)

        z_pred_list = [z_init]
        for i in range(k_max):
            u_enc = off_multi_model.action_encoder(
                u[..., t_start + 1 + i, :].unsqueeze(-2), z_pred_list[-1]
            )
            z_pred_list.append(
                off_multi_model.dynamics.sample_forward(
                    z_pred_list[-1], action=u_enc, k_step=1, return_traj=False
                )[1]
            )
        z_pred = torch.cat(z_pred_list, dim=-2)  # (S, B, k, D)
        y_pred = off_multi_model.decoder(z_pred)  # (S, B, k, D)
        y_pred_mean = y_pred.mean(dim=0)  # (B, k, D)

        y_arx_pred = arx_model.predict(
            validation_ro["obs"][:, t_start - arx_model.order + 1 : t_start + 1, :],
            u[:, t_start - arx_model.order + 1 : t_start + k_max + 1, :],
            k_max,
        )

        fix, ax = create_subplot(y)
        for i in range(y.shape[-1]):
            ax[i].plot(
                np.arange(t_start - 100, t_start + k_max + 1),
                to_np(y[batch_idx, t_start - 100 : t_start + k_max + 1, i]),
                label="True",
                alpha=0.5,
            )
            for j in range(y_pred.shape[0]):
                ax[i].plot(
                    np.arange(t_start, t_start + k_max + 1),
                    to_np(y_pred[j, batch_idx, :, i]),
                    color="C1",
                    alpha=0.1,
                )
            ax[i].plot(
                np.arange(t_start, t_start + k_max + 1),
                to_np(y_pred_mean[batch_idx, :, i]),
                label="Pred",
                alpha=0.7,
            )
            ax[i].plot(
                np.arange(t_start, t_start + k_max + 1),
                to_np(y_lik.mean(dim=0)[batch_idx, t_start : t_start + k_max + 1, i]),
                label="Rec",
                alpha=0.7,
            )
            ax[i].plot(
                np.arange(t_start, t_start + k_max + 1),
                to_np(y_arx_pred[batch_idx, :, i]),
                label="ARX",
                alpha=0.7,
            )
            ax[i].axvline(x=t_start, color="k", linestyle="--")
            ax[i].set_title(f"Dimension {i+1}")
            ax[i].set_xlabel("Time Step")
            ax[i].set_ylabel("Value")
            ax[i].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(exp_config.results_dir, "kstep_prediction.pdf"))
        plt.close()
    save_load.save_config(exp_config)
    # Memory cleanup after experiment following the project pattern
    if "cuda" in str(exp_config.device):
        torch.cuda.empty_cache()


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "conf/config.yaml")
    exp_config = ExperimentConfig.from_yaml(config_path)
    results_dir = os.path.dirname(__file__)
    base_dir = os.path.join(results_dir, "../../results", "cartpole_partial", "latent8")
    base_cfg = exp_config.clone()

    # %% regular training experiment.
    if run_residual := True:
        exp_config = base_cfg.clone()
        exp_config.model.is_residual = False
        exp_config.dt = 0.01
        exp_config.model.dyn_dt = 0.01
        exp_config.environment.env_dt = 0.01
        exp_config.results_dir = os.path.join(base_dir, "non_residual_slow")
        run_experiment(exp_config)

        exp_config = base_cfg.clone()
        exp_config.model.is_residual = True
        exp_config.dt = 0.01
        exp_config.model.dyn_dt = 0.01
        exp_config.environment.env_dt = 0.01
        exp_config.results_dir = os.path.join(base_dir, "residual_slow")
        run_experiment(exp_config)

        exp_config = base_cfg.clone()
        exp_config.model.is_residual = False
        exp_config.dt = 0.05
        exp_config.model.dyn_dt = 0.05
        exp_config.environment.env_dt = 0.05
        exp_config.results_dir = os.path.join(base_dir, "non_residual_fast")
        run_experiment(exp_config)

        exp_config = base_cfg.clone()
        exp_config.model.is_residual = True
        exp_config.dt = 0.05
        exp_config.model.dyn_dt = 0.05
        exp_config.environment.env_dt = 0.05
        exp_config.results_dir = os.path.join(base_dir, "residual_fast")
        run_experiment(exp_config)

        exp_config = base_cfg.clone()
        exp_config.model.is_residual = True
        exp_config.dt = 0.05
        exp_config.model.dyn_dt = 0.05
        exp_config.environment.env_dt = 0.05
        exp_config.results_dir = os.path.join(base_dir, "residual_fast")
        run_experiment(exp_config)

    # %% Async dynamics experiment
    if run_async := True:
        exp_config = base_cfg.clone()
        exp_config.model.is_residual = True
        exp_config.dt = 0.01
        exp_config.model.dyn_dt = 0.05
        exp_config.environment.env_dt = 0.01
        exp_config.results_dir = os.path.join(base_dir, "residual_slow_async")
        data_dir = os.path.join(exp_config.results_dir)
        run_experiment(exp_config)

    # %% Run  k-step experiment
    if run_kstep := True:
        exp_config = base_cfg.clone()
        exp_config.model.is_residual = True
        exp_config.dt = 0.01
        exp_config.model.dyn_dt = 0.05
        exp_config.environment.env_dt = 0.01
        exp_config.training.k_steps = 3
        exp_config.results_dir = os.path.join(base_dir, "residual_slow_async_k3")
        data_dir = os.path.join(exp_config.results_dir)
        run_experiment(exp_config)

        exp_config = base_cfg.clone()
        exp_config.model.is_residual = True
        exp_config.dt = 0.05
        exp_config.model.dyn_dt = 0.05
        exp_config.environment.env_dt = 0.05
        exp_config.training.k_steps = 3
        exp_config.results_dir = os.path.join(base_dir, "residual_fast_k3")
        run_experiment(exp_config)

        exp_config = base_cfg.clone()
        exp_config.model.is_residual = False
        exp_config.dt = 0.05
        exp_config.model.dyn_dt = 0.05
        exp_config.environment.env_dt = 0.05
        exp_config.training.k_steps = 3
        exp_config.results_dir = os.path.join(base_dir, "non_residual_fast_k3")
        run_experiment(exp_config)

    # %% Run Masking experiment
    if run_masking := True:
        exp_config = base_cfg.clone()
        exp_config.model.is_residual = True
        exp_config.dt = 0.01
        exp_config.model.dyn_dt = 0.05
        exp_config.environment.env_dt = 0.01
        exp_config.training.p_mask = 0.5
        exp_config.results_dir = os.path.join(base_dir, "residual_slow_async_mask")
        data_dir = os.path.join(exp_config.results_dir)
        run_experiment(exp_config)

        exp_config = base_cfg.clone()
        exp_config.model.is_residual = True
        exp_config.dt = 0.05
        exp_config.model.dyn_dt = 0.05
        exp_config.environment.env_dt = 0.05
        exp_config.training.p_mask = 0.5
        exp_config.results_dir = os.path.join(base_dir, "residual_fast_mask")
        run_experiment(exp_config)

        exp_config = base_cfg.clone()
        exp_config.model.is_residual = False
        exp_config.dt = 0.05
        exp_config.model.dyn_dt = 0.05
        exp_config.environment.env_dt = 0.05
        exp_config.training.p_mask = 0.5
        exp_config.results_dir = os.path.join(base_dir, "non_residual_fast_mask")
        run_experiment(exp_config)

    # %% State-dependent Action
    if run_state_action := True:
        exp_config = base_cfg.clone()
        exp_config.model.is_residual = True
        exp_config.dt = 0.05
        exp_config.model.dyn_dt = 0.05
        exp_config.environment.env_dt = 0.05
        exp_config.model.act_state_dependent = True
        exp_config.results_dir = os.path.join(base_dir, "residual_fast_state_action")
        data_dir = os.path.join(exp_config.results_dir)
        run_experiment(exp_config)

        exp_config = base_cfg.clone()
        exp_config.model.is_residual = True
        exp_config.dt = 0.05
        exp_config.model.dyn_dt = 0.05
        exp_config.environment.env_dt = 0.05
        exp_config.model.act_state_dependent = True
        exp_config.training.p_mask = 0.5
        exp_config.results_dir = os.path.join(base_dir, "residual_fast_state_action_mask")
        data_dir = os.path.join(exp_config.results_dir)
        run_experiment(exp_config)

        exp_config = base_cfg.clone()
        exp_config.model.is_residual = False
        exp_config.dt = 0.05
        exp_config.model.dyn_dt = 0.05
        exp_config.environment.env_dt = 0.05
        exp_config.model.act_state_dependent = True
        exp_config.training.p_mask = 0.5
        exp_config.results_dir = os.path.join(base_dir, "non_residual_fast_state_action_mask")
        data_dir = os.path.join(exp_config.results_dir)
        run_experiment(exp_config)
