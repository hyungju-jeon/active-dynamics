# %%
import os
import logging

# Suppress verbose font manager logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

from einops import rearrange
import numpy as np
import torch
import actdyn
from actdyn.config import ExperimentConfig
from actdyn.utils import save_load, setup_experiment
from actdyn.utils.validation import compute_model_r2
from actdyn.utils.rollout import Rollout

import matplotlib.pyplot as plt
from actdyn.utils.helper import to_np
from actdyn.utils.visualize import create_subplot
from vjf.model import VJF

from actdyn.utils.visualize import plot_vector_field, plot_per_dimension

plt.rcParams["text.usetex"] = False
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
plt.rcParams["font.size"] = 16


def list_to_str(lst):
    return "x".join([str(x) for x in lst])


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


# %% SETUP EXPERIMENT
if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "conf/config.yaml")
    exp_config = ExperimentConfig.from_yaml(config_path)
    project_root = os.path.dirname(os.path.dirname(actdyn.__file__))
    exp_config.results_dir = os.path.join(project_root, "results", "double_limitcycle")
    data_dir = exp_config.results_dir
    fig_dir = os.path.join(exp_config.results_dir, "figs")

    os.makedirs(fig_dir, exist_ok=True)
    # copy config file to results dir for reference
    os.system(f"cp {config_path} {fig_dir}")

    # Set random seeds
    torch.manual_seed(exp_config.seed)
    np.random.seed(exp_config.seed)
    if torch.cuda.is_available() and exp_config.device == "cuda":
        torch.set_default_device(exp_config.device)

    # Set up experiment
    experiment, agent, env, model_env = setup_experiment(exp_config)
    online_model = model_env.model
    # if not os.path.exists(exp_config.results_dir + "/validation.pkl"):
    experiment.generate_rollout(50, 500)
    validation_ro = save_load.load_rollout(exp_config.results_dir + "/validation.pkl").to(
        device=exp_config.device
    )
    multi_ro = save_load.load_rollout(exp_config.results_dir + "/train.pkl").to(
        device=exp_config.device
    )

    # -------------------------------
    # Single Trajectory - Online
    # -------------------------------
    experiment.run()
    single_ro = save_load.load_and_concatenate_rollouts(
        os.path.join(exp_config.results_dir, "rollouts")
    ).to(device=exp_config.device)

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
    off_multi_experiment, _, _, _ = setup_experiment(exp_config)
    multi_cfg = off_multi_experiment.cfg.training.get_offline_optim_cfg()
    loss = off_multi_experiment.agent.model.train_model(multi_ro, **multi_cfg)
    off_multi_model = off_multi_experiment.agent.model.model
    off_multi_experiment.agent.model.save_model(
        os.path.join(off_multi_experiment.results_path, f"model/model_full.pth")
    )

    # -------------------------------
    # VJF model
    # -------------------------------
    # model = VJF.make_model(
    #     exp_config.observation_dim,
    #     exp_config.latent_dim,
    #     udim=exp_config.action_dim,
    #     n_rbf=100,
    #     hidden_sizes=[20],
    #     likelihood="gaussian",
    # )
    # y = rearrange(multi_ro["obs"], "b t c -> t b c")
    # u = rearrange(multi_ro["action"], "b t c -> t b c")
    # m, logvar, _ = model.fit(y, u, max_iter=150)

    # y_lik = model.decoder(m)
    # x_pred, y_pred = model.forecast(x0=m[0, ...], u=u, n_step=1000, noise=False)

    # %
    # -------------------------------
    # k-step Prediction R2
    # -------------------------------
    validation_ro = save_load.load_rollout(data_dir + "/validation.pkl").to(
        device=exp_config.device
    )
    k_max = 50
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

    # %%
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

    start_idx = torch.randint(arx_model.order, T - k_max - 1, (n_idx,))
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

    # -------------------------------
    # Plot k-step Prediction
    # -------------------------------
    k_step = 50

    with torch.no_grad():
        t_start = 200
        batch_idx = 0

        y = validation_ro["next_obs"]
        u = validation_ro["action"]
        torch.manual_seed(0)
        B, T, D = y.shape
        z_post = off_multi_model.encoder(y, u, n_samples=10)[0]
        z_init = z_post[..., t_start : t_start + 1, :]
        y_lik = off_multi_model.decoder(z_post)
        u_enc = off_multi_model.action_encoder(u[..., t_start + 1 :, :])
        z_pred_list = off_multi_model.dynamics.sample_forward(
            z_init, action=u_enc, k_step=k_step, return_traj=True
        )[0]
        z_pred = torch.cat(z_pred_list, dim=-2)  # (S, B, k, D)
        y_pred = off_multi_model.decoder(z_pred)  # (S, B, k, D)
        y_pred_mean = y_pred.mean(dim=0)  # (B, k, D)

        y_arx_pred = arx_model.predict(
            validation_ro["obs"][:, t_start - arx_model.order + 1 : t_start + 1, :],
            u[:, t_start - arx_model.order + 1 : t_start + k_step + 1, :],
            k_step,
        )

        fix, ax = create_subplot(y)
        for i in range(y.shape[-1]):
            ax[i].plot(
                np.arange(t_start - 100, t_start + k_step + 1),
                to_np(y[batch_idx, t_start - 100 : t_start + k_step + 1, i]),
                label="True",
                alpha=0.5,
            )
            for j in range(y_pred.shape[0]):
                ax[i].plot(
                    np.arange(t_start, t_start + k_step + 1),
                    to_np(y_pred[j, batch_idx, :, i]),
                    color="C1",
                    alpha=0.15,
                )
            ax[i].plot(
                np.arange(t_start, t_start + k_step + 1),
                to_np(y_pred_mean[batch_idx, :, i]),
                label="Pred",
                alpha=0.7,
            )
            ax[i].plot(
                np.arange(t_start, t_start + k_step + 1),
                to_np(y_lik.mean(dim=0)[batch_idx, t_start : t_start + k_step + 1, i]),
                label="Rec",
                alpha=0.7,
            )
            ax[i].plot(
                np.arange(t_start, t_start + k_step + 1),
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
        # plt.close()
    # Memory cleanup after experiment following the project pattern
    if "cuda" in str(exp_config.device):
        torch.cuda.empty_cache()
