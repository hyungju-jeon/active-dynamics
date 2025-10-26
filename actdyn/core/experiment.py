from tqdm import tqdm
from actdyn.core.agent import Agent

from torch.utils.tensorboard.writer import SummaryWriter
from actdyn.utils import save_load
from actdyn.utils.rollout import Rollout, RolloutBuffer
from actdyn.utils.video import VideoRecorder
from actdyn.utils.helper import format_list, to_np
from actdyn.config import ExperimentConfig
from actdyn.utils.validation import compute_kstep_r2
import matplotlib.pyplot as plt

import torch
import os
import shutil
import copy

from actdyn.utils.visualize import plot_vector_field


class Experiment:
    def __init__(self, agent: Agent, config: ExperimentConfig, resume=False):
        self.agent = agent
        self.cfg = copy.deepcopy(config)
        self.env_step = 0
        self.prev_step = 0
        self.rollout = Rollout(device="cpu")
        self.writer = None
        self.video_recorder = None
        self.training_info = {}
        self.results_dir = os.path.join(os.path.dirname(__file__), config.results_dir)
        if resume:
            ## TODO : Implement resume functionality
            print("Resuming from previous experiment state...")

            # self.agent.model.load(self.cfg.logging.model_path)

    def init_experiment(self, reset=True):
        os.makedirs(self.results_dir, exist_ok=True)
        for subdir in ["rollouts", "logs", "model"]:
            p = os.path.join(self.results_dir, subdir)
            if os.path.exists(p):
                shutil.rmtree(p, ignore_errors=True) if reset else None
            os.makedirs(p, exist_ok=True)

        self.writer = SummaryWriter(log_dir=os.path.join(self.results_dir, "logs"))

        # Initialize environment
        if reset:
            self.agent.reset(seed=int(self.cfg.seed))
            self.env_step = 0
            self.rollout.clear()
        else:
            print("Continuing from previous step:", self.env_step)
            self.rollout.clear()

    # TODO: Add video recording functionality
    def _setup_video_recording(self, video_path, fps=30):
        if video_path:
            self.video_recorder = VideoRecorder(self.agent.env, video_path, fps=fps)
            self.video_recorder.capture_frame()
        else:
            self.video_recorder = None

    # TODO: Add video recording functionality
    def _stop_video_recording(self):
        if self.video_recorder:
            self.video_recorder.save()
            self.video_recorder = None

    def generate_rollout(self, num_episodes=20, episode_length=1000, rollout_dir=None):
        num_validate = num_episodes // 3
        num_train = num_episodes - num_validate

        rb = RolloutBuffer(max_size=num_episodes, device="cpu")
        pbar = tqdm(total=num_episodes, desc="Validation Episodes")
        for ep in range(num_episodes):
            ro = Rollout(device="cpu")
            state, info = self.agent.env.reset()
            latent_state = info["latent_state"]
            for t in range(episode_length):
                ro.add(obs=state)
                ro.add(env_state=latent_state)
                action = self.agent.env.action_space.sample()
                action = self.agent.env._to_tensor(action)
                new_state, reward, _, done, info = self.agent.env.step(action)
                ro.add(next_obs=new_state)
                ro.add(action=action)
                ro.add(next_env_state=info["latent_state"])
                state = new_state
                latent_state = info["latent_state"]
            rb.add(ro)
            pbar.update(1)
        pbar.close()

        rb_train = RolloutBuffer(max_size=num_train, device="cpu")
        rb_train.add(rb[:num_train])
        rb_validate = RolloutBuffer(max_size=num_validate, device="cpu")
        rb_validate.add(rb[num_train:])

        if rollout_dir is None:
            validate_rollout_path = os.path.join(self.results_dir, "validation.pkl")
            train_rollout_path = os.path.join(self.results_dir, "train.pkl")
        else:
            validate_rollout_path = os.path.join(rollout_dir, "validation.pkl")
            train_rollout_path = os.path.join(rollout_dir, "train.pkl")

        save_load.save_rollout(rb_train, train_rollout_path)
        save_load.save_rollout(rb_validate, validate_rollout_path)
        print(f"rollout saved to {validate_rollout_path} and {train_rollout_path}")
        return rb_train, rb_validate

    def update_writer(self, info: dict, prefix=""):
        # Update logs
        for key, value in info.items():
            # if there is multiple values (e.g., list or tensor), log them using add_scalars
            if isinstance(value, (list, torch.Tensor)) and len(value) > 1:
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().numpy().tolist()
                scalar_dict = {f"{i}": v for i, v in enumerate(value)}
                self.writer.add_scalars(prefix + key, scalar_dict, self.env_step)
            else:
                self.writer.add_scalar(prefix + key, value, self.env_step)

    def update_pbar(self, pbar: tqdm, interval: int = 100, postfix: dict = {}):
        if self.env_step % interval == 0 and self.env_step > 0:
            pbar.set_postfix(
                {k: f"{format_list(v)}" for k, v in self.training_info.items()} | postfix
            )
            pbar.update(interval)

    @property
    def is_training_step(self):
        train_cfg = self.cfg.training
        return (
            self.env_step % train_cfg.train_every == 0 and self.env_step > train_cfg.rollout_horizon
        )

    @property
    def is_save_step(self):
        return self.env_step % self.cfg.logging.save_every == 0

    def run(self, reset=True):
        train_cfg = self.cfg.training

        # Initialize environment
        self.init_experiment(reset=reset)
        self._setup_video_recording(os.path.join(self.results_dir, "dynamics.mp4"))
        # Setup progress bar
        pbar = tqdm(total=train_cfg.total_steps - self.env_step, desc="Training")
        while self.env_step < train_cfg.total_steps:
            self.env_step += 1
            print("hi")
            # 1. Plan
            action = self.agent.plan()
            # 2. Execute
            transition, done = self.agent.step(action)

            # Append transition to rollout
            self.rollout.add(**transition)

            # Update policy
            self.agent.update_policy(transition)

            # Update logs
            self.update_writer(self.training_info)
            self.update_pbar(pbar)

            # Train model periodically
            if self.is_training_step:
                sampling_ratio = self.agent.model.dynamics.dt / self.agent.env.dt
                self.training_info = self.agent.train_model(
                    **train_cfg.get_optim_cfg(), sampling_ratio=sampling_ratio
                )

            # Periodic rollout saving for crash recovery and memory management
            if self.is_save_step:
                save_load.save_rollout(
                    self.rollout,
                    os.path.join(self.results_dir, f"rollouts/rollout_{self.env_step}.pkl"),
                )
                if self.env_step < train_cfg.total_steps:
                    self.rollout.clear()
                # for param in self.agent.policy.metric.metrics[0].predictor_network.parameters():
                #     param.data += 1e-3 * torch.randn_like(param.data)

            if self.env_step % self.cfg.logging.plot_every == 0 and self.env_step > 0:
                # Save intermediate model
                z = self.agent.recent["env_state"]
                z_hat = self.agent.recent["model_state"]

                plot_vector_field(self.agent.model.dynamics, x_range=5)
                # plot_vector_field(duffing_env.dynamics, x_range=3)
                plt.plot(to_np(z[0])[:, 0], to_np(z[0])[:, 1], alpha=0.7, label="true")
                plt.plot(to_np(z_hat[0])[:, 0], to_np(z_hat[0])[:, 1], alpha=0.7, label="model")
                plt.legend()
                plt.show()

            # Clean up tensors to prevent memory accumulation
            if "cuda" in str(self.agent.device):
                del transition, action
                torch.cuda.empty_cache()

            if done:
                break
        pbar.close()
        self.rollout.finalize()
        self.agent.model.save(os.path.join(self.results_dir, f"model/model_final.pth"))

    def post_run(self, data_dir=None):
        # Load data if exist
        if data_dir is None:
            data_dir = os.path.join(self.results_dir)
        data_path = os.path.join(data_dir, "validation.pkl")

        if os.path.exists(data_path):
            rb = save_load.load_rollout(data_path)
        else:
            _, rb = self.generate_rollout(num_episodes=50, episode_length=500, rollout_dir=data_dir)

        self.writer = SummaryWriter(log_dir=os.path.join(self.results_dir, "logs"))

        # # ELBO on validation set
        # validate_elbo(self, rb, self.writer)

        # # K-step prediction R2
        # validate_kstep_r2(self, rb, self.writer)

        self.writer.close()

    def offline_learning(self, reset=True):
        if reset:
            if self.writer:
                self.writer.close()
            self.writer = SummaryWriter(log_dir=os.path.join(self.results_dir, "logs"))
            self.rollout.clear()
            # Check if rollout exists in the results directory
            self.rollout = save_load.load_and_concatenate_rollouts(
                os.path.join(self.results_dir, "rollouts")
            )
            offline_cfg = self.cfg.training.get_offline_optim_cfg()
            print(f"Training params: {offline_cfg['param_list']}")

            sampling_ratio = self.agent.model.model.dynamics.dt / self.agent.env.dt
            self.rollout.downsample(n=int(sampling_ratio))

            # Perform offline learning
            self.training_loss = self.agent.model.train_model(self.rollout, **offline_cfg)
            elbo_list, loglike_list, kl_list = [], [], []
            for t in self.training_loss:
                elbo_list.append(float(-t[0]))
                loglike_list.append(float(t[1]))
                kl_list.append(float(t[2]))

        else:
            self.env_step = self.prev_step
            print("Continuing from previous step:", self.env_step)
            offline_cfg = self.cfg.training.get_offline_optim_cfg()
            # Perform offline learning
            self.training_loss = self.agent.model.train_model(self.rollout, **offline_cfg)
            elbo_list, loglike_list, kl_list = [], [], []
            for t in self.training_loss:
                elbo_list.append(float(-t[0]))
                loglike_list.append(float(t[1]))
                kl_list.append(float(t[2]))

        for i, (e, l, k) in enumerate(zip(elbo_list, loglike_list, kl_list), start=1):
            self.writer.add_scalar("offline/ELBO", e, i + self.env_step)
            self.writer.add_scalar("offline/log_like", l, i + self.env_step)
            self.writer.add_scalar("offline/kl_d", k, i + self.env_step)

        self.prev_step += offline_cfg["n_epochs"]

    def __del__(self):
        if self.writer:
            self.writer.close()


# def validate_elbo(experiment: Experiment, rb: Rollout | RolloutBuffer, writer: SummaryWriter):
#     B, T, D = rb["obs"].shape
#     loss, log_like, kl_d = experiment.agent.model_env.model.compute_elbo(
#         y=rb["next_obs"],
#         u=rb["action"],
#         idx=None,
#         n_samples=16,
#         beta=1,
#         k_steps=1,
#     )
#     writer.add_scalar("validation/ELBO", -loss.item() / T, 0)
#     writer.add_scalar("validation/log_like", log_like.item() / T, 0)
#     writer.add_scalar("validation/kl_d", kl_d.item() / T, 0)


# def validate_kstep_r2(experiment: Experiment, rb: Rollout | RolloutBuffer, writer: SummaryWriter):
#     r2_fig_path = os.path.join(experiment.results_dir, "validation_kstep.png")
#     r2m, _ = compute_kstep_r2(
#         model=experiment.agent.model_env.model,
#         rollout=rb,
#         k_max=10,
#         n_samples=50,
#         fig_path=r2_fig_path,
#     )
#     for i in range(r2m.shape[1]):
#         for k in range(r2m.shape[0]):
#             writer.add_scalar(f"validation/kstep/r2_mean_{i}", r2m[k, i], k + 1)


class MetaEmbeddingExperiment(Experiment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, reset=True):
        train_cfg = self.cfg.training
        self._setup_video_recording(os.path.join(self.results_dir, "dynamics.mp4"), fps=60)
        # Initialize environment
        self.init_experiment(reset=reset)

        # Setup progress bar
        self.pbar = tqdm(total=train_cfg.total_steps - self.env_step, desc="Training")
        while self.env_step < train_cfg.total_steps:
            self.env_step += 1

            # 1. Plan
            action = self.agent.plan()
            # 2. Execute
            transition, done = self.agent.step(action)
            e_bel = self.agent.model.embedding.reshape(-1)

            # Append transition to rollout
            self.rollout.add(**transition)

            # Update policy
            self.agent.update_policy(transition)

            # Update logs
            self.training_info["e"] = e_bel
            self.update_writer(self.training_info)
            self.writer.add_scalars(
                "e",
                {
                    "true_0": self.agent.env.env.dyn_param[0],
                    "true_1": self.agent.env.env.dyn_param[1],
                },
                self.env_step,
            )
            self.update_pbar(self.pbar)

            # Train model periodically
            if self.is_training_step:
                sampling_ratio = self.agent.model.dynamics.dt / self.agent.env.dt
                # self.training_info = self.agent.train_model(
                #     **train_cfg.get_optim_cfg(), sampling_ratio=sampling_ratio
                # )

            # Periodic rollout saving for crash recovery and memory management
            if self.is_save_step:
                save_load.save_rollout(
                    self.rollout,
                    os.path.join(self.results_dir, f"rollouts/rollout_{self.env_step}.pkl"),
                )
                self.rollout.clear()

            if self.env_step % self.cfg.logging.plot_every == 0 and self.env_step > 0:
                # Save intermediate model
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                plot_vector_field(
                    self.agent.model.dynamics, title="Learned Dynnimics", ax=ax, x_range=5
                )
                self.video_recorder.capture_frame(fig=fig)
                plt.close(fig)
                # plot_vector_field(duffing_env.dynamics, x_range=3)
                # plt.plot(to_np(z[0])[:, 0], to_np(z[0])[:, 1], alpha=0.7, label="true")
                # plt.plot(to_np(z_hat[0])[:, 0], to_np(z_hat[0])[:, 1], alpha=0.7, label="model")
                # plt.legend()
                # plt.show()

            # Clean up tensors to prevent memory accumulation
            if "cuda" in str(self.agent.device):
                del transition, action
                torch.cuda.empty_cache()

            if done:
                break

        self.pbar.close()
        self.agent.model.save(os.path.join(self.results_dir, f"model/model_final.pth"))
        self.video_recorder.close()

    # When closed or deleted
    def __del__(self):
        if self.writer:
            self.writer.close()
        if hasattr(self, "pbar") and self.pbar is not None:
            self.pbar.close()
