from tqdm import tqdm
from actdyn.core.agent import Agent

from torch.utils.tensorboard.writer import SummaryWriter
from actdyn.utils import save_load
from actdyn.utils.rollout import Rollout, RolloutBuffer
from actdyn.utils.video import VideoRecorder
from actdyn.config import ExperimentConfig
from actdyn.utils.validation_helpers import compute_kstep_r2
import matplotlib.pyplot as plt

import torch
import os
import shutil


class Experiment:
    def __init__(self, agent: Agent, config: ExperimentConfig, resume=False):
        self.agent = agent
        self.cfg = config
        self.env_step = 0
        self.prev_step = 0
        self.rollout = Rollout(device="cpu")
        self.writer = None
        self.video_recorder = None
        self.training_loss = 0
        self.results_dir = os.path.join(os.path.dirname(__file__), config.results_dir)
        if resume:
            ## TODO : Implement resume functionality
            print("Resuming from previous experiment state...")
            # self.agent.model.load(self.cfg.logging.model_path)

    def _setup_video_recording(self, video_path):
        if video_path:
            self.video_recorder = VideoRecorder(self.agent.env, video_path)
            self.video_recorder.capture_frame()
        else:
            self.video_recorder = None

    def _stop_video_recording(self):
        if self.video_recorder:
            self.video_recorder.save()
            self.video_recorder = None

    def run(self, reset=True):
        os.makedirs(self.results_dir, exist_ok=True)
        for subdir in ["rollouts", "logs", "model"]:
            p = os.path.join(self.results_dir, subdir)
            if os.path.exists(p):
                shutil.rmtree(p, ignore_errors=True)
            os.makedirs(p, exist_ok=True)

        self.writer = SummaryWriter(log_dir=os.path.join(self.results_dir, "logs"))

        # Initialize environment
        if reset:
            self.agent.reset(seed=int(self.cfg.seed))
            self.env_step = 0
            self.rollout.clear()
        else:
            print("Continueing from previous step:", self.env_step)
            self.prev_step = self.env_step
            self.rollout.clear()

        # Setup progress bar
        pbar = tqdm(total=self.cfg.training.total_steps, desc="Training")

        while self.env_step < self.cfg.training.total_steps + self.prev_step:
            self.env_step += 1

            with torch.no_grad():
                # 1. Plan
                action = self.agent.plan()
                # 2. Execute
                transition, done = self.agent.step(action)

            # Append transition to rollout
            self.rollout.add(**transition)

            if isinstance(self.training_loss, list):
                self.writer.add_scalar("train/ELBO", -self.training_loss[0][0], self.env_step)
                self.writer.add_scalar("train/log_like", self.training_loss[0][1], self.env_step)
                self.writer.add_scalar("train/kl_d", self.training_loss[0][2], self.env_step)
            else:
                self.writer.add_scalar("train/ELBO", 0, -self.env_step)
                self.writer.add_scalar("train/log_like", 0, self.env_step)
                self.writer.add_scalar("train/kl_d", 0, self.env_step)

            self.agent.update_policy(transition)

            if self.env_step % 10 == 0:
                if isinstance(self.training_loss, list) and len(self.training_loss) > 0:
                    elbo_loss = -self.training_loss[0][0]
                    pbar.set_postfix(
                        {"ELBO": f"{elbo_loss:.4f}, beta: {self.agent.model_env.model.beta:.4f}"}
                    )
                else:
                    pbar.set_postfix({"ELBO": "N/A"})
                pbar.update(10)

            # Train model periodically
            if (
                self.env_step % self.cfg.training.train_every == 0
                and self.env_step > self.cfg.training.rollout_horizon
            ):
                self.training_loss = self.agent.train_model(**self.cfg.training.get_optim_cfg())

            if self.cfg.logging.plot_every > 0:
                if self.env_step % self.cfg.logging.plot_every == 0:
                    pass

            # Periodic rollout saving for crash recovery and memory management
            if self.env_step % self.cfg.logging.save_every == 0:
                save_load.save_rollout(
                    self.rollout,
                    os.path.join(self.results_dir, f"rollouts/rollout_{self.env_step}.pkl"),
                )
                if self.env_step < self.cfg.training.total_steps:
                    self.rollout.clear()

            # Clean up tensors to prevent memory accumulation
            if "cuda" in str(self.agent.device):
                del transition, action
                torch.cuda.empty_cache()

            if done:
                break
        pbar.close()
        self.rollout.finalize()
        self.writer.close()
        self.agent.model_env.save_model(os.path.join(self.results_dir, f"model/model_final.pth"))

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

    def post_run(self, data_dir=None):
        # Load data if exist
        if data_dir is None:
            data_dir = os.path.join(self.results_dir)
        data_path = os.path.join(data_dir, "validation.pkl")

        if os.path.exists(data_path):
            rb = save_load.load_rollout(data_path)
        else:
            _, rb = self.generate_rollout(num_episodes=50, episode_length=500, rollout_dir=data_dir)
        #
        self.writer = SummaryWriter(log_dir=os.path.join(self.results_dir, "logs"))

        # ELBO on validation set
        validate_elbo(self, rb, self.writer)

        # K-step prediction R2
        validate_kstep_r2(self, rb, self.writer)

        self.writer.close()

    def offline_learning(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.results_dir, "logs"))
        self.rollout.clear()
        # Check if rollout exists in the results directory
        self.rollout = save_load.load_and_concatenate_rollouts(
            os.path.join(self.results_dir, "rollouts")
        )
        offline_cfg = self.cfg.training.get_offline_optim_cfg()

        # Perform offline learning
        self.training_loss = self.agent.model_env.train_model(self.rollout, **offline_cfg)

        elbo_list, loglike_list, kl_list = [], [], []
        for t in self.training_loss:
            elbo_list.append(float(-t[0]))
            loglike_list.append(float(t[1]))
            kl_list.append(float(t[2]))

        for i, (e, l, k) in enumerate(zip(elbo_list, loglike_list, kl_list), start=1):
            self.writer.add_scalar("offline/train/ELBO", e, i)
            self.writer.add_scalar("offline/train/log_like", l, i)
            self.writer.add_scalar("offline/train/kl_d", k, i)
        self.writer.close()


def validate_elbo(experiment: Experiment, rb: Rollout | RolloutBuffer, writer: SummaryWriter):
    B, T, D = rb["obs"].shape
    loss, log_like, kl_d = experiment.agent.model_env.model.compute_elbo(
        y=rb["next_obs"],
        u=rb["action"],
        idx=None,
        n_samples=16,
        beta=1,
        k_steps=1,
    )
    writer.add_scalar("validation/ELBO", -loss.item() / T, 0)
    writer.add_scalar("validation/log_like", log_like.item() / T, 0)
    writer.add_scalar("validation/kl_d", kl_d.item() / T, 0)


def validate_kstep_r2(experiment: Experiment, rb: Rollout | RolloutBuffer, writer: SummaryWriter):
    r2_fig_path = os.path.join(experiment.results_dir, "validation_kstep.png")
    r2m, _ = compute_kstep_r2(
        model=experiment.agent.model_env.model,
        rollout=rb,
        k_max=10,
        n_samples=50,
        fig_path=r2_fig_path,
    )
    for i in range(r2m.shape[1]):
        for k in range(r2m.shape[0]):
            writer.add_scalar(f"validation/kstep/r2_mean_{i}", r2m[k, i], k + 1)
