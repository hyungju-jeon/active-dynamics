from tqdm import tqdm
from actdyn.core.agent import Agent

from actdyn.utils.logger import Logger
from actdyn.utils import save_load
from actdyn.utils.rollout import Rollout, RolloutBuffer
from actdyn.utils.video import VideoRecorder
from actdyn.config import ExperimentConfig

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
        self.logger = Logger()
        self.video_recorder = None
        self.training_loss = 0

        # Create results directory
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

    def _get_log_path(self, step):
        # Calculate which JSON file this step belongs to
        log_file_index = ((step - 1) // self.cfg.logging.save_every) * self.cfg.logging.save_every
        return os.path.join(self.results_dir, f"logs/log_{log_file_index}.json")

    def run(self, reset=True):
        os.makedirs(self.results_dir, exist_ok=True)
        for subdir in ["rollouts", "logs"]:
            if os.path.exists(os.path.join(self.results_dir, subdir)):
                shutil.rmtree(os.path.join(self.results_dir, subdir), ignore_errors=True)
            os.makedirs(os.path.join(self.results_dir, subdir), exist_ok=True)

        # Initialize environment
        if reset:
            self.agent.reset(seed=int(self.cfg.seed))
            self.env_step = 0
            self.rollout.clear()
        else:
            print("Continueing from previous step:", self.env_step)
            self.prev_step = self.env_step
            self.rollout.clear()
        video_path = self.cfg.logging.video_path

        # Setup progress bar
        pbar = tqdm(total=self.cfg.training.total_steps, desc="Training")

        while self.env_step < self.cfg.training.total_steps + self.prev_step:
            self.env_step += 1

            # Use no_grad context for inference to save memory
            with torch.no_grad():
                # 1. Plan
                action = self.agent.plan()
                # 2. Execute
                transition, done = self.agent.step(action)

            # Append transition to rollout
            self.rollout.add(**transition)

            # Log scalar metrics immediately (lightweight)
            self.logger.log("step", self.env_step)
            if isinstance(self.training_loss, list):
                self.logger.log("elbo", -self.training_loss[0][0])
                self.logger.log("log_like", -self.training_loss[0][1])
                self.logger.log("kl_divergence", self.training_loss[0][2])
            else:
                self.logger.log("elbo", 0)
                self.logger.log("log_like", 0)
                self.logger.log("kl_divergence", 0)
            # Append to JSON every step for real-time monitoring
            self.logger.save(self._get_log_path(self.env_step))

            # Update policy if necessary
            self.agent.update_policy(transition)

            # Update progress bar only every 10 steps to reduce overhead
            if self.env_step % 10 == 0:
                # Update progress bar with training loss info
                if isinstance(self.training_loss, list) and len(self.training_loss) > 0:
                    elbo_loss = self.training_loss[0][0]
                    pbar.set_postfix({"ELBO": f"{elbo_loss:.4f}"})
                else:
                    pbar.set_postfix({"ELBO": "N/A"})
                pbar.update(10)  # Update by 10 steps at once

            # Train model periodically
            if (
                self.env_step % self.cfg.training.train_every == 0
                and self.env_step > self.cfg.training.rollout_horizon
            ):
                self.training_loss = self.agent.train_model(**self.cfg.training.get_optim_cfg())

            # Visualization every 1000 steps
            if self.cfg.logging.plot_every > 0:
                if self.env_step % self.cfg.logging.plot_every == 0:
                    ## Implement visualization logic for all environments
                    # self.visualize()
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
        # Final save
        pbar.close()
        self.rollout.finalize()

    def post_run(self):
        # TODO: Implement k-step prediction and evaluation
        pass

    def offline_learning(self):
        self.logger.clear()
        self.rollout.clear()
        # Check if rollout exists in the results directory
        rollout_files = [
            f for f in os.listdir(os.path.join(self.results_dir, "rollouts")) if f.endswith(".pkl")
        ]
        if not rollout_files:
            print("No rollout files found for offline learning.")
            return
        rollout_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
        print("Starting offline learning...")

        # Load and merge rollout
        self.rollout = Rollout()
        for rollout_file in rollout_files:
            rollout_path = os.path.join(self.results_dir, "rollouts", rollout_file)
            rollout = save_load.load_rollout(rollout_path)
            self.rollout.add(**rollout._data)

        offline_cfg = self.cfg.training.get_offline_optim_cfg()

        # Perform offline learning
        self.training_loss = self.agent.model_env.train_model(self.rollout, **offline_cfg)

        elbo_list, loglike_list, kl_list = [], [], []
        for t in self.training_loss:
            elbo_list.append(-float(t[0]))
            loglike_list.append(-float(t[1]))
            kl_list.append(float(t[2]))

        # Save final metrics
        self.logger.log("experiment", "offline_learning")
        self.logger.log("elbo", elbo_list)
        self.logger.log("log_like", loglike_list)
        self.logger.log("kl_divergence", kl_list)
        self.logger.save(os.path.join(self.results_dir, "logs", "offline_metrics.json"))
