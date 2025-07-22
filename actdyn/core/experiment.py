from tqdm import tqdm
from actdyn.core.agent import Agent
from actdyn.utils.logger import Logger
from actdyn.utils import save_load
from actdyn.utils.rollout import Rollout
from actdyn.utils.video import VideoRecorder
from actdyn.config import ExperimentConfig
import torch


class Experiment:
    def __init__(self, agent: Agent, config: ExperimentConfig, resume=False):
        self.agent = agent
        self.config = config
        self.env_step = 0
        self.logger = Logger()
        self.video_recorder = None
        self.rollout = Rollout()

        if resume:
            self.agent.model.load(self.config.logging.model_path)
            self.agent.buffer = save_load.load_buffer(self.config.logging.buffer_path)
            self.logger = save_load.load_logger(self.config.logging.logger_path)

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

    def run(self):
        # Initialize environment
        self.agent.reset()
        video_path = self.config.logging.video_path

        # Setup progress bar
        pbar = tqdm(
            total=self.config.training.total_steps,
            desc="Training Progress",
            mininterval=1,
        )

        while self.env_step < self.config.training.total_steps:
            animate = self.env_step % self.config.training.animate_every == 0
            # Setup video recording if needed
            if animate:
                self._setup_video_recording(video_path)

            # Run for rollout horizon
            total_reward = 0

            # Use no_grad context for inference to save memory
            with torch.no_grad():
                # 1. Plan
                action = self.agent.plan()
                # 2. Execute
                transition, done = self.agent.step(action)

            self.rollout.add(**transition)

            total_reward += transition["reward"]

            # Capture frame if recording
            if self.video_recorder:
                self.video_recorder.capture_frame()

            if done:
                break

            # Stop video recording if it was active
            if animate:
                self._stop_video_recording()

            # Log episode metrics
            self.logger.log("reward", total_reward)
            self.env_step += 1

            # Update progress bar
            pbar.update(1)

            # Clean up tensors to prevent memory accumulation
            if "cuda" in str(self.agent.device):
                # Delete local variables that might hold GPU tensors
                del transition
                if "action" in locals():
                    del action

            # Train model periodically
            if (
                self.env_step % self.config.training.train_every == 0
                and self.env_step > self.config.training.rollout_horizon
            ):
                self.agent.train_model(
                    optimizer="SGD",
                    lr=1e-3,
                    weight_decay=1e-5,
                    n_epochs=1,
                    verbose=False,
                )
                # Clear GPU cache to prevent memory leaks
                if "cuda" in str(self.agent.device):
                    torch.cuda.empty_cache()
            if self.env_step % 1000 == 0:
                self.agent.model_env.render()

            # # Save periodically
            # if self.env_step % self.config.logging.save_every == 0:
            #     save_load.save_model(self.agent.model)
            #     save_load.save_buffer(self.agent.buffer)
            #     save_load.save_logger(self.logger)

        # Close progress bar
        pbar.close()
        self.logger.save()
