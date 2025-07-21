from actdyn.core.agent import Agent
from actdyn.utils.logger import Logger
from actdyn.utils import save_load
from actdyn.utils.video import VideoRecorder
from actdyn.config import *


class Experiment:
    def __init__(self, agent: Agent, config: ExperimentConfig, resume=False):
        self.agent = agent
        self.config = config
        self.env_step = 0
        self.logger = Logger()
        self.video_recorder = None

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

        while self.env_step < self.config.training.total_steps:
            animate = self.env_step % self.config.training.animate_every == 0

            # Setup video recording if needed
            if animate:
                self._setup_video_recording(video_path)

            # Run for rollout horizon
            total_reward = 0
            for _ in range(self.config.training.rollout_horizon):
                # 1. Plan
                action = self.agent.plan()

                # 2. Execute
                # state, reward, done, info = self.agent.step(action)
                obs, reward, done, env_info, model_info = self.agent.step(action)
                total_reward += reward

                # Capture frame if recording
                if self.video_recorder:
                    self.video_recorder.capture_frame()

                # # Log metrics
                # preds = self.agent.model.predict(state, action)
                # if isinstance(preds, list):
                #     import numpy as np

                #     std = np.std(np.array(preds), axis=0).mean()
                #     self.logger.log("uncertainty", std)

                if done:
                    break

            # Stop video recording if it was active
            if animate:
                self._stop_video_recording()

            # Log episode metrics
            self.logger.log("reward", total_reward)
            self.env_step += self.config.training.rollout_horizon

            # Train model periodically
            if self.env_step % self.config.training.train_every == 0:
                self.agent.train_model()

            # Save periodically
            if self.env_step % self.config.logging.save_every == 0:
                save_load.save_model(self.agent.model)
                save_load.save_buffer(self.agent.buffer)
                save_load.save_logger(self.logger)

        self.logger.save()
