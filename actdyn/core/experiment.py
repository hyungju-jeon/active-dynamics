import copy
import re
from datetime import datetime
from pathlib import Path
from typing import Callable

from matplotlib.figure import Figure
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from actdyn.config import ExperimentConfig
from actdyn.core.agent import Agent
from actdyn.utils import save_load
from actdyn.utils.helper import format_list, to_np
from actdyn.utils.rollout import Rollout, RolloutBuffer
from actdyn.utils.video import VideoRecorder

SESSION_DIR_PATTERN = re.compile(r"\d{8}_\d{4}_session\d{2}")


class Experiment:
    def __init__(self, agent: Agent, config: ExperimentConfig, resume: bool = False):
        self.agent = agent
        self.cfg = copy.deepcopy(config)
        self.env_step = 0
        self.prev_step = 0
        self.rollout = Rollout(device="cpu")
        self.writer = None
        self.video_recorder = None
        self.training_info = {}

        # Setup results directory
        config_results = Path(config.results_dir).expanduser()
        if config_results.is_absolute():
            base_results_path = config_results
        else:
            base_results_path = Path(__file__).resolve().parent / config_results
        self.base_results_path = base_results_path
        self.results_path = self._create_session_dir()
        self.cfg.results_dir = str(self.results_path)

        # Resume from previous experiment
        if resume:
            print("Resuming from previous experiment state...")
            try:
                self._resume_from_checkpoint()
                latest_session = self._find_latest_session_dir()
                if latest_session is not None:
                    self.results_path = latest_session
            except FileNotFoundError as exc:
                print(f"Resume skipped: {exc}")
            except Exception as exc:
                print(f"Resume encountered an issue: {exc}")

    def _create_session_dir(self) -> Path:
        # Create a new session directory
        base_dir = self.base_results_path
        base_dir.mkdir(parents=True, exist_ok=True)

        def next_session_dir() -> Path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            index = 1
            while True:
                candidate = base_dir / f"{timestamp}_session{index:02d}"
                try:
                    candidate.mkdir(parents=True, exist_ok=False)
                    return candidate
                except FileExistsError:
                    index += 1

        return next_session_dir()

    def _find_latest_session_dir(self) -> Path | None:
        # Find the latest session directory
        base_dir = getattr(self, "base_results_dir", None)
        if base_dir is None or not base_dir.is_dir():
            return None
        session_dirs = [
            path
            for path in base_dir.iterdir()
            if path.is_dir() and SESSION_DIR_PATTERN.fullmatch(path.name)
        ]
        if not session_dirs:
            return None
        session_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return session_dirs[0]

    def _resume_from_checkpoint(self):
        # Resume from previous experiment
        results_path = self.results_path.expanduser().resolve()
        if not results_path.exists():
            raise FileNotFoundError(f"Results directory not found at {results_path}")

        rollout_step = self._find_latest_rollout_step(results_path / "rollouts")
        if rollout_step is not None:
            self.env_step = rollout_step
            self.prev_step = rollout_step
            print(f"Restored training step to {self.env_step}")

        model_path = self._find_latest_model_checkpoint(results_path / "model")
        if model_path is not None:
            self.agent.model.load(model_path)
            print(f"Loaded model checkpoint from {model_path}")
        else:
            print("No model checkpoint found; using current model parameters.")

        try:
            self.agent.reset(seed=int(self.cfg.seed))
        except Exception as exc:  # pragma: no cover - environment specific
            print(f"Warning: could not reset agent during resume ({exc})")

    def _find_latest_rollout_step(self, rollout_dir: Path) -> int | None:
        # Find the latest rollout step from saved rollouts
        if not rollout_dir.is_dir():
            return None
        rollout_files = list(rollout_dir.glob("*.pkl"))
        if not rollout_files:
            return None

        def extract_step(path: Path) -> int | None:
            matches = re.findall(r"\d+", path.stem)
            return int(matches[-1]) if matches else None

        candidates: list[tuple[int, Path]] = []
        for path in rollout_files:
            step = extract_step(path)
            if step is not None:
                candidates.append((step, path))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        return candidates[-1][0]

    def _find_latest_model_checkpoint(self, model_dir: Path) -> str | None:
        # Find the latest model checkpoint file
        if not model_dir.is_dir():
            return None

        checkpoint_files = list(model_dir.glob("*.pth")) + list(model_dir.glob("*.pt"))
        if not checkpoint_files:
            return None

        def extract_step(path: Path) -> int | None:
            matches = re.findall(r"\d+", path.stem)
            return int(matches[-1]) if matches else None

        numeric_candidates: list[tuple[int, Path]] = []
        for path in checkpoint_files:
            step = extract_step(path)
            if step is not None:
                numeric_candidates.append((step, path))

        if numeric_candidates:
            numeric_candidates.sort(key=lambda item: item[0])
            return str(numeric_candidates[-1][1])

        final_candidates = [path for path in checkpoint_files if "final" in path.stem.lower()]
        if final_candidates:
            final_candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
            return str(final_candidates[0])

        checkpoint_files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return str(checkpoint_files[0])

    def _setup_video_recording(self, fps=30):
        # Setup video recording
        video_filename = self.cfg.logging.video_filename
        if video_filename:
            video_path = self.results_path / "video" / video_filename
            self.video_recorder = VideoRecorder(self.agent.env, video_path, fps=fps)
            self.video_recorder.capture_frame()
        else:
            self.video_recorder = None

    def _finalize_experiment(self):
        # Finalize experiment
        if self.writer:
            self.writer.close()
            self.writer = None
        if self.video_recorder:
            self.video_recorder.close()
            self.video_recorder = None
        if self.pbar:
            self.pbar.close()
            self.pbar = None

        self.rollout.finalize()
        self.agent.model.save(self.results_path / "model" / "model_final.pth")

    def init_experiment(self, reset=True):
        # Create necessary directories
        for subdir in ["rollouts", "logs", "model", "video"]:
            (self.results_path / subdir).mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.results_path / "logs")

        # Initialize environment
        if reset:
            self.agent.reset(seed=int(self.cfg.seed))
            self.env_step = 0
            self.rollout.clear()
        else:
            print("Continuing from previous step:", self.env_step)
            self.rollout.clear()

    def generate_rollout(
        self, num_episodes: int = 20, episode_length: int = 1000, rollout_dir: str = None
    ):
        # Generate random action rollouts for validation and offline training
        num_validate = num_episodes // 3
        num_train = num_episodes - num_validate

        rb = RolloutBuffer(max_size=num_episodes, device="cpu")
        pbar = tqdm(total=num_episodes, desc="Validation Episodes")
        for _ in range(num_episodes):
            ro = Rollout(device="cpu")
            obs, info = self.agent.env.reset()
            latent_state = info["latent_state"]
            for _ in range(episode_length):
                ro.add(obs=obs)
                ro.add(env_state=latent_state)
                action = self.agent.env.action_space.sample()
                action = self.agent.env._to_tensor(action)
                next_obs, reward, _, done, info = self.agent.env.step(action)
                ro.add(next_obs=next_obs)
                ro.add(action=action)
                ro.add(next_env_state=info["latent_state"])
                obs = next_obs
                latent_state = info["latent_state"]
            rb.add(ro)
            pbar.update(1)
        pbar.close()

        rb_train = RolloutBuffer(max_size=num_train, device="cpu")
        rb_train.add(rb[:num_train])
        rb_validate = RolloutBuffer(max_size=num_validate, device="cpu")
        rb_validate.add(rb[num_train:])

        target_dir = self.results_path if rollout_dir is None else Path(rollout_dir)
        validate_rollout_path = target_dir / "validation.pkl"
        train_rollout_path = target_dir / "train.pkl"

        save_load.save_rollout(rb_train, str(train_rollout_path))
        save_load.save_rollout(rb_validate, str(validate_rollout_path))
        print(f"rollout saved to {validate_rollout_path} and {train_rollout_path}")
        return rb_train, rb_validate

    def update_writer(self, info: dict, prefix=""):
        # Update TensorBoard writer with info dictionary
        if self.writer is None:
            return
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
        # Update progress bar with training info
        if self.env_step % interval == 0 and self.env_step > 0:
            pbar.set_postfix(
                {k: f"{format_list(v)}" for k, v in self.training_info.items()} | postfix
            )
            pbar.update(interval)

    def check_step(self, step_type: str) -> bool:
        # Check if it's time to train the model
        if step_type == "train":
            return (
                self.env_step % self.cfg.training.train_every == 0
                and self.env_step > self.cfg.training.rollout_horizon
            )
        elif step_type == "save":
            return self.env_step % self.cfg.logging.save_every == 0 and self.env_step > 0
        elif step_type == "plot":
            return self.env_step % self.cfg.logging.plot_every == 0 and self.env_step > 0
        return False

    def run(
        self,
        plot_fcn: Callable[[Agent], Figure] | None = None,
        reset: bool = True,
    ):
        train_cfg = self.cfg.training

        # Initialize environment
        self.init_experiment(reset=reset)
        self._setup_video_recording()

        # Setup progress bar
        self.pbar = tqdm(total=train_cfg.total_steps - self.env_step, desc="Online")
        while self.env_step < train_cfg.total_steps:
            self.env_step += 1

            # 1. Plan
            action = self.agent.plan()
            # 2. Execute
            transition, done = self.agent.step(action)
            self.rollout.add(**transition)

            # 3-1. Update policy
            self.agent.update_policy(transition)

            # 3-2. Train model
            if self.check_step("train"):
                sampling_ratio = self.agent.model.dynamics.dt / self.agent.env.dt
                self.training_info = self.agent.train_model(
                    **train_cfg.get_optim_cfg(), sampling_ratio=sampling_ratio
                )

            # 3-3. Update logs and plot
            self.update_writer(self.training_info)
            self.update_pbar(self.pbar)

            # Periodic rollout saving for crash recovery and memory management
            if self.check_step("save"):
                save_load.save_rollout(
                    self.rollout,
                    str(self.results_path / "rollouts" / f"rollout_{self.env_step}.pkl"),
                )
                if self.env_step < train_cfg.total_steps:
                    self.rollout.clear(keep_last=100)

            if self.check_step("plot"):
                if plot_fcn:
                    fig = plot_fcn(self.agent)
                    self.video_recorder.capture_frame(fig=fig)

            # Clean up tensors to prevent memory accumulation
            if "cuda" in str(self.agent.device):
                del transition, action
                torch.cuda.empty_cache()

            if done:
                break

        # Close progress bar and finalize experiment
        self._finalize_experiment()

    # TODO Update offline_run to match new experiment structure
    def offline_run(self, reset=True):
        if reset:
            if self.writer:
                self.writer.close()
            self.writer = SummaryWriter(log_dir=str(self.results_path / "logs"))
            self.rollout.clear()
            # Check if rollout exists in the results directory
            self.rollout = save_load.load_and_concatenate_rollouts(
                str(self.results_path / "rollouts")
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
        self._finalize_experiment()
        if "cuda" in str(self.agent.device):
            torch.cuda.empty_cache()


class MetaEmbeddingExperiment(Experiment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, plot_fcn: Callable[[Agent], Figure] | None = None, reset: bool = True):
        self.e_norm = []
        train_cfg = self.cfg.training

        # Initialize environment
        self.init_experiment(reset=reset)
        self._setup_video_recording()

        # Setup progress bar
        self.pbar = tqdm(total=train_cfg.total_steps - self.env_step, desc="Embedding")
        while self.env_step < train_cfg.total_steps:
            self.env_step += 1

            # 1. Plan
            action = self.agent.plan()
            # 2. Execute
            transition, done = self.agent.step(action)
            self.rollout.add(**transition)
            e_bel = self.agent.model.embedding.reshape(-1)

            # 3-1. Update policy
            self.agent.update_policy(transition)

            # 3-2. Train model
            if self.check_step("train"):
                sampling_ratio = self.agent.model.dynamics.dt / self.agent.env.dt
                self.training_info = self.agent.train_model(
                    **train_cfg.get_optim_cfg(), sampling_ratio=sampling_ratio
                )

            # 3-3. Update logs and plot
            self.training_info["e"] = e_bel
            e = self.agent.env.env.get_params()
            self.writer.add_scalars(
                "e",
                {f"true_{i}": v for i, v in enumerate(to_np(e).tolist())},
                self.env_step,
            )
            self.e_norm.append(
                torch.norm(e_bel.cpu() - self.agent.env.env.get_params().cpu()).item()
            )
            self.training_info["e_norm"] = self.e_norm[-1]
            self.update_writer(self.training_info)
            self.update_pbar(self.pbar)

            # Periodic rollout saving for crash recovery and memory management
            if self.check_step("save"):
                save_load.save_rollout(
                    self.rollout,
                    str(self.results_path / "rollouts" / f"rollout_{self.env_step}.pkl"),
                )
                if self.env_step < train_cfg.total_steps:
                    self.rollout.clear(keep_last=100)

            if self.check_step("plot"):
                if plot_fcn:
                    fig = plot_fcn(self.agent)
                    self.video_recorder.capture_frame(fig=fig)

            # Clean up tensors to prevent memory accumulation
            if "cuda" in str(self.agent.device):
                del transition, action
                torch.cuda.empty_cache()

            if done:
                break

        # Close progress bar and finalize experiment
        self._finalize_experiment()
