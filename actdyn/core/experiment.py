from __future__ import annotations

import copy
import queue
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from matplotlib.figure import Figure
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from actdyn.config import ExperimentConfig
from actdyn.core.agent import Agent
from actdyn.utils import to_np, format_list, VideoRecorder, Rollout, RolloutBuffer, save_rollout
from actdyn.utils.persistence import load_and_concatenate_rollouts

SESSION_DIR_PATTERN = re.compile(r"\d{8}_\d{4}_session\d{2}")


def _numeric_suffix(path: Path) -> int | None:
    matches = re.findall(r"\d+", path.stem)
    return int(matches[-1]) if matches else None


class _AsyncExperimentWriter:
    """Background TensorBoard and rollout writer for online experiments."""

    def __init__(self, log_dir: Path):
        self._log_dir = Path(log_dir)
        self._queue: queue.Queue[tuple] = queue.Queue()
        self._closed = False
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="actdyn-experiment-writer",
            daemon=True,
        )
        self._thread.start()

    def _check_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("background experiment writer failed") from self._error

    def _put(self, item: tuple) -> None:
        if self._closed:
            return
        self._check_error()
        self._queue.put(item)

    def add_scalar(self, *args: Any, **kwargs: Any) -> None:
        self._put(("scalar", args, kwargs))

    def add_scalars(self, *args: Any, **kwargs: Any) -> None:
        self._put(("scalars", args, kwargs))

    def add_transition(self, transition: dict) -> None:
        self._put(("transition", dict(transition)))

    def save_rollout(self, path: Path, *, keep_last: int | None = None) -> None:
        self._put(("save_rollout", str(path), keep_last))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(("close",))
        self._thread.join()
        self._check_error()

    def _run(self) -> None:
        writer = SummaryWriter(log_dir=str(self._log_dir))
        rollout = Rollout(device="cpu")
        try:
            while True:
                item = self._queue.get()
                op = item[0]
                if op == "close":
                    return
                if op == "scalar":
                    _, args, kwargs = item
                    writer.add_scalar(*args, **kwargs)
                elif op == "scalars":
                    _, args, kwargs = item
                    writer.add_scalars(*args, **kwargs)
                elif op == "transition":
                    _, transition = item
                    rollout.add(**transition)
                elif op == "save_rollout":
                    _, path, keep_last = item
                    save_rollout(rollout, path)
                    if keep_last is not None:
                        rollout.clear(keep_last=keep_last)
        except BaseException as exc:
            self._error = exc
        finally:
            writer.close()


class Experiment:
    def __init__(self, agent: Agent, config: ExperimentConfig, resume: bool = False):
        self.agent = agent
        self.cfg = copy.deepcopy(config)
        self.env_step = 0
        self.prev_step = 0
        self.rollout = Rollout(device="cpu")
        self.writer = None
        self.video_recorder = None
        self.pbar = None
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
        base_dir = getattr(self, "base_results_path", None)
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

        candidates: list[tuple[int, Path]] = []
        for path in rollout_files:
            step = _numeric_suffix(path)
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

        numeric_candidates: list[tuple[int, Path]] = []
        for path in checkpoint_files:
            step = _numeric_suffix(path)
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
            self.video_recorder = VideoRecorder(video_path, fps=fps)
            self.video_recorder.capture_frame()
        else:
            self.video_recorder = None

    def _finalize_experiment(self):
        # Finalize experiment
        if hasattr(self, "writer") and self.writer:
            self.writer.close()
            self.writer = None
        if hasattr(self, "video_recorder") and self.video_recorder:
            self.video_recorder.close()
            self.video_recorder = None
        if hasattr(self, "pbar") and self.pbar:
            self.pbar.close()
            self.pbar = None

        if hasattr(self, "rollout") and self.rollout:
            self.rollout.finalize()
        if hasattr(self, "agent") and self.agent and hasattr(self, "results_path"):
            close_agent = getattr(self.agent, "close", None)
            if callable(close_agent):
                close_agent()
            self.agent.model.save(self.results_path / "model" / "model_final.pth")

    def init_experiment(self, reset=True):
        # Create necessary directories
        for subdir in ["rollouts", "logs", "model", "video"]:
            (self.results_path / subdir).mkdir(parents=True, exist_ok=True)
        self.writer = _AsyncExperimentWriter(self.results_path / "logs")

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

        save_rollout(rb_train, str(train_rollout_path))
        save_rollout(rb_validate, str(validate_rollout_path))
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

    def update_pbar(self, pbar: tqdm, interval: int = 100, postfix: dict | None = None):
        # Update progress bar with training info
        if postfix is None:
            postfix = {}
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
        self._run_online_loop(
            train_cfg=self.cfg.training,
            pbar_desc="Online",
            plot_fcn=plot_fcn,
            reset=reset,
        )

    def _run_online_loop(
        self,
        train_cfg,
        pbar_desc: str,
        plot_fcn: Callable[[Agent], Figure] | None,
        reset: bool,
        on_step_end: Callable[[dict], None] | None = None,
    ) -> None:
        self.init_experiment(reset=reset)
        self._setup_video_recording()
        self.pbar = tqdm(total=train_cfg.total_steps - self.env_step, desc=pbar_desc)

        set_foreground_active = getattr(self.agent, "set_foreground_active", None)

        while self.env_step < train_cfg.total_steps:
            self.env_step += 1
            plan_start = time.perf_counter()
            if callable(set_foreground_active):
                set_foreground_active(True)
            try:
                action = self.agent.plan()
            finally:
                if callable(set_foreground_active):
                    set_foreground_active(False)
            plan_sec = time.perf_counter() - plan_start
            plan_info = getattr(getattr(self.agent, "policy", None), "last_plan_info", {}) or {}
            step_start = time.perf_counter()
            if callable(set_foreground_active):
                set_foreground_active(True)
            try:
                transition, done = self.agent.step(action)
            finally:
                if callable(set_foreground_active):
                    set_foreground_active(False)
            step_sec = time.perf_counter() - step_start
            launch_start = time.perf_counter()
            launch_background_plan = getattr(self.agent, "launch_background_plan", None)
            launch_info = {}
            if callable(launch_background_plan):
                launch_info = launch_background_plan() or {}
            launch_sec = time.perf_counter() - launch_start
            if isinstance(launch_info, dict):
                transition.update(launch_info)
            transition["loop_plan_sec"] = float(plan_sec)
            transition["loop_step_sec"] = float(step_sec)
            transition["loop_async_launch_sec"] = float(launch_sec)
            transition["loop_compute_sec"] = float(plan_sec + step_sec + launch_sec)
            transition["loop_plan_executed"] = bool(plan_info.get("plan_executed", False))
            transition["loop_plan_reason"] = str(plan_info.get("plan_reason", "none"))
            self.rollout.add(**transition)
            add_transition = getattr(getattr(self, "writer", None), "add_transition", None)
            if callable(add_transition):
                add_transition(transition)

            if self.check_step("train"):
                sampling_ratio = self.agent.model.dynamics.dt / self.agent.env.dt
                self.training_info = self.agent.train_model(
                    **train_cfg.get_optim_cfg(), sampling_ratio=sampling_ratio
                )

            if on_step_end is not None:
                on_step_end(transition)

            self.update_writer(self.training_info)
            self.update_pbar(self.pbar)

            if self.check_step("save"):
                rollout_path = self.results_path / "rollouts" / f"rollout_{self.env_step}.pkl"
                save_async = getattr(getattr(self, "writer", None), "save_rollout", None)
                keep_last = 100 if self.env_step < train_cfg.total_steps else None
                if callable(save_async):
                    save_async(rollout_path, keep_last=keep_last)
                else:
                    save_rollout(self.rollout, str(rollout_path))
                if self.env_step < train_cfg.total_steps:
                    self.rollout.clear(keep_last=100)

            if self.check_step("plot") and plot_fcn:
                fig = plot_fcn(self.agent)
                self.video_recorder.capture_frame(fig=fig)

            if done:
                break

        self._finalize_experiment()

    # TODO Update offline_run to match new experiment structure
    def offline_run(self, reset=True):
        if reset:
            if self.writer:
                self.writer.close()
            self.writer = SummaryWriter(log_dir=str(self.results_path / "logs"))
            self.rollout.clear()
            # Check if rollout exists in the results directory
            self.rollout = load_and_concatenate_rollouts(str(self.results_path / "rollouts"))
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

    def run(self, plot_fcn: Callable[[Agent], Figure] | None = None, reset: bool = True):
        self.e_norm = []
        self.e_trace = []

        def _embedding_step_hook(transition: dict) -> None:
            e_bel = self.agent.model.embedding.reshape(-1)
            self.training_info["e"] = e_bel

            e_true = self.agent.env.env.get_params()
            self.writer.add_scalars(
                "e",
                {f"true_{i}": v for i, v in enumerate(to_np(e_true).tolist())},
                self.env_step,
            )
            self.e_trace.append([float(v) for v in to_np(e_bel).reshape(-1).tolist()])
            self.e_norm.append(torch.norm(e_bel.cpu() - e_true.cpu()).item())
            self.training_info["e_norm"] = self.e_norm[-1]

        self._run_online_loop(
            train_cfg=self.cfg.training,
            pbar_desc="Embedding",
            plot_fcn=plot_fcn,
            reset=reset,
            on_step_end=_embedding_step_hook,
        )
