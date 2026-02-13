#!/usr/bin/env python3
"""Process benchmark runs for lightweight latent-state/parameter inference baselines."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "output_dir": "results/benchmark_actdyn",
    "run_name": "benchmark_v1",
    "append_timestamp": True,
    "methods": [
        "baseline-random",
        "baseline-prbs",
        "baseline-ce-mpc",
        "baseline-thompson",
        "baseline-ucb",
    ],
    "environments": ["linear_easy", "linear_shifted"],
    "seeds": [0],
    "episodes_per_env": 1,
    "horizon": 20,
    "action": {
        "low": -1.0,
        "high": 1.0,
    },
    "reward": {
        "action_penalty": 0.02,
    },
    "estimator": {
        "ridge": 1.0,
        "noise_var": 1.0,
    },
    "env_params": {
        "linear_easy": {
            "theta": 0.65,
            "action_gain": 0.7,
            "process_noise": 0.03,
            "observation_noise": 0.03,
        },
        "linear_shifted": {
            "theta": 0.8,
            "theta_shift": 1.05,
            "theta_shift_step": 10,
            "action_gain": 0.7,
            "process_noise": 0.04,
            "observation_noise": 0.05,
        },
    },
    "method_params": {
        "baseline-prbs": {
            "hold_steps": 4,
            "amplitude": 1.0,
        },
        "baseline-ce-mpc": {
            "horizon": 6,
            "num_samples": 48,
            "num_iterations": 3,
            "num_elite": 8,
            "alpha": 0.25,
            "init_std": 0.5,
            "action_penalty": 0.05,
        },
        "baseline-thompson": {
            "prior_var": 0.2,
            "ridge": 1.0,
        },
        "baseline-ucb": {
            "beta": 1.0,
            "ridge": 1.0,
        },
    },
}

METHOD_ALIASES = {
    "random": "baseline-random",
    "prbs": "baseline-prbs",
    "ce-mpc": "baseline-ce-mpc",
    "thompson": "baseline-thompson",
    "ucb": "baseline-ucb",
}

METRIC_FIELDS = [
    "run_id",
    "timestamp_utc",
    "env_name",
    "method",
    "seed",
    "episode",
    "step",
    "theta_true",
    "theta_hat",
    "param_abs_error",
    "latent_abs_error",
    "posterior_var",
    "info_gain",
    "reward",
    "action",
    "action_norm",
    "policy_cost",
    "runtime_ms",
]

EPISODE_FIELDS = [
    "run_id",
    "env_name",
    "method",
    "seed",
    "episode",
    "steps",
    "final_theta_true",
    "final_theta_hat",
    "final_param_abs_error",
    "final_latent_abs_error",
    "mean_info_gain",
    "mean_reward",
    "mean_action_norm",
]


class ActionBox:
    """Minimal action-space utility to avoid external simulator dependencies."""

    def __init__(self, low: float, high: float, seed: int):
        self.low = float(low)
        self.high = float(high)
        self._rng = np.random.default_rng(seed)

    def sample(self) -> float:
        return float(self._rng.uniform(self.low, self.high))

    def clip(self, value: float) -> float:
        return float(np.clip(value, self.low, self.high))


class PlaceholderLatentEnv:
    """Simple scalar latent-state environment used for benchmark smoke runs."""

    def __init__(
        self,
        name: str,
        horizon: int,
        action_low: float,
        action_high: float,
        rng: np.random.Generator,
        **params,
    ):
        self.name = name
        self.horizon = int(horizon)
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self._rng = rng

        self.theta = float(params.get("theta", 0.8))
        self.theta_shift = params.get("theta_shift")
        self.theta_shift_step = int(params.get("theta_shift_step", max(1, self.horizon // 2)))
        self.action_gain = float(params.get("action_gain", 0.7))
        self.process_noise = float(params.get("process_noise", 0.03))
        self.observation_noise = float(params.get("observation_noise", 0.03))

        self._step = 0
        self._state = 0.0

    def reset(self) -> tuple[float, dict[str, float]]:
        self._step = 0
        self._state = float(self._rng.normal(loc=0.0, scale=0.5))
        obs = float(self._state + self._rng.normal(loc=0.0, scale=self.observation_noise))
        return obs, {"latent_state": self._state, "theta_true": self._current_theta()}

    def _current_theta(self) -> float:
        if self.theta_shift is not None and self._step >= self.theta_shift_step:
            return float(self.theta_shift)
        return self.theta

    def step(self, action: float) -> tuple[float, float, bool, bool, dict[str, float]]:
        action_scalar = float(np.clip(action, self.action_low, self.action_high))
        theta = self._current_theta()
        next_state = (
            theta * self._state
            + self.action_gain * action_scalar
            + float(self._rng.normal(loc=0.0, scale=self.process_noise))
        )
        obs = float(next_state + self._rng.normal(loc=0.0, scale=self.observation_noise))

        self._state = next_state
        self._step += 1
        done = self._step >= self.horizon

        info = {
            "latent_state": next_state,
            "theta_true": theta,
        }
        return obs, 0.0, done, False, info


class OnlineThetaEstimator:
    """Ridge estimator for scalar latent dynamics parameter theta."""

    def __init__(self, ridge: float = 1.0, noise_var: float = 1.0):
        self._ridge = max(float(ridge), 1e-6)
        self._noise_var = max(float(noise_var), 1e-8)
        self._sxx = self._ridge
        self._sxy = 0.0
        self.theta_hat = 0.0

    def posterior_var(self) -> float:
        return float(self._noise_var / max(self._sxx, 1e-6))

    def update(self, obs: float, action: float, next_obs: float, action_gain: float) -> float:
        feature = float(obs)
        target = float(next_obs) - float(action_gain) * float(action)

        self._sxx += feature * feature
        self._sxy += feature * target

        self.theta_hat = float(self._sxy / max(self._sxx, 1e-6))
        return self.theta_hat

    def predict_next_obs(self, obs: float, action: float, action_gain: float) -> float:
        return float(self.theta_hat * float(obs) + float(action_gain) * float(action))


class RandomRunner:
    def __init__(self, action_box: ActionBox, rng: np.random.Generator):
        self.action_box = action_box
        self.rng = rng

    def get_action(self, obs: float, posterior_var: float) -> tuple[float, float]:
        del obs, posterior_var
        return self.action_box.sample(), 0.0

    def update(self, obs: float, action: float, reward: float) -> None:
        del obs, action, reward


class PRBSRunner:
    def __init__(
        self,
        action_box: ActionBox,
        rng: np.random.Generator,
        hold_steps: int = 5,
        amplitude: float = 1.0,
    ):
        self.action_box = action_box
        self.rng = rng
        self.hold_steps = max(1, int(hold_steps))
        self.amplitude = float(amplitude)
        self._step = 0
        self._current = 0.0

    def get_action(self, obs: float, posterior_var: float) -> tuple[float, float]:
        del obs, posterior_var
        if self._step % self.hold_steps == 0:
            center = 0.5 * (self.action_box.low + self.action_box.high)
            radius = 0.5 * (self.action_box.high - self.action_box.low)
            sign = 1.0 if self.rng.random() > 0.5 else -1.0
            self._current = self.action_box.clip(center + sign * radius * self.amplitude)
        self._step += 1
        return self._current, 0.0

    def update(self, obs: float, action: float, reward: float) -> None:
        del obs, action, reward


class CEMPCPolicyRunner:
    def __init__(
        self,
        action_box: ActionBox,
        rng: np.random.Generator,
        horizon: int = 6,
        num_samples: int = 48,
        num_iterations: int = 3,
        num_elite: int = 8,
        alpha: float = 0.25,
        init_std: float = 0.5,
        action_penalty: float = 0.05,
    ):
        self.action_box = action_box
        self.rng = rng
        self.horizon = max(1, int(horizon))
        self.num_samples = max(4, int(num_samples))
        self.num_iterations = max(1, int(num_iterations))
        self.num_elite = max(2, min(int(num_elite), self.num_samples))
        self.alpha = float(alpha)
        self.init_std = max(float(init_std), 1e-3)
        self.action_penalty = float(action_penalty)

        center = 0.5 * (self.action_box.low + self.action_box.high)
        spread = max(1e-3, 0.5 * (self.action_box.high - self.action_box.low) * self.init_std)
        self._mean = np.full(self.horizon, center, dtype=np.float64)
        self._std = np.full(self.horizon, spread, dtype=np.float64)

    def _objective(self, action_sequences: np.ndarray, obs: float, posterior_var: float) -> np.ndarray:
        cumulative = np.cumsum(action_sequences, axis=1)
        excitation = np.var(cumulative, axis=1)
        energy = np.mean(action_sequences**2, axis=1)
        state_scale = 1.0 + abs(float(obs))
        uncertainty_scale = 1.0 + float(posterior_var)
        return -(uncertainty_scale * state_scale * excitation) + self.action_penalty * energy

    def get_action(self, obs: float, posterior_var: float) -> tuple[float, float]:
        mean = self._mean.copy()
        std = self._std.copy()

        best_cost = float("inf")
        best_sequence = mean.copy()

        for _ in range(self.num_iterations):
            samples = self.rng.normal(loc=mean, scale=std, size=(self.num_samples, self.horizon))
            samples = np.clip(samples, self.action_box.low, self.action_box.high)

            costs = self._objective(samples, obs=obs, posterior_var=posterior_var)
            elite_idx = np.argpartition(costs, self.num_elite - 1)[: self.num_elite]
            elite = samples[elite_idx]

            new_mean = elite.mean(axis=0)
            new_std = np.maximum(elite.std(axis=0), 1e-3)

            mean = (1.0 - self.alpha) * new_mean + self.alpha * mean
            std = (1.0 - self.alpha) * new_std + self.alpha * std

            idx = int(np.argmin(costs))
            if costs[idx] < best_cost:
                best_cost = float(costs[idx])
                best_sequence = samples[idx]

        shifted = mean.copy()
        shifted[:-1] = mean[1:]
        self._mean = shifted
        self._std = std

        return float(best_sequence[0]), best_cost

    def update(self, obs: float, action: float, reward: float) -> None:
        del obs, action, reward


class ThompsonRunner:
    def __init__(
        self,
        action_box: ActionBox,
        rng: np.random.Generator,
        prior_var: float = 0.2,
        ridge: float = 1.0,
    ):
        self.action_box = action_box
        self.rng = rng
        self.prior_var = float(prior_var)
        self.ridge = float(ridge)

        self._feature_dim = 5
        self._a = np.eye(self._feature_dim, dtype=np.float64) * self.ridge
        self._b = np.zeros(self._feature_dim, dtype=np.float64)

    def _features(self, obs: float, action: float) -> np.ndarray:
        return np.asarray([1.0, obs, action, abs(action), obs * action], dtype=np.float64)

    def _candidates(self) -> np.ndarray:
        return np.asarray([self.action_box.low, 0.0, self.action_box.high], dtype=np.float64)

    def _mean(self) -> np.ndarray:
        return np.linalg.solve(self._a, self._b)

    def get_action(self, obs: float, posterior_var: float) -> tuple[float, float]:
        del posterior_var
        candidates = self._candidates()
        features = np.stack([self._features(obs, a) for a in candidates], axis=0)

        cov = self.prior_var * np.linalg.inv(self._a)
        sampled_theta = self.rng.multivariate_normal(self._mean(), cov)
        scores = features @ sampled_theta

        idx = int(np.argmax(scores))
        return float(candidates[idx]), float(-scores[idx])

    def update(self, obs: float, action: float, reward: float) -> None:
        phi = self._features(obs, action)
        self._a += np.outer(phi, phi)
        self._b += float(reward) * phi


class UCBRunner:
    def __init__(
        self,
        action_box: ActionBox,
        rng: np.random.Generator,
        beta: float = 1.0,
        ridge: float = 1.0,
    ):
        self.action_box = action_box
        self.rng = rng
        self.beta = float(beta)
        self.ridge = float(ridge)

        self._feature_dim = 5
        self._a = np.eye(self._feature_dim, dtype=np.float64) * self.ridge
        self._b = np.zeros(self._feature_dim, dtype=np.float64)

    def _features(self, obs: float, action: float) -> np.ndarray:
        return np.asarray([1.0, obs, action, abs(action), obs * action], dtype=np.float64)

    def _candidates(self) -> np.ndarray:
        return np.asarray([self.action_box.low, 0.0, self.action_box.high], dtype=np.float64)

    def _mean(self) -> np.ndarray:
        return np.linalg.solve(self._a, self._b)

    def get_action(self, obs: float, posterior_var: float) -> tuple[float, float]:
        del posterior_var
        candidates = self._candidates()
        features = np.stack([self._features(obs, a) for a in candidates], axis=0)

        a_inv = np.linalg.inv(self._a)
        mu = self._mean()
        means = features @ mu
        conf = np.sqrt(np.einsum("bi,ij,bj->b", features, a_inv, features))
        scores = means + self.beta * conf

        idx = int(np.argmax(scores))
        return float(candidates[idx]), float(-scores[idx])

    def update(self, obs: float, action: float, reward: float) -> None:
        phi = self._features(obs, action)
        self._a += np.outer(phi, phi)
        self._b += float(reward) * phi


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path | None) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if config_path is None:
        return cfg

    config_file = Path(config_path).expanduser().resolve()
    if not config_file.exists():
        raise FileNotFoundError(f"Benchmark config not found: {config_file}")

    with config_file.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}

    if not isinstance(payload, dict):
        raise ValueError("Benchmark config root must be a mapping")

    return _deep_merge(cfg, payload)


def _normalize_method_name(method: str) -> str:
    method = method.strip().lower()
    return METHOD_ALIASES.get(method, method)


def _build_policy(
    method: str,
    action_box: ActionBox,
    method_params: dict[str, Any],
    rng: np.random.Generator,
) -> Any:
    if method == "baseline-random":
        return RandomRunner(action_box=action_box, rng=rng)
    if method == "baseline-prbs":
        return PRBSRunner(action_box=action_box, rng=rng, **method_params)
    if method == "baseline-ce-mpc":
        return CEMPCPolicyRunner(action_box=action_box, rng=rng, **method_params)
    if method == "baseline-thompson":
        return ThompsonRunner(action_box=action_box, rng=rng, **method_params)
    if method == "baseline-ucb":
        return UCBRunner(action_box=action_box, rng=rng, **method_params)
    raise ValueError(f"Unknown benchmark method: {method}")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_run_dir(config: dict[str, Any], output_dir: str | None, run_name: str | None) -> Path:
    base_output = Path(output_dir or config["output_dir"]).expanduser()
    run_label = run_name or config["run_name"]

    if bool(config.get("append_timestamp", True)):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_label = f"{run_label}_{stamp}"

    run_dir = base_output / run_label
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_benchmark(
    config_path: str | Path | None = None,
    output_dir: str | None = None,
    run_name: str | None = None,
) -> Path:
    config = load_config(config_path)
    run_dir = _resolve_run_dir(config=config, output_dir=output_dir, run_name=run_name)
    run_id = run_dir.name

    metrics_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []

    env_names = [str(name) for name in config["environments"]]
    methods = [_normalize_method_name(str(method)) for method in config["methods"]]
    seeds = [int(seed) for seed in config["seeds"]]

    horizon = int(config["horizon"])
    episodes_per_env = int(config["episodes_per_env"])
    action_low = float(config["action"]["low"])
    action_high = float(config["action"]["high"])
    reward_action_penalty = float(config["reward"]["action_penalty"])

    estimator_ridge = float(config["estimator"]["ridge"])
    estimator_noise_var = float(config["estimator"]["noise_var"])

    for env_name in env_names:
        env_cfg = dict(config.get("env_params", {}).get(env_name, {}))

        for method in methods:
            method_cfg = dict(config.get("method_params", {}).get(method, {}))

            for seed in seeds:
                action_box = ActionBox(low=action_low, high=action_high, seed=seed)

                for episode in range(episodes_per_env):
                    episode_seed = seed + episode * 1000
                    episode_rng = np.random.default_rng(episode_seed)

                    env = PlaceholderLatentEnv(
                        name=env_name,
                        horizon=horizon,
                        action_low=action_low,
                        action_high=action_high,
                        rng=episode_rng,
                        **env_cfg,
                    )
                    policy = _build_policy(
                        method=method,
                        action_box=action_box,
                        method_params=method_cfg,
                        rng=episode_rng,
                    )
                    estimator = OnlineThetaEstimator(
                        ridge=estimator_ridge,
                        noise_var=estimator_noise_var,
                    )

                    obs, info = env.reset()
                    theta_true = float(info["theta_true"])

                    run_step_rows: list[dict[str, Any]] = []

                    for step in range(horizon):
                        step_start = time.perf_counter()

                        action_value, policy_cost = policy.get_action(
                            obs=obs,
                            posterior_var=estimator.posterior_var(),
                        )
                        action_value = action_box.clip(action_value)

                        next_obs, _, done, _, next_info = env.step(action_value)
                        theta_true = float(next_info["theta_true"])

                        prev_var = estimator.posterior_var()
                        theta_hat = estimator.update(
                            obs=obs,
                            action=action_value,
                            next_obs=next_obs,
                            action_gain=env.action_gain,
                        )
                        new_var = estimator.posterior_var()
                        info_gain = max(prev_var - new_var, 0.0)

                        predicted_next = estimator.predict_next_obs(
                            obs=obs,
                            action=action_value,
                            action_gain=env.action_gain,
                        )
                        latent_abs_error = abs(predicted_next - next_obs)
                        param_abs_error = abs(theta_hat - theta_true)
                        reward = info_gain - reward_action_penalty * abs(action_value)

                        policy.update(obs=obs, action=action_value, reward=reward)

                        runtime_ms = (time.perf_counter() - step_start) * 1000.0

                        row = {
                            "run_id": run_id,
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "env_name": env_name,
                            "method": method,
                            "seed": seed,
                            "episode": episode,
                            "step": step,
                            "theta_true": theta_true,
                            "theta_hat": theta_hat,
                            "param_abs_error": param_abs_error,
                            "latent_abs_error": latent_abs_error,
                            "posterior_var": new_var,
                            "info_gain": info_gain,
                            "reward": reward,
                            "action": action_value,
                            "action_norm": abs(action_value),
                            "policy_cost": float(policy_cost),
                            "runtime_ms": runtime_ms,
                        }
                        metrics_rows.append(row)
                        run_step_rows.append(row)

                        obs = next_obs
                        if done:
                            break

                    if not run_step_rows:
                        continue

                    steps = len(run_step_rows)
                    final_row = run_step_rows[-1]
                    episode_row = {
                        "run_id": run_id,
                        "env_name": env_name,
                        "method": method,
                        "seed": seed,
                        "episode": episode,
                        "steps": steps,
                        "final_theta_true": final_row["theta_true"],
                        "final_theta_hat": final_row["theta_hat"],
                        "final_param_abs_error": final_row["param_abs_error"],
                        "final_latent_abs_error": final_row["latent_abs_error"],
                        "mean_info_gain": float(np.mean([r["info_gain"] for r in run_step_rows])),
                        "mean_reward": float(np.mean([r["reward"] for r in run_step_rows])),
                        "mean_action_norm": float(np.mean([r["action_norm"] for r in run_step_rows])),
                    }
                    episode_rows.append(episode_row)

    _write_jsonl(run_dir / "metrics.jsonl", metrics_rows)
    _write_csv(run_dir / "metrics.csv", metrics_rows, METRIC_FIELDS)
    _write_csv(run_dir / "episode_summary.csv", episode_rows, EPISODE_FIELDS)

    metadata = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "n_step_rows": len(metrics_rows),
        "n_episode_rows": len(episode_rows),
        # TODO(FLEX-v2): Add FLEX-specific metadata fields (dataset/version/model registry IDs).
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Benchmark processing complete: {run_dir}")
    print(f"- step metrics: {run_dir / 'metrics.csv'}")
    print(f"- episode summary: {run_dir / 'episode_summary.csv'}")
    return run_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run benchmark processing for actdyn baselines")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "conf" / "config.yaml"),
        help="Path to benchmark config yaml",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--run-name", type=str, default=None, help="Override run label")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    run_benchmark(config_path=args.config, output_dir=args.output_dir, run_name=args.run_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
