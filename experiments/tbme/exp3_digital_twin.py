from __future__ import annotations

import copy
import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from actdyn.config import ExperimentConfig
from actdyn.core.agent import Agent
from actdyn.metrics import metric_from_str
from actdyn.models.model_wrapper import ModelWrapper
from actdyn.policy import policy_from_str
from actdyn.policy.base import BasePolicy, BaseMPC
from actdyn.utils.experiment_helpers import setup_model
from actdyn.utils.rollout import Rollout, RolloutBuffer

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from experiment_common import load_json, resolve_session_root, write_json
else:
    from ..experiment_common import load_json, resolve_session_root, write_json


DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("exp3_digital_twin.yaml")
DEFAULT_BASE_DIR = "results/tbme/exp3"


@dataclass
class DatasetConfig:
    dataset_path: str = "data/mcrtt/mcrtt_replay.npz"
    spike_key: str = "spikes"
    behavior_key: str = "behavior"
    train_fraction: float = 0.7
    sequence_length: int = 128
    sequence_stride: int = 64
    max_units: int | None = 96
    max_train_sequences: int | None = None
    max_eval_sequences: int | None = 64


@dataclass
class GeneratorConfig:
    latent_dim: int = 8
    encoder_type: str = "rnn"
    enc_hidden_dims: list[int] = field(default_factory=lambda: [64])
    enc_rnn_hidden_dims: list[int] = field(default_factory=lambda: [64])
    enc_rnn_type: str = "gru"
    enc_h_init: str = "reset"
    mapping_type: str = "log-linear"
    map_hidden_dims: list[int] = field(default_factory=lambda: [64])
    noise_type: str = "poisson"
    dynamics_type: str = "mlp"
    dyn_hidden_dims: list[int] = field(default_factory=lambda: [128, 128])
    dyn_activation: str = "relu"
    is_residual: bool = False
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 16
    n_epochs: int = 60
    beta: float = 0.1
    n_samples: int = 3
    k_steps: int = 5
    grad_clip_norm: float = 10.0
    p_mask: float = 0.0
    annealing_type: str = "linear"
    annealing_steps: int = 1000
    warmup: int = 10
    behavior_readout_ridge: float = 1e-3


@dataclass
class TwinConfig:
    control_dim: int = 2
    action_low: float = -1.0
    action_high: float = 1.0
    control_scale: float = 0.35
    latent_clip_abs: float | None = 8.0
    stochastic_dynamics: bool = True
    sample_observations: bool = True
    num_initial_latents: int = 256
    probe_num_initial_latents: int = 16
    probe_horizon: int = 25
    probe_action_scale: float = 0.75
    probe_action_seed: int = 123


@dataclass
class LearnerConfig:
    freeze_encoder: bool = True
    freeze_decoder: bool = True
    freeze_action_encoder: bool = True
    dynamics_init_noise_scale: float = 0.2
    inference_window: int = 32
    train_every: int = 5
    min_train_steps: int = 20
    batch_size: int = 16
    n_epochs_per_update: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip_norm: float = 10.0
    n_samples: int = 1
    k_steps: int = 1
    beta: float = 0.1
    p_mask: float = 0.0
    annealing_type: str = "none"
    annealing_steps: int = 1000
    warmup: int = 0


@dataclass
class BenchmarkConfig:
    policy_ids: list[str] = field(
        default_factory=lambda: ["active_myopic", "active_planning", "baseline_prbs", "random"]
    )
    seeds: list[int] = field(default_factory=lambda: [0, 10, 20])
    total_steps: int = 200
    eval_every: int = 10
    metric_type: str = "d-optimality"
    metric_use_diag: bool = True
    metric_discount_factor: float = 0.99
    mpc_num_samples: int = 64
    mpc_num_iterations: int = 5
    mpc_num_elite: int = 8
    mpc_alpha: float = 0.1
    mpc_init_std: float = 0.5
    mpc_noise_beta: float = 1.0
    myopic_horizon: int = 2
    planning_horizon: int = 20
    prbs_hold_steps: int = 5
    prbs_amplitude: float = 1.0
    save_state_action_trace: bool = True


@dataclass
class SummaryConfig:
    figure_formats: list[str] = field(default_factory=lambda: [".pdf"])


@dataclass
class RuntimeConfig:
    seed: int = 0
    device: str = "auto"


@dataclass
class Exp3DigitalTwinConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    twin: TwinConfig = field(default_factory=TwinConfig)
    learner: LearnerConfig = field(default_factory=LearnerConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    summary: SummaryConfig = field(default_factory=SummaryConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Exp3DigitalTwinConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        return cls(
            runtime=RuntimeConfig(**payload.get("runtime", {})),
            dataset=DatasetConfig(**payload.get("dataset", {})),
            generator=GeneratorConfig(**payload.get("generator", {})),
            twin=TwinConfig(**payload.get("twin", {})),
            learner=LearnerConfig(**payload.get("learner", {})),
            benchmark=BenchmarkConfig(**payload.get("benchmark", {})),
            summary=SummaryConfig(**payload.get("summary", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SequenceBundle:
    train_spikes: np.ndarray
    eval_spikes: np.ndarray
    train_behavior: np.ndarray
    eval_behavior: np.ndarray
    dt: float
    metadata: dict[str, Any]


def load_config(path: str | Path | None = None) -> Exp3DigitalTwinConfig:
    config_path = DEFAULT_CONFIG_PATH if path is None else Path(path)
    return Exp3DigitalTwinConfig.from_yaml(config_path)


def _resolve_device(raw: str) -> str:
    if raw != "auto":
        return str(raw)
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (_repo_root() / path).resolve()


def _parse_csv_ints(raw: str | None) -> list[int]:
    if raw is None:
        return []
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def _parse_csv_strs(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _plot_and_save(fig: plt.Figure, stem: Path, figure_formats: Sequence[str]) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in figure_formats:
        suffix = fmt if str(fmt).startswith(".") else f".{fmt}"
        fig.savefig(stem.with_suffix(suffix), bbox_inches="tight")
    plt.close(fig)


def _history_to_rows(history: Any) -> list[dict[str, float]]:
    if isinstance(history, dict):
        elbo = np.asarray(history.get("ELBO", []), dtype=np.float32).reshape(-1)
        log_like = np.asarray(
            history.get("log_L", history.get("log_like", [])),
            dtype=np.float32,
        ).reshape(-1)
        kl = np.asarray(history.get("KL", history.get("kl", [])), dtype=np.float32).reshape(-1)
        n_epochs = max(int(elbo.size), int(log_like.size), int(kl.size))
        rows: list[dict[str, float]] = []
        for idx in range(n_epochs):
            rows.append(
                {
                    "epoch": idx + 1,
                    "neg_elbo": float(elbo[idx]) if idx < elbo.size else float("nan"),
                    "log_like": float(log_like[idx]) if idx < log_like.size else float("nan"),
                    "kl": float(kl[idx]) if idx < kl.size else float("nan"),
                }
            )
        return rows
    rows: list[dict[str, float]] = []
    for idx, item in enumerate(history, start=1):
        values = item.detach().cpu().numpy().reshape(-1) if isinstance(item, torch.Tensor) else np.asarray(item)
        rows.append(
            {
                "epoch": idx,
                "neg_elbo": float(values[0]),
                "log_like": float(values[1]),
                "kl": float(values[2]),
            }
        )
    return rows


def _cap_sequence_count(sequences: np.ndarray, max_items: int | None) -> np.ndarray:
    if max_items is None or sequences.shape[0] <= int(max_items):
        return sequences
    idx = np.linspace(0, sequences.shape[0] - 1, int(max_items), dtype=np.float64)
    picked = np.unique(np.round(idx).astype(np.int64))
    return sequences[picked]


def _sliding_windows(values: np.ndarray, *, length: int, stride: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected rank-2 array for windows, got {arr.shape}")
    if arr.shape[0] < int(length):
        raise ValueError(f"Need at least {length} steps for windows, got {arr.shape[0]}")
    starts = list(range(0, arr.shape[0] - int(length) + 1, max(1, int(stride))))
    return np.stack([arr[start : start + int(length)] for start in starts], axis=0).astype(np.float32)


def build_sequence_bundle(dataset_cfg: DatasetConfig) -> SequenceBundle:
    path = _resolve_path(dataset_cfg.dataset_path)
    with np.load(path, allow_pickle=False) as data:
        spikes = np.asarray(data[dataset_cfg.spike_key], dtype=np.float32)
        behavior = np.asarray(data[dataset_cfg.behavior_key], dtype=np.float32)
        dt = float(np.asarray(data["dt"]).reshape(-1)[0]) if "dt" in data else 0.02
        metadata = {"dataset_path": str(path), "available_keys": [str(key) for key in data.files]}

    if spikes.shape[0] != behavior.shape[0]:
        raise ValueError(f"Spike and behavior arrays must share time axis, got {spikes.shape} vs {behavior.shape}")

    if dataset_cfg.max_units is not None and spikes.shape[1] > int(dataset_cfg.max_units):
        order = np.argsort(np.var(spikes, axis=0))[::-1][: int(dataset_cfg.max_units)]
        spikes = spikes[:, order]
        metadata["selected_unit_indices"] = [int(idx) for idx in order.tolist()]

    split = int(np.clip(round(spikes.shape[0] * float(dataset_cfg.train_fraction)), 8, spikes.shape[0] - 8))
    train_spikes = spikes[:split]
    eval_spikes = spikes[split:]
    train_behavior = behavior[:split]
    eval_behavior = behavior[split:]

    train_spike_seq = _sliding_windows(
        train_spikes,
        length=int(dataset_cfg.sequence_length),
        stride=int(dataset_cfg.sequence_stride),
    )
    eval_spike_seq = _sliding_windows(
        eval_spikes,
        length=int(dataset_cfg.sequence_length),
        stride=int(dataset_cfg.sequence_stride),
    )
    train_behavior_seq = _sliding_windows(
        train_behavior,
        length=int(dataset_cfg.sequence_length),
        stride=int(dataset_cfg.sequence_stride),
    )
    eval_behavior_seq = _sliding_windows(
        eval_behavior,
        length=int(dataset_cfg.sequence_length),
        stride=int(dataset_cfg.sequence_stride),
    )

    train_spike_seq = _cap_sequence_count(train_spike_seq, dataset_cfg.max_train_sequences)
    eval_spike_seq = _cap_sequence_count(eval_spike_seq, dataset_cfg.max_eval_sequences)
    train_behavior_seq = train_behavior_seq[: train_spike_seq.shape[0]]
    eval_behavior_seq = eval_behavior_seq[: eval_spike_seq.shape[0]]

    metadata.update(
        {
            "train_sequence_count": int(train_spike_seq.shape[0]),
            "eval_sequence_count": int(eval_spike_seq.shape[0]),
            "observation_dim": int(train_spike_seq.shape[-1]),
            "behavior_dim": int(train_behavior_seq.shape[-1]),
            "sequence_length": int(dataset_cfg.sequence_length),
            "sequence_stride": int(dataset_cfg.sequence_stride),
            "dt": float(dt),
        }
    )
    return SequenceBundle(
        train_spikes=train_spike_seq,
        eval_spikes=eval_spike_seq,
        train_behavior=train_behavior_seq,
        eval_behavior=eval_behavior_seq,
        dt=float(dt),
        metadata=metadata,
    )


def _build_rollout_buffer(sequences: np.ndarray, *, action_dim: int = 0) -> RolloutBuffer:
    buffer = RolloutBuffer(device="cpu")
    for seq in sequences:
        rollout = Rollout(device="cpu")
        payload: dict[str, Any] = {"next_obs": seq.astype(np.float32, copy=False)}
        if int(action_dim) > 0:
            payload["action"] = np.zeros((seq.shape[0], int(action_dim)), dtype=np.float32)
        rollout.add(**payload)
        rollout.finalize()
        buffer.add_rollout(rollout)
    return buffer


def _build_seqvae_config(
    *,
    obs_dim: int,
    latent_dim: int,
    generator_cfg: GeneratorConfig,
    device: str,
    action_dim: int,
    action_low: float = -1.0,
    action_high: float = 1.0,
    action_type: str = "identity",
) -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.device = str(device)
    cfg.dt = 1.0
    cfg.observation_dim = int(obs_dim)
    cfg.latent_dim = int(latent_dim)
    cfg.action_dim = int(action_dim)
    cfg.environment.env_action_bounds = [float(action_low), float(action_high)]
    cfg.model.model_type = "seq-vae"
    cfg.model.encoder_type = str(generator_cfg.encoder_type)
    cfg.model.enc_hidden_dims = list(generator_cfg.enc_hidden_dims)
    cfg.model.enc_rnn_hidden_dims = list(generator_cfg.enc_rnn_hidden_dims)
    cfg.model.enc_rnn_type = str(generator_cfg.enc_rnn_type)
    cfg.model.enc_h_init = str(generator_cfg.enc_h_init)
    cfg.model.mapping_type = str(generator_cfg.mapping_type)
    cfg.model.map_hidden_dims = list(generator_cfg.map_hidden_dims)
    cfg.model.noise_type = str(generator_cfg.noise_type)
    cfg.model.dynamics_type = str(generator_cfg.dynamics_type)
    cfg.model.dyn_hidden_dims = list(generator_cfg.dyn_hidden_dims)
    cfg.model.dyn_activation = str(generator_cfg.dyn_activation)
    cfg.model.is_residual = bool(generator_cfg.is_residual)
    cfg.model.dyn_dt = 1.0
    cfg.model.action_type = str(action_type)
    return cfg


def _set_action_encoder_weights(model: Any, control_matrix: np.ndarray) -> None:
    if model.action_encoder is None or not hasattr(model.action_encoder, "network"):
        raise ValueError("Learner model does not expose a configurable action encoder")
    layer = model.action_encoder.network
    if not isinstance(layer, torch.nn.Linear):
        raise ValueError("This benchmark expects a linear action encoder")
    with torch.no_grad():
        layer.weight.copy_(torch.as_tensor(control_matrix, dtype=layer.weight.dtype, device=layer.weight.device))
        layer.bias.zero_()


def _freeze_module(module: torch.nn.Module | None) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = False


def _encode_means(model: Any, obs: np.ndarray, *, device: str) -> np.ndarray:
    y = torch.as_tensor(obs, dtype=torch.float32, device=device)
    with torch.no_grad():
        _samples, mu, _var = model.encoder(y=y, u=None, n_samples=1)
    return mu.detach().cpu().numpy().astype(np.float32, copy=False)


def _next_latent_mean(model: Any, z: torch.Tensor, action: torch.Tensor | None = None) -> torch.Tensor:
    pred = model.dynamics(z)
    if bool(getattr(model.dynamics, "is_residual", False)):
        pred = z + pred * float(getattr(model.dynamics, "dt", 1.0))
    if action is not None and action.shape[-1] > 0:
        pred = pred + action * float(getattr(model.dynamics, "dt", 1.0))
    return pred


def _sanitize_latent_tensor(latent: torch.Tensor, *, clip_abs: float | None = None) -> torch.Tensor:
    cleaned = torch.nan_to_num(latent, nan=0.0, posinf=1e3, neginf=-1e3)
    if clip_abs is not None and float(clip_abs) > 0:
        cleaned = cleaned.clamp(min=-float(clip_abs), max=float(clip_abs))
    return cleaned


def _sanitize_rate_tensor(rate: torch.Tensor, *, min_rate: float = 1e-6, max_rate: float = 1e3) -> torch.Tensor:
    return torch.nan_to_num(rate, nan=min_rate, posinf=max_rate, neginf=min_rate).clamp(
        min=min_rate,
        max=max_rate,
    )


def _poisson_nll(rate: np.ndarray, counts: np.ndarray) -> float:
    rate_t = torch.as_tensor(rate, dtype=torch.float64)
    counts_t = torch.as_tensor(counts, dtype=torch.float64)
    nll = rate_t - counts_t * torch.log(rate_t + 1e-8) + torch.lgamma(counts_t + 1.0)
    return float(torch.mean(nll).cpu())


def _mse(x: np.ndarray, y: np.ndarray) -> float:
    delta = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    return float(np.mean(delta * delta))


def _r2(x: np.ndarray, y: np.ndarray) -> float:
    pred = np.asarray(x, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64)
    ss_res = float(np.sum((pred - target) ** 2))
    ss_tot = float(np.sum((target - np.mean(target, axis=0, keepdims=True)) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def _flatten_behavior_sequences(sequences: np.ndarray) -> np.ndarray:
    return np.asarray(sequences, dtype=np.float32).reshape(-1, sequences.shape[-1])


def _fit_ridge_with_bias(x: np.ndarray, y: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    aug = np.concatenate([x_arr, np.ones((x_arr.shape[0], 1), dtype=np.float64)], axis=1)
    gram = aug.T @ aug + float(ridge) * np.eye(aug.shape[1], dtype=np.float64)
    coef = np.linalg.solve(gram, aug.T @ y_arr)
    weight = coef[:-1].astype(np.float32, copy=False)
    bias = coef[-1].astype(np.float32, copy=False)
    return weight, bias


def _predict_ridge_with_bias(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32) @ np.asarray(weight, dtype=np.float32) + np.asarray(bias, dtype=np.float32)


def _calibrate_control_matrix(
    *,
    model: Any,
    latent_points: np.ndarray,
    control_dim: int,
    control_scale: float,
    seed: int,
    device: str,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    latent_dim = int(latent_points.shape[-1])
    raw = rng.normal(size=(latent_dim, int(control_dim))).astype(np.float32)
    q, _ = np.linalg.qr(raw)
    basis = q[:, : int(control_dim)].astype(np.float32)
    z = torch.as_tensor(latent_points[: min(256, latent_points.shape[0])], dtype=torch.float32, device=device).unsqueeze(1)
    with torch.no_grad():
        next_z = _next_latent_mean(model, z).squeeze(1)
    delta = (next_z - z.squeeze(1)).detach().cpu().numpy().astype(np.float32, copy=False)
    median_step = float(np.median(np.linalg.norm(delta, axis=-1)))
    scale = max(1e-2, float(control_scale) * median_step)
    return (basis * scale).astype(np.float32, copy=False)


def _build_probe_actions(
    *,
    num_initial_latents: int,
    horizon: int,
    control_dim: int,
    low: float,
    high: float,
    scale: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(int(num_initial_latents), int(horizon), int(control_dim))).astype(np.float32)
    values = np.tanh(values) * float(scale)
    return np.clip(values, float(low), float(high)).astype(np.float32, copy=False)


def _flatten_parameters(module: torch.nn.Module) -> torch.Tensor:
    parts = [param.detach().reshape(-1) for param in module.parameters()]
    return torch.cat(parts) if parts else torch.zeros(1)


def _relative_param_error(reference: torch.nn.Module, estimate: torch.nn.Module) -> float:
    ref = _flatten_parameters(reference).cpu()
    est = _flatten_parameters(estimate).cpu()
    denom = float(torch.linalg.norm(ref)) + 1e-8
    return float(torch.linalg.norm(est - ref) / denom)


def _rollout_rates(
    *,
    model: Any,
    init_latents: torch.Tensor,
    raw_actions: torch.Tensor,
    control_matrix: np.ndarray | None,
    device: str,
    latent_clip_abs: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    z = init_latents.to(device)
    if z.ndim == 2:
        z = z.unsqueeze(1)
    z = _sanitize_latent_tensor(z, clip_abs=latent_clip_abs)
    if control_matrix is not None:
        control = torch.as_tensor(control_matrix, dtype=torch.float32, device=device)
        encoded_actions = torch.einsum("bta,za->btz", raw_actions.to(device), control)
    else:
        encoded_actions = model.action_encoder(raw_actions.to(device)) if model.action_encoder is not None else None
    with torch.no_grad():
        if encoded_actions is not None and encoded_actions.shape[-1] == 0:
            encoded_actions = None
        _samples, mus, _vars = model.dynamics.sample_forward(
            init_z=z,
            action=encoded_actions,
            k_step=int(raw_actions.shape[1]),
            return_traj=True,
            add_noise=False,
        )
        latent = _sanitize_latent_tensor(torch.cat(mus, dim=1), clip_abs=latent_clip_abs)
        rates = _sanitize_rate_tensor(model.decoder(latent))
    return (
        latent.detach().cpu().numpy().astype(np.float32, copy=False),
        rates.detach().cpu().numpy().astype(np.float32, copy=False),
    )


def _evaluate_generator(
    *,
    model: Any,
    bundle: SequenceBundle,
    cfg: GeneratorConfig,
    device: str,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    y = torch.as_tensor(bundle.eval_spikes, dtype=torch.float32, device=device)
    with torch.no_grad():
        loss, log_like, kl = model.compute_elbo(
            y,
            u=None,
            n_samples=int(cfg.n_samples),
            k_steps=int(cfg.k_steps),
            beta=float(cfg.beta),
            p_mask=float(cfg.p_mask),
        )
        _samples, mu, _var = model.encoder(y=y, u=None, n_samples=1)
        next_latent = _next_latent_mean(model, mu[:, :-1, :])
        one_step_rate = _sanitize_rate_tensor(model.decoder(next_latent))

    target = bundle.eval_spikes[:, 1:, :]
    pred = one_step_rate.detach().cpu().numpy().astype(np.float32, copy=False)
    latent = mu.detach().cpu().numpy().astype(np.float32, copy=False)
    metrics = {
        "eval_neg_elbo": float(loss.detach().cpu()),
        "eval_log_like": float(log_like.detach().cpu()),
        "eval_kl": float(kl.detach().cpu()),
        "one_step_count_mse": _mse(pred, target),
        "one_step_count_r2": _r2(pred, target),
        "one_step_poisson_nll": _poisson_nll(pred, target),
    }
    return metrics, {"posterior_latent": latent, "one_step_rate": pred, "one_step_target": target}


def _evaluate_behavior_readout(
    *,
    model: Any,
    bundle: SequenceBundle,
    ridge: float,
    device: str,
) -> dict[str, float]:
    train_latent = _encode_means(model, bundle.train_spikes, device=device)
    eval_latent = _encode_means(model, bundle.eval_spikes, device=device)
    x_train = _flatten_behavior_sequences(train_latent)
    x_eval = _flatten_behavior_sequences(eval_latent)
    y_train = _flatten_behavior_sequences(bundle.train_behavior)
    y_eval = _flatten_behavior_sequences(bundle.eval_behavior)
    weight, bias = _fit_ridge_with_bias(x_train, y_train, ridge=float(ridge))
    pred = _predict_ridge_with_bias(x_eval, weight, bias)
    return {
        "behavior_readout_mse": _mse(pred, y_eval),
        "behavior_readout_r2": _r2(pred, y_eval),
    }


class NeuralDigitalTwinEnv(gym.Env):
    def __init__(
        self,
        *,
        generator: Any,
        initial_latents: np.ndarray,
        control_matrix: np.ndarray,
        action_low: float,
        action_high: float,
        latent_clip_abs: float | None,
        stochastic_dynamics: bool,
        sample_observations: bool,
        device: str,
    ) -> None:
        super().__init__()
        self.generator = generator
        self.device = torch.device(device)
        self.initial_latents = torch.as_tensor(initial_latents, dtype=torch.float32, device=self.device)
        self.control_matrix = torch.as_tensor(control_matrix, dtype=torch.float32, device=self.device)
        self.latent_dim = int(self.initial_latents.shape[-1])
        self.control_dim = int(self.control_matrix.shape[-1])
        self.observation_dim = int(generator.decoder.obs_dim)
        self.action_space = gym.spaces.Box(
            low=float(action_low),
            high=float(action_high),
            shape=(self.control_dim,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )
        self.stochastic_dynamics = bool(stochastic_dynamics)
        self.sample_observations = bool(sample_observations)
        self.latent_clip_abs = None if latent_clip_abs is None else float(latent_clip_abs)
        self._rng = np.random.default_rng(0)
        self._state = torch.zeros(1, 1, self.latent_dim, dtype=torch.float32, device=self.device)

    def _to_tensor(self, action: torch.Tensor | np.ndarray | Sequence[float]) -> torch.Tensor:
        act = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if act.ndim == 1:
            act = act.view(1, 1, -1)
        elif act.ndim == 2:
            act = act.unsqueeze(0)
        return act

    def _encode_action(self, action: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bta,za->btz", action, self.control_matrix)

    def _sample_obs(self, z: torch.Tensor) -> torch.Tensor:
        rate = _sanitize_rate_tensor(self.generator.decoder(z))
        if self.sample_observations:
            return torch.poisson(rate)
        return rate

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del options
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
            self.action_space.seed(int(seed))
        idx = int(self._rng.integers(0, self.initial_latents.shape[0]))
        self._state = _sanitize_latent_tensor(
            self.initial_latents[idx : idx + 1].unsqueeze(1).clone(),
            clip_abs=self.latent_clip_abs,
        )
        obs = self._sample_obs(self._state)
        zeros = torch.zeros((1, 1, self.latent_dim), dtype=torch.float32, device=self.device)
        return obs, {"latent_state": self._state.clone(), "env_action": zeros}

    def step(self, action):
        act = self._to_tensor(action)
        env_action = self._encode_action(act)
        with torch.no_grad():
            next_state, _mu, _var = self.generator.dynamics.sample_forward(
                init_z=self._state,
                action=env_action,
                k_step=1,
                add_noise=bool(self.stochastic_dynamics),
                return_traj=False,
            )
            next_state = _sanitize_latent_tensor(next_state, clip_abs=self.latent_clip_abs)
            obs = self._sample_obs(next_state)
        self._state = next_state.detach()
        reward = torch.zeros(1, device=self.device).squeeze()
        terminated = torch.tensor(False, device=self.device)
        truncated = torch.tensor(False, device=self.device)
        info = {"latent_state": self._state.clone(), "env_action": env_action.detach()}
        return obs, reward, terminated, truncated, info


class ClippedModelWrapper(ModelWrapper):
    def __init__(
        self,
        model: Any,
        observation_space: gym.Space,
        action_space: gym.Space,
        *,
        latent_clip_abs: float | None,
        device: str,
    ) -> None:
        super().__init__(model=model, observation_space=observation_space, action_space=action_space, device=device)
        self.latent_clip_abs = None if latent_clip_abs is None else float(latent_clip_abs)

    def reset(self, observation: torch.Tensor):
        out = super().reset(observation)
        self._state = _sanitize_latent_tensor(self._state, clip_abs=self.latent_clip_abs)
        self.model.set_state(self._state)
        return out

    def set_state(self, state: torch.Tensor):
        clipped = _sanitize_latent_tensor(state, clip_abs=self.latent_clip_abs)
        super().set_state(clipped)

    def predict(self, action: torch.Tensor) -> torch.Tensor:
        pred = self.model.predict(action)
        return _sanitize_latent_tensor(pred, clip_abs=self.latent_clip_abs)

    def step(self, action: torch.Tensor):
        next_observation, reward, terminated, truncated, info = super().step(action)
        self._state = _sanitize_latent_tensor(self._state, clip_abs=self.latent_clip_abs)
        self.model.set_state(self._state)
        info["latent_state"] = self._state
        return next_observation, reward, terminated, truncated, info


def _fit_generator(
    *,
    session_root: Path,
    config: Exp3DigitalTwinConfig,
    device: str,
) -> tuple[Any, SequenceBundle, np.ndarray, np.ndarray, np.ndarray]:
    bundle = build_sequence_bundle(config.dataset)
    obs_dim = int(bundle.train_spikes.shape[-1])
    generator_cfg = _build_seqvae_config(
        obs_dim=obs_dim,
        latent_dim=int(config.generator.latent_dim),
        generator_cfg=config.generator,
        device=device,
        action_dim=0,
        action_type="identity",
    )
    model = setup_model(generator_cfg)
    train_rollout = _build_rollout_buffer(bundle.train_spikes, action_dim=0)
    history = model.train_model(
        train_rollout,
        batch_size=int(config.generator.batch_size),
        shuffle=True,
        optimizer="AdamW",
        lr=float(config.generator.learning_rate),
        weight_decay=float(config.generator.weight_decay),
        n_epochs=int(config.generator.n_epochs),
        verbose=False,
        grad_clip_norm=float(config.generator.grad_clip_norm),
        n_samples=int(config.generator.n_samples),
        k_steps=int(config.generator.k_steps),
        beta=float(config.generator.beta),
        p_mask=float(config.generator.p_mask),
        annealing_type=str(config.generator.annealing_type),
        annealing_steps=int(config.generator.annealing_steps),
        warmup=int(config.generator.warmup),
        param_list="all",
    )
    history_rows = _history_to_rows(history)
    fit_metrics, eval_artifacts = _evaluate_generator(model=model, bundle=bundle, cfg=config.generator, device=device)
    behavior_metrics = _evaluate_behavior_readout(
        model=model,
        bundle=bundle,
        ridge=float(config.generator.behavior_readout_ridge),
        device=device,
    )
    train_latents = _encode_means(model, bundle.train_spikes, device=device)
    eval_latents = _encode_means(model, bundle.eval_spikes, device=device)
    initial_latents = train_latents[:, 0, :]
    probe_latents = eval_latents[:, 0, :]
    initial_latents = _cap_sequence_count(initial_latents, config.twin.num_initial_latents)
    probe_latents = _cap_sequence_count(probe_latents, config.twin.probe_num_initial_latents)
    control_matrix = _calibrate_control_matrix(
        model=model,
        latent_points=initial_latents,
        control_dim=int(config.twin.control_dim),
        control_scale=float(config.twin.control_scale),
        seed=int(config.runtime.seed),
        device=device,
    )
    probe_actions = _build_probe_actions(
        num_initial_latents=int(probe_latents.shape[0]),
        horizon=int(config.twin.probe_horizon),
        control_dim=int(config.twin.control_dim),
        low=float(config.twin.action_low),
        high=float(config.twin.action_high),
        scale=float(config.twin.probe_action_scale),
        seed=int(config.twin.probe_action_seed),
    )

    generator_dir = session_root / "generator"
    figures_dir = generator_dir / "figures"
    generator_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(generator_dir / "checkpoint.pt"))
    write_json(
        generator_dir / "fit_metrics.json",
        {**fit_metrics, **behavior_metrics},
    )
    write_json(
        generator_dir / "fit_config.json",
        {
            "observation_dim": obs_dim,
            "behavior_dim": int(bundle.train_behavior.shape[-1]),
            "latent_dim": int(config.generator.latent_dim),
            "control_dim": int(config.twin.control_dim),
            "config": config.to_dict(),
        },
    )
    _write_csv(generator_dir / "train_history.csv", history_rows, ["epoch", "neg_elbo", "log_like", "kl"])
    np.savez_compressed(
        generator_dir / "eval_artifacts.npz",
        posterior_latent=eval_artifacts["posterior_latent"],
        one_step_rate=eval_artifacts["one_step_rate"],
        one_step_target=eval_artifacts["one_step_target"],
        initial_latents=initial_latents,
        probe_latents=probe_latents,
        probe_actions=probe_actions,
        control_matrix=control_matrix,
    )

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot([row["epoch"] for row in history_rows], [row["neg_elbo"] for row in history_rows], label="neg ELBO")
    ax.plot([row["epoch"] for row in history_rows], [row["log_like"] for row in history_rows], label="log-like")
    ax.plot([row["epoch"] for row in history_rows], [row["kl"] for row in history_rows], label="KL")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_title("Spike SeqVAE training history")
    ax.legend(loc="best")
    _plot_and_save(fig, figures_dir / "training_history", config.summary.figure_formats)

    idx = 0
    target = eval_artifacts["one_step_target"][idx]
    pred = eval_artifacts["one_step_rate"][idx]
    fig, axes = plt.subplots(min(4, target.shape[-1]), 1, figsize=(9.0, 6.5), sharex=True)
    axes = np.atleast_1d(axes)
    for dim, ax in enumerate(axes):
        ax.plot(target[:, dim], label="spikes", linewidth=1.5)
        ax.plot(pred[:, dim], label="rate pred", linewidth=1.2)
        ax.set_ylabel(f"unit {dim}")
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("Step")
    fig.suptitle("Representative held-out one-step spike prediction")
    _plot_and_save(fig, figures_dir / "representative_one_step_prediction", config.summary.figure_formats)

    return model, bundle, initial_latents, probe_latents, probe_actions


def _load_generator_assets(
    *,
    session_root: Path,
    device: str,
) -> tuple[Any, dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    generator_dir = session_root / "generator"
    payload = load_json(generator_dir / "fit_config.json")
    cfg_payload = payload["config"]
    config = Exp3DigitalTwinConfig(
        runtime=RuntimeConfig(**cfg_payload["runtime"]),
        dataset=DatasetConfig(**cfg_payload["dataset"]),
        generator=GeneratorConfig(**cfg_payload["generator"]),
        twin=TwinConfig(**cfg_payload["twin"]),
        learner=LearnerConfig(**cfg_payload["learner"]),
        benchmark=BenchmarkConfig(**cfg_payload["benchmark"]),
        summary=SummaryConfig(**cfg_payload["summary"]),
    )
    obs_dim = int(payload["observation_dim"])
    model_cfg = _build_seqvae_config(
        obs_dim=obs_dim,
        latent_dim=int(config.generator.latent_dim),
        generator_cfg=config.generator,
        device=device,
        action_dim=0,
        action_type="identity",
    )
    model = setup_model(model_cfg)
    model.load(str(generator_dir / "checkpoint.pt"))
    with np.load(generator_dir / "eval_artifacts.npz", allow_pickle=False) as data:
        initial_latents = np.asarray(data["initial_latents"], dtype=np.float32)
        probe_latents = np.asarray(data["probe_latents"], dtype=np.float32)
        probe_actions = np.asarray(data["probe_actions"], dtype=np.float32)
        control_matrix = np.asarray(data["control_matrix"], dtype=np.float32)
    return model, payload, initial_latents, probe_latents, probe_actions, control_matrix


def _build_learner(
    *,
    generator: Any,
    control_matrix: np.ndarray,
    config: Exp3DigitalTwinConfig,
    device: str,
) -> Any:
    learner_cfg = _build_seqvae_config(
        obs_dim=int(generator.decoder.obs_dim),
        latent_dim=int(config.generator.latent_dim),
        generator_cfg=config.generator,
        device=device,
        action_dim=int(config.twin.control_dim),
        action_low=float(config.twin.action_low),
        action_high=float(config.twin.action_high),
        action_type="linear",
    )
    learner = setup_model(learner_cfg)
    learner.encoder = copy.deepcopy(generator.encoder).to(device)
    learner.decoder = copy.deepcopy(generator.decoder).to(device)
    learner.dynamics.load_state_dict(generator.dynamics.state_dict())
    _set_action_encoder_weights(learner, control_matrix)
    noise_scale = float(config.learner.dynamics_init_noise_scale)
    if noise_scale > 0:
        with torch.no_grad():
            for param in learner.dynamics.parameters():
                param.add_(noise_scale * torch.randn_like(param))
    if config.learner.freeze_encoder:
        _freeze_module(learner.encoder)
    if config.learner.freeze_decoder:
        _freeze_module(learner.decoder)
    if config.learner.freeze_action_encoder:
        _freeze_module(learner.action_encoder)
    return learner


def _build_policy(
    *,
    policy_id: str,
    model: Any,
    benchmark_cfg: BenchmarkConfig,
    seed: int,
    device: str,
) -> BasePolicy:
    if policy_id in {"active_myopic", "active_planning"}:
        metric_cls = metric_from_str(str(benchmark_cfg.metric_type))
        metric = metric_cls(
            model=model,
            compute_type="sum",
            use_diag=bool(benchmark_cfg.metric_use_diag),
            discount_factor=float(benchmark_cfg.metric_discount_factor),
            covariance="invariant",
            sensitivity=False,
            device=device,
        )
        horizon = int(benchmark_cfg.myopic_horizon if policy_id == "active_myopic" else benchmark_cfg.planning_horizon)
        return policy_from_str("mpc-icem")(
            metric=metric,
            model=model,
            horizon=horizon,
            num_samples=int(benchmark_cfg.mpc_num_samples),
            num_iterations=int(benchmark_cfg.mpc_num_iterations),
            num_elite=int(benchmark_cfg.mpc_num_elite),
            alpha=float(benchmark_cfg.mpc_alpha),
            init_std=float(benchmark_cfg.mpc_init_std),
            noise_beta=float(benchmark_cfg.mpc_noise_beta),
            device=device,
        )
    if policy_id == "baseline_prbs":
        return policy_from_str("baseline-prbs")(
            action_space=model.action_encoder.action_space,
            hold_steps=int(benchmark_cfg.prbs_hold_steps),
            amplitude=float(benchmark_cfg.prbs_amplitude),
            seed=int(seed),
            device=device,
        )
    if policy_id == "random":
        return policy_from_str("random")(action_space=model.action_encoder.action_space, device=device)
    raise ValueError(f"Unsupported benchmark policy_id={policy_id}")


def _evaluate_identification(
    *,
    generator: Any,
    learner: Any,
    probe_latents: np.ndarray,
    probe_actions: np.ndarray,
    control_matrix: np.ndarray,
    latent_clip_abs: float | None,
    device: str,
) -> dict[str, float]:
    init_latents = torch.as_tensor(probe_latents, dtype=torch.float32, device=device).unsqueeze(1)
    raw_actions = torch.as_tensor(probe_actions, dtype=torch.float32, device=device)
    gen_latent, gen_rate = _rollout_rates(
        model=generator,
        init_latents=init_latents,
        raw_actions=raw_actions,
        control_matrix=control_matrix,
        latent_clip_abs=latent_clip_abs,
        device=device,
    )
    learner_latent, learner_rate = _rollout_rates(
        model=learner,
        init_latents=init_latents,
        raw_actions=raw_actions,
        control_matrix=None,
        latent_clip_abs=latent_clip_abs,
        device=device,
    )
    return {
        "param_error": _relative_param_error(generator.dynamics, learner.dynamics),
        "latent_rollout_mse": _mse(learner_latent, gen_latent),
        "rate_rollout_mse": _mse(learner_rate, gen_rate),
        "rate_rollout_r2": _r2(learner_rate, gen_rate),
    }


def _fit_learner_dynamics(
    *,
    learner: Any,
    rollout: Rollout,
    config: Exp3DigitalTwinConfig,
    device: str,
) -> dict[str, float]:
    params = [param for param in learner.dynamics.parameters() if param.requires_grad]
    if not params:
        return {}

    if not rollout.finalized:
        rollout.finalize()

    obs = rollout["obs"].to(device).float()
    next_obs = rollout["next_obs"].to(device).float()
    actions = rollout["action"].to(device).float() if "action" in rollout.as_dict() else None

    optimizer = torch.optim.AdamW(
        params,
        lr=float(config.learner.learning_rate),
        weight_decay=float(config.learner.weight_decay),
    )
    last_loss = torch.tensor(0.0, device=device)
    for _ in range(int(config.learner.n_epochs_per_update)):
        optimizer.zero_grad()
        with torch.no_grad():
            _samples, z_curr, _var = learner.encoder(y=obs, u=None, n_samples=1)
            _samples_next, z_next, _var_next = learner.encoder(y=next_obs, u=None, n_samples=1)
            if actions is not None and learner.action_encoder is not None:
                action_enc = learner.action_encoder(actions, z_curr)
            else:
                action_enc = None
        pred_next = _next_latent_mean(learner, z_curr, action_enc)
        pred_next = _sanitize_latent_tensor(pred_next, clip_abs=config.twin.latent_clip_abs)
        last_loss = torch.mean((pred_next - z_next) ** 2)
        last_loss.backward()
        if float(config.learner.grad_clip_norm) > 0:
            torch.nn.utils.clip_grad_norm_(params, float(config.learner.grad_clip_norm))
        optimizer.step()

    return {"train_latent_mse": float(last_loss.detach().cpu())}


def _run_single_benchmark(
    *,
    session_root: Path,
    policy_id: str,
    seed: int,
    generator: Any,
    initial_latents: np.ndarray,
    probe_latents: np.ndarray,
    probe_actions: np.ndarray,
    control_matrix: np.ndarray,
    config: Exp3DigitalTwinConfig,
    device: str,
) -> dict[str, float]:
    _set_seed(int(seed))
    learner = _build_learner(generator=generator, control_matrix=control_matrix, config=config, device=device)
    env = NeuralDigitalTwinEnv(
        generator=generator,
        initial_latents=initial_latents,
        control_matrix=control_matrix,
        action_low=float(config.twin.action_low),
        action_high=float(config.twin.action_high),
        latent_clip_abs=config.twin.latent_clip_abs,
        stochastic_dynamics=bool(config.twin.stochastic_dynamics),
        sample_observations=bool(config.twin.sample_observations),
        device=device,
    )
    model_env = ClippedModelWrapper(
        learner,
        env.observation_space,
        env.action_space,
        latent_clip_abs=config.twin.latent_clip_abs,
        device=device,
    )
    policy = _build_policy(
        policy_id=policy_id,
        model=model_env,
        benchmark_cfg=config.benchmark,
        seed=int(seed),
        device=device,
    )
    agent = Agent(
        env=env,
        model=model_env,
        policy=policy,
        buffer_length=int(config.learner.inference_window),
        device=device,
    )
    agent.reset(seed=int(seed))
    full_rollout = Rollout(device="cpu")
    rows: list[dict[str, Any]] = []
    state_action_rows: list[dict[str, Any]] = []
    last_train: dict[str, float] = {}

    initial_eval = _evaluate_identification(
        generator=generator,
        learner=learner,
        probe_latents=probe_latents,
        probe_actions=probe_actions,
        control_matrix=control_matrix,
        latent_clip_abs=config.twin.latent_clip_abs,
        device=device,
    )
    rows.append({"step": 0, **initial_eval, "policy_cost": 0.0})

    for step in range(1, int(config.benchmark.total_steps) + 1):
        action = agent.plan()
        transition, _done = agent.step(action)
        full_rollout.add(**transition)
        if bool(config.benchmark.save_state_action_trace):
            env_state = transition["env_state"].detach().cpu().numpy().reshape(-1)
            next_state = transition["next_env_state"].detach().cpu().numpy().reshape(-1)
            raw_action = transition["action"].detach().cpu().numpy().reshape(-1)
            state_action_rows.append(
                {
                    "step": step,
                    "z0": float(env_state[0]) if env_state.size > 0 else 0.0,
                    "z1": float(env_state[1]) if env_state.size > 1 else 0.0,
                    "next_z0": float(next_state[0]) if next_state.size > 0 else 0.0,
                    "next_z1": float(next_state[1]) if next_state.size > 1 else 0.0,
                    "action0": float(raw_action[0]) if raw_action.size > 0 else 0.0,
                    "action1": float(raw_action[1]) if raw_action.size > 1 else 0.0,
                }
            )

        if step >= int(config.learner.min_train_steps) and step % int(config.learner.train_every) == 0:
            train_rollout = full_rollout.copy()
            train_rollout.finalize()
            last_train = _fit_learner_dynamics(
                learner=learner,
                rollout=train_rollout,
                config=config,
                device=device,
            )

        if step % int(config.benchmark.eval_every) == 0 or step == int(config.benchmark.total_steps):
            metrics = _evaluate_identification(
                generator=generator,
                learner=learner,
                probe_latents=probe_latents,
                probe_actions=probe_actions,
                control_matrix=control_matrix,
                latent_clip_abs=config.twin.latent_clip_abs,
                device=device,
            )
            rows.append(
                {
                    "step": step,
                    **metrics,
                    "policy_cost": float(getattr(policy, "cost", 0.0)),
                    **last_train,
                }
            )

    run_dir = session_root / "benchmark" / policy_id / f"seed_{int(seed)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(run_dir / "metrics_over_steps.csv", rows, sorted({key for row in rows for key in row.keys()}))
    if state_action_rows:
        _write_csv(run_dir / "state_action_trace.csv", state_action_rows, list(state_action_rows[0].keys()))
    final_row = dict(rows[-1])
    write_json(run_dir / "final_metrics.json", final_row)
    learner.save(str(run_dir / "learner_checkpoint.pt"))
    return final_row


def summarize_session(*, session_root: Path, config: Exp3DigitalTwinConfig) -> int:
    summary_dir = session_root / "summary"
    figures_dir = summary_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fit_metrics = load_json(session_root / "generator" / "fit_metrics.json")
    benchmark_rows: list[dict[str, Any]] = []
    traces: dict[str, list[dict[str, Any]]] = {}
    for metrics_path in sorted(session_root.glob("benchmark/*/seed_*/metrics_over_steps.csv")):
        policy_id = metrics_path.parents[1].name
        seed = int(metrics_path.parent.name.split("_", 1)[1])
        with metrics_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            trace = []
            for row in reader:
                parsed = {key: (float(value) if key != "step" else int(value)) for key, value in row.items() if value != ""}
                parsed["policy_id"] = policy_id
                parsed["seed"] = seed
                trace.append(parsed)
            if trace:
                traces.setdefault(policy_id, []).extend(trace)
                benchmark_rows.append(
                    {
                        "policy_id": policy_id,
                        "seed": seed,
                        **{key: trace[-1][key] for key in trace[-1] if key not in {"policy_id", "seed", "step"}},
                    }
                )

    if not benchmark_rows:
        raise FileNotFoundError(f"No benchmark metrics found under {session_root / 'benchmark'}")

    _write_csv(summary_dir / "benchmark_final_metrics.csv", benchmark_rows, sorted({key for row in benchmark_rows for key in row.keys()}))
    write_json(summary_dir / "generator_fit_metrics.json", fit_metrics)

    def _aggregate_trace(metric_key: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        grouped: dict[str, dict[int, list[float]]] = {}
        for policy_id, rows in traces.items():
            by_step: dict[int, list[float]] = {}
            for row in rows:
                if metric_key not in row:
                    continue
                by_step.setdefault(int(row["step"]), []).append(float(row[metric_key]))
            grouped[policy_id] = by_step
        agg: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for policy_id, by_step in grouped.items():
            steps = np.asarray(sorted(by_step.keys()), dtype=np.int64)
            means = np.asarray([np.mean(by_step[int(step)]) for step in steps], dtype=np.float32)
            agg[policy_id] = (steps, means)
        return agg

    param_curves = _aggregate_trace("param_error")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for policy_id, (steps, values) in param_curves.items():
        ax.plot(steps, values, marker="o", label=policy_id)
    ax.set_xlabel("Interaction step")
    ax.set_ylabel("Relative dynamics parameter error")
    ax.set_title("Exp 3 digital-twin identification")
    ax.legend(loc="best")
    _plot_and_save(fig, figures_dir / "parameter_error_over_steps", config.summary.figure_formats)

    rate_curves = _aggregate_trace("rate_rollout_mse")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for policy_id, (steps, values) in rate_curves.items():
        ax.plot(steps, values, marker="o", label=policy_id)
    ax.set_xlabel("Interaction step")
    ax.set_ylabel("Spike-rate rollout MSE")
    ax.set_title("Exp 3 digital-twin predictive fidelity")
    ax.legend(loc="best")
    _plot_and_save(fig, figures_dir / "rate_rollout_mse_over_steps", config.summary.figure_formats)

    final_by_policy: dict[str, list[float]] = {}
    for row in benchmark_rows:
        final_by_policy.setdefault(str(row["policy_id"]), []).append(float(row["param_error"]))
    labels = sorted(final_by_policy.keys())
    values = [float(np.mean(final_by_policy[label])) for label in labels]
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.bar(labels, values)
    ax.set_ylabel("Final relative parameter error")
    ax.set_title("Final Exp 3 identification error by policy")
    ax.tick_params(axis="x", rotation=20)
    _plot_and_save(fig, figures_dir / "final_parameter_error_by_policy", config.summary.figure_formats)

    best_policy = min(labels, key=lambda label: float(np.mean(final_by_policy[label])))
    write_json(
        summary_dir / "summary_metadata.json",
        {
            "best_policy": best_policy,
            "best_policy_final_param_error": float(np.mean(final_by_policy[best_policy])),
            "session_root": str(session_root),
        },
    )
    return 0


def run_workflow(
    *,
    config: Exp3DigitalTwinConfig,
    session_root: Path,
    mode: str,
) -> int:
    device = _resolve_device(config.runtime.device)
    _set_seed(int(config.runtime.seed))
    session_root.mkdir(parents=True, exist_ok=True)
    write_json(session_root / "session_metadata.json", {"config": config.to_dict(), "device": device})

    generator = None
    initial_latents = None
    probe_latents = None
    probe_actions = None
    control_matrix = None

    if mode in {"fit", "all"}:
        generator, _bundle, initial_latents, probe_latents, probe_actions = _fit_generator(
            session_root=session_root,
            config=config,
            device=device,
        )
        with np.load(session_root / "generator" / "eval_artifacts.npz", allow_pickle=False) as data:
            control_matrix = np.asarray(data["control_matrix"], dtype=np.float32)

    if mode in {"benchmark", "summary"} and generator is None:
        generator, _payload, initial_latents, probe_latents, probe_actions, control_matrix = _load_generator_assets(
            session_root=session_root,
            device=device,
        )

    if mode == "fit":
        return 0

    if mode in {"benchmark", "all"}:
        if generator is None or control_matrix is None or initial_latents is None or probe_latents is None or probe_actions is None:
            generator, _payload, initial_latents, probe_latents, probe_actions, control_matrix = _load_generator_assets(
                session_root=session_root,
                device=device,
            )
        for policy_id in config.benchmark.policy_ids:
            for seed in config.benchmark.seeds:
                _run_single_benchmark(
                    session_root=session_root,
                    policy_id=str(policy_id),
                    seed=int(seed),
                    generator=generator,
                    initial_latents=initial_latents,
                    probe_latents=probe_latents,
                    probe_actions=probe_actions,
                    control_matrix=control_matrix,
                    config=config,
                    device=device,
                )

    if mode in {"summary", "all"}:
        return summarize_session(session_root=session_root, config=config)

    return 0
