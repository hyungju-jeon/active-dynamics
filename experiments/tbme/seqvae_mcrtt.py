from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

import actdyn.models
from actdyn.config import ExperimentConfig
from actdyn.models.dynamics import MLPDynamics
from actdyn.utils.experiment_helpers import setup_model
from actdyn.utils.rollout import Rollout, RolloutBuffer

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from experiment_common import load_json, resolve_session_root, write_json
    from experiments._OLD.cosyne.realdata_spiking import fit_linear_dynamics_ridge
else:
    from ..experiment_common import load_json, resolve_session_root, write_json
    from .._OLD.cosyne.realdata_spiking import fit_linear_dynamics_ridge


DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("seqvae_mcrtt.yaml")
DEFAULT_BASE_DIR = "results/tbme/seqvae_mcrtt"


@dataclass
class DatasetConfig:
    dataset_path: str = "data/mcrtt/mcrtt_replay.npz"
    observation_key: str = "behavior"
    train_fraction: float = 0.7
    sequence_length: int = 128
    sequence_stride: int = 64
    normalize_observations: bool = True
    max_train_sequences: int | None = None
    max_eval_sequences: int | None = 64


@dataclass
class SeqVaeModelConfig:
    latent_dims: list[int] = field(default_factory=lambda: [2, 4, 8])
    encoder_type: str = "rnn"
    enc_hidden_dims: list[int] = field(default_factory=lambda: [64])
    enc_rnn_hidden_dims: list[int] = field(default_factory=lambda: [64])
    enc_rnn_type: str = "gru"
    enc_h_init: str = "reset"
    mapping_type: str = "linear"
    map_hidden_dims: list[int] = field(default_factory=lambda: [64])
    noise_type: str = "gaussian"
    dynamics_type: str = "mlp"
    dyn_hidden_dims: list[int] = field(default_factory=lambda: [128, 128])
    dyn_activation: str = "relu"
    is_residual: bool = False
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 16
    n_epochs: int = 40
    beta: float = 1.0
    n_samples: int = 3
    k_steps: int = 5
    grad_clip_norm: float = 10.0
    p_mask: float = 0.0
    annealing_type: str = "none"
    annealing_steps: int = 1000
    warmup: int = 0


@dataclass
class BaselineConfig:
    linear_ridge: float = 1e-3
    mlp_hidden_dims: list[int] = field(default_factory=lambda: [128, 128])
    mlp_learning_rate: float = 1e-3
    mlp_weight_decay: float = 1e-5
    mlp_batch_size: int = 256
    mlp_n_epochs: int = 200
    mlp_grad_clip_norm: float = 10.0


@dataclass
class RecoveryConfig:
    synthetic_num_sequences: int = 64
    synthetic_sequence_length: int = 128
    sample_decoder_noise: bool = False
    refit_n_epochs: int = 30
    rollout_horizons: list[int] = field(default_factory=lambda: [1, 5, 10, 25, 50])


@dataclass
class RuntimeConfig:
    seed: int = 0
    device: str = "auto"


@dataclass
class SummaryConfig:
    figure_formats: list[str] = field(default_factory=lambda: [".pdf"])
    representative_index: int = 0


@dataclass
class SeqVaeMcrttConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    seqvae: SeqVaeModelConfig = field(default_factory=SeqVaeModelConfig)
    baselines: BaselineConfig = field(default_factory=BaselineConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    summary: SummaryConfig = field(default_factory=SummaryConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SeqVaeMcrttConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        return cls(
            runtime=RuntimeConfig(**payload.get("runtime", {})),
            dataset=DatasetConfig(**payload.get("dataset", {})),
            seqvae=SeqVaeModelConfig(**payload.get("seqvae", {})),
            baselines=BaselineConfig(**payload.get("baselines", {})),
            recovery=RecoveryConfig(**payload.get("recovery", {})),
            summary=SummaryConfig(**payload.get("summary", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path | None = None) -> "SeqVaeMcrttConfig":
    config_path = DEFAULT_CONFIG_PATH if path is None else Path(path)
    return SeqVaeMcrttConfig.from_yaml(config_path)


@dataclass
class ObservationBundle:
    train_raw: np.ndarray
    eval_raw: np.ndarray
    train_norm: np.ndarray
    eval_norm: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    metadata: dict[str, Any]


@dataclass
class SequenceBundle:
    train: np.ndarray
    eval: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    metadata: dict[str, Any]


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


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _load_behavior_array(dataset_cfg: DatasetConfig) -> ObservationBundle:
    path = _resolve_path(dataset_cfg.dataset_path)
    with np.load(path, allow_pickle=False) as data:
        observation_keys = [
            key.strip()
            for key in str(dataset_cfg.observation_key).replace("+", ",").split(",")
            if key.strip()
        ]
        if not observation_keys:
            raise ValueError("dataset.observation_key must not be empty")
        missing = [key for key in observation_keys if key not in data]
        if missing:
            raise KeyError(f"Expected keys {missing!r} in {path}, found {sorted(data.files)}")
        arrays = [np.asarray(data[key], dtype=np.float32) for key in observation_keys]
        row_counts = {arr.shape[0] for arr in arrays}
        if len(row_counts) != 1:
            raise ValueError(
                f"Observation arrays must share the same time dimension, got {[arr.shape for arr in arrays]}"
            )
        obs_raw = arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=1)
        metadata = {
            "dataset_path": str(path),
            "available_keys": [str(key) for key in data.files],
            "observation_key": str(dataset_cfg.observation_key),
            "resolved_observation_keys": observation_keys,
        }

    if obs_raw.ndim != 2:
        raise ValueError(f"Expected rank-2 observation array, got {obs_raw.shape}")
    if obs_raw.shape[0] < 16:
        raise ValueError(f"Observation array is too short for sequence training: {obs_raw.shape}")

    split = int(
        np.clip(
            round(obs_raw.shape[0] * float(dataset_cfg.train_fraction)), 8, obs_raw.shape[0] - 8
        )
    )
    train_raw = obs_raw[:split].astype(np.float32, copy=False)
    eval_raw = obs_raw[split:].astype(np.float32, copy=False)

    if bool(dataset_cfg.normalize_observations):
        mean = np.mean(train_raw, axis=0, keepdims=True).astype(np.float32)
        std = np.std(train_raw, axis=0, keepdims=True).astype(np.float32)
        std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
        train_norm = ((train_raw - mean) / std).astype(np.float32)
        eval_norm = ((eval_raw - mean) / std).astype(np.float32)
    else:
        mean = np.zeros((1, obs_raw.shape[1]), dtype=np.float32)
        std = np.ones((1, obs_raw.shape[1]), dtype=np.float32)
        train_norm = train_raw.copy()
        eval_norm = eval_raw.copy()

    metadata.update(
        {
            "num_timepoints": int(obs_raw.shape[0]),
            "observation_dim": int(obs_raw.shape[1]),
            "split_index": int(split),
            "normalize_observations": bool(dataset_cfg.normalize_observations),
        }
    )
    return ObservationBundle(
        train_raw=train_raw,
        eval_raw=eval_raw,
        train_norm=train_norm,
        eval_norm=eval_norm,
        mean=mean,
        std=std,
        metadata=metadata,
    )


def _sliding_windows(values: np.ndarray, *, length: int, stride: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected rank-2 array for sliding windows, got {arr.shape}")
    seq_len = int(length)
    seq_stride = max(1, int(stride))
    if seq_len < 2:
        raise ValueError(f"sequence_length must be at least 2, got {seq_len}")
    if arr.shape[0] < seq_len:
        raise ValueError(f"Need at least {seq_len} time steps, got {arr.shape[0]}")
    starts = list(range(0, arr.shape[0] - seq_len + 1, seq_stride))
    windows = np.stack([arr[start : start + seq_len] for start in starts], axis=0)
    return windows.astype(np.float32, copy=False)


def _cap_sequence_count(sequences: np.ndarray, max_items: int | None) -> np.ndarray:
    if max_items is None or sequences.shape[0] <= int(max_items):
        return sequences
    idx = np.linspace(0, sequences.shape[0] - 1, int(max_items), dtype=np.float64)
    picked = np.unique(np.round(idx).astype(np.int64))
    return sequences[picked]


def build_sequence_bundle(dataset_cfg: DatasetConfig) -> SequenceBundle:
    obs = _load_behavior_array(dataset_cfg)
    train_seq = _sliding_windows(
        obs.train_norm,
        length=int(dataset_cfg.sequence_length),
        stride=int(dataset_cfg.sequence_stride),
    )
    eval_seq = _sliding_windows(
        obs.eval_norm,
        length=int(dataset_cfg.sequence_length),
        stride=int(dataset_cfg.sequence_stride),
    )
    train_seq = _cap_sequence_count(train_seq, dataset_cfg.max_train_sequences)
    eval_seq = _cap_sequence_count(eval_seq, dataset_cfg.max_eval_sequences)
    metadata = dict(obs.metadata)
    metadata.update(
        {
            "train_sequence_count": int(train_seq.shape[0]),
            "eval_sequence_count": int(eval_seq.shape[0]),
            "sequence_length": int(dataset_cfg.sequence_length),
            "sequence_stride": int(dataset_cfg.sequence_stride),
        }
    )
    return SequenceBundle(
        train=train_seq,
        eval=eval_seq,
        mean=obs.mean,
        std=obs.std,
        metadata=metadata,
    )


def _build_rollout_buffer(sequences: np.ndarray, *, action_dim: int) -> RolloutBuffer:
    buffer = RolloutBuffer(device="cpu")
    for seq in sequences:
        rollout = Rollout(device="cpu")
        payload = {"next_obs": seq.astype(np.float32, copy=False)}
        if int(action_dim) > 0:
            payload["action"] = np.zeros((sequences.shape[1], int(action_dim)), dtype=np.float32)
        rollout.add_dict(**payload)
        rollout.finalize()
        buffer.add_rollout(rollout)
    return buffer


def _build_seqvae_config(
    *,
    obs_dim: int,
    latent_dim: int,
    seqvae_cfg: SeqVaeModelConfig,
    device: str,
) -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.device = str(device)
    cfg.action_dim = 0
    cfg.observation_dim = int(obs_dim)
    cfg.latent_dim = int(latent_dim)
    cfg.dt = 1.0
    cfg.environment.env_action_bounds = [0.0, 0.0]
    cfg.model.model_type = "seq-vae"
    cfg.model.encoder_type = str(seqvae_cfg.encoder_type)
    cfg.model.enc_hidden_dims = list(seqvae_cfg.enc_hidden_dims)
    cfg.model.enc_rnn_hidden_dims = list(seqvae_cfg.enc_rnn_hidden_dims)
    cfg.model.enc_rnn_type = str(seqvae_cfg.enc_rnn_type)
    cfg.model.enc_h_init = str(seqvae_cfg.enc_h_init)
    cfg.model.mapping_type = str(seqvae_cfg.mapping_type)
    cfg.model.map_hidden_dims = list(seqvae_cfg.map_hidden_dims)
    cfg.model.noise_type = str(seqvae_cfg.noise_type)
    cfg.model.dynamics_type = str(seqvae_cfg.dynamics_type)
    cfg.model.dyn_hidden_dims = list(seqvae_cfg.dyn_hidden_dims)
    cfg.model.dyn_activation = str(seqvae_cfg.dyn_activation)
    cfg.model.is_residual = bool(seqvae_cfg.is_residual)
    cfg.model.dyn_dt = 1.0
    cfg.model.action_type = "identity"
    cfg.training.beta = float(seqvae_cfg.beta)
    cfg.training.n_samples = int(seqvae_cfg.n_samples)
    cfg.training.k_steps = int(seqvae_cfg.k_steps)
    cfg.training.grad_clip_norm = float(seqvae_cfg.grad_clip_norm)
    cfg.training.p_mask = float(seqvae_cfg.p_mask)
    return cfg


def _history_to_rows(history: Any) -> list[dict[str, float]]:
    if isinstance(history, dict):
        elbo = np.asarray(history.get("ELBO", []), dtype=np.float32).reshape(-1)
        log_like = np.asarray(
            history.get("log_L", history.get("log_like", [])),
            dtype=np.float32,
        ).reshape(-1)
        kl = np.asarray(history.get("KL", history.get("kl", [])), dtype=np.float32).reshape(-1)
        n_epochs = max(int(elbo.size), int(log_like.size), int(kl.size))
        if n_epochs == 0:
            return []
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
        values = (
            item.detach().cpu().numpy().reshape(-1)
            if isinstance(item, torch.Tensor)
            else np.asarray(item)
        )
        if values.size < 3:
            raise ValueError(
                f"Expected training history entry with at least 3 values, got {values}"
            )
        rows.append(
            {
                "epoch": idx,
                "neg_elbo": float(values[0]),
                "log_like": float(values[1]),
                "kl": float(values[2]),
            }
        )
    return rows


def _rows_to_curve(rows: Sequence[dict[str, float]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float32)


def _encode_means(model: actdyn.models.SeqVae, obs: np.ndarray, *, device: str) -> np.ndarray:
    model.eval()
    y = torch.as_tensor(obs, dtype=torch.float32, device=device)
    u = (
        None
        if int(getattr(model, "action_dim", 0)) <= 0
        else torch.zeros(
            (y.shape[0], y.shape[1], model.action_dim), dtype=torch.float32, device=device
        )
    )
    with torch.no_grad():
        _samples, mu, _var = model.encoder(y, u=u, n_samples=1)
    return mu.detach().cpu().numpy().astype(np.float32, copy=False)


def _seqvae_next_latent_mean(model: actdyn.models.SeqVae, z: torch.Tensor) -> torch.Tensor:
    pred = model.dynamics(z)
    if bool(getattr(model.dynamics, "is_residual", False)):
        pred = z + pred * float(getattr(model.dynamics, "dt", 1.0))
    return pred


def _rollout_seqvae_observations(
    model: actdyn.models.SeqVae,
    z0: torch.Tensor,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    latent_steps: list[torch.Tensor] = []
    obs_steps: list[torch.Tensor] = []
    z_cur = z0
    for _ in range(int(horizon)):
        z_cur = _seqvae_next_latent_mean(model, z_cur)
        latent_steps.append(z_cur)
        obs_steps.append(model.decoder(z_cur))
    return torch.cat(obs_steps, dim=1), torch.cat(latent_steps, dim=1)


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


def _unnormalize(values: np.ndarray, *, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.float32) * std.astype(np.float32)) + mean.astype(np.float32)


def _rollout_mse_curve(
    predict_step: Callable[[np.ndarray], np.ndarray],
    eval_sequences: np.ndarray,
    horizons: Sequence[int],
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    max_h = max(int(item) for item in horizons)
    for horizon in horizons:
        if int(horizon) < 1 or int(horizon) >= eval_sequences.shape[1]:
            raise ValueError(
                f"Invalid horizon {horizon} for sequences of length {eval_sequences.shape[1]}"
            )
    for horizon in horizons:
        preds: list[np.ndarray] = []
        target = eval_sequences[:, 1 : int(horizon) + 1, :]
        for seq in eval_sequences:
            current = seq[0].copy()
            rollout: list[np.ndarray] = []
            for _ in range(int(horizon)):
                current = predict_step(current)
                rollout.append(current.copy())
            preds.append(np.stack(rollout, axis=0))
        pred_arr = np.stack(preds, axis=0)
        rows.append(
            {
                "horizon": int(horizon),
                "rollout_mse": _mse(pred_arr, target),
                "rollout_r2": _r2(pred_arr, target),
            }
        )
    return rows


def _evaluate_seqvae_realdata(
    model: actdyn.models.SeqVae,
    eval_sequences: np.ndarray,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    seqvae_cfg: SeqVaeModelConfig,
    device: str,
    rollout_horizons: Sequence[int],
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    y = torch.as_tensor(eval_sequences, dtype=torch.float32, device=device)
    u = (
        None
        if int(getattr(model, "action_dim", 0)) <= 0
        else torch.zeros(
            (y.shape[0], y.shape[1], model.action_dim), dtype=torch.float32, device=device
        )
    )
    model.eval()
    with torch.no_grad():
        loss, log_like, kl = model.compute_elbo(
            y,
            u=u,
            n_samples=int(seqvae_cfg.n_samples),
            k_steps=int(seqvae_cfg.k_steps),
            beta=float(seqvae_cfg.beta),
            p_mask=float(seqvae_cfg.p_mask),
        )
        _samples, mu, _var = model.encoder(y, u=u, n_samples=1)
        z_next = _seqvae_next_latent_mean(model, mu[:, :-1, :])
        one_step_pred = model.decoder(z_next)
        rollout_pred, rollout_latent = _rollout_seqvae_observations(
            model,
            mu[:, :1, :],
            horizon=max(int(item) for item in rollout_horizons),
        )

    target_one = y[:, 1:, :]
    pred_one = one_step_pred.detach().cpu().numpy().astype(np.float32, copy=False)
    target_one_np = target_one.detach().cpu().numpy().astype(np.float32, copy=False)
    rollout_pred_np = rollout_pred.detach().cpu().numpy().astype(np.float32, copy=False)
    rollout_target_np = eval_sequences[:, 1 : rollout_pred_np.shape[1] + 1, :]
    latent_np = mu.detach().cpu().numpy().astype(np.float32, copy=False)
    rollout_latent_np = rollout_latent.detach().cpu().numpy().astype(np.float32, copy=False)

    one_step_pred_raw = _unnormalize(pred_one, mean=mean, std=std)
    target_one_raw = _unnormalize(target_one_np, mean=mean, std=std)
    rollout_pred_raw = _unnormalize(rollout_pred_np, mean=mean, std=std)
    rollout_target_raw = _unnormalize(rollout_target_np, mean=mean, std=std)

    metrics = {
        "eval_neg_elbo": float(loss.detach().cpu()),
        "eval_log_like": float(log_like.detach().cpu()),
        "eval_kl": float(kl.detach().cpu()),
        "one_step_mse_norm": _mse(pred_one, target_one_np),
        "one_step_mse_raw": _mse(one_step_pred_raw, target_one_raw),
        "one_step_r2_norm": _r2(pred_one, target_one_np),
        "one_step_r2_raw": _r2(one_step_pred_raw, target_one_raw),
    }
    for horizon in rollout_horizons:
        h = int(horizon)
        metrics[f"rollout_mse_h{h}_norm"] = _mse(
            rollout_pred_np[:, :h, :], rollout_target_np[:, :h, :]
        )
        metrics[f"rollout_mse_h{h}_raw"] = _mse(
            rollout_pred_raw[:, :h, :], rollout_target_raw[:, :h, :]
        )
        metrics[f"rollout_r2_h{h}_norm"] = _r2(
            rollout_pred_np[:, :h, :], rollout_target_np[:, :h, :]
        )
        metrics[f"rollout_r2_h{h}_raw"] = _r2(
            rollout_pred_raw[:, :h, :], rollout_target_raw[:, :h, :]
        )
    artifacts = {
        "posterior_latent": latent_np,
        "rollout_latent": rollout_latent_np,
        "one_step_pred_norm": pred_one,
        "one_step_target_norm": target_one_np,
        "rollout_pred_norm": rollout_pred_np,
        "rollout_target_norm": rollout_target_np,
    }
    return metrics, artifacts


def _prepare_flat_transitions(sequences: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = sequences[:, :-1, :].reshape(-1, sequences.shape[-1]).astype(np.float32)
    y = sequences[:, 1:, :].reshape(-1, sequences.shape[-1]).astype(np.float32)
    return x, y


def _train_mlp_baseline(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    hidden_dims: Sequence[int],
    device: str,
    lr: float,
    weight_decay: float,
    batch_size: int,
    n_epochs: int,
    grad_clip_norm: float,
) -> tuple[MLPDynamics, list[dict[str, float]]]:
    model = MLPDynamics(
        state_dim=int(train_x.shape[1]),
        hidden_dims=list(hidden_dims),
        activation="relu",
        dt=1.0,
        is_residual=False,
        device=device,
    ).to(device)
    dataset = TensorDataset(
        torch.as_tensor(train_x, dtype=torch.float32),
        torch.as_tensor(train_y, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )
    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(1, int(n_epochs) + 1):
        losses: list[float] = []
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = torch.mean((pred - y_batch) ** 2)
            loss.backward()
            if float(grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "train_mse": float(np.mean(losses) if losses else 0.0)})
    model.eval()
    return model, history


def _evaluate_predictor_realdata(
    *,
    name: str,
    predict_step: Callable[[np.ndarray], np.ndarray],
    eval_sequences: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    rollout_horizons: Sequence[int],
) -> dict[str, float]:
    one_pred = np.stack([predict_step(seq[:-1]) for seq in eval_sequences], axis=0).astype(
        np.float32
    )
    one_target = eval_sequences[:, 1:, :]
    one_pred_raw = _unnormalize(one_pred, mean=mean, std=std)
    one_target_raw = _unnormalize(one_target, mean=mean, std=std)

    metrics = {
        "model_name": str(name),
        "one_step_mse_norm": _mse(one_pred, one_target),
        "one_step_mse_raw": _mse(one_pred_raw, one_target_raw),
        "one_step_r2_norm": _r2(one_pred, one_target),
        "one_step_r2_raw": _r2(one_pred_raw, one_target_raw),
    }
    rollout_rows = _rollout_mse_curve(
        lambda cur: np.asarray(predict_step(cur[None, :]), dtype=np.float32).reshape(-1),
        eval_sequences,
        rollout_horizons,
    )
    for row in rollout_rows:
        h = int(row["horizon"])
        metrics[f"rollout_mse_h{h}_norm"] = float(row["rollout_mse"])
        metrics[f"rollout_r2_h{h}_norm"] = float(row["rollout_r2"])

    raw_rows = []
    for row in rollout_rows:
        h = int(row["horizon"])
        preds: list[np.ndarray] = []
        target = eval_sequences[:, 1 : h + 1, :]
        for seq in eval_sequences:
            current = seq[0].copy()
            rollout: list[np.ndarray] = []
            for _ in range(h):
                current = np.asarray(predict_step(current[None, :]), dtype=np.float32).reshape(-1)
                rollout.append(current.copy())
            preds.append(np.stack(rollout, axis=0))
        pred_arr = np.stack(preds, axis=0)
        raw_rows.append(
            (
                h,
                _mse(
                    _unnormalize(pred_arr, mean=mean, std=std),
                    _unnormalize(target, mean=mean, std=std),
                ),
                _r2(
                    _unnormalize(pred_arr, mean=mean, std=std),
                    _unnormalize(target, mean=mean, std=std),
                ),
            )
        )
    for horizon, mse_raw, r2_raw in raw_rows:
        metrics[f"rollout_mse_h{horizon}_raw"] = float(mse_raw)
        metrics[f"rollout_r2_h{horizon}_raw"] = float(r2_raw)
    return metrics


def _sample_seed_sequences(sequences: np.ndarray, num_items: int) -> np.ndarray:
    if sequences.shape[0] <= num_items:
        if sequences.shape[0] >= 2:
            return sequences
        return np.repeat(sequences, 2, axis=0)
    idx = np.linspace(0, sequences.shape[0] - 1, int(num_items), dtype=np.float64)
    picked = np.unique(np.round(idx).astype(np.int64))
    sampled = sequences[picked]
    if sampled.shape[0] >= 2:
        return sampled
    return np.repeat(sampled, 2, axis=0)


def _generate_synthetic_sequences(
    model: actdyn.models.SeqVae,
    seeds: np.ndarray,
    *,
    sequence_length: int,
    sample_decoder_noise: bool,
    device: str,
) -> np.ndarray:
    seed_obs = np.asarray(seeds[:, :1, :], dtype=np.float32)
    y = torch.as_tensor(seed_obs, dtype=torch.float32, device=device)
    u = (
        None
        if int(getattr(model, "action_dim", 0)) <= 0
        else torch.zeros(
            (y.shape[0], y.shape[1], model.action_dim), dtype=torch.float32, device=device
        )
    )
    with torch.no_grad():
        _samples, mu, _var = model.encoder(y, u=u, n_samples=1)
        rollout_obs, _rollout_latent = _rollout_seqvae_observations(
            model,
            mu[:, :1, :],
            horizon=max(0, int(sequence_length) - 1),
        )
    obs_np = rollout_obs.detach().cpu().numpy().astype(np.float32, copy=False)
    if sample_decoder_noise:
        sigma = (
            torch.sqrt(model.decoder.var()).detach().cpu().numpy().astype(np.float32, copy=False)
        )
        obs_np = obs_np + np.random.randn(*obs_np.shape).astype(np.float32) * sigma.reshape(
            1, 1, -1
        )
    return np.concatenate([seed_obs, obs_np], axis=1).astype(np.float32, copy=False)


def _orthogonal_procrustes(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    ref_mean = np.mean(ref, axis=0, keepdims=True)
    cand_mean = np.mean(cand, axis=0, keepdims=True)
    ref_centered = ref - ref_mean
    cand_centered = cand - cand_mean
    u, _s, vt = np.linalg.svd(cand_centered.T @ ref_centered, full_matrices=False)
    rot = u @ vt
    return (cand_centered @ rot + ref_mean).astype(np.float32, copy=False)


def _evaluate_recovery(
    *,
    generator: actdyn.models.SeqVae,
    recovered: actdyn.models.SeqVae,
    synthetic_eval: np.ndarray,
    rollout_horizons: Sequence[int],
    device: str,
) -> dict[str, float]:
    y = torch.as_tensor(synthetic_eval, dtype=torch.float32, device=device)
    u_gen = (
        None
        if int(getattr(generator, "action_dim", 0)) <= 0
        else torch.zeros(
            (y.shape[0], y.shape[1], generator.action_dim), dtype=torch.float32, device=device
        )
    )
    u_rec = (
        None
        if int(getattr(recovered, "action_dim", 0)) <= 0
        else torch.zeros(
            (y.shape[0], y.shape[1], recovered.action_dim), dtype=torch.float32, device=device
        )
    )
    with torch.no_grad():
        _sg, mu_gen, _vg = generator.encoder(y, u=u_gen, n_samples=1)
        _sr, mu_rec, _vr = recovered.encoder(y, u=u_rec, n_samples=1)
        gen_rollout_obs, gen_rollout_latent = _rollout_seqvae_observations(
            generator,
            mu_gen[:, :1, :],
            horizon=max(int(item) for item in rollout_horizons),
        )
        rec_rollout_obs, rec_rollout_latent = _rollout_seqvae_observations(
            recovered,
            mu_rec[:, :1, :],
            horizon=max(int(item) for item in rollout_horizons),
        )
    mu_gen_np = mu_gen.detach().cpu().numpy().astype(np.float32, copy=False)
    mu_rec_np = mu_rec.detach().cpu().numpy().astype(np.float32, copy=False)
    aligned = _orthogonal_procrustes(
        mu_gen_np.reshape(-1, mu_gen_np.shape[-1]),
        mu_rec_np.reshape(-1, mu_rec_np.shape[-1]),
    ).reshape(mu_rec_np.shape)
    gen_rollout_np = gen_rollout_obs.detach().cpu().numpy().astype(np.float32, copy=False)
    rec_rollout_np = rec_rollout_obs.detach().cpu().numpy().astype(np.float32, copy=False)
    gen_latent_np = gen_rollout_latent.detach().cpu().numpy().astype(np.float32, copy=False)
    rec_latent_np = rec_rollout_latent.detach().cpu().numpy().astype(np.float32, copy=False)
    aligned_roll_latent = _orthogonal_procrustes(
        gen_latent_np.reshape(-1, gen_latent_np.shape[-1]),
        rec_latent_np.reshape(-1, rec_latent_np.shape[-1]),
    ).reshape(rec_latent_np.shape)
    metrics = {
        "aligned_latent_mse": _mse(aligned, mu_gen_np),
        "aligned_rollout_latent_mse": _mse(aligned_roll_latent, gen_latent_np),
        "generator_vs_recovered_rollout_obs_mse": _mse(rec_rollout_np, gen_rollout_np),
        "generator_vs_recovered_rollout_obs_r2": _r2(rec_rollout_np, gen_rollout_np),
    }
    for horizon in rollout_horizons:
        h = int(horizon)
        metrics[f"generator_vs_recovered_rollout_obs_mse_h{h}"] = _mse(
            rec_rollout_np[:, :h, :], gen_rollout_np[:, :h, :]
        )
    return metrics


def _plot_and_save(fig: plt.Figure, stem: Path, figure_formats: Sequence[str]) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in figure_formats:
        suffix = fmt if str(fmt).startswith(".") else f".{fmt}"
        fig.savefig(stem.with_suffix(suffix), bbox_inches="tight")
    plt.close(fig)


def _save_seqvae_run_artifacts(
    run_dir: Path,
    *,
    train_history_rows: Sequence[dict[str, float]],
    metrics: dict[str, float],
    eval_artifacts: dict[str, np.ndarray],
    config_payload: dict[str, Any],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        run_dir / "train_history.csv", train_history_rows, ["epoch", "neg_elbo", "log_like", "kl"]
    )
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "config.json", config_payload)
    np.savez_compressed(
        run_dir / "eval_artifacts.npz",
        posterior_latent=eval_artifacts["posterior_latent"],
        rollout_latent=eval_artifacts["rollout_latent"],
        one_step_pred_norm=eval_artifacts["one_step_pred_norm"],
        one_step_target_norm=eval_artifacts["one_step_target_norm"],
        rollout_pred_norm=eval_artifacts["rollout_pred_norm"],
        rollout_target_norm=eval_artifacts["rollout_target_norm"],
    )


def _plot_seqvae_run(
    run_dir: Path,
    *,
    train_history_rows: Sequence[dict[str, float]],
    eval_artifacts: dict[str, np.ndarray],
    figure_formats: Sequence[str],
    representative_index: int,
) -> None:
    figures_dir = run_dir / "figures"

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(
        [row["epoch"] for row in train_history_rows],
        [row["neg_elbo"] for row in train_history_rows],
        label="neg ELBO",
    )
    ax.plot(
        [row["epoch"] for row in train_history_rows],
        [row["log_like"] for row in train_history_rows],
        label="log-like",
    )
    ax.plot(
        [row["epoch"] for row in train_history_rows],
        [row["kl"] for row in train_history_rows],
        label="KL",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_title("SeqVAE training history")
    ax.legend(loc="best")
    _plot_and_save(fig, figures_dir / "training_history", figure_formats)

    idx = int(np.clip(representative_index, 0, eval_artifacts["one_step_target_norm"].shape[0] - 1))
    target = eval_artifacts["one_step_target_norm"][idx]
    pred = eval_artifacts["one_step_pred_norm"][idx]
    n_dim = target.shape[-1]
    fig, axes = plt.subplots(n_dim, 1, figsize=(9.0, max(3.5, 2.0 * n_dim)), sharex=True)
    if n_dim == 1:
        axes = [axes]
    for dim, ax in enumerate(axes):
        ax.plot(target[:, dim], label="target", linewidth=2)
        ax.plot(pred[:, dim], label="one-step pred", linewidth=1.5)
        ax.set_ylabel(f"dim {dim}")
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("Step")
    fig.suptitle("Representative one-step prediction")
    _plot_and_save(fig, figures_dir / "representative_one_step_prediction", figure_formats)

    latent = eval_artifacts["posterior_latent"][idx]
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    if latent.shape[-1] >= 2:
        ax.plot(latent[:, 0], latent[:, 1], linewidth=1.5)
        ax.set_xlabel("z0")
        ax.set_ylabel("z1")
    else:
        ax.plot(latent[:, 0], linewidth=1.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("z0")
    ax.set_title("Representative posterior latent trajectory")
    _plot_and_save(fig, figures_dir / "representative_posterior_latent", figure_formats)


def _save_baseline_run(
    run_dir: Path,
    *,
    metrics: dict[str, float],
    history_rows: Sequence[dict[str, float]] | None = None,
    state_dict: dict[str, Any] | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "metrics.json", metrics)
    if history_rows:
        _write_csv(run_dir / "train_history.csv", history_rows, ["epoch", "train_mse"])
    if state_dict is not None:
        torch.save(state_dict, run_dir / "checkpoint.pt")


def _collect_run_metrics(session_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    real_rows: list[dict[str, Any]] = []
    recovery_rows: list[dict[str, Any]] = []
    for run_dir in sorted(session_root.glob("seqvae_latent_*")):
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = load_json(metrics_path)
        latent_dim = int(str(run_dir.name).split("_")[-1])
        row = {"model_family": "seqvae", "latent_dim": latent_dim, **metrics}
        real_rows.append(row)
        rec_path = run_dir / "recovery" / "metrics.json"
        if rec_path.exists():
            recovery_rows.append({"latent_dim": latent_dim, **load_json(rec_path)})
    for run_dir in [session_root / "linear_behavior", session_root / "mlp_behavior"]:
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            row = {"model_family": str(run_dir.name), "latent_dim": "", **load_json(metrics_path)}
            real_rows.append(row)
    return real_rows, recovery_rows


def summarize_session(
    *,
    session_root: Path,
    config: SeqVaeMcrttConfig,
) -> int:
    summary_dir = session_root / "summary"
    figures_dir = summary_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    real_rows, recovery_rows = _collect_run_metrics(session_root)
    if not real_rows:
        raise FileNotFoundError(f"No SeqVAE/MC_RTT run metrics found under {session_root}")

    real_fieldnames = sorted({key for row in real_rows for key in row.keys()})
    recovery_fieldnames = (
        sorted({key for row in recovery_rows for key in row.keys()})
        if recovery_rows
        else ["latent_dim"]
    )
    _write_csv(summary_dir / "realdata_metrics.csv", real_rows, real_fieldnames)
    if recovery_rows:
        _write_csv(summary_dir / "recovery_metrics.csv", recovery_rows, recovery_fieldnames)

    def _label(row: dict[str, Any]) -> str:
        if row["model_family"] == "seqvae":
            return f"seqvae-d{row['latent_dim']}"
        return str(row["model_family"])

    ordered_real = sorted(
        real_rows,
        key=lambda row: (
            (
                0
                if row["model_family"] == "linear_behavior"
                else 1 if row["model_family"] == "mlp_behavior" else 2
            ),
            int(row["latent_dim"]) if str(row["latent_dim"]).strip() else -1,
        ),
    )
    labels = [_label(row) for row in ordered_real]
    values = [float(row["one_step_mse_raw"]) for row in ordered_real]
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    ax.bar(labels, values)
    ax.set_ylabel("One-step MSE (raw behavior)")
    ax.set_title("MC_RTT held-out one-step prediction")
    ax.tick_params(axis="x", rotation=25)
    _plot_and_save(
        fig, figures_dir / "realdata_one_step_mse_by_model", config.summary.figure_formats
    )

    horizon_keys = sorted(
        {
            key
            for row in real_rows
            for key in row.keys()
            if key.startswith("rollout_mse_h") and key.endswith("_raw")
        },
        key=lambda item: int(item.split("_h", 1)[1].split("_", 1)[0]),
    )
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for row in ordered_real:
        horizons = [int(key.split("_h", 1)[1].split("_", 1)[0]) for key in horizon_keys]
        ys = [float(row[key]) for key in horizon_keys]
        ax.plot(horizons, ys, marker="o", label=_label(row))
    ax.set_xlabel("Rollout horizon")
    ax.set_ylabel("Rollout MSE (raw behavior)")
    ax.set_title("Held-out rollout error by horizon")
    ax.legend(loc="best")
    _plot_and_save(
        fig, figures_dir / "realdata_rollout_mse_by_horizon", config.summary.figure_formats
    )

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for run_dir in sorted(session_root.glob("seqvae_latent_*")):
        history_path = run_dir / "train_history.csv"
        if not history_path.exists():
            continue
        epochs: list[int] = []
        neg_elbo: list[float] = []
        with history_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row["epoch"]))
                neg_elbo.append(float(row["neg_elbo"]))
        latent_dim = str(run_dir.name).split("_")[-1]
        ax.plot(epochs, neg_elbo, label=f"latent {latent_dim}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Negative ELBO")
    ax.set_title("SeqVAE training curves")
    ax.legend(loc="best")
    _plot_and_save(
        fig, figures_dir / "seqvae_training_elbo_by_latent_dim", config.summary.figure_formats
    )

    if recovery_rows:
        ordered_recovery = sorted(recovery_rows, key=lambda row: int(row["latent_dim"]))
        x = [int(row["latent_dim"]) for row in ordered_recovery]
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
        axes[0].plot(x, [float(row["aligned_latent_mse"]) for row in ordered_recovery], marker="o")
        axes[0].set_xlabel("Latent dim")
        axes[0].set_ylabel("Aligned latent MSE")
        axes[0].set_title("Synthetic recovery")
        axes[1].plot(
            x,
            [float(row["generator_vs_recovered_rollout_obs_mse"]) for row in ordered_recovery],
            marker="o",
        )
        axes[1].set_xlabel("Latent dim")
        axes[1].set_ylabel("Generator vs recovered rollout MSE")
        axes[1].set_title("Observation rollout recovery")
        _plot_and_save(
            fig, figures_dir / "seqvae_recovery_by_latent_dim", config.summary.figure_formats
        )

    best_seqvae = min(
        (row for row in real_rows if row["model_family"] == "seqvae"),
        key=lambda row: (
            float(row["rollout_mse_h50_raw"])
            if "rollout_mse_h50_raw" in row
            else float(row["one_step_mse_raw"])
        ),
    )
    write_json(
        summary_dir / "summary_metadata.json",
        {
            "best_seqvae_latent_dim": int(best_seqvae["latent_dim"]),
            "best_seqvae_one_step_mse_raw": float(best_seqvae["one_step_mse_raw"]),
            "session_root": str(session_root),
        },
    )
    return 0


def run_suite(
    *,
    config: SeqVaeMcrttConfig,
    session_root: Path,
    summarize: bool = True,
) -> int:
    device = _resolve_device(config.runtime.device)
    _set_seed(int(config.runtime.seed))
    bundle = build_sequence_bundle(config.dataset)
    session_root.mkdir(parents=True, exist_ok=True)
    write_json(
        session_root / "session_metadata.json",
        {"config": config.to_dict(), "device": device, "dataset": bundle.metadata},
    )

    obs_dim = int(bundle.train.shape[-1])
    rollout_horizons = list(config.recovery.rollout_horizons)
    train_x, train_y = _prepare_flat_transitions(bundle.train)

    linear_coef = fit_linear_dynamics_ridge(
        train_x.astype(np.float64),
        train_y.astype(np.float64),
        ridge=float(config.baselines.linear_ridge),
    ).astype(np.float32, copy=False)
    linear_metrics = _evaluate_predictor_realdata(
        name="linear_behavior",
        predict_step=lambda state: np.asarray(state, dtype=np.float32) @ linear_coef,
        eval_sequences=bundle.eval,
        mean=bundle.mean,
        std=bundle.std,
        rollout_horizons=rollout_horizons,
    )
    _save_baseline_run(session_root / "linear_behavior", metrics=linear_metrics)

    mlp_model, mlp_history = _train_mlp_baseline(
        train_x=train_x,
        train_y=train_y,
        hidden_dims=config.baselines.mlp_hidden_dims,
        device=device,
        lr=float(config.baselines.mlp_learning_rate),
        weight_decay=float(config.baselines.mlp_weight_decay),
        batch_size=int(config.baselines.mlp_batch_size),
        n_epochs=int(config.baselines.mlp_n_epochs),
        grad_clip_norm=float(config.baselines.mlp_grad_clip_norm),
    )
    mlp_metrics = _evaluate_predictor_realdata(
        name="mlp_behavior",
        predict_step=lambda state: mlp_model(
            torch.as_tensor(state, dtype=torch.float32, device=device)
        )
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False),
        eval_sequences=bundle.eval,
        mean=bundle.mean,
        std=bundle.std,
        rollout_horizons=rollout_horizons,
    )
    _save_baseline_run(
        session_root / "mlp_behavior",
        metrics=mlp_metrics,
        history_rows=mlp_history,
        state_dict=mlp_model.state_dict(),
    )

    synthetic_seed_sequences = _sample_seed_sequences(
        bundle.eval,
        num_items=int(config.recovery.synthetic_num_sequences),
    )

    for latent_dim in config.seqvae.latent_dims:
        seqvae_cfg = _build_seqvae_config(
            obs_dim=obs_dim,
            latent_dim=int(latent_dim),
            seqvae_cfg=config.seqvae,
            device=device,
        )
        model = setup_model(seqvae_cfg)
        train_buffer = _build_rollout_buffer(bundle.train, action_dim=int(seqvae_cfg.action_dim))
        history = model.train_model(
            train_buffer,
            batch_size=int(config.seqvae.batch_size),
            shuffle=True,
            optimizer="AdamW",
            lr=float(config.seqvae.learning_rate),
            weight_decay=float(config.seqvae.weight_decay),
            n_epochs=int(config.seqvae.n_epochs),
            verbose=False,
            grad_clip_norm=float(config.seqvae.grad_clip_norm),
            n_samples=int(config.seqvae.n_samples),
            k_steps=int(config.seqvae.k_steps),
            beta=float(config.seqvae.beta),
            p_mask=float(config.seqvae.p_mask),
            annealing_type=str(config.seqvae.annealing_type),
            annealing_steps=int(config.seqvae.annealing_steps),
            warmup=int(config.seqvae.warmup),
            param_list="all",
        )
        history_rows = _history_to_rows(history)
        metrics, eval_artifacts = _evaluate_seqvae_realdata(
            model,
            bundle.eval,
            mean=bundle.mean,
            std=bundle.std,
            seqvae_cfg=config.seqvae,
            device=device,
            rollout_horizons=rollout_horizons,
        )
        run_dir = session_root / f"seqvae_latent_{int(latent_dim)}"
        run_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(run_dir / "checkpoint.pt"))
        _save_seqvae_run_artifacts(
            run_dir,
            train_history_rows=history_rows,
            metrics=metrics,
            eval_artifacts=eval_artifacts,
            config_payload={"latent_dim": int(latent_dim), "model_config": seqvae_cfg.to_dict()},
        )
        _plot_seqvae_run(
            run_dir,
            train_history_rows=history_rows,
            eval_artifacts=eval_artifacts,
            figure_formats=config.summary.figure_formats,
            representative_index=int(config.summary.representative_index),
        )

        synthetic_sequences = _generate_synthetic_sequences(
            model,
            synthetic_seed_sequences,
            sequence_length=int(config.recovery.synthetic_sequence_length),
            sample_decoder_noise=bool(config.recovery.sample_decoder_noise),
            device=device,
        )
        split = int(
            np.clip(round(synthetic_sequences.shape[0] * 0.7), 1, synthetic_sequences.shape[0] - 1)
        )
        synthetic_train = synthetic_sequences[:split]
        synthetic_eval = synthetic_sequences[split:]
        recovery_cfg = _build_seqvae_config(
            obs_dim=obs_dim,
            latent_dim=int(latent_dim),
            seqvae_cfg=SeqVaeModelConfig(
                **{
                    **config.seqvae.__dict__,
                    "n_epochs": int(config.recovery.refit_n_epochs),
                }
            ),
            device=device,
        )
        recovery_model = setup_model(recovery_cfg)
        recovery_buffer = _build_rollout_buffer(
            synthetic_train,
            action_dim=int(recovery_cfg.action_dim),
        )
        recovery_history = recovery_model.train_model(
            recovery_buffer,
            batch_size=int(config.seqvae.batch_size),
            shuffle=True,
            optimizer="AdamW",
            lr=float(config.seqvae.learning_rate),
            weight_decay=float(config.seqvae.weight_decay),
            n_epochs=int(config.recovery.refit_n_epochs),
            verbose=False,
            grad_clip_norm=float(config.seqvae.grad_clip_norm),
            n_samples=int(config.seqvae.n_samples),
            k_steps=int(config.seqvae.k_steps),
            beta=float(config.seqvae.beta),
            p_mask=float(config.seqvae.p_mask),
            annealing_type=str(config.seqvae.annealing_type),
            annealing_steps=int(config.seqvae.annealing_steps),
            warmup=int(config.seqvae.warmup),
            param_list="all",
        )
        recovery_dir = run_dir / "recovery"
        recovery_dir.mkdir(parents=True, exist_ok=True)
        recovery_model.save(str(recovery_dir / "checkpoint.pt"))
        _write_csv(
            recovery_dir / "train_history.csv",
            _history_to_rows(recovery_history),
            ["epoch", "neg_elbo", "log_like", "kl"],
        )
        recovery_metrics = _evaluate_recovery(
            generator=model,
            recovered=recovery_model,
            synthetic_eval=synthetic_eval,
            rollout_horizons=rollout_horizons,
            device=device,
        )
        write_json(recovery_dir / "metrics.json", recovery_metrics)
        np.savez_compressed(
            recovery_dir / "synthetic_sequences.npz",
            synthetic_train=synthetic_train,
            synthetic_eval=synthetic_eval,
        )

    if summarize:
        summarize_session(session_root=session_root, config=config)
    return 0
