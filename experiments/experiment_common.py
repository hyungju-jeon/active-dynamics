from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


def parse_csv_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def parse_csv_ints(raw: str | None) -> list[int]:
    return [int(item) for item in parse_csv_list(raw)]


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return None


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2, sort_keys=True)


def list_session_dirs(base_dir: str | Path) -> list[Path]:
    root = Path(base_dir)
    if not root.exists():
        return []
    sessions: list[tuple[int, Path]] = []
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith("session_"):
            continue
        suffix = child.name.split("session_", 1)[1]
        if suffix.isdigit():
            sessions.append((int(suffix), child))
    sessions.sort(key=lambda item: item[0])
    return [path for _, path in sessions]


def has_experiment_dirs(base_dir: str | Path, exp_ids: Sequence[str] | None = None) -> bool:
    root = Path(base_dir)
    if not root.exists():
        return False
    if exp_ids:
        return any((root / exp_id).is_dir() for exp_id in exp_ids)
    return any(child.is_dir() and child.name.startswith("exp") for child in root.iterdir())


def resolve_session_root(
    base_dir: str | Path,
    *,
    create: bool,
    exp_ids: Sequence[str] | None = None,
) -> Path:
    root = Path(base_dir)
    if root.name.startswith("session_"):
        if create:
            root.mkdir(parents=True, exist_ok=True)
        return root
    if create:
        root.mkdir(parents=True, exist_ok=True)
    if has_experiment_dirs(root, exp_ids=exp_ids):
        return root
    sessions = list_session_dirs(root)
    if create:
        next_idx = 1
        if sessions:
            next_idx = max(int(path.name.split("session_", 1)[1]) for path in sessions) + 1
        session_root = root / f"session_{next_idx}"
        session_root.mkdir(parents=True, exist_ok=False)
        return session_root
    if sessions:
        return sessions[-1]
    return root


def resolve_artifact_path(
    run_dir: str | Path,
    metadata: Mapping[str, Any],
    *,
    key: str,
    fallback_name: str,
) -> Path:
    base_dir = Path(run_dir)
    raw = metadata.get(key)
    if isinstance(raw, str) and raw.strip():
        path = Path(raw)
        if path.is_absolute():
            return path
        # Metadata may store either repo-relative paths like results/cosyne/...
        # or run-local filenames like parameter_error_trace.csv.
        if path.exists():
            return path.resolve()
        candidate = (base_dir / path).resolve()
        if candidate.exists():
            return candidate
        return candidate
    return base_dir / fallback_name


def find_nested_metadata_paths(
    root_dir: str | Path,
    *,
    metadata_filename: str = "run_metadata.json",
    nested_pattern: str = "repeat_*",
    include_root: bool = True,
) -> list[Path]:
    base = Path(root_dir)
    paths = sorted(base.glob(f"{nested_pattern}/{metadata_filename}"))
    root_path = base / metadata_filename
    if include_root and root_path.exists():
        paths.append(root_path)
    return paths


def nested_get(
    payload: Mapping[str, Any],
    key_path: Sequence[str],
    default: Any = None,
) -> Any:
    current: Any = payload
    for key in key_path:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def align_trace_length(
    trace: Any,
    num_steps: int,
    *,
    dtype: np.dtype = np.float32,
    empty_error: str,
) -> np.ndarray:
    values = np.asarray(trace, dtype=dtype)
    if values.ndim == 1 and values.size > 0:
        values = values.reshape(1, -1)
    if values.size == 0:
        raise ValueError(empty_error)
    if values.shape[0] < num_steps:
        pad = np.repeat(values[-1:], num_steps - values.shape[0], axis=0)
        values = np.concatenate([values, pad], axis=0)
    if values.shape[0] > num_steps:
        values = values[:num_steps]
    return values


def frame_indices(n_steps: int, stride: int) -> list[int]:
    if n_steps <= 0:
        return [0]
    idxs = list(range(0, n_steps, max(1, int(stride))))
    if idxs[-1] != n_steps - 1:
        idxs.append(n_steps - 1)
    return idxs


def get_environment_preset_from_metadata(metadata: Mapping[str, Any]) -> Any:
    try:
        from experiment_specs import get_environment_preset, get_experiment_spec
    except ImportError:
        if __package__ in {None, ""}:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from experiment_specs import get_environment_preset, get_experiment_spec
        else:
            from .experiment_specs import get_environment_preset, get_experiment_spec

    preset_id = str(metadata.get("env_preset_id", "")).strip()
    if preset_id:
        return get_environment_preset(preset_id)

    exp_id = str(metadata.get("exp_id", "")).strip()
    if not exp_id:
        raise ValueError("run metadata is missing exp_id")
    exp_spec = get_experiment_spec(exp_id)
    return get_environment_preset(exp_spec.env_preset_id)


def reconstruct_loglinear_rate_model(
    metadata: Mapping[str, Any],
    *,
    obs_dim: int = 50,
    latent_dim: int = 2,
    dt_default: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, float]:
    import torch

    env_preset = get_environment_preset_from_metadata(metadata)
    seed = int(metadata.get("seed", 0))
    dt = float(metadata.get("dt", getattr(env_preset, "dt", dt_default)))
    torch.manual_seed(seed)
    layer = torch.nn.Linear(int(latent_dim), int(obs_dim))
    c = layer.weight.detach().clone()
    if bool(getattr(env_preset, "asymmetric_loading", False)):
        c[:, 0] = torch.abs(c[:, 0])
        if int(latent_dim) > 1:
            c[:, 1] = c[:, 1] * 2.0
    mean_firing = float(
        metadata.get(
            "mean_firing_rate_target",
            getattr(
                env_preset,
                "mean_firing_rate_target",
                50.0 * float(getattr(env_preset, "firing_rate_scale", 1.0)),
            ),
        )
    )
    max_firing_rate = float(
        metadata.get(
            "max_firing_rate_target",
            getattr(
                env_preset,
                "max_firing_rate_target",
                100.0 * float(getattr(env_preset, "firing_rate_scale", 1.0)),
            ),
        )
    )
    state_range_for_cap = 5.0
    mean_log_rate = torch.log(torch.full((obs_dim,), mean_firing, dtype=torch.float32))
    max_log_rate = torch.log(torch.full((obs_dim,), max_firing_rate, dtype=torch.float32))
    for _ in range(6):
        c_row_l1 = torch.sum(torch.abs(c), dim=1)
        c_row_l2_sq = torch.sum(c * c, dim=1)
        bias_from_mean = mean_log_rate - 0.5 * c_row_l2_sq
        capped_log_rate = state_range_for_cap * c_row_l1 + bias_from_mean
        if torch.all(capped_log_rate <= max_log_rate):
            break
        safe_den = torch.clamp(state_range_for_cap * c_row_l1, min=1e-8)
        row_scale = torch.clamp(
            (max_log_rate - bias_from_mean) / safe_den,
            min=0.0,
            max=1.0,
        )
        c = c * row_scale.unsqueeze(1)
    bias = mean_log_rate - 0.5 * torch.sum(c * c, dim=1)
    return c.cpu().numpy(), bias.cpu().numpy(), dt


def expected_loglinear_rate_hz(
    latent_state: Any,
    *,
    weights: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    latent = np.asarray(latent_state, dtype=np.float32)
    if latent.ndim == 1:
        latent = latent.reshape(1, -1)
    log_rate_hz = latent @ np.asarray(weights, dtype=np.float32).T + np.asarray(
        bias, dtype=np.float32
    ).reshape(1, -1)
    return np.exp(np.clip(log_rate_hz, -20.0, 20.0)).astype(np.float32)
