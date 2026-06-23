from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from actdyn.utils.experiment_runtime import safe_float


def parse_csv_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def parse_csv_ints(raw: str | None) -> list[int]:
    return [int(item) for item in parse_csv_list(raw)]


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

    if exp_ids:
        has_existing_results = any((root / exp_id).is_dir() for exp_id in exp_ids)
    else:
        has_existing_results = root.exists() and any(
            child.is_dir() and child.name.startswith("exp") for child in root.iterdir()
        )
    if has_existing_results:
        return root

    sessions: list[tuple[int, Path]] = []
    if root.exists():
        for child in root.iterdir():
            if not child.is_dir() or not child.name.startswith("session_"):
                continue
            suffix = child.name.split("session_", 1)[1]
            if suffix.isdigit():
                sessions.append((int(suffix), child))
    sessions.sort(key=lambda item: item[0])

    if create:
        next_idx = sessions[-1][0] + 1 if sessions else 1
        session_root = root / f"session_{next_idx}"
        session_root.mkdir(parents=True, exist_ok=False)
        return session_root
    if sessions:
        return sessions[-1][1]
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


def get_environment_preset_from_metadata(metadata: Mapping[str, Any]) -> Any:
    try:
        from experiment_definitions import get_environment_preset, get_experiment_spec
    except ImportError:
        if __package__ in {None, ""}:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from experiment_definitions import get_environment_preset, get_experiment_spec
        else:
            from .experiment_definitions import get_environment_preset, get_experiment_spec

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
    obs_dim: int | None = None,
    latent_dim: int | None = None,
    dt_default: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, float]:
    env_preset = get_environment_preset_from_metadata(metadata)
    dt = float(metadata.get("dt", getattr(env_preset, "dt", dt_default)))
    saved_weights = metadata.get("observation_loading_matrix")
    saved_bias = metadata.get("observation_loading_bias")
    if saved_weights is not None and saved_bias is not None:
        weights = np.asarray(saved_weights, dtype=np.float32)
        bias = np.asarray(saved_bias, dtype=np.float32).reshape(-1)
        expected_obs_dim = weights.shape[0] if obs_dim is None else int(obs_dim)
        expected_latent_dim = weights.shape[1] if latent_dim is None else int(latent_dim)
        expected_shape = (expected_obs_dim, expected_latent_dim)
        if weights.shape != expected_shape:
            raise ValueError(f"Saved observation loading has shape {weights.shape}, expected {expected_shape}.")
        if bias.shape != (expected_obs_dim,):
            raise ValueError(f"Saved observation bias has shape {bias.shape}, expected {(expected_obs_dim,)}.")
        return weights, bias, dt

    from actdyn.utils.experiment_runtime import (
        DEFAULT_LOG_LINEAR_LOADING_SEED,
        DEFAULT_LOG_LINEAR_SNR_SEED,
        shared_loglinear_loading,
    )

    loading_seed = int(
        metadata.get(
            "observation_loading_seed",
            metadata.get("seed", DEFAULT_LOG_LINEAR_LOADING_SEED),
        )
    )
    snr_seed = int(metadata.get("loading_snr_trajectory_seed", DEFAULT_LOG_LINEAR_SNR_SEED))

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
    target_snr = safe_float(metadata.get("loading_target_snr_db"))
    c, bias = shared_loglinear_loading(
        env_preset,
        device="cpu",
        loading_seed=loading_seed,
        snr_seed=snr_seed,
        mean_firing_rate=mean_firing,
        max_firing_rate=max_firing_rate,
        target_snr=target_snr,
    )
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
