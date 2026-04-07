from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, rel_path: str):
    module_path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(name, None)


def test_planar_systems_include_tbme_dynamics():
    module = _load_module("cosyne_planar_tbme", "experiments/cosyne/planar_systems.py")
    assert module.has_planar_system_spec("damped_pendulum")
    assert module.has_planar_system_spec("double_integrator")

    state = np.asarray([[0.2, -0.1]], dtype=np.float32)
    pend = module.residual_np(
        "damped_pendulum",
        state,
        module.true_embedding("damped_pendulum"),
        dynamics_alpha=1.0,
    )
    integ = module.residual_np(
        "double_integrator",
        state,
        module.true_embedding("double_integrator"),
        dynamics_alpha=1.0,
    )
    assert pend.shape == (1, 2)
    assert integ.shape == (1, 2)


def test_realdata_loader_accepts_standardized_npz(tmp_path: Path):
    module = _load_module("cosyne_realdata_tbme", "experiments/cosyne/realdata_spiking.py")
    behavior = np.stack(
        [
            np.linspace(-1.0, 1.0, 32),
            np.linspace(0.5, -0.5, 32),
            np.sin(np.linspace(0.0, 1.0, 32)),
            np.cos(np.linspace(0.0, 1.0, 32)),
        ],
        axis=1,
    ).astype(np.float32)
    spikes = np.tile(np.arange(12, dtype=np.float32), (32, 1))
    path = tmp_path / "replay_fixture.npz"
    np.savez(path, behavior=behavior, spikes=spikes, dt=np.asarray([0.02], dtype=np.float32))

    dataset = module.load_replay_dataset(
        dataset_id="fixture",
        dataset_path=path,
        state_key="behavior",
        observation_key="spikes",
        latent_dim=2,
        max_observation_dim=6,
        time_bin_ms=20.0,
    )
    assert dataset.states.shape == (32, 2)
    assert dataset.spikes.shape == (32, 6)
    assert dataset.dt == pytest.approx(0.02)
    assert dataset.metadata["num_units"] == 6


def test_realdata_split_and_ridge_helpers(tmp_path: Path):
    module = _load_module("cosyne_realdata_tbme_helpers", "experiments/cosyne/realdata_spiking.py")
    x = np.stack(
        [
            np.linspace(-1.0, 1.0, 40),
            np.linspace(1.0, -1.0, 40),
        ],
        axis=1,
    )
    a_true = np.asarray([[0.85, 0.10], [-0.05, 0.90]], dtype=np.float64)
    states = np.zeros((41, 2), dtype=np.float64)
    states[:-1] = x
    states[1:] = x @ a_true
    spikes = np.abs(np.random.default_rng(0).normal(size=(41, 16))).astype(np.float32)
    path = tmp_path / "linear_replay.npz"
    np.savez(path, behavior=states.astype(np.float32), spikes=spikes, dt=np.asarray([0.02], dtype=np.float32))

    dataset = module.load_replay_dataset(
        dataset_id="linear",
        dataset_path=path,
        state_key="behavior",
        observation_key="spikes",
        latent_dim=2,
        max_observation_dim=8,
        time_bin_ms=20.0,
    )
    train_idx, eval_idx = module.split_replay_dataset(dataset, train_fraction=0.7)
    x_all, y_all = module.build_transition_matrices(dataset)
    coef = module.fit_linear_dynamics_ridge(x_all[train_idx], y_all[train_idx], ridge=1e-4)
    mse = module.evaluate_prediction_mse(x_all[eval_idx], y_all[eval_idx], coef)
    r2 = module.evaluate_prediction_r2(x_all[eval_idx], y_all[eval_idx], coef)
    assert mse < 0.2
    assert r2 > 0.6


def test_video_renderer_skips_realdata_experiment(tmp_path: Path):
    module = _load_module("cosyne_video_tbme", "experiments/cosyne/render_experiment_videos.py")
    exit_code = module.main(
        [
            "--base-dir",
            str(tmp_path / "results"),
            "--exp-id",
            "tbme_exp3_realdata_policy",
        ]
    )
    assert exit_code == 0
