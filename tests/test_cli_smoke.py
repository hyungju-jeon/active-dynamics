import json
import math
from pathlib import Path

import pytest

from actdyn import cli


def test_cli_help_commands():
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--help"])
    assert excinfo.value.code == 0

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["run", "--help"])
    assert excinfo.value.code == 0

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["sweep", "--help"])
    assert excinfo.value.code == 0


def test_cli_run_smoke_with_stubbed_setup(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dt: 0.1\n", encoding="utf-8")

    class _Experiment:
        def run(self):
            return None

        def offline_run(self):
            return None

    def _setup_experiment(_cfg):
        return _Experiment(), None, None, None

    monkeypatch.setattr(cli, "_setup_experiment_fn", _setup_experiment)

    exit_code = cli.main(
        [
            "run",
            "--config",
            str(config_path),
            "--device",
            "cpu",
            "--online",
            "--no-offline",
            "--no-analysis",
            "--results-dir",
            str(tmp_path / "results"),
        ]
    )
    assert exit_code == 0


def test_cli_analyze_smoke(tmp_path: Path):
    results_root = tmp_path / "results"
    log_file = results_root / "model_a" / "seed_1" / "logs" / "log_0.json"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(
        json.dumps(
            [
                {"step": 0, "train_elbo": -1.0},
                {"step": 1, "train_elbo": -0.7},
            ]
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(["analyze", str(results_root), "--summary"])
    assert exit_code == 0


def test_cli_analyze_save_summary(tmp_path: Path):
    results_root = tmp_path / "results"
    log_file = results_root / "model_a" / "seed_1" / "logs" / "log_0.json"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(
        json.dumps(
            [
                {"step": 0, "train_elbo": -1.0},
                {"step": 1, "train_elbo": -0.7},
            ]
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(["analyze", str(results_root), "--save-summary"])
    assert exit_code == 0
    assert (results_root / "analysis_summary.json").exists()


def test_tbme_loading_target_snr_is_available_in_session_metadata():
    import numpy as np

    from actdyn.utils.experiment_runtime import (
        DEFAULT_LOG_LINEAR_LOADING_SEED,
        DEFAULT_LOG_LINEAR_SNR_SEED,
        _normalized_zero_action_trajectories,
        compute_loglinear_loading_fisher_snr_db,
        shared_loglinear_loading,
    )
    from experiments import run as experiment_run
    from experiments.experiment_definitions import configure_catalogs
    from experiments.experiment_io import reconstruct_loglinear_rate_model
    from experiments.tbme.run_tbme_experiments import configure_tbme_catalogs

    try:
        bundle = configure_tbme_catalogs()
        env_preset = bundle.environment_presets["tbme_duffing"]
        hard_env_preset = bundle.environment_presets["tbme_duffing_hard"]
        assert env_preset.loading_fisher_snr_db is None
        assert env_preset.loading_target_snr_db == pytest.approx(-5.0)
        assert hard_env_preset.loading_target_snr_db == pytest.approx(-10.0)
        small_snr = compute_loglinear_loading_fisher_snr_db(
            env_preset,
            num_trajectories=2,
            trajectory_length=4,
        )
        assert math.isfinite(small_snr)
        assert small_snr == pytest.approx(env_preset.loading_target_snr_db, abs=1.0)

        c, b = shared_loglinear_loading(
            env_preset,
            target_snr=-8.0,
            snr_num_trajectories=2,
            snr_trajectory_length=4,
        )
        assert c.shape == (env_preset.observation_dim, env_preset.latent_dim)
        assert b.shape == (env_preset.observation_dim,)

        latents = _normalized_zero_action_trajectories(
            env_preset,
            seed=DEFAULT_LOG_LINEAR_SNR_SEED,
            num_trajectories=2,
            trajectory_length=4,
        )
        log_rates = (
            latents @ c.detach().cpu().numpy().T
            + b.detach().cpu().numpy().reshape(1, -1)
            + np.log(float(env_preset.dt))
        )
        rates = np.exp(log_rates)
        np.testing.assert_allclose(
            rates.mean(axis=0),
            float(env_preset.mean_firing_rate_target) * float(env_preset.dt),
            rtol=1e-5,
            atol=1e-6,
        )
        assert float(rates.max()) <= (
            float(env_preset.max_firing_rate_target) * float(env_preset.dt) + 1e-5
        )

        metadata = {
            "env_preset_id": env_preset.preset_id,
            "dt": env_preset.dt,
            "observation_loading_matrix": c.tolist(),
            "observation_loading_bias": b.tolist(),
        }
        weights, bias, dt = reconstruct_loglinear_rate_model(
            metadata,
            obs_dim=env_preset.observation_dim,
            latent_dim=env_preset.latent_dim,
        )
        assert dt == pytest.approx(env_preset.dt)
        assert np.allclose(weights, c.numpy())
        assert np.allclose(bias, b.numpy())

        entry = experiment_run._build_session_experiment_entry(
            exp_id="exp01_duffing",
            seeds=[0],
            repeats=1,
            total_steps_override=None,
        )

        assert entry["environment"]["loading_target_snr_db"] == pytest.approx(-5.0)
        assert entry["environment"]["loading_fisher_snr_db"] is None
        assert entry["environment"]["observation_loading_seed"] == DEFAULT_LOG_LINEAR_LOADING_SEED
        assert entry["environment"]["loading_snr_trajectory_seed"] == DEFAULT_LOG_LINEAR_SNR_SEED
        assert entry["environment"]["observation_loading_shared_across_seeds"] is True
    finally:
        configure_catalogs()
