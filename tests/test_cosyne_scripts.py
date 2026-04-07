from __future__ import annotations

import csv
import importlib.util
import json
import numpy as np
from pathlib import Path
import sys


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


def _write_run_fixture(
    base_dir: Path,
    *,
    exp_id: str,
    policy_id: str,
    seed: int,
    repeat: int = 1,
    final_value: float = 0.1,
    trace_name: str = "parameter_error_trace.csv",
    trace_value_key: str = "parameter_error",
    metadata_value_key: str = "embedding_error_final",
) -> None:
    run_dir = base_dir / exp_id / "track" / policy_id / f"seed_{seed}" / f"repeat_{repeat:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "commit": "deadbee",
        "seed": seed,
        "exp_id": exp_id,
        "policy_id": policy_id,
        "total_steps": 1000,
        "base_dir": str(run_dir),
        "status": "completed",
        "start_time": "2026-03-14T00:00:00Z",
        "end_time": "2026-03-14T00:01:00Z",
        "runtime_sec": 60.0,
        metadata_value_key: final_value,
        trace_name.replace(".csv", "_path"): str(run_dir / trace_name),
        "trajectory_r2_trace_path": str(run_dir / "trajectory_r2_trace.csv"),
        "writing_ref": "docs/active-dynamics-writing/methods.tex",
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with (run_dir / trace_name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "cpu_time_sec", trace_value_key])
        writer.writeheader()
        writer.writerow({"step": 1, "cpu_time_sec": 1.0, trace_value_key: final_value + 0.1})
        writer.writerow({"step": 2, "cpu_time_sec": 2.0, trace_value_key: final_value})
    with (run_dir / "trajectory_r2_trace.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "cpu_time_sec", "trajectory_r2"])
        writer.writeheader()
        writer.writerow({"step": 1, "cpu_time_sec": 1.0, "trajectory_r2": 0.1})
        writer.writerow({"step": 2, "cpu_time_sec": 2.0, "trajectory_r2": 0.2})


def test_run_experiments_parser_accepts_expected_args():
    module = _load_module("experiment_runner_v2", "experiments/run_experiments.py")
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--exp-id",
            "exp01_1",
            "--mode",
            "all",
            "--seeds",
            "0,10",
            "--repeats",
            "1",
            "--sampling-variance-samples",
            "12",
            "--base-dir",
            "results/cosyne",
        ]
    )
    assert args.exp_id == "exp01_1"
    assert args.mode == "all"
    assert args.seeds == "0,10"
    assert args.repeats == 1
    assert args.sampling_variance_samples == 12
    assert args.base_dir == "results/cosyne"


def test_session_root_resolution_defaults_to_latest_or_next(tmp_path: Path):
    module = _load_module("cosyne_common_v2", "experiments/cosyne/cosyne_common.py")
    root = tmp_path / "results"
    root.mkdir()
    (root / "session_1").mkdir()
    (root / "session_3").mkdir()
    latest = module.resolve_session_root(root, create=False, exp_ids=["exp01_1"])
    assert latest == root / "session_3"
    created = module.resolve_session_root(root, create=True, exp_ids=["exp01_1"])
    assert created == root / "session_4"
    assert created.exists()


def test_experiment_specs_define_expected_matrices():
    module = _load_module("cosyne_specs_v2", "experiments/cosyne/experiment_specs.py")
    assert Path(module.DEFAULT_ENV_CATALOG_PATH).exists()
    assert Path(module.DEFAULT_MODEL_CATALOG_PATH).exists()
    assert Path(module.DEFAULT_SUITE_CATALOG_PATH).exists()
    assert module.ENVIRONMENT_PRESETS["hard_duffing"].firing_rate_scale == 0.2
    assert module.ENVIRONMENT_PRESETS["hard_duffing"].action_max == 1.5
    assert module.SCHEDULE_SPECS["u5_r5_h40"].planning_horizon == 40
    assert module.SCHEDULE_SPECS["u10_r10_h20"].planning_chunk == 10
    assert module.SCHEDULE_SPECS["u10_r10_h40"].predictive_only_window is True
    assert module.EXPERIMENT_SPECS["exp01_1"].policy_ids == (
        "active_myopic",
        "active_planning",
        "active_planning_update",
        "random",
        "off_policy",
    )
    assert module.EXPERIMENT_SPECS["exp01_2"].policy_ids == (
        "fully_observable",
        "state_information",
        "dynamics",
        "sampling_variance",
        "active_planning",
    )
    assert module.EXPERIMENT_SPECS["exp02_1"].policy_ids == (
        "fully_observable_u5_r5_h40",
        "state_information_u5_r5_h40",
        "dynamics_u5_r5_h40",
        "active_myopic",
        "active_planning_update_u5_r5_h40",
    )
    assert module.EXPERIMENT_SPECS["exp02_2"].policy_ids == (
        "u5_r5_h20",
        "u10_r10_h20",
        "u5_r5_h40",
        "u10_r10_h40",
    )
    assert module.EXPERIMENT_SPECS["exp01_3"].policy_ids == (
        "active_planning_u5_r5_h20",
        "sampling_variance_u5_r5_h20",
    )
    assert module.EXPERIMENT_SPECS["exp02_3"].policy_ids == (
        "active_planning_update_u5_r5_h20",
        "sampling_variance_u5_r5_h20",
        "active_planning_update_u10_r10_h40",
        "sampling_variance_u10_r10_h40",
    )
    assert module.EXPERIMENT_SPECS["exp03"].policy_ids == (
        "active_planning",
        "random",
        "off_policy",
    )
    assert "tbme_duffing_easy" not in module.ENVIRONMENT_PRESETS
    assert "baseline_prbs" not in module.POLICY_SPECS
    assert "tbme_exp1_duffing_policy" not in module.EXPERIMENT_SPECS
    assert module.EXPERIMENT_SPECS["exp01_1"].model_ids == module.EXPERIMENT_SPECS["exp01_1"].policy_ids


def test_runtime_experiment_config_is_built_from_catalog_defaults(tmp_path: Path):
    specs = _load_module("cosyne_specs_runtime", "experiments/cosyne/experiment_specs.py")
    runner = _load_module("experiment_runner_runtime", "experiments/run_experiments.py")
    env_preset = specs.get_environment_preset("hard_duffing")
    schedule_spec = specs.get_schedule_spec("u5_r5_h40")
    cfg = runner._build_runtime_experiment_config(
        run_dir=tmp_path / "run",
        seed=7,
        total_steps=123,
        experiment_kind="duffing",
        policy_id="active_planning_update_u5_r5_h40",
        env_preset=env_preset,
        schedule_spec=schedule_spec,
    )
    assert cfg.seed == 7
    assert cfg.results_dir == str(tmp_path / "run")
    assert cfg.dt == env_preset.dt
    assert cfg.action_dim == env_preset.action_dim
    assert cfg.observation_dim == env_preset.observation_dim
    assert cfg.environment.env_action_bounds == [-env_preset.action_max, env_preset.action_max]
    assert cfg.environment.obs_noise_type == env_preset.observation_noise_type
    assert cfg.training.total_steps == 123
    assert cfg.training.train_every == 124
    assert cfg.policy.policy_type == "mpc-icem"

def test_rbf_active_indices_use_manhattan_radius():
    module = _load_module("experiment_runner_v2_rbf", "experiments/run_experiments.py")
    axis = np.linspace(-2.0, 2.0, 5)
    idxs = module._rbf_active_indices(np.asarray([0.0, 0.0]), axis, 2)
    pairs = [(int(idx) // axis.size, int(idx) % axis.size) for idx in idxs]
    assert len(pairs) == 13
    assert all(abs(i - 2) + abs(j - 2) <= 2 for i, j in pairs)


def test_summarizer_fails_on_missing_expected_matrix(tmp_path: Path):
    module = _load_module("cosyne_summary_v2_missing", "experiments/cosyne/summarize_experiments.py")
    base_dir = tmp_path / "results"
    _write_run_fixture(base_dir, exp_id="exp01_1", policy_id="active_myopic", seed=0)
    exit_code = module.main(
        [
            "--base-dir",
            str(base_dir),
            "--exp-id",
            "exp01_1",
            "--seeds",
            "0,10",
            "--fail-on-missing",
        ]
    )
    assert exit_code == 1


def test_summarizer_writes_parameter_error_outputs(tmp_path: Path):
    module = _load_module("cosyne_summary_v2_param", "experiments/cosyne/summarize_experiments.py")
    base_dir = tmp_path / "results" / "session_1"
    for policy_id in ("active_myopic", "active_planning", "active_planning_update", "random", "off_policy"):
        for seed in (0, 10):
            _write_run_fixture(
                base_dir,
                exp_id="exp01_1",
                policy_id=policy_id,
                seed=seed,
                final_value=0.12 if seed == 0 else 0.16,
            )
    exit_code = module.main(
        [
            "--base-dir",
            str(base_dir.parent),
            "--exp-id",
            "exp01_1",
            "--seeds",
            "0,10",
        ]
    )
    assert exit_code == 0
    summary_dir = base_dir / "exp01_1" / "summary"
    assert (summary_dir / "metrics.csv").exists()
    assert (summary_dir / "metrics.md").exists()
    assert (summary_dir / "parameter_error_over_steps.csv").exists()
    assert (summary_dir / "trajectory_r2_over_steps.csv").exists()
    assert (summary_dir / "figures" / "parameter_error_over_cpu_time.pdf").exists()
    assert not (summary_dir / "figures" / "parameter_error_over_cpu_time.svg").exists()
    assert not (summary_dir / "figures" / "parameter_error_over_cpu_time.png").exists()
    with (summary_dir / "parameter_error_over_steps.csv").open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows
    assert "value_sem" in rows[0]
    assert "value_std" not in rows[0]
    assert np.isclose(float(rows[0]["value_sem"]), 0.02)


def test_summarizer_writes_dynamics_mse_outputs(tmp_path: Path):
    module = _load_module("cosyne_summary_v2_dyn", "experiments/cosyne/summarize_experiments.py")
    base_dir = tmp_path / "results" / "session_1"
    for policy_id in ("active_planning", "random", "off_policy"):
        for seed in (0, 10):
            _write_run_fixture(
                base_dir,
                exp_id="exp03",
                policy_id=policy_id,
                seed=seed,
                final_value=0.25,
                trace_name="dynamics_mse_trace.csv",
                trace_value_key="dynamics_mse",
                metadata_value_key="dynamics_mse_final",
            )
    exit_code = module.main(
        [
            "--base-dir",
            str(base_dir.parent),
            "--exp-id",
            "exp03",
            "--seeds",
            "0,10",
            "--figure-formats",
            "svg,pdf",
        ]
    )
    assert exit_code == 0
    summary_dir = base_dir / "exp03" / "summary"
    assert (summary_dir / "dynamics_mse_over_steps.csv").exists()
    assert (summary_dir / "figures" / "dynamics_mse_over_cpu_time.svg").exists()
    assert (summary_dir / "figures" / "dynamics_mse_over_cpu_time.pdf").exists()
    assert not (summary_dir / "figures" / "dynamics_mse_over_cpu_time.png").exists()
