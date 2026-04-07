from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

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


def test_tbme_runner_parser_accepts_expected_args():
    module = _load_module("tbme_run_exp1_v1", "experiments/tbme/run_exp1.py")
    captured: dict[str, object] = {}

    def _fake_run_family(*, argv, suite_ids, default_base_dir):
        captured["argv"] = list(argv)
        captured["suite_ids"] = list(suite_ids)
        captured["default_base_dir"] = default_base_dir
        return 17

    module.run_family = _fake_run_family
    exit_code = module.main(["--mode", "summary", "--seeds", "0,10", "--base-dir", "results/tbme"])
    assert exit_code == 17
    assert captured["argv"] == ["--mode", "summary", "--seeds", "0,10", "--base-dir", "results/tbme"]
    assert captured["suite_ids"] == [
        "tbme_exp1_duffing_policy",
        "tbme_exp1_pendulum_policy",
        "tbme_exp1_double_integrator_policy",
        "tbme_exp1_objective_duffing",
    ]
    assert captured["default_base_dir"] == "results/tbme/exp1"


def test_tbme_experiment_specs_define_expected_matrices():
    module = _load_module("tbme_specs_v1", "experiments/tbme/experiment_specs.py")
    assert Path(module.DEFAULT_ENV_CATALOG_PATH).exists()
    assert Path(module.DEFAULT_MODEL_CATALOG_PATH).exists()
    assert Path(module.DEFAULT_SUITE_CATALOG_PATH).exists()
    assert module.ENVIRONMENT_PRESETS["tbme_pendulum_damped"].system_id == "damped_pendulum"
    assert module.ENVIRONMENT_PRESETS["tbme_double_integrator"].system_id == "double_integrator"
    assert (
        module.ENVIRONMENT_PRESETS["tbme_duffing_family_mismatch"].estimator_system_id
        == "damped_pendulum"
    )
    assert module.ENVIRONMENT_PRESETS["tbme_mcrtt_spikes"].real_data is True
    assert module.POLICY_SPECS["baseline_prbs"].policy_type == "baseline-prbs"
    assert module.POLICY_SPECS["e_optimality"].objective_kind == "e_optimality"
    assert module.EXPERIMENT_SPECS["tbme_exp1_duffing_policy"].policy_ids == (
        "active_myopic",
        "active_planning",
        "baseline_prbs",
        "random",
        "off_policy",
    )
    assert (
        module.EXPERIMENT_SPECS["tbme_exp2_robustness_duffing"].env_preset_id
        == "tbme_duffing_family_mismatch"
    )
    assert module.EXPERIMENT_SPECS["tbme_exp3_realdata_policy"].experiment_kind == "realdata"


def test_tbme_runtime_config_respects_catalog_policy_type_for_prbs(tmp_path: Path):
    specs = _load_module("tbme_specs_runtime_prbs", "experiments/tbme/experiment_specs.py")
    runner = _load_module("experiment_runner_runtime_prbs", "experiments/run_experiments.py")
    env_preset = specs.get_environment_preset("tbme_duffing_easy")
    schedule_spec = specs.get_schedule_spec("u1_r1_h1")
    policy_spec = specs.get_policy_spec("baseline_prbs")
    cfg = runner._build_runtime_experiment_config(
        run_dir=tmp_path / "run",
        seed=3,
        total_steps=50,
        experiment_kind="duffing",
        policy_id="baseline_prbs",
        env_preset=env_preset,
        schedule_spec=schedule_spec,
        policy_spec=policy_spec,
    )
    assert cfg.policy.policy_type == "baseline-prbs"


def test_tbme_family_scripts_define_expected_suite_sets():
    exp2 = _load_module("tbme_run_exp2_v1", "experiments/tbme/run_exp2.py")
    exp3 = _load_module("tbme_run_exp3_v1", "experiments/tbme/run_exp3.py")
    assert exp2.EXP2_SUITES == [
        "tbme_exp2_robustness_duffing",
        "tbme_exp2_robustness_pendulum",
        "tbme_exp2_robustness_double_integrator",
    ]
    assert exp3.EXP3_SUITES == ["tbme_exp3_realdata_policy"]


def test_root_runner_catalog_preparse_does_not_treat_mode_as_model_catalog():
    module = _load_module("experiment_runner_catalog_preparse", "experiments/run_experiments.py")
    captured: dict[str, object] = {}

    def _fake_configure_catalogs(*, env_catalog_paths=None, model_catalog_paths=None, suite_catalog_paths=None):
        captured["env_catalog_paths"] = env_catalog_paths
        captured["model_catalog_paths"] = model_catalog_paths
        captured["suite_catalog_paths"] = suite_catalog_paths
        raise RuntimeError("stop after catalog capture")

    module.configure_catalogs = _fake_configure_catalogs
    with pytest.raises(RuntimeError, match="stop after catalog capture"):
        module.main(
            [
                "--env-catalog",
                "experiments/experiment_env.yaml",
                "--env-catalog",
                "experiments/tbme/experiment_env.yaml",
                "--model-catalog",
                "experiments/experiment_model.yaml",
                "--model-catalog",
                "experiments/tbme/experiment_model.yaml",
                "--suite-catalog",
                "experiments/tbme/experiment_suite.yaml",
                "--mode",
                "summary",
            ]
        )

    assert captured["env_catalog_paths"] == [
        "experiments/experiment_env.yaml",
        "experiments/tbme/experiment_env.yaml",
    ]
    assert captured["model_catalog_paths"] == [
        "experiments/experiment_model.yaml",
        "experiments/tbme/experiment_model.yaml",
    ]
    assert captured["suite_catalog_paths"] == ["experiments/tbme/experiment_suite.yaml"]
