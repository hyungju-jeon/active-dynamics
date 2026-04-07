from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from experiment_specs import (
        EnvironmentPreset,
        ExperimentCatalogBundle,
        ExperimentSpec,
        PolicySpec,
        ScheduleSpec,
        load_catalog_bundle,
    )
else:
    from ..experiment_specs import (
        EnvironmentPreset,
        ExperimentCatalogBundle,
        ExperimentSpec,
        PolicySpec,
        ScheduleSpec,
        load_catalog_bundle,
    )


CATALOG_DIR = Path(__file__).resolve().parent
SHARED_CATALOG_DIR = CATALOG_DIR.parent
DEFAULT_ENV_CATALOG_PATH = SHARED_CATALOG_DIR / "experiment_env.yaml"
DEFAULT_MODEL_CATALOG_PATH = SHARED_CATALOG_DIR / "experiment_model.yaml"
DEFAULT_SUITE_CATALOG_PATH = CATALOG_DIR / "experiment_suite.yaml"

_CATALOGS: ExperimentCatalogBundle = load_catalog_bundle(
    env_catalog_paths=[DEFAULT_ENV_CATALOG_PATH],
    model_catalog_paths=[DEFAULT_MODEL_CATALOG_PATH],
    suite_catalog_paths=[DEFAULT_SUITE_CATALOG_PATH],
)

ENVIRONMENT_PRESETS = _CATALOGS.environment_presets
SCHEDULE_SPECS = _CATALOGS.schedule_specs
POLICY_SPECS = _CATALOGS.policy_specs
MODEL_SPECS = POLICY_SPECS
EXPERIMENT_SPECS = _CATALOGS.experiment_specs


def describe_catalogs() -> dict[str, str]:
    return {
        "environment": str(DEFAULT_ENV_CATALOG_PATH),
        "model": str(DEFAULT_MODEL_CATALOG_PATH),
        "suite": str(DEFAULT_SUITE_CATALOG_PATH),
    }


def list_experiment_ids() -> list[str]:
    return list(EXPERIMENT_SPECS)


def get_experiment_spec(exp_id: str) -> ExperimentSpec:
    return EXPERIMENT_SPECS[exp_id]


def get_policy_spec(policy_id: str) -> PolicySpec:
    return POLICY_SPECS[policy_id]


def get_model_spec(policy_id: str) -> PolicySpec:
    return get_policy_spec(policy_id)


def get_schedule_spec(schedule_id: str) -> ScheduleSpec:
    return SCHEDULE_SPECS[schedule_id]


def get_environment_preset(preset_id: str) -> EnvironmentPreset:
    return ENVIRONMENT_PRESETS[preset_id]
