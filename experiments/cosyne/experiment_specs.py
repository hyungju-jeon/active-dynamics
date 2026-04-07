from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml


ObjectiveKind = Literal[
    "parameter_eig",
    "fully_observable_parameter_eig",
    "state_information",
    "dynamics",
    "sampling_variance",
]
ExperimentKind = Literal["duffing", "rbf"]
SummaryValueKind = Literal["parameter_error", "dynamics_mse"]


@dataclass(frozen=True)
class EnvironmentPreset:
    preset_id: str
    system_id: str
    asymmetric_loading: bool
    firing_rate_scale: float
    action_max: float
    dynamics_alpha: float = 1.0
    state_noise: float = 0.2
    state_init_uncertainty: float = 25.0
    x_range: float = 5.0
    dt: float = 0.01
    action_dim: int = 2
    latent_dim: int = 2
    embedding_dim: int = 2
    observation_dim: int = 50
    observation_noise_scale: float = 0.1
    observation_noise_type: str = "poisson"
    mean_firing_rate_target: float = 25.0
    max_firing_rate_target: float = 100.0


@dataclass(frozen=True)
class ScheduleSpec:
    schedule_id: str
    update_interval: int
    replan_interval: int
    planning_horizon: int
    planning_chunk: int
    predictive_only_window: bool = False


@dataclass(frozen=True)
class PolicySpec:
    policy_id: str
    objective_kind: ObjectiveKind | None
    schedule_id: str
    passive: bool = False
    save_acq_map: bool = True


@dataclass(frozen=True)
class ExperimentSpec:
    exp_id: str
    experiment_kind: ExperimentKind
    total_steps: int
    env_preset_id: str
    policy_ids: tuple[str, ...]
    summary_value_kind: SummaryValueKind
    summary_value_label: str
    trajectory_eval_interval: int = 100
    trajectory_eval_horizon: int = 100
    trajectory_eval_samples: int = 16

    @property
    def model_ids(self) -> tuple[str, ...]:
        return self.policy_ids


CATALOG_DIR = Path(__file__).resolve().parent
DEFAULT_ENV_CATALOG_PATH = CATALOG_DIR / "experiment_env.yaml"
DEFAULT_MODEL_CATALOG_PATH = CATALOG_DIR / "experiment_model.yaml"
DEFAULT_SUITE_CATALOG_PATH = CATALOG_DIR / "experiment_suite.yaml"


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected mapping at {path}")
    return dict(payload)


def _require_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected mapping for {label}")
    return dict(value)


def _resolve_named_entries(
    raw_entries: Mapping[str, Any],
    *,
    section_name: str,
) -> dict[str, dict[str, Any]]:
    entries = {
        str(name): _require_mapping(value, label=f"{section_name}.{name}")
        for name, value in raw_entries.items()
    }
    resolved: dict[str, dict[str, Any]] = {}
    resolving: set[str] = set()

    def _resolve(name: str) -> dict[str, Any]:
        if name in resolved:
            return dict(resolved[name])
        if name in resolving:
            raise ValueError(f"Cyclic extends detected in {section_name}.{name}")
        if name not in entries:
            raise KeyError(f"Unknown {section_name} entry: {name}")
        resolving.add(name)
        payload = dict(entries[name])
        parent_name = payload.pop("extends", None)
        merged: dict[str, Any] = {}
        if parent_name is not None:
            merged.update(_resolve(str(parent_name)))
        merged.update(payload)
        resolving.remove(name)
        resolved[name] = merged
        return dict(merged)

    for key in entries:
        _resolve(key)
    return resolved


def _as_str_tuple(values: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"Expected list/tuple for {label}")
    return tuple(str(value) for value in values)


def describe_catalogs() -> dict[str, str]:
    return {
        "environment": str(DEFAULT_ENV_CATALOG_PATH),
        "model": str(DEFAULT_MODEL_CATALOG_PATH),
        "suite": str(DEFAULT_SUITE_CATALOG_PATH),
    }


_ENVIRONMENT_PAYLOAD = _load_yaml_mapping(DEFAULT_ENV_CATALOG_PATH)
_MODEL_PAYLOAD = _load_yaml_mapping(DEFAULT_MODEL_CATALOG_PATH)
_SUITE_PAYLOAD = _load_yaml_mapping(DEFAULT_SUITE_CATALOG_PATH)

_ENVIRONMENT_RAW = _resolve_named_entries(
    _require_mapping(_ENVIRONMENT_PAYLOAD.get("environments", {}), label="environments"),
    section_name="environments",
)
_SCHEDULE_RAW = _resolve_named_entries(
    _require_mapping(_MODEL_PAYLOAD.get("schedules", {}), label="schedules"),
    section_name="schedules",
)
_MODEL_RAW = _resolve_named_entries(
    _require_mapping(_MODEL_PAYLOAD.get("models", {}), label="models"),
    section_name="models",
)
_SUITE_RAW = _resolve_named_entries(
    _require_mapping(_SUITE_PAYLOAD.get("suites", {}), label="suites"),
    section_name="suites",
)


ENVIRONMENT_PRESETS: dict[str, EnvironmentPreset] = {
    preset_id: EnvironmentPreset(
        preset_id=str(spec.get("preset_id", preset_id)),
        system_id=str(spec["system_id"]),
        asymmetric_loading=bool(spec.get("asymmetric_loading", False)),
        firing_rate_scale=float(spec.get("firing_rate_scale", 1.0)),
        action_max=float(spec.get("action_max", 1.0)),
        dynamics_alpha=float(spec.get("dynamics_alpha", 1.0)),
        state_noise=float(spec.get("state_noise", 0.2)),
        state_init_uncertainty=float(spec.get("state_init_uncertainty", 25.0)),
        x_range=float(spec.get("x_range", 5.0)),
        dt=float(spec.get("dt", 0.01)),
        action_dim=int(spec.get("action_dim", 2)),
        latent_dim=int(spec.get("latent_dim", 2)),
        embedding_dim=int(spec.get("embedding_dim", 2)),
        observation_dim=int(spec.get("observation_dim", 50)),
        observation_noise_scale=float(spec.get("observation_noise_scale", 0.1)),
        observation_noise_type=str(spec.get("observation_noise_type", "poisson")),
        mean_firing_rate_target=float(spec.get("mean_firing_rate_target", 25.0)),
        max_firing_rate_target=float(spec.get("max_firing_rate_target", 100.0)),
    )
    for preset_id, spec in _ENVIRONMENT_RAW.items()
}

SCHEDULE_SPECS: dict[str, ScheduleSpec] = {
    schedule_id: ScheduleSpec(
        schedule_id=str(spec.get("schedule_id", schedule_id)),
        update_interval=int(spec["update_interval"]),
        replan_interval=int(spec["replan_interval"]),
        planning_horizon=int(spec["planning_horizon"]),
        planning_chunk=int(spec["planning_chunk"]),
        predictive_only_window=bool(spec.get("predictive_only_window", False)),
    )
    for schedule_id, spec in _SCHEDULE_RAW.items()
}

POLICY_SPECS: dict[str, PolicySpec] = {
    policy_id: PolicySpec(
        policy_id=str(spec.get("policy_id", policy_id)),
        objective_kind=(
            None
            if spec.get("objective_kind") is None
            else str(spec.get("objective_kind"))
        ),
        schedule_id=str(spec["schedule_id"]),
        passive=bool(spec.get("passive", False)),
        save_acq_map=bool(spec.get("save_acq_map", True)),
    )
    for policy_id, spec in _MODEL_RAW.items()
}
MODEL_SPECS = POLICY_SPECS

EXPERIMENT_SPECS: dict[str, ExperimentSpec] = {
    exp_id: ExperimentSpec(
        exp_id=str(spec.get("exp_id", exp_id)),
        experiment_kind=str(spec["experiment_kind"]),
        total_steps=int(spec["total_steps"]),
        env_preset_id=str(spec["env_preset_id"]),
        policy_ids=_as_str_tuple(
            spec.get("model_ids", spec.get("policy_ids")),
            label=f"suites.{exp_id}.model_ids",
        ),
        summary_value_kind=str(spec["summary_value_kind"]),
        summary_value_label=str(spec["summary_value_label"]),
        trajectory_eval_interval=int(spec.get("trajectory_eval_interval", 100)),
        trajectory_eval_horizon=int(spec.get("trajectory_eval_horizon", 100)),
        trajectory_eval_samples=int(spec.get("trajectory_eval_samples", 16)),
    )
    for exp_id, spec in _SUITE_RAW.items()
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
