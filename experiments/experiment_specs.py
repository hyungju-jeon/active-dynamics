from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Literal

import yaml


ObjectiveKind = Literal[
    "parameter_eig",
    "e_optimality",
    "fully_observable_parameter_eig",
    "state_information",
    "dynamics",
    "sampling_variance",
]
ExperimentKind = Literal["duffing", "rbf", "realdata"]
SummaryValueKind = Literal["parameter_error", "dynamics_mse"]


@dataclass(frozen=True)
class EnvironmentPreset:
    preset_id: str
    system_id: str
    asymmetric_loading: bool
    observation_primary_scale: float
    observation_secondary_scale: float
    observation_row_skew: float
    firing_rate_scale: float
    action_max: float
    system_label: str | None = None
    estimator_system_id: str | None = None
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
    real_data: bool = False
    dataset_id: str | None = None
    dataset_path: str | None = None
    state_key: str = "behavior"
    observation_key: str = "spikes"
    train_fraction: float = 0.7
    time_bin_ms: float = 20.0
    max_observation_dim: int | None = None


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
    policy_type: str | None = None
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


@dataclass(frozen=True)
class ExperimentCatalogBundle:
    environment_catalog_paths: tuple[Path, ...]
    model_catalog_paths: tuple[Path, ...]
    suite_catalog_paths: tuple[Path, ...]
    environment_presets: dict[str, EnvironmentPreset]
    schedule_specs: dict[str, ScheduleSpec]
    policy_specs: dict[str, PolicySpec]
    experiment_specs: dict[str, ExperimentSpec]


CATALOG_DIR = Path(__file__).resolve().parent
COSYNE_CATALOG_DIR = CATALOG_DIR / "cosyne"
DEFAULT_ENV_CATALOG_PATHS = (CATALOG_DIR / "experiment_env.yaml",)
DEFAULT_MODEL_CATALOG_PATHS = (CATALOG_DIR / "experiment_model.yaml",)
DEFAULT_SUITE_CATALOG_PATHS = (COSYNE_CATALOG_DIR / "experiment_suite.yaml",)
DEFAULT_ENV_CATALOG_PATH = DEFAULT_ENV_CATALOG_PATHS[0]
DEFAULT_MODEL_CATALOG_PATH = DEFAULT_MODEL_CATALOG_PATHS[0]
DEFAULT_SUITE_CATALOG_PATH = DEFAULT_SUITE_CATALOG_PATHS[0]
ENV_CATALOGS_ENVVAR = "ACTDYN_ENV_CATALOGS"
MODEL_CATALOGS_ENVVAR = "ACTDYN_MODEL_CATALOGS"
SUITE_CATALOGS_ENVVAR = "ACTDYN_SUITE_CATALOGS"


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


def _coerce_catalog_paths(
    paths: Path | str | list[Path | str] | tuple[Path | str, ...],
) -> tuple[Path, ...]:
    if isinstance(paths, (str, Path)):
        items = [paths]
    else:
        items = list(paths)
    out: list[Path] = []
    for item in items:
        path = Path(item).expanduser()
        out.append(path if path.is_absolute() else path.resolve())
    return tuple(out)


def _catalog_paths_from_env(
    env_var: str,
    default_paths: tuple[Path, ...],
) -> tuple[Path, ...]:
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return tuple(path.resolve() for path in default_paths)
    parts = [item.strip() for item in raw.split(os.pathsep) if item.strip()]
    return _coerce_catalog_paths(parts)


def _merge_section_entries(
    catalog_paths: tuple[Path, ...],
    *,
    section_name: str,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path in catalog_paths:
        payload = _load_yaml_mapping(path)
        section = _require_mapping(payload.get(section_name, {}), label=f"{path}:{section_name}")
        merged.update(dict(section))
    return merged


def _format_catalog_paths(paths: tuple[Path, ...]) -> str | list[str]:
    rendered = [str(path) for path in paths]
    if len(rendered) == 1:
        return rendered[0]
    return rendered


def load_catalog_bundle(
    *,
    env_catalog_paths: Path | str | list[Path | str] | tuple[Path | str, ...],
    model_catalog_paths: Path | str | list[Path | str] | tuple[Path | str, ...],
    suite_catalog_paths: Path | str | list[Path | str] | tuple[Path | str, ...],
) -> ExperimentCatalogBundle:
    env_paths = _coerce_catalog_paths(env_catalog_paths)
    model_paths = _coerce_catalog_paths(model_catalog_paths)
    suite_paths = _coerce_catalog_paths(suite_catalog_paths)

    environment_raw = _resolve_named_entries(
        _merge_section_entries(env_paths, section_name="environments"),
        section_name="environments",
    )
    schedule_raw = _resolve_named_entries(
        _merge_section_entries(model_paths, section_name="schedules"),
        section_name="schedules",
    )
    model_raw = _resolve_named_entries(
        _merge_section_entries(model_paths, section_name="models"),
        section_name="models",
    )
    suite_raw = _resolve_named_entries(
        _merge_section_entries(suite_paths, section_name="suites"),
        section_name="suites",
    )

    environment_presets: dict[str, EnvironmentPreset] = {
        preset_id: EnvironmentPreset(
            preset_id=str(spec.get("preset_id", preset_id)),
            system_id=str(spec["system_id"]),
            system_label=(
                None if spec.get("system_label") is None else str(spec.get("system_label"))
            ),
            estimator_system_id=(
                None
                if spec.get("estimator_system_id") is None
                else str(spec.get("estimator_system_id"))
            ),
            asymmetric_loading=bool(spec.get("asymmetric_loading", False)),
            observation_primary_scale=float(spec.get("observation_primary_scale", 1.0)),
            observation_secondary_scale=float(spec.get("observation_secondary_scale", 2.0)),
            observation_row_skew=float(spec.get("observation_row_skew", 0.0)),
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
            real_data=bool(spec.get("real_data", False)),
            dataset_id=(
                None if spec.get("dataset_id") is None else str(spec.get("dataset_id"))
            ),
            dataset_path=(
                None if spec.get("dataset_path") is None else str(spec.get("dataset_path"))
            ),
            state_key=str(spec.get("state_key", "behavior")),
            observation_key=str(spec.get("observation_key", "spikes")),
            train_fraction=float(spec.get("train_fraction", 0.7)),
            time_bin_ms=float(spec.get("time_bin_ms", 20.0)),
            max_observation_dim=(
                None
                if spec.get("max_observation_dim") is None
                else int(spec.get("max_observation_dim"))
            ),
        )
        for preset_id, spec in environment_raw.items()
    }

    schedule_specs: dict[str, ScheduleSpec] = {
        schedule_id: ScheduleSpec(
            schedule_id=str(spec.get("schedule_id", schedule_id)),
            update_interval=int(spec["update_interval"]),
            replan_interval=int(spec["replan_interval"]),
            planning_horizon=int(spec["planning_horizon"]),
            planning_chunk=int(spec["planning_chunk"]),
            predictive_only_window=bool(spec.get("predictive_only_window", False)),
        )
        for schedule_id, spec in schedule_raw.items()
    }

    policy_specs: dict[str, PolicySpec] = {
        policy_id: PolicySpec(
            policy_id=str(spec.get("policy_id", policy_id)),
            objective_kind=(
                None
                if spec.get("objective_kind") is None
                else str(spec.get("objective_kind"))
            ),
            schedule_id=str(spec["schedule_id"]),
            policy_type=(
                None if spec.get("policy_type") is None else str(spec.get("policy_type"))
            ),
            passive=bool(spec.get("passive", False)),
            save_acq_map=bool(spec.get("save_acq_map", True)),
        )
        for policy_id, spec in model_raw.items()
    }

    experiment_specs: dict[str, ExperimentSpec] = {
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
        for exp_id, spec in suite_raw.items()
    }

    return ExperimentCatalogBundle(
        environment_catalog_paths=env_paths,
        model_catalog_paths=model_paths,
        suite_catalog_paths=suite_paths,
        environment_presets=environment_presets,
        schedule_specs=schedule_specs,
        policy_specs=policy_specs,
        experiment_specs=experiment_specs,
    )


def configure_catalogs(
    *,
    env_catalog_paths: Path | str | list[Path | str] | tuple[Path | str, ...] | None = None,
    model_catalog_paths: Path | str | list[Path | str] | tuple[Path | str, ...] | None = None,
    suite_catalog_paths: Path | str | list[Path | str] | tuple[Path | str, ...] | None = None,
) -> ExperimentCatalogBundle:
    global _CATALOGS, ENVIRONMENT_PRESETS, SCHEDULE_SPECS, POLICY_SPECS, MODEL_SPECS, EXPERIMENT_SPECS

    resolved_env_paths = (
        _catalog_paths_from_env(ENV_CATALOGS_ENVVAR, DEFAULT_ENV_CATALOG_PATHS)
        if env_catalog_paths is None
        else _coerce_catalog_paths(env_catalog_paths)
    )
    resolved_model_paths = (
        _catalog_paths_from_env(MODEL_CATALOGS_ENVVAR, DEFAULT_MODEL_CATALOG_PATHS)
        if model_catalog_paths is None
        else _coerce_catalog_paths(model_catalog_paths)
    )
    resolved_suite_paths = (
        _catalog_paths_from_env(SUITE_CATALOGS_ENVVAR, DEFAULT_SUITE_CATALOG_PATHS)
        if suite_catalog_paths is None
        else _coerce_catalog_paths(suite_catalog_paths)
    )
    _CATALOGS = load_catalog_bundle(
        env_catalog_paths=resolved_env_paths,
        model_catalog_paths=resolved_model_paths,
        suite_catalog_paths=resolved_suite_paths,
    )
    ENVIRONMENT_PRESETS = _CATALOGS.environment_presets
    SCHEDULE_SPECS = _CATALOGS.schedule_specs
    POLICY_SPECS = _CATALOGS.policy_specs
    MODEL_SPECS = POLICY_SPECS
    EXPERIMENT_SPECS = _CATALOGS.experiment_specs
    return _CATALOGS


def describe_catalogs() -> dict[str, str | list[str]]:
    return {
        "environment": _format_catalog_paths(_CATALOGS.environment_catalog_paths),
        "model": _format_catalog_paths(_CATALOGS.model_catalog_paths),
        "suite": _format_catalog_paths(_CATALOGS.suite_catalog_paths),
    }


_CATALOGS = configure_catalogs()


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
