from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Literal

import yaml
import numpy as np
import torch


ObjectiveKind = Literal[
    "parameter_eig",
    "shrinkage_parameter_eig",
    "ambiguity_aware_parameter_eig",
    "e_optimality",
    "fully_observable_parameter_eig",
    "state_information",
    "dynamics",
    "observation_variance",
    "corrected_observation_variance",
    "state_variance",
]
ExperimentKind = Literal["parameter", "rbf"]


@dataclass(frozen=True)
class EnvironmentPreset:
    preset_id: str
    system_id: str
    asymmetric_loading: bool
    observation_primary_scale: float
    observation_secondary_scale: float
    observation_nuisance_scale: float
    observation_row_skew: float
    observation_loading_mismatch_variance: float
    firing_rate_scale: float
    action_max: float
    observation_loading_direction_mismatch_max_deg: float = 0.0
    observation_loading_gain_mismatch_max_factor: float = 1.0
    loading_fisher_snr_db: float | None = None
    loading_target_snr_db: float | None = None
    system_label: str | None = None
    estimator_system_id: str | None = None
    dynamics_type: str | None = None
    estimator_dynamics_type: str | None = None
    true_params: tuple[float, ...] | None = None
    estimator_true_params: tuple[float, ...] | None = None
    initial_parameter_mean: float | tuple[float, ...] = 1.0
    initial_parameter_variance: float = 0.0
    state_low: tuple[float, ...] | None = None
    state_high: tuple[float, ...] | None = None
    min_embedding_dim: int | None = None
    dynamics_alpha: float = 1.0
    state_noise: float = 0.2
    state_init_uncertainty: float = 25.0
    filter_initial_state_mean: tuple[float, ...] | None = None
    trajectory_eval_state_noise: float | None = None
    trajectory_eval_state_low: tuple[float, ...] | None = None
    trajectory_eval_state_high: tuple[float, ...] | None = None
    trajectory_eval_state_indices: tuple[int, ...] | None = None
    trajectory_eval_coordinate_balanced: bool = False
    x_range: float = 5.0
    dt: float = 0.01
    action_dim: int = 2
    latent_dim: int = 2
    embedding_dim: int = 2
    observation_dim: int = 50
    observation_model: str = "log_linear"
    observation_information_diag: tuple[float, ...] | None = None
    observation_loading_design: str = "calibrated_random"
    observation_loading_gains: tuple[float, ...] | None = None
    observation_loading_repeats_per_sign: int = 1
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
    boundary_enabled: bool = False
    boundary_type: str = "none"
    boundary_radius: float | None = None
    boundary_barrier_enabled: bool = False
    boundary_projection_enabled: bool = False
    boundary_barrier_width: float = 0.5
    boundary_barrier_strength: float = 5.0
    boundary_barrier_temperature: float = 0.1
    information_boundary_visibility_enabled: bool = False
    information_boundary_margin: float = 1.0
    information_boundary_temperature: float = 0.15

    def resolved_dynamics_type(self, *, estimator: bool = False) -> str:
        configured = self.estimator_dynamics_type if estimator else self.dynamics_type
        if isinstance(configured, str) and configured.strip():
            return configured.strip()
        if estimator:
            if isinstance(self.dynamics_type, str) and self.dynamics_type.strip():
                return self.dynamics_type.strip()
            if self.estimator_system_id is not None:
                return str(self.estimator_system_id)
        return str(self.system_id)

    def resolved_true_params(self, *, estimator: bool = False) -> tuple[float, ...]:
        configured = self.estimator_true_params if estimator else self.true_params
        if configured is not None:
            return tuple(float(x) for x in configured)
        if estimator and self.true_params is not None:
            return tuple(float(x) for x in self.true_params)
        raise ValueError(
            f"Environment preset {self.preset_id} is missing {'estimator_' if estimator else ''}true_params."
        )

    def resolved_state_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if self.state_low is not None and self.state_high is not None:
            return (
                np.asarray(self.state_low, dtype=np.float32),
                np.asarray(self.state_high, dtype=np.float32),
            )
        lim = float(self.x_range)
        return (
            np.asarray([-lim, -lim], dtype=np.float32),
            np.asarray([lim, lim], dtype=np.float32),
        )

    def resolved_plot_limit(self) -> float:
        if self.x_range is not None:
            return float(self.x_range)
        low, high = self.resolved_state_bounds()
        return float(max(np.max(np.abs(low)), np.max(np.abs(high))))

    def resolved_min_embedding_dim(self) -> int:
        if self.min_embedding_dim is not None:
            return int(self.min_embedding_dim)
        return 1

    def true_embedding_vector(
        self, *, embedding_dim: int | None = None, estimator: bool = False
    ) -> np.ndarray:
        full = np.asarray(
            self.resolved_true_params(estimator=estimator), dtype=np.float32
        )
        dim = int(self.embedding_dim) if embedding_dim is None else int(embedding_dim)
        min_dim = self.resolved_min_embedding_dim()
        if dim < min_dim or dim > full.shape[0]:
            raise ValueError(
                f"embedding_dim must be in [{min_dim}, {full.shape[0]}] for {self.preset_id}, got {dim}."
            )
        return full[:dim]

    def params_from_embedding(self, embedding: Any, *, estimator: bool = False) -> Any:
        full = np.asarray(
            self.resolved_true_params(estimator=estimator), dtype=np.float32
        )
        min_dim = self.resolved_min_embedding_dim()
        if torch.is_tensor(embedding):
            e = embedding.to(dtype=torch.float32)
            if e.shape[-1] < min_dim:
                raise ValueError(
                    f"Embedding for {self.preset_id} must have at least {min_dim} coordinates, got shape {tuple(e.shape)}."
                )
            if e.shape[-1] >= full.shape[0]:
                return e[..., : full.shape[0]]
            tail = torch.as_tensor(full[e.shape[-1] :], dtype=e.dtype, device=e.device)
            if e.ndim == 1:
                return torch.cat((e, tail), dim=0)
            tail = tail.reshape(*([1] * (e.ndim - 1)), -1).expand(*e.shape[:-1], -1)
            return torch.cat((e, tail), dim=-1)
        e_np = np.asarray(embedding, dtype=np.float32)
        if e_np.shape[-1] < min_dim:
            raise ValueError(
                f"Embedding for {self.preset_id} must have at least {min_dim} coordinates, got shape {e_np.shape}."
            )
        if e_np.shape[-1] >= full.shape[0]:
            return e_np[..., : full.shape[0]]
        tail = full[e_np.shape[-1] :]
        if e_np.ndim == 1:
            return np.concatenate((e_np, tail), axis=0).astype(np.float32, copy=False)
        tail = np.broadcast_to(
            tail.reshape(*([1] * (e_np.ndim - 1)), -1),
            (*e_np.shape[:-1], tail.shape[0]),
        )
        return np.concatenate((e_np, tail.astype(np.float32, copy=False)), axis=-1)

    def initial_parameter_mean_vector(
        self, *, embedding_dim: int | None = None
    ) -> np.ndarray:
        """Return the configured initial parameter mean with shape ``(embedding_dim,)``."""
        dim = int(self.embedding_dim) if embedding_dim is None else int(embedding_dim)
        configured = self.initial_parameter_mean
        if isinstance(configured, tuple):
            if len(configured) != dim:
                raise ValueError(
                    f"initial_parameter_mean for {self.preset_id} has {len(configured)} values, "
                    f"expected {dim}."
                )
            return np.asarray(configured, dtype=np.float32)
        return np.full((dim,), float(configured), dtype=np.float32)

    def filter_initial_state_mean_vector(self) -> np.ndarray | None:
        """Return the configured latent-state prior mean with shape ``(latent_dim,)``."""
        if self.filter_initial_state_mean is None:
            return None
        mean = np.asarray(self.filter_initial_state_mean, dtype=np.float32)
        if mean.shape != (int(self.latent_dim),):
            raise ValueError(
                f"filter_initial_state_mean for {self.preset_id} has shape {mean.shape}, "
                f"expected ({self.latent_dim},)."
            )
        return mean

    def sample_initial_state(self, seed: int) -> np.ndarray:
        low, high = self.resolved_state_bounds()
        rng = np.random.default_rng(int(seed))
        return (low + (high - low) * rng.random(low.shape[0])).astype(np.float32)


@dataclass(frozen=True, init=False)
class ScheduleSpec:
    schedule_id: str
    update_interval: int
    replan_interval: int
    planning_horizon: int
    predictive_only_window: bool = False
    adaptive_cadence: bool = False
    adaptive_update_min_interval: int = 1
    adaptive_update_eig_threshold: float | None = None
    adaptive_replan_min_interval: int = 1
    adaptive_replan_state_error_threshold: float | None = None

    def __init__(
        self,
        schedule_id: str,
        update_interval: int,
        replan_interval: int | None = None,
        planning_horizon: int | None = None,
        planning_chunk_or_predictive_only_window: int | bool | None = None,
        predictive_only_window: bool = False,
        *,
        planning_interval: int | None = None,
        planning_chunk: int | None = None,
        adaptive_cadence: bool = False,
        adaptive_update_min_interval: int = 1,
        adaptive_update_eig_threshold: float | None = None,
        adaptive_replan_min_interval: int = 1,
        adaptive_replan_state_error_threshold: float | None = None,
    ) -> None:
        if planning_chunk_or_predictive_only_window is not None:
            if isinstance(planning_chunk_or_predictive_only_window, bool):
                predictive_only_window = planning_chunk_or_predictive_only_window
            elif planning_chunk is None:
                planning_chunk = int(planning_chunk_or_predictive_only_window)
            else:
                raise TypeError("ScheduleSpec got planning_chunk twice")
        aliases = {
            key: int(value)
            for key, value in {
                "replan_interval": replan_interval,
                "planning_interval": planning_interval,
                "planning_chunk": planning_chunk,
            }.items()
            if value is not None
        }
        if not aliases:
            raise TypeError("ScheduleSpec requires replan_interval")
        first_value = next(iter(aliases.values()))
        if any(value != first_value for value in aliases.values()):
            parts = ", ".join(f"{key}={value}" for key, value in aliases.items())
            raise ValueError(
                f"Schedule {schedule_id!r} has conflicting planning interval aliases: {parts}. "
                "Use a single replan_interval value."
            )
        if planning_horizon is None:
            raise TypeError("ScheduleSpec requires planning_horizon")
        object.__setattr__(self, "schedule_id", str(schedule_id))
        object.__setattr__(self, "update_interval", int(update_interval))
        object.__setattr__(self, "replan_interval", int(first_value))
        object.__setattr__(self, "planning_horizon", int(planning_horizon))
        object.__setattr__(self, "predictive_only_window", bool(predictive_only_window))
        object.__setattr__(self, "adaptive_cadence", bool(adaptive_cadence))
        object.__setattr__(
            self, "adaptive_update_min_interval", int(adaptive_update_min_interval)
        )
        object.__setattr__(
            self,
            "adaptive_update_eig_threshold",
            None
            if adaptive_update_eig_threshold is None
            else float(adaptive_update_eig_threshold),
        )
        object.__setattr__(
            self, "adaptive_replan_min_interval", int(adaptive_replan_min_interval)
        )
        object.__setattr__(
            self,
            "adaptive_replan_state_error_threshold",
            None
            if adaptive_replan_state_error_threshold is None
            else float(adaptive_replan_state_error_threshold),
        )

    @property
    def planning_interval(self) -> int:
        return int(self.replan_interval)

    @property
    def planning_chunk(self) -> int:
        return int(self.replan_interval)


@dataclass(frozen=True)
class PolicySpec:
    policy_id: str
    objective_kind: ObjectiveKind | None
    schedule_id: str
    policy_type: str | None = None
    passive: bool = False
    shrinkage_kind: str | None = None
    shrinkage_min: float | None = None
    ambiguity_temperature: float | None = None
    ensemble_kind: str | None = None
    eig_freeze_covariance: bool = False
    eig_diagonal_covariance: bool = False
    action_constraint: str | None = None
    action_radius: float | None = None
    action_cost_weight: float = 0.0
    flex_regularization: float | None = None
    flex_parameter_step_clip: float | None = None
    flex_parameter_min: float | None = None
    flex_parameter_max: float | None = None
    flex_lr: float | None = None
    use_true_state: bool = False
    coarse_dt_factor: int = 1
    coarse_action_mapping: str = "hold"
    coarse_mapping_opt_steps: int = 25
    coarse_mapping_opt_lr: float = 0.05
    async_planning: bool = False
    async_stale_tolerance: float = 0.5
    async_worker_iterations: int | None = None
    async_worker_full_interval: int | None = None
    async_worker_device: str | None = None
    async_start_after_first_plan: bool = True
    async_realtime_prefix_steps: int = 10
    async_anytime_prefix_steps: int | None = 0
    async_anytime_min_iteration: int = 1
    async_anytime_std_tolerance: float = 0.5


@dataclass(frozen=True)
class ExperimentSpec:
    exp_id: str
    experiment_kind: ExperimentKind
    total_steps: int
    env_preset_id: str
    policy_ids: tuple[str, ...]
    trajectory_eval_interval: int = 10
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
DEFAULT_ENV_CATALOG_PATHS = (CATALOG_DIR / "experiment_env.yaml",)
DEFAULT_MODEL_CATALOG_PATHS = (CATALOG_DIR / "experiment_model.yaml",)
DEFAULT_SUITE_CATALOG_PATHS: tuple[Path, ...] = ()
DEFAULT_ENV_CATALOG_PATH = DEFAULT_ENV_CATALOG_PATHS[0]
DEFAULT_MODEL_CATALOG_PATH = DEFAULT_MODEL_CATALOG_PATHS[0]
DEFAULT_SUITE_CATALOG_PATH = None
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


def _resolve_schedule_replan_interval(schedule_id: str, spec: Mapping[str, Any]) -> int:
    aliases = {
        key: int(spec[key])
        for key in ("replan_interval", "planning_interval", "planning_chunk")
        if key in spec
    }
    if not aliases:
        raise KeyError(
            f"Schedule {schedule_id!r} must define replan_interval "
            "(planning_interval is accepted as an alias)."
        )
    first_key, first_value = next(iter(aliases.items()))
    mismatched = {
        key: value for key, value in aliases.items() if int(value) != int(first_value)
    }
    if mismatched:
        parts = ", ".join(f"{key}={value}" for key, value in aliases.items())
        raise ValueError(
            f"Schedule {schedule_id!r} has conflicting planning interval aliases: {parts}. "
            "Use a single replan_interval value."
        )
    if first_key == "planning_chunk":
        # Backward compatible reader for older catalogs. New catalogs should use
        # replan_interval so there is one explicit scheduling knob.
        return int(first_value)
    return int(first_value)


def _as_str_tuple(values: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"Expected list/tuple for {label}")
    return tuple(str(value) for value in values)


def _as_float_tuple(values: Any, *, label: str) -> tuple[float, ...] | None:
    if values is None:
        return None
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"Expected list/tuple for {label}")
    return tuple(float(value) for value in values)


def _as_int_tuple(values: Any, *, label: str) -> tuple[int, ...] | None:
    if values is None:
        return None
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{label} must be a list or tuple, got {type(values).__name__}")
    return tuple(int(value) for value in values)


def _as_float_or_tuple(
    values: Any, *, label: str, default: float
) -> float | tuple[float, ...]:
    if values is None:
        return float(default)
    if isinstance(values, (list, tuple)):
        return tuple(float(value) for value in values)
    try:
        return float(values)
    except Exception as exc:
        raise ValueError(f"Expected scalar or list/tuple for {label}") from exc


def _as_state_vector(values: Any, *, label: str) -> tuple[float, ...] | None:
    parsed = _as_float_tuple(values, label=label)
    if parsed is None:
        return None
    if len(parsed) == 0:
        raise ValueError(f"Expected at least 1 value for {label}")
    return tuple(float(value) for value in parsed)


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
    if raw in {"-", "none", "None", "NONE"}:
        return ()
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
        section = _require_mapping(
            payload.get(section_name, {}), label=f"{path}:{section_name}"
        )
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
    suite_entries: Mapping[str, Any] | None = None,
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
    suite_entries_raw = _merge_section_entries(suite_paths, section_name="suites")
    if suite_entries is not None:
        suite_entries_raw.update(
            {
                str(name): _require_mapping(value, label=f"suites.{name}")
                for name, value in suite_entries.items()
            }
        )
    suite_raw = _resolve_named_entries(suite_entries_raw, section_name="suites")

    environment_presets: dict[str, EnvironmentPreset] = {
        preset_id: EnvironmentPreset(
            preset_id=str(spec.get("preset_id", preset_id)),
            system_id=str(spec["system_id"]),
            system_label=(
                None
                if spec.get("system_label") is None
                else str(spec.get("system_label"))
            ),
            estimator_system_id=(
                None
                if spec.get("estimator_system_id") is None
                else str(spec.get("estimator_system_id"))
            ),
            dynamics_type=(
                None
                if spec.get("dynamics_type") is None
                else str(spec.get("dynamics_type"))
            ),
            estimator_dynamics_type=(
                None
                if spec.get("estimator_dynamics_type") is None
                else str(spec.get("estimator_dynamics_type"))
            ),
            true_params=_as_float_tuple(
                spec.get("true_params"), label=f"environments.{preset_id}.true_params"
            ),
            estimator_true_params=_as_float_tuple(
                spec.get("estimator_true_params"),
                label=f"environments.{preset_id}.estimator_true_params",
            ),
            initial_parameter_mean=_as_float_or_tuple(
                spec.get("initial_parameter_mean"),
                label=f"environments.{preset_id}.initial_parameter_mean",
                default=1.0,
            ),
            initial_parameter_variance=float(
                spec.get("initial_parameter_variance", 0.0)
            ),
            state_low=_as_state_vector(
                spec.get("state_low"), label=f"environments.{preset_id}.state_low"
            ),
            state_high=_as_state_vector(
                spec.get("state_high"), label=f"environments.{preset_id}.state_high"
            ),
            min_embedding_dim=(
                None
                if spec.get("min_embedding_dim") is None
                else int(spec.get("min_embedding_dim"))
            ),
            asymmetric_loading=bool(spec.get("asymmetric_loading", False)),
            observation_primary_scale=float(spec.get("observation_primary_scale", 1.0)),
            observation_secondary_scale=float(
                spec.get("observation_secondary_scale", 2.0)
            ),
            observation_nuisance_scale=float(
                spec.get("observation_nuisance_scale", 1.0)
            ),
            observation_row_skew=float(spec.get("observation_row_skew", 0.0)),
            observation_loading_mismatch_variance=float(
                spec.get("observation_loading_mismatch_variance", 0.0)
            ),
            observation_loading_direction_mismatch_max_deg=float(
                spec.get("observation_loading_direction_mismatch_max_deg", 0.0)
            ),
            observation_loading_gain_mismatch_max_factor=float(
                spec.get("observation_loading_gain_mismatch_max_factor", 1.0)
            ),
            loading_fisher_snr_db=(
                None
                if spec.get("loading_fisher_snr_db") is None
                else float(spec.get("loading_fisher_snr_db"))
            ),
            loading_target_snr_db=(
                None
                if spec.get("loading_target_snr_db") is None
                else float(spec.get("loading_target_snr_db"))
            ),
            firing_rate_scale=float(spec.get("firing_rate_scale", 1.0)),
            action_max=float(spec.get("action_max", 1.0)),
            dynamics_alpha=float(spec.get("dynamics_alpha", 1.0)),
            state_noise=float(spec.get("state_noise", 0.2)),
            state_init_uncertainty=float(spec.get("state_init_uncertainty", 25.0)),
            filter_initial_state_mean=_as_state_vector(
                spec.get("filter_initial_state_mean"),
                label=f"environments.{preset_id}.filter_initial_state_mean",
            ),
            trajectory_eval_state_noise=(
                None
                if spec.get("trajectory_eval_state_noise") is None
                else float(spec.get("trajectory_eval_state_noise"))
            ),
            trajectory_eval_state_low=_as_state_vector(
                spec.get("trajectory_eval_state_low"),
                label=f"environments.{preset_id}.trajectory_eval_state_low",
            ),
            trajectory_eval_state_high=_as_state_vector(
                spec.get("trajectory_eval_state_high"),
                label=f"environments.{preset_id}.trajectory_eval_state_high",
            ),
            trajectory_eval_state_indices=_as_int_tuple(
                spec.get("trajectory_eval_state_indices"),
                label=f"environments.{preset_id}.trajectory_eval_state_indices",
            ),
            trajectory_eval_coordinate_balanced=bool(
                spec.get("trajectory_eval_coordinate_balanced", False)
            ),
            x_range=float(spec.get("x_range", 5.0)),
            dt=float(spec.get("dt", 0.01)),
            action_dim=int(spec.get("action_dim", 2)),
            latent_dim=int(spec.get("latent_dim", 2)),
            embedding_dim=int(spec.get("embedding_dim", 2)),
            observation_dim=int(spec.get("observation_dim", 50)),
            observation_model=str(spec.get("observation_model", "log_linear")),
            observation_information_diag=_as_float_tuple(
                spec.get("observation_information_diag"),
                label=f"environments.{preset_id}.observation_information_diag",
            ),
            observation_loading_design=str(
                spec.get("observation_loading_design", "calibrated_random")
            ),
            observation_loading_gains=_as_float_tuple(
                spec.get("observation_loading_gains"),
                label=f"environments.{preset_id}.observation_loading_gains",
            ),
            observation_loading_repeats_per_sign=int(
                spec.get("observation_loading_repeats_per_sign", 1)
            ),
            observation_noise_scale=float(spec.get("observation_noise_scale", 0.1)),
            observation_noise_type=str(spec.get("observation_noise_type", "poisson")),
            mean_firing_rate_target=float(spec.get("mean_firing_rate_target", 25.0)),
            max_firing_rate_target=float(spec.get("max_firing_rate_target", 100.0)),
            real_data=bool(spec.get("real_data", False)),
            dataset_id=(
                None if spec.get("dataset_id") is None else str(spec.get("dataset_id"))
            ),
            dataset_path=(
                None
                if spec.get("dataset_path") is None
                else str(spec.get("dataset_path"))
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
            boundary_enabled=bool(spec.get("boundary_enabled", False)),
            boundary_type=str(spec.get("boundary_type", "none")),
            boundary_radius=(
                None
                if spec.get("boundary_radius") is None
                else float(spec.get("boundary_radius"))
            ),
            boundary_barrier_enabled=bool(spec.get("boundary_barrier_enabled", False)),
            boundary_projection_enabled=bool(
                spec.get("boundary_projection_enabled", False)
            ),
            boundary_barrier_width=float(spec.get("boundary_barrier_width", 0.5)),
            boundary_barrier_strength=float(spec.get("boundary_barrier_strength", 5.0)),
            boundary_barrier_temperature=float(
                spec.get("boundary_barrier_temperature", 0.1)
            ),
            information_boundary_visibility_enabled=bool(
                spec.get("information_boundary_visibility_enabled", False)
            ),
            information_boundary_margin=float(
                spec.get("information_boundary_margin", 1.0)
            ),
            information_boundary_temperature=float(
                spec.get("information_boundary_temperature", 0.15)
            ),
        )
        for preset_id, spec in environment_raw.items()
    }

    schedule_specs: dict[str, ScheduleSpec] = {
        schedule_id: ScheduleSpec(
            schedule_id=str(spec.get("schedule_id", schedule_id)),
            update_interval=int(spec["update_interval"]),
            replan_interval=_resolve_schedule_replan_interval(schedule_id, spec),
            planning_horizon=int(spec["planning_horizon"]),
            predictive_only_window=bool(spec.get("predictive_only_window", False)),
            adaptive_cadence=bool(spec.get("adaptive_cadence", False)),
            adaptive_update_min_interval=int(
                spec.get("adaptive_update_min_interval", 1)
            ),
            adaptive_update_eig_threshold=(
                None
                if spec.get("adaptive_update_eig_threshold") is None
                else float(spec.get("adaptive_update_eig_threshold"))
            ),
            adaptive_replan_min_interval=int(
                spec.get("adaptive_replan_min_interval", 1)
            ),
            adaptive_replan_state_error_threshold=(
                None
                if spec.get("adaptive_replan_state_error_threshold") is None
                else float(spec.get("adaptive_replan_state_error_threshold"))
            ),
        )
        for schedule_id, spec in schedule_raw.items()
    }

    schedule_keys = {
        "update_interval",
        "replan_interval",
        "planning_interval",
        "planning_chunk",
        "planning_horizon",
        "predictive_only_window",
        "adaptive_cadence",
        "adaptive_update_min_interval",
        "adaptive_update_eig_threshold",
        "adaptive_replan_min_interval",
        "adaptive_replan_state_error_threshold",
    }
    for policy_id, spec in model_raw.items():
        if not any(key in spec for key in schedule_keys):
            continue
        schedule_id = str(spec.get("policy_id", policy_id))
        schedule_specs[schedule_id] = ScheduleSpec(
            schedule_id=schedule_id,
            update_interval=int(spec["update_interval"]),
            replan_interval=_resolve_schedule_replan_interval(schedule_id, spec),
            planning_horizon=int(spec["planning_horizon"]),
            predictive_only_window=bool(spec.get("predictive_only_window", False)),
            adaptive_cadence=bool(spec.get("adaptive_cadence", False)),
            adaptive_update_min_interval=int(
                spec.get("adaptive_update_min_interval", 1)
            ),
            adaptive_update_eig_threshold=(
                None
                if spec.get("adaptive_update_eig_threshold") is None
                else float(spec.get("adaptive_update_eig_threshold"))
            ),
            adaptive_replan_min_interval=int(
                spec.get("adaptive_replan_min_interval", 1)
            ),
            adaptive_replan_state_error_threshold=(
                None
                if spec.get("adaptive_replan_state_error_threshold") is None
                else float(spec.get("adaptive_replan_state_error_threshold"))
            ),
        )
        spec["schedule_id"] = schedule_id

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
                None
                if spec.get("policy_type") is None
                else str(spec.get("policy_type"))
            ),
            passive=bool(spec.get("passive", False)),
            shrinkage_kind=(
                None
                if spec.get("shrinkage_kind") is None
                else str(spec.get("shrinkage_kind"))
            ),
            shrinkage_min=(
                None
                if spec.get("shrinkage_min") is None
                else float(spec.get("shrinkage_min"))
            ),
            ambiguity_temperature=(
                None
                if spec.get("ambiguity_temperature") is None
                else float(spec.get("ambiguity_temperature"))
            ),
            ensemble_kind=(
                None
                if spec.get("ensemble_kind") is None
                else str(spec.get("ensemble_kind"))
            ),
            eig_freeze_covariance=bool(spec.get("eig_freeze_covariance", False)),
            eig_diagonal_covariance=bool(spec.get("eig_diagonal_covariance", False)),
            action_constraint=(
                None
                if spec.get("action_constraint") is None
                else str(spec.get("action_constraint"))
            ),
            action_radius=(
                None
                if spec.get("action_radius") is None
                else float(spec.get("action_radius"))
            ),
            action_cost_weight=float(
                spec.get(
                    "action_cost_weight",
                    PolicySpec.__dataclass_fields__["action_cost_weight"].default,
                )
            ),
            flex_regularization=(
                None
                if spec.get("flex_regularization") is None
                else float(spec.get("flex_regularization"))
            ),
            flex_parameter_step_clip=(
                None
                if spec.get("flex_parameter_step_clip") is None
                else float(spec.get("flex_parameter_step_clip"))
            ),
            flex_parameter_min=(
                None
                if spec.get("flex_parameter_min") is None
                else float(spec.get("flex_parameter_min"))
            ),
            flex_parameter_max=(
                None
                if spec.get("flex_parameter_max") is None
                else float(spec.get("flex_parameter_max"))
            ),
            flex_lr=(
                None if spec.get("flex_lr") is None else float(spec.get("flex_lr"))
            ),
            use_true_state=bool(spec.get("use_true_state", False)),
            coarse_dt_factor=int(spec.get("coarse_dt_factor", 1)),
            coarse_action_mapping=str(spec.get("coarse_action_mapping", "hold")),
            coarse_mapping_opt_steps=int(spec.get("coarse_mapping_opt_steps", 25)),
            coarse_mapping_opt_lr=float(spec.get("coarse_mapping_opt_lr", 0.05)),
            async_planning=bool(spec.get("async_planning", False)),
            async_stale_tolerance=float(spec.get("async_stale_tolerance", 0.5)),
            async_worker_iterations=(
                None
                if spec.get("async_worker_iterations") is None
                else int(spec.get("async_worker_iterations"))
            ),
            async_worker_full_interval=(
                None
                if spec.get("async_worker_full_interval") is None
                else int(spec.get("async_worker_full_interval"))
            ),
            async_worker_device=(
                None
                if spec.get("async_worker_device") is None
                else str(spec.get("async_worker_device"))
            ),
            async_start_after_first_plan=bool(
                spec.get("async_start_after_first_plan", True)
            ),
            async_realtime_prefix_steps=int(
                spec.get("async_realtime_prefix_steps", 10)
            ),
            async_anytime_prefix_steps=(
                None
                if spec.get("async_anytime_prefix_steps", 0) is None
                else int(spec.get("async_anytime_prefix_steps", 0))
            ),
            async_anytime_min_iteration=int(spec.get("async_anytime_min_iteration", 1)),
            async_anytime_std_tolerance=float(
                spec.get("async_anytime_std_tolerance", 0.5)
            ),
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
            trajectory_eval_interval=int(spec.get("trajectory_eval_interval", 10)),
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
    env_catalog_paths: Path
    | str
    | list[Path | str]
    | tuple[Path | str, ...]
    | None = None,
    model_catalog_paths: Path
    | str
    | list[Path | str]
    | tuple[Path | str, ...]
    | None = None,
    suite_catalog_paths: Path
    | str
    | list[Path | str]
    | tuple[Path | str, ...]
    | None = None,
    suite_entries: Mapping[str, Any] | None = None,
) -> ExperimentCatalogBundle:
    global \
        _CATALOGS, \
        ENVIRONMENT_PRESETS, \
        SCHEDULE_SPECS, \
        POLICY_SPECS, \
        MODEL_SPECS, \
        EXPERIMENT_SPECS

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
        suite_entries=suite_entries,
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
