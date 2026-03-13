from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Callable, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import actdyn
import actdyn.core.experiment
import actdyn.environment
import actdyn.environment.action
import actdyn.environment.observation
import actdyn.metrics
import actdyn.metrics.cost
import actdyn.metrics.information
import actdyn.policy
import actdyn.policy.mpc
from actdyn.config import ExperimentConfig
from actdyn.models.dynamics import FunctionDynamics
from actdyn.utils.helper import jacobian_wrt_param, make_uniform_sampler
from actdyn.utils.runtime import configure_runtime, ensure_dir
from actdyn.utils.save_load import load_and_concatenate_rollouts, save_rollout

try:
    import gymnasium as gym
except ModuleNotFoundError:

    class _GymStub:
        class Env:
            pass

    gym = _GymStub()

try:
    from external.integrative_inference.experiments.model_utils import build_hypernetwork
    import external.integrative_inference.src.modules as metadyn

    HAS_INTEGRATIVE_INFERENCE = True
except ModuleNotFoundError:
    HAS_INTEGRATIVE_INFERENCE = False

    class _FallbackLowRankHypernet(nn.Module):
        def __init__(self, d_embed: int, d_context: int, d_hidden: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_embed, d_hidden),
                nn.SiLU(),
                nn.Linear(d_hidden, d_hidden),
                nn.SiLU(),
                nn.Linear(d_hidden, d_context),
            )

        def forward(self, e: torch.Tensor):
            ctx = self.net(e)
            return ctx, None

    class _FallbackHyperMlpDynamics(nn.Module):
        def __init__(
            self,
            d_latent: int,
            d_hidden: int,
            n_hidden: int,
            update_input: bool,
            update_output: bool,
            update_hidden: bool,
            du: int,
            device: str,
            d_context: int = 16,
        ):
            super().__init__()
            layers: list[nn.Module] = []
            in_dim = d_latent + d_context
            for _ in range(max(n_hidden, 1)):
                layers.append(nn.Linear(in_dim, d_hidden))
                layers.append(nn.SiLU())
                in_dim = d_hidden
            layers.append(nn.Linear(in_dim, d_latent))
            self.net = nn.Sequential(*layers)

        def compute_param(self, z: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            return self.net(torch.cat([z, out], dim=-1))

        def forward(self, z: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            return self.compute_param(z, out)

    class _FallbackMetadynModule:
        LowRankHypernet = _FallbackLowRankHypernet
        HyperMlpDynamics = _FallbackHyperMlpDynamics

    def build_hypernetwork(cfg, device):
        return _FallbackLowRankHypernet(
            d_embed=int(cfg["d_embed"]),
            d_context=int(cfg["d_context"]),
            d_hidden=max(int(cfg.get("d_hidden_hypernet_dynamics", 16)), int(cfg["d_context"])),
        ).to(device)

    metadyn = _FallbackMetadynModule()


device = "cuda" if torch.cuda.is_available() else "cpu"

CANONICAL_VECTORFIELD_SYSTEMS: dict[str, tuple[str, ...]] = {
    "mixed200": (
        "double_limit_cycle_00",
        "duffing_bistable_00",
        "duffing_single_00",
        "van_der_pol_00",
    ),
    "mixed80": (
        "double_limit_cycle_10",
        "duffing_bistable_10",
        "duffing_single_10",
        "van_der_pol_10",
    ),
    "mixed40": (
        "double_limit_cycle_10",
        "duffing_bistable_10",
        "duffing_single_10",
        "van_der_pol_10",
    ),
    "legacy4": (
        "duffing_soft",
        "duffing_hard",
        "vanderpol_soft",
        "vanderpol_hard",
    ),
    "known_duffing40": (
        "duffing_bistable_10",
        "duffing_single_10",
    ),
}
CANONICAL_VECTORFIELD_GRID_RANGE: tuple[float, float] = (-3.0, 3.0)
CANONICAL_VECTORFIELD_GRID_N: int = 25
CANONICAL_VECTORFIELD_LAYOUT: str = "families_x_(true,reconstructed)_streamplot"
CANONICAL_VECTORFIELD_STYLE: dict[str, float | str] = {
    "cmap": "viridis",
    "stream_density": 0.9,
    "line_width": 1.0,
    "arrow_size": 0.8,
    "dpi": 220,
}


@dataclass(frozen=True)
class SystemSpec:
    name: str
    family: str
    embedding: tuple[float, ...]
    params: tuple[float, float]
    note: str


LEGACY_SYSTEM_SPECS: tuple[SystemSpec, ...] = (
    SystemSpec(
        name="duffing_soft",
        family="duffing",
        embedding=(-1.2, -0.5, -0.2, 0.1),
        params=(-0.45, 0.90),
        note="damped Duffing monostable regime",
    ),
    SystemSpec(
        name="duffing_stiff",
        family="duffing",
        embedding=(-0.6, 0.9, 0.2, -0.1),
        params=(-0.35, -0.22),
        note="damped Duffing bistable regime",
    ),
    SystemSpec(
        name="vdp_mild",
        family="van_der_pol",
        embedding=(0.5, -1.0, 0.6, 0.2),
        params=(0.8, 1.0),
        note="mild Van der Pol limit cycle",
    ),
    SystemSpec(
        name="vdp_relaxation",
        family="van_der_pol",
        embedding=(1.1, 0.7, 1.0, -0.2),
        params=(2.2, 1.0),
        note="strong relaxation Van der Pol cycle",
    ),
)


def _make_family_embedding(
    family: str,
    idx: int,
    total: int = 10,
) -> tuple[float, float, float, float]:
    family_centers = {
        "duffing_single": (-1.8, -0.6, -0.8, -0.2),
        "duffing_bistable": (-0.8, 1.2, -0.2, -0.4),
        "van_der_pol": (1.4, -0.9, 0.8, 0.1),
        "double_limit_cycle": (1.1, 1.2, -0.4, 0.8),
    }
    if family not in family_centers:
        raise ValueError(f"Unknown family for embedding seed: {family}")
    c0, c1, c2, c3 = family_centers[family]
    phase = (2.0 * math.pi * idx) / max(total, 1)
    radial = 0.28
    linear = (idx - 0.5 * (total - 1)) / max(total - 1, 1)
    return (
        float(c0 + radial * math.cos(phase)),
        float(c1 + radial * math.sin(phase)),
        float(c2 + 0.22 * linear),
        float(c3 + 0.16 * ((idx % 2) * 2 - 1)),
    )


def build_mixed80_family_param_bank() -> list[tuple[str, str, list[tuple[float, float]]]]:
    return [
        (
            "duffing_single",
            "Duffing single-attractor regime (positive linear stiffness, damped)",
            [
                (-0.66, 0.48),
                (-0.64, 0.54),
                (-0.62, 0.60),
                (-0.60, 0.68),
                (-0.58, 0.76),
                (-0.56, 0.84),
                (-0.55, 0.92),
                (-0.54, 1.00),
                (-0.53, 1.08),
                (-0.52, 1.16),
                (-0.51, 1.24),
                (-0.50, 1.32),
                (-0.49, 1.40),
                (-0.48, 1.48),
                (-0.47, 1.56),
                (-0.46, 1.64),
                (-0.45, 1.72),
                (-0.44, 1.80),
                (-0.43, 1.88),
                (-0.42, 1.96),
            ],
        ),
        (
            "duffing_bistable",
            "Duffing bistable regime (negative linear stiffness, damped; speed-constrained)",
            [
                (-0.62, -0.10),
                (-0.61, -0.12),
                (-0.60, -0.14),
                (-0.59, -0.16),
                (-0.58, -0.18),
                (-0.57, -0.20),
                (-0.56, -0.22),
                (-0.55, -0.24),
                (-0.54, -0.26),
                (-0.53, -0.28),
                (-0.52, -0.30),
                (-0.51, -0.32),
                (-0.50, -0.34),
                (-0.49, -0.36),
                (-0.48, -0.38),
                (-0.47, -0.40),
                (-0.46, -0.42),
                (-0.45, -0.44),
                (-0.44, -0.46),
                (-0.43, -0.48),
            ],
        ),
        (
            "van_der_pol",
            "Van der Pol limit-cycle strength sweep (speed-constrained, fewer relaxation-extreme cases)",
            [
                (0.50, 0.92),
                (0.56, 0.94),
                (0.62, 0.96),
                (0.68, 0.98),
                (0.74, 1.00),
                (0.80, 1.02),
                (0.86, 1.04),
                (0.92, 1.06),
                (0.98, 1.08),
                (1.04, 1.10),
                (1.10, 0.96),
                (1.16, 0.98),
                (1.22, 1.00),
                (1.28, 1.02),
                (1.34, 1.04),
                (1.40, 1.06),
                (1.44, 1.08),
                (1.48, 1.10),
                (1.52, 1.11),
                (1.56, 1.12),
            ],
        ),
        (
            "double_limit_cycle",
            "double-ring limit-cycle family (speed-constrained angular/radial sweep; bidirectional rotation, capped max radius tail)",
            [
                (-0.34, 0.78),
                (-0.38, 0.82),
                (-0.42, 0.86),
                (-0.46, 0.90),
                (-0.50, 0.94),
                (-0.54, 0.98),
                (-0.58, 1.02),
                (-0.62, 1.06),
                (-0.66, 1.10),
                (-0.70, 1.13),
                (0.46, 0.88),
                (0.50, 0.92),
                (0.54, 0.96),
                (0.58, 1.00),
                (0.62, 1.04),
                (0.66, 1.08),
                (0.70, 1.12),
                (0.74, 1.14),
                (0.78, 1.16),
                (0.82, 1.18),
            ],
        ),
    ]


def build_system_specs_from_family_param_bank(
    family_param_bank: list[tuple[str, str, list[tuple[float, float]]]],
    *,
    expected_per_family: int | None = None,
    expected_total: int | None = None,
) -> tuple[SystemSpec, ...]:
    specs: list[SystemSpec] = []
    for family, note, params in family_param_bank:
        if expected_per_family is not None and len(params) != int(expected_per_family):
            raise ValueError(
                f"Family {family} must define exactly {expected_per_family} parameter sets."
            )
        for i, pair in enumerate(params):
            specs.append(
                SystemSpec(
                    name=f"{family}_{i:02d}",
                    family=family,
                    embedding=_make_family_embedding(family, i, total=len(params)),
                    params=(float(pair[0]), float(pair[1])),
                    note=note,
                )
            )
    if expected_total is not None and len(specs) != int(expected_total):
        raise ValueError(f"Expected {expected_total} systems, got {len(specs)}.")
    return tuple(specs)


def build_mixed80_system_specs() -> tuple[SystemSpec, ...]:
    return build_system_specs_from_family_param_bank(
        build_mixed80_family_param_bank(),
        expected_per_family=20,
        expected_total=80,
    )


def build_mixed200_system_specs() -> tuple[SystemSpec, ...]:
    base_bank = build_mixed80_family_param_bank()
    extra_bank: dict[str, list[tuple[float, float]]] = {
        "duffing_single": [
            (p0, p1)
            for p0 in (-0.71, -0.65, -0.59, -0.53, -0.47, -0.41)
            for p1 in (0.36, 0.79, 1.13, 1.57, 2.01)
        ],
        "duffing_bistable": [
            (p0, p1)
            for p0 in (-0.665, -0.605, -0.545, -0.485, -0.425, -0.385)
            for p1 in (-0.11, -0.19, -0.29, -0.41, -0.53)
        ],
        "van_der_pol": [
            (p0, p1)
            for p0 in (0.38, 0.64, 0.90, 1.14, 1.38, 1.62)
            for p1 in (0.90, 0.98, 1.06, 1.14, 1.20)
        ],
        "double_limit_cycle": (
            [(p0, p1) for p0 in (-0.86, -0.75, -0.64, -0.53, -0.42) for p1 in (0.74, 0.91, 1.08)]
            + [(p0, p1) for p0 in (0.44, 0.57, 0.70, 0.83, 0.96) for p1 in (0.74, 0.91, 1.08)]
        ),
    }
    family_param_bank: list[tuple[str, str, list[tuple[float, float]]]] = []
    for family, note, base_params in base_bank:
        params = list(base_params) + list(extra_bank[family])
        if len(params) != 50:
            raise ValueError(f"Family {family} must define exactly 50 parameter sets for mixed200.")
        family_param_bank.append(
            (
                family,
                f"{note}; expanded mixed200 coverage with broader parameter support",
                params,
            )
        )
    return build_system_specs_from_family_param_bank(
        family_param_bank,
        expected_per_family=50,
        expected_total=200,
    )


MIXED200_SYSTEM_SPECS: tuple[SystemSpec, ...] = build_mixed200_system_specs()
MIXED80_SYSTEM_SPECS: tuple[SystemSpec, ...] = build_mixed80_system_specs()


def build_known_duffing40_system_specs() -> tuple[SystemSpec, ...]:
    specs: list[SystemSpec] = []
    for spec in MIXED80_SYSTEM_SPECS:
        if spec.family not in {"duffing_single", "duffing_bistable"}:
            continue
        specs.append(
            SystemSpec(
                name=spec.name,
                family=spec.family,
                embedding=(float(spec.params[0]), float(spec.params[1])),
                params=spec.params,
                note=f"{spec.note}; fixed embedding equals true Duffing parameters (a, b)",
            )
        )
    if len(specs) != 40:
        raise ValueError(f"Expected 40 known-Duffing systems, got {len(specs)}.")
    return tuple(specs)


KNOWN_DUFFING40_SYSTEM_SPECS: tuple[SystemSpec, ...] = build_known_duffing40_system_specs()
BASE_SYSTEM_SPECS: tuple[SystemSpec, ...] = MIXED80_SYSTEM_SPECS


@dataclass
class ModelBundle:
    meta_dynamics: "MetaDynamics"
    cfg: dict
    train_summary: dict
    embedding_mode: str
    system_embeddings: dict[str, list[float]]
    embedding_metadata: dict[str, object]


@dataclass(frozen=True)
class ActivePolicyConfig:
    horizon: int = 3
    num_iterations: int = 2
    num_samples: int = 12
    num_elite: int = 4
    chunk: int = 2
    action_cost_weight: float = 0.01
    action_strength: float = 0.3


ONLINE_ID_ACTIVE_POLICIES: tuple[str, ...] = (
    "active_long",
    "active_short",
    "active_chunk",
    "async_windowed_update",
)


def resolve_online_id_policy_config(
    policy_name: str,
    active_cfg: ActivePolicyConfig,
) -> dict[str, object]:
    if policy_name == "active_long":
        return {
            "name": policy_name,
            "planning_horizon": 20,
            "planning_chunk": 1,
            "num_iterations": max(10, int(active_cfg.num_iterations)),
            "num_samples": max(40, int(active_cfg.num_samples)),
            "num_elite": max(10, int(active_cfg.num_elite)),
            "update_scheme": "active_long",
            "state_update_interval": 1,
            "predictive_only_window": False,
            "k_theta": 1,
        }
    if policy_name == "active_short":
        return {
            "name": policy_name,
            "planning_horizon": 1,
            "planning_chunk": 1,
            "num_iterations": max(10, int(active_cfg.num_iterations)),
            "num_samples": max(40, int(active_cfg.num_samples)),
            "num_elite": max(10, int(active_cfg.num_elite)),
            "update_scheme": "active_short",
            "state_update_interval": 1,
            "predictive_only_window": False,
            "k_theta": 1,
        }
    if policy_name == "active_chunk":
        return {
            "name": policy_name,
            "planning_horizon": 20,
            "planning_chunk": 5,
            "num_iterations": max(10, int(active_cfg.num_iterations)),
            "num_samples": max(40, int(active_cfg.num_samples)),
            "num_elite": max(10, int(active_cfg.num_elite)),
            "update_scheme": "active_chunk",
            "state_update_interval": 1,
            "predictive_only_window": False,
            "k_theta": 1,
        }
    if policy_name == "async_windowed_update":
        return {
            "name": policy_name,
            "planning_horizon": max(20, int(active_cfg.horizon)),
            # Keep a longer open-loop plan while the embedding filter updates every k_theta steps.
            "planning_chunk": 5,
            "num_iterations": max(10, int(active_cfg.num_iterations)),
            "num_samples": max(40, int(active_cfg.num_samples)),
            "num_elite": max(10, int(active_cfg.num_elite)),
            "update_scheme": "async_windowed_update",
            "state_update_interval": 1,
            "predictive_only_window": False,
            "k_theta": 5,
        }
    return {
        "name": policy_name,
        "planning_horizon": int(active_cfg.horizon),
        "planning_chunk": int(active_cfg.chunk),
        "num_iterations": int(active_cfg.num_iterations),
        "num_samples": int(active_cfg.num_samples),
        "num_elite": int(active_cfg.num_elite),
        "update_scheme": "step_update" if int(active_cfg.chunk) == 1 else "standard_online_id",
        "state_update_interval": 1,
        "predictive_only_window": False,
        "k_theta": 1,
    }


class MixedSystemDataset(Dataset):
    def __init__(
        self,
        systems: tuple[SystemSpec, ...],
        n_per_system: int,
        z_sampler: Callable,
        d_embed: int,
        embedding_map: dict[str, list[float]] | None = None,
        embedding_mode: Literal["fixed", "learned_system_id", "family_param"] = "fixed",
        dynamics_scale: float = 10.0,
    ):
        if d_embed <= 0:
            raise ValueError("d_embed must be >= 1")
        if embedding_mode not in {"fixed", "learned_system_id", "family_param"}:
            raise ValueError(f"Unsupported embedding_mode: {embedding_mode}")
        self.records: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = []
        for spec_idx, spec in enumerate(systems):
            z = z_sampler(n_per_system).float()
            if embedding_mode == "learned_system_id":
                # Pure non-parametric mode: no hand-coded coordinates are used as model input.
                base_e = torch.zeros(d_embed, dtype=torch.float32)
            else:
                if embedding_map is None or spec.name not in embedding_map:
                    raise ValueError(
                        f"embedding_map must contain {spec.name} for {embedding_mode} mode"
                    )
                base_e = torch.tensor(embedding_map[spec.name], dtype=torch.float32)
                if base_e.numel() != d_embed:
                    raise ValueError(
                        f"System {spec.name} resolves to embedding length {base_e.numel()} but d_embed={d_embed}."
                    )
            e = base_e.repeat(n_per_system, 1)
            fx = true_dynamics_from_spec(spec, z, dynamics_scale=dynamics_scale)
            for zi, ei, fi in zip(z, e, fx):
                self.records.append((zi, ei, fi, spec_idx))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


class MetaDynamics:
    def __init__(
        self,
        hypernet: metadyn.LowRankHypernet,
        mean_dynamics: metadyn.HyperMlpDynamics,
        output_scale: float = 10.0,
    ):
        self.hypernet = hypernet
        self.mean_dynamics = mean_dynamics
        self.output_scale = output_scale
        self.e = None
        self.out = None

    def set_params(self, *args):
        self.e = torch.tensor(args, device=device, dtype=torch.float32).unsqueeze(0)
        self.out, _ = self.hypernet(self.e)

    def _expand_hyper_tensor(self, x: torch.Tensor, value: torch.Tensor, trailing_dims: int):
        target_prefix = list(x.shape[:-1])
        value = value.to(device=x.device, dtype=x.dtype)
        prefix_ndim = max(value.ndim - trailing_dims, 0)
        value_prefix = list(value.shape[:prefix_ndim])
        trailing_shape = list(value.shape[prefix_ndim:])
        if len(value_prefix) > len(target_prefix):
            raise ValueError(
                f"Cannot broadcast hyper output with prefix shape {value_prefix} to target prefix {target_prefix}."
            )
        reshape_shape = value_prefix + [1] * (len(target_prefix) - len(value_prefix)) + trailing_shape
        value = value.reshape(*reshape_shape)
        return value.expand(*target_prefix, *trailing_shape)

    def _broadcast_hyper_output(self, x: torch.Tensor, out):
        if isinstance(out, torch.Tensor):
            return self._expand_hyper_tensor(x, out, trailing_dims=1)
        if isinstance(out, (list, tuple)):
            expanded = []
            for param in out:
                if not isinstance(param, torch.Tensor):
                    param = torch.as_tensor(param, dtype=x.dtype, device=x.device)
                expanded.append(self._expand_hyper_tensor(x, param, trailing_dims=2))
            return expanded
        raise TypeError(f"Unsupported hypernet output type: {type(out)!r}")

    def __call__(self, x, e=None):
        if e is None:
            if self.e is None or self.out is None:
                raise ValueError("Embedding not set")
            out = self.out
        else:
            out, _ = self.hypernet(e)
        out = self._broadcast_hyper_output(x, out)
        return self.mean_dynamics.compute_param(x, out) * self.output_scale


class MixedDynamicsEnv(gym.Env):
    def __init__(
        self,
        spec: SystemSpec,
        embedding_vector: torch.Tensor | None = None,
        dt: float = 0.02,
        Q: float = 0.01,
        state_bounds: tuple[float, float] = (-3.0, 3.0),
        action_bounds: tuple[float, float] = (-4.0, 4.0),
        dynamics_scale: float = 10.0,
        device: str = "cpu",
    ):
        try:
            from gymnasium import spaces
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "gymnasium is required for online identification environments. "
                "Install gymnasium to use MixedDynamicsEnv."
            ) from exc

        super().__init__()
        self.spec = spec
        self.dt = dt
        self.Q = Q
        self.dynamics_scale = dynamics_scale
        self.device = torch.device(device)
        if embedding_vector is None:
            embedding_vector = torch.tensor(spec.embedding, dtype=torch.float32)
        self.embedding_vector = (
            embedding_vector.detach().to(self.device, dtype=torch.float32).reshape(-1)
        )
        self.action_space = spaces.Box(
            low=np.full((2,), action_bounds[0], dtype=np.float32),
            high=np.full((2,), action_bounds[1], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.full((2,), state_bounds[0], dtype=np.float32),
            high=np.full((2,), state_bounds[1], dtype=np.float32),
            dtype=np.float32,
        )
        self.state = torch.zeros(2, device=self.device, dtype=torch.float32)
        self.np_random = None

    def get_params(self) -> torch.Tensor:
        return self.embedding_vector

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()
        low = self.observation_space.low
        high = self.observation_space.high
        state = self.np_random.uniform(low, high).astype(np.float32)
        self.state = torch.tensor(state, device=self.device)
        return self.state.clone(), {}

    def step(self, action):
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, device=self.device, dtype=torch.float32)
        action = action.to(self.device).reshape(-1)
        dyn = true_dynamics_from_spec(
            self.spec,
            self.state.unsqueeze(0),
            dynamics_scale=self.dynamics_scale,
        ).squeeze(0)
        self.state = self.state + (dyn + action) * self.dt
        if self.Q > 0:
            self.state = self.state + torch.randn_like(self.state) * math.sqrt(self.Q * self.dt)
        return self.state.clone(), 0.0, False, False, {}


class ExactFe:
    def __init__(self, meta_dynamics: MetaDynamics):
        self.meta_dynamics = meta_dynamics

    def __call__(self, z: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            return jacobian_wrt_param(self.meta_dynamics, [z, e], 1)


class ExactFz:
    def __init__(self, meta_dynamics: MetaDynamics):
        self.meta_dynamics = meta_dynamics

    def __call__(self, z: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            return jacobian_wrt_param(self.meta_dynamics, [z, e], 0)


# -------------------------
# Ground-truth dynamics
# -------------------------
def true_dynamics_from_spec(
    spec: SystemSpec, z: torch.Tensor, dynamics_scale: float = 10.0
) -> torch.Tensor:
    x = z[..., 0]
    y = z[..., 1]
    p0, p1 = spec.params
    if spec.family in {"duffing", "duffing_single", "duffing_bistable"}:
        dx = y
        dy = p0 * y - x * (p1 + 0.1 * x**2)
    elif spec.family == "van_der_pol":
        dx = y
        dy = p0 * (1 - x**2) * y - p1 * x
    elif spec.family == "double_limit_cycle":
        r2 = x**2 + y**2
        inner_r = p1
        barrier_r = p1 + 0.55
        outer_r = p1 + 1.10
        inner2 = inner_r**2
        barrier2 = barrier_r**2
        outer2 = outer_r**2
        radial = 0.02 * (r2 - inner2) * (barrier2 - r2) * (r2 - outer2)
        dx = x * radial - p0 * y
        dy = y * radial + p0 * x
    else:
        raise ValueError(f"Unknown family: {spec.family}")
    return dynamics_scale * torch.stack([dx, dy], dim=-1)


def rollout_true(
    spec: SystemSpec,
    z0: torch.Tensor,
    horizon: int,
    dt: float,
    dynamics_scale: float = 10.0,
) -> torch.Tensor:
    z = z0.clone()
    traj = [z.clone()]
    for _ in range(horizon):
        z = z + dt * true_dynamics_from_spec(spec, z, dynamics_scale=dynamics_scale)
        traj.append(z.clone())
    return torch.stack(traj, dim=1)


def rollout_meta(
    meta_dynamics: MetaDynamics, e: torch.Tensor, z0: torch.Tensor, horizon: int, dt: float
) -> torch.Tensor:
    z = z0.clone()
    traj = [z.clone()]
    for _ in range(horizon):
        z = z + dt * meta_dynamics(z, e=e)
        traj.append(z.clone())
    return torch.stack(traj, dim=1)


def rollout_true_controlled(
    spec: SystemSpec,
    z0: torch.Tensor,
    actions: torch.Tensor,
    dt: float,
    dynamics_scale: float = 10.0,
) -> torch.Tensor:
    z = z0.clone()
    traj = [z.clone()]
    for t in range(actions.shape[1]):
        z = z + dt * (
            true_dynamics_from_spec(spec, z, dynamics_scale=dynamics_scale) + actions[:, t, :]
        )
        traj.append(z.clone())
    return torch.stack(traj, dim=1)


def rollout_meta_controlled(
    meta_dynamics: MetaDynamics,
    e: torch.Tensor,
    z0: torch.Tensor,
    actions: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    z = z0.clone()
    traj = [z.clone()]
    for t in range(actions.shape[1]):
        z = z + dt * (meta_dynamics(z, e=e) + actions[:, t, :])
        traj.append(z.clone())
    return torch.stack(traj, dim=1)


def trajectory_r2(pred_traj: torch.Tensor, true_traj: torch.Tensor) -> torch.Tensor:
    pred_flat = pred_traj.reshape(pred_traj.shape[0], -1)
    true_flat = true_traj.reshape(true_traj.shape[0], -1)
    sse = ((pred_flat - true_flat) ** 2).sum(dim=-1)
    centered = true_flat - true_flat.mean(dim=-1, keepdim=True)
    sst = (centered**2).sum(dim=-1).clamp_min(1e-8)
    return 1.0 - sse / sst


# -------------------------
# Builders / training
# -------------------------
def truncate_embedding(specs: tuple[SystemSpec, ...], d_embed: int) -> tuple[SystemSpec, ...]:
    if d_embed <= 0:
        raise ValueError("d_embed must be >= 1")
    return tuple(
        SystemSpec(
            name=spec.name,
            family=spec.family,
            embedding=tuple(spec.embedding[:d_embed]),
            params=spec.params,
            note=spec.note,
        )
        for spec in specs
    )


def build_family_param_embedding_metadata(
    systems: tuple[SystemSpec, ...],
) -> dict[str, object]:
    family_order = tuple(dict.fromkeys(spec.family for spec in systems))
    param_stats: dict[str, dict[str, list[float]]] = {}
    for family in family_order:
        params = np.asarray(
            [spec.params for spec in systems if spec.family == family],
            dtype=np.float32,
        )
        if params.size == 0:
            continue
        param_min = params.min(axis=0)
        param_max = params.max(axis=0)
        param_center = 0.5 * (param_min + param_max)
        param_scale = np.maximum(0.5 * (param_max - param_min), 1e-3)
        param_stats[family] = {
            "min": param_min.tolist(),
            "max": param_max.tolist(),
            "center": param_center.tolist(),
            "scale": param_scale.tolist(),
        }
    return {
        "family_order": list(family_order),
        "param_stats": param_stats,
        "embedding_dim": int(len(family_order) + 2),
    }


def family_param_embedding_vector(
    spec: SystemSpec,
    embedding_metadata: dict[str, object],
) -> list[float]:
    family_order = [str(x) for x in embedding_metadata.get("family_order", [])]
    if spec.family not in family_order:
        raise ValueError(f"Family {spec.family} is not present in family_order: {family_order}")
    param_stats = dict(embedding_metadata.get("param_stats", {}))
    if spec.family not in param_stats:
        raise ValueError(f"Missing param_stats for family {spec.family}")
    family_idx = family_order.index(spec.family)
    one_hot = np.zeros(len(family_order), dtype=np.float32)
    one_hot[family_idx] = 1.0
    center = np.asarray(param_stats[spec.family]["center"], dtype=np.float32)
    scale = np.asarray(param_stats[spec.family]["scale"], dtype=np.float32)
    normed_params = (np.asarray(spec.params, dtype=np.float32) - center) / scale
    return np.concatenate([one_hot, normed_params], axis=0).astype(np.float32).tolist()


def resolve_effective_d_embed(
    *,
    systems: tuple[SystemSpec, ...],
    embedding_mode: str,
    requested_d_embed: int,
    embedding_metadata: dict[str, object] | None = None,
) -> int:
    if embedding_mode == "family_param":
        if embedding_metadata is None:
            raise ValueError("embedding_metadata is required for family_param mode")
        return int(embedding_metadata["embedding_dim"])
    return int(requested_d_embed)


def build_embedding_metadata(
    systems: tuple[SystemSpec, ...],
    embedding_mode: str,
) -> dict[str, object]:
    if embedding_mode == "family_param":
        return build_family_param_embedding_metadata(systems)
    return {}


def resolve_system_embedding_map(
    systems: tuple[SystemSpec, ...],
    embedding_mode: Literal["fixed", "learned_system_id", "family_param"],
    learned_table: nn.Embedding | None = None,
    embedding_metadata: dict[str, object] | None = None,
) -> dict[str, list[float]]:
    if embedding_mode == "fixed":
        return {spec.name: [float(x) for x in spec.embedding] for spec in systems}
    if embedding_mode == "family_param":
        if embedding_metadata is None:
            raise ValueError("embedding_metadata must be provided for family_param mode")
        return {
            spec.name: family_param_embedding_vector(spec, embedding_metadata)
            for spec in systems
        }
    if learned_table is None:
        raise ValueError("learned_table must be provided for learned_system_id mode")
    weights = learned_table.weight.detach().cpu()
    return {spec.name: [float(x) for x in weights[i].tolist()] for i, spec in enumerate(systems)}


def system_embedding_tensor(
    embedding_map: dict[str, list[float]],
    systems: tuple[SystemSpec, ...],
    target_device: str | torch.device,
) -> torch.Tensor:
    rows = [embedding_map[spec.name] for spec in systems]
    return torch.tensor(rows, dtype=torch.float32, device=target_device)


def build_training_cfg(
    d_embed: int, d_hidden_dynamics: int, d_hidden_hypernet_dynamics: int, n_hidden: int
):
    return {
        "d_latent": 2,
        "d_embed": d_embed,
        "du": 0,
        "d_hidden_embed": 16,
        "d_context": d_embed,
        "d_hidden_dynamics": d_hidden_dynamics,
        "d_hidden_hypernet_dynamics": d_hidden_hypernet_dynamics,
        "n_hidden": n_hidden,
        "likelihood": "gaussian",
        "l2_c": 1e-4,
        "l2_dw_dynamics": 1e-4,
        "rank_dynamics": 2,
        "update_input": True,
        "update_hidden": True,
        "update_output": False,
        "linear_hypernetwork": False,
    }


def build_system_geometry_targets(
    *,
    systems: tuple[SystemSpec, ...],
    n_anchor_samples: int,
    dynamics_scale: float,
    z_sampler: Callable,
    neighbor_k: int,
) -> dict[str, torch.Tensor]:
    if len(systems) < 2:
        empty_long = torch.empty(0, dtype=torch.long)
        empty_float = torch.empty(0, dtype=torch.float32)
        return {
            "anchor_z": z_sampler(max(int(n_anchor_samples), 1)).float(),
            "anchor_fx": torch.empty((len(systems), max(int(n_anchor_samples), 1), 2), dtype=torch.float32),
            "distance_matrix": torch.zeros((len(systems), len(systems)), dtype=torch.float32),
            "edge_i": empty_long,
            "edge_j": empty_long,
            "edge_target": empty_float,
        }

    anchor_z = z_sampler(max(int(n_anchor_samples), 1)).float()
    anchor_fx = torch.stack(
        [
            true_dynamics_from_spec(spec, anchor_z, dynamics_scale=dynamics_scale).detach().cpu()
            for spec in systems
        ],
        dim=0,
    )
    flat_fx = anchor_fx.reshape(len(systems), -1)
    distance_matrix = torch.cdist(flat_fx, flat_fx, p=2)
    if flat_fx.shape[-1] > 0:
        distance_matrix = distance_matrix / math.sqrt(float(flat_fx.shape[-1]))
    positive = distance_matrix[distance_matrix > 0]
    scale = float(positive.mean().item()) if positive.numel() > 0 else 1.0
    distance_matrix = distance_matrix / max(scale, 1e-6)

    edge_set: set[tuple[int, int]] = set()
    max_neighbors = max(int(neighbor_k), 0)
    for i, spec in enumerate(systems):
        same_family = [
            j for j, other in enumerate(systems) if j != i and other.family == spec.family
        ]
        global_neighbors = [j for j in range(len(systems)) if j != i]
        preferred = same_family if same_family else global_neighbors
        preferred = sorted(preferred, key=lambda j: float(distance_matrix[i, j].item()))
        if len(preferred) < max_neighbors:
            extras = [j for j in global_neighbors if j not in preferred]
            extras = sorted(extras, key=lambda j: float(distance_matrix[i, j].item()))
            preferred = preferred + extras
        for j in preferred[:max_neighbors]:
            edge_set.add(tuple(sorted((i, j))))

    if edge_set:
        edge_pairs = sorted(edge_set)
        edge_i = torch.tensor([i for i, _ in edge_pairs], dtype=torch.long)
        edge_j = torch.tensor([j for _, j in edge_pairs], dtype=torch.long)
        edge_target = distance_matrix[edge_i, edge_j].to(dtype=torch.float32)
    else:
        edge_i = torch.empty(0, dtype=torch.long)
        edge_j = torch.empty(0, dtype=torch.long)
        edge_target = torch.empty(0, dtype=torch.float32)

    return {
        "anchor_z": anchor_z,
        "anchor_fx": anchor_fx,
        "distance_matrix": distance_matrix.to(dtype=torch.float32),
        "edge_i": edge_i,
        "edge_j": edge_j,
        "edge_target": edge_target,
    }


def geometry_regularizer_loss(
    learned_table: nn.Embedding,
    geometry_targets: dict[str, torch.Tensor],
    target_device: str | torch.device,
) -> torch.Tensor:
    edge_i = geometry_targets["edge_i"].to(device=target_device, dtype=torch.long)
    edge_j = geometry_targets["edge_j"].to(device=target_device, dtype=torch.long)
    edge_target = geometry_targets["edge_target"].to(device=target_device, dtype=torch.float32)
    if edge_i.numel() == 0 or edge_target.numel() == 0:
        return learned_table.weight.sum() * 0.0
    emb = learned_table.weight
    edge_dist = torch.norm(emb[edge_i] - emb[edge_j], dim=-1)
    edge_dist = edge_dist / edge_dist.mean().clamp_min(1e-6)
    edge_target = edge_target / edge_target.mean().clamp_min(1e-6)
    return F.mse_loss(edge_dist, edge_target)


def interpolation_augmentation_loss(
    *,
    learned_table: nn.Embedding,
    hypernet,
    mean_dynamics,
    geometry_targets: dict[str, torch.Tensor],
    n_aug_samples: int,
    dynamics_scale: float,
    target_device: str | torch.device,
) -> torch.Tensor:
    edge_i = geometry_targets["edge_i"].to(device=target_device, dtype=torch.long)
    edge_j = geometry_targets["edge_j"].to(device=target_device, dtype=torch.long)
    if edge_i.numel() == 0 or int(n_aug_samples) <= 0:
        return learned_table.weight.sum() * 0.0

    sample_idx = torch.randint(edge_i.shape[0], (int(n_aug_samples),), device=target_device)
    sys_i = edge_i[sample_idx]
    sys_j = edge_j[sample_idx]
    anchor_z = geometry_targets["anchor_z"].to(device=target_device, dtype=torch.float32)
    anchor_fx = geometry_targets["anchor_fx"].to(device=target_device, dtype=torch.float32)
    anchor_idx = torch.randint(anchor_z.shape[0], (int(n_aug_samples),), device=target_device)
    lam = torch.rand(int(n_aug_samples), 1, device=target_device, dtype=torch.float32)

    z_aug = anchor_z[anchor_idx]
    target_fx = lam * anchor_fx[sys_i, anchor_idx] + (1.0 - lam) * anchor_fx[sys_j, anchor_idx]
    e_i = learned_table(sys_i)
    e_j = learned_table(sys_j)
    e_aug = lam * e_i + (1.0 - lam) * e_j
    out_aug, _ = hypernet(e_aug)
    pred_fx = mean_dynamics.compute_param(z_aug, out_aug) * dynamics_scale
    return F.mse_loss(pred_fx, target_fx)


def train_meta_dynamics(
    systems: tuple[SystemSpec, ...],
    d_embed: int,
    d_hidden_dynamics: int,
    d_hidden_hypernet_dynamics: int,
    n_hidden: int,
    embedding_mode: Literal["fixed", "learned_system_id", "family_param"] = "fixed",
    dynamics_scale: float = 10.0,
    n_per_system: int = 5000,
    batch_size: int = 512,
    n_epochs: int = 80,
    embedding_reference_systems: tuple[SystemSpec, ...] | None = None,
    geometry_reg_weight: float = 0.05,
    geometry_anchor_samples: int = 512,
    geometry_neighbor_k: int = 4,
    interpolation_aug_weight: float = 0.25,
    interpolation_aug_samples: int = 128,
    train_state_bounds: tuple[float, float] = (-3.0, 3.0),
) -> ModelBundle:
    global device
    reference_systems = embedding_reference_systems or systems
    embedding_metadata = build_embedding_metadata(
        systems=reference_systems,
        embedding_mode=embedding_mode,
    )
    effective_d_embed = resolve_effective_d_embed(
        systems=systems,
        embedding_mode=embedding_mode,
        requested_d_embed=d_embed,
        embedding_metadata=embedding_metadata,
    )
    static_embedding_map = None
    if embedding_mode != "learned_system_id":
        static_embedding_map = resolve_system_embedding_map(
            systems=systems,
            embedding_mode=embedding_mode,
            learned_table=None,
            embedding_metadata=embedding_metadata,
        )
    z_sampler = make_uniform_sampler(float(train_state_bounds[0]), float(train_state_bounds[1]), 2)
    geometry_targets: dict[str, torch.Tensor] | None = None
    if embedding_mode == "learned_system_id" and (
        float(geometry_reg_weight) > 0.0 or float(interpolation_aug_weight) > 0.0
    ):
        geometry_targets = build_system_geometry_targets(
            systems=systems,
            n_anchor_samples=int(geometry_anchor_samples),
            dynamics_scale=dynamics_scale,
            z_sampler=z_sampler,
            neighbor_k=int(geometry_neighbor_k),
        )
    ds = MixedSystemDataset(
        systems,
        n_per_system=n_per_system,
        z_sampler=z_sampler,
        d_embed=effective_d_embed,
        embedding_map=static_embedding_map,
        embedding_mode=embedding_mode,
        dynamics_scale=dynamics_scale,
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    cfg = build_training_cfg(
        d_embed=effective_d_embed,
        d_hidden_dynamics=d_hidden_dynamics,
        d_hidden_hypernet_dynamics=d_hidden_hypernet_dynamics,
        n_hidden=n_hidden,
    )
    cfg["dynamics_scale"] = dynamics_scale
    cfg["train_samples_per_system"] = int(n_per_system)
    cfg["train_epochs"] = int(n_epochs)
    cfg["batch_size"] = int(batch_size)
    cfg["conditioning_type"] = "one_hot_system_id_lookup"
    cfg["geometry_reg_weight"] = float(geometry_reg_weight)
    cfg["geometry_anchor_samples"] = int(geometry_anchor_samples)
    cfg["geometry_neighbor_k"] = int(geometry_neighbor_k)
    cfg["interpolation_aug_weight"] = float(interpolation_aug_weight)
    cfg["interpolation_aug_samples"] = int(interpolation_aug_samples)
    cfg["train_state_bounds"] = [float(train_state_bounds[0]), float(train_state_bounds[1])]
    hypernet = build_hypernetwork(cfg, device)
    mean_kwargs = dict(
        d_latent=cfg["d_latent"],
        d_hidden=cfg["d_hidden_dynamics"],
        n_hidden=cfg["n_hidden"],
        update_input=cfg["update_input"],
        update_output=cfg["update_output"],
        update_hidden=cfg["update_hidden"],
        du=0,
        device=device,
    )
    if not HAS_INTEGRATIVE_INFERENCE:
        mean_kwargs["d_context"] = cfg["d_context"]
    mean_dynamics = metadyn.HyperMlpDynamics(**mean_kwargs).to(device)
    learned_table: nn.Embedding | None = None
    if embedding_mode == "learned_system_id":
        learned_table = nn.Embedding(len(systems), effective_d_embed, device=device)
        nn.init.normal_(learned_table.weight, mean=0.0, std=0.5)
    params = list(hypernet.parameters()) + list(mean_dynamics.parameters())
    if learned_table is not None:
        params += list(learned_table.parameters())
    opt = torch.optim.AdamW(
        params,
        lr=1e-3,
        weight_decay=1e-4,
    )

    epoch_losses: list[float] = []
    epoch_primary_losses: list[float] = []
    epoch_geometry_losses: list[float] = []
    epoch_interp_losses: list[float] = []
    per_system_last = {spec.name: None for spec in systems}
    for _ in tqdm(range(n_epochs), desc="meta-train"):
        total_loss = 0.0
        total_primary_loss = 0.0
        total_geometry_loss = 0.0
        total_interp_loss = 0.0
        total_n = 0
        per_system_acc = {spec.name: [] for spec in systems}
        for z, e, fx, spec_idx in dl:
            z = z.to(device)
            fx = fx.to(device)
            if embedding_mode == "learned_system_id":
                if learned_table is None:
                    raise RuntimeError("learned_table missing for learned_system_id mode")
                e = learned_table(spec_idx.to(device=device, dtype=torch.long))
            else:
                e = e.to(device)
            out, _ = hypernet(e)
            pred = mean_dynamics.compute_param(z, out) * dynamics_scale
            primary_loss = F.mse_loss(pred, fx)
            geometry_loss = primary_loss * 0.0
            interp_loss = primary_loss * 0.0
            if embedding_mode == "learned_system_id" and learned_table is not None and geometry_targets is not None:
                if float(geometry_reg_weight) > 0.0:
                    geometry_loss = geometry_regularizer_loss(
                        learned_table=learned_table,
                        geometry_targets=geometry_targets,
                        target_device=device,
                    )
                if float(interpolation_aug_weight) > 0.0 and int(interpolation_aug_samples) > 0:
                    interp_loss = interpolation_augmentation_loss(
                        learned_table=learned_table,
                        hypernet=hypernet,
                        mean_dynamics=mean_dynamics,
                        geometry_targets=geometry_targets,
                        n_aug_samples=int(interpolation_aug_samples),
                        dynamics_scale=dynamics_scale,
                        target_device=device,
                    )
            loss = (
                primary_loss
                + float(geometry_reg_weight) * geometry_loss
                + float(interpolation_aug_weight) * interp_loss
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                params,
                5.0,
            )
            opt.step()
            total_loss += loss.item() * z.shape[0]
            total_primary_loss += primary_loss.item() * z.shape[0]
            total_geometry_loss += geometry_loss.item() * z.shape[0]
            total_interp_loss += interp_loss.item() * z.shape[0]
            total_n += z.shape[0]
            with torch.no_grad():
                batch_err = ((pred - fx) ** 2).mean(dim=-1).detach().cpu().numpy()
                spec_idx_np = spec_idx.cpu().numpy()
                for local_i, system_i in enumerate(spec_idx_np):
                    per_system_acc[systems[int(system_i)].name].append(float(batch_err[local_i]))
        epoch_losses.append(total_loss / max(total_n, 1))
        epoch_primary_losses.append(total_primary_loss / max(total_n, 1))
        epoch_geometry_losses.append(total_geometry_loss / max(total_n, 1))
        epoch_interp_losses.append(total_interp_loss / max(total_n, 1))
        per_system_last = {
            name: float(np.mean(vals)) if len(vals) > 0 else None
            for name, vals in per_system_acc.items()
        }

    return ModelBundle(
        meta_dynamics=MetaDynamics(hypernet, mean_dynamics, output_scale=dynamics_scale),
        cfg=cfg,
        train_summary={
            "epoch_losses": epoch_losses,
            "epoch_primary_losses": epoch_primary_losses,
            "epoch_geometry_losses": epoch_geometry_losses,
            "epoch_interpolation_losses": epoch_interp_losses,
            "final_train_loss": epoch_losses[-1] if epoch_losses else None,
            "final_primary_loss": epoch_primary_losses[-1] if epoch_primary_losses else None,
            "final_geometry_loss": epoch_geometry_losses[-1] if epoch_geometry_losses else None,
            "final_interpolation_loss": epoch_interp_losses[-1] if epoch_interp_losses else None,
            "final_per_system_train_mse": per_system_last,
            "effective_d_embed": effective_d_embed,
        },
        embedding_mode=embedding_mode,
        system_embeddings=resolve_system_embedding_map(
            systems=systems,
            embedding_mode=embedding_mode,
            learned_table=learned_table,
            embedding_metadata=embedding_metadata,
        ),
        embedding_metadata=embedding_metadata,
    )


# -------------------------
# Offline evaluation
# -------------------------
def evaluate_vectorfield(
    meta_dynamics: MetaDynamics,
    systems: tuple[SystemSpec, ...],
    system_embeddings: torch.Tensor,
    grid_n: int = 41,
    dynamics_scale: float = 10.0,
):
    x = torch.linspace(-3.0, 3.0, grid_n, device=device)
    y = torch.linspace(-3.0, 3.0, grid_n, device=device)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    z = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    results = {}
    for spec_idx, spec in enumerate(systems):
        e = system_embeddings[spec_idx].reshape(1, -1).repeat(z.shape[0], 1)
        true_fx = true_dynamics_from_spec(spec, z, dynamics_scale=dynamics_scale)
        pred_fx = meta_dynamics(z, e=e)
        mse = F.mse_loss(pred_fx, true_fx).item()
        mae = (pred_fx - true_fx).abs().mean().item()
        results[spec.name] = {
            "vectorfield_mse": float(mse),
            "vectorfield_mae": float(mae),
        }
    return results


def evaluate_rollout(
    meta_dynamics: MetaDynamics,
    systems: tuple[SystemSpec, ...],
    system_embeddings: torch.Tensor,
    n_init: int = 32,
    horizon: int = 200,
    dt: float = 0.01,
    init_bounds: tuple[float, float] = (-1.2, 1.2),
    dynamics_scale: float = 10.0,
):
    z_sampler = make_uniform_sampler(init_bounds[0], init_bounds[1], 2)
    z0 = z_sampler(n_init).to(device)
    results = {}
    for spec_idx, spec in enumerate(systems):
        e = system_embeddings[spec_idx].reshape(1, -1).repeat(n_init, 1)
        true_traj = rollout_true(spec, z0, horizon=horizon, dt=dt, dynamics_scale=dynamics_scale)
        pred_traj = rollout_meta(meta_dynamics, e=e, z0=z0, horizon=horizon, dt=dt)
        mse = F.mse_loss(pred_traj, true_traj).item()
        final_mse = F.mse_loss(pred_traj[:, -1], true_traj[:, -1]).item()
        max_abs = (pred_traj - true_traj).abs().max().item()
        results[spec.name] = {
            "rollout_mse": float(mse),
            "final_state_mse": float(final_mse),
            "rollout_max_abs_err": float(max_abs),
        }
    return results


def pairwise_embedding_distances(systems: tuple[SystemSpec, ...], system_embeddings: torch.Tensor):
    names = [spec.name for spec in systems]
    D = torch.cdist(system_embeddings, system_embeddings).cpu().numpy().tolist()
    return {"names": names, "distance_matrix": D}


def summarize_eval_by_family(
    systems: tuple[SystemSpec, ...],
    per_system_eval: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    by_family: dict[str, list[dict[str, float]]] = {}
    for spec in systems:
        if spec.name in per_system_eval:
            by_family.setdefault(spec.family, []).append(per_system_eval[spec.name])
    out: dict[str, dict[str, float]] = {}
    for family, rows in by_family.items():
        keys = sorted({k for row in rows for k in row.keys()})
        payload: dict[str, float] = {"n_systems": float(len(rows))}
        for key in keys:
            vals = [float(row[key]) for row in rows if key in row]
            if not vals:
                continue
            payload[f"mean_{key}"] = float(np.mean(vals))
            payload[f"std_{key}"] = float(np.std(vals))
            payload[f"min_{key}"] = float(np.min(vals))
            payload[f"max_{key}"] = float(np.max(vals))
        out[family] = payload
    return out


def project_embeddings_to_2d(emb: np.ndarray) -> tuple[np.ndarray, str]:
    if emb.ndim != 2:
        raise ValueError(f"Expected 2D embedding matrix, got shape {emb.shape}")
    if emb.shape[1] == 1:
        return np.concatenate([emb, np.zeros_like(emb)], axis=1), "pad_y_zero"
    if emb.shape[1] == 2:
        return emb.copy(), "native_2d"
    centered = emb - emb.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ vt[:2].T
    return proj, "pca_2d"


def save_embedding_cluster_figure(
    systems: tuple[SystemSpec, ...],
    system_embeddings: torch.Tensor,
    out_path: str,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    emb_np = system_embeddings.detach().cpu().numpy()
    emb_2d, projection = project_embeddings_to_2d(emb_np)

    families = sorted({spec.family for spec in systems})
    cmap = plt.get_cmap("tab10")
    family_to_color = {family: cmap(i % 10) for i, family in enumerate(families)}

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    for family in families:
        idxs = [i for i, spec in enumerate(systems) if spec.family == family]
        pts = emb_2d[idxs, :]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=42,
            alpha=0.85,
            color=family_to_color[family],
            label=family,
            edgecolors="none",
        )
        centroid = pts.mean(axis=0)
        ax.scatter(
            [centroid[0]],
            [centroid[1]],
            s=120,
            marker="X",
            color=family_to_color[family],
            edgecolors="black",
            linewidths=0.8,
        )
        ax.text(
            float(centroid[0]) + 0.04,
            float(centroid[1]) + 0.04,
            family,
            fontsize=9,
            color=family_to_color[family],
            weight="bold",
        )

    axis_names = (
        ("Embedding dim 1", "Embedding dim 2") if projection == "native_2d" else ("PC1", "PC2")
    )
    ax.set_title("Meta-Dynamics Conditioning Embeddings")
    ax.set_xlabel(axis_names[0])
    ax.set_ylabel(axis_names[1])
    ax.grid(True, alpha=0.25)
    ax.legend(title="Family", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return {"path": out_path, "projection": projection}


def save_family_vectorfield_comparison_figure(
    meta_dynamics: MetaDynamics,
    systems: tuple[SystemSpec, ...],
    system_embeddings: torch.Tensor,
    out_path: str,
    dynamics_scale: float = 10.0,
    grid_n: int = CANONICAL_VECTORFIELD_GRID_N,
    grid_limits: tuple[float, float] = CANONICAL_VECTORFIELD_GRID_RANGE,
    figure_layout: str = CANONICAL_VECTORFIELD_LAYOUT,
) -> dict[str, object]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(os.path.dirname(out_path))
    families = sorted({spec.family for spec in systems})
    representative_specs: list[SystemSpec] = []
    representative_indices: list[int] = []
    for family in families:
        family_indices = [i for i, spec in enumerate(systems) if spec.family == family]
        if len(family_indices) != 1:
            raise ValueError(
                "save_family_vectorfield_comparison_figure expects exactly one explicit representative per family; "
                f"got {len(family_indices)} for family {family}."
            )
        representative_indices.append(family_indices[0])
        representative_specs.append(systems[family_indices[0]])

    grid_min, grid_max = grid_limits
    x_np = np.linspace(grid_min, grid_max, grid_n)
    y_np = np.linspace(grid_min, grid_max, grid_n)
    X, Y = np.meshgrid(x_np, y_np)
    z = torch.tensor(
        np.stack([X.reshape(-1), Y.reshape(-1)], axis=-1), dtype=torch.float32, device=device
    )

    fig, axes = plt.subplots(
        len(families), 2, figsize=(8.8, 2.45 * len(families)), sharex=True, sharey=True
    )
    if len(families) == 1:
        axes = np.asarray([axes])

    metadata = []
    for row_idx, (family, spec, spec_idx) in enumerate(
        zip(families, representative_specs, representative_indices)
    ):
        e = system_embeddings[spec_idx].reshape(1, -1).repeat(z.shape[0], 1)
        true_fx = (
            true_dynamics_from_spec(spec, z, dynamics_scale=dynamics_scale).detach().cpu().numpy()
        )
        pred_fx = meta_dynamics(z, e=e).detach().cpu().numpy()
        comps = [
            (axes[row_idx, 0], true_fx, "True"),
            (axes[row_idx, 1], pred_fx, "Reconstructed"),
        ]
        for ax, field, label in comps:
            U = field[:, 0].reshape(grid_n, grid_n)
            V = field[:, 1].reshape(grid_n, grid_n)
            speed = np.sqrt(U**2 + V**2)
            ax.streamplot(
                x_np,
                y_np,
                U,
                V,
                color=np.log1p(speed),
                cmap=str(CANONICAL_VECTORFIELD_STYLE["cmap"]),
                density=1.0,
                linewidth=1.0,
                arrowsize=0.8,
            )
            ax.set_aspect("equal")
            ax.grid(alpha=0.15)
            ax.set_xlim(grid_min, grid_max)
            ax.set_ylim(grid_min, grid_max)
            ax.set_title(f"{family} — {label}", fontsize=10)
        axes[row_idx, 0].set_ylabel("x2")
        metadata.append({"family": family, "system": spec.name, "params": list(spec.params)})

    for ax in axes[-1, :]:
        ax.set_xlabel("x1")
    fig.suptitle("Representative family vector fields: true vs reconstructed", fontsize=12, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    fig.savefig(out_path, dpi=int(CANONICAL_VECTORFIELD_STYLE["dpi"]), bbox_inches="tight")
    plt.close(fig)
    return {
        "path": out_path,
        "representatives": metadata,
        "grid_n": grid_n,
        "grid_limits": [float(grid_min), float(grid_max)],
        "layout": figure_layout,
        "style": dict(CANONICAL_VECTORFIELD_STYLE),
    }


def _rollout_dt_key(dt: float) -> str:
    return f"{dt:.6g}"


def verify_parameter_bank(
    systems: tuple[SystemSpec, ...],
    out_dir: str,
    dynamics_scale: float = 10.0,
    horizon: int = 300,
    dt: float = 0.01,
) -> dict:
    import csv

    ensure_dir(out_dir)
    init_grid = torch.tensor(
        [
            [-2.0, -2.0],
            [-2.0, 0.0],
            [-2.0, 2.0],
            [0.0, -2.0],
            [0.0, -0.75],
            [0.0, 0.75],
            [0.0, 2.0],
            [2.0, -2.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [-1.2, 1.2],
            [1.2, -1.2],
            [1.5, 0.5],
            [-1.5, -0.5],
        ],
        dtype=torch.float32,
    )
    rows: list[dict] = []
    for spec in systems:
        traj = rollout_true(
            spec, init_grid, horizon=horizon, dt=dt, dynamics_scale=dynamics_scale
        ).cpu()
        finite_ok = bool(torch.isfinite(traj).all().item())
        radii = torch.linalg.norm(traj, dim=-1)
        final_xy = traj[:, -1, :]
        tail = traj[:, -80:, :]
        tail_radii = torch.linalg.norm(tail, dim=-1)
        max_radius = float(radii.max().item())
        mean_final_radius = float(torch.linalg.norm(final_xy, dim=-1).mean().item())
        std_final_radius = float(torch.linalg.norm(final_xy, dim=-1).std(unbiased=False).item())
        tail_radius_std = float(tail_radii.std(unbiased=False).item())
        speed = torch.linalg.norm(traj[:, 1:, :] - traj[:, :-1, :], dim=-1) / dt
        mean_speed = float(speed.mean().item())
        p95_speed = float(torch.quantile(speed.reshape(-1), 0.95).item())
        max_speed = float(speed.max().item())
        angular_velocity = (
            traj[:, :-1, 0] * (traj[:, 1:, 1] - traj[:, :-1, 1])
            - traj[:, :-1, 1] * (traj[:, 1:, 0] - traj[:, :-1, 0])
        ) / dt
        median_angular_velocity = float(torch.median(angular_velocity).item())
        rotation_direction = "ccw" if median_angular_velocity >= 0.0 else "cw"
        sign_diversity = int(torch.unique(torch.sign(final_xy[:, 0])).numel())

        family_ok = False
        family_reason = ""
        speed_cap = 180.0
        p95_cap = 80.0
        if spec.family == "duffing_single":
            family_ok = mean_final_radius < 0.8 and std_final_radius < 0.55 and max_speed < 80.0
            family_reason = "single-attractor convergence toward origin with modest transient speed"
        elif spec.family == "duffing_bistable":
            family_ok = (
                mean_final_radius > 0.7
                and sign_diversity >= 2
                and std_final_radius < 0.8
                and max_speed < 40.0
            )
            family_reason = (
                "bistable settling into separated wells without steep well-crossing transients"
            )
        elif spec.family == "van_der_pol":
            speed_cap = 130.0
            p95_cap = 82.0
            family_ok = (
                0.8 < mean_final_radius < 3.2
                and tail_radius_std < 0.80
                and max_speed < speed_cap
                and p95_speed < p95_cap
            )
            family_reason = "stable oscillatory limit cycle with controlled relaxation speed"
        elif spec.family == "double_limit_cycle":
            speed_cap = 120.0
            p95_cap = 55.0
            family_ok = (
                0.5 < mean_final_radius < 4.0
                and 0.12 < std_final_radius < 1.6
                and max_speed < speed_cap
                and p95_speed < p95_cap
            )
            family_reason = "bounded multi-ring radial dynamics with controlled angular speed and bidirectional rotation support"
        generic_ok = (
            finite_ok
            and max_radius < 8.0
            and max_speed < speed_cap
            and p95_speed < p95_cap
            and mean_speed > 0.05
        )
        passed = bool(generic_ok and family_ok)
        rows.append(
            {
                "name": spec.name,
                "family": spec.family,
                "param_0": float(spec.params[0]),
                "param_1": float(spec.params[1]),
                "finite_ok": finite_ok,
                "max_radius": max_radius,
                "mean_final_radius": mean_final_radius,
                "std_final_radius": std_final_radius,
                "tail_radius_std": tail_radius_std,
                "mean_speed": mean_speed,
                "p95_speed": p95_speed,
                "max_speed": max_speed,
                "median_angular_velocity": median_angular_velocity,
                "rotation_direction": rotation_direction,
                "speed_cap": speed_cap,
                "p95_cap": p95_cap,
                "sign_diversity": sign_diversity,
                "generic_ok": generic_ok,
                "family_ok": family_ok,
                "passed": passed,
                "check": family_reason,
            }
        )

    csv_path = os.path.join(out_dir, "parameter_bank_verification.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    family_summary = {}
    for family in sorted({r["family"] for r in rows}):
        family_rows = [r for r in rows if r["family"] == family]
        family_summary[family] = {
            "n_systems": len(family_rows),
            "n_passed": int(sum(1 for r in family_rows if r["passed"])),
            "max_radius_max": float(max(r["max_radius"] for r in family_rows)),
            "max_final_radius_max": float(max(r["mean_final_radius"] for r in family_rows)),
            "mean_speed_mean": float(np.mean([r["mean_speed"] for r in family_rows])),
            "p95_speed_max": float(max(r["p95_speed"] for r in family_rows)),
            "max_speed_max": float(max(r["max_speed"] for r in family_rows)),
            "mean_final_radius_mean": float(np.mean([r["mean_final_radius"] for r in family_rows])),
            "std_final_radius_mean": float(np.mean([r["std_final_radius"] for r in family_rows])),
            "rotation_directions": sorted({str(r["rotation_direction"]) for r in family_rows}),
        }
    payload = {
        "verification_horizon": horizon,
        "verification_dt": dt,
        "dynamics_scale": dynamics_scale,
        "n_systems": len(rows),
        "n_passed": int(sum(1 for r in rows if r["passed"])),
        "all_passed": bool(all(r["passed"] for r in rows)),
        "family_summary": family_summary,
        "csv_path": csv_path,
        "rows": rows,
    }
    json_path = os.path.join(out_dir, "parameter_bank_verification.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def save_model_bundle_checkpoint(bundle: ModelBundle, out_dir: str) -> str:
    ckpt_path = os.path.join(out_dir, "meta_dynamics_checkpoint.pt")
    torch.save(
        {
            "cfg": bundle.cfg,
            "train_summary": bundle.train_summary,
            "embedding_mode": bundle.embedding_mode,
            "system_embeddings": bundle.system_embeddings,
            "embedding_metadata": bundle.embedding_metadata,
            "hypernet_state_dict": bundle.meta_dynamics.hypernet.state_dict(),
            "mean_dynamics_state_dict": bundle.meta_dynamics.mean_dynamics.state_dict(),
            "output_scale": bundle.meta_dynamics.output_scale,
        },
        ckpt_path,
    )
    return ckpt_path


def resolve_checkpoint_embedding_map(
    *,
    systems: tuple[SystemSpec, ...],
    embedding_mode: str,
    inferred_d_embed: int,
    saved_embeddings: dict[str, list[float]],
    embedding_metadata: dict[str, object] | None,
) -> dict[str, list[float]]:
    if embedding_mode == "fixed":
        aligned_systems = truncate_embedding(systems, d_embed=inferred_d_embed)
        return resolve_system_embedding_map(
            systems=aligned_systems,
            embedding_mode="fixed",
            learned_table=None,
        )
    if embedding_mode == "family_param" and embedding_metadata:
        resolved = resolve_system_embedding_map(
            systems=systems,
            embedding_mode="family_param",
            learned_table=None,
            embedding_metadata=embedding_metadata,
        )
        for spec in systems:
            if spec.name in saved_embeddings:
                resolved[spec.name] = [float(x) for x in saved_embeddings[spec.name]]
        return resolved
    resolved: dict[str, list[float]] = {}
    for spec in systems:
        if spec.name in saved_embeddings:
            resolved[spec.name] = [float(x) for x in saved_embeddings[spec.name]]
        else:
            resolved[spec.name] = [float(x) for x in spec.embedding[:inferred_d_embed]]
    return resolved


def load_model_bundle_checkpoint(ckpt_path: str, systems: tuple[SystemSpec, ...]) -> ModelBundle:
    payload = torch.load(ckpt_path, map_location=device)
    cfg = dict(payload.get("cfg", {}))
    embedding_mode = str(payload["embedding_mode"])
    saved_embeddings = dict(payload.get("system_embeddings", {}))
    embedding_metadata = dict(payload.get("embedding_metadata", {}))
    if saved_embeddings:
        inferred_d_embed = len(next(iter(saved_embeddings.values())))
    elif embedding_mode == "family_param" and embedding_metadata.get("embedding_dim") is not None:
        inferred_d_embed = int(embedding_metadata["embedding_dim"])
    else:
        inferred_d_embed = int(cfg.get("d_embed", len(systems[0].embedding)))
    cfg.setdefault("d_embed", inferred_d_embed)
    cfg.setdefault("d_hidden_dynamics", 64)
    cfg.setdefault("d_hidden_hypernet_dynamics", 16)
    cfg.setdefault("n_hidden", 2)
    cfg.setdefault("d_context", max(int(cfg["d_hidden_hypernet_dynamics"]), 16))
    cfg.setdefault("d_latent", 2)
    cfg.setdefault("update_input", True)
    cfg.setdefault("update_output", True)
    cfg.setdefault("update_hidden", True)
    hypernet_state = payload["hypernet_state_dict"]
    if any(str(k).startswith("net.") for k in hypernet_state.keys()):

        class _CheckpointFallbackLowRankHypernet(nn.Module):
            def __init__(self, d_embed: int, d_context: int, d_hidden: int):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(d_embed, d_hidden),
                    nn.SiLU(),
                    nn.Linear(d_hidden, d_hidden),
                    nn.SiLU(),
                    nn.Linear(d_hidden, d_context),
                )

            def forward(self, e: torch.Tensor):
                ctx = self.net(e)
                return ctx, None

        hypernet = _CheckpointFallbackLowRankHypernet(
            d_embed=int(cfg["d_embed"]),
            d_context=int(cfg["d_context"]),
            d_hidden=max(int(cfg.get("d_hidden_hypernet_dynamics", 16)), int(cfg["d_context"])),
        ).to(device)
    else:
        hypernet = build_hypernetwork(cfg, device)
    mean_state = payload["mean_dynamics_state_dict"]
    mean_kwargs = dict(
        d_latent=cfg["d_latent"],
        d_hidden=cfg["d_hidden_dynamics"],
        n_hidden=cfg["n_hidden"],
        update_input=cfg["update_input"],
        update_output=cfg["update_output"],
        update_hidden=cfg["update_hidden"],
        du=0,
        device=device,
    )
    if any(str(k).startswith("net.") for k in mean_state.keys()):

        class _CheckpointFallbackHyperMlpDynamics(nn.Module):
            def __init__(self, d_latent: int, d_hidden: int, n_hidden: int, d_context: int):
                super().__init__()
                layers = []
                in_dim = d_latent + d_context
                for _ in range(max(n_hidden, 1)):
                    layers += [nn.Linear(in_dim, d_hidden), nn.SiLU()]
                    in_dim = d_hidden
                layers += [nn.Linear(in_dim, d_latent)]
                self.net = nn.Sequential(*layers)

            def compute_param(self, z: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
                return self.net(torch.cat([z, out], dim=-1))

            def forward(self, z: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
                return self.compute_param(z, out)

        mean_dynamics = _CheckpointFallbackHyperMlpDynamics(
            d_latent=cfg["d_latent"],
            d_hidden=cfg["d_hidden_dynamics"],
            n_hidden=cfg["n_hidden"],
            d_context=cfg["d_context"],
        ).to(device)
    else:
        if not HAS_INTEGRATIVE_INFERENCE:
            mean_kwargs["d_context"] = cfg["d_context"]
        mean_dynamics = metadyn.HyperMlpDynamics(**mean_kwargs).to(device)
    hypernet.load_state_dict(hypernet_state)
    mean_dynamics.load_state_dict(mean_state)
    hypernet.eval()
    mean_dynamics.eval()
    meta_dynamics = MetaDynamics(
        hypernet=hypernet,
        mean_dynamics=mean_dynamics,
        output_scale=float(payload.get("output_scale", cfg.get("dynamics_scale", 10.0))),
    )
    system_embeddings = resolve_checkpoint_embedding_map(
        systems=systems,
        embedding_mode=embedding_mode,
        inferred_d_embed=inferred_d_embed,
        saved_embeddings=saved_embeddings,
        embedding_metadata=embedding_metadata,
    )
    return ModelBundle(
        meta_dynamics=meta_dynamics,
        cfg=cfg,
        train_summary=dict(payload.get("train_summary", {})),
        embedding_mode=embedding_mode,
        system_embeddings=system_embeddings,
        embedding_metadata=embedding_metadata,
    )


def write_pretrain_summary_markdown(
    *,
    out_path: str,
    payload: dict,
    verification: dict,
    checkpoint_path: str,
) -> str:
    family_lines = []
    for family, stats in payload["family_rollout_eval"].items():
        family_lines.append(
            f"- {family}: mean rollout MSE {stats['mean_rollout_mse']:.4f}, mean final-state MSE {stats['mean_final_state_mse']:.4f}"
        )
    verification_lines = []
    for family, stats in verification["family_summary"].items():
        rotation_text = ""
        if stats.get("rotation_directions"):
            rotation_text = f", rotations {', '.join(stats['rotation_directions'])}"
        verification_lines.append(
            f"- {family}: {stats['n_passed']}/{stats['n_systems']} passed, max radius {stats['max_radius_max']:.3f}, max final radius {stats.get('max_final_radius_max', float('nan')):.3f}, p95 speed max {stats['p95_speed_max']:.3f}, max speed {stats['max_speed_max']:.3f}{rotation_text}"
        )
    text = f"""# Mixed-family meta-dynamics pretraining summary

- System bank: {payload['system_bank']}
- Embedding mode: {payload['embedding_mode']}
- Systems: {len(payload['systems'])}
- Final train loss: {payload['train_summary']['final_train_loss']:.6f}
- Training samples/system: {payload['train_cfg']['train_samples_per_system']}
- Training epochs: {payload['train_cfg']['train_epochs']}
- Batch size: {payload['train_cfg']['batch_size']}
- Verification passed: {verification['n_passed']}/{verification['n_systems']}
- Checkpoint: `{checkpoint_path}`
- Embedding figure: `{payload['embedding_cluster_figure']['path']}`

## Family rollout metrics
{os.linesep.join(family_lines)}

## Parameter-bank verification
{os.linesep.join(verification_lines)}
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return out_path


def summarize_offline(
    bundle: ModelBundle,
    systems: tuple[SystemSpec, ...],
    out_dir: str,
    rollout_horizon: int,
    rollout_inits: int,
    rollout_dt: float,
    rollout_dt_sweep: tuple[float, ...] = (),
    rollout_init_bounds: tuple[float, float] = (-1.2, 1.2),
    dynamics_scale: float = 10.0,
    system_bank: str = "custom",
):
    embeddings = system_embedding_tensor(bundle.system_embeddings, systems, target_device=device)
    vf = evaluate_vectorfield(
        bundle.meta_dynamics,
        systems,
        system_embeddings=embeddings,
        dynamics_scale=dynamics_scale,
    )
    dt_candidates = [float(rollout_dt), *[float(x) for x in rollout_dt_sweep]]
    rollout_dts = tuple(dict.fromkeys(dt_candidates))
    rollout_eval_by_dt = {}
    for dt in rollout_dts:
        rollout_eval_by_dt[_rollout_dt_key(dt)] = evaluate_rollout(
            bundle.meta_dynamics,
            systems,
            system_embeddings=embeddings,
            n_init=rollout_inits,
            horizon=rollout_horizon,
            dt=dt,
            init_bounds=rollout_init_bounds,
            dynamics_scale=dynamics_scale,
        )
    primary_key = _rollout_dt_key(rollout_dt)
    ro = rollout_eval_by_dt[primary_key]
    family_rollout_eval_by_dt = {
        dt_key: summarize_eval_by_family(systems, per_system_eval)
        for dt_key, per_system_eval in rollout_eval_by_dt.items()
    }
    family_vf = summarize_eval_by_family(systems, vf)
    cluster_plot_path = os.path.join(out_dir, "embedding_family_clusters.png")
    cluster_plot = save_embedding_cluster_figure(
        systems=systems,
        system_embeddings=embeddings,
        out_path=cluster_plot_path,
    )
    payload = {
        "mode": "pretrain_eval",
        "system_bank": system_bank,
        "embedding_mode": bundle.embedding_mode,
        "embedding_metadata": bundle.embedding_metadata,
        "train_summary": bundle.train_summary,
        "model_cfg": bundle.cfg,
        "train_cfg": {
            "train_samples_per_system": int(bundle.cfg.get("train_samples_per_system", -1)),
            "train_epochs": int(bundle.cfg.get("train_epochs", -1)),
            "batch_size": int(bundle.cfg.get("batch_size", -1)),
            "integrative_inference_backend": bool(HAS_INTEGRATIVE_INFERENCE),
        },
        "rollout_cfg": {
            "horizon": rollout_horizon,
            "n_init": rollout_inits,
            "rollout_dt": rollout_dt,
            "rollout_dt_sweep": list(rollout_dts),
            "rollout_init_bounds": list(rollout_init_bounds),
            "dynamics_scale": dynamics_scale,
        },
        "systems": [
            {
                "name": s.name,
                "system_id": i,
                "family": s.family,
                "embedding": list(bundle.system_embeddings[s.name]),
                "fixed_embedding": list(s.embedding),
                "params": list(s.params),
                "note": s.note,
            }
            for i, s in enumerate(systems)
        ],
        "vectorfield_eval": vf,
        "family_vectorfield_eval": family_vf,
        "rollout_eval": ro,
        "family_rollout_eval": family_rollout_eval_by_dt[primary_key],
        "rollout_eval_by_dt": rollout_eval_by_dt,
        "family_rollout_eval_by_dt": family_rollout_eval_by_dt,
        "embedding_geometry": pairwise_embedding_distances(systems, embeddings),
        "embedding_cluster_figure": cluster_plot,
    }
    out_path = os.path.join(out_dir, "pretrain_eval_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


# -------------------------
# Online identification (existing prototype)
# -------------------------
def build_observation_model(
    dy: int,
    dt: float,
    noise_scale: float,
    mean_firing: float = 1000.0,
):
    obs_model = actdyn.environment.observation.LogLinearObservation(
        d_obs=dy,
        d_latent=2,
        R=noise_scale,
        noise_type="poisson",
        dt=dt,
        device=device,
    )
    C = obs_model.network[0].weight.detach()
    # C[:, 0] = torch.abs(C[:, 0])
    # C[:, 1] = C[:, 1] * 2.0
    mean_firing = max(float(mean_firing), 1e-6)
    bias = torch.log(mean_firing * torch.ones(dy, device=device)) - 0.5 * torch.diag(C @ C.T)
    obs_model.network[0].weight = nn.Parameter(C)
    obs_model.network[0].bias = nn.Parameter(bias)
    return obs_model


def _scalar_stats(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_p90": 0.0,
            f"{prefix}_max": 0.0,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_std": float(arr.std()),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_p90": float(np.percentile(arr, 90.0)),
        f"{prefix}_max": float(arr.max()),
    }


def summarize_embedding_diagnostics(
    step_history: list[dict],
    block_history: list[dict],
    e_init: torch.Tensor,
    e_final: torch.Tensor,
) -> dict[str, float]:
    keys = [
        "embedding_step_norm",
        "innovation_norm",
        "latent_correction_norm",
        "score_norm",
        "sensitivity_fro",
        "info_trace",
        "info_fro",
        "info_cond",
        "fe_fro",
        "fz_fro",
    ]
    summary: dict[str, float] = {
        "n_steps": float(len(step_history)),
        "n_block_updates": float(len(block_history)),
        "embedding_drift_norm": float(torch.norm(e_final - e_init).item()),
    }
    for key in keys:
        summary.update(_scalar_stats([float(r.get(key, 0.0)) for r in step_history], key))
    summary["embedding_path_norm"] = float(
        sum(float(r.get("embedding_step_norm", 0.0)) for r in step_history)
    )
    summary["block_update_rate"] = float(summary["n_block_updates"] / max(summary["n_steps"], 1.0))
    summary.update(
        _scalar_stats(
            [float(r.get("embed_block_delta_norm", 0.0)) for r in block_history],
            "block_embed_delta_norm",
        )
    )
    summary.update(
        _scalar_stats(
            [float(r.get("info_trace", 0.0)) for r in block_history],
            "block_info_trace",
        )
    )
    summary.update(
        _scalar_stats(
            [float(r.get("info_cond", 0.0)) for r in block_history],
            "block_info_cond",
        )
    )
    return summary


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _mahalanobis_distance_sq(
    target: torch.Tensor,
    mean: torch.Tensor,
    covariance: torch.Tensor | None,
) -> float:
    if covariance is None:
        return 0.0
    diff = (target.reshape(-1) - mean.reshape(-1)).to(dtype=torch.float64)
    cov = covariance.detach().reshape(diff.numel(), diff.numel()).to(dtype=torch.float64)
    eye = torch.eye(diff.numel(), dtype=torch.float64, device=cov.device)
    cov = 0.5 * (cov + cov.T) + 1e-6 * eye
    try:
        chol = torch.linalg.cholesky(cov)
        solved = torch.cholesky_solve(diff.unsqueeze(-1), chol).squeeze(-1)
    except RuntimeError:
        solved = torch.linalg.pinv(cov) @ diff
    return float(torch.dot(diff, solved).item())


def _covariance_logdet(covariance: torch.Tensor | None) -> float:
    if covariance is None:
        return 0.0
    cov = covariance.detach().to(dtype=torch.float64)
    eye = torch.eye(cov.shape[-1], dtype=torch.float64, device=cov.device)
    cov = 0.5 * (cov + cov.transpose(-1, -2)) + 1e-6 * eye
    sign, logdet = torch.linalg.slogdet(cov)
    if float(sign.item()) <= 0:
        return float("-inf")
    return float(logdet.item())


def summarize_online_id_debug_traces(
    *,
    observed_rate_hz: list[float],
    observed_zero_fraction: list[float],
    action_norm: list[float],
    field_norm: list[float],
    action_to_field_ratio: list[float],
    action_field_cosine: list[float],
    posterior_mahalanobis_sq: list[float],
    posterior_cov_trace: list[float],
    posterior_cov_logdet: list[float],
    info_theta_trace: list[float],
    info_state_trace: list[float],
    action_at_bound: list[float],
) -> dict[str, float]:
    summary: dict[str, float] = {}
    summary.update(_scalar_stats(observed_rate_hz, "observed_rate_hz"))
    summary.update(_scalar_stats(observed_zero_fraction, "observed_zero_fraction"))
    summary.update(_scalar_stats(action_norm, "action_norm"))
    summary.update(_scalar_stats(field_norm, "field_norm"))
    summary.update(_scalar_stats(action_to_field_ratio, "action_to_field_ratio"))
    summary.update(_scalar_stats(action_field_cosine, "action_field_cosine"))
    summary.update(_scalar_stats(posterior_mahalanobis_sq, "posterior_mahalanobis_sq"))
    summary.update(_scalar_stats(posterior_cov_trace, "posterior_cov_trace"))
    summary.update(_scalar_stats(posterior_cov_logdet, "posterior_cov_logdet"))
    summary.update(_scalar_stats(info_theta_trace, "info_theta_trace"))
    summary.update(_scalar_stats(info_state_trace, "info_state_trace"))
    summary.update(_scalar_stats(action_at_bound, "action_at_bound"))
    return summary


def evaluate_embedding_bank_landscape(
    *,
    spec: SystemSpec,
    meta_dynamics: MetaDynamics,
    rollout,
    bank_specs: tuple[SystemSpec, ...],
    bank_embeddings: torch.Tensor,
    dt: float,
    posterior_mean: torch.Tensor,
    posterior_covariance: torch.Tensor | None,
    max_steps: int = 400,
) -> dict[str, object]:
    if len(bank_specs) == 0 or bank_embeddings.numel() == 0:
        return {}

    env_state = rollout["env_state"]
    next_env_state = rollout["next_env_state"]
    env_action = rollout.get("env_action")

    if env_state.ndim != 3 or next_env_state.ndim != 3:
        return {}

    z_t = env_state[:, :, :2].to(device)
    z_next = next_env_state[:, :, :2].to(device)
    if env_action is None:
        u_t = torch.zeros_like(z_t)
    else:
        u_t = env_action[:, :, :2].to(device)

    n_steps = int(z_t.shape[1])
    if n_steps <= 0:
        return {}
    if n_steps > max_steps:
        idx = torch.linspace(0, n_steps - 1, steps=max_steps, device=z_t.device).round().long()
        z_t = z_t[:, idx, :]
        z_next = z_next[:, idx, :]
        u_t = u_t[:, idx, :]

    bank_embeddings = bank_embeddings.to(device=device, dtype=torch.float32)
    n_bank = int(bank_embeddings.shape[0])
    z_batch = z_t.expand(n_bank, -1, -1)
    z_next_batch = z_next.expand(n_bank, -1, -1)
    u_batch = u_t.expand(n_bank, -1, -1)

    with torch.no_grad():
        pred_dyn = meta_dynamics(z_batch, e=bank_embeddings)
        pred_next = z_batch + dt * (pred_dyn + u_batch)
        mse = ((pred_next - z_next_batch) ** 2).mean(dim=(1, 2))

    mse_np = mse.detach().cpu().numpy().astype(np.float64, copy=False)
    if mse_np.size == 0 or not np.isfinite(mse_np).any():
        return {}

    order = np.argsort(mse_np)
    true_idx = next((idx for idx, bank_spec in enumerate(bank_specs) if bank_spec.name == spec.name), None)
    true_rank = int(np.where(order == true_idx)[0][0] + 1) if true_idx is not None else None

    topk = order[: min(5, len(order))]
    top1_idx = int(order[0])
    top2_gap = (
        float(mse_np[order[1]] - mse_np[order[0]])
        if len(order) > 1 and np.isfinite(mse_np[order[1]])
        else None
    )

    emb_np = bank_embeddings.detach().cpu().numpy()
    shifted = mse_np - float(np.nanmin(mse_np))
    temp = float(np.nanmedian(shifted[np.isfinite(shifted)]))
    if not np.isfinite(temp) or temp <= 1e-8:
        temp = max(float(np.nanstd(mse_np)), 1e-3)
    logits = -shifted / temp
    logits = logits - float(np.max(logits))
    weights = np.exp(logits)
    weights = weights / max(float(np.sum(weights)), 1e-12)
    discrete_mean = np.sum(weights[:, None] * emb_np, axis=0)
    centered = emb_np - discrete_mean[None, :]
    discrete_cov = np.einsum("n,ni,nj->ij", weights, centered, centered)
    posterior_mean_np = posterior_mean.detach().cpu().numpy().reshape(-1)
    if posterior_covariance is None:
        posterior_cov_np = np.eye(posterior_mean_np.shape[0], dtype=np.float64)
    else:
        posterior_cov_np = posterior_covariance.detach().cpu().numpy().reshape(
            posterior_mean_np.shape[0], posterior_mean_np.shape[0]
        )
    entropy = float(-np.sum(weights * np.log(np.clip(weights, 1e-12, None))))
    effective_support = float(np.exp(entropy))

    return {
        "landscape_n_steps": int(z_t.shape[1]),
        "true_candidate_rank": true_rank,
        "top1_system": bank_specs[top1_idx].name,
        "top1_family": bank_specs[top1_idx].family,
        "top1_mse": float(mse_np[top1_idx]),
        "top2_gap": top2_gap,
        "top5_unique_families": int(len({bank_specs[idx].family for idx in topk})),
        "top5_systems": [bank_specs[idx].name for idx in topk],
        "discrete_posterior_entropy": entropy,
        "discrete_posterior_effective_support": effective_support,
        "discrete_posterior_mean": [float(x) for x in discrete_mean.tolist()],
        "discrete_posterior_cov_trace": float(np.trace(discrete_cov)),
        "discrete_gaussian_mean_gap": float(np.linalg.norm(discrete_mean - posterior_mean_np)),
        "discrete_gaussian_cov_gap_fro": float(
            np.linalg.norm(discrete_cov - posterior_cov_np, ord="fro")
        ),
        "true_weight": float(weights[true_idx]) if true_idx is not None else None,
        "top1_weight": float(weights[top1_idx]),
    }


def evaluate_post_probe_rollout_prediction(
    spec: SystemSpec,
    meta_dynamics: MetaDynamics,
    inferred_embedding: torch.Tensor,
    inferred_state: torch.Tensor,
    true_state: torch.Tensor,
    n_rollouts: int,
    horizon: int,
    dt: float,
    action_low: float,
    action_high: float,
    dynamics_scale: float,
    seed: int,
) -> dict[str, float]:
    n_rollouts = max(1, int(n_rollouts))
    horizon = max(1, int(horizon))
    if dt <= 0:
        raise ValueError("post-probe rollout dt must be > 0")
    if action_high <= action_low:
        raise ValueError("post-probe action_high must be > action_low")

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    actions = action_low + (action_high - action_low) * torch.rand(
        (n_rollouts, horizon, 2), generator=g
    )
    actions = actions.to(device=device, dtype=torch.float32)

    z_true = true_state.reshape(1, -1).to(device).repeat(n_rollouts, 1)
    z_hat = inferred_state.reshape(1, -1).to(device).repeat(n_rollouts, 1)
    e_hat = inferred_embedding.reshape(1, -1).to(device).repeat(n_rollouts, 1)

    with torch.no_grad():
        true_traj = rollout_true_controlled(
            spec,
            z_true,
            actions=actions,
            dt=dt,
            dynamics_scale=dynamics_scale,
        )
        pred_traj = rollout_meta_controlled(
            meta_dynamics,
            e=e_hat,
            z0=z_hat,
            actions=actions,
            dt=dt,
        )
        pred_traj_true_init = rollout_meta_controlled(
            meta_dynamics,
            e=e_hat,
            z0=z_true,
            actions=actions,
            dt=dt,
        )

    if not (
        torch.isfinite(true_traj).all()
        and torch.isfinite(pred_traj).all()
        and torch.isfinite(pred_traj_true_init).all()
    ):
        return {
            "rollout_mse": float("inf"),
            "final_state_mse": float("inf"),
            "trajectory_r2": float("-inf"),
            "rollout_mse_true_init": float("inf"),
            "final_state_mse_true_init": float("inf"),
            "eval_rollout_count": float(n_rollouts),
            "eval_rollout_horizon": float(horizon),
            "eval_rollout_dt": float(dt),
            "eval_action_low": float(action_low),
            "eval_action_high": float(action_high),
        }

    return {
        "rollout_mse": float(F.mse_loss(pred_traj, true_traj).item()),
        "final_state_mse": float(F.mse_loss(pred_traj[:, -1], true_traj[:, -1]).item()),
        "trajectory_r2": float(trajectory_r2(pred_traj, true_traj).mean().item()),
        "rollout_mse_true_init": float(F.mse_loss(pred_traj_true_init, true_traj).item()),
        "final_state_mse_true_init": float(
            F.mse_loss(pred_traj_true_init[:, -1], true_traj[:, -1]).item()
        ),
        "eval_rollout_count": float(n_rollouts),
        "eval_rollout_horizon": float(horizon),
        "eval_rollout_dt": float(dt),
        "eval_action_low": float(action_low),
        "eval_action_high": float(action_high),
    }


def persist_online_id_run_artifacts(
    *,
    experiment: actdyn.core.experiment.MetaEmbeddingExperiment,
    record: dict,
) -> dict:
    session_dir = str(experiment.results_path)
    rollout_path = os.path.join(session_dir, "rollouts", f"rollout_{int(experiment.env_step)}.pkl")
    save_rollout(experiment.rollout, rollout_path)
    rollouts_dir = os.path.join(session_dir, "rollouts")
    if os.path.isdir(rollouts_dir):
        full_rollout = load_and_concatenate_rollouts(rollouts_dir, device="cpu")
        save_rollout(full_rollout, rollout_path)
    record_path = os.path.join(session_dir, "online_id_record.json")
    payload = {
        **record,
        "n_steps": int(experiment.env_step),
        "session_dir": session_dir,
        "rollout_path": rollout_path,
        "record_path": record_path,
    }
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def load_existing_online_id_record(
    run_dir: str,
    *,
    system: str,
    policy: str,
    seed: int,
) -> dict[str, object] | None:
    if not os.path.isdir(run_dir):
        return None

    record_paths: list[str] = []
    for root, _dirs, files in os.walk(run_dir):
        if "online_id_record.json" in files:
            record_paths.append(os.path.join(root, "online_id_record.json"))
    if not record_paths:
        return None

    record_paths.sort(key=lambda path: (os.path.getmtime(path), path))
    record_path = record_paths[-1]
    with open(record_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {record_path}")

    loaded_system = str(payload.get("system", ""))
    loaded_policy = str(payload.get("policy", ""))
    try:
        loaded_seed = int(payload.get("seed", -1))
    except (TypeError, ValueError):
        loaded_seed = -1
    if loaded_system != system or loaded_policy != policy or loaded_seed != int(seed):
        raise ValueError(
            "Existing online-ID record does not match requested run: "
            f"{record_path} has system={loaded_system!r}, policy={loaded_policy!r}, seed={loaded_seed}"
        )

    return dict(payload)


def _current_plan_index(policy) -> int:
    chunk = max(1, int(getattr(policy, "chunk", 1)))
    count = max(0, int(getattr(policy, "count", 0)))
    if count <= 0:
        return 0
    return (count - 1) % chunk


def _extract_remaining_plan_actions(policy):
    plan = None
    elite_actions = getattr(policy, "elite_actions", None)
    if elite_actions is not None:
        elite_actions = torch.as_tensor(elite_actions).detach()
        if elite_actions.ndim == 3 and elite_actions.shape[0] > 0:
            plan = elite_actions[0]

    if plan is None:
        mean_actions = getattr(policy, "mean", None)
        if mean_actions is not None:
            mean_actions = torch.as_tensor(mean_actions).detach()
            if mean_actions.ndim == 2:
                plan = mean_actions
            elif mean_actions.ndim == 3 and mean_actions.shape[0] > 0:
                plan = mean_actions[0]

    if plan is None or plan.ndim != 2 or plan.shape[0] == 0:
        return None

    start = min(_current_plan_index(policy), int(plan.shape[0] - 1))
    return plan[start:].unsqueeze(0)


def _predict_planned_xy_trajectory(*, model, policy, transition: dict[str, object]):
    if getattr(policy, "metric", None) is None:
        return None

    planned_actions = _extract_remaining_plan_actions(policy)
    if planned_actions is None:
        return None

    model_state = transition.get("model_state")
    if model_state is None:
        return None

    state = torch.as_tensor(model_state).detach()
    if state.ndim == 1:
        state = state.reshape(1, 1, -1)
    elif state.ndim == 2:
        state = state.unsqueeze(0)
    if state.ndim != 3 or state.shape[-1] == 0:
        return None

    model_device = getattr(model, "device", state.device)
    state = state.to(model_device)
    planned_actions = planned_actions.to(model_device)

    prev_state = None
    try:
        current_state = model.get_state()
        if current_state is not None:
            prev_state = current_state.detach().clone()
    except Exception:
        prev_state = None

    try:
        with torch.no_grad():
            model.set_state(state)
            if model.action_encoder is not None:
                encoded_actions = model.action_encoder(planned_actions)
            else:
                encoded_actions = planned_actions
            predicted = model.predict(encoded_actions)
            trajectory = torch.cat([state, predicted], dim=-2)
    except Exception:
        return None
    finally:
        if prev_state is not None:
            model.set_state(prev_state)

    trajectory = trajectory.detach().cpu().reshape(-1, trajectory.shape[-1]).numpy()
    xy = np.zeros((trajectory.shape[0], 2), dtype=np.float32)
    if trajectory.shape[1] > 0:
        xy[:, 0] = trajectory[:, 0].astype(np.float32, copy=False)
    if trajectory.shape[1] > 1:
        xy[:, 1] = trajectory[:, 1].astype(np.float32, copy=False)
    return xy


def run_identification(
    spec: SystemSpec,
    meta_dynamics: MetaDynamics,
    system_embedding: torch.Tensor,
    embedding_mode: str,
    policy_name: str,
    results_dir: str,
    total_steps: int,
    seed: int,
    active_cfg: ActivePolicyConfig,
    eval_rollout_horizon: int,
    eval_rollout_dt: float,
    eval_rollout_count: int,
    dynamics_scale: float,
    save_acq_map: bool,
    acq_map_interval: int,
    acq_map_grid: int,
    acq_map_lim: float,
    observation_mean_firing: float = 1000.0,
    q_theta: float = 1e-4,
    k_theta: int = 10,
    q_theta_meas_coeff: float = 0.0,
    q_theta_max_scale: float = 10.0,
    state_init_uncertainty: float = 1.0,
    reference_bank_specs: tuple[SystemSpec, ...] | None = None,
    reference_bank_embeddings: torch.Tensor | None = None,
    bank_landscape_max_steps: int = 400,
    record_metadata: dict[str, object] | None = None,
):
    dt = float(eval_rollout_dt)
    dy = 30
    e_true = system_embedding.detach().cpu().reshape(-1).to(torch.float32)
    de = int(e_true.numel())
    du = 2
    noise_scale = 0.02
    action_strength = float(active_cfg.action_strength)
    initial_state_bounds = (-2.0, 2.0)

    action_model = actdyn.environment.action.IdentityActionEncoder(
        d_action=du,
        d_latent=2,
        action_bounds=[-action_strength, action_strength],
        device=device,
    )
    obs_model = build_observation_model(
        dy=dy,
        dt=dt,
        noise_scale=noise_scale,
        mean_firing=observation_mean_firing,
    )
    env = actdyn.environment.EnvWrapper(
        MixedDynamicsEnv(
            spec=spec,
            embedding_vector=e_true,
            dt=dt,
            Q=noise_scale,
            state_bounds=initial_state_bounds,
            action_bounds=(
                float(action_model.action_space.low.min()),
                float(action_model.action_space.high.max()),
            ),
            dynamics_scale=dynamics_scale,
            device=device,
        ),
        obs_model,
        action_model,
        dt=dt,
        device=device,
    )

    mapping = actdyn.models.decoder.LogLinearMapping(latent_dim=2, obs_dim=dy, dt=dt, device=device)
    noise = actdyn.models.decoder.PoissonNoise(obs_dim=dy, sigma=0.01, device=device)
    decoder = actdyn.models.Decoder(mapping=mapping, noise=noise, device=device)
    dynamics = FunctionDynamics(state_dim=2, dt=dt, dynamics_fn=meta_dynamics, device=device)
    dynamics.logvar = nn.Parameter(torch.log(torch.ones(1, 2) * noise_scale).to(device))

    sigma_0 = 5e-2
    e_bel = {
        "m": torch.zeros(1, de, device=device),
        "P": sigma_0 * torch.eye(de, device=device).unsqueeze(0),
        "L": (1.0 / sigma_0) * torch.eye(de, device=device).unsqueeze(0),
    }
    policy_cfg = resolve_online_id_policy_config(policy_name=policy_name, active_cfg=active_cfg)
    resolved_k_theta = int(policy_cfg.get("k_theta", k_theta))
    model = actdyn.models.FilteringEmbedding(
        dynamics=dynamics,
        decoder=decoder,
        e=e_bel,
        action_encoder=action_model,
        Fe=ExactFe(meta_dynamics),
        Fz=ExactFz(meta_dynamics),
        q_theta=q_theta,
        k_theta=resolved_k_theta,
        q_theta_meas_coeff=q_theta_meas_coeff,
        q_theta_max_scale=q_theta_max_scale,
        state_init_uncertainty=state_init_uncertainty,
        device=device,
    )
    model.set_params(e_bel["m"])
    e_init = model.embedding.reshape(-1).detach().cpu().clone()
    acq_interval = max(1, int(acq_map_interval))
    acq_grid_n = max(25, int(acq_map_grid))
    acq_grid_lim = float(acq_map_lim)
    acq_map_steps: list[int] = []
    acq_map_frames: list[np.ndarray] = []
    planned_traj_steps: list[int] = []
    planned_traj_frames: list[np.ndarray] = []
    observed_rate_hz_trace: list[float] = []
    observed_zero_fraction_trace: list[float] = []
    action_norm_trace: list[float] = []
    field_norm_trace: list[float] = []
    action_to_field_ratio_trace: list[float] = []
    action_field_cosine_trace: list[float] = []
    posterior_mahalanobis_sq_trace: list[float] = []
    posterior_cov_trace_trace: list[float] = []
    posterior_cov_logdet_trace: list[float] = []
    info_theta_trace: list[float] = []
    info_state_trace: list[float] = []
    action_at_bound_trace: list[float] = []
    acq_axis = np.linspace(-acq_grid_lim, acq_grid_lim, acq_grid_n, dtype=np.float32)

    emb_metric = actdyn.metrics.information.EmbeddingFisherMetric(
        model=model,
        Fe_net=ExactFe(meta_dynamics),
        Fz_net=ExactFz(meta_dynamics),
        gamma=0.99,
        sensitivity_mode="local",
        device=device,
    )
    action_metric = actdyn.metrics.cost.ActionCost()
    composite_metric = actdyn.metrics.CompositeMetric(
        [emb_metric, action_metric],
        weights=[1.0, float(active_cfg.action_cost_weight)],
        device=device,
    )
    random_policy = actdyn.policy.RandomPolicy(action_space=env.action_space, device=device)
    no_policy = actdyn.policy.OffPolicy(action_space=env.action_space, device=device)
    ciss_active_policy = actdyn.policy.mpc.MpcICem(
        metric=composite_metric,
        model=model,
        device=device,
        horizon=int(policy_cfg["planning_horizon"]),
        num_iterations=int(policy_cfg["num_iterations"]),
        num_samples=int(policy_cfg["num_samples"]),
        num_elite=int(policy_cfg["num_elite"]),
        chunk=int(policy_cfg["planning_chunk"]),
        verbose=False,
    )
    policy_map = {
        "active_long": ciss_active_policy,
        "active_short": ciss_active_policy,
        "active_chunk": ciss_active_policy,
        "async_windowed_update": ciss_active_policy,
        "random": random_policy,
        "no_policy": no_policy,
    }
    if policy_name not in policy_map:
        raise ValueError(
            f"Unknown online-ID policy: {policy_name}. Expected one of {sorted(policy_map)}"
        )
    policy = policy_map[policy_name]
    capture_acq_map = bool(save_acq_map and policy_name in ONLINE_ID_ACTIVE_POLICIES)
    if capture_acq_map:
        acq_X, acq_Y = np.meshgrid(acq_axis, acq_axis, indexing="xy")
        acq_points = torch.as_tensor(
            np.stack([acq_X.reshape(-1), acq_Y.reshape(-1)], axis=1),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(1)
    else:
        acq_points = None
    agent = actdyn.Agent(env=env, model=model, buffer_length=10, policy=policy, device=device)
    config = ExperimentConfig.from_yaml(resolve_online_id_config_path())
    config.results_dir = ensure_dir(results_dir)
    config.training.total_steps = total_steps
    config.training.rollout_horizon = 100
    config.training.train_every = max(total_steps + 1, 10_000)
    config.training.n_epochs = 0
    decoder.set_params(obs_model)
    torch.manual_seed(seed)
    np.random.seed(seed)
    experiment = actdyn.core.experiment.MetaEmbeddingExperiment(agent=agent, config=config)
    experiment.e_norm = []
    experiment.e_trace = []

    def _on_step_end(_transition: dict) -> None:
        e_bel = experiment.agent.model.embedding.reshape(-1)
        experiment.training_info["e"] = e_bel

        e_true_step = experiment.agent.env.env.get_params().detach().cpu().reshape(-1)
        if experiment.writer is not None:
            experiment.writer.add_scalars(
                "e",
                {f"true_{i}": float(v) for i, v in enumerate(e_true_step.tolist())},
                experiment.env_step,
            )
        experiment.e_trace.append([float(v) for v in e_bel.detach().cpu().reshape(-1).tolist()])
        experiment.e_norm.append(float(torch.norm(e_bel.detach().cpu() - e_true_step).item()))
        experiment.training_info["e_norm"] = experiment.e_norm[-1]

        next_obs = torch.as_tensor(_transition.get("next_obs", torch.zeros(dy))).detach().cpu().reshape(-1)
        if next_obs.numel() > 0:
            observed_rate_hz_trace.append(float(next_obs.mean().item() / max(dt, 1e-8)))
            observed_zero_fraction_trace.append(float((next_obs <= 0).float().mean().item()))

        env_state = torch.as_tensor(_transition.get("env_state", torch.zeros(2))).detach().cpu().reshape(-1)[:2]
        env_action = torch.as_tensor(_transition.get("env_action", torch.zeros(2))).detach().cpu().reshape(-1)[:2]
        true_field = true_dynamics_from_spec(
            spec,
            env_state.to(device=device, dtype=torch.float32).reshape(1, 2),
            dynamics_scale=dynamics_scale,
        ).detach().cpu().reshape(-1)[:2]
        action_norm = float(torch.linalg.norm(env_action).item())
        field_norm = float(torch.linalg.norm(true_field).item())
        action_norm_trace.append(action_norm)
        field_norm_trace.append(field_norm)
        action_to_field_ratio_trace.append(action_norm / max(field_norm, 1e-6))
        action_field_cosine_trace.append(
            _cosine_similarity(env_action.numpy(), true_field.numpy())
        )
        action_at_bound_trace.append(
            float(
                max(float(torch.abs(env_action[0]).item()), float(torch.abs(env_action[1]).item()))
                >= action_strength - 1e-6
            )
        )

        cov = model.e.get("P")
        cov_step = cov[0].detach().cpu() if cov is not None and cov.ndim >= 3 else None
        posterior_mahalanobis_sq_trace.append(
            _mahalanobis_distance_sq(e_true_step, e_bel.detach().cpu(), cov_step)
        )
        posterior_cov_trace_trace.append(
            float(torch.trace(cov_step).item()) if cov_step is not None else 0.0
        )
        posterior_cov_logdet_trace.append(_covariance_logdet(cov_step))

        info_diag = getattr(model, "last_information", {}) or {}
        info_theta_trace.append(float(info_diag.get("I_theta_t", 0.0)))
        info_state_trace.append(float(info_diag.get("I_z_t", 0.0)))

        if capture_acq_map and acq_points is not None and experiment.env_step % acq_interval == 0:
            with torch.no_grad():
                map_rollout = {
                    "model_state": acq_points,
                    "next_model_state": acq_points,
                }
                acq_cost = emb_metric(map_rollout).detach().reshape(-1)
            acq_map = (-acq_cost).cpu().numpy().reshape(acq_grid_n, acq_grid_n)
            acq_map = np.nan_to_num(acq_map, nan=0.0, posinf=1e6, neginf=0.0).astype(np.float32)
            acq_map_frames.append(acq_map)
            acq_map_steps.append(int(experiment.env_step))

        planned_traj_xy = _predict_planned_xy_trajectory(
            model=model,
            policy=policy,
            transition=_transition,
        )
        if planned_traj_xy is not None and planned_traj_xy.shape[0] >= 2:
            planned_traj_steps.append(int(experiment.env_step))
            planned_traj_frames.append(planned_traj_xy)

    experiment._run_online_loop(
        train_cfg=config.training,
        pbar_desc="Embedding",
        plot_fcn=None,
        reset=True,
        on_step_end=_on_step_end,
    )

    e_hat = model.embedding.reshape(-1).detach().cpu()
    step_trace = [dict(item) for item in getattr(model, "embedding_diag_history", [])]
    block_trace = [dict(item) for item in getattr(model, "embedding_block_update_history", [])]
    diag_summary = summarize_embedding_diagnostics(
        step_history=step_trace,
        block_history=block_trace,
        e_init=e_init,
        e_final=e_hat,
    )
    debug_diag_summary = summarize_online_id_debug_traces(
        observed_rate_hz=observed_rate_hz_trace,
        observed_zero_fraction=observed_zero_fraction_trace,
        action_norm=action_norm_trace,
        field_norm=field_norm_trace,
        action_to_field_ratio=action_to_field_ratio_trace,
        action_field_cosine=action_field_cosine_trace,
        posterior_mahalanobis_sq=posterior_mahalanobis_sq_trace,
        posterior_cov_trace=posterior_cov_trace_trace,
        posterior_cov_logdet=posterior_cov_logdet_trace,
        info_theta_trace=info_theta_trace,
        info_state_trace=info_state_trace,
        action_at_bound=action_at_bound_trace,
    )
    final_error = float(torch.norm(e_hat - e_true).item())
    posterior_covariance = model.e.get("P")
    posterior_covariance = (
        posterior_covariance[0].detach().cpu() if posterior_covariance is not None and posterior_covariance.ndim >= 3 else None
    )
    final_posterior_mahalanobis_sq = _mahalanobis_distance_sq(e_true, e_hat, posterior_covariance)
    post_probe_eval = evaluate_post_probe_rollout_prediction(
        spec=spec,
        meta_dynamics=meta_dynamics,
        inferred_embedding=e_hat,
        inferred_state=agent._model_state.detach(),
        true_state=agent._env_state.detach(),
        n_rollouts=eval_rollout_count,
        horizon=eval_rollout_horizon,
        dt=eval_rollout_dt,
        action_low=float(action_model.action_space.low.min()),
        action_high=float(action_model.action_space.high.max()),
        dynamics_scale=dynamics_scale,
        seed=int(seed) + 10_000,
    )
    landscape_summary = {}
    if reference_bank_specs and reference_bank_embeddings is not None:
        landscape_summary = evaluate_embedding_bank_landscape(
            spec=spec,
            meta_dynamics=meta_dynamics,
            rollout=experiment.rollout,
            bank_specs=reference_bank_specs,
            bank_embeddings=reference_bank_embeddings,
            dt=dt,
            posterior_mean=e_hat,
            posterior_covariance=posterior_covariance,
            max_steps=bank_landscape_max_steps,
        )
    acquisition_map_trace_path = None
    if capture_acq_map and acq_map_frames:
        acquisition_map_trace_path = os.path.join(
            str(experiment.results_path), "acquisition_map_trace.npz"
        )
        np.savez_compressed(
            acquisition_map_trace_path,
            steps=np.asarray(acq_map_steps, dtype=np.int64),
            axis=acq_axis.astype(np.float32),
            maps=np.asarray(acq_map_frames, dtype=np.float32),
            policy=np.asarray([policy_name], dtype=object),
            system=np.asarray([spec.name], dtype=object),
        )
    planned_trajectory_trace_path = None
    if planned_traj_frames:
        planned_trajectory_trace_path = os.path.join(
            str(experiment.results_path), "planned_trajectory_trace.npz"
        )
        max_points = max(frame.shape[0] for frame in planned_traj_frames)
        paths = np.full((len(planned_traj_frames), max_points, 2), np.nan, dtype=np.float32)
        lengths = np.zeros((len(planned_traj_frames),), dtype=np.int64)
        for idx, frame in enumerate(planned_traj_frames):
            n_points = int(frame.shape[0])
            paths[idx, :n_points, :] = frame[:, :2]
            lengths[idx] = n_points
        np.savez_compressed(
            planned_trajectory_trace_path,
            steps=np.asarray(planned_traj_steps, dtype=np.int64),
            paths=paths,
            lengths=lengths,
            policy=np.asarray([policy_name], dtype=object),
            system=np.asarray([spec.name], dtype=object),
        )
    record = {
        "system": spec.name,
        "family": spec.family,
        "policy": policy_name,
        "seed": seed,
        "embedding_mode": embedding_mode,
        "active_policy_cfg": {**asdict(active_cfg), **policy_cfg},
        "embedding_true": e_true.tolist(),
        "embedding_est": e_hat.tolist(),
        "embedding_trace": list(getattr(experiment, "e_trace", [])),
        "final_error": final_error,
        "error_trace": [float(x) for x in experiment.e_norm],
        "post_probe_eval": post_probe_eval,
        "diagnostics": diag_summary,
        "debug_diagnostics": debug_diag_summary,
        "diagnostic_step_trace": step_trace,
        "diagnostic_block_trace": block_trace,
        "final_posterior_mahalanobis_sq": final_posterior_mahalanobis_sq,
        "final_posterior_cov_trace": (
            float(torch.trace(posterior_covariance).item()) if posterior_covariance is not None else 0.0
        ),
        "final_posterior_cov_logdet": _covariance_logdet(posterior_covariance),
        "observation_mean_firing": float(observation_mean_firing),
        "filter_q_theta": float(q_theta),
        "filter_k_theta": int(resolved_k_theta),
        "filter_q_theta_meas_coeff": float(q_theta_meas_coeff),
        "filter_q_theta_max_scale": float(q_theta_max_scale),
        "filter_state_init_uncertainty": float(state_init_uncertainty),
        "update_scheme": str(policy_cfg["update_scheme"]),
        "state_update_interval": int(policy_cfg["state_update_interval"]),
        "predictive_only_window": bool(policy_cfg["predictive_only_window"]),
        "planning_horizon": int(policy_cfg["planning_horizon"]),
        "planning_chunk": int(policy_cfg["planning_chunk"]),
        "online_id_dt": float(dt),
        "initial_state_bounds": [float(initial_state_bounds[0]), float(initial_state_bounds[1])],
        "save_acq_map": bool(capture_acq_map),
        "acq_map_interval": int(acq_interval),
        "acq_map_grid": int(acq_grid_n),
        "acq_map_lim": float(acq_grid_lim),
        "acquisition_map_trace_path": acquisition_map_trace_path,
        "planned_trajectory_trace_path": planned_trajectory_trace_path,
        "embedding_bank_landscape": landscape_summary,
    }
    if record_metadata:
        record.update(dict(record_metadata))
    return persist_online_id_run_artifacts(experiment=experiment, record=record)


def summarize_record_group(records: list[dict]) -> dict[str, float]:
    payload: dict[str, float] = {
        "n": len(records),
        "mean_rollout_mse": float(np.mean([r["post_probe_eval"]["rollout_mse"] for r in records])),
        "std_rollout_mse": float(np.std([r["post_probe_eval"]["rollout_mse"] for r in records])),
        "mean_final_state_mse": float(
            np.mean([r["post_probe_eval"]["final_state_mse"] for r in records])
        ),
        "std_final_state_mse": float(
            np.std([r["post_probe_eval"]["final_state_mse"] for r in records])
        ),
        "mean_trajectory_r2": float(
            np.mean([r["post_probe_eval"]["trajectory_r2"] for r in records])
        ),
        "std_trajectory_r2": float(
            np.std([r["post_probe_eval"]["trajectory_r2"] for r in records])
        ),
        "mean_rollout_mse_true_init": float(
            np.mean([r["post_probe_eval"]["rollout_mse_true_init"] for r in records])
        ),
        "std_rollout_mse_true_init": float(
            np.std([r["post_probe_eval"]["rollout_mse_true_init"] for r in records])
        ),
        "mean_final_state_mse_true_init": float(
            np.mean([r["post_probe_eval"]["final_state_mse_true_init"] for r in records])
        ),
        "std_final_state_mse_true_init": float(
            np.std([r["post_probe_eval"]["final_state_mse_true_init"] for r in records])
        ),
        "mean_final_error": float(np.mean([r["final_error"] for r in records])),
        "std_final_error": float(np.std([r["final_error"] for r in records])),
    }
    diag_keys = [
        "embedding_drift_norm",
        "embedding_path_norm",
        "embedding_step_norm_mean",
        "info_trace_mean",
        "info_cond_median",
        "fe_fro_mean",
        "fz_fro_mean",
        "block_embed_delta_norm_mean",
        "block_info_trace_mean",
    ]
    for key in diag_keys:
        vals = [float(r.get("diagnostics", {}).get(key, 0.0)) for r in records]
        payload[f"mean_{key}"] = float(np.mean(vals))
        payload[f"std_{key}"] = float(np.std(vals))
    return payload


def summarize_by_group(records: list[dict], field: str) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[str, dict[str, dict[str, float]]] = {}
    groups = sorted({str(r[field]) for r in records})
    policies = sorted({r["policy"] for r in records})
    for group in groups:
        grouped[group] = {}
        for policy in policies:
            subset = [r for r in records if str(r[field]) == group and r["policy"] == policy]
            if subset:
                grouped[group][policy] = summarize_record_group(subset)
    return grouped


def _finite_metric(value: object, *, cutoff: float = 1e308) -> float | None:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(scalar):
        return None
    if abs(scalar) >= float(cutoff):
        return None
    return scalar


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr))


def build_online_id_metrics_rows(records: list[dict]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    systems = sorted({str(r["system"]) for r in records})
    policies = sorted({str(r["policy"]) for r in records})
    for system in systems:
        for policy in policies:
            subset = [r for r in records if str(r["system"]) == system and str(r["policy"]) == policy]
            if not subset:
                continue
            family = str(subset[0]["family"])
            final_error_values = [float(r["final_error"]) for r in subset]
            rollout_values = [
                _finite_metric(r.get("post_probe_eval", {}).get("rollout_mse")) for r in subset
            ]
            final_state_values = [
                _finite_metric(r.get("post_probe_eval", {}).get("final_state_mse")) for r in subset
            ]
            traj_r2_values = [
                _finite_metric(r.get("post_probe_eval", {}).get("trajectory_r2")) for r in subset
            ]
            rollout_values_finite = [v for v in rollout_values if v is not None]
            final_state_values_finite = [v for v in final_state_values if v is not None]
            traj_r2_values_finite = [v for v in traj_r2_values if v is not None]
            final_error_mean, final_error_std = _mean_std(final_error_values)
            rollout_mean, rollout_std = _mean_std(rollout_values_finite)
            final_state_mean, final_state_std = _mean_std(final_state_values_finite)
            traj_r2_mean, traj_r2_std = _mean_std(traj_r2_values_finite)
            best_rollout_record = min(
                (
                    r
                    for r in subset
                    if _finite_metric(r.get("post_probe_eval", {}).get("rollout_mse")) is not None
                ),
                key=lambda r: float(r["post_probe_eval"]["rollout_mse"]),
                default=None,
            )
            rows.append(
                {
                    "system": system,
                    "family": family,
                    "policy": policy,
                    "n_runs": len(subset),
                    "n_finite_rollout": len(rollout_values_finite),
                    "rollout_mse_mean": rollout_mean,
                    "rollout_mse_std": rollout_std,
                    "final_state_mse_mean": final_state_mean,
                    "final_state_mse_std": final_state_std,
                    "trajectory_r2_mean": traj_r2_mean,
                    "trajectory_r2_std": traj_r2_std,
                    "final_error_mean": final_error_mean,
                    "final_error_std": final_error_std,
                    "best_rollout_mse": (
                        float(best_rollout_record["post_probe_eval"]["rollout_mse"])
                        if best_rollout_record is not None
                        else None
                    ),
                    "best_seed": int(best_rollout_record["seed"]) if best_rollout_record is not None else None,
                    "best_session_dir": (
                        str(best_rollout_record.get("session_dir")) if best_rollout_record is not None else None
                    ),
                }
            )
    return rows


def build_online_id_error_trace_rows(records: list[dict]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for policy in sorted({str(r["policy"]) for r in records}):
        traces: list[np.ndarray] = []
        for record in records:
            if str(record["policy"]) != policy:
                continue
            trace = np.asarray(record.get("error_trace", []), dtype=np.float64)
            if trace.size == 0:
                continue
            trace = np.where(np.isfinite(trace), trace, np.nan)
            traces.append(trace)
        if not traces:
            continue
        max_len = max(trace.shape[0] for trace in traces)
        mat = np.full((len(traces), max_len), np.nan, dtype=np.float64)
        for idx, trace in enumerate(traces):
            mat[idx, : trace.shape[0]] = trace
        means = np.nanmean(mat, axis=0)
        stds = np.nanstd(mat, axis=0)
        counts = np.sum(np.isfinite(mat), axis=0)
        for step_idx in range(max_len):
            if int(counts[step_idx]) == 0:
                continue
            rows.append(
                {
                    "policy": policy,
                    "step": step_idx,
                    "value_mean": float(means[step_idx]),
                    "value_std": float(stds[step_idx]),
                    "n": int(counts[step_idx]),
                }
            )
    return rows


def write_online_id_metrics_csv(path: str, rows: list[dict[str, object]]) -> str:
    ensure_dir(os.path.dirname(path))
    fields = [
        "system",
        "family",
        "policy",
        "n_runs",
        "n_finite_rollout",
        "rollout_mse_mean",
        "rollout_mse_std",
        "final_state_mse_mean",
        "final_state_mse_std",
        "trajectory_r2_mean",
        "trajectory_r2_std",
        "final_error_mean",
        "final_error_std",
        "best_rollout_mse",
        "best_seed",
        "best_session_dir",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_online_id_error_trace_csv(path: str, rows: list[dict[str, object]]) -> str:
    ensure_dir(os.path.dirname(path))
    fields = ["policy", "step", "value_mean", "value_std", "n"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_online_id_summary_markdown(
    out_path: str,
    *,
    records: list[dict],
    metric_rows: list[dict[str, object]],
    figure_paths: list[str],
) -> str:
    def _fmt_metric(value: object) -> str:
        scalar = _finite_metric(value)
        if scalar is None:
            return "NA"
        return f"{scalar:.6f}"

    ensure_dir(os.path.dirname(out_path))
    policies = sorted({str(r["policy"]) for r in records})
    systems = sorted({str(r["system"]) for r in records})
    finite_rollout_records = [
        r for r in records if _finite_metric(r.get("post_probe_eval", {}).get("rollout_mse")) is not None
    ]
    lines: list[str] = []
    lines.append("# Cosyne Online-ID Summary")
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append(f"- Records: {len(records)}")
    lines.append(f"- Systems: {len(systems)}")
    lines.append(f"- Policies: {', '.join(policies)}")
    lines.append(f"- Finite rollout records: {len(finite_rollout_records)} / {len(records)}")
    lines.append("")
    lines.append("## Per-System Metrics")
    lines.append("")
    lines.append(
        "| system | policy | n_runs | finite_rollout | rollout_mse_mean | final_error_mean | best_seed |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in metric_rows:
        rollout_val = row["rollout_mse_mean"]
        final_error_val = row["final_error_mean"]
        best_seed = row["best_seed"]
        lines.append(
            "| "
            f"{row['system']} | {row['policy']} | {int(row['n_runs'])} | {int(row['n_finite_rollout'])} | "
            f"{_fmt_metric(rollout_val)} | "
            f"{_fmt_metric(final_error_val)} | "
            f"{best_seed if best_seed is not None else 'NA'} |"
        )
    lines.append("")
    lines.append("## Best Finite Record by Policy")
    lines.append("")
    lines.append("| policy | system | seed | rollout_mse | session_dir |")
    lines.append("| --- | --- | ---: | ---: | --- |")
    for policy in policies:
        best_record = min(
            (
                r
                for r in records
                if str(r["policy"]) == policy
                and _finite_metric(r.get("post_probe_eval", {}).get("rollout_mse")) is not None
            ),
            key=lambda r: float(r["post_probe_eval"]["rollout_mse"]),
            default=None,
        )
        if best_record is None:
            lines.append(f"| {policy} | NA | NA | NA | NA |")
            continue
        lines.append(
            f"| {policy} | {best_record['system']} | {int(best_record['seed'])} | "
            f"{float(best_record['post_probe_eval']['rollout_mse']):.6f} | "
            f"`{best_record.get('session_dir', 'NA')}` |"
        )
    if figure_paths:
        lines.append("")
        lines.append("## Figures")
        lines.append("")
        for figure_path in figure_paths:
            lines.append(f"- `{os.path.basename(figure_path)}`")
    text = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return out_path


def save_online_id_summary_figures(
    out_dir: str,
    *,
    metric_rows: list[dict[str, object]],
    error_trace_rows: list[dict[str, object]],
) -> list[str]:
    ensure_dir(out_dir)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    figure_paths: list[str] = []
    policies = sorted({str(row["policy"]) for row in metric_rows})
    systems = sorted({str(row["system"]) for row in metric_rows})

    if metric_rows:
        x = np.arange(len(systems), dtype=np.float64)
        width = 0.8 / max(len(policies), 1)

        fig, ax = plt.subplots(figsize=(10.5, 5.2))
        for idx, policy in enumerate(policies):
            subset = {str(row["system"]): row for row in metric_rows if str(row["policy"]) == policy}
            ys = []
            yerr = []
            for system in systems:
                row = subset.get(system)
                ys.append(np.nan if row is None or row["final_error_mean"] is None else float(row["final_error_mean"]))
                yerr.append(np.nan if row is None or row["final_error_std"] is None else float(row["final_error_std"]))
            offset = (idx - (len(policies) - 1) / 2.0) * width
            ax.bar(x + offset, ys, width=width, yerr=yerr, capsize=3, label=policy)
        ax.set_xticks(x)
        ax.set_xticklabels(systems, rotation=20)
        ax.set_ylabel("Final embedding error (mean +/- std)")
        ax.set_title("Online-ID final embedding error by system and policy")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(loc="best")
        fig.tight_layout()
        final_error_path = os.path.join(out_dir, "final_error_by_system_policy.png")
        fig.savefig(final_error_path, dpi=150)
        plt.close(fig)
        figure_paths.append(final_error_path)

        fig, ax = plt.subplots(figsize=(10.5, 5.2))
        for idx, policy in enumerate(policies):
            subset = {str(row["system"]): row for row in metric_rows if str(row["policy"]) == policy}
            ys = []
            yerr = []
            for system in systems:
                row = subset.get(system)
                ys.append(np.nan if row is None or row["rollout_mse_mean"] is None else float(row["rollout_mse_mean"]))
                yerr.append(np.nan if row is None or row["rollout_mse_std"] is None else float(row["rollout_mse_std"]))
            offset = (idx - (len(policies) - 1) / 2.0) * width
            ax.bar(x + offset, ys, width=width, yerr=yerr, capsize=3, label=policy)
        ax.set_xticks(x)
        ax.set_xticklabels(systems, rotation=20)
        ax.set_ylabel("Post-probe rollout MSE (mean +/- std)")
        ax.set_title("Online-ID rollout MSE by system and policy")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(loc="best")
        fig.tight_layout()
        rollout_path = os.path.join(out_dir, "rollout_mse_by_system_policy.png")
        fig.savefig(rollout_path, dpi=150)
        plt.close(fig)
        figure_paths.append(rollout_path)

    if error_trace_rows:
        fig, ax = plt.subplots(figsize=(9.5, 5.0))
        for policy in policies:
            series = [row for row in error_trace_rows if str(row["policy"]) == policy]
            series.sort(key=lambda row: int(row["step"]))
            xs = [int(row["step"]) for row in series]
            ys = np.asarray([float(row["value_mean"]) for row in series], dtype=np.float64)
            std = np.asarray([float(row["value_std"]) for row in series], dtype=np.float64)
            ax.plot(xs, ys, label=policy)
            ax.fill_between(xs, ys - std, ys + std, alpha=0.18)
        ax.set_xlabel("Environment step")
        ax.set_ylabel("Embedding error norm (mean +/- std)")
        ax.set_title("Online-ID embedding error over steps")
        ax.grid(alpha=0.2)
        ax.legend(loc="best")
        fig.tight_layout()
        trace_path = os.path.join(out_dir, "embedding_error_over_steps.png")
        fig.savefig(trace_path, dpi=150)
        plt.close(fig)
        figure_paths.append(trace_path)

    return figure_paths


def summarize_identification_results(records: list[dict], out_path: str):
    summary: dict[str, dict[str, float]] = {}
    for policy in sorted({r["policy"] for r in records}):
        subset = [r for r in records if r["policy"] == policy]
        summary[policy] = summarize_record_group(subset)
    payload = {
        "primary_metric": "rollout_mse",
        "embedding_modes": sorted({str(r.get("embedding_mode", "unknown")) for r in records}),
        "records": records,
        "summary": summary,
        "per_system_policy": summarize_by_group(records, "system"),
        "per_family_policy": summarize_by_group(records, "family"),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    out_dir = os.path.dirname(out_path)
    metric_rows = build_online_id_metrics_rows(records)
    error_trace_rows = build_online_id_error_trace_rows(records)
    metrics_csv_path = write_online_id_metrics_csv(os.path.join(out_dir, "metrics.csv"), metric_rows)
    error_trace_csv_path = write_online_id_error_trace_csv(
        os.path.join(out_dir, "embedding_error_over_steps.csv"),
        error_trace_rows,
    )
    figure_paths = save_online_id_summary_figures(
        os.path.join(out_dir, "figures"),
        metric_rows=metric_rows,
        error_trace_rows=error_trace_rows,
    )
    markdown_path = write_online_id_summary_markdown(
        os.path.join(out_dir, "summary.md"),
        records=records,
        metric_rows=metric_rows,
        figure_paths=figure_paths,
    )
    payload["metrics_csv"] = metrics_csv_path
    payload["embedding_error_trace_csv"] = error_trace_csv_path
    payload["summary_markdown"] = markdown_path
    payload["figure_paths"] = figure_paths
    return payload


def _format_variant_value(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def load_reference_embedding_bank(
    *,
    checkpoint_path: str | None,
    system_bank: str,
    fallback_systems: tuple[SystemSpec, ...],
    target_device: str | torch.device = "cpu",
) -> tuple[tuple[SystemSpec, ...], torch.Tensor]:
    if checkpoint_path:
        bank_specs = resolve_system_bank(system_bank)
        payload = torch.load(checkpoint_path, map_location="cpu")
        saved_embeddings = dict(payload.get("system_embeddings", {}))
        embedding_metadata = dict(payload.get("embedding_metadata", {}))
        embedding_mode = str(payload.get("embedding_mode", "learned_system_id"))
        if saved_embeddings:
            inferred_d_embed = len(next(iter(saved_embeddings.values())))
        elif embedding_mode == "family_param" and embedding_metadata.get("embedding_dim") is not None:
            inferred_d_embed = int(embedding_metadata["embedding_dim"])
        else:
            inferred_d_embed = len(fallback_systems[0].embedding)
        if embedding_mode == "fixed":
            aligned_specs = truncate_embedding(bank_specs, d_embed=inferred_d_embed)
            embedding_map = resolve_system_embedding_map(
                systems=aligned_specs,
                embedding_mode="fixed",
                learned_table=None,
            )
            bank_specs = aligned_specs
        elif embedding_mode == "family_param" and embedding_metadata:
            embedding_map = resolve_system_embedding_map(
                systems=bank_specs,
                embedding_mode="family_param",
                learned_table=None,
                embedding_metadata=embedding_metadata,
            )
            for spec in bank_specs:
                if spec.name in saved_embeddings:
                    embedding_map[spec.name] = [float(x) for x in saved_embeddings[spec.name]]
        else:
            embedding_map = {}
            for spec in bank_specs:
                if spec.name in saved_embeddings:
                    embedding_map[spec.name] = [float(x) for x in saved_embeddings[spec.name]]
                else:
                    embedding_map[spec.name] = [float(x) for x in spec.embedding[:inferred_d_embed]]
        return bank_specs, system_embedding_tensor(embedding_map, bank_specs, target_device=target_device)

    fallback_map = {spec.name: [float(x) for x in spec.embedding] for spec in fallback_systems}
    return fallback_systems, system_embedding_tensor(
        fallback_map, fallback_systems, target_device=target_device
    )


def build_online_id_debug_scenarios(args) -> list[dict[str, object]]:
    baseline_mean_firing = float(getattr(args, "baseline_mean_firing", 1000.0))
    baseline_action_strength = float(getattr(args, "baseline_action_strength", 1.0))
    baseline_policies = [str(p) for p in getattr(args, "policies", ["active_short", "random", "no_policy"])]
    firing_policies = [str(p) for p in getattr(args, "firing_policies", baseline_policies)]
    action_policies = [str(p) for p in getattr(args, "action_policies", ["active_short"])]

    scenarios: list[dict[str, object]] = [
        {
            "hypothesis": "baseline",
            "variant": "baseline",
            "scenario_label": "baseline",
            "observation_mean_firing": baseline_mean_firing,
            "action_strength": baseline_action_strength,
            "policies": baseline_policies,
        }
    ]

    for mean_firing in [float(x) for x in getattr(args, "mean_firing_sweep", [])]:
        if math.isclose(mean_firing, baseline_mean_firing, rel_tol=0.0, abs_tol=1e-12):
            continue
        variant = f"mean_firing_{_format_variant_value(mean_firing)}"
        scenarios.append(
            {
                "hypothesis": "firing_rate",
                "variant": variant,
                "scenario_label": variant,
                "observation_mean_firing": mean_firing,
                "action_strength": baseline_action_strength,
                "policies": firing_policies,
            }
        )

    for action_strength in [float(x) for x in getattr(args, "action_strength_sweep", [])]:
        if math.isclose(action_strength, baseline_action_strength, rel_tol=0.0, abs_tol=1e-12):
            continue
        variant = f"action_strength_{_format_variant_value(action_strength)}"
        scenarios.append(
            {
                "hypothesis": "action_strength",
                "variant": variant,
                "scenario_label": variant,
                "observation_mean_firing": baseline_mean_firing,
                "action_strength": action_strength,
                "policies": action_policies,
            }
        )
    return scenarios


def build_online_id_debug_run_rows(records: list[dict]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        debug = dict(record.get("debug_diagnostics", {}))
        landscape = dict(record.get("embedding_bank_landscape", {}))
        post_probe = dict(record.get("post_probe_eval", {}))
        rows.append(
            {
                "hypothesis": str(record.get("hypothesis", "baseline")),
                "variant": str(record.get("variant", "baseline")),
                "scenario_label": str(record.get("scenario_label", record.get("variant", "baseline"))),
                "system": str(record["system"]),
                "family": str(record["family"]),
                "policy": str(record["policy"]),
                "seed": int(record["seed"]),
                "configured_mean_firing": float(record.get("observation_mean_firing", 0.0)),
                "configured_action_strength": float(
                    record.get("active_policy_cfg", {}).get("action_strength", 0.0)
                ),
                "rollout_mse": _finite_metric(post_probe.get("rollout_mse")),
                "final_state_mse": _finite_metric(post_probe.get("final_state_mse")),
                "trajectory_r2": _finite_metric(post_probe.get("trajectory_r2")),
                "final_error": float(record.get("final_error", 0.0)),
                "observed_rate_hz_mean": _finite_metric(debug.get("observed_rate_hz_mean")),
                "observed_zero_fraction_mean": _finite_metric(
                    debug.get("observed_zero_fraction_mean")
                ),
                "action_norm_mean": _finite_metric(debug.get("action_norm_mean")),
                "field_norm_mean": _finite_metric(debug.get("field_norm_mean")),
                "action_to_field_ratio_mean": _finite_metric(
                    debug.get("action_to_field_ratio_mean")
                ),
                "action_field_cosine_mean": _finite_metric(debug.get("action_field_cosine_mean")),
                "action_at_bound_mean": _finite_metric(debug.get("action_at_bound_mean")),
                "info_theta_trace_mean": _finite_metric(debug.get("info_theta_trace_mean")),
                "info_state_trace_mean": _finite_metric(debug.get("info_state_trace_mean")),
                "final_posterior_mahalanobis_sq": _finite_metric(
                    record.get("final_posterior_mahalanobis_sq")
                ),
                "final_posterior_cov_trace": _finite_metric(record.get("final_posterior_cov_trace")),
                "final_posterior_cov_logdet": _finite_metric(
                    record.get("final_posterior_cov_logdet")
                ),
                "embedding_bank_true_candidate_rank": landscape.get("true_candidate_rank"),
                "embedding_bank_top1_system": landscape.get("top1_system"),
                "embedding_bank_top1_family": landscape.get("top1_family"),
                "embedding_bank_top1_mse": _finite_metric(landscape.get("top1_mse")),
                "embedding_bank_top2_gap": _finite_metric(landscape.get("top2_gap")),
                "embedding_bank_top5_unique_families": landscape.get("top5_unique_families"),
                "embedding_bank_discrete_posterior_effective_support": _finite_metric(
                    landscape.get("discrete_posterior_effective_support")
                ),
                "embedding_bank_discrete_gaussian_mean_gap": _finite_metric(
                    landscape.get("discrete_gaussian_mean_gap")
                ),
                "embedding_bank_discrete_gaussian_cov_gap_fro": _finite_metric(
                    landscape.get("discrete_gaussian_cov_gap_fro")
                ),
                "session_dir": str(record.get("session_dir", "")),
                "record_path": str(record.get("record_path", "")),
            }
        )
    return rows


def aggregate_online_id_debug_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["hypothesis"]), str(row["variant"]), str(row["policy"]))
        groups.setdefault(key, []).append(row)

    agg_rows: list[dict[str, object]] = []
    for (hypothesis, variant, policy), subset in sorted(groups.items()):
        metric_names = [
            "rollout_mse",
            "final_state_mse",
            "trajectory_r2",
            "final_error",
            "observed_rate_hz_mean",
            "observed_zero_fraction_mean",
            "action_norm_mean",
            "field_norm_mean",
            "action_to_field_ratio_mean",
            "action_field_cosine_mean",
            "action_at_bound_mean",
            "info_theta_trace_mean",
            "info_state_trace_mean",
            "final_posterior_mahalanobis_sq",
            "final_posterior_cov_trace",
            "final_posterior_cov_logdet",
            "embedding_bank_top1_mse",
            "embedding_bank_top2_gap",
            "embedding_bank_discrete_posterior_effective_support",
            "embedding_bank_discrete_gaussian_mean_gap",
            "embedding_bank_discrete_gaussian_cov_gap_fro",
        ]
        row_out: dict[str, object] = {
            "hypothesis": hypothesis,
            "variant": variant,
            "policy": policy,
            "n_runs": len(subset),
            "n_finite_rollout": sum(1 for row in subset if row.get("rollout_mse") is not None),
            "configured_mean_firing": float(subset[0]["configured_mean_firing"]),
            "configured_action_strength": float(subset[0]["configured_action_strength"]),
            "mean_true_candidate_rank": None,
            "mean_top5_unique_families": None,
        }
        for metric_name in metric_names:
            values = [float(row[metric_name]) for row in subset if row.get(metric_name) is not None]
            mean_val, std_val = _mean_std(values)
            row_out[f"{metric_name}_mean"] = mean_val
            row_out[f"{metric_name}_std"] = std_val

        rank_values = [
            float(row["embedding_bank_true_candidate_rank"])
            for row in subset
            if row.get("embedding_bank_true_candidate_rank") is not None
        ]
        family_values = [
            float(row["embedding_bank_top5_unique_families"])
            for row in subset
            if row.get("embedding_bank_top5_unique_families") is not None
        ]
        rank_mean, _ = _mean_std(rank_values)
        family_mean, _ = _mean_std(family_values)
        row_out["mean_true_candidate_rank"] = rank_mean
        row_out["mean_top5_unique_families"] = family_mean
        agg_rows.append(row_out)
    return agg_rows


def _write_csv_rows(path: str, rows: list[dict[str, object]], fieldnames: list[str]) -> str:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_online_id_debug_csvs(
    out_dir: str,
    *,
    run_rows: list[dict[str, object]],
    aggregate_rows: list[dict[str, object]],
) -> dict[str, str]:
    run_fields = list(run_rows[0].keys()) if run_rows else []
    agg_fields = list(aggregate_rows[0].keys()) if aggregate_rows else []
    paths: dict[str, str] = {}
    if run_fields:
        paths["run_metrics_csv"] = _write_csv_rows(
            os.path.join(out_dir, "debug_run_metrics.csv"), run_rows, run_fields
        )
    if agg_fields:
        paths["aggregate_metrics_csv"] = _write_csv_rows(
            os.path.join(out_dir, "debug_aggregate_metrics.csv"),
            aggregate_rows,
            agg_fields,
        )
    return paths


def save_online_id_debug_figures(
    out_dir: str,
    *,
    run_rows: list[dict[str, object]],
    aggregate_rows: list[dict[str, object]],
) -> list[str]:
    ensure_dir(out_dir)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    figure_paths: list[str] = []

    firing_rows = [
        row
        for row in aggregate_rows
        if row["hypothesis"] in {"baseline", "firing_rate"}
        and row.get("configured_mean_firing") is not None
    ]
    if firing_rows:
        fig, ax = plt.subplots(figsize=(8.8, 4.8))
        for policy in sorted({str(row["policy"]) for row in firing_rows}):
            subset = [row for row in firing_rows if str(row["policy"]) == policy]
            subset.sort(key=lambda row: float(row["configured_mean_firing"]))
            xs = [float(row["configured_mean_firing"]) for row in subset]
            ys = [
                np.nan
                if row.get("rollout_mse_mean") is None
                else float(row["rollout_mse_mean"])
                for row in subset
            ]
            ax.plot(xs, ys, marker="o", label=policy)
        ax.set_xscale("log")
        ax.set_xlabel("Configured mean firing rate")
        ax.set_ylabel("Rollout MSE")
        ax.set_title("Online-ID sensitivity to observation firing rate")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        path = os.path.join(out_dir, "rollout_mse_vs_mean_firing.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figure_paths.append(path)

        fig, ax = plt.subplots(figsize=(8.8, 4.8))
        for policy in sorted({str(row["policy"]) for row in firing_rows}):
            subset = [row for row in firing_rows if str(row["policy"]) == policy]
            subset.sort(key=lambda row: float(row["configured_mean_firing"]))
            xs = [float(row["configured_mean_firing"]) for row in subset]
            ys = [
                np.nan
                if row.get("observed_rate_hz_mean_mean") is None
                else float(row["observed_rate_hz_mean_mean"])
                for row in subset
            ]
            ax.plot(xs, ys, marker="o", label=policy)
        ax.set_xscale("log")
        ax.set_xlabel("Configured mean firing rate")
        ax.set_ylabel("Observed mean firing rate (Hz)")
        ax.set_title("Observed firing rate under firing-rate sweep")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        path = os.path.join(out_dir, "observed_rate_vs_mean_firing.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figure_paths.append(path)

    action_rows = [
        row
        for row in aggregate_rows
        if row["hypothesis"] in {"baseline", "action_strength"}
        and str(row["policy"]) == "active_short"
    ]
    if action_rows:
        action_rows.sort(key=lambda row: float(row["configured_action_strength"]))
        xs = [float(row["configured_action_strength"]) for row in action_rows]

        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        ys = [
            np.nan if row.get("rollout_mse_mean") is None else float(row["rollout_mse_mean"])
            for row in action_rows
        ]
        ax.plot(xs, ys, marker="o", color="#0b6e4f")
        ax.set_xscale("log")
        ax.set_xlabel("Active action strength")
        ax.set_ylabel("Rollout MSE")
        ax.set_title("Active-short performance vs action strength")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = os.path.join(out_dir, "rollout_mse_vs_action_strength.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figure_paths.append(path)

        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        ys = [
            np.nan
            if row.get("action_to_field_ratio_mean_mean") is None
            else float(row["action_to_field_ratio_mean_mean"])
            for row in action_rows
        ]
        ax.plot(xs, ys, marker="o", color="#b74f2a")
        ax.set_xscale("log")
        ax.set_xlabel("Active action strength")
        ax.set_ylabel("Action / vector-field norm")
        ax.set_title("Action strength relative to native dynamics")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = os.path.join(out_dir, "action_to_field_ratio_vs_action_strength.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figure_paths.append(path)

    gaussian_rows = [
        row
        for row in run_rows
        if row.get("final_posterior_mahalanobis_sq") is not None and row.get("rollout_mse") is not None
    ]
    if gaussian_rows:
        fig, ax = plt.subplots(figsize=(8.8, 4.8))
        for policy in sorted({str(row["policy"]) for row in gaussian_rows}):
            subset = [row for row in gaussian_rows if str(row["policy"]) == policy]
            ax.scatter(
                [float(row["final_posterior_mahalanobis_sq"]) for row in subset],
                [float(row["rollout_mse"]) for row in subset],
                s=28,
                alpha=0.8,
                label=policy,
            )
        ax.set_xlabel("Final posterior Mahalanobis distance squared")
        ax.set_ylabel("Rollout MSE")
        ax.set_title("Posterior Gaussian fit vs online-ID performance")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        path = os.path.join(out_dir, "gaussian_fit_vs_rollout_mse.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figure_paths.append(path)

    landscape_rows = [
        row
        for row in run_rows
        if row.get("embedding_bank_true_candidate_rank") is not None
        and row.get("embedding_bank_discrete_gaussian_mean_gap") is not None
    ]
    if landscape_rows:
        fig, ax = plt.subplots(figsize=(8.8, 4.8))
        for policy in sorted({str(row["policy"]) for row in landscape_rows}):
            subset = [row for row in landscape_rows if str(row["policy"]) == policy]
            ax.scatter(
                [float(row["embedding_bank_true_candidate_rank"]) for row in subset],
                [float(row["embedding_bank_discrete_gaussian_mean_gap"]) for row in subset],
                s=28,
                alpha=0.8,
                label=policy,
            )
        ax.set_xlabel("True-system rank in discrete bank landscape")
        ax.set_ylabel("Discrete posterior / Gaussian mean gap")
        ax.set_title("Discrete bank posterior vs local Gaussian belief")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        path = os.path.join(out_dir, "discrete_bank_vs_gaussian_gap.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figure_paths.append(path)

    return figure_paths


def write_online_id_debug_summary_markdown(
    out_path: str,
    *,
    records: list[dict],
    aggregate_rows: list[dict[str, object]],
    figure_paths: list[str],
) -> str:
    ensure_dir(os.path.dirname(out_path))

    def _fmt(value: object) -> str:
        scalar = _finite_metric(value)
        return "NA" if scalar is None else f"{scalar:.4f}"

    lines: list[str] = []
    lines.append("# Cosyne Online-ID Debug Summary")
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append(f"- Records: {len(records)}")
    lines.append(f"- Hypotheses: {', '.join(sorted({str(r.get('hypothesis', 'baseline')) for r in records}))}")
    lines.append(f"- Systems: {', '.join(sorted({str(r['system']) for r in records}))}")
    lines.append("")

    firing_rows = [row for row in aggregate_rows if row["hypothesis"] in {"baseline", "firing_rate"}]
    if firing_rows:
        lines.append("## Firing-Rate Check")
        lines.append("")
        lines.append(
            "| variant | policy | mean_firing | rollout_mse | observed_rate_hz | info_theta | final_error |"
        )
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for row in sorted(firing_rows, key=lambda r: (float(r["configured_mean_firing"]), str(r["policy"]))):
            lines.append(
                f"| {row['variant']} | {row['policy']} | {float(row['configured_mean_firing']):.1f} | "
                f"{_fmt(row.get('rollout_mse_mean'))} | "
                f"{_fmt(row.get('observed_rate_hz_mean_mean'))} | "
                f"{_fmt(row.get('info_theta_trace_mean_mean'))} | "
                f"{_fmt(row.get('final_error_mean'))} |"
            )
        lines.append("")

    action_rows = [
        row for row in aggregate_rows if row["hypothesis"] in {"baseline", "action_strength"}
    ]
    if action_rows:
        lines.append("## Action-Strength Check")
        lines.append("")
        lines.append(
            "| variant | policy | action_strength | rollout_mse | action/field | at_bound | final_error |"
        )
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for row in sorted(action_rows, key=lambda r: (float(r["configured_action_strength"]), str(r["policy"]))):
            lines.append(
                f"| {row['variant']} | {row['policy']} | {float(row['configured_action_strength']):.3f} | "
                f"{_fmt(row.get('rollout_mse_mean'))} | "
                f"{_fmt(row.get('action_to_field_ratio_mean_mean'))} | "
                f"{_fmt(row.get('action_at_bound_mean_mean'))} | "
                f"{_fmt(row.get('final_error_mean'))} |"
            )
        lines.append("")

    gaussian_records = []
    for record in records:
        landscape = dict(record.get("embedding_bank_landscape", {}))
        gaussian_records.append(
            {
                "system": record["system"],
                "policy": record["policy"],
                "variant": record.get("variant", "baseline"),
                "mahal": record.get("final_posterior_mahalanobis_sq"),
                "rank": landscape.get("true_candidate_rank"),
                "gap": landscape.get("discrete_gaussian_mean_gap"),
                "top1": landscape.get("top1_system"),
            }
        )
    if gaussian_records:
        lines.append("## Gaussian-Approximation Check")
        lines.append("")
        lines.append("| system | policy | variant | mahal_sq | true_rank | mean_gap | top1_system |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: | --- |")
        for row in sorted(gaussian_records, key=lambda r: (str(r["system"]), str(r["policy"]), str(r["variant"]))):
            lines.append(
                f"| {row['system']} | {row['policy']} | {row['variant']} | "
                f"{_fmt(row['mahal'])} | "
                f"{row['rank'] if row['rank'] is not None else 'NA'} | "
                f"{_fmt(row['gap'])} | "
                f"{row['top1'] if row['top1'] is not None else 'NA'} |"
            )
        lines.append("")

    if figure_paths:
        lines.append("## Figures")
        lines.append("")
        for figure_path in figure_paths:
            lines.append(f"- `{os.path.basename(figure_path)}`")
    lines.append("")

    high_mahal = [
        r for r in records if _finite_metric(r.get("final_posterior_mahalanobis_sq")) is not None
        and float(r["final_posterior_mahalanobis_sq"]) > 5.99
    ]
    rank_miss = [
        r for r in records
        if r.get("embedding_bank_landscape", {}).get("true_candidate_rank") not in {None, 1}
    ]
    lines.append("## Headline Diagnostics")
    lines.append("")
    lines.append(
        f"- Final Gaussian posterior misses the true embedding at 95% ellipse level in {len(high_mahal)} / {len(records)} records."
    )
    lines.append(
        f"- Discrete bank landscape top-1 differs from the true system in {len(rank_miss)} / {len(records)} records."
    )

    text = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return out_path


def summarize_online_id_debug_results(records: list[dict], out_path: str) -> dict[str, object]:
    summary_by_variant = aggregate_online_id_debug_rows(build_online_id_debug_run_rows(records))
    payload = {
        "primary_metric": "rollout_mse",
        "records": records,
        "summary_by_variant": summary_by_variant,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    out_dir = os.path.dirname(out_path)
    run_rows = build_online_id_debug_run_rows(records)
    aggregate_rows = aggregate_online_id_debug_rows(run_rows)
    csv_paths = write_online_id_debug_csvs(
        out_dir,
        run_rows=run_rows,
        aggregate_rows=aggregate_rows,
    )
    figure_paths = save_online_id_debug_figures(
        os.path.join(out_dir, "figures"),
        run_rows=run_rows,
        aggregate_rows=aggregate_rows,
    )
    markdown_path = write_online_id_debug_summary_markdown(
        os.path.join(out_dir, "summary.md"),
        records=records,
        aggregate_rows=aggregate_rows,
        figure_paths=figure_paths,
    )
    payload["run_metrics_csv"] = csv_paths.get("run_metrics_csv")
    payload["aggregate_metrics_csv"] = csv_paths.get("aggregate_metrics_csv")
    payload["summary_markdown"] = markdown_path
    payload["figure_paths"] = figure_paths
    return payload


def default_results_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results", "cosyne"))


def resolve_online_id_config_path() -> str:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidates = (
        os.path.join(os.path.dirname(__file__), "conf", "config.yaml"),
        os.path.join(repo_root, "experiments", "active_embedding", "conf", "config.yaml"),
        os.path.join(repo_root, "experiments", "ciss", "conf", "config.yaml"),
    )
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Could not locate an experiment config for online identification. "
        f"Checked: {list(candidates)}"
    )


def configure_runtime_device(seed: int = 0) -> str:
    global device
    device = configure_runtime(seed=seed, device=device)
    return device


def resolve_system_bank(system_bank: str) -> tuple[SystemSpec, ...]:
    if system_bank == "mixed200":
        return MIXED200_SYSTEM_SPECS
    if system_bank in {"mixed80", "mixed40"}:
        return MIXED80_SYSTEM_SPECS
    if system_bank == "known_duffing40":
        return KNOWN_DUFFING40_SYSTEM_SPECS
    if system_bank == "legacy4":
        return LEGACY_SYSTEM_SPECS
    raise ValueError(f"Unknown system bank: {system_bank}")


def select_systems(
    bank: tuple[SystemSpec, ...],
    requested_systems: list[str] | tuple[str, ...] | None,
    system_bank: str,
) -> tuple[SystemSpec, ...]:
    if requested_systems:
        available = {spec.name for spec in bank}
        requested = set(requested_systems)
        missing = sorted(requested.difference(available))
        if missing:
            raise ValueError(f"Requested systems are not in {system_bank}: {missing}")
        selected = tuple(spec for spec in bank if spec.name in requested)
    else:
        selected = bank
    if len(selected) == 0:
        raise ValueError("No systems selected.")
    return selected


def validate_rollout_args(
    *,
    rollout_init_low: float,
    rollout_init_high: float,
    rollout_horizon: int,
    rollout_inits: int,
    rollout_dt: float,
) -> None:
    if rollout_init_low >= rollout_init_high:
        raise ValueError("rollout-init-low must be smaller than rollout-init-high")
    if rollout_horizon <= 0:
        raise ValueError("rollout-horizon must be >= 1")
    if rollout_inits <= 0:
        raise ValueError("rollout-inits must be >= 1")
    if rollout_dt <= 0:
        raise ValueError("rollout-dt must be > 0")


def build_active_policy_cfg(
    *,
    active_horizon: int,
    active_num_iterations: int,
    active_num_samples: int,
    active_num_elite: int,
    active_chunk: int,
    active_action_cost_weight: float,
    active_action_strength: float,
) -> ActivePolicyConfig:
    if active_horizon <= 0:
        raise ValueError("active-horizon must be >= 1")
    if active_num_iterations <= 0:
        raise ValueError("active-num-iterations must be >= 1")
    if active_num_elite <= 0:
        raise ValueError("active-num-elite must be >= 1")
    if active_num_samples < active_num_elite:
        raise ValueError("active-num-samples must be >= active-num-elite")
    if active_chunk <= 0:
        raise ValueError("active-chunk must be >= 1")
    if active_action_cost_weight < 0:
        raise ValueError("active-action-cost-weight must be >= 0")
    if active_action_strength <= 0:
        raise ValueError("active-action-strength must be > 0")
    return ActivePolicyConfig(
        horizon=active_horizon,
        num_iterations=active_num_iterations,
        num_samples=active_num_samples,
        num_elite=active_num_elite,
        chunk=min(active_chunk, active_horizon),
        action_cost_weight=active_action_cost_weight,
        action_strength=active_action_strength,
    )


def prepare_selected_systems(
    *,
    system_bank: str,
    systems: list[str] | tuple[str, ...] | None,
    embedding_mode: str,
    d_embed: int,
) -> tuple[SystemSpec, ...]:
    if system_bank == "known_duffing40" and embedding_mode == "fixed" and int(d_embed) != 2:
        raise ValueError(
            "known_duffing40 with fixed embeddings requires d_embed=2 because the embedding "
            "is the true Duffing parameter pair (a, b)."
        )
    bank = resolve_system_bank(system_bank)
    selected = select_systems(bank=bank, requested_systems=systems, system_bank=system_bank)
    if embedding_mode == "fixed":
        selected = truncate_embedding(selected, d_embed=d_embed)
    return selected


def canonical_vectorfield_system_names(system_bank: str) -> tuple[str, ...]:
    if system_bank not in CANONICAL_VECTORFIELD_SYSTEMS:
        raise ValueError(
            f"No canonical vectorfield representative mapping for system bank: {system_bank}"
        )
    return CANONICAL_VECTORFIELD_SYSTEMS[system_bank]


def resolve_results_dir(results_root: str | None, results_subdir: str | None) -> str:
    if not results_subdir:
        raise ValueError("results_subdir must be provided when resolve_results_dir is used")
    root = results_root or default_results_root()
    return ensure_dir(os.path.join(root, results_subdir))


def resolve_output_dir(
    *,
    results_root: str | None,
    results_subdir: str | None,
    output_dir: str | None,
    checkpoint: str | None,
) -> str:
    if output_dir:
        return ensure_dir(output_dir)
    if results_subdir:
        return resolve_results_dir(results_root, results_subdir)
    if checkpoint:
        return ensure_dir(os.path.dirname(os.path.abspath(checkpoint)))
    raise ValueError(
        "Either output_dir, results_subdir, or checkpoint must be provided to resolve the figure output directory"
    )


def maybe_load_or_train_bundle(args, selected: tuple[SystemSpec, ...]) -> ModelBundle:
    if getattr(args, "checkpoint", None):
        return load_model_bundle_checkpoint(args.checkpoint, selected)
    embedding_reference_systems = resolve_system_bank(getattr(args, "system_bank", "mixed80"))
    return train_meta_dynamics(
        selected,
        d_embed=args.d_embed,
        d_hidden_dynamics=args.d_hidden_dynamics,
        d_hidden_hypernet_dynamics=args.d_hidden_hypernet_dynamics,
        n_hidden=args.n_hidden,
        embedding_mode=args.embedding_mode,
        dynamics_scale=args.dynamics_scale,
        n_per_system=args.train_samples_per_system,
        batch_size=args.batch_size,
        n_epochs=args.train_epochs,
        embedding_reference_systems=embedding_reference_systems,
        geometry_reg_weight=float(getattr(args, "geometry_reg_weight", 0.05)),
        geometry_anchor_samples=int(getattr(args, "geometry_anchor_samples", 512)),
        geometry_neighbor_k=int(getattr(args, "geometry_neighbor_k", 4)),
        interpolation_aug_weight=float(getattr(args, "interpolation_aug_weight", 0.25)),
        interpolation_aug_samples=int(getattr(args, "interpolation_aug_samples", 128)),
        train_state_bounds=(
            float(getattr(args, "train_state_low", -3.0)),
            float(getattr(args, "train_state_high", 3.0)),
        ),
    )


def run_pretrain_eval_experiment(args) -> dict[str, object]:
    configure_runtime_device(seed=int(getattr(args, "seed", 0)))
    validate_rollout_args(
        rollout_init_low=args.rollout_init_low,
        rollout_init_high=args.rollout_init_high,
        rollout_horizon=args.rollout_horizon,
        rollout_inits=args.rollout_inits,
        rollout_dt=args.rollout_dt,
    )
    base_dir = resolve_results_dir(getattr(args, "results_root", None), args.results_subdir)
    selected = prepare_selected_systems(
        system_bank=args.system_bank,
        systems=args.systems,
        embedding_mode=args.embedding_mode,
        d_embed=args.d_embed,
    )
    bundle = maybe_load_or_train_bundle(args=args, selected=selected)

    verification = verify_parameter_bank(
        selected,
        out_dir=base_dir,
        dynamics_scale=args.dynamics_scale,
        horizon=max(250, args.rollout_horizon),
        dt=min(args.rollout_dt, 0.01),
    )
    if not verification["all_passed"]:
        failed = [row["name"] for row in verification["rows"] if not row["passed"]]
        raise RuntimeError(f"Parameter verification failed for: {failed}")
    payload = summarize_offline(
        bundle,
        selected,
        out_dir=base_dir,
        rollout_horizon=args.rollout_horizon,
        rollout_inits=args.rollout_inits,
        rollout_dt=args.rollout_dt,
        rollout_dt_sweep=tuple(args.rollout_dt_sweep),
        rollout_init_bounds=(args.rollout_init_low, args.rollout_init_high),
        dynamics_scale=args.dynamics_scale,
        system_bank=args.system_bank,
    )
    checkpoint_path = save_model_bundle_checkpoint(bundle, out_dir=base_dir)
    summary_markdown = getattr(
        args,
        "summary_markdown",
        os.path.join(os.path.dirname(__file__), "mixed_family_metadynamics_summary.md"),
    )
    markdown_path = write_pretrain_summary_markdown(
        out_path=summary_markdown,
        payload=payload,
        verification=verification,
        checkpoint_path=checkpoint_path,
    )
    mean_rollout_mse = float(
        np.mean([payload["rollout_eval"][spec.name]["rollout_mse"] for spec in selected])
    )
    return {
        "results_dir": base_dir,
        "system_bank": payload["system_bank"],
        "n_systems": len(selected),
        "embedding_mode": payload["embedding_mode"],
        "final_train_loss": payload["train_summary"]["final_train_loss"],
        "mean_rollout_mse": mean_rollout_mse,
        "family_rollout_eval": payload["family_rollout_eval"],
        "verification_passed": verification["n_passed"],
        "verification_total": verification["n_systems"],
        "embedding_cluster_figure": payload["embedding_cluster_figure"],
        "checkpoint_path": checkpoint_path,
        "summary_markdown": markdown_path,
    }


def run_vectorfield_figure_experiment(args) -> dict[str, object]:
    configure_runtime_device(seed=int(getattr(args, "seed", 0)))
    system_bank = getattr(args, "system_bank", "mixed80")
    requested_systems = getattr(args, "systems", None) or canonical_vectorfield_system_names(
        system_bank
    )
    base_dir = resolve_output_dir(
        results_root=getattr(args, "results_root", None),
        results_subdir=getattr(args, "results_subdir", None),
        output_dir=getattr(args, "output_dir", None),
        checkpoint=getattr(args, "checkpoint", None),
    )
    selected = prepare_selected_systems(
        system_bank=system_bank,
        systems=requested_systems,
        embedding_mode=args.embedding_mode,
        d_embed=args.d_embed,
    )
    bundle = maybe_load_or_train_bundle(args=args, selected=selected)
    embeddings = system_embedding_tensor(bundle.system_embeddings, selected, target_device=device)
    figure_path = os.path.join(
        base_dir, getattr(args, "figure_filename", "vectorfield_family_comparison_official.png")
    )
    payload = save_family_vectorfield_comparison_figure(
        meta_dynamics=bundle.meta_dynamics,
        systems=selected,
        system_embeddings=embeddings,
        out_path=figure_path,
        dynamics_scale=args.dynamics_scale,
        grid_n=int(getattr(args, "grid_n", CANONICAL_VECTORFIELD_GRID_N)),
        grid_limits=(
            float(getattr(args, "grid_min", CANONICAL_VECTORFIELD_GRID_RANGE[0])),
            float(getattr(args, "grid_max", CANONICAL_VECTORFIELD_GRID_RANGE[1])),
        ),
        figure_layout=str(getattr(args, "figure_layout", CANONICAL_VECTORFIELD_LAYOUT)),
    )
    metadata_path = os.path.join(
        base_dir, getattr(args, "metadata_filename", "vectorfield_family_comparison_official.json")
    )
    metadata_payload = {
        **payload,
        "system_bank": system_bank,
        "selected_systems": [spec.name for spec in selected],
        "checkpoint": getattr(args, "checkpoint", None),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2)
    return {
        "results_dir": base_dir,
        "figure_path": figure_path,
        "metadata_path": metadata_path,
        "families": payload["representatives"],
        "selected_systems": [spec.name for spec in selected],
        "grid_n": payload["grid_n"],
        "grid_limits": payload["grid_limits"],
        "layout": payload["layout"],
    }


def run_embedding_cluster_figure_experiment(args) -> dict[str, object]:
    configure_runtime_device(seed=int(getattr(args, "seed", 0)))
    base_dir = resolve_results_dir(getattr(args, "results_root", None), args.results_subdir)
    selected = prepare_selected_systems(
        system_bank=args.system_bank,
        systems=args.systems,
        embedding_mode=args.embedding_mode,
        d_embed=args.d_embed,
    )
    bundle = maybe_load_or_train_bundle(args=args, selected=selected)
    embeddings = system_embedding_tensor(bundle.system_embeddings, selected, target_device=device)
    figure_path = os.path.join(
        base_dir, getattr(args, "figure_filename", "embedding_family_clusters.png")
    )
    payload = save_embedding_cluster_figure(
        systems=selected,
        system_embeddings=embeddings,
        out_path=figure_path,
    )
    metadata_path = os.path.join(
        base_dir, getattr(args, "metadata_filename", "embedding_family_clusters.json")
    )
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "figure": payload,
                "n_systems": len(selected),
                "system_bank": args.system_bank,
            },
            f,
            indent=2,
        )
    return {
        "results_dir": base_dir,
        "figure_path": figure_path,
        "metadata_path": metadata_path,
        "projection": payload["projection"],
    }


def run_online_identification_experiment(args, *, print_progress: bool = True) -> dict[str, object]:
    configure_runtime_device(seed=int(getattr(args, "seed", 0)))
    validate_rollout_args(
        rollout_init_low=args.rollout_init_low,
        rollout_init_high=args.rollout_init_high,
        rollout_horizon=args.rollout_horizon,
        rollout_inits=args.rollout_inits,
        rollout_dt=args.rollout_dt,
    )
    active_cfg = build_active_policy_cfg(
        active_horizon=args.active_horizon,
        active_num_iterations=args.active_num_iterations,
        active_num_samples=args.active_num_samples,
        active_num_elite=args.active_num_elite,
        active_chunk=args.active_chunk,
        active_action_cost_weight=args.active_action_cost_weight,
        active_action_strength=args.active_action_strength,
    )
    base_dir = resolve_results_dir(getattr(args, "results_root", None), args.results_subdir)
    checkpoint_path = getattr(args, "checkpoint", None)
    if checkpoint_path is not None:
        checkpoint_path = os.path.abspath(str(checkpoint_path))
    resume_existing = bool(getattr(args, "resume", False))
    selected = prepare_selected_systems(
        system_bank=args.system_bank,
        systems=args.systems,
        embedding_mode=args.embedding_mode,
        d_embed=args.d_embed,
    )
    bundle = maybe_load_or_train_bundle(args=args, selected=selected)
    records: list[dict] = []
    resolved_embeddings = system_embedding_tensor(
        bundle.system_embeddings, selected, target_device="cpu"
    )
    for system_idx, spec in enumerate(selected):
        for policy in args.policies:
            for rep in range(args.repeats):
                run_dir = os.path.join(base_dir, spec.name, policy, f"seed_{rep}")
                if resume_existing:
                    existing_record = load_existing_online_id_record(
                        run_dir,
                        system=spec.name,
                        policy=str(policy),
                        seed=rep,
                    )
                    if existing_record is not None:
                        records.append(existing_record)
                        if print_progress:
                            post_probe_eval = dict(existing_record.get("post_probe_eval", {}))
                            rollout_mse = float(post_probe_eval.get("rollout_mse", float("nan")))
                            final_state_mse = float(
                                post_probe_eval.get("final_state_mse", float("nan"))
                            )
                            final_error = float(existing_record.get("final_error", float("nan")))
                            print(
                                f"[{spec.name}][{policy}][seed={rep}] "
                                f"resumed rollout_mse={rollout_mse:.4f} "
                                f"final_state_mse={final_state_mse:.4f} "
                                f"final_error={final_error:.4f}"
                            )
                        continue
                record = run_identification(
                    spec=spec,
                    meta_dynamics=bundle.meta_dynamics,
                    system_embedding=resolved_embeddings[system_idx],
                    embedding_mode=bundle.embedding_mode,
                    policy_name=policy,
                    results_dir=run_dir,
                    total_steps=args.total_steps,
                    seed=100 * system_idx + rep,
                    active_cfg=active_cfg,
                    eval_rollout_horizon=args.rollout_horizon,
                    eval_rollout_dt=args.rollout_dt,
                    eval_rollout_count=args.rollout_inits,
                    dynamics_scale=args.dynamics_scale,
                    save_acq_map=bool(getattr(args, "save_acq_map", False)),
                    acq_map_interval=int(getattr(args, "acq_map_interval", 5)),
                    acq_map_grid=int(getattr(args, "acq_map_grid", 61)),
                    acq_map_lim=float(getattr(args, "acq_map_lim", 3.0)),
                    observation_mean_firing=float(
                        getattr(args, "observation_mean_firing", 1000.0)
                    ),
                    q_theta=float(getattr(args, "q_theta", 1e-4)),
                    k_theta=int(getattr(args, "k_theta", 10)),
                    q_theta_meas_coeff=float(getattr(args, "q_theta_meas_coeff", 0.0)),
                    q_theta_max_scale=float(getattr(args, "q_theta_max_scale", 10.0)),
                    state_init_uncertainty=float(
                        getattr(args, "state_init_uncertainty", 1.0)
                    ),
                    record_metadata={
                        "checkpoint": checkpoint_path,
                        "system_bank": str(args.system_bank),
                        "dynamics_scale": float(args.dynamics_scale),
                    },
                )
                records.append(record)
                if print_progress:
                    print(
                        f"[{spec.name}][{policy}][seed={rep}] "
                        f"rollout_mse={record['post_probe_eval']['rollout_mse']:.4f} "
                        f"final_state_mse={record['post_probe_eval']['final_state_mse']:.4f} "
                        f"final_error={record['final_error']:.4f}"
                    )

    summary_path = os.path.join(base_dir, "summary.json")
    payload = summarize_identification_results(records, summary_path)
    return {
        "results_dir": base_dir,
        "summary_path": summary_path,
        "summary": payload["summary"],
        "n_records": len(records),
        "embedding_modes": payload["embedding_modes"],
    }


def run_online_identification_debug_experiment(
    args,
    *,
    print_progress: bool = True,
) -> dict[str, object]:
    configure_runtime_device(seed=int(getattr(args, "seed", 0)))
    validate_rollout_args(
        rollout_init_low=args.rollout_init_low,
        rollout_init_high=args.rollout_init_high,
        rollout_horizon=args.rollout_horizon,
        rollout_inits=args.rollout_inits,
        rollout_dt=args.rollout_dt,
    )
    base_dir = resolve_results_dir(getattr(args, "results_root", None), args.results_subdir)
    checkpoint_path = getattr(args, "checkpoint", None)
    if checkpoint_path is not None:
        checkpoint_path = os.path.abspath(str(checkpoint_path))

    selected = prepare_selected_systems(
        system_bank=args.system_bank,
        systems=args.systems,
        embedding_mode=args.embedding_mode,
        d_embed=args.d_embed,
    )
    bundle = maybe_load_or_train_bundle(args=args, selected=selected)
    resolved_embeddings = system_embedding_tensor(
        bundle.system_embeddings, selected, target_device="cpu"
    )
    reference_bank_specs, reference_bank_embeddings = load_reference_embedding_bank(
        checkpoint_path=checkpoint_path,
        system_bank=str(args.system_bank),
        fallback_systems=selected,
        target_device="cpu",
    )
    scenarios = build_online_id_debug_scenarios(args)
    records: list[dict] = []

    for scenario_idx, scenario in enumerate(scenarios):
        active_cfg = build_active_policy_cfg(
            active_horizon=args.active_horizon,
            active_num_iterations=args.active_num_iterations,
            active_num_samples=args.active_num_samples,
            active_num_elite=args.active_num_elite,
            active_chunk=args.active_chunk,
            active_action_cost_weight=args.active_action_cost_weight,
            active_action_strength=float(scenario["action_strength"]),
        )
        for system_idx, spec in enumerate(selected):
            for policy in scenario["policies"]:
                for rep in range(args.repeats):
                    run_dir = os.path.join(
                        base_dir,
                        str(scenario["scenario_label"]),
                        spec.name,
                        str(policy),
                        f"seed_{rep}",
                    )
                    seed = 10_000 * scenario_idx + 100 * system_idx + rep
                    record = run_identification(
                        spec=spec,
                        meta_dynamics=bundle.meta_dynamics,
                        system_embedding=resolved_embeddings[system_idx],
                        embedding_mode=bundle.embedding_mode,
                        policy_name=str(policy),
                        results_dir=run_dir,
                        total_steps=args.total_steps,
                        seed=seed,
                        active_cfg=active_cfg,
                        eval_rollout_horizon=args.rollout_horizon,
                        eval_rollout_dt=args.rollout_dt,
                        eval_rollout_count=args.rollout_inits,
                        dynamics_scale=args.dynamics_scale,
                        save_acq_map=bool(getattr(args, "save_acq_map", False)),
                        acq_map_interval=int(getattr(args, "acq_map_interval", 5)),
                        acq_map_grid=int(getattr(args, "acq_map_grid", 61)),
                        acq_map_lim=float(getattr(args, "acq_map_lim", 3.0)),
                        observation_mean_firing=float(scenario["observation_mean_firing"]),
                        q_theta=float(getattr(args, "q_theta", 1e-4)),
                        k_theta=int(getattr(args, "k_theta", 10)),
                        q_theta_meas_coeff=float(getattr(args, "q_theta_meas_coeff", 0.0)),
                        q_theta_max_scale=float(getattr(args, "q_theta_max_scale", 10.0)),
                        state_init_uncertainty=float(
                            getattr(args, "state_init_uncertainty", 1.0)
                        ),
                        reference_bank_specs=reference_bank_specs,
                        reference_bank_embeddings=reference_bank_embeddings,
                        bank_landscape_max_steps=int(
                            getattr(args, "bank_landscape_max_steps", 400)
                        ),
                        record_metadata={
                            "checkpoint": checkpoint_path,
                            "system_bank": str(args.system_bank),
                            "dynamics_scale": float(args.dynamics_scale),
                            "hypothesis": str(scenario["hypothesis"]),
                            "variant": str(scenario["variant"]),
                            "scenario_label": str(scenario["scenario_label"]),
                            "scenario_index": int(scenario_idx),
                            "configured_mean_firing": float(scenario["observation_mean_firing"]),
                            "configured_action_strength": float(scenario["action_strength"]),
                        },
                    )
                    records.append(record)
                    if print_progress:
                        print(
                            f"[{scenario['scenario_label']}][{spec.name}][{policy}][seed={rep}] "
                            f"rollout_mse={record['post_probe_eval']['rollout_mse']:.4f} "
                            f"final_error={record['final_error']:.4f} "
                            f"mahal={record['final_posterior_mahalanobis_sq']:.4f}"
                        )

    summary_path = os.path.join(base_dir, "summary.json")
    payload = summarize_online_id_debug_results(records, summary_path)
    return {
        "results_dir": base_dir,
        "summary_path": summary_path,
        "n_records": len(records),
        "n_scenarios": len(scenarios),
        "figure_paths": payload.get("figure_paths", []),
    }
