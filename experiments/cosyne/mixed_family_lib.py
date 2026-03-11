from __future__ import annotations

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


def build_mixed80_system_specs() -> tuple[SystemSpec, ...]:
    family_param_bank: list[tuple[str, str, list[tuple[float, float]]]] = [
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

    specs: list[SystemSpec] = []
    for family, note, params in family_param_bank:
        if len(params) != 20:
            raise ValueError(f"Family {family} must define exactly 20 parameter sets.")
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
    if len(specs) != 80:
        raise ValueError(f"Expected 80 systems, got {len(specs)}.")
    return tuple(specs)


MIXED80_SYSTEM_SPECS: tuple[SystemSpec, ...] = build_mixed80_system_specs()
BASE_SYSTEM_SPECS: tuple[SystemSpec, ...] = MIXED80_SYSTEM_SPECS


@dataclass
class ModelBundle:
    meta_dynamics: "MetaDynamics"
    cfg: dict
    train_summary: dict
    embedding_mode: str
    system_embeddings: dict[str, list[float]]


@dataclass(frozen=True)
class ActivePolicyConfig:
    horizon: int = 3
    num_iterations: int = 2
    num_samples: int = 12
    num_elite: int = 4
    chunk: int = 2
    action_cost_weight: float = 0.01
    action_strength: float = 0.3


class MixedSystemDataset(Dataset):
    def __init__(
        self,
        systems: tuple[SystemSpec, ...],
        n_per_system: int,
        z_sampler: Callable,
        d_embed: int,
        embedding_mode: Literal["fixed", "learned_system_id"] = "fixed",
        dynamics_scale: float = 10.0,
    ):
        if d_embed <= 0:
            raise ValueError("d_embed must be >= 1")
        if embedding_mode not in {"fixed", "learned_system_id"}:
            raise ValueError(f"Unsupported embedding_mode: {embedding_mode}")
        self.records: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = []
        for spec_idx, spec in enumerate(systems):
            z = z_sampler(n_per_system).float()
            if embedding_mode == "fixed":
                if len(spec.embedding) != d_embed:
                    raise ValueError(
                        f"System {spec.name} has embedding length {len(spec.embedding)} but d_embed={d_embed}."
                    )
                base_e = torch.tensor(spec.embedding, dtype=torch.float32)
            else:
                # Pure non-parametric mode: no hand-coded coordinates are used as model input.
                base_e = torch.zeros(d_embed, dtype=torch.float32)
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

    def __call__(self, x, e=None):
        if e is None:
            if self.e is None or self.out is None:
                raise ValueError("Embedding not set")
            out = self.out
        else:
            out, _ = self.hypernet(e)
        if x.ndim > out.ndim:
            expand_shape = list(out.shape)
            while len(expand_shape) < x.ndim:
                expand_shape.insert(1, 1)
            out = out.reshape(*expand_shape)
            out = out.expand(*x.shape[:-1], out.shape[-1])
        return self.mean_dynamics(x, out) * self.output_scale


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
        self.embedding_vector = embedding_vector.detach().to(self.device, dtype=torch.float32).reshape(-1)
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
def true_dynamics_from_spec(spec: SystemSpec, z: torch.Tensor, dynamics_scale: float = 10.0) -> torch.Tensor:
    x = z[..., 0]
    y = z[..., 1]
    p0, p1 = spec.params
    if spec.family in {"duffing_single", "duffing_bistable"}:
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


def rollout_meta(meta_dynamics: MetaDynamics, e: torch.Tensor, z0: torch.Tensor, horizon: int, dt: float) -> torch.Tensor:
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
        z = z + dt * (true_dynamics_from_spec(spec, z, dynamics_scale=dynamics_scale) + actions[:, t, :])
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


def resolve_system_embedding_map(
    systems: tuple[SystemSpec, ...],
    embedding_mode: Literal["fixed", "learned_system_id"],
    learned_table: nn.Embedding | None = None,
) -> dict[str, list[float]]:
    if embedding_mode == "fixed":
        return {spec.name: [float(x) for x in spec.embedding] for spec in systems}
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


def build_training_cfg(d_embed: int, d_hidden_dynamics: int, d_hidden_hypernet_dynamics: int, n_hidden: int):
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


def train_meta_dynamics(
    systems: tuple[SystemSpec, ...],
    d_embed: int,
    d_hidden_dynamics: int,
    d_hidden_hypernet_dynamics: int,
    n_hidden: int,
    embedding_mode: Literal["fixed", "learned_system_id"] = "fixed",
    dynamics_scale: float = 10.0,
    n_per_system: int = 5000,
    batch_size: int = 512,
    n_epochs: int = 80,
) -> ModelBundle:
    global device
    z_sampler = make_uniform_sampler(-3.0, 3.0, 2)
    ds = MixedSystemDataset(
        systems,
        n_per_system=n_per_system,
        z_sampler=z_sampler,
        d_embed=d_embed,
        embedding_mode=embedding_mode,
        dynamics_scale=dynamics_scale,
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    cfg = build_training_cfg(
        d_embed=d_embed,
        d_hidden_dynamics=d_hidden_dynamics,
        d_hidden_hypernet_dynamics=d_hidden_hypernet_dynamics,
        n_hidden=n_hidden,
    )
    cfg["dynamics_scale"] = dynamics_scale
    cfg["train_samples_per_system"] = int(n_per_system)
    cfg["train_epochs"] = int(n_epochs)
    cfg["batch_size"] = int(batch_size)
    hypernet = build_hypernetwork(cfg, device)
    mean_dynamics = metadyn.HyperMlpDynamics(
        d_latent=cfg["d_latent"],
        d_hidden=cfg["d_hidden_dynamics"],
        n_hidden=cfg["n_hidden"],
        update_input=cfg["update_input"],
        update_output=cfg["update_output"],
        update_hidden=cfg["update_hidden"],
        du=0,
        device=device,
        d_context=cfg["d_context"],
    ).to(device)
    learned_table: nn.Embedding | None = None
    if embedding_mode == "learned_system_id":
        learned_table = nn.Embedding(len(systems), d_embed, device=device)
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
    per_system_last = {spec.name: None for spec in systems}
    for _ in tqdm(range(n_epochs), desc="meta-train"):
        total_loss = 0.0
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
            loss = F.mse_loss(pred, fx)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                params,
                5.0,
            )
            opt.step()
            total_loss += loss.item() * z.shape[0]
            total_n += z.shape[0]
            with torch.no_grad():
                batch_err = ((pred - fx) ** 2).mean(dim=-1).detach().cpu().numpy()
                spec_idx_np = spec_idx.cpu().numpy()
                for local_i, system_i in enumerate(spec_idx_np):
                    per_system_acc[systems[int(system_i)].name].append(float(batch_err[local_i]))
        epoch_losses.append(total_loss / max(total_n, 1))
        per_system_last = {
            name: float(np.mean(vals)) if len(vals) > 0 else None for name, vals in per_system_acc.items()
        }

    return ModelBundle(
        meta_dynamics=MetaDynamics(hypernet, mean_dynamics, output_scale=dynamics_scale),
        cfg=cfg,
        train_summary={
            "epoch_losses": epoch_losses,
            "final_train_loss": epoch_losses[-1] if epoch_losses else None,
            "final_per_system_train_mse": per_system_last,
        },
        embedding_mode=embedding_mode,
        system_embeddings=resolve_system_embedding_map(
            systems=systems,
            embedding_mode=embedding_mode,
            learned_table=learned_table,
        ),
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

    axis_names = ("Embedding dim 1", "Embedding dim 2") if projection == "native_2d" else ("PC1", "PC2")
    ax.set_title("Learned Non-Parametric System-ID Embeddings")
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
    grid_n: int = 25,
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
        family_specs = [systems[i] for i in family_indices]
        family_specs_sorted = sorted(family_specs, key=lambda s: s.name)
        representative = family_specs_sorted[len(family_specs_sorted) // 2]
        representative_specs.append(representative)
        representative_indices.append(next(i for i, spec in enumerate(systems) if spec.name == representative.name))

    x = torch.linspace(-3.0, 3.0, grid_n, device=device)
    y = torch.linspace(-3.0, 3.0, grid_n, device=device)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    z = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()

    fig, axes = plt.subplots(len(families), 2, figsize=(8.8, 2.45 * len(families)), sharex=True, sharey=True)
    if len(families) == 1:
        axes = np.asarray([axes])

    metadata = []
    for row_idx, (family, spec, spec_idx) in enumerate(zip(families, representative_specs, representative_indices)):
        e = system_embeddings[spec_idx].reshape(1, -1).repeat(z.shape[0], 1)
        true_fx = true_dynamics_from_spec(spec, z, dynamics_scale=dynamics_scale).detach().cpu().numpy()
        pred_fx = meta_dynamics(z, e=e).detach().cpu().numpy()
        comps = [
            (axes[row_idx, 0], true_fx, 'True'),
            (axes[row_idx, 1], pred_fx, 'Reconstructed'),
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
                color=speed,
                cmap='viridis',
                density=0.9,
                linewidth=1.0,
                arrowsize=0.8,
            )
            ax.set_aspect('equal')
            ax.grid(alpha=0.15)
            ax.set_xlim(-3.0, 3.0)
            ax.set_ylim(-3.0, 3.0)
            ax.set_title(f"{family} — {label}", fontsize=10)
        axes[row_idx, 0].set_ylabel('x2')
        metadata.append({'family': family, 'system': spec.name, 'params': list(spec.params)})

    for ax in axes[-1, :]:
        ax.set_xlabel('x1')
    fig.suptitle('Representative family vector fields: true vs reconstructed', fontsize=12, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    return {'path': out_path, 'representatives': metadata, 'grid_n': grid_n}


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
            [-2.0, -2.0], [-2.0, 0.0], [-2.0, 2.0],
            [0.0, -2.0], [0.0, -0.75], [0.0, 0.75], [0.0, 2.0],
            [2.0, -2.0], [2.0, 0.0], [2.0, 2.0],
            [-1.2, 1.2], [1.2, -1.2], [1.5, 0.5], [-1.5, -0.5],
        ],
        dtype=torch.float32,
    )
    rows: list[dict] = []
    for spec in systems:
        traj = rollout_true(spec, init_grid, horizon=horizon, dt=dt, dynamics_scale=dynamics_scale).cpu()
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
        angular_velocity = (traj[:, :-1, 0] * (traj[:, 1:, 1] - traj[:, :-1, 1]) - traj[:, :-1, 1] * (traj[:, 1:, 0] - traj[:, :-1, 0])) / dt
        median_angular_velocity = float(torch.median(angular_velocity).item())
        rotation_direction = 'ccw' if median_angular_velocity >= 0.0 else 'cw'
        sign_diversity = int(torch.unique(torch.sign(final_xy[:, 0])).numel())

        family_ok = False
        family_reason = ''
        speed_cap = 180.0
        p95_cap = 80.0
        if spec.family == 'duffing_single':
            family_ok = mean_final_radius < 0.8 and std_final_radius < 0.55 and max_speed < 80.0
            family_reason = 'single-attractor convergence toward origin with modest transient speed'
        elif spec.family == 'duffing_bistable':
            family_ok = mean_final_radius > 0.7 and sign_diversity >= 2 and std_final_radius < 0.8 and max_speed < 40.0
            family_reason = 'bistable settling into separated wells without steep well-crossing transients'
        elif spec.family == 'van_der_pol':
            speed_cap = 130.0
            p95_cap = 82.0
            family_ok = 0.8 < mean_final_radius < 3.2 and tail_radius_std < 0.80 and max_speed < speed_cap and p95_speed < p95_cap
            family_reason = 'stable oscillatory limit cycle with controlled relaxation speed'
        elif spec.family == 'double_limit_cycle':
            speed_cap = 120.0
            p95_cap = 55.0
            family_ok = 0.5 < mean_final_radius < 4.0 and 0.12 < std_final_radius < 1.6 and max_speed < speed_cap and p95_speed < p95_cap
            family_reason = 'bounded multi-ring radial dynamics with controlled angular speed and bidirectional rotation support'
        generic_ok = finite_ok and max_radius < 8.0 and max_speed < speed_cap and p95_speed < p95_cap and mean_speed > 0.05
        passed = bool(generic_ok and family_ok)
        rows.append({
            'name': spec.name,
            'family': spec.family,
            'param_0': float(spec.params[0]),
            'param_1': float(spec.params[1]),
            'finite_ok': finite_ok,
            'max_radius': max_radius,
            'mean_final_radius': mean_final_radius,
            'std_final_radius': std_final_radius,
            'tail_radius_std': tail_radius_std,
            'mean_speed': mean_speed,
            'p95_speed': p95_speed,
            'max_speed': max_speed,
            'median_angular_velocity': median_angular_velocity,
            'rotation_direction': rotation_direction,
            'speed_cap': speed_cap,
            'p95_cap': p95_cap,
            'sign_diversity': sign_diversity,
            'generic_ok': generic_ok,
            'family_ok': family_ok,
            'passed': passed,
            'check': family_reason,
        })

    csv_path = os.path.join(out_dir, 'parameter_bank_verification.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    family_summary = {}
    for family in sorted({r['family'] for r in rows}):
        family_rows = [r for r in rows if r['family'] == family]
        family_summary[family] = {
            'n_systems': len(family_rows),
            'n_passed': int(sum(1 for r in family_rows if r['passed'])),
            'max_radius_max': float(max(r['max_radius'] for r in family_rows)),
            'max_final_radius_max': float(max(r['mean_final_radius'] for r in family_rows)),
            'mean_speed_mean': float(np.mean([r['mean_speed'] for r in family_rows])),
            'p95_speed_max': float(max(r['p95_speed'] for r in family_rows)),
            'max_speed_max': float(max(r['max_speed'] for r in family_rows)),
            'mean_final_radius_mean': float(np.mean([r['mean_final_radius'] for r in family_rows])),
            'std_final_radius_mean': float(np.mean([r['std_final_radius'] for r in family_rows])),
            'rotation_directions': sorted({str(r['rotation_direction']) for r in family_rows}),
        }
    payload = {
        'verification_horizon': horizon,
        'verification_dt': dt,
        'dynamics_scale': dynamics_scale,
        'n_systems': len(rows),
        'n_passed': int(sum(1 for r in rows if r['passed'])),
        'all_passed': bool(all(r['passed'] for r in rows)),
        'family_summary': family_summary,
        'csv_path': csv_path,
        'rows': rows,
    }
    json_path = os.path.join(out_dir, 'parameter_bank_verification.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    return payload


def save_model_bundle_checkpoint(bundle: ModelBundle, out_dir: str) -> str:
    ckpt_path = os.path.join(out_dir, 'meta_dynamics_checkpoint.pt')
    torch.save(
        {
            'cfg': bundle.cfg,
            'train_summary': bundle.train_summary,
            'embedding_mode': bundle.embedding_mode,
            'system_embeddings': bundle.system_embeddings,
            'hypernet_state_dict': bundle.meta_dynamics.hypernet.state_dict(),
            'mean_dynamics_state_dict': bundle.meta_dynamics.mean_dynamics.state_dict(),
            'output_scale': bundle.meta_dynamics.output_scale,
        },
        ckpt_path,
    )
    return ckpt_path


def load_model_bundle_checkpoint(ckpt_path: str, systems: tuple[SystemSpec, ...]) -> ModelBundle:
    payload = torch.load(ckpt_path, map_location=device)
    cfg = dict(payload.get('cfg', {}))
    embedding_mode = str(payload['embedding_mode'])
    inferred_d_embed = len(next(iter(payload['system_embeddings'].values())))
    cfg.setdefault('d_embed', inferred_d_embed)
    cfg.setdefault('d_hidden_dynamics', 64)
    cfg.setdefault('d_hidden_hypernet_dynamics', 16)
    cfg.setdefault('n_hidden', 2)
    cfg.setdefault('d_context', max(int(cfg['d_hidden_hypernet_dynamics']), 16))
    cfg.setdefault('d_latent', 2)
    cfg.setdefault('update_input', True)
    cfg.setdefault('update_output', True)
    cfg.setdefault('update_hidden', True)
    hypernet = build_hypernetwork(cfg, device)
    mean_dynamics = metadyn.HyperMlpDynamics(
        d_latent=cfg['d_latent'],
        d_hidden=cfg['d_hidden_dynamics'],
        n_hidden=cfg['n_hidden'],
        update_input=cfg['update_input'],
        update_output=cfg['update_output'],
        update_hidden=cfg['update_hidden'],
        du=0,
        device=device,
        d_context=cfg['d_context'],
    ).to(device)
    hypernet.load_state_dict(payload['hypernet_state_dict'])
    mean_dynamics.load_state_dict(payload['mean_dynamics_state_dict'])
    hypernet.eval()
    mean_dynamics.eval()
    meta_dynamics = MetaDynamics(
        hypernet=hypernet,
        mean_dynamics=mean_dynamics,
        output_scale=float(payload.get('output_scale', cfg.get('dynamics_scale', 10.0))),
    )
    if embedding_mode == 'fixed':
        aligned_systems = truncate_embedding(systems, d_embed=inferred_d_embed)
        system_embeddings = resolve_system_embedding_map(
            systems=aligned_systems,
            embedding_mode=embedding_mode,
            learned_table=None,
        )
    else:
        saved_embeddings = dict(payload.get('system_embeddings', {}))
        system_embeddings = {spec.name: [float(x) for x in saved_embeddings[spec.name]] for spec in systems}
    return ModelBundle(
        meta_dynamics=meta_dynamics,
        cfg=cfg,
        train_summary=dict(payload.get('train_summary', {})),
        embedding_mode=embedding_mode,
        system_embeddings=system_embeddings,
    )


def write_pretrain_summary_markdown(
    *,
    out_path: str,
    payload: dict,
    verification: dict,
    checkpoint_path: str,
) -> str:
    family_lines = []
    for family, stats in payload['family_rollout_eval'].items():
        family_lines.append(
            f"- {family}: mean rollout MSE {stats['mean_rollout_mse']:.4f}, mean final-state MSE {stats['mean_final_state_mse']:.4f}"
        )
    verification_lines = []
    for family, stats in verification['family_summary'].items():
        rotation_text = ''
        if stats.get('rotation_directions'):
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
    with open(out_path, 'w', encoding='utf-8') as f:
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
        dt_key: summarize_eval_by_family(systems, per_system_eval) for dt_key, per_system_eval in rollout_eval_by_dt.items()
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
def build_observation_model(dy: int, dt: float, noise_scale: float):
    obs_model = actdyn.environment.observation.LogLinearObservation(
        d_obs=dy,
        d_latent=2,
        R=noise_scale,
        noise_type="poisson",
        dt=dt,
        device=device,
    )
    C = obs_model.network[0].weight.detach()
    C[:, 0] = torch.abs(C[:, 0])
    C[:, 1] = C[:, 1] * 2.0
    mean_firing = 30.0
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
    summary["block_update_rate"] = float(
        summary["n_block_updates"] / max(summary["n_steps"], 1.0)
    )
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
    actions = action_low + (action_high - action_low) * torch.rand((n_rollouts, horizon, 2), generator=g)
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

    return {
        "rollout_mse": float(F.mse_loss(pred_traj, true_traj).item()),
        "final_state_mse": float(F.mse_loss(pred_traj[:, -1], true_traj[:, -1]).item()),
        "trajectory_r2": float(trajectory_r2(pred_traj, true_traj).mean().item()),
        "rollout_mse_true_init": float(F.mse_loss(pred_traj_true_init, true_traj).item()),
        "final_state_mse_true_init": float(F.mse_loss(pred_traj_true_init[:, -1], true_traj[:, -1]).item()),
        "eval_rollout_count": float(n_rollouts),
        "eval_rollout_horizon": float(horizon),
        "eval_rollout_dt": float(dt),
        "eval_action_low": float(action_low),
        "eval_action_high": float(action_high),
    }


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
):
    dt = 0.005
    dy = 30
    e_true = system_embedding.detach().cpu().reshape(-1).to(torch.float32)
    de = int(e_true.numel())
    du = 2
    noise_scale = 0.02
    action_strength = float(active_cfg.action_strength)

    action_model = actdyn.environment.action.IdentityActionEncoder(
        d_action=du,
        d_latent=2,
        action_bounds=[-action_strength * 10.0, action_strength * 10.0],
        device=device,
    )
    obs_model = build_observation_model(dy=dy, dt=dt, noise_scale=noise_scale)
    env = actdyn.environment.EnvWrapper(
        MixedDynamicsEnv(
            spec=spec,
            embedding_vector=e_true,
            dt=dt,
            Q=noise_scale,
            action_bounds=(
                float(action_model.action_space.low.min()),
                float(action_model.action_space.high.max()),
            ),
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
    model = actdyn.models.FilteringEmbedding(
        dynamics=dynamics,
        decoder=decoder,
        e=e_bel,
        action_encoder=action_model,
        Fe=ExactFe(meta_dynamics),
        Fz=ExactFz(meta_dynamics),
        device=device,
    )
    model.set_params(e_bel["m"])
    e_init = model.embedding.reshape(-1).detach().cpu().clone()

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
    active_policy = actdyn.policy.mpc.MpcICem(
        metric=composite_metric,
        model=model,
        device=device,
        horizon=int(active_cfg.horizon),
        num_iterations=int(active_cfg.num_iterations),
        num_samples=int(active_cfg.num_samples),
        num_elite=int(active_cfg.num_elite),
        chunk=int(active_cfg.chunk),
        verbose=False,
    )

    policy = active_policy if policy_name == "active_short" else random_policy
    agent = actdyn.Agent(env=env, model=model, buffer_length=10, policy=policy, device=device)
    config_path = os.path.join(os.path.dirname(__file__), "conf/config.yaml")
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "active_embedding", "conf", "config.yaml")
    config = ExperimentConfig.from_yaml(config_path)
    config.results_dir = ensure_dir(results_dir)
    config.training.total_steps = total_steps
    config.training.rollout_horizon = 100
    config.training.train_every = max(total_steps + 1, 10_000)
    config.training.n_epochs = 0
    decoder.set_params(obs_model)
    torch.manual_seed(seed)
    np.random.seed(seed)
    experiment = actdyn.core.experiment.MetaEmbeddingExperiment(agent=agent, config=config)
    experiment.run()

    e_hat = model.embedding.reshape(-1).detach().cpu()
    step_trace = [dict(item) for item in getattr(model, "embedding_diag_history", [])]
    block_trace = [dict(item) for item in getattr(model, "embedding_block_update_history", [])]
    diag_summary = summarize_embedding_diagnostics(
        step_history=step_trace,
        block_history=block_trace,
        e_init=e_init,
        e_final=e_hat,
    )
    final_error = float(torch.norm(e_hat - e_true).item())
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
    return {
        "system": spec.name,
        "family": spec.family,
        "policy": policy_name,
        "seed": seed,
        "embedding_mode": embedding_mode,
        "active_policy_cfg": asdict(active_cfg),
        "embedding_true": e_true.tolist(),
        "embedding_est": e_hat.tolist(),
        "final_error": final_error,
        "error_trace": [float(x) for x in experiment.e_norm],
        "post_probe_eval": post_probe_eval,
        "diagnostics": diag_summary,
        "diagnostic_step_trace": step_trace,
        "diagnostic_block_trace": block_trace,
    }


def summarize_record_group(records: list[dict]) -> dict[str, float]:
    payload: dict[str, float] = {
        "n": len(records),
        "mean_rollout_mse": float(np.mean([r["post_probe_eval"]["rollout_mse"] for r in records])),
        "std_rollout_mse": float(np.std([r["post_probe_eval"]["rollout_mse"] for r in records])),
        "mean_final_state_mse": float(np.mean([r["post_probe_eval"]["final_state_mse"] for r in records])),
        "std_final_state_mse": float(np.std([r["post_probe_eval"]["final_state_mse"] for r in records])),
        "mean_trajectory_r2": float(np.mean([r["post_probe_eval"]["trajectory_r2"] for r in records])),
        "std_trajectory_r2": float(np.std([r["post_probe_eval"]["trajectory_r2"] for r in records])),
        "mean_rollout_mse_true_init": float(np.mean([r["post_probe_eval"]["rollout_mse_true_init"] for r in records])),
        "std_rollout_mse_true_init": float(np.std([r["post_probe_eval"]["rollout_mse_true_init"] for r in records])),
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
    return payload


def default_results_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results"))


def configure_runtime_device(seed: int = 0) -> str:
    global device
    device = configure_runtime(seed=seed, device=device)
    return device


def resolve_system_bank(system_bank: str) -> tuple[SystemSpec, ...]:
    if system_bank in {"mixed80", "mixed40"}:
        return MIXED80_SYSTEM_SPECS
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
    bank = resolve_system_bank(system_bank)
    selected = select_systems(bank=bank, requested_systems=systems, system_bank=system_bank)
    if embedding_mode == "fixed":
        selected = truncate_embedding(selected, d_embed=d_embed)
    return selected


def resolve_results_dir(results_root: str | None, results_subdir: str) -> str:
    root = results_root or default_results_root()
    return ensure_dir(os.path.join(root, results_subdir))


def maybe_load_or_train_bundle(args, selected: tuple[SystemSpec, ...]) -> ModelBundle:
    if getattr(args, "checkpoint", None):
        return load_model_bundle_checkpoint(args.checkpoint, selected)
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
    mean_rollout_mse = float(np.mean([payload["rollout_eval"][spec.name]["rollout_mse"] for spec in selected]))
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
    base_dir = resolve_results_dir(getattr(args, "results_root", None), args.results_subdir)
    selected = prepare_selected_systems(
        system_bank=args.system_bank,
        systems=args.systems,
        embedding_mode=args.embedding_mode,
        d_embed=args.d_embed,
    )
    bundle = maybe_load_or_train_bundle(args=args, selected=selected)
    embeddings = system_embedding_tensor(bundle.system_embeddings, selected, target_device=device)
    figure_path = os.path.join(base_dir, getattr(args, "figure_filename", "vectorfield_family_comparison.png"))
    payload = save_family_vectorfield_comparison_figure(
        meta_dynamics=bundle.meta_dynamics,
        systems=selected,
        system_embeddings=embeddings,
        out_path=figure_path,
        dynamics_scale=args.dynamics_scale,
        grid_n=int(getattr(args, "grid_n", 25)),
    )
    metadata_path = os.path.join(base_dir, getattr(args, "metadata_filename", "vectorfield_family_comparison.json"))
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return {
        "results_dir": base_dir,
        "figure_path": figure_path,
        "metadata_path": metadata_path,
        "families": payload["representatives"],
        "grid_n": payload["grid_n"],
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
    figure_path = os.path.join(base_dir, getattr(args, "figure_filename", "embedding_family_clusters.png"))
    payload = save_embedding_cluster_figure(
        systems=selected,
        system_embeddings=embeddings,
        out_path=figure_path,
    )
    metadata_path = os.path.join(base_dir, getattr(args, "metadata_filename", "embedding_family_clusters.json"))
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
    selected = prepare_selected_systems(
        system_bank=args.system_bank,
        systems=args.systems,
        embedding_mode=args.embedding_mode,
        d_embed=args.d_embed,
    )
    bundle = maybe_load_or_train_bundle(args=args, selected=selected)
    records: list[dict] = []
    resolved_embeddings = system_embedding_tensor(bundle.system_embeddings, selected, target_device="cpu")
    for system_idx, spec in enumerate(selected):
        for policy in args.policies:
            for rep in range(args.repeats):
                run_dir = os.path.join(base_dir, spec.name, policy, f"seed_{rep}")
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
