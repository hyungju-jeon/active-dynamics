from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any
import pickle
import time
import types

import gymnasium as gym
import numpy as np
import pytest
import torch

import actdyn.policy.mpc as mpc_module
from actdyn.core.agent import Agent
from actdyn.core.experiment import _numeric_suffix
from actdyn.policy.mpc import AsyncMpcICem, MpcICem
from actdyn.utils.experiment_runtime import (
    apply_loglinear_loading_mismatch,
    apply_loglinear_loading_tuning_mismatch,
)
from actdyn.utils.rollout import Rollout
from actdyn.utils.runtime import configure_runtime, ensure_dir
from experiments.experiment_definitions import (
    ScheduleSpec,
    configure_catalogs,
    get_environment_preset,
    get_policy_spec,
)
from experiments.run import _EnvParameterFormatter, _build_env_jacobians
from experiments.tbme.run_tbme_experiments import configure_tbme_catalogs


def test_configure_runtime_returns_valid_device():
    device = configure_runtime(seed=123, device=None)
    assert device in {"cpu", "cuda", "mps"}


def test_filtering_embedding_uses_explicit_initial_state_mean():
    from actdyn.models.model import FilteringEmbedding

    class _Dynamics:
        state_dim = 3
        dt = 0.01

        def set_params(self, params):
            self.params = params

    class _Decoder:
        obs_dim = 2
        latent_dim = 3

    model = FilteringEmbedding(
        e={
            "m": torch.zeros(1, 1),
            "P": torch.eye(1).unsqueeze(0),
            "L": torch.eye(1).unsqueeze(0),
        },
        dynamics=_Dynamics(),
        decoder=_Decoder(),
        state_initial_mean=[-0.5, 0.0, 0.0],
        state_init_uncertainty=4.0,
        device="cpu",
    )

    _, info = model.reset(torch.zeros(1, 1, 2))

    assert torch.allclose(
        info["latent_state"], torch.tensor([[[-0.5, 0.0, 0.0]]])
    )
    assert torch.allclose(model.z["P"], 4.0 * torch.eye(3).reshape(1, 1, 3, 3))


def test_worker_result_tensor_uses_shared_cpu_storage() -> None:
    tensor = torch.ones(2, 3)
    result = mpc_module._worker_result_tensor(tensor)

    assert result is not None
    assert result.is_shared()
    assert torch.allclose(result, tensor)
    assert result.data_ptr() != tensor.data_ptr()


def test_ensure_dir_creates_directory(tmp_path: Path):
    target = tmp_path / "a" / "b" / "c"
    result = ensure_dir(target)
    assert Path(result).exists()
    assert Path(result).is_dir()


def test_numeric_suffix_uses_last_number() -> None:
    assert _numeric_suffix(Path("rollout_step_0010.pkl")) == 10
    assert _numeric_suffix(Path("model_final.pt")) is None


def test_eig_covariance_ablations_are_catalog_knobs(tmp_path: Path):
    catalog = tmp_path / "models.yaml"
    catalog.write_text(
        """
schedules:
  s1:
    update_interval: 1
    replan_interval: 1
    planning_horizon: 2
models:
  eig_default:
    objective_kind: parameter_eig
    schedule_id: s1
  eig_freeze:
    objective_kind: parameter_eig
    schedule_id: s1
    eig_freeze_covariance: true
  eig_diagonal:
    objective_kind: parameter_eig
    schedule_id: s1
    eig_diagonal_covariance: true
  async_planning:
    objective_kind: parameter_eig
    schedule_id: s1
    async_planning: true
    async_worker_iterations: 3
    async_worker_full_interval: 5
    async_worker_device: cpu
    async_realtime_prefix_steps: 5
""",
        encoding="utf-8",
    )

    try:
        configure_catalogs(env_catalog_paths=(), model_catalog_paths=(catalog,), suite_catalog_paths=())
        assert get_policy_spec("eig_default").eig_freeze_covariance is False
        assert get_policy_spec("eig_default").eig_diagonal_covariance is False
        assert get_policy_spec("eig_freeze").eig_freeze_covariance is True
        assert get_policy_spec("eig_diagonal").eig_diagonal_covariance is True
        assert get_policy_spec("async_planning").async_planning is True
        assert get_policy_spec("async_planning").async_worker_iterations == 3
        assert get_policy_spec("async_planning").async_worker_full_interval == 5
        assert get_policy_spec("async_planning").async_worker_device == "cpu"
        assert get_policy_spec("async_planning").async_realtime_prefix_steps == 5
    finally:
        configure_catalogs()


def test_schedule_spec_uses_replan_interval_as_single_planning_knob():
    schedule = ScheduleSpec(
        schedule_id="demo",
        update_interval=1,
        replan_interval=5,
        planning_horizon=20,
        planning_chunk=5,
        adaptive_cadence=True,
        adaptive_update_min_interval=2,
        adaptive_update_eig_threshold=0.03,
        adaptive_replan_min_interval=2,
        adaptive_replan_state_error_threshold=3.0,
    )

    assert schedule.planning_interval == 5
    assert schedule.planning_chunk == 5
    assert schedule.adaptive_cadence is True
    assert schedule.adaptive_update_eig_threshold == pytest.approx(0.03)
    legacy = ScheduleSpec("legacy", 1, 5, 20, 5, True)
    assert legacy.replan_interval == 5
    assert legacy.predictive_only_window is True

    with pytest.raises(ValueError, match="conflicting planning interval"):
        ScheduleSpec(
            schedule_id="bad",
            update_interval=1,
            replan_interval=5,
            planning_horizon=20,
            planning_chunk=4,
        )


def test_loglinear_loading_mismatch_uses_seeded_gaussian_power() -> None:
    weight = torch.zeros((3, 2), dtype=torch.float32)
    no_mismatch = apply_loglinear_loading_mismatch(weight, variance=0.0, seed=7)
    mild = apply_loglinear_loading_mismatch(weight, variance=0.01, seed=7)
    strong = apply_loglinear_loading_mismatch(weight, variance=0.04, seed=7)
    mild_repeat = apply_loglinear_loading_mismatch(weight, variance=0.01, seed=7)

    assert torch.allclose(no_mismatch, weight)
    assert torch.allclose(mild, mild_repeat)
    assert torch.allclose(strong, 2.0 * mild)


def test_loglinear_loading_tuning_mismatch_bounds_direction_and_gain() -> None:
    weight = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [-3.0, 0.0],
        ],
        dtype=torch.float32,
    )

    no_mismatch = apply_loglinear_loading_tuning_mismatch(
        weight, max_angle_deg=0.0, max_gain_factor=1.0, seed=11
    )
    perturbed = apply_loglinear_loading_tuning_mismatch(
        weight, max_angle_deg=30.0, max_gain_factor=1.5, seed=11
    )
    repeat = apply_loglinear_loading_tuning_mismatch(
        weight, max_angle_deg=30.0, max_gain_factor=1.5, seed=11
    )

    assert torch.allclose(no_mismatch, weight)
    assert torch.allclose(perturbed, repeat)

    gain = torch.linalg.norm(perturbed, dim=1) / torch.linalg.norm(weight, dim=1)
    dot = (weight * perturbed).sum(dim=1)
    cross = weight[:, 0] * perturbed[:, 1] - weight[:, 1] * perturbed[:, 0]
    angle_deg = torch.rad2deg(torch.atan2(torch.abs(cross), dot))

    assert torch.all(angle_deg <= 30.0 + 1e-5)
    assert torch.all(gain >= 1.0 / 1.5 - 1e-6)
    assert torch.all(gain <= 1.5 + 1e-6)


def test_experiment_loop_does_not_update_policy_outside_agent_step():
    from actdyn.core.experiment import Experiment

    class _Agent:
        def __init__(self):
            self.device = "cpu"
            self.plan_calls = 0
            self.step_calls = 0
            self.update_policy_calls = 0
            self.foreground_events = []

        def set_foreground_active(self, active):
            self.foreground_events.append(bool(active))

        def plan(self):
            self.plan_calls += 1
            return torch.zeros(1, 1)

        def step(self, action):
            self.step_calls += 1
            return {"action": action, "step": self.step_calls}, False

        def update_policy(self, _transition):
            self.update_policy_calls += 1

    class _Rollout:
        def __init__(self):
            self.add_calls = 0

        def add(self, **_transition):
            self.add_calls += 1

    agent = _Agent()
    experiment = Experiment.__new__(Experiment)
    experiment.agent = agent
    experiment.env_step = 0
    experiment.rollout = _Rollout()
    experiment.training_info = {}
    experiment.init_experiment = lambda reset=True: None
    experiment._setup_video_recording = lambda: None
    experiment.check_step = lambda _kind: False
    experiment.update_writer = lambda _info: None
    experiment.update_pbar = lambda _pbar: None
    experiment._finalize_experiment = lambda: None

    experiment._run_online_loop(
        train_cfg=types.SimpleNamespace(total_steps=3),
        pbar_desc="test",
        plot_fcn=None,
        reset=False,
    )

    assert agent.plan_calls == 3
    assert agent.step_calls == 3
    assert agent.update_policy_calls == 0
    assert agent.foreground_events == [
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    ]
    assert experiment.rollout.add_calls == 3


def test_experiment_loop_skips_intermediate_async_rollout_save(monkeypatch, tmp_path: Path):
    from actdyn.core import experiment as experiment_module

    def fail_sync_save(*_args, **_kwargs):
        raise AssertionError("online loop should not synchronously save rollout")

    monkeypatch.setattr(experiment_module, "save_rollout", fail_sync_save)

    class _Agent:
        device = "cpu"

        def plan(self):
            return torch.zeros(1, 1)

        def step(self, action):
            return {"action": action}, False

    class _Rollout:
        def __init__(self):
            self.add_calls = 0
            self.clear_keep_last = []

        def add(self, **_transition):
            self.add_calls += 1

        def clear(self, keep_last=0):
            self.clear_keep_last.append(keep_last)

    class _Writer:
        def __init__(self):
            self.transitions = 0
            self.scalar_calls = 0
            self.saves = []

        def add_transition(self, _transition):
            self.transitions += 1

        def add_scalar(self, *_args, **_kwargs):
            self.scalar_calls += 1

        def save_rollout(self, path, *, keep_last=None):
            self.saves.append((Path(path).name, keep_last))

    experiment = experiment_module.Experiment.__new__(experiment_module.Experiment)
    experiment.agent = _Agent()
    experiment.env_step = 0
    experiment.results_path = tmp_path
    experiment.rollout = _Rollout()
    experiment.writer = _Writer()
    experiment.training_info = {"loss": 1.0}
    experiment.init_experiment = lambda reset=True: None
    experiment._setup_video_recording = lambda: None
    experiment.check_step = lambda kind: kind == "save" and experiment.env_step == 1
    experiment.update_writer = experiment_module.Experiment.update_writer.__get__(
        experiment, experiment_module.Experiment
    )
    experiment.update_pbar = lambda _pbar: None
    experiment._finalize_experiment = lambda: None

    experiment._run_online_loop(
        train_cfg=types.SimpleNamespace(total_steps=2),
        pbar_desc="test",
        plot_fcn=None,
        reset=False,
    )

    assert experiment.rollout.add_calls == 2
    assert experiment.rollout.clear_keep_last == [100]
    assert experiment.writer.transitions == 2
    assert experiment.writer.scalar_calls == 2
    assert experiment.writer.saves == [
        ("rollout_2.pkl", None),
    ]


def test_async_experiment_writer_saves_rollout(tmp_path: Path):
    from actdyn.core.experiment import _AsyncExperimentWriter

    rollout_path = tmp_path / "rollout.pkl"
    writer = _AsyncExperimentWriter(tmp_path / "logs")
    writer.add_transition(
        {
            "action": torch.ones(1, 2),
            "obs": torch.zeros(1, 2),
        }
    )
    writer.save_rollout(rollout_path, keep_last=1)
    writer.close()

    with rollout_path.open("rb") as f:
        rollout = pickle.load(f)

    assert rollout.length == 1
    assert torch.allclose(rollout["action"], torch.ones(1, 1, 2))


class _IdentityActionEncoder:
    def __init__(self, action_space: gym.Space) -> None:
        self.action_space = action_space

    def __call__(self, actions: torch.Tensor) -> torch.Tensor:
        return actions


class _LinearActionDynamics:
    def __init__(self, *, dt: float = 0.1, state_dim: int = 2) -> None:
        self.dt = float(dt)
        self.state_dim = int(state_dim)

    def sample_forward(
        self,
        init_z: torch.Tensor,
        action: torch.Tensor | None = None,
        k_step: int = 1,
        return_traj: bool = False,
        add_noise: bool = False,
    ):
        del add_noise
        current = init_z
        samples = [current]
        mus = []
        vars_ = []
        for idx in range(int(k_step)):
            if action is None or action.shape[-1] == 0:
                step_action = torch.zeros_like(current)
            else:
                step_action = action[:, idx : idx + 1, :]
            current = current + step_action * self.dt
            samples.append(current)
            mus.append(current)
            vars_.append(torch.zeros_like(current))
        if return_traj:
            return samples, mus, vars_
        return samples[-1], mus[-1], vars_[-1]


class _DummyModel:
    def __init__(self, *, action_space: gym.Space, dt: float = 0.1) -> None:
        self.action_encoder = _IdentityActionEncoder(action_space)
        self.dynamics = _LinearActionDynamics(dt=dt, state_dim=action_space.shape[0])
        self.dt = float(dt)
        self._state = torch.zeros(1, 1, action_space.shape[0], dtype=torch.float32)

    def predict(self, action: torch.Tensor) -> torch.Tensor:
        _samples, next_states, _vars = self.dynamics.sample_forward(
            init_z=self._state,
            action=action,
            k_step=action.shape[-2],
            add_noise=False,
            return_traj=True,
        )
        return torch.cat(next_states, dim=-2)

    def get_state(self) -> torch.Tensor:
        return self._state

    def set_params(self, params: torch.Tensor) -> None:
        self.params = params.detach().clone()


class _WarmupEnv:
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float)

    def reset(self, seed=None):
        del seed
        return torch.zeros(1, 1, 2), {"latent_state": torch.zeros(1, 1, 2)}


class _WarmupModel:
    def __init__(self) -> None:
        self.device = "cpu"
        self.update_calls = 0
        self.e = {
            "m": torch.zeros(1, 2),
            "P": torch.eye(2).unsqueeze(0),
        }
        self.z = {
            "m": torch.zeros(1, 1, 2),
            "P": torch.eye(2).reshape(1, 1, 2, 2),
        }
        self._state = torch.zeros(1, 1, 2)
        self._theta_score_block = torch.zeros(1, 2)
        self._theta_info_block = torch.zeros(1, 2, 2)
        self._theta_sensitivity = torch.zeros(1, 2, 2)
        self._theta_block_steps = 0
        self.last_information = {"I_z_t": 0.0}
        self.action_encoder = None

    def reset(self, observation):
        return observation, {"latent_state": self._state.clone()}

    def update(self, recent, update_theta: bool = True):
        del recent, update_theta
        self.update_calls += 1
        self._state = torch.ones(1, 1, 2)
        self.z["m"] = self._state
        self.e["m"] = torch.ones(1, 2)
        self._theta_block_steps = 3
        self.last_information = {"I_z_t": 9.0}
        return {"latent_state": self._state, "env_action": None}

    def get_state(self):
        return self._state


class _WarmupPolicy:
    async_realtime_prefix_steps = 10
    action_dim = 2

    def __init__(self):
        self.prime_calls = 0
        self.prime_kwargs = {}

    def beginning_of_rollout(self, state):
        self.state = state

    def prime_initial_plan(self, state, **kwargs):
        self.prime_calls += 1
        self.prime_state = state
        self.prime_kwargs = kwargs


class _UnusedMetric:
    metric_list = ()

    def __call__(self, rollout, **kwargs):
        return torch.zeros(1)


class _CountingMetric:
    def __init__(self) -> None:
        self.metric_list = [self]
        self.update_calls = 0

    def update(self, rollout) -> None:
        assert rollout.get("model_state") is not None
        self.update_calls += 1

    def __call__(self, rollout, **kwargs):
        return torch.zeros(1)


def _make_coarse_policy(
    *,
    chunk: int,
    horizon: int,
    factor: int,
    mapping: str = "hold",
    seed: int | None = None,
) -> MpcICem:
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float)
    return MpcICem(
        metric=_UnusedMetric(),
        model=_DummyModel(action_space=action_space),
        horizon=horizon,
        num_samples=2,
        num_iterations=1,
        num_elite=1,
        chunk=chunk,
        device="cpu",
        noise_beta=0.0,
        coarse_dt_factor=factor,
        coarse_action_mapping=mapping,
        coarse_mapping_opt_steps=2,
        coarse_mapping_opt_lr=0.01,
        seed=seed,
    )


def _make_async_policy(*, chunk: int = 3, horizon: int = 4) -> AsyncMpcICem:
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float)
    return AsyncMpcICem(
        metric=_UnusedMetric(),
        model=_DummyModel(action_space=action_space),
        horizon=horizon,
        num_samples=2,
        num_iterations=1,
        num_elite=1,
        chunk=chunk,
        device="cpu",
        noise_beta=0.0,
        async_stale_tolerance=0.25,
        async_realtime_prefix_steps=3,
    )


def _process_anchor_worker(planner, state: torch.Tensor) -> dict[str, Any]:
    time.sleep(0.2)
    kwargs = {"parameter_update_version": 0}
    updated_state = planner._maybe_refresh_worker_belief(state, kwargs)
    return {
        "state": updated_state,
        "parameter_update_version": kwargs["parameter_update_version"],
        "model_state": planner.model._state,
        "parameter_mean": planner.model.e["m"],
        "parameter_cov": planner.model.e["P"],
        "params": getattr(planner.model, "params", None),
    }


def test_agent_reset_warms_realtime_model_update_without_changing_belief() -> None:
    model = _WarmupModel()
    agent = Agent(
        env=_WarmupEnv(),
        model=model,
        policy=_WarmupPolicy(),
        buffer_length=1,
        device="cpu",
    )

    agent.reset(seed=0)

    assert model.update_calls == 1
    assert torch.allclose(model._state, torch.zeros(1, 1, 2))
    assert torch.allclose(model.e["m"], torch.zeros(1, 2))
    assert model._theta_block_steps == 0
    assert model.last_information == {"I_z_t": 0.0}
    assert agent.policy.prime_calls == 1
    assert torch.allclose(agent.policy.prime_state, torch.zeros(1, 1, 2))
    assert agent.policy.prime_kwargs["parameter_update_version"] == 0


def test_coarse_dt_mapping_uses_chunk_to_select_macro_actions() -> None:
    policy = _make_coarse_policy(chunk=20, horizon=3, factor=10)
    seen_shift_steps: list[int] = []

    def fake_search(self, state, *, shift_steps=1, debug=False, **kwargs):
        del state, debug, kwargs
        seen_shift_steps.append(int(shift_steps))
        assert self.model.dt == pytest.approx(1.0)
        assert self.model.dynamics.dt == pytest.approx(1.0)
        plan = torch.tensor([[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]], dtype=torch.float32)
        return plan, torch.tensor([0.0])

    policy._run_icem_search = types.MethodType(fake_search, policy)

    fine_actions, cost = policy.get_action(torch.zeros(1, 1, 2))

    assert cost.item() == pytest.approx(0.0)
    assert seen_shift_steps == [2]
    assert policy.model.dt == pytest.approx(0.1)
    assert policy.model.dynamics.dt == pytest.approx(0.1)
    assert fine_actions.shape == (1, 20, 2)
    assert torch.allclose(fine_actions[:, :10], torch.tensor([[[0.1, 0.2]]]).expand(1, 10, 2))
    assert torch.allclose(fine_actions[:, 10:], torch.tensor([[[0.3, 0.4]]]).expand(1, 10, 2))


def test_coarse_dt_mapping_ceilings_and_truncates_nondivisible_chunks() -> None:
    policy = _make_coarse_policy(chunk=25, horizon=3, factor=10)
    seen_shift_steps: list[int] = []

    def fake_search(self, state, *, shift_steps=1, debug=False, **kwargs):
        del state, debug, kwargs
        seen_shift_steps.append(int(shift_steps))
        plan = torch.tensor([[[0.1, 0.0], [0.2, 0.0], [0.3, 0.0]]], dtype=torch.float32)
        return plan, torch.tensor([0.0])

    policy._run_icem_search = types.MethodType(fake_search, policy)

    fine_actions, _cost = policy.get_action(torch.zeros(1, 1, 2))

    assert seen_shift_steps == [3]
    assert fine_actions.shape == (1, 25, 2)
    assert torch.allclose(fine_actions[:, :10], torch.tensor([[[0.1, 0.0]]]).expand(1, 10, 2))
    assert torch.allclose(fine_actions[:, 10:20], torch.tensor([[[0.2, 0.0]]]).expand(1, 10, 2))
    assert torch.allclose(fine_actions[:, 20:], torch.tensor([[[0.3, 0.0]]]).expand(1, 5, 2))


def test_coarse_dt_restores_model_dt_when_planning_raises() -> None:
    policy = _make_coarse_policy(chunk=20, horizon=3, factor=10)

    def fake_search(self, state, *, shift_steps=1, debug=False, **kwargs):
        del self, state, shift_steps, debug, kwargs
        raise RuntimeError("planner failed")

    policy._run_icem_search = types.MethodType(fake_search, policy)

    with pytest.raises(RuntimeError, match="planner failed"):
        policy.get_action(torch.zeros(1, 1, 2))

    assert policy.model.dt == pytest.approx(0.1)
    assert policy.model.dynamics.dt == pytest.approx(0.1)


def test_state_tracking_error_forces_next_replan() -> None:
    policy = _make_coarse_policy(chunk=4, horizon=4, factor=1)
    policy.adaptive_replanning = True
    policy.adaptive_replan_min_interval = 1
    policy.adaptive_replan_state_error_threshold = 0.1
    plan_calls = 0

    def fake_get_action(self, state, **kwargs):
        del state, kwargs
        nonlocal plan_calls
        plan_calls += 1
        return torch.zeros(1, 4, 2), torch.tensor([0.0])

    policy.get_action = types.MethodType(fake_get_action, policy)

    policy(torch.zeros(1, 1, 2))
    update_info = policy.update({"next_model_state": torch.tensor([[[1.0, 0.0]]])})

    assert update_info["adaptive_replan_triggered"] is True
    assert update_info["adaptive_replan_reason"] == "state_tracking_error"

    policy(torch.zeros(1, 1, 2))
    assert plan_calls == 2


def test_endpoint_opt_mapping_returns_bounded_fine_actions() -> None:
    policy = _make_coarse_policy(chunk=5, horizon=2, factor=3, mapping="endpoint_opt")

    def fake_search(self, state, *, shift_steps=1, debug=False, **kwargs):
        del self, state, shift_steps, debug, kwargs
        plan = torch.tensor([[[1.5, 0.0], [0.5, 0.0]]], dtype=torch.float32)
        return plan, torch.tensor([0.0])

    policy._run_icem_search = types.MethodType(fake_search, policy)

    fine_actions, _cost = policy.get_action(torch.zeros(1, 1, 2))

    assert fine_actions.shape == (1, 5, 2)
    assert torch.all(fine_actions <= 1.0 + 1e-6)
    assert torch.all(fine_actions >= -1.0 - 1e-6)
    assert policy.model.dt == pytest.approx(0.1)
    assert policy.model.dynamics.dt == pytest.approx(0.1)


def test_active_planning_catalog_uses_coarse_dt_by_default() -> None:
    spec = get_policy_spec("active_planning")
    assert spec.coarse_dt_factor == 10
    assert spec.coarse_action_mapping == "hold"


def test_async_mpc_returns_cached_action_while_future_runs() -> None:
    policy = _make_async_policy(chunk=3)
    policy.beginning_of_rollout(torch.zeros(1, 1, 2))
    policy._set_current_buffer(
        torch.tensor([[[0.1, 0.0], [0.2, 0.0], [0.3, 0.0]]], dtype=torch.float32),
        torch.tensor([0.0]),
    )
    policy._planning_future = Future()

    action = policy(torch.zeros(1, 1, 2))

    assert torch.allclose(action, torch.tensor([[[0.1, 0.0]]], dtype=torch.float32))
    assert policy._buffer_index == 1
    assert policy._planning_future is not None
    assert policy.last_plan_status["async_plan_status"] == "waiting_ready_tail"
    assert policy.last_plan_status["async_plan_used"] is False
    policy.close()


def test_async_mpc_reports_ready_plan_waiting_for_chunk_boundary() -> None:
    policy = _make_async_policy(chunk=3)
    policy.beginning_of_rollout(torch.zeros(1, 1, 2))
    policy._set_current_buffer(torch.zeros(1, 3, 2), torch.tensor([0.0]))
    future = Future()
    future.set_result(
        types.SimpleNamespace(
            actions=torch.ones(1, 3, 2),
            cost=torch.tensor([1.0]),
            predicted_boundary_state=torch.zeros(1, 1, 2),
            runtime_sec=0.01,
            model_update_version=0,
            mean=None,
            std=None,
            elite_actions=None,
            elite_costs_traj=None,
        )
    )
    policy._planning_future = future

    _ = policy(torch.zeros(1, 1, 2))

    assert policy.last_plan_status["async_plan_status"] == "ready_waiting_boundary"
    assert policy.last_plan_status["async_plan_used"] is False
    policy.close()


def test_async_mpc_uses_buffer_tail_while_future_runs_at_boundary() -> None:
    policy = _make_async_policy(chunk=3, horizon=5)
    policy.beginning_of_rollout(torch.zeros(1, 1, 2))
    policy._set_current_buffer(
        torch.tensor(
            [[[0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0], [0.5, 0.0]]],
            dtype=torch.float32,
        ),
        torch.tensor([0.0]),
    )
    policy._buffer_index = 3
    policy._planning_future = Future()
    policy._launch_background_plan = types.MethodType(lambda self, state, kwargs: None, policy)

    action = policy(torch.zeros(1, 1, 2))

    assert torch.allclose(action, torch.tensor([[[0.4, 0.0]]], dtype=torch.float32))
    assert policy._buffer_index == 4
    assert policy.last_plan_status["async_plan_status"] == "waiting_ready_tail"
    assert policy.last_plan_status["async_blocking_fallback"] is False
    policy.close()


def test_async_mpc_publishes_dynamic_anytime_prefix() -> None:
    policy = _make_async_policy(chunk=3, horizon=3)
    policy.coarse_dt_factor = 10
    policy.async_anytime_prefix_steps = None
    policy.async_anytime_std_tolerance = 0.5
    policy.beginning_of_rollout(torch.zeros(1, 1, 2))
    policy._yield_to_foreground = True
    policy._active_anytime_worker_id = 4
    policy._active_anytime_model_update_version = 7
    policy._active_anytime_launch_step = 0
    policy._async_anytime_anchor_state = torch.zeros(1, 1, 2)
    policy._ensure_shared_anytime_storage(torch.zeros(1, 1, 2))
    policy.std = torch.zeros_like(policy.std)
    policy.std[2] = 1.0

    policy._maybe_publish_anytime_prefix(
        torch.tensor([[[0.8, 0.0], [0.6, 0.0], [0.4, 0.0]]], dtype=torch.float32),
        torch.tensor([1.25]),
        iteration=1,
    )

    assert int(policy._async_anytime_shared_meta[0].item()) == 1
    assert int(policy._async_anytime_shared_meta[1].item()) == 4
    assert int(policy._async_anytime_shared_meta[2].item()) == 7
    assert int(policy._async_anytime_shared_meta[3].item()) == 1
    assert int(policy._async_anytime_shared_meta[4].item()) == 20
    assert int(policy._async_anytime_shared_meta[5].item()) == 0
    assert policy._async_anytime_shared_actions.shape == (1, 30, 2)
    assert policy._async_anytime_shared_stats[0].item() == pytest.approx(0.0)
    assert policy._async_anytime_shared_stats[1].item() == pytest.approx(1.25)
    assert torch.allclose(
        policy._async_anytime_shared_actions[:, :10],
        torch.tensor([[[0.8, 0.0]]], dtype=torch.float32).expand(1, 10, 2),
    )
    assert torch.allclose(
        policy._async_anytime_shared_actions[:, 10:20],
        torch.tensor([[[0.6, 0.0]]], dtype=torch.float32).expand(1, 10, 2),
    )
    policy.close()


def test_async_mpc_uses_anytime_prefix_while_future_runs() -> None:
    policy = _make_async_policy(chunk=3)
    policy.async_anytime_prefix_steps = 2
    policy.beginning_of_rollout(torch.zeros(1, 1, 2))
    policy._set_current_buffer(torch.zeros(1, 3, 2), torch.tensor([0.0]))
    policy._buffer_index = 3
    policy._planning_future = Future()
    policy._launch_background_plan = types.MethodType(lambda self, state, kwargs: None, policy)
    policy._active_anytime_worker_id = 5
    policy._ensure_shared_anytime_storage(torch.zeros(1, 1, 2))
    policy._async_anytime_shared_actions.copy_(
        torch.tensor([[[0.7, 0.0], [0.9, 0.0]]], dtype=torch.float32)
    )
    policy._async_anytime_shared_anchor.copy_(torch.zeros(1, 1, 2))
    policy._async_anytime_shared_stats[0] = 0.2
    policy._async_anytime_shared_stats[1] = 1.0
    policy._async_anytime_shared_meta[1] = 5
    policy._async_anytime_shared_meta[2] = 0
    policy._async_anytime_shared_meta[3] = 1
    policy._async_anytime_shared_meta[4] = 2
    policy._async_anytime_shared_meta[0] = 1

    action = policy(torch.zeros(1, 1, 2))

    assert torch.allclose(action, torch.tensor([[[0.7, 0.0]]], dtype=torch.float32))
    assert policy.last_plan_status["async_plan_status"] == "used_anytime_prefix"
    assert policy.last_plan_status["async_anytime_plan_ready"] is True
    assert policy.last_plan_status["async_anytime_plan_used"] is True
    assert policy.last_plan_status["async_plan_used"] is False
    policy.close()


def test_async_mpc_uses_anytime_prefix_instead_of_tail() -> None:
    policy = _make_async_policy(chunk=3, horizon=5)
    policy.coarse_dt_factor = 10
    policy.async_anytime_prefix_steps = None
    policy.beginning_of_rollout(torch.zeros(1, 1, 2))
    policy._set_current_buffer(
        torch.tensor(
            [[[0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0], [0.5, 0.0]]],
            dtype=torch.float32,
        ),
        torch.tensor([0.0]),
    )
    policy._buffer_index = 1
    policy.count = 7
    policy._planning_future = Future()
    policy._active_anytime_worker_id = 6
    policy._ensure_shared_anytime_storage(torch.zeros(1, 1, 2))
    policy._async_anytime_shared_actions[:, :10].copy_(
        torch.tensor([[[0.8, 0.0]]], dtype=torch.float32).expand(1, 10, 2)
    )
    policy._async_anytime_shared_actions[:, 10:20].copy_(
        torch.tensor([[[0.9, 0.0]]], dtype=torch.float32).expand(1, 10, 2)
    )
    policy._async_anytime_shared_anchor.copy_(torch.zeros(1, 1, 2))
    policy._async_anytime_shared_stats[0] = 0.1
    policy._async_anytime_shared_stats[1] = 2.0
    policy._async_anytime_shared_meta[1] = 6
    policy._async_anytime_shared_meta[2] = 0
    policy._async_anytime_shared_meta[3] = 1
    policy._async_anytime_shared_meta[4] = 20
    policy._async_anytime_shared_meta[5] = 0
    policy._async_anytime_shared_meta[0] = 1

    action = policy(torch.zeros(1, 1, 2))

    assert torch.allclose(action, torch.tensor([[[0.8, 0.0]]], dtype=torch.float32))
    assert policy._current_buffer.shape[-2] == 13
    assert torch.allclose(
        policy._current_buffer[:, :3],
        torch.tensor([[[0.8, 0.0]]], dtype=torch.float32).expand(1, 3, 2),
    )
    assert torch.allclose(
        policy._current_buffer[:, 3:],
        torch.tensor([[[0.9, 0.0]]], dtype=torch.float32).expand(1, 10, 2),
    )
    assert policy.last_plan_status["async_plan_status"] == "used_anytime_prefix"
    assert policy.last_plan_status["async_anytime_plan_used"] is True
    policy.close()


def test_async_mpc_replaces_anytime_prefix_when_final_plan_is_ready() -> None:
    policy = _make_async_policy(chunk=3, horizon=5)
    policy.async_anytime_prefix_steps = None
    policy.beginning_of_rollout(torch.zeros(1, 1, 2))
    policy._set_current_buffer(
        torch.tensor(
            [[[0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0], [0.5, 0.0]]],
            dtype=torch.float32,
        ),
        torch.tensor([0.0]),
    )
    policy._buffer_index = 1
    policy.count = 7
    policy._launch_background_plan = types.MethodType(lambda self, state, kwargs: None, policy)
    future = Future()
    future.set_result(
        types.SimpleNamespace(
            actions=torch.cat(
                [
                    torch.tensor([[[0.8, 0.0]]], dtype=torch.float32).expand(1, 10, 2),
                    torch.tensor([[[0.9, 0.0]]], dtype=torch.float32).expand(1, 10, 2),
                ],
                dim=1,
            ),
            cost=torch.tensor([2.0]),
            predicted_boundary_state=torch.zeros(1, 1, 2),
            runtime_sec=0.01,
            model_update_version=0,
            mean=None,
            std=None,
            elite_actions=None,
            elite_costs_traj=None,
            anytime_launch_step=0,
        )
    )
    policy._planning_future = future

    action = policy(torch.zeros(1, 1, 2))

    assert torch.allclose(action, torch.tensor([[[0.8, 0.0]]], dtype=torch.float32))
    assert policy._current_buffer.shape[-2] == 13
    assert torch.allclose(
        policy._current_buffer[:, :3],
        torch.tensor([[[0.8, 0.0]]], dtype=torch.float32).expand(1, 3, 2),
    )
    assert torch.allclose(
        policy._current_buffer[:, 3:],
        torch.tensor([[[0.9, 0.0]]], dtype=torch.float32).expand(1, 10, 2),
    )
    assert policy.last_plan_status["async_plan_status"] == "used_ready"
    assert policy.last_plan_status["async_plan_used"] is True
    policy.close()


def test_async_mpc_swaps_ready_background_plan_at_boundary() -> None:
    policy = _make_async_policy(chunk=3)
    policy.beginning_of_rollout(torch.zeros(1, 1, 2))
    policy._set_current_buffer(torch.zeros(1, 3, 2), torch.tensor([0.0]))
    policy._buffer_index = 3
    policy._launch_background_plan = types.MethodType(lambda self, state, kwargs: None, policy)
    future = Future()
    future.set_result(
        types.SimpleNamespace(
            actions=torch.tensor([[[0.4, 0.0], [0.5, 0.0], [0.6, 0.0]]], dtype=torch.float32),
            cost=torch.tensor([1.0]),
            predicted_boundary_state=torch.zeros(1, 1, 2),
            runtime_sec=0.01,
            mean=None,
            std=None,
            elite_actions=None,
            elite_costs_traj=None,
        )
    )
    policy._planning_future = future

    action = policy(torch.zeros(1, 1, 2))

    assert torch.allclose(action, torch.tensor([[[0.4, 0.0]]], dtype=torch.float32))
    assert policy.last_plan_status["async_plan_used"] is True
    assert policy.last_plan_status["async_blocking_fallback"] is False
    policy.close()


def test_async_mpc_stale_plan_uses_realtime_fallback() -> None:
    policy = _make_async_policy(chunk=3)
    policy.beginning_of_rollout(torch.zeros(1, 1, 2))
    policy._set_current_buffer(torch.zeros(1, 3, 2), torch.tensor([0.0]))
    policy._buffer_index = 3
    policy._launch_background_plan = types.MethodType(lambda self, state, kwargs: None, policy)

    def fake_sync(self, state, *, num_iterations=None, **kwargs):
        del self, state, num_iterations, kwargs
        raise AssertionError("stale async result should use realtime fallback")

    policy._sync_plan = types.MethodType(fake_sync, policy)
    future = Future()
    future.set_result(
        types.SimpleNamespace(
            actions=torch.ones(1, 3, 2) * 0.4,
            cost=torch.tensor([1.0]),
            predicted_boundary_state=torch.ones(1, 1, 2) * 10.0,
            runtime_sec=0.01,
            mean=None,
            std=None,
            elite_actions=None,
            elite_costs_traj=None,
        )
    )
    policy._planning_future = future

    action = policy(torch.zeros(1, 1, 2))

    assert torch.allclose(action, torch.zeros(1, 1, 2))
    assert policy.last_plan_status["async_plan_stale"] is True
    assert policy.last_plan_status["async_refined"] is False
    assert policy.last_plan_status["async_realtime_fallback"] is True
    policy.close()


def test_async_mpc_uses_parameter_stale_ready_plan() -> None:
    policy = _make_async_policy(chunk=3)
    policy.beginning_of_rollout(torch.zeros(1, 1, 2))
    policy._set_current_buffer(torch.zeros(1, 3, 2), torch.tensor([0.0]))
    policy._buffer_index = 3
    policy._launch_background_plan = types.MethodType(lambda self, state, kwargs: None, policy)

    def fake_sync(self, state, *, num_iterations=None, **kwargs):
        del self, state, num_iterations, kwargs
        raise AssertionError("parameter-version mismatch alone should not refine synchronously")

    policy._sync_plan = types.MethodType(fake_sync, policy)
    future = Future()
    future.set_result(
        types.SimpleNamespace(
            actions=torch.ones(1, 3, 2) * 0.4,
            cost=torch.tensor([1.0]),
            predicted_boundary_state=torch.zeros(1, 1, 2),
            runtime_sec=0.01,
            model_update_version=1,
            mean=None,
            std=None,
            elite_actions=None,
            elite_costs_traj=None,
        )
    )
    policy._planning_future = future

    action = policy(torch.zeros(1, 1, 2), model_update_version=2)

    assert torch.allclose(action, torch.ones(1, 1, 2) * 0.4)
    assert policy.last_plan_status["async_plan_used"] is True
    assert policy.last_plan_status["async_model_version_mismatch"] is True
    policy.close()


def test_async_mpc_can_use_parameter_stale_ready_plan_when_configured() -> None:
    policy = _make_async_policy(chunk=3)
    policy.beginning_of_rollout(torch.zeros(1, 1, 2))
    policy._set_current_buffer(torch.zeros(1, 3, 2), torch.tensor([0.0]))
    policy._buffer_index = 3
    policy._launch_background_plan = types.MethodType(lambda self, state, kwargs: None, policy)
    future = Future()
    future.set_result(
        types.SimpleNamespace(
            actions=torch.ones(1, 3, 2) * 0.4,
            cost=torch.tensor([1.0]),
            predicted_boundary_state=torch.zeros(1, 1, 2),
            runtime_sec=0.01,
            model_update_version=1,
            mean=None,
            std=None,
            elite_actions=None,
            elite_costs_traj=None,
        )
    )
    policy._planning_future = future

    action = policy(torch.zeros(1, 1, 2), model_update_version=2)

    assert torch.allclose(action, torch.ones(1, 1, 2) * 0.4)
    assert policy.last_plan_status["async_plan_used"] is True
    assert policy.last_plan_status["async_model_version_mismatch"] is True
    assert policy.last_plan_status["async_plan_status"] == "used_ready_parameter_stale"
    policy.close()


def test_async_mpc_update_skips_unused_state_tracking() -> None:
    policy = _make_async_policy(chunk=3)
    policy.adaptive_replanning = True
    policy.adaptive_replan_min_interval = 1
    policy.adaptive_replan_state_error_threshold = 0.1
    policy.beginning_of_rollout(torch.zeros(1, 1, 2))
    policy._launch_background_plan = types.MethodType(lambda self, state, kwargs: None, policy)
    policy._set_current_buffer(torch.zeros(1, 3, 2), torch.tensor([0.0]))
    _ = policy(torch.zeros(1, 1, 2))

    update_info = policy.update({"next_model_state": torch.tensor([[[1.0, 0.0]]])})

    assert update_info["adaptive_replan_triggered"] is False
    assert update_info["adaptive_replan_reason"] == "none"
    assert policy._force_replan_next is False
    policy.close()


def test_async_mpc_defers_parameter_replans_to_boundary_reconciliation() -> None:
    policy = _make_async_policy(chunk=3)

    policy.request_replan("parameter_update")
    assert policy._force_replan_next is False

    policy.request_replan("state_tracking_error")
    assert policy._force_replan_next is False
    policy.close()


def test_async_mpc_defers_state_replans_when_stale_parameter_plans_are_allowed() -> None:
    policy = _make_async_policy(chunk=3)

    policy.request_replan("state_tracking_error")
    assert policy._force_replan_next is False
    policy.close()


def test_async_mpc_fast_stale_mode_skips_unused_state_tracking() -> None:
    policy = _make_async_policy(chunk=3)
    policy.adaptive_replanning = True
    policy.adaptive_replan_min_interval = 1
    policy.adaptive_replan_state_error_threshold = 0.1
    policy.beginning_of_rollout(torch.zeros(1, 1, 2))
    policy._set_current_buffer(torch.zeros(1, 3, 2), torch.tensor([0.0]))

    update_info = policy.update({"next_model_state": torch.tensor([[[1.0, 0.0]]])})

    assert policy._planned_state_trace is None
    assert update_info["adaptive_replan_triggered"] is False
    assert update_info["adaptive_replan_reason"] == "none"
    policy.close()


def test_async_mpc_stale_parameter_mode_keeps_chunk_swap_cadence() -> None:
    policy = _make_async_policy(chunk=3)
    policy.adaptive_replanning = True

    assert policy._async_execution_interval() == 3
    policy.close()


def test_async_mpc_warms_process_executor_before_rollout() -> None:
    policy = _make_async_policy(chunk=3)
    calls = []

    class _ImmediateFuture:
        def result(self):
            calls.append("result")

    class _Executor:
        def submit(self, fn):
            calls.append(fn.__name__)
            return _ImmediateFuture()

    policy._executor = _Executor()
    policy._warm_executor()

    assert calls == ["_warm_process_worker", "result"]
    policy._executor = None
    policy.close()


def test_async_mpc_warms_process_snapshot_without_advancing_action_rng() -> None:
    policy = _make_async_policy(chunk=3)
    policy.beginning_of_rollout(torch.zeros(1, 1, 2))
    counter_before = int(policy._async_action_seed_counter)
    calls = []

    class _ImmediateFuture:
        def result(self):
            calls.append("result")

    class _Executor:
        def submit(self, fn, planner):
            calls.append((fn.__name__, planner._executor, planner._planning_future))
            return _ImmediateFuture()

    policy._executor = _Executor()
    policy._warm_process_snapshot(torch.zeros(1, 1, 2))

    assert calls == [("_warm_process_planner_worker", None, None), "result"]
    assert policy._async_action_seed_counter == counter_before
    policy._executor = None
    policy.close()


def test_async_mpc_process_executor_sets_tensor_sharing_strategy(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(
        mpc_module,
        "_use_file_system_tensor_sharing",
        lambda: calls.append("sharing"),
    )
    policy = _make_async_policy(chunk=3)

    executor = policy._new_executor()
    executor.shutdown(wait=True, cancel_futures=True)

    assert calls[-1:] == ["sharing"]
    policy.close()


def test_async_mpc_rollout_close_waits_for_executor() -> None:
    policy = _make_async_policy(chunk=3)
    calls = []

    class _Executor:
        def shutdown(self, *, wait, cancel_futures):
            calls.append((wait, cancel_futures))

    policy._executor = _Executor()
    policy.end_of_rollout(0.0, 0.0, "test")

    assert calls == [(True, True)]
    assert policy._executor is None


def test_async_mpc_process_snapshot_keeps_warm_start_payload() -> None:
    policy = _make_async_policy(chunk=3)
    policy.beginning_of_rollout(torch.zeros(1, 1, 2))

    planner = policy._make_snapshot_planner(torch.zeros(1, 1, 2))
    assert planner._return_worker_warm_start is True

    planner = policy._make_snapshot_planner(torch.zeros(1, 1, 2))
    assert planner._return_worker_warm_start is True
    policy.close()


def test_async_mpc_missing_plan_uses_realtime_fallback() -> None:
    policy = _make_async_policy(chunk=3)
    policy.beginning_of_rollout(torch.zeros(1, 1, 2))
    policy._set_current_buffer(torch.zeros(1, 3, 2), torch.tensor([0.0]))
    policy._buffer_index = 3
    policy._launch_background_plan = types.MethodType(lambda self, state, kwargs: None, policy)
    def fake_sync(self, state, *, num_iterations=None, **kwargs):
        del self, state, num_iterations, kwargs
        raise AssertionError("missing async plan should use realtime fallback")

    policy._sync_plan = types.MethodType(fake_sync, policy)

    action = policy(torch.zeros(1, 1, 2))

    assert torch.allclose(action, torch.zeros(1, 1, 2))
    assert policy.last_plan_status["async_realtime_fallback"] is True
    assert policy.last_plan_status["async_blocking_fallback"] is False
    policy.close()


def test_async_mpc_can_defer_background_launch() -> None:
    policy = _make_async_policy(chunk=3)
    policy.beginning_of_rollout(torch.zeros(1, 1, 2))
    calls = []

    def fake_launch(self, state, kwargs):
        calls.append((state.detach().clone(), dict(kwargs)))

    policy._launch_background_plan = types.MethodType(fake_launch, policy)

    _ = policy(torch.zeros(1, 1, 2), defer_background_launch=True)
    assert calls == []

    policy.launch_background_plan(torch.zeros(1, 1, 2), parameter_update_version=1)
    assert len(calls) == 1
    assert calls[0][1]["parameter_update_version"] == 1
    policy.close()


def test_async_mpc_waits_with_last_action_when_future_is_running() -> None:
    policy = _make_async_policy(chunk=3)
    policy.beginning_of_rollout(torch.zeros(1, 1, 2))
    policy._set_current_buffer(
        torch.tensor([[[0.1, 0.0], [0.2, 0.0], [0.3, 0.0]]], dtype=torch.float32),
        torch.tensor([0.0]),
    )
    policy._buffer_index = 3
    policy._planning_future = Future()
    policy._launch_background_plan = types.MethodType(lambda self, state, kwargs: None, policy)

    def fail_sync(self, state, *, num_iterations=None, **kwargs):
        del self, state, num_iterations, kwargs
        raise AssertionError("running async plan should not block on foreground iCEM")

    policy._sync_plan = types.MethodType(fail_sync, policy)

    action = policy(torch.zeros(1, 1, 2))

    assert torch.allclose(action, torch.tensor([[[0.3, 0.0]]], dtype=torch.float32))
    assert policy.last_plan_status["async_plan_status"] == "waiting_ready_hold"
    assert policy.last_plan_status["async_blocking_fallback"] is False
    policy.close()


def test_async_mpc_realtime_zero_only_fallback_does_not_run_icem() -> None:
    policy = _make_async_policy(chunk=3, horizon=6)
    policy.async_realtime_prefix_steps = 1
    policy.beginning_of_rollout(torch.zeros(1, 1, 2))
    policy._set_current_buffer(torch.zeros(1, 3, 2), torch.tensor([0.0]))
    policy._buffer_index = 3
    policy._launch_background_plan = types.MethodType(lambda self, state, kwargs: None, policy)

    def fail_sync(self, state, *, num_iterations=None, **kwargs):
        del self, state, num_iterations, kwargs
        raise AssertionError("zero-only realtime fallback should not call _sync_plan")

    def fail_search(self, state, *, shift_steps=1, debug=False, **kwargs):
        del self, state, shift_steps, debug, kwargs
        raise AssertionError("zero-only realtime fallback should not run iCEM")

    policy._sync_plan = types.MethodType(fail_sync, policy)
    policy._run_icem_search = types.MethodType(fail_search, policy)

    action = policy(torch.zeros(1, 1, 2))

    assert torch.allclose(action, torch.zeros(1, 1, 2))
    assert policy._current_buffer.shape[-2] == 1
    assert policy.last_plan_status["async_realtime_fallback"] is True
    assert policy.last_plan_status["async_realtime_fallback_steps"] == 1
    assert policy.last_plan_status["async_realtime_zero_prefix"] is True
    assert policy.last_plan_status["async_blocking_fallback"] is False
    policy.close()


def test_async_mpc_realtime_initial_fallback_uses_one_fine_step() -> None:
    policy = _make_async_policy(chunk=3, horizon=6)
    policy.async_realtime_prefix_steps = 3
    policy.beginning_of_rollout(torch.zeros(1, 1, 2))
    policy._launch_background_plan = types.MethodType(lambda self, state, kwargs: None, policy)

    def fail_sync(self, state, *, num_iterations=None, **kwargs):
        del self, state, num_iterations, kwargs
        raise AssertionError("initial realtime fallback should not call _sync_plan")

    def fail_search(self, state, *, shift_steps=1, debug=False, **kwargs):
        del self, state, shift_steps, debug, kwargs
        raise AssertionError("initial zero-only realtime fallback should not run iCEM")

    policy._sync_plan = types.MethodType(fail_sync, policy)
    policy._run_icem_search = types.MethodType(fail_search, policy)

    action = policy(torch.zeros(1, 1, 2))

    assert torch.allclose(action, torch.zeros(1, 1, 2))
    assert policy.last_plan_status["async_plan_status"] == "initial_realtime_fallback"
    assert policy.last_plan_status["async_realtime_fallback_steps"] == 1
    assert policy._current_buffer.shape[-2] == 1
    assert policy._realtime_feedback_target_state is None
    policy.close()


def test_async_mpc_prime_initial_plan_sets_buffer_before_realtime_loop() -> None:
    policy = _make_async_policy(chunk=3, horizon=6)
    policy.beginning_of_rollout(torch.zeros(1, 1, 2))

    def fake_sync(self, state, **kwargs):
        del self, state, kwargs
        return torch.tensor([[[0.2, 0.0], [0.3, 0.0], [0.4, 0.0]]]), torch.tensor([0.0])

    policy._sync_plan = types.MethodType(fake_sync, policy)

    policy.prime_initial_plan(torch.zeros(1, 1, 2), parameter_update_version=0)

    assert policy.last_plan_status["async_plan_status"] == "initial_primed"
    assert torch.allclose(
        policy(torch.zeros(1, 1, 2)),
        torch.tensor([[[0.2, 0.0]]]),
    )
    assert policy.last_plan_status["async_realtime_fallback"] is False
    policy.close()


def test_async_mpc_returns_full_coarse_planning_tail() -> None:
    policy = _make_async_policy(chunk=3, horizon=5)
    policy.coarse_dt_factor = 2
    policy.beginning_of_rollout(torch.zeros(1, 1, 2))

    actions, _cost = policy._sync_plan(torch.zeros(1, 1, 2))

    assert actions.shape[-2] == 10
    policy.close()


def test_async_mpc_does_not_duplicate_buffer_into_action_list() -> None:
    policy = _make_async_policy(chunk=3, horizon=6)
    policy.beginning_of_rollout(torch.zeros(1, 1, 2))
    policy._set_current_buffer(torch.zeros(1, 100, 2), torch.tensor([0.0]))

    action = policy(torch.zeros(1, 1, 2))

    assert policy.action_list == []
    assert torch.allclose(action, torch.zeros(1, 1, 2))
    policy.close()


def test_async_mpc_parameter_update_replan_is_ignored() -> None:
    policy = _make_async_policy(chunk=3)
    policy.request_replan("parameter_update")
    assert policy._force_replan_next is False

    policy.request_replan("parameter_update")
    assert policy._force_replan_next is False
    policy.close()


def test_async_mpc_realtime_zero_prefix_feedback_tracks_zero_boundary() -> None:
    policy = _make_async_policy(chunk=3, horizon=6)
    policy.async_realtime_prefix_steps = 3
    policy.beginning_of_rollout(torch.zeros(1, 1, 2))
    policy._set_current_buffer(torch.zeros(1, 3, 2), torch.tensor([0.0]))
    policy._buffer_index = 3
    policy._launch_background_plan = types.MethodType(lambda self, state, kwargs: None, policy)

    first_action = policy(torch.zeros(1, 1, 2))
    second_action = policy(torch.tensor([[[0.3, 0.0]]], dtype=torch.float32))

    assert torch.allclose(first_action, torch.zeros(1, 1, 2))
    assert policy.last_plan_status["async_realtime_fallback_steps"] == 3
    assert torch.allclose(second_action, torch.tensor([[[-1.0, 0.0]]], dtype=torch.float32))
    policy.close()


def test_async_ready_plan_compensates_missed_realtime_prefix_steps() -> None:
    policy = _make_async_policy(chunk=3, horizon=6)
    policy.beginning_of_rollout(torch.zeros(1, 1, 2))
    policy._realtime_feedback_target_state = torch.zeros(1, 1, 2)
    policy._realtime_feedback_prefix_steps = 3
    policy._buffer_index = 1
    result = types.SimpleNamespace(
        actions=torch.tensor(
            [[[0.5, 0.0], [0.5, 0.0], [0.5, 0.0], [0.2, 0.0]]],
            dtype=torch.float32,
        ),
        predicted_boundary_state=torch.zeros(1, 1, 2),
        realtime_prefix_index=0,
    )

    adjusted = policy._reconcile_ready_plan_with_realtime_prefix(
        result,
        torch.zeros(1, 1, 2),
    )

    assert torch.allclose(adjusted[:, :2], torch.tensor([[[0.75, 0.0], [0.75, 0.0]]]))
    assert torch.allclose(adjusted[:, 2:], torch.tensor([[[0.2, 0.0]]]))
    policy.close()


def test_async_background_launch_passes_rollout_to_worker() -> None:
    policy = _make_async_policy(chunk=3)
    rollout = Rollout(device="cpu")
    rollout.add(model_state=torch.zeros(1, 2))
    policy.beginning_of_rollout(torch.zeros(1, 1, 2))
    policy._set_current_buffer(torch.zeros(1, 3, 2), torch.tensor([0.0]))

    class _Executor:
        def submit(self, fn, planner, predicted_boundary_state, kwargs):
            del fn, planner, predicted_boundary_state
            assert kwargs["recent_rollout"].get("model_state") is not None
            future = Future()
            future.set_result(
                types.SimpleNamespace(
                    actions=torch.zeros(1, 3, 2),
                    cost=torch.tensor([0.0]),
                    predicted_boundary_state=torch.zeros(1, 1, 2),
                    runtime_sec=0.0,
                    model_update_version=0,
                    mean=None,
                    std=None,
                    elite_actions=None,
                    elite_costs_traj=None,
                )
            )
            return future

    policy._executor = _Executor()
    policy._launch_background_plan(torch.zeros(1, 1, 2), {"recent_rollout": rollout})

    policy._executor = None
    policy.close()


def test_async_process_background_launch_defers_process_submit() -> None:
    policy = _make_async_policy(chunk=3)
    policy.beginning_of_rollout(torch.zeros(1, 1, 2))
    policy._set_current_buffer(torch.zeros(1, 3, 2), torch.tensor([0.0]))
    submitted: list[Any] = []

    class _Submitter:
        def submit(self, fn):
            submitted.append(fn)
            return Future()

        def shutdown(self, *, wait, cancel_futures):
            submitted.append((wait, cancel_futures))

    class _ProcessExecutor:
        def submit(self, *args, **kwargs):
            raise AssertionError("process submit should run in the submitter thread")

    policy._submit_executor = _Submitter()
    policy._executor = _ProcessExecutor()
    launch_info = policy._launch_background_plan(torch.zeros(1, 1, 2), {})

    assert len(submitted) == 1
    assert launch_info["async_launch_started"] is True
    assert launch_info["async_launch_snapshot_sec"] >= 0.0
    assert launch_info["async_launch_submit_sec"] >= 0.0
    assert policy._planning_future is not None
    assert not policy._planning_future.done()
    policy._submit_executor = None
    policy._executor = None
    policy.close()


def test_async_process_background_launch_uses_persistent_worker_payload() -> None:
    policy = _make_async_policy(chunk=3)
    policy.beginning_of_rollout(torch.zeros(1, 1, 2))
    policy.mean.fill_(0.25)
    mean_before = policy.mean.clone()
    policy._set_current_buffer(
        torch.tensor(
            [[[0.1, 0.0], [0.8, 0.0], [0.3, 0.0], [0.2, 0.0]]], dtype=torch.float32
        ),
        torch.tensor([0.0]),
    )
    policy._buffer_index = 1
    submitted_process: list[tuple[Any, ...]] = []

    def fail_snapshot(*args, **kwargs):
        raise AssertionError("foreground launch should not deepcopy a process snapshot")

    class _Submitter:
        def submit(self, fn):
            fn()
            return Future()

        def shutdown(self, *, wait, cancel_futures):
            pass

    class _ProcessExecutor:
        def submit(self, *args):
            submitted_process.append(args)
            return Future()

    policy._make_snapshot_planner = fail_snapshot
    policy._submit_executor = _Submitter()
    policy._executor = _ProcessExecutor()
    launch_info = policy._launch_background_plan(torch.zeros(1, 1, 2), {})

    assert len(submitted_process) == 1
    assert submitted_process[0][0] is mpc_module._background_plan_persistent_worker
    payload = submitted_process[0][3]
    assert payload["model"]["_state"].shape == policy.model._state.shape
    assert payload["mean"].shape == policy.mean.shape
    assert torch.allclose(payload["mean"][0], torch.tensor([0.8, 0.0]))
    assert torch.allclose(policy.mean, mean_before)
    assert launch_info["async_launch_started"] is True
    policy._submit_executor = None
    policy._executor = None
    policy.close()


def test_async_process_background_cancel_before_submit_skips_process_submit() -> None:
    policy = _make_async_policy(chunk=3)
    policy.beginning_of_rollout(torch.zeros(1, 1, 2))
    policy._set_current_buffer(torch.zeros(1, 3, 2), torch.tensor([0.0]))
    submitted: list[Any] = []

    class _Submitter:
        def submit(self, fn):
            submitted.append(fn)
            return Future()

        def shutdown(self, *, wait, cancel_futures):
            pass

    class _ProcessExecutor:
        def submit(self, *args, **kwargs):
            raise AssertionError("canceled proxy should not submit a process job")

    policy._submit_executor = _Submitter()
    policy._executor = _ProcessExecutor()
    policy._launch_background_plan(torch.zeros(1, 1, 2), {})
    policy._cancel_planning_future()

    submitted[0]()

    assert policy._planning_future is None
    policy._submit_executor = None
    policy._executor = None
    policy.close()


def test_async_process_background_cancel_cancels_worker_future() -> None:
    policy = _make_async_policy(chunk=3)
    policy.beginning_of_rollout(torch.zeros(1, 1, 2))
    policy._set_current_buffer(torch.zeros(1, 3, 2), torch.tensor([0.0]))
    submitted: list[Any] = []
    worker_future = Future()

    class _Submitter:
        def submit(self, fn):
            submitted.append(fn)
            return Future()

        def shutdown(self, *, wait, cancel_futures):
            pass

    class _ProcessExecutor:
        def submit(self, *args, **kwargs):
            return worker_future

    policy._submit_executor = _Submitter()
    policy._executor = _ProcessExecutor()
    policy._launch_background_plan(torch.zeros(1, 1, 2), {})
    submitted[0]()
    policy._cancel_planning_future()

    assert worker_future.cancelled()
    assert policy._planning_future is None
    policy._submit_executor = None
    policy._executor = None
    policy.close()


def test_async_process_background_submitter_error_returns_failed_proxy() -> None:
    policy = _make_async_policy(chunk=3)
    policy.beginning_of_rollout(torch.zeros(1, 1, 2))
    policy._set_current_buffer(torch.zeros(1, 3, 2), torch.tensor([0.0]))

    class _Submitter:
        def submit(self, fn):
            del fn
            raise RuntimeError("submitter closed")

        def shutdown(self, *, wait, cancel_futures):
            pass

    class _ProcessExecutor:
        def submit(self, *args, **kwargs):
            raise AssertionError("submitter error should stop process submission")

    policy._submit_executor = _Submitter()
    policy._executor = _ProcessExecutor()
    policy._launch_background_plan(torch.zeros(1, 1, 2), {})

    assert policy._planning_future is not None
    assert policy._planning_future.done()
    with pytest.raises(RuntimeError, match="submitter closed"):
        policy._planning_future.result()
    policy._submit_executor = None
    policy._executor = None
    policy.close()


def test_async_background_launch_predicts_next_chunk_boundary() -> None:
    policy = _make_async_policy(chunk=3, horizon=6)
    policy.beginning_of_rollout(torch.zeros(1, 1, 2))
    policy._set_current_buffer(torch.zeros(1, 6, 2), torch.tensor([0.0]))
    policy._buffer_index = 1
    seen_lengths: list[int] = []

    def fake_rollout(self, state, actions):
        del self, state
        seen_lengths.append(actions.shape[-2])
        return torch.ones(1, 1, 2)

    class _Executor:
        def submit(self, fn, planner, predicted_boundary_state, kwargs):
            del fn, planner, predicted_boundary_state, kwargs
            future = Future()
            future.set_result(
                types.SimpleNamespace(
                    actions=torch.zeros(1, 3, 2),
                    cost=torch.tensor([0.0]),
                    predicted_boundary_state=torch.ones(1, 1, 2),
                    runtime_sec=0.0,
                    model_update_version=0,
                    mean=None,
                    std=None,
                    elite_actions=None,
                    elite_costs_traj=None,
                )
            )
            return future

    policy._rollout_from_state = types.MethodType(fake_rollout, policy)
    policy._executor = _Executor()
    policy._launch_background_plan(torch.zeros(1, 1, 2), {})

    assert seen_lengths == [2]
    policy._executor = None
    policy.close()


def test_async_adaptive_launches_from_chunk_boundary() -> None:
    policy = _make_async_policy(chunk=3, horizon=6)
    policy.adaptive_replanning = True
    policy.beginning_of_rollout(torch.zeros(1, 1, 2))
    policy._set_current_buffer(torch.zeros(1, 6, 2), torch.tensor([0.0]))
    policy._buffer_index = 1
    state = torch.tensor([[[2.0, 3.0]]], dtype=torch.float32)
    boundary = torch.tensor([[[4.0, 5.0]]], dtype=torch.float32)

    def fake_rollout(self, state_arg, actions):
        del self
        assert torch.allclose(state_arg, state)
        assert actions.shape[-2] == 2
        return boundary

    class _Executor:
        def submit(self, fn, planner, predicted_boundary_state, kwargs):
            del fn, planner, kwargs
            assert torch.allclose(predicted_boundary_state, boundary)
            future = Future()
            future.set_result(
                types.SimpleNamespace(
                    actions=torch.zeros(1, 3, 2),
                    cost=torch.tensor([0.0]),
                    predicted_boundary_state=boundary,
                    runtime_sec=0.0,
                    model_update_version=0,
                    mean=None,
                    std=None,
                    elite_actions=None,
                    elite_costs_traj=None,
                )
            )
            return future

    policy._rollout_from_state = types.MethodType(fake_rollout, policy)
    policy._executor = _Executor()
    policy._launch_background_plan(state, {})

    policy._executor = None
    policy.close()


def test_async_background_snapshot_shares_foreground_event() -> None:
    policy = _make_async_policy(chunk=3)
    policy.async_worker_device = torch.device("cpu")
    policy.beginning_of_rollout(torch.zeros(1, 1, 2))

    planner = policy._make_snapshot_planner(torch.zeros(1, 1, 2))
    policy.set_foreground_active(True)

    assert policy._yield_to_foreground is False
    assert planner._yield_to_foreground is True
    assert planner._foreground_active is policy._foreground_active
    assert planner._foreground_active.is_set()
    assert planner.device.type == "cpu"
    assert planner.model._state.device.type == "cpu"

    policy.set_foreground_active(False)
    assert not planner._foreground_active.is_set()
    policy.close()


def test_async_realtime_snapshot_scores_worker_candidates_in_full_batches() -> None:
    policy = _make_async_policy(chunk=3)
    policy.beginning_of_rollout(torch.zeros(1, 1, 2))

    planner = policy._make_snapshot_planner(torch.zeros(1, 1, 2))
    assert getattr(planner, "_yield_score_batch_size", 0) == 0

    planner = policy._make_snapshot_planner(torch.zeros(1, 1, 2))
    assert planner._yield_score_batch_size == 0
    policy.close()


def test_async_snapshot_sampling_uses_local_rngs() -> None:
    policy = _make_async_policy(chunk=3)
    policy.beginning_of_rollout(torch.zeros(1, 1, 2))

    planner_a = policy._make_snapshot_planner(torch.zeros(1, 1, 2))
    planner_b = policy._make_snapshot_planner(torch.zeros(1, 1, 2))

    torch.manual_seed(123)
    expected_torch = torch.randn(3)
    torch.manual_seed(123)
    _ = planner_a.sample_action_sequences(2)
    assert torch.allclose(torch.randn(3), expected_torch)

    planner_a.noise_beta = 1.0
    np.random.seed(123)
    expected_np = np.random.random(3)
    np.random.seed(123)
    _ = planner_a.sample_action_sequences(2)
    assert np.allclose(np.random.random(3), expected_np)

    planner_a.noise_beta = 0.0
    planner_b.noise_beta = 0.0
    assert not torch.allclose(
        planner_a.sample_action_sequences(2),
        planner_b.sample_action_sequences(2),
    )
    policy.close()


def test_sync_sampling_uses_explicit_local_rng() -> None:
    planner_a = _make_coarse_policy(chunk=2, horizon=4, factor=1, seed=17)
    planner_b = _make_coarse_policy(chunk=2, horizon=4, factor=1, seed=17)
    planner_a.noise_beta = 1.0
    planner_b.noise_beta = 1.0
    planner_a.beginning_of_rollout(torch.zeros(1, 1, 2))
    planner_b.beginning_of_rollout(torch.zeros(1, 1, 2))

    np.random.seed(123)
    expected_np = np.random.random(3)
    np.random.seed(123)
    samples_a = planner_a.sample_action_sequences(2)
    assert np.allclose(np.random.random(3), expected_np)
    assert torch.allclose(samples_a, planner_b.sample_action_sequences(2))


def test_async_worker_anchor_refreshes_parameter_belief() -> None:
    policy = _make_async_policy(chunk=3)
    policy.model.e = {"m": torch.zeros(1, 2), "P": torch.eye(2).unsqueeze(0)}
    policy.beginning_of_rollout(torch.zeros(1, 1, 2))
    planner = policy._make_snapshot_planner(torch.zeros(1, 1, 2))

    policy.model.e = {"m": torch.ones(1, 2), "P": 2.0 * torch.eye(2).unsqueeze(0)}
    policy._publish_async_anchor(
        torch.zeros(1, 1, 2),
        {"parameter_update_version": 1},
    )
    kwargs = {"parameter_update_version": 0}
    planner._maybe_refresh_worker_belief(torch.zeros(1, 1, 2), kwargs)

    assert kwargs["parameter_update_version"] == 1
    assert torch.allclose(planner.model.e["m"], torch.ones(1, 2))
    assert torch.allclose(planner.model.e["P"], 2.0 * torch.eye(2).unsqueeze(0))
    assert torch.allclose(planner.model.params, torch.ones(1, 2))
    policy.close()


def test_async_process_worker_refreshes_parameters_from_late_foreground_publish() -> None:
    policy = _make_async_policy(chunk=3)
    policy.model.e = {"m": torch.zeros(1, 2), "P": torch.eye(2).unsqueeze(0)}
    launch_boundary = torch.zeros(1, 1, 2)
    live_boundary = torch.tensor([[[0.6, -0.2]]], dtype=torch.float32)
    policy.beginning_of_rollout(launch_boundary)

    planner = policy._make_snapshot_planner(launch_boundary)
    mpc_module._use_file_system_tensor_sharing()
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_process_anchor_worker, planner, launch_boundary)
        time.sleep(0.05)
        policy.model.e = {"m": torch.ones(1, 2), "P": 2.0 * torch.eye(2).unsqueeze(0)}
        policy._publish_async_anchor(live_boundary, {"parameter_update_version": 1})
        result = future.result(timeout=10)

    assert torch.allclose(result["state"], launch_boundary)
    assert result["parameter_update_version"] == 1
    assert torch.allclose(result["parameter_mean"], torch.ones(1, 2))
    assert torch.allclose(result["parameter_cov"], 2.0 * torch.eye(2).unsqueeze(0))
    assert torch.allclose(result["params"], torch.ones(1, 2))
    policy.close()


def test_async_background_planning_does_not_mutate_live_policy_state() -> None:
    policy = _make_async_policy(chunk=3)
    policy.beginning_of_rollout(torch.zeros(1, 1, 2))
    policy.mean.fill_(0.25)
    policy.std.fill_(0.5)
    mean_before = policy.mean.clone()
    std_before = policy.std.clone()
    dt_before = policy.model.dt
    dyn_dt_before = policy.model.dynamics.dt
    policy._set_current_buffer(torch.zeros(1, 3, 2), torch.tensor([0.0]))

    policy._launch_background_plan(torch.zeros(1, 1, 2), {})
    assert policy._planning_future is not None
    _ = policy._planning_future.result(timeout=5.0)

    assert torch.allclose(policy.mean, mean_before)
    assert torch.allclose(policy.std, std_before)
    assert policy.model.dt == pytest.approx(dt_before)
    assert policy.model.dynamics.dt == pytest.approx(dyn_dt_before)
    policy.close()


def test_async_background_plan_uses_worker_iteration_override() -> None:
    policy = _make_async_policy(chunk=3)
    policy.async_worker_iterations = 1
    policy.async_worker_full_interval = 2
    policy.beginning_of_rollout(torch.zeros(1, 1, 2))
    policy._set_current_buffer(torch.zeros(1, 3, 2), torch.tensor([0.0]))
    seen_iterations: list[int] = []

    class _Submitter:
        def submit(self, fn):
            fn()
            future = Future()
            return future

        def shutdown(self, *, wait, cancel_futures):
            pass

    class _Executor:
        def submit(self, fn, predicted_boundary_state, kwargs, payload):
            del fn, predicted_boundary_state, kwargs
            seen_iterations.append(int(payload["num_iterations"]))
            future = Future()
            future.set_result(
                types.SimpleNamespace(
                    actions=torch.zeros(1, 3, 2),
                    cost=torch.tensor([0.0]),
                    predicted_boundary_state=torch.zeros(1, 1, 2),
                    runtime_sec=0.0,
                    model_update_version=0,
                    mean=None,
                    std=None,
                    elite_actions=None,
                    elite_costs_traj=None,
                )
            )
            return future

    policy._submit_executor = _Submitter()
    policy._executor = _Executor()
    policy._launch_background_plan(torch.zeros(1, 1, 2), {})
    policy._planning_future = None
    policy._launch_background_plan(torch.zeros(1, 1, 2), {})

    assert seen_iterations == [1, policy.num_iterations]
    policy._executor = None
    policy.close()


def test_async_policy_catalog_entry_is_available() -> None:
    spec = get_policy_spec("active_planning_async")
    assert spec.policy_type == "async-mpc-icem"
    assert spec.async_planning is True
    assert spec.async_worker_iterations is None
    assert spec.async_worker_full_interval is None
    assert spec.async_worker_device is None
    try:
        configure_tbme_catalogs()
        realtime = get_policy_spec("adaptive_async_realtime")
        assert realtime.async_worker_iterations == 2
        assert realtime.async_realtime_prefix_steps == 10
        anytime = get_policy_spec("adaptive_async_anytime")
        assert anytime.async_worker_iterations == 2
        assert anytime.async_anytime_prefix_steps is None
        assert anytime.async_anytime_min_iteration == 1
        assert anytime.async_anytime_std_tolerance == pytest.approx(0.75)
    finally:
        configure_catalogs()


def test_exp02_defaults_use_realtime_async() -> None:
    from experiments.tbme.exp02_hardEnv import EXPERIMENT_SUITES

    for suite in EXPERIMENT_SUITES.values():
        model_ids = suite["model_ids"]
        assert "adaptive_async_realtime" in model_ids
        assert "adaptive_async_anytime" in model_ids


def test_tbme_runner_callables_are_pickle_safe() -> None:
    try:
        configure_tbme_catalogs()
        env = get_environment_preset("tbme_asymmetric_basin_hard")
        fe, fz = _build_env_jacobians(env, estimator=True, dynamics_alpha=env.dynamics_alpha)
        fe_roundtrip = pickle.loads(pickle.dumps(fe))
        fz_roundtrip = pickle.loads(pickle.dumps(fz))
        formatter = _EnvParameterFormatter(
            full_params=env.resolved_true_params(estimator=True),
            min_embedding_dim=env.resolved_min_embedding_dim(),
        )
        formatter_roundtrip = pickle.loads(pickle.dumps(formatter))

        z = torch.zeros(1, 1, env.latent_dim)
        e = torch.zeros(1, env.embedding_dim)

        assert torch.allclose(fe(z, e), fe_roundtrip(z, e))
        assert torch.allclose(fz(z, e), fz_roundtrip(z, e))
        assert torch.allclose(formatter(e), formatter_roundtrip(e))
    finally:
        configure_catalogs()
