from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path
import types

import gymnasium as gym
import pytest
import torch

from actdyn.policy.mpc import AsyncMpcICem, MpcICem
from actdyn.utils.rollout import Rollout
from actdyn.utils.runtime import configure_runtime, ensure_dir
from experiments.experiment_definitions import ScheduleSpec, configure_catalogs, get_policy_spec


def test_configure_runtime_returns_valid_device():
    device = configure_runtime(seed=123, device=None)
    assert device in {"cpu", "cuda", "mps"}


def test_ensure_dir_creates_directory(tmp_path: Path):
    target = tmp_path / "a" / "b" / "c"
    result = ensure_dir(target)
    assert Path(result).exists()
    assert Path(result).is_dir()


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
    async_reanchor_live_state: true
    async_reanchor_tolerance: 0.1
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
        assert get_policy_spec("async_planning").async_reanchor_live_state is True
        assert get_policy_spec("async_planning").async_reanchor_tolerance == pytest.approx(0.1)
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
    *, chunk: int, horizon: int, factor: int, mapping: str = "hold"
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
        async_stale_refine_iterations=2,
    )


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


def test_async_mpc_stale_plan_runs_two_iteration_refinement() -> None:
    policy = _make_async_policy(chunk=3)
    policy.beginning_of_rollout(torch.zeros(1, 1, 2))
    policy._set_current_buffer(torch.zeros(1, 3, 2), torch.tensor([0.0]))
    policy._buffer_index = 3
    policy._launch_background_plan = types.MethodType(lambda self, state, kwargs: None, policy)
    seen_iterations: list[int | None] = []

    def fake_sync(self, state, *, num_iterations=None, **kwargs):
        del state, kwargs
        seen_iterations.append(num_iterations)
        return torch.ones(1, 3, 2) * 0.7, torch.tensor([2.0])

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

    assert seen_iterations == [2]
    assert torch.allclose(action, torch.ones(1, 1, 2) * 0.7)
    assert policy.last_plan_status["async_plan_stale"] is True
    assert policy.last_plan_status["async_refined"] is True
    policy.close()


def test_async_mpc_refines_when_background_model_version_is_stale() -> None:
    policy = _make_async_policy(chunk=3)
    policy.beginning_of_rollout(torch.zeros(1, 1, 2))
    policy._set_current_buffer(torch.zeros(1, 3, 2), torch.tensor([0.0]))
    policy._buffer_index = 3
    policy._launch_background_plan = types.MethodType(lambda self, state, kwargs: None, policy)
    seen_iterations: list[int | None] = []

    def fake_sync(self, state, *, num_iterations=None, **kwargs):
        del state, kwargs
        seen_iterations.append(num_iterations)
        return torch.ones(1, 3, 2) * 0.8, torch.tensor([2.0])

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

    assert seen_iterations == [2]
    assert torch.allclose(action, torch.ones(1, 1, 2) * 0.8)
    assert policy.last_plan_status["async_plan_stale"] is True
    assert policy.last_plan_status["async_model_version_mismatch"] is True
    assert policy.last_plan_status["async_refined"] is True
    policy.close()


def test_async_mpc_can_use_parameter_stale_ready_plan_when_configured() -> None:
    policy = _make_async_policy(chunk=3)
    policy.async_refine_on_parameter_update = False
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


def test_async_mpc_update_keeps_adaptive_state_tracking() -> None:
    policy = _make_async_policy(chunk=3)
    policy.adaptive_replanning = True
    policy.adaptive_replan_min_interval = 1
    policy.adaptive_replan_state_error_threshold = 0.1
    policy.beginning_of_rollout(torch.zeros(1, 1, 2))
    policy._launch_background_plan = types.MethodType(lambda self, state, kwargs: None, policy)
    policy._set_current_buffer(torch.zeros(1, 3, 2), torch.tensor([0.0]))
    _ = policy(torch.zeros(1, 1, 2))

    update_info = policy.update({"next_model_state": torch.tensor([[[1.0, 0.0]]])})

    assert update_info["adaptive_replan_triggered"] is True
    assert update_info["adaptive_replan_reason"] == "state_tracking_error"
    assert policy._force_replan_next is True
    policy.close()


def test_async_mpc_defers_parameter_replans_to_boundary_reconciliation() -> None:
    policy = _make_async_policy(chunk=3)

    policy.request_replan("parameter_update")
    assert policy._force_replan_next is False

    policy.request_replan("state_tracking_error")
    assert policy._force_replan_next is True
    policy.close()


def test_async_mpc_defers_state_replans_when_stale_parameter_plans_are_allowed() -> None:
    policy = _make_async_policy(chunk=3)
    policy.async_refine_on_parameter_update = False

    policy.request_replan("state_tracking_error")
    assert policy._force_replan_next is False
    policy.close()


def test_async_mpc_fast_stale_mode_skips_unused_state_tracking() -> None:
    policy = _make_async_policy(chunk=3)
    policy.async_refine_on_parameter_update = False
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


def test_async_mpc_missing_plan_uses_blocking_fallback() -> None:
    policy = _make_async_policy(chunk=3)
    policy.beginning_of_rollout(torch.zeros(1, 1, 2))
    policy._set_current_buffer(torch.zeros(1, 3, 2), torch.tensor([0.0]))
    policy._buffer_index = 3
    policy._launch_background_plan = types.MethodType(lambda self, state, kwargs: None, policy)
    seen_iterations: list[int | None] = []

    def fake_sync(self, state, *, num_iterations=None, **kwargs):
        del state, kwargs
        seen_iterations.append(num_iterations)
        return torch.ones(1, 3, 2) * 0.9, torch.tensor([3.0])

    policy._sync_plan = types.MethodType(fake_sync, policy)

    action = policy(torch.zeros(1, 1, 2))

    assert seen_iterations == [None]
    assert torch.allclose(action, torch.ones(1, 1, 2) * 0.9)
    assert policy.last_plan_status["async_blocking_fallback"] is True
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


def test_async_background_launch_does_not_pass_rollout_to_worker() -> None:
    policy = _make_async_policy(chunk=3)
    rollout = Rollout(device="cpu")
    rollout.add(model_state=torch.zeros(1, 2))
    policy.beginning_of_rollout(torch.zeros(1, 1, 2))
    policy._set_current_buffer(torch.zeros(1, 3, 2), torch.tensor([0.0]))

    class _Executor:
        def submit(self, fn, planner, predicted_boundary_state, kwargs):
            del fn, planner, predicted_boundary_state
            assert "recent_rollout" not in kwargs
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


def test_async_adaptive_fast_launches_from_current_state() -> None:
    policy = _make_async_policy(chunk=3, horizon=6)
    policy.adaptive_replanning = True
    policy.async_refine_on_parameter_update = False
    policy.beginning_of_rollout(torch.zeros(1, 1, 2))
    policy._set_current_buffer(torch.zeros(1, 6, 2), torch.tensor([0.0]))
    policy._buffer_index = 1
    state = torch.tensor([[[2.0, 3.0]]], dtype=torch.float32)

    def fail_rollout(self, state, actions):
        del self, state, actions
        raise AssertionError("fast adaptive async should launch from current state")

    class _Executor:
        def submit(self, fn, planner, predicted_boundary_state, kwargs):
            del fn, planner, kwargs
            assert torch.allclose(predicted_boundary_state, state)
            future = Future()
            future.set_result(
                types.SimpleNamespace(
                    actions=torch.zeros(1, 3, 2),
                    cost=torch.tensor([0.0]),
                    predicted_boundary_state=state,
                    runtime_sec=0.0,
                    model_update_version=0,
                    mean=None,
                    std=None,
                    elite_actions=None,
                    elite_costs_traj=None,
                )
            )
            return future

    policy._rollout_from_state = types.MethodType(fail_rollout, policy)
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


def test_async_background_worker_reanchors_to_live_boundary() -> None:
    policy = _make_async_policy(chunk=3)
    policy.async_reanchor_live_state = True
    policy.async_reanchor_tolerance = 0.0
    launch_boundary = torch.zeros(1, 1, 2)
    live_boundary = torch.tensor([[[0.8, -0.4]]], dtype=torch.float32)
    policy.beginning_of_rollout(launch_boundary)
    planner = policy._make_snapshot_planner(launch_boundary)

    policy._publish_async_anchor(live_boundary, {})
    result = AsyncMpcICem._background_plan_worker(planner, launch_boundary, {})

    assert result.reanchor_count == 1
    assert torch.allclose(result.predicted_boundary_state, live_boundary)
    assert torch.allclose(planner.model._state, live_boundary)
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

    class _Executor:
        def submit(self, fn, planner, predicted_boundary_state, kwargs):
            del predicted_boundary_state, kwargs
            seen_iterations.append(planner.num_iterations)
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
    assert spec.async_refine_on_parameter_update is True
    assert spec.async_worker_iterations is None
    assert spec.async_worker_full_interval is None
    assert spec.async_worker_device is None
