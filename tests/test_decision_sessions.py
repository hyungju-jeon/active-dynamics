"""Contracts for decision sessions: environment rule, planner rollouts, EIG, and filter reset."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from actdyn.core.agent import _apply_session_reset, _session_fields
from actdyn.environment.action import IdentityActionEncoder
from actdyn.environment.session import (
    SessionRule,
    advance_session_clock,
    new_session_clock,
    session_clock_from_context,
)
from actdyn.environment.spiking_decision import SpikingDecisionEnv
from actdyn.metrics.information import EmbeddingFisherMetric
from actdyn.models.model import FilteringEmbedding
from actdyn.policy.mpc import MpcICem


class ScriptedNetwork:
    """Stand-in for WangDecisionNetwork: the gap z1 - z2 grows by ``rate`` per bin."""

    dt_latent = 0.05
    state_scale = 4.0
    n_observed = 2
    seed = 5

    def __init__(self, rate: float) -> None:
        self.rate = float(rate)
        self.t = 0
        self.reset_seeds: list[int] = []

    def latent_proxy(self) -> np.ndarray:
        return np.array([-2.0, -2.0], dtype=np.float32)

    def reset(self, seed: int) -> np.ndarray:
        self.reset_seeds.append(int(seed))
        self.t = 0
        return self.latent_proxy()

    def step(self, action: np.ndarray):
        self.t += 1
        gap = self.rate * self.t
        z = np.array([-2.0 + gap, -2.0], dtype=np.float32)
        return np.full(2, float(self.t), dtype=np.float32), z


def _run_clock(gaps, rule, step_bins=1):
    clock = new_session_clock(1)
    resets, decisions = [], []
    for g in gaps:
        clock, decided, reset = advance_session_clock(clock, torch.tensor([float(g)]), rule, step_bins)
        resets.append(bool(reset[0]))
        decisions.append(bool(decided[0]))
    return decisions, resets


def test_clock_resets_after_the_post_decision_delay_and_at_the_time_limit():
    rule = SessionRule(decision_gap=1.0, post_decision_bins=3, max_session_bins=10)
    decisions, resets = _run_clock([0.0, 2.0, 2.0, 0.0, -2.0, 0.0], rule)
    assert decisions == [False, True, False, False, False, False]  # later crossings do not count
    assert resets == [False, False, False, False, True, False]
    _, resets = _run_clock([0.0] * 12, rule)
    assert resets.index(True) == 9  # 10 bins without a decision
    _, resets = _run_clock([0.0] * 3, rule, step_bins=5)
    assert resets == [False, True, False]  # coarse steps of 5 bins


def test_clock_from_context_continues_the_environment_session():
    rule = SessionRule(decision_gap=1.0, post_decision_bins=3, max_session_bins=10)
    clock = session_clock_from_context({"bins": 4, "decided": True, "since": 2}, 3)
    assert clock["bins"].tolist() == [4, 4, 4] and clock["decided"].all()
    _, _, reset = advance_session_clock(clock, torch.zeros(3), rule)
    assert reset.all()
    fresh = session_clock_from_context(None, 2)
    assert fresh["bins"].tolist() == [0, 0] and not fresh["decided"].any()


def test_environment_and_planner_clock_end_sessions_at_the_same_bins():
    rule = SessionRule(decision_gap=2.0, post_decision_bins=4, max_session_bins=30, max_sessions=3)
    network = ScriptedNetwork(rate=0.3)
    env = SpikingDecisionEnv(network, session_rule=rule)
    env.reset(seed=7)
    ends, gaps = [], []
    terminated = False
    while not terminated:
        _z, _r, terminated, _trunc, info = env.step(np.zeros(2))
        ends.append(info["session_end"])
        gaps.append(float(network.rate * network.t) if not info["session_end"] else None)
        if info["session_end"]:
            np.testing.assert_allclose(info["latent_state"], rule.reset_state)
            np.testing.assert_allclose(info["session_pre_reset_state"], [-2.0 + 0.3 * 11, -2.0], rtol=1e-6)
            assert info["session_context"] == {"bins": 0, "decided": False, "since": 0}
    # The gap first exceeds 2 at bin 7; the reset follows 4 bins later.
    assert [i + 1 for i, e in enumerate(ends) if e] == [11, 22, 33]
    _, clock_resets = _run_clock([0.3 * (t + 1) for t in range(11)], rule)
    assert clock_resets.index(True) == 10
    assert [s["decision"] for s in env.session_log] == [1, 1, 1]
    assert [s["decision_bin"] for s in env.session_log] == [7, 7, 7]
    assert network.reset_seeds == [SpikingDecisionEnv.session_seed(7, k) for k in range(4)]


def test_environment_without_sessions_never_resets_itself():
    network = ScriptedNetwork(rate=0.3)
    env = SpikingDecisionEnv(network)
    env.reset(seed=1)
    for _ in range(50):
        _z, _r, terminated, _trunc, info = env.step(np.zeros(2))
        assert not terminated and "session_end" not in info
    assert network.reset_seeds == [1]


class AdditiveDynamics:
    """z_{t+1} = z_t + u_t (unit step), for exact rollout checks."""

    dt = 1.0
    logvar = torch.zeros(1, 2)

    def sample_forward(self, init_z, action, k_step, add_noise=False, return_traj=True):
        states, z = [], init_z
        for t in range(k_step):
            z = z + action[:, t : t + 1]
            states.append(z)
        return None, states, None


def _session_planner(rule):
    encoder = IdentityActionEncoder(
        d_action=2, d_latent=2, action_dim=2, latent_dim=2, action_bounds=[-1.0, 1.0], device="cpu"
    )
    model = SimpleNamespace(
        action_encoder=encoder, dynamics=AdditiveDynamics(), dt=1.0,
        _state=torch.tensor([[[-2.0, -2.0]]]), z={"m": torch.tensor([[[-2.0, -2.0]]])},
    )
    metric = SimpleNamespace()
    planner = MpcICem(metric=metric, model=model, horizon=6, num_samples=4, device="cpu", seed=0)
    planner.session_rule = rule
    return planner, metric


def test_planner_rollout_resets_the_state_and_marks_the_step():
    rule = SessionRule(decision_gap=2.0, post_decision_bins=2, max_session_bins=100, reset_variance=0.02)
    planner, metric = _session_planner(rule)
    assert metric.session_reset_variance == pytest.approx(0.02)
    actions = torch.tensor([[1.0, 0.0]]).repeat(1, 6, 1)
    rollout = planner.simulate(None, actions)
    start, pred = rollout["model_state"][0], rollout["next_model_state"][0]
    mask = rollout["session_reset_after"][0, :, 0]
    # Gap 3 > 2 after step 2; reset 2 bins later, after step 4.
    assert mask.tolist() == [0, 0, 0, 0, 1, 0]
    torch.testing.assert_close(pred[4], torch.tensor([3.0, -2.0]))  # the step's observation sees z before the reset
    torch.testing.assert_close(start[5], torch.tensor([-2.0, -2.0]))
    torch.testing.assert_close(pred[5], torch.tensor([-1.0, -2.0]))


def test_planner_rollout_uses_the_session_context_and_coarse_bins():
    rule = SessionRule(decision_gap=2.0, post_decision_bins=40, max_session_bins=400)
    planner, _ = _session_planner(rule)
    planner.session_context = {"bins": 100, "decided": True, "since": 25}
    planner._planning_step_bins = 10
    rollout = planner.simulate(None, torch.zeros(1, 6, 2))
    assert rollout["session_reset_after"][0, :, 0].tolist() == [0, 1, 0, 0, 0, 0]  # 25 + 10 + 10 >= 40
    planner.session_rule = None
    planner.model.predict = lambda a: planner.model._state + torch.cumsum(a, dim=-2)
    plain = planner.simulate(None, torch.zeros(1, 6, 2))
    assert plain.flat.get("session_reset_after") is None


def test_planner_session_reset_forces_a_new_plan():
    rule = SessionRule()
    planner, _ = _session_planner(rule)
    planner.action_list = [torch.zeros(1, 2)] * 3
    planner._chunk_step = 1
    planner.on_session_reset(torch.tensor([[[-2.0, -2.0]]]))
    assert planner.action_list == [] and planner._chunk_step == 0
    assert planner._force_replan_next and planner._force_replan_reason == "session_reset"


class LinearGaussianDecoder:
    def __init__(self, h, r):
        self.h, self.r, self.noise = h, r, object()

    def __call__(self, z):
        return z @ self.h.to(z).T

    def jacobian(self, z):
        return self.h.to(z).expand(*z.shape[:-1], *self.h.shape)

    def var(self, z):
        return self.r.diagonal().to(z).expand(*z.shape[:-1], self.r.shape[-1])


def _information_problem(p0):
    a = torch.tensor([[0.1, 0.3], [-0.2, -0.1]])
    b = torch.tensor([[1.0, 0.2], [0.1, 0.5]])
    model = SimpleNamespace(
        e={"m": torch.zeros(1, 2), "P": torch.eye(2).unsqueeze(0)},
        z={"m": torch.zeros(1, 1, 2), "P": p0.unsqueeze(0)},
        dynamics=SimpleNamespace(logvar=torch.log(torch.expm1(torch.tensor([[0.05, 0.02]])))),
        decoder=LinearGaussianDecoder(torch.tensor([[1.0, 0.2], [0.0, 1.0]]), torch.diag(torch.tensor([0.5, 0.8]))),
        dt=1.0,
    )

    def fe(z, e):
        return z[..., :1, None] * b.to(z)

    def fz(z, e):
        return a.to(z).expand(*z.shape[:-1], 2, 2)

    return model, fe, fz


@pytest.mark.parametrize("planning_rollout", ["prediction_only", "measurement_conditioned"])
def test_information_after_a_reset_adds_as_a_fresh_start(planning_rollout):
    torch.manual_seed(0)
    z = torch.randn(1, 7, 2) + 1.0
    p0 = torch.tensor([[0.9, 0.1], [0.1, 0.6]])
    reset_after = torch.zeros(1, 7, 1)
    reset_after[0, 3, 0] = 1.0

    def information(model, fe, fz, states, mask=None):
        metric = EmbeddingFisherMetric(model, Fe_net=fe, Fz_net=fz, device="cpu", gamma=1.0,
                                       planning_rollout=planning_rollout, session_reset_variance=0.03)
        rollout = {"model_state": states, "next_model_state": states.clone()}
        if mask is not None:
            rollout["session_reset_after"] = mask
        return metric._rollout_parameter_information(rollout, fe, fz)

    model, fe, fz = _information_problem(p0)
    joint = information(model, fe, fz, z, reset_after)
    prefix = information(model, fe, fz, z[:, :4])
    fresh_model, fe, fz = _information_problem(0.03 * torch.eye(2))
    suffix = information(fresh_model, fe, fz, z[:, 4:])
    torch.testing.assert_close(joint, prefix + suffix)
    no_reset = information(model, fe, fz, z)
    assert not torch.allclose(joint, no_reset)


def test_filter_reset_keeps_the_parameter_belief_and_clears_the_sensitivity():
    stub = SimpleNamespace(
        e={"m": torch.tensor([[1.2, 0.8]]), "P": torch.eye(2).unsqueeze(0)},
        latent_dim=2, device=torch.device("cpu"),
        _theta_sensitivity=torch.ones(1, 2, 2), _state=None, z=None,
    )
    FilteringEmbedding.reset_session_state(stub, np.array([-2.0, -2.0], dtype=np.float32), 0.01)
    torch.testing.assert_close(stub.z["m"], torch.tensor([[[-2.0, -2.0]]]))
    torch.testing.assert_close(stub.z["P"], 0.01 * torch.eye(2).reshape(1, 1, 2, 2))
    torch.testing.assert_close(stub._state, stub.z["m"])
    assert not stub._theta_sensitivity.any()
    torch.testing.assert_close(stub.e["m"], torch.tensor([[1.2, 0.8]]))


def test_agent_session_hook_resets_the_filter_then_the_policy():
    calls = []
    model = SimpleNamespace(
        reset_session_state=lambda mean, var: calls.append(("model", tuple(mean), var)),
        get_state=lambda: "reset-state",
    )
    policy = SimpleNamespace(on_session_reset=lambda state: calls.append(("policy", state)))
    info = {"session_index": 2, "session_end": True, "decision": 1,
            "session_reset_state": np.array([-2.0, -2.0]), "session_reset_variance": 0.01}
    assert _session_fields(info) == {"session_index": 2, "session_end": True, "session_decision": 1}
    _apply_session_reset(model, policy, info)
    assert calls == [("model", (-2.0, -2.0), 0.01), ("policy", "reset-state")]
    assert _session_fields({}) == {"session_index": 0, "session_end": False, "session_decision": 0}


def test_poisson_filter_step_without_readout_is_the_model_prediction():
    from actdyn.utils.validation import poisson_ekf_update
    from experiments.tnsre.eval_spiking_sessions import learner_transition

    m, P = np.array([-1.5, -1.8]), np.diag([0.02, 0.03])
    theta = np.array([1.444, 0.586, -1.278, 2.447, -1.599])
    step = learner_transition("wong_wang_inside_gain", theta, theta, 5, 0.05, torch.float64)
    kw = dict(transition=step, readout_bias=np.zeros(3), dt=0.05, process_var=0.1)
    m0, P0 = poisson_ekf_update(m, P, np.zeros(3), np.array([0.3, 0.0]), readout_weight=np.zeros((3, 2)), **kw)
    m1, P1 = poisson_ekf_update(m, P, np.array([5.0, 0.0, 0.0]), np.array([0.3, 0.0]),
                                readout_weight=np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]), **kw)
    np.testing.assert_allclose(m0, step(torch.tensor(m), torch.tensor([0.3, 0.0])).numpy(), atol=1e-6)
    assert m0[0] > m[0]  # the drive pushes pool 1 up
    assert P0.trace() > P.trace()  # prediction only adds uncertainty
    assert m1[0] > m0[0] and P1[0, 0] < P0[0, 0]  # many pool-1 spikes raise z1 and shrink its variance
    np.testing.assert_allclose(P0, P0.T)


def test_task_controllers_respect_the_budget_and_target_the_other_pool():
    from experiments.tnsre import eval_spiking_sessions as ev

    assert ev.evidence_pool(90003) == ev.evidence_pool(90003) in (0, 1)
    np.testing.assert_allclose(ev.target_direction(0), [1 / np.sqrt(2), -1 / np.sqrt(2)])
    np.testing.assert_allclose(ev.evidence_drive(0, 1), [0.0, 0.3])
    np.testing.assert_allclose(ev.evidence_drive(200, 1), [0.0, 0.0])
    dt = 0.05
    for task in ev.TASKS:
        window = ev.PROTOCOL["window"][task]
        for budget in ev.PROTOCOL["budgets"][task]:
            for name in ("spread", "front"):
                u = np.stack([ev.model_free_input(name, task, k, 1, budget, dt) for k in range(window + 5)])
                assert dt * np.sum(u * u) <= budget + 1e-9
                assert np.all(np.abs(u) <= 1.0 + 1e-12) and np.all(u[:, 1] >= 0) and np.all(u[:, 0] <= 0)
                assert not u[window:].any()


def test_task_jobs_pair_every_controller_on_the_same_sessions():
    from experiments.tnsre import eval_spiking_sessions as ev

    run = {"policy": "adaptive", "seed": 3, "est": {k: np.full(5, float(k)) for k in (0, 1, 2, 5, 20)},
           "C": np.zeros((4, 2)), "b": np.zeros(4), "dynamics_type": "wong_wang_inside_gain",
           "full_params": np.zeros(5), "min_embedding_dim": 5}
    units = ev.controller_units([run], np.arange(5.0))
    names = [(u["policy"], u["checkpoint"]) for u in units]
    assert names == [("adaptive", 1), ("adaptive", 2), ("adaptive", 5), ("adaptive", 20), ("prior", 0),
                     ("reduced_fit", -1), ("none", -1), ("spread", -1), ("front", -1)]
    jobs = ev.task_jobs(units)
    sessions = {(u, task, b): sorted(s for uu, tt, bb, s in jobs if (uu, tt, bb) == (u, task, b))
                for u, task, b, _ in jobs}
    assert len(set(map(tuple, sessions.values()))) == 2  # one session set per task
    none_idx = names.index(("none", -1))
    assert {b for u, t, b, _ in jobs if u == none_idx} == {ev.PROTOCOL["budgets"]["force"][0],
                                                          ev.PROTOCOL["budgets"]["overturn"][0]}


def test_first_passage_statistics_follow_the_task_rules():
    from actdyn.utils.validation import update_decision_statistic

    # Target lead over time: other pool decides first (path 0), target reaches 2.5 then falls
    # back (path 1), target decides first and the other pool later (path 2), nobody decides (path 3).
    leads = torch.tensor([[-2.5, 3.0, 3.0], [1.0, 2.5, -1.0], [2.5, -3.0, -3.0], [0.5, -1.0, 1.5]])
    for objective, expected in (("reach", [3.0, 2.5, 2.5, 1.5]), ("first_decision", [-np.inf, 2.5, 2.5, 1.5])):
        best = torch.full((4,), -float("inf"))
        decided = torch.zeros(4, dtype=torch.bool)
        for t in range(3):
            best, decided = update_decision_statistic(best, decided, leads[:, t], objective=objective, threshold=2.0)
        np.testing.assert_array_equal(best.numpy(), expected)
    with pytest.raises(ValueError):
        update_decision_statistic(best, decided, leads[:, 0], objective="terminal", threshold=2.0)


def test_seed_pushes_spend_at_most_the_budget_toward_pool_two():
    from experiments.tnsre import eval_spiking_sessions as ev

    for window_left, budget in ((200, 0.5), (200, 1.0), (400, 12.0), (120, 12.0), (40, 0.07)):
        pushes = ev.seed_pushes(window_left, budget, 0.05, 10)
        assert pushes.shape == (6, window_left // 10, 2)
        energy = 0.5 * np.sum(pushes ** 2, axis=(1, 2))  # coarse step = 10 bins of dt 0.05
        assert np.all(energy <= budget + 1e-9) and np.all(np.abs(pushes) <= 1.0 + 1e-12)
        assert np.all(pushes[..., 1] >= 0) and np.all(pushes[..., 0] <= 0)
        np.testing.assert_allclose(energy[0], min(budget, 0.5 * (window_left // 10)))  # full push: all it can


def test_carry_over_plan_returns_the_rest_in_planning_coordinates():
    from experiments.tnsre.eval_spiking_sessions import carry_over_plan

    fine = np.stack([np.arange(400.0), -np.arange(400.0)], axis=1)
    rest = carry_over_plan((40, fine[40:]), 80, target=1, factor=10)  # plan made at bin 40, replan at 80
    assert rest.shape == (32, 2)
    np.testing.assert_array_equal(rest[0], fine[80])
    np.testing.assert_array_equal(rest[1], fine[90])
    np.testing.assert_array_equal(carry_over_plan((40, fine[40:]), 80, target=0, factor=10)[0], fine[80, ::-1])


def test_warm_controller_paths_keep_the_fresh_results():
    from pathlib import Path

    from experiments.tnsre.eval_spiking_sessions import controller_paths

    fresh, warm = controller_paths(Path("e"), "fresh"), controller_paths(Path("e"), "warm")
    assert fresh == {"task": Path("e/task"), "summary": Path("e/task_summary.csv")}
    assert warm == {"task": Path("e/task_warm"), "summary": Path("e/task_summary_warm.csv")}
