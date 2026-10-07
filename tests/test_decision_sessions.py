"""Contracts for decision sessions: environment rule, planner rollouts, and filter reset."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from actdyn.core.agent import _apply_session_reset, _model_input, _session_fields
from actdyn.environment.action import IdentityActionEncoder
from actdyn.environment.session import NO_EVIDENCE, SessionRule, session_evidence, session_evidence_pool
from actdyn.environment.spiking_decision import SpikingDecisionEnv
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

    def step(self, action: np.ndarray, *, evidence=None):
        self.t += 1
        gap = self.rate * self.t
        z = np.array([-2.0 + gap, -2.0], dtype=np.float32)
        return np.full(2, float(self.t), dtype=np.float32), z


def test_environment_ends_sessions_after_the_post_decision_delay():
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
            assert info["session_context"] == {"bins": 0, "decided": False, "since": 0, "evidence_pool": -1,
                                               "session_index": len(env.session_log)}
    # The gap first exceeds 2 at bin 7; the reset follows 4 bins later.
    assert [i + 1 for i, e in enumerate(ends) if e] == [11, 22, 33]
    assert [s["decision"] for s in env.session_log] == [1, 1, 1]
    assert [s["decision_bin"] for s in env.session_log] == [7, 7, 7]
    assert network.reset_seeds == [SpikingDecisionEnv.session_seed(7, k) for k in range(4)]


def test_session_evidence_follows_the_pool_and_the_evidence_period():
    rule = SessionRule(evidence_amplitude=0.3, evidence_bins=4)
    bins = torch.tensor([0, 3, 4, 0, 2])
    pool = torch.tensor([1, 0, 0, -1, 0])
    expected = [[0.0, 0.3], [0.3, 0.0], [0.0, 0.0], [0.15, 0.15], [0.3, 0.0]]
    torch.testing.assert_close(session_evidence(bins, pool, rule), torch.tensor(expected))
    # A coarse step of 4 bins starting at bin 2 has the evidence on for half its bins.
    torch.testing.assert_close(session_evidence(torch.tensor([2]), torch.tensor([1]), rule, 4),
                               torch.tensor([[0.0, 0.15]]))
    assert session_evidence(bins, pool, SessionRule()).abs().max() == 0.0
    pools = [session_evidence_pool(SpikingDecisionEnv.session_seed(3, k)) for k in range(40)]
    assert set(pools) == {0, 1}
    assert pools == [session_evidence_pool(SpikingDecisionEnv.session_seed(3, k)) for k in range(40)]


class RecordingNetwork(ScriptedNetwork):
    """ScriptedNetwork that records the drive of every bin."""

    def __init__(self, rate: float) -> None:
        super().__init__(rate)
        self.drives: list[np.ndarray] = []

    def step(self, action: np.ndarray, *, evidence=None):
        self.drives.append(np.asarray(action, dtype=np.float64).copy() + (0 if evidence is None else evidence))
        return super().step(action, evidence=evidence)


def test_environment_adds_the_known_evidence_to_the_drive():
    rule = SessionRule(decision_gap=2.0, post_decision_bins=2, max_session_bins=30, max_sessions=2,
                       evidence_amplitude=0.3, evidence_bins=3)
    network = RecordingNetwork(rate=0.3)
    env = SpikingDecisionEnv(network, session_rule=rule)
    _z, info = env.reset(seed=4)
    pools = [session_evidence_pool(SpikingDecisionEnv.session_seed(4, k)) for k in range(2)]
    assert info["session_context"]["evidence_pool"] == pools[0]
    action = np.array([0.1, -0.2])
    evidences, context_pools, terminated = [], [info["session_context"]["evidence_pool"]], False
    while not terminated:
        _z, _r, terminated, _trunc, info = env.step(action)
        evidences.append(info["evidence"])
        context_pools.append(info["session_context"]["evidence_pool"])
    # Decision at bin 7, reset after bin 9: each session has 9 bins, the first 3 with evidence.
    for k, pool in enumerate(pools):
        on = np.zeros(2)
        on[pool] = 0.3
        for b in range(9):
            np.testing.assert_allclose(evidences[9 * k + b], on if b < 3 else np.zeros(2), atol=1e-7)
            np.testing.assert_allclose(network.drives[9 * k + b], action + evidences[9 * k + b], atol=1e-7)
    assert context_pools[:9] == [pools[0]] * 9 and context_pools[9] == pools[1]
    assert [s["evidence_pool"] for s in env.session_log] == pools


def test_agent_model_input_is_the_action_plus_the_evidence():
    action = torch.tensor([[[0.5, -0.5]]])
    torch.testing.assert_close(_model_input(action, {"evidence": np.array([0.3, 0.0])}),
                               torch.tensor([[[0.8, -0.5]]]))
    assert _model_input(action, {}) is action


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


def test_planner_rollout_stays_in_the_current_session():
    """Agents learn of a reset only when it happens: a rollout never simulates the decision rule."""
    rule = SessionRule(decision_gap=2.0, post_decision_bins=2, max_session_bins=4)
    planner, _ = _session_planner(rule)
    planner.session_context = {"bins": 3, "decided": True, "since": 1, "evidence_pool": -1, "session_index": 0}
    actions = torch.tensor([[1.0, 0.0]]).repeat(1, 6, 1)
    rollout = planner.simulate(None, actions)
    # The gap passes 2 after step 2 and the session time limit is near, but the state keeps going.
    expected = torch.tensor([[-2.0 + t, -2.0] for t in range(1, 7)])
    torch.testing.assert_close(rollout["next_model_state"][0], expected)
    torch.testing.assert_close(rollout["model_state"][0, 1:], expected[:-1])
    torch.testing.assert_close(rollout["env_action"], actions)
    assert rollout.flat.get("session_reset_after") is None


def test_planner_rollout_adds_the_current_session_evidence_at_coarse_steps():
    rule = SessionRule(decision_gap=2.0, post_decision_bins=1, max_session_bins=100,
                       evidence_amplitude=2.0, evidence_bins=125)
    planner, _ = _session_planner(rule)
    planner.session_context = {"bins": 100, "decided": False, "since": 0, "evidence_pool": 0}
    planner._planning_step_bins = 10
    rollout = planner.simulate(None, torch.zeros(1, 5, 2))
    # Steps start at bins 100, 110, 120, 130, 140; the step over bins 120-129 has evidence for half its bins.
    # The gap passes 2 in step 1 and the session would end, but the rollout stays in the session.
    expected_inputs = [[2.0, 0.0], [2.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    torch.testing.assert_close(rollout["env_action"][0], torch.tensor(expected_inputs))
    torch.testing.assert_close(rollout["next_model_state"][0, 4], torch.tensor([3.0, -2.0]))
    planner.session_context = {"bins": 0, "decided": False, "since": 0, "evidence_pool": 0}
    torch.testing.assert_close(planner._known_evidence(3)[0], torch.zeros(3, 2) + torch.tensor([2.0, 0.0]))


def test_planner_session_reset_forces_a_new_plan():
    rule = SessionRule()
    planner, _ = _session_planner(rule)
    planner.action_list = [torch.zeros(1, 2)] * 3
    planner._chunk_step = 1
    planner.on_session_reset(torch.tensor([[[-2.0, -2.0]]]))
    assert planner.action_list == [] and planner._chunk_step == 0
    assert planner._force_replan_next and planner._force_replan_reason == "session_reset"


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


def test_fixed_length_preset_ends_sessions_only_at_the_time_limit():
    """Preset ``tbme_wong_wang_snn_sessions_m2_fixed``: a decision does not end the session."""
    from dataclasses import replace

    from actdyn.environment.spiking_decision import session_rule_from_preset
    from experiments.experiment_definitions import get_environment_preset
    from experiments.tnsre.run_tbme_experiments import configure_tbme_catalogs

    configure_tbme_catalogs(suite_entries={})
    fixed = session_rule_from_preset(get_environment_preset("tbme_wong_wang_snn_sessions_m2_fixed"), 4.0)
    base = session_rule_from_preset(get_environment_preset("tbme_wong_wang_snn_sessions_m2"), 4.0)
    assert replace(fixed, post_decision_bins=base.post_decision_bins) == base  # only the session end differs
    assert fixed.max_session_bins == 400 and fixed.post_decision_bins >= fixed.max_session_bins
    env = SpikingDecisionEnv(ScriptedNetwork(rate=0.3), session_rule=replace(fixed, max_sessions=2))
    env.reset(seed=7)
    ends, terminated = [], False
    while not terminated:
        _z, _r, terminated, _trunc, info = env.step(np.zeros(2))
        ends.append(info["session_end"])
    assert [i + 1 for i, e in enumerate(ends) if e] == [400, 800]  # environment
    assert [s["decision"] for s in env.session_log] == [1, 1]  # decisions are still recorded


def test_observation_presets_change_only_the_recorded_neurons():
    """Low-SNR and biased fixed-trial presets record (4, 4), (2, 2), and (40, 2) neurons; nothing else changes."""
    from dataclasses import replace

    from actdyn.environment.spiking_decision import observed_counts
    from experiments.experiment_definitions import get_environment_preset
    from experiments.tnsre.run_tbme_experiments import configure_tbme_catalogs

    configure_tbme_catalogs(suite_entries={})
    second = (("n10_10", (10, 10)), ("n20_20", (20, 20)), ("n20_5", (20, 5)))
    for protocol in ("_fixed", "", "_evmix", "_fixed_evmix", "_evfirst"):  # fixed, decision-ended, mixed evidence
        base = get_environment_preset(f"tbme_wong_wang_snn_sessions_m2{protocol}")
        assert observed_counts(base) == (40, 40)
        names = {"_fixed": (("lowsnr4", (4, 4)), ("lowsnr2", (2, 2)), ("biased", (40, 2))),
                 "": (("lowsnr4", (4, 4)), ("biased", (40, 2))), "_evmix": (), "_fixed_evmix": (), "_evfirst": ()}[protocol] + second
        for name, counts in names:
            p = get_environment_preset(f"tbme_wong_wang_snn_sessions_m2{protocol}_{name}")
            assert observed_counts(p) == counts and p.observation_dim == sum(counts)
            same = replace(p, preset_id=base.preset_id, system_label=base.system_label, observation_dim=base.observation_dim,
                           spiking_observed_per_pool=base.spiking_observed_per_pool, spiking_observed_pool2=None)
            assert same == base


def test_reactive_paldi_changes_only_the_minimum_intervals():
    from experiments.tnsre.run_tbme_experiments import configure_tbme_catalogs

    bundle = configure_tbme_catalogs(suite_entries={})
    base, fast = bundle.policy_specs["adaptive"], bundle.policy_specs["adaptive_reactive"]
    sb, sf = bundle.schedule_specs[base.schedule_id], bundle.schedule_specs[fast.schedule_id]
    assert (sb.adaptive_update_min_interval, sb.adaptive_replan_min_interval) == (5, 5)
    assert (sf.adaptive_update_min_interval, sf.adaptive_replan_min_interval) == (1, 1)
    assert (sf.planning_horizon, sf.replan_interval, sf.update_interval) == (sb.planning_horizon, sb.replan_interval, sb.update_interval)
    assert fast.objective_kind == base.objective_kind and fast.coarse_dt_factor == base.coarse_dt_factor


def test_paldi_c2_changes_only_the_planning_step_horizon_and_cadence():
    from experiments.tnsre.eval_spiking_sessions import evaluated_policy
    from experiments.tnsre.exp_spiking_sessions import EXPERIMENT_SUITES
    from experiments.tnsre.run_tbme_experiments import configure_tbme_catalogs

    bundle = configure_tbme_catalogs(suite_entries={})
    base, c2 = bundle.policy_specs["adaptive"], bundle.policy_specs["adaptive_c2"]
    sb, sc = bundle.schedule_specs[base.schedule_id], bundle.schedule_specs[c2.schedule_id]
    assert (base.coarse_dt_factor, c2.coarse_dt_factor) == (20, 2)
    assert base.coarse_dt_factor * sb.planning_horizon == c2.coarse_dt_factor * sc.planning_horizon == 800  # 4 s
    assert (sc.adaptive_update_min_interval, sc.adaptive_replan_min_interval) == (3, 3)
    assert (sc.replan_interval, sc.update_interval) == (sb.replan_interval, sb.update_interval)
    assert c2.objective_kind == base.objective_kind
    assert all("adaptive_c2" in s["model_ids"] for k, s in EXPERIMENT_SUITES.items() if "_obs_" not in k)
    assert evaluated_policy("adaptive_c2") == "adaptive" and evaluated_policy("rhc") == "rhc"
    h20 = bundle.policy_specs["adaptive_c2h20"]
    sh = bundle.schedule_specs[h20.schedule_id]
    assert h20.coarse_dt_factor * sh.planning_horizon == 40  # 200 ms
    assert (sh.adaptive_update_min_interval, sh.adaptive_replan_min_interval, sh.replan_interval) == (3, 3, 20)
    assert all("adaptive_c2h20" in s["model_ids"] for k, s in EXPERIMENT_SUITES.items() if "_obs_" not in k)
    assert evaluated_policy("adaptive_c2h20") == "adaptive"


def test_runs_root_and_preset_come_from_the_experiment(tmp_path):
    from experiments.tnsre.eval_spiking_sessions import runs_preset, runs_root_of

    (tmp_path / "tracks" / "wong_wang_snn_sessions_m2_biased").mkdir(parents=True)
    assert runs_root_of(tmp_path).name == "wong_wang_snn_sessions_m2_biased"
    (tmp_path / "tracks" / "other").mkdir()
    with pytest.raises(ValueError):
        runs_root_of(tmp_path)
    assert runs_preset([{"preset_id": "a"}, {"preset_id": "a"}]) == "a"
    with pytest.raises(ValueError):
        runs_preset([{"preset_id": "a"}, {"preset_id": "b"}])


def test_mixed_evidence_preset_starts_the_evidence_at_session_eleven():
    from dataclasses import replace

    from actdyn.environment.spiking_decision import session_rule_from_preset
    from experiments.experiment_definitions import get_environment_preset
    from experiments.tnsre.run_tbme_experiments import configure_tbme_catalogs

    configure_tbme_catalogs(suite_entries={})
    evidence = session_rule_from_preset(get_environment_preset("tbme_wong_wang_snn_sessions_m2_evidence"), 4.0)
    mixed = session_rule_from_preset(get_environment_preset("tbme_wong_wang_snn_sessions_m2_evmix"), 4.0)
    assert mixed.evidence_first_session == 10 and evidence.evidence_first_session == 0
    assert replace(mixed, evidence_first_session=0) == evidence  # same evidence and session end otherwise
    fixed = session_rule_from_preset(get_environment_preset("tbme_wong_wang_snn_sessions_m2_fixed"), 4.0)
    fixed_mixed = session_rule_from_preset(get_environment_preset("tbme_wong_wang_snn_sessions_m2_fixed_evmix"), 4.0)
    assert fixed_mixed == replace(mixed, post_decision_bins=fixed.post_decision_bins)  # fixed 2-s sessions


def test_sessions_before_the_first_evidence_session_have_no_evidence():
    rule = SessionRule(evidence_amplitude=0.3, evidence_bins=50)
    zero = session_evidence(torch.tensor([0, 10]), torch.tensor([NO_EVIDENCE, NO_EVIDENCE]), rule)
    torch.testing.assert_close(zero, torch.zeros(2, 2))
    rule = SessionRule(decision_gap=2.0, post_decision_bins=4, max_session_bins=30, max_sessions=2,
                       evidence_amplitude=0.3, evidence_bins=5, evidence_first_session=1)
    env = SpikingDecisionEnv(ScriptedNetwork(rate=0.3), session_rule=rule)
    env.reset(seed=7)
    assert env.session_context()["evidence_pool"] == NO_EVIDENCE and env.session_context()["session_index"] == 0
    seen = {0: [], 1: []}
    terminated = False
    while not terminated:
        index = env.session_index
        _z, _r, terminated, _trunc, info = env.step(np.zeros(2))
        seen[index].append(np.asarray(info["evidence"]))
    assert np.all(np.stack(seen[0]) == 0.0)
    first = np.stack(seen[1][:5])
    assert np.allclose(first.sum(axis=1), 0.3) and np.all(np.stack(seen[1][5:]) == 0.0)
    assert [s["evidence_pool"] for s in env.session_log][0] == NO_EVIDENCE
    assert env.session_log[1]["evidence_pool"] in (0, 1)


def test_planner_rollout_adds_no_evidence_in_a_session_without_evidence():
    """A rollout in a no-evidence session stays without evidence, also past the session end."""
    rule = SessionRule(decision_gap=2.0, post_decision_bins=1, max_session_bins=100,
                       evidence_amplitude=2.0, evidence_bins=2, evidence_first_session=1)
    planner, _ = _session_planner(rule)
    planner.session_context = {"bins": 0, "decided": False, "since": 0, "evidence_pool": NO_EVIDENCE,
                               "session_index": 0}
    actions = torch.tensor([[1.0, 0.0]]).repeat(1, 5, 1)
    rollout = planner.simulate(None, actions)
    torch.testing.assert_close(rollout["env_action"], actions)
    torch.testing.assert_close(rollout["next_model_state"][0, 4], torch.tensor([3.0, -2.0]))


def test_evidence_sessions_then_fixed_sessions():
    """Evidence (decision-ended) sessions before ``evidence_last_session``; fixed-length sessions from ``fixed_first_session``."""
    from dataclasses import replace

    from actdyn.environment.session import session_post_decision_bins
    from actdyn.environment.spiking_decision import session_rule_from_preset
    from experiments.experiment_definitions import get_environment_preset
    from experiments.tnsre.run_tbme_experiments import configure_tbme_catalogs

    rule = SessionRule(decision_gap=2.0, post_decision_bins=4, max_session_bins=30, max_sessions=2,
                       evidence_amplitude=0.3, evidence_bins=5, evidence_last_session=1, fixed_first_session=1)
    assert [session_post_decision_bins(rule, k) for k in (0, 1)] == [4, 30]
    env = SpikingDecisionEnv(ScriptedNetwork(rate=0.3), session_rule=rule)
    env.reset(seed=7)
    ends, seen, terminated = [], {0: [], 1: []}, False
    while not terminated:
        index = env.session_index
        _z, _r, terminated, _trunc, info = env.step(np.zeros(2))
        ends.append(info["session_end"])
        seen[index].append(np.asarray(info.get("evidence", np.zeros(2))))
    assert [i + 1 for i, e in enumerate(ends) if e] == [11, 41]  # decision at bin 7 + 4; then the full 30 bins
    assert np.allclose(np.stack(seen[0][:5]).sum(axis=1), 0.3) and np.all(np.stack(seen[1]) == 0.0)
    assert env.session_log[0]["evidence_pool"] in (0, 1) and env.session_log[1]["evidence_pool"] == NO_EVIDENCE
    assert [s["decision"] for s in env.session_log] == [1, 1]  # the decision is still recorded

    configure_tbme_catalogs(suite_entries={})
    evidence = session_rule_from_preset(get_environment_preset("tbme_wong_wang_snn_sessions_m2_evidence"), 4.0)
    first = session_rule_from_preset(get_environment_preset("tbme_wong_wang_snn_sessions_m2_evfirst_n20_5"), 4.0)
    assert first == replace(evidence, evidence_last_session=10, fixed_first_session=10)


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


@pytest.mark.parametrize('bound', [.25, .5, 1.])
def test_lower_action_bound_applies_to_fixed_pushes_and_planner_seeds(monkeypatch, bound):
    from experiments.tnsre import eval_spiking_sessions as ev

    monkeypatch.setitem(ev.PROTOCOL, 'action_max', bound)
    dt, window = .05, 400
    for budget in [0., .0037, *[b * bound**2 for b in (5., 10., 12., 15., 25.)]]:
        expected = min(budget, dt * window * bound**2)
        for name in ('front', 'spread'):
            u = np.stack([ev.model_free_input(name, 'overturn', k, 1, budget, dt)
                          for k in range(window + 2)])
            assert np.linalg.norm(u, axis=-1).max() <= bound + 1e-12
            assert dt * np.square(u).sum() == pytest.approx(expected, abs=1e-12)
            assert not u[window:].any()
        seeds = ev.seed_pushes(window, budget, dt, 10)
        assert np.linalg.norm(seeds, axis=-1).max() <= bound + 1e-12
        np.testing.assert_allclose(10 * dt * np.square(seeds).sum(axis=(1, 2)), expected, atol=1e-12)


def test_task_icem_passes_the_reduced_bound_to_candidate_projection(monkeypatch):
    from experiments.tnsre import eval_spiking_sessions as ev
    from actdyn.utils import validation

    monkeypatch.setitem(ev.PROTOCOL, 'action_max', .25)
    calls = []

    def capture(transition, z0, drive, **kw):
        calls.append(kw)
        return {'u': np.zeros((kw['window'], 2))}

    monkeypatch.setattr(validation, 'icem_reversal_plan', capture)
    model = object.__new__(ev.IcemController)
    model.m, model.dt, model.seed = np.zeros(2), .05, 91000
    model.warm_start, model._last_plan = False, None
    model.transition = {'plan': None}
    model.plan('overturn', 120, 0, 1, 0, .75)
    assert calls[0]['action_max'] == .25
    assert np.linalg.norm(calls[0]['seed_candidates'], axis=-1).max() <= .25 + 1e-12


def test_task_loop_rejects_a_controller_that_exceeds_the_reduced_bound(monkeypatch):
    from experiments.tnsre import eval_spiking_sessions as ev

    monkeypatch.setitem(ev.PROTOCOL, 'action_max', .25)
    model = SimpleNamespace(observe=lambda *args: None,
                            plan=lambda *args: np.tile([.3, 0.], (400, 1)))
    with pytest.raises(RuntimeError, match='action bound'):
        ev.run_task_session(ScriptedNetwork(rate=3.), 'overturn', 91000, .75, 'icem', model)


def test_control_dose_accounts_for_actual_neuron_weights():
    from experiments.tnsre.wong_wang_budget import control_costs

    trace = [{'u': [.25, -.25]}, {'u': [0., .5]}]
    weights = np.array([[0., 2.], [.5, 1.5]])
    result = control_costs(trace, weights, .05)
    assert result['peak_action_norm'] == pytest.approx(.5)
    assert result['peak_neuron_current_pa'] == pytest.approx(15.)
    # dt_phys=.005 s; actual currents are [0,10,-2.5,-7.5] then [0,0,5,15] pA.
    assert result['total_squared_current_pa2_s'] == pytest.approx(.005 * (100 + 6.25 + 56.25 + 25 + 225))


def test_task_evaluator_keeps_sensory_evidence_separate_from_control(monkeypatch):
    from experiments.tnsre import eval_spiking_sessions as ev

    class RecordingNetwork:
        dt_latent = .05

        def reset(self, seed):
            self.calls = []
            return np.array([-2., -2.])

        def step(self, action, *, evidence):
            self.calls.append((np.asarray(action), np.asarray(evidence)))
            # First bin decides pool 1; second bin reverses the choice.
            state = np.array([1., -2.]) if len(self.calls) == 1 else np.array([-2., 1.])
            return np.zeros(2), state

    net = RecordingNetwork()
    result = ev.run_task_session(net, 'overturn', 91000, 12., 'front', None)
    assert result['valid'] == 1 and result['success'] == 1
    np.testing.assert_allclose(net.calls[0][0], [0., 0.])
    assert net.calls[0][1].sum() == pytest.approx(.3)
    np.testing.assert_allclose(net.calls[1][0], ev.target_direction(1))
    np.testing.assert_allclose(net.calls[1][1], net.calls[0][1])


def test_full_wong_wang_training_presets_have_no_evidence():
    from experiments.tnsre.wong_wang_full import TRAIN_ARMS, SUITES
    from experiments.tnsre.run_tbme_experiments import configure_tbme_catalogs
    from experiments.experiment_definitions import get_environment_preset

    configure_tbme_catalogs(suite_entries=SUITES)
    for arm in TRAIN_ARMS:
        p = get_environment_preset(f'tbme_ww_full_{arm}')
        assert p.spiking_session_evidence_amplitude == 0
        assert p.spiking_session_evidence_bins == 0
        assert p.spiking_evidence_mode == 'current'
        assert p.spiking_sessions == 20
        assert len(SUITES[arm]['model_ids']) == 6


def test_numerical_controller_failure_keeps_valid_task_in_denominator():
    from experiments.tnsre.wong_wang_full import failed_control_outcome

    trace = [{'u': [0., 0.], 'z': [-2., 1.]}, {'u': [.5, -.5], 'z': [-1., 1.]}]
    result = failed_control_outcome(trace, seed=91000, dt=.05)
    assert result['valid'] == 1 and result['success'] == 0
    assert result['target'] == 0 and result['onset'] == 1
    assert result['energy'] == pytest.approx(.025)
    # A physical reversal already observed in the failed bin remains a success.
    trace[-1]['z'] = [1., -2.]
    result = failed_control_outcome(trace, seed=91000, dt=.05)
    assert result['success'] == 1 and result['outcome_bin'] == 2


def test_predecision_failure_requires_independent_task_validity():
    from experiments.tnsre.wong_wang_full import failed_control_outcome

    trace = [{'u': [0., 0.], 'z': [-2., -2.]}]
    with pytest.raises(ValueError, match='uncontrolled validity'):
        failed_control_outcome(trace, seed=91000, dt=.05)
    result = failed_control_outcome(trace, seed=91000, dt=.05,
                                   baseline={'valid': 1, 'onset': 120, 'target': 0})
    assert result['valid'] == 1 and result['success'] == 0 and result['onset'] == 120
    result = failed_control_outcome(trace, seed=91000, dt=.05,
                                   baseline={'valid': 0, 'onset': -1, 'target': -1})
    assert result['valid'] == 0 and result['success'] == 0
