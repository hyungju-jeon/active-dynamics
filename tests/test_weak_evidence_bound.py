"""Contracts for the session-dependent identification bound (weak control in evidence trials)."""

from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from actdyn.core.agent import _apply_session_reset
from actdyn.environment.session import SessionRule, session_action_scale, session_evidence_pool
from actdyn.environment.spiking_decision import SpikingDecisionEnv
from actdyn.policy.baseline_prbs import BaselinePRBSPolicy
from actdyn.policy.policy import RandomPolicy

from test_decision_sessions import RecordingNetwork, _session_planner

WEAK = SessionRule(decision_gap=2.0, post_decision_bins=2, max_session_bins=30, max_sessions=4,
                   evidence_amplitude=0.3, evidence_bins=3, evidence_last_session=2,
                   fixed_first_session=2, action_scale=0.1, action_scale_last_session=2)


def _context(k: int) -> dict:
    return {"bins": 0, "decided": False, "since": 0, "evidence_pool": -1, "session_index": k}


def test_scale_applies_before_the_last_weak_session_only():
    rule = SessionRule(action_scale=0.1, action_scale_last_session=10)
    assert [session_action_scale(rule, _context(k)) for k in (0, 9, 10, 19)] == [0.1, 0.1, 1.0, 1.0]
    assert session_action_scale(rule, None) == 0.1  # a run starts in session 0
    assert session_action_scale(None, _context(0)) == 1.0
    assert all(session_action_scale(SessionRule(), _context(k)) == 1.0 for k in range(20))


def test_environment_checks_the_control_but_not_control_plus_evidence():
    network = RecordingNetwork(rate=0.3)
    env = SpikingDecisionEnv(network, session_rule=WEAK)
    env.reset(seed=4)
    pool = session_evidence_pool(SpikingDecisionEnv.session_seed(4, 0))
    action = np.array([0.1, -0.1])
    _z, _r, _t, _tr, info = env.step(action)
    # The weak bound holds for the control; the 0.3 evidence adds on top of it.
    expected = action.copy()
    expected[pool] += 0.3
    np.testing.assert_allclose(network.drives[-1], expected, atol=1e-7)
    np.testing.assert_allclose(info["evidence"][pool], 0.3, atol=1e-7)
    with pytest.raises(ValueError, match="session-0 bound"):
        env.step(np.array([0.2, 0.0]))


def test_environment_restores_the_original_bound_at_the_switch_session():
    network = RecordingNetwork(rate=0.3)
    env = SpikingDecisionEnv(network, session_rule=WEAK)
    env.reset(seed=4)
    while env.session_index < 2:
        env.step(np.array([0.1, 0.0]))
    # Session 2 (third trial): no evidence, full length, original bound 1.
    _z, _r, _t, _tr, info = env.step(np.array([1.0, -1.0]))
    np.testing.assert_allclose(info["evidence"], 0.0)
    np.testing.assert_allclose(network.drives[-1], [1.0, -1.0])
    assert [s["bins"] for s in env.session_log] == [9, 9]
    with pytest.raises(ValueError):
        env.step(np.array([1.01, 0.0]))


@pytest.mark.parametrize("constraint", ["box", "radius"])
def test_planner_samples_and_projects_in_the_session_bound(constraint):
    planner, _ = _session_planner(WEAK)
    planner.action_constraint = constraint
    for k, bound in ((0, 0.1), (2, 1.0)):
        planner.session_context = _context(k)
        planner.mean, planner.std = planner.get_init_mean(), planner.get_init_std()
        torch.testing.assert_close(planner.std, torch.full((6, 2), 0.5 * bound))
        actions = planner.sample_action_sequences(500)
        size = actions.norm(dim=-1) if constraint == "radius" else actions.abs().amax(dim=-1)
        assert size.max() <= bound + 1e-6 and size.max() > 0.9 * bound
        # A warm-start mean outside the bound is projected onto it.
        projected = planner._project_actions(torch.full((1, 6, 2), 5.0))
        target = bound / np.sqrt(2.0) if constraint == "radius" else bound
        torch.testing.assert_close(projected, torch.full((1, 6, 2), target))


def test_planner_search_returns_actions_in_the_session_bound():
    planner, _ = _session_planner(WEAK)
    planner.action_constraint = "radius"
    planner.num_samples, planner.num_elites, planner.num_iterations = 32, 4, 3
    # Cost favours large inputs, so the search pushes the plan onto the bound.
    planner.metric = lambda rollout, **_: -rollout["action"].norm(dim=-1).sum(dim=-1)
    for k, bound in ((0, 0.1), (2, 1.0)):
        planner.session_context = _context(k)
        planner.on_session_reset(None)
        action, _cost = planner._run_icem_search(torch.zeros(1, 1, 2))
        norms = action.norm(dim=-1)
        assert norms.max() <= bound + 1e-6 and norms.min() > 0.8 * bound


def test_session_reset_gives_the_policy_the_new_session_before_it_replans():
    seen = []
    policy = SimpleNamespace(session_context=_context(1))
    policy.on_session_reset = lambda state: seen.append(session_action_scale(WEAK, policy.session_context))
    model = SimpleNamespace(reset_session_state=lambda mean, var: None, get_state=lambda: None)
    info = {"session_reset_state": np.array([-2.0, -2.0]), "session_reset_variance": 0.01,
            "session_context": _context(2)}
    _apply_session_reset(model, policy, info)
    assert seen == [1.0]
    # The planner restarts with the unscaled sampling width.
    planner, _ = _session_planner(WEAK)
    planner.session_context = _context(1)
    planner.mean = planner.get_init_mean()
    _apply_session_reset(model, planner, info)
    torch.testing.assert_close(planner.std, torch.full((6, 2), 0.5))


@pytest.mark.parametrize("radius", [None, 1.0])
def test_passive_policies_scale_the_same_draws(radius):
    space = lambda: gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    for make in (lambda s: RandomPolicy(s, action_radius=radius),
                 lambda s: BaselinePRBSPolicy(s, seed=3, hold_steps=1, action_radius=radius)):
        weak, strong = make(space()), make(space())
        for p in (weak, strong):
            p.session_rule = WEAK
            p.action_space.seed(11)
        weak.session_context, strong.session_context = _context(0), _context(2)
        for _ in range(20):
            u_weak = weak.get_action(None)[0].reshape(-1)
            u_strong = strong.get_action(None)[0].reshape(-1)
            torch.testing.assert_close(u_weak, 0.1 * u_strong)
            size = u_strong.norm() if radius is not None else u_strong.abs().max()
            assert size <= 1.0 + 1e-6


def test_rhc_plans_inside_the_session_box():
    from actdyn.policy.baseline_rhc import RecedingHorizonCuriosityPolicy

    policy = RecedingHorizonCuriosityPolicy(
        action_space=gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32), device="cpu",
        horizon=4, objective="rhc_us", num_features=16, planner_maxiter=60, seed=0)
    policy.session_rule = SessionRule(action_scale=0.1, action_scale_last_session=2)
    for k, bound in ((0, 0.1), (2, 1.0)):
        policy.session_context = _context(k)
        action, _ = policy.get_action(torch.tensor([[[-2.0, -2.0]]]))
        assert action.abs().max() <= bound + 1e-6
        assert action.abs().max() > 0.5 * bound  # uncertainty sampling uses the available input


def test_flex_inputs_lie_on_the_session_sphere():
    from test_flex_safe_policy import _env_preset
    from actdyn.policy.baseline_flex import FLEXPolicy

    policy = FLEXPolicy(action_space=gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
                        model=SimpleNamespace(e={"m": torch.zeros(1, 4)}), env_preset=_env_preset())
    policy.session_rule = SessionRule(action_scale=0.1, action_scale_last_session=2)
    for k, bound in ((0, 0.1), (1, 0.1), (2, 1.0)):
        policy.session_context = _context(k)
        action, _ = policy.get_action(torch.tensor([[[0.3, -0.2]]]))
        np.testing.assert_allclose(float(action.norm()), bound, rtol=1e-5)


def test_weak_preset_changes_only_the_identification_bound():
    import dataclasses

    from actdyn.environment.spiking_decision import session_rule_from_preset
    from experiments.experiment_definitions import get_environment_preset
    from experiments.tnsre.run_tbme_experiments import configure_tbme_catalogs

    configure_tbme_catalogs(suite_entries={})
    base = get_environment_preset("tbme_ww_observation_transfer_uniform")
    weak = get_environment_preset("tbme_ww_weak_evidence_transfer_uniform")
    a, b = dataclasses.asdict(base), dataclasses.asdict(weak)
    assert {k for k in a if a[k] != b[k]} == {"preset_id", "system_label", "spiking_session_action_scale",
                                              "spiking_session_action_scale_last_session"}
    assert weak.action_max == 1.0 and weak.spiking_calibration_amplitude is None  # calibration at amplitude 1
    rule_a, rule_b = session_rule_from_preset(base, 4.0), session_rule_from_preset(weak, 4.0)
    assert dataclasses.replace(rule_b, action_scale=1.0, action_scale_last_session=0) == rule_a
    assert (rule_b.action_scale, rule_b.action_scale_last_session) == (0.1, 10)
    assert rule_b.evidence_last_session == rule_b.fixed_first_session == rule_b.action_scale_last_session
    # PALDI's 2-pA control norm is below the 6-pA evidence; the original bound is 20 pA.
    gain = weak.spiking_stim_gain_pa
    assert gain * rule_b.action_scale * weak.action_max == pytest.approx(2.0)
    assert gain * rule_b.evidence_amplitude == pytest.approx(6.0)


def test_weak_evidence_campaign_grid_and_groups():
    from experiments.tnsre import snn_weak_evidence as w

    assert sorted(s for g in range(4) for s in w.group_seeds(g, 4)) == list(range(20))
    unit = lambda policy, k: {"policy": policy, "checkpoint": k}
    cells = [(k, b) for k in (0, 1, 5, 10, 15, 20, -1) for b in (5., 10., 12., 15.) if w.learned_cell(unit("adaptive", k), b)]
    # 4 earlier checkpoints at 480 pA^2 s and the final model at four budgets: 8 cells x 8 task seeds per run.
    assert cells == [(1, 12.), (5, 12.), (10, 12.), (15, 12.), (20, 5.), (20, 10.), (20, 12.), (20, 15.)]
    assert not any(w.learned_cell(unit(p, 20), 12.) for p in ("prior", "reduced_fit", "none", "spread", "front"))
    assert w.runs_root(w.Path("r")).name == "ww_weak_evidence_transfer_uniform"


def test_control_size_and_session_bins():
    from experiments.tnsre.snn_weak_evidence import control_size, session_bins

    u = np.array([[0.06, -0.08], [0.1, 0.0]])
    np.testing.assert_allclose(control_size(u, "ball"), [0.1, 0.1])
    np.testing.assert_allclose(control_size(u, "box"), [0.08, 0.1])
    np.testing.assert_array_equal(session_bins(np.array([0, 0, 0, 1, 1, 2])), [0, 1, 2, 0, 1, 0])
