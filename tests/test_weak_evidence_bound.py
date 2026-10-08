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
    weak, passive = w.SCENARIOS["weak2_e10"], w.SCENARIOS["passive_e20"]
    cells = [(k, b) for k in (0, 1, 5, 10, 15, 20, -1) for b in (5., 10., 12., 15.)
             if w.learned_cell(unit("adaptive", k), b, weak)]
    # 4 earlier checkpoints at 480 pA^2 s and the final model at four budgets: 8 cells x 8 task seeds per run.
    assert cells == [(1, 12.), (5, 12.), (10, 12.), (15, 12.), (20, 5.), (20, 10.), (20, 12.), (20, 15.)]
    assert not any(w.learned_cell(unit(p, 20), 12., weak) for p in ("prior", "reduced_fit", "none", "spread", "front"))
    assert w.learned_cell(unit("off_policy", 20), 12., passive) and not w.learned_cell(unit("adaptive", 20), 12., passive)
    # The 2026-10-07 run folder keeps its name.
    assert w.runs_root(w.Path("r"), weak).name == "ww_weak_evidence_transfer_uniform"


@pytest.mark.parametrize("name, changed", [
    ("weak10_e10", {"spiking_session_action_scale", "spiking_session_action_scale_last_session"}),
    ("weak2_e5", {"spiking_session_action_scale", "spiking_session_action_scale_last_session",
                  "spiking_session_evidence_last_session", "spiking_fixed_first_session"}),
    ("passive_e20", {"spiking_session_evidence_last_session", "spiking_fixed_first_session"}),
])
def test_scenario_presets_change_only_their_protocol(name, changed):
    import dataclasses

    from actdyn.environment.spiking_decision import session_rule_from_preset
    from experiments.experiment_definitions import get_environment_preset
    from experiments.tnsre import snn_weak_evidence as w
    from experiments.tnsre.run_tbme_experiments import configure_tbme_catalogs

    configure_tbme_catalogs(suite_entries={})
    scenario = w.SCENARIOS[name]
    base, preset = get_environment_preset(w.TASK_PRESET), get_environment_preset(scenario.preset)
    a, b = dataclasses.asdict(base), dataclasses.asdict(preset)
    assert {k for k in a if a[k] != b[k]} == {"preset_id", "system_label", *changed}
    rule = session_rule_from_preset(preset, 4.0)
    assert (rule.action_scale, rule.action_scale_last_session) == (scenario.scale, scenario.weak_trials)
    assert rule.evidence_last_session == rule.fixed_first_session == scenario.evidence_trials
    assert rule.max_sessions == 20 and preset.action_max == 1.0
    gain = preset.spiking_stim_gain_pa
    assert gain * rule.evidence_amplitude == pytest.approx(6.0)
    weak_pa = {"weak10_e10": 10.0, "weak2_e5": 2.0, "passive_e20": 20.0}[name]  # passive: no scaled trials
    assert gain * rule.action_scale * preset.action_max == pytest.approx(weak_pa)


def test_scenario_campaign_jobs():
    from experiments.tnsre import snn_weak_evidence as w

    names = ["weak10_e10", "weak2_e5", "passive_e20"]
    jobs = w.campaign_jobs(names, 0, 4, 16)
    train = jobs["identification"]
    # 5 seeds x (6 + 6 + 1) policies; RHC-US first, the passive runs last; one job per key.
    assert len(train) == 65 and len({k for k, _ in train}) == 65
    assert all("rhc" in k for k, _ in train[:10]) and all("off_policy" in k for k, _ in train[-5:])
    assert [k for k, _ in jobs["prediction"]] == [f"{n}_r2" for n in names]
    assert len(jobs["control_and_replay"]) == 3 + 3 * 16
    assert [k for k, _ in jobs["example"]] == ["weak10_e10_examples", "weak2_e5_examples"]
    assert w.campaign_jobs(names, 1, 4, 16)["example"] == []
    for _key, argv in train:
        scenario = w.SCENARIOS[argv[argv.index("--scenario") + 1]]
        assert argv[argv.index("--policy") + 1] in scenario.policies


def _fake_run(path, control: float):
    """A passive run folder: 20 evidence trials of 3 bins each; model input = control + evidence."""
    import json

    import pandas as pd

    log = [{"session": k, "bins": 3, "decision": 1, "decision_bin": 1, "evidence_pool": k % 2} for k in range(20)]
    rows = []
    for k in range(20):
        for t in range(3):
            u = np.array([control, 0.0])
            e = np.zeros(2)
            e[k % 2] = 0.3
            rows.append(dict(step=len(rows), session_index=k, action_x=(u + e)[0], action_v=(u + e)[1],
                             policy_action_x=u[0], policy_action_v=u[1]))
    rule = dict(action_scale=1.0, action_scale_last_session=0, evidence_last_session=20, fixed_first_session=20,
                evidence_bins=200, evidence_amplitude=0.3)
    path.mkdir(parents=True)
    (path / "run_metadata.json").write_text(json.dumps(dict(seed=3, session_rule=rule, session_log=log,
                                                            env_preset_id="tbme_ww_passive_evidence20_transfer_uniform")))
    pd.DataFrame(rows).to_csv(path / "state_action_trace.csv", index=False)
    return path


def test_check_run_requires_zero_control_in_the_passive_scenario(tmp_path):
    from experiments.tnsre import snn_weak_evidence as w

    passive = w.SCENARIOS["passive_e20"]
    record = w.check_run(_fake_run(tmp_path / "zero", 0.0), "off_policy", passive)
    assert record["compliant"] and record["evidence_max_error"] == 0.0 and set(record["phases"]) == {"strong"}
    assert record["phases"]["strong"]["trials"] == "1-20"
    with pytest.raises(AssertionError, match="bound violations"):
        w.check_run(_fake_run(tmp_path / "pushed", 1e-3), "off_policy", passive)
    with pytest.raises(AssertionError):  # the run's protocol must be the scenario's
        w.check_run(_fake_run(tmp_path / "other", 0.0), "off_policy", w.SCENARIOS["weak2_e10"])


def test_control_size_and_session_bins():
    from experiments.tnsre.snn_weak_evidence import control_size, session_bins

    u = np.array([[0.06, -0.08], [0.1, 0.0]])
    np.testing.assert_allclose(control_size(u, "ball"), [0.1, 0.1])
    np.testing.assert_allclose(control_size(u, "box"), [0.08, 0.1])
    np.testing.assert_array_equal(session_bins(np.array([0, 0, 0, 1, 1, 2])), [0, 1, 2, 0, 1, 0])


def test_manuscript_suite_uses_five_weak_evidence_trials():
    from actdyn.environment.spiking_decision import session_rule_from_preset
    from experiments.experiment_definitions import get_environment_preset
    from experiments.tnsre import snn_weak_evidence as w
    from experiments.tnsre.exp_spiking_sessions import DEFAULT_EXP_IDS, EXPERIMENT_SUITES
    from experiments.tnsre.run_tbme_experiments import configure_tbme_catalogs

    main, strong = EXPERIMENT_SUITES["wong_wang_snn_native_uniform"], EXPERIMENT_SUITES["wong_wang_snn_native_uniform_strong"]
    assert DEFAULT_EXP_IDS == ("wong_wang_snn_native_uniform",)
    # The manuscript suite is the weak2_e5 scenario; the 2026-10-07 suite differs only in its preset.
    assert main["env_preset_id"] == w.SCENARIOS["weak2_e5"].preset
    assert strong["env_preset_id"] == "tbme_ww_observation_transfer_uniform"
    assert {**main, "env_preset_id": None} == {**strong, "env_preset_id": None}
    assert main["model_ids"] == list(w.SCENARIOS["weak2_e5"].policies)
    configure_tbme_catalogs(suite_entries={})
    preset = get_environment_preset(main["env_preset_id"])
    rule = session_rule_from_preset(preset, 4.0)
    assert (rule.evidence_last_session, rule.fixed_first_session) == (5, 5)
    assert (rule.action_scale, rule.action_scale_last_session) == (0.1, 5)
    assert preset.spiking_stim_gain_pa * rule.action_scale * preset.action_max == pytest.approx(2.0)


def test_ball_scenario_changes_only_the_baseline_constraint():
    import dataclasses

    from experiments.tnsre import snn_weak_evidence as w

    box, ball = w.SCENARIOS["weak2_e5"], w.SCENARIOS["weak2_e5_ball"]
    assert dataclasses.replace(ball, constraint="box") == box and ball.constraint == "ball"
    specs = {c: w.configure_catalogs(c).policy_specs for c in ("box", "ball")}
    for policy in w.POLICIES:
        a, b = dataclasses.asdict(specs["box"][policy]), dataclasses.asdict(specs["ball"][policy])
        changed = {k for k in a if a[k] != b[k]}
        if policy in ("rhc", "prbs", "random"):
            assert changed == {"action_constraint"}
            assert (a["action_constraint"], b["action_constraint"]) == ("box", "radius")
        else:
            assert not changed, (policy, changed)
    assert [w.geometry_of(p, ball) for p in ("rhc", "prbs", "random")] == ["ball", "sphere", "ball"]
    assert [w.geometry_of(p, box) for p in ("rhc", "prbs", "random")] == ["box", "box", "box"]
    w.configure_catalogs("box")
