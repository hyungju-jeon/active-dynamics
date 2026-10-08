"""Contracts for the matched updated-manuscript Wong–Wang comparison."""
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.tnsre import wong_wang_observation as study
from experiments.tnsre import eval_spiking_sessions as ev
from experiments.tnsre.run_tbme_experiments import configure_tbme_catalogs
from experiments.experiment_definitions import get_environment_preset,get_policy_spec,get_schedule_spec
from actdyn.environment.spiking_decision import SpikingDecisionEnv,session_rule_from_preset


@pytest.mark.parametrize('arm',study.ARMS)
def test_real_flex_dependency_generates_finite_bounded_controls(arm):
    import gymnasium as gym
    import torch
    from actdyn.policy.baseline_flex import FLEXPolicy
    configure_tbme_catalogs(suite_entries=study.SUITES)
    p=get_environment_preset(f'tbme_ww_observation_{arm}')
    policy=FLEXPolicy(
        action_space=gym.spaces.Box(-1.,1.,shape=(2,),dtype=np.float32),
        model=SimpleNamespace(),env_preset=p,
        initial_parameter_mean=torch.tensor(p.initial_parameter_mean),
        rollback_unstable_update=True,
    )
    policy.reset_policy_state(seed=0)
    action=policy._flex_agent.policy(np.array([-2.,-2.]),0)
    assert action.shape==(2,) and np.isfinite(action).all()
    assert np.linalg.norm(action)==pytest.approx(p.action_max,rel=1e-5)


def test_updated_presets_change_observation_and_trial_protocol_without_changing_inputs():
    configure_tbme_catalogs(suite_entries=study.SUITES)
    for arm in study.ARMS:
        p=get_environment_preset(f'tbme_ww_observation_{arm}')
        old=get_environment_preset(f'tbme_ww_full_{arm}')
        assert (p.observation_dim,p.spiking_observed_per_pool)==(40,20)
        assert (old.observation_dim,old.spiking_observed_per_pool)==(80,40)
        for attr in ['dynamics_type','estimator_dynamics_type','initial_parameter_mean','state_noise',
                     'spiking_stim_fraction','spiking_stim_weight_cv','spiking_stim_mask_seed',
                     'spiking_evidence_mode','action_max']:
            assert getattr(p,attr)==getattr(old,attr)
        rule=session_rule_from_preset(p,4.)
        assert (rule.evidence_amplitude,rule.evidence_bins,rule.evidence_last_session,rule.fixed_first_session)==(.3,200,10,10)


def test_updated_paldi_preserves_the_old_policy_and_sets_the_manuscript_horizon():
    configure_tbme_catalogs(suite_entries=study.SUITES)
    old=get_policy_spec('adaptive');new=get_policy_spec('adaptive_c2h20')
    so=get_schedule_spec(old.schedule_id);sn=get_schedule_spec(new.schedule_id)
    assert (so.planning_horizon,old.coarse_dt_factor)==(40,20)
    assert (sn.planning_horizon,new.coarse_dt_factor)==(20,2)
    assert (sn.adaptive_update_min_interval,sn.adaptive_replan_min_interval)==(3,3)


def test_twenty_trials_change_evidence_and_duration_at_the_same_boundary():
    from test_decision_sessions import ScriptedNetwork  # tests/ is on sys.path; a vendored package is also named "tests"
    configure_tbme_catalogs(suite_entries=study.SUITES)
    p=get_environment_preset('tbme_ww_observation_transfer_partial')
    net=ScriptedNetwork(rate=3.)
    env=SpikingDecisionEnv(net,session_rule=session_rule_from_preset(p,4.))
    env.reset(seed=0)
    seen={}
    for i in range(8000):
        _,_,terminated,_,info=env.step(np.zeros(2))
        if info['session_end']:
            log=env.session_log[-1];seen[log['session']]=log
        if terminated:break
    assert len(seen)==20
    assert all(seen[k]['bins']==41 and seen[k]['evidence_pool'] in (0,1) for k in range(10))
    assert all(seen[k]['bins']==400 and seen[k]['evidence_pool']==-2 for k in range(10,20))


def test_evaluation_selects_the_updated_recording_and_keeps_both_control_budgets(tmp_path,monkeypatch):
    from copy import deepcopy
    monkeypatch.setattr(ev,'PROTOCOL',deepcopy(ev.PROTOCOL))
    monkeypatch.setattr(ev,'TASKS',ev.TASKS)
    monkeypatch.setattr(ev,'PRESET_ID',ev.PRESET_ID)
    for arm in study.ARMS:
        tracks=study.configure(tmp_path,arm)
        assert tracks==tmp_path/'tracks'/f'ww_observation_{arm}'
        assert ev.PRESET_ID==f'tbme_ww_observation_{arm}'
        assert ev.PROTOCOL['task_checkpoints']==[0,20]
        assert ev.PROTOCOL['budgets']['overturn']==[10.,12.]
        assert ev.TASKS==('overturn',)


def test_network_default_uses_the_configured_arm(tmp_path, monkeypatch):
    from copy import deepcopy
    from actdyn.environment import spiking_decision
    monkeypatch.setattr(ev, 'PROTOCOL', deepcopy(ev.PROTOCOL))
    monkeypatch.setattr(ev, 'TASKS', ev.TASKS)
    monkeypatch.setattr(ev, 'PRESET_ID', ev.PRESET_ID)
    monkeypatch.setattr(spiking_decision, 'WangDecisionNetwork', lambda **kw: kw)
    study.configure(tmp_path, 'transfer_partial')
    settings = ev.build_network()
    assert settings['observed_per_pool'] == (20, 20)
    assert settings['stim_fraction'] == .5
    explicit = ev.build_network(preset_id='tbme_ww_observation_transfer_uniform')
    assert explicit['observed_per_pool'] == (20, 20)
    assert explicit['stim_fraction'] == 1.
