"""Causal feedback, pathwise constraints, and immutable-plan compatibility."""
from copy import deepcopy
import numpy as np
import pytest
import torch

from experiments.tnsre import eval_spiking_sessions as ev
from experiments.tnsre.wong_wang_control import ControlTask, rollout_statistics, run_session
from experiments.tnsre.wong_wang_feedback import (
    FeedbackTrackingController, feedback_input, feedback_rollout_statistics,
    feedback_tracking_plan,
)


def test_feedback_cancels_past_error_but_cannot_cancel_future_noise():
    steps, samples, seed = 20, 32, 4
    noise_var = .1
    args = dict(dt=1., budget=100., bound=10., gain=1/np.sqrt(2.),
                seed=seed, samples=samples, noise_var=noise_var)
    result = feedback_rollout_statistics(lambda z, u: z+u, np.zeros(2),
        np.zeros((steps, 2)), np.zeros((1, steps, 2)), np.ones(steps), np.ones(steps), **args)
    rng = torch.Generator().manual_seed(seed)
    noise = torch.randn(steps, samples, 2, generator=rng)*np.sqrt(noise_var)
    # After feedback, gap_t = reference_t + noise_t[1] - noise_t[0].
    exact = torch.mean((noise[..., 1]-noise[..., 0])**2).item()
    np.testing.assert_allclose(result['mse'], exact, rtol=2e-6)
    noiseless = feedback_rollout_statistics(lambda z, u: z+u, np.zeros(2),
        np.zeros((steps, 2)), np.zeros((1, steps, 2)), np.ones(steps), np.ones(steps),
        **(args | dict(noise_var=0.)))
    np.testing.assert_allclose(noiseless['mse'], 0., atol=1e-12)
    np.testing.assert_allclose(noiseless['energy'], .5, atol=1e-6)


@pytest.mark.parametrize('gain', [0., .5, 2., 8.])
@pytest.mark.parametrize('budget', [0., .017, 1.])
def test_every_feedback_rollout_obeys_peak_and_energy_limits(gain, budget):
    args = dict(dt=.05, budget=budget, bound=.25, gain=gain, seed=17, samples=8)
    result = feedback_rollout_statistics(lambda z, u: z+.05*u,
        np.array([-2., 2.]), np.zeros((20, 2)), np.ones((3, 20, 2))*5.,
        np.zeros(20), np.ones(20), **args)
    assert result['energy'].min() >= -1e-7
    assert result['energy'].max() <= budget+1e-7
    assert result['peak'].max() <= .25+1e-7
    if budget == 0:
        np.testing.assert_allclose(result['energy'], 0.)
        np.testing.assert_allclose(result['peak'], 0.)


def test_zero_gain_matches_existing_rollouts_for_feasible_inputs():
    transition = lambda z, u: z+.05*(torch.tanh(z+u)-.4*z)
    actions = np.random.default_rng(4).normal(size=(3, 20, 2))*.05
    arguments = (transition, np.array([-.5, .3]), np.zeros((20, 2)), actions)
    before = rollout_statistics(*arguments, dt=.05, seed=7, samples=8,
        reference=np.ones(20), weights=np.ones(20))
    after = feedback_rollout_statistics(*arguments, np.ones(20), np.ones(20),
        dt=.05, budget=1., bound=1., gain=0., seed=7, samples=8)
    np.testing.assert_allclose(after['mse'], before['mse'], rtol=1e-7, atol=1e-7)


def test_input_projection_matches_closed_form_and_pool_exchange():
    expected = np.array([-.2, .2])/np.sqrt(2.)
    u = feedback_input(torch.zeros(2, dtype=torch.float64),
        torch.tensor([1., -1.], dtype=torch.float64), 0., .002,
        gain=2., bound=1., dt=.05)
    np.testing.assert_allclose(u, expected, atol=1e-12)
    ctrl = object.__new__(FeedbackTrackingController)
    ctrl.dt, ctrl.feedback_gain, ctrl.m = .05, 2., np.array([.2, -.3])
    task = ControlTask(name='moderate')
    first = ctrl.apply_input(np.array([.1, -.2]), task=task, k=0, target=1, remaining=.3)
    ctrl.m = ctrl.m[::-1].copy()
    swapped = ctrl.apply_input(np.array([-.2, .1]), task=task, k=0, target=0, remaining=.3)
    np.testing.assert_allclose(first, swapped[::-1], atol=1e-12)


def test_feedback_optimizer_retains_exact_zero_feedforward_policy():
    task = ControlTask(name='moderate', window=20, warmup=0, replan=10, budget=2.)
    args = dict(task=task, dt=.05, budget=2., gain=1/np.sqrt(2.), seed=3,
        action_seed=7, seed_candidates=np.zeros((1, 2, 2)),
        settings=dict(noise_var=0., mc_samples=2, num_samples=8, num_elites=2, num_iterations=2))
    result = feedback_tracking_plan(lambda z, u: z+u, np.zeros(2),
        np.zeros((20, 2)), np.ones(20), np.ones(20), **args)
    np.testing.assert_allclose(result['u'], 0., atol=1e-7)
    assert result['score'] == pytest.approx(0., abs=1e-12)
    assert result['feedforward_energy'] == pytest.approx(0.)
    again = feedback_tracking_plan(lambda z, u: z+u, np.zeros(2),
        np.zeros((20, 2)), np.ones(20), np.ones(20), **args)
    np.testing.assert_allclose(result['u'], again['u'])


def test_feedback_applies_between_replans_without_snn_truth(monkeypatch):
    monkeypatch.setattr(ev, 'PROTOCOL', dict(ev.PROTOCOL))

    class Network:
        dt_latent = .05

        def reset(self, seed):
            return np.array([-2., -2.])

        def step(self, action, *, evidence):
            # Constant truth; the fake observer below changes only from its own update.
            return np.zeros(2), np.array([1., -2.])

    class Controller(FeedbackTrackingController):
        def __init__(self):
            self.dt, self.feedback_gain, self.m = .05, .25, np.zeros(2)
            self.replans = 0

        def observe(self, counts, drive):
            self.m += np.array([0., .1])

        def plan_tracking(self, task, t, k, target, pool, budget_left):
            self.replans += 1
            return np.zeros((task.window-k, 2))

    task = ControlTask(name='neutral', window=20, replan=10, warmup=0)
    trace, ctrl = [], Controller()
    result = run_session(Network(), task, 3, controller=ctrl, trace=trace)
    actions = np.array([step['u'] for step in trace[1:]])
    assert ctrl.replans == 2 and len(actions) == 20
    assert len(np.unique(actions[:, 1])) == 20
    assert actions[1, 1] < actions[0, 1] < 0
    np.testing.assert_allclose(result['energy'], .05*np.sum(actions**2), atol=1e-12)


def test_plan_variant_changes_only_learned_jobs_and_retains_original_keys():
    from experiments.tnsre.wong_wang_control_study import planned_jobs
    units = [dict(policy='adaptive', seed=0, theta=[1.]),
             dict(policy='prior', seed=0, theta=[1.], C=[[1., 0.]], b=[0.]),
             dict(policy='front', seed=-1, theta=None)]
    plan = dict(tasks=[dict(name='neutral')], training_seeds=[0], task_seeds=[3],
        policies=['adaptive', 'prior', 'front'], baselines=[dict(label='pi_tuned',
            controller='pi', parameters=dict(kp=2., ki=0.))])
    original = planned_jobs(plan, units)
    configured = deepcopy(plan)
    configured['learned_controller'] = dict(name='feedback_tracking', gain=2.)
    changed = planned_jobs(configured, units)
    for before, after in zip(original, changed):
        if before['unit']['theta'] is None:
            assert before == after
        else:
            assert before['parameters'] == {}
            assert after['parameters'] == {'feedback_gain': 2.}
            assert before['key'] != after['key']
    assert original == planned_jobs(plan, units)
    configured['tasks'] = [dict(name='overturn')]
    with pytest.raises(ValueError, match='tracking task'):
        planned_jobs(configured, units)


@pytest.mark.parametrize('gain', [-1., float('nan'), float('inf')])
def test_nonfinite_or_negative_gain_is_rejected_before_model_initialization(gain):
    with pytest.raises(ValueError, match='Feedback gain'):
        FeedbackTrackingController(feedback_gain=gain)
