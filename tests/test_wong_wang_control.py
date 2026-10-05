"""Control-study mathematical and task contracts; no full SNN simulations."""
from dataclasses import replace
import numpy as np
import pytest
import torch

from experiments.tnsre import eval_spiking_sessions as ev
from experiments.tnsre.wong_wang_control import (
    ControlTask, CountObserver, baseline_action, rollout_statistics, run_session,
    tracking_metrics, tracking_plan,
)


def test_tracking_scores_all_bins_after_warmup_and_penalizes_overshoot():
    task = ControlTask(name='moderate', window=20, warmup=5, replan=10)
    exact = np.ones(20)
    exact[:5] = -2.
    assert tracking_metrics(exact, task)['rmse'] == 0
    assert tracking_metrics(exact, task)['success'] == 1
    transient = exact.copy()
    transient[10:] = 2.
    score = tracking_metrics(transient, task)
    assert score['success'] == 0
    np.testing.assert_allclose(score['rmse'], np.sqrt(10/15))


def test_rollout_statistics_matches_exact_linear_solution():
    transition = lambda z, u: z+u
    u = np.array([[[0., .25]]*4, [[0., .5]]*4])
    result = rollout_statistics(transition, np.zeros(2), np.zeros((4, 2)), u,
        dt=1., seed=5, noise_var=0., reference=np.ones(4), weights=np.ones(4))
    expected = np.mean((np.array([[.25, .5, .75, 1.], [.5, 1., 1.5, 2.]])-1.)**2, axis=1)
    np.testing.assert_allclose(result['mse'], expected)
    reach = rollout_statistics(transition, np.zeros(2), np.zeros((4, 2)), u,
                               dt=1., seed=5, noise_var=0., threshold=1.5)
    np.testing.assert_allclose(reach['hard'], [0., 1.])


def test_tracking_optimizer_retains_exact_seed_and_is_deterministic():
    task = ControlTask(name='moderate', window=20, replan=10, coarse=10, warmup=0, budget=1.)
    # A memoryless plant permits exactly known zero-error control.
    transition = lambda z, u: u
    exact = np.array([[-.5, .5], [-.5, .5]])
    kwargs = dict(task=task, dt=.05, budget=1., seed=7, action_seed=8,
        seed_candidates=exact[None], settings=dict(noise_var=0., mc_samples=2,
        num_samples=8, num_elites=2, num_iterations=2))
    a = tracking_plan(transition, np.zeros(2), np.zeros((20, 2)), np.ones(20), np.ones(20), **kwargs)
    b = tracking_plan(transition, np.zeros(2), np.zeros((20, 2)), np.ones(20), np.ones(20), **kwargs)
    np.testing.assert_allclose(a['u'], np.repeat(exact, 10, axis=0), atol=1e-7)
    np.testing.assert_allclose(a['u'], b['u'])
    assert a['score'] == pytest.approx(0.)
    assert a['energy'] <= 1.


@pytest.mark.parametrize('name', ['none', 'front', 'spread', 'pulse', 'relay', 'pi'])
@pytest.mark.parametrize('bound', [.25, .5, 1.])
def test_every_baseline_respects_peak_and_energy_limits(name, bound):
    task = ControlTask(name='switch', bound=bound, budget=.071, window=20, replan=10, warmup=0)
    energy, integral = 0., 0.
    for k in range(task.window):
        u, integral = baseline_action(name, task=task, k=k, target=1, dt=.05,
            remaining=task.budget-energy, estimate=np.array([0., -2.]), integral=integral)
        assert np.linalg.norm(u) <= bound+1e-12
        energy += .05*float(u@u)
        assert energy <= task.budget+1e-12


def test_count_observer_uses_poisson_rates_in_model_time_units():
    C = np.array([[1., 0.], [0., 1.], [.5, .5]])
    b = np.full(3, 3.)
    z = np.array([-.5, .5])
    observer = CountObserver(C, b, .05)
    counts = .05*np.exp(C@z+b)
    for _ in range(200):
        observer.observe(counts)
    np.testing.assert_allclose(observer.m, z, atol=1e-6)


class SequenceNetwork:
    dt_latent = .05

    def reset(self, seed):
        self.k = 0
        return np.array([-2., -2.])

    def step(self, action, *, evidence):
        self.k += 1
        # First pool wins immediately; opposite pool then leads by 3.
        return np.zeros(2), np.array([1., -2.]) if self.k == 1 else np.array([-2., 1.])


def test_tracking_never_stops_at_a_single_threshold_crossing(monkeypatch):
    monkeypatch.setattr(ev, 'PROTOCOL', dict(ev.PROTOCOL))
    task = ControlTask(name='moderate', window=20, warmup=0, replan=10)
    net = SequenceNetwork()
    result = run_session(net, task, 3)
    assert net.k == 21
    assert result['success'] == 0
    assert result['rmse'] == pytest.approx(2.)
    result = run_session(net, replace(task, name='overturn'), 3)
    assert net.k == 2 and result['success'] == 1


@pytest.mark.parametrize('failure_step', [1, 2])
def test_observer_failure_retains_the_applied_step_and_energy(monkeypatch, failure_step):
    from experiments.tnsre.wong_wang_control_study import failure_outcome
    monkeypatch.setattr(ev, 'PROTOCOL', dict(ev.PROTOCOL))

    class FailingObserver:
        m = np.zeros(2)
        observations = 0

        def observe(self, counts, drive):
            self.observations += 1
            if self.observations == failure_step:
                raise np.linalg.LinAlgError('forced observer failure')

    net, trace = SequenceNetwork(), []
    task = ControlTask(name='neutral', budget=1., window=20, warmup=0, replan=10)
    with pytest.raises(np.linalg.LinAlgError, match='forced observer failure'):
        run_session(net, task, 3, controller=FailingObserver(), baseline='front', trace=trace)
    assert net.k == len(trace) == failure_step
    result = failure_outcome(net, task, 3, trace)
    assert result['valid'] == 1 and result['onset'] == 1 and result['target'] == 1
    assert result['success'] == 0 and result['rmse'] == 5.
    assert result['energy'] == pytest.approx((failure_step-1)*net.dt_latent)


def test_front_and_spread_match_existing_overturn_inputs(monkeypatch):
    monkeypatch.setattr(ev, 'PROTOCOL', dict(ev.PROTOCOL, action_max=.75))
    task = ControlTask(bound=.75, budget=10.)
    for name in ['front', 'spread']:
        used = 0.
        for k in range(400):
            u, _ = baseline_action(name, task=task, k=k, target=0, dt=.05, remaining=task.budget-used)
            expected = ev.model_free_input(name, 'overturn', k, 0, task.budget, .05)
            np.testing.assert_allclose(u, expected, atol=1e-7)
            used += .05*float(u@u)


def test_diagnostic_prediction_matches_the_planners_reported_score():
    from actdyn.utils.validation import icem_reversal_plan
    transition = lambda z, u: z + .05*(torch.tanh(z+u)-.4*z)
    drive = np.full((30, 2), .05)
    z0 = np.array([-.5, .3])
    result = icem_reversal_plan(transition, z0, drive, dt=.05, window=20,
        coarse_factor=10, budget=.31, action_max=.75, noise_var=.1,
        objective='reach', success_gap=2., mc_samples=8, mc_seed=17,
        action_seed=19, num_samples=8, num_elites=2, num_iterations=2)
    prediction = rollout_statistics(transition, z0, drive, result['u'][None],
        dt=.05, seed=17, samples=8, noise_var=.1)
    np.testing.assert_allclose(prediction['soft'][0], result['soft_probability'], rtol=1e-6)
    np.testing.assert_allclose(prediction['hard'][0], result['probability'], atol=1e-7)


def test_paired_comparison_broadcasts_pulse_without_inventing_replications():
    import pandas as pd
    from experiments.tnsre.wong_wang_control_report import paired_matrix
    rows = pd.DataFrame([dict(policy='front', seed=-1, task_seed=40, success=0),
                         dict(policy='front', seed=-1, task_seed=41, success=1)])
    matrix = paired_matrix(rows, 'front', [0, 1, 2], [40, 41], 'success')
    np.testing.assert_allclose(matrix, [[0, 1]]*3)
    with pytest.raises(ValueError, match='Incomplete paired'):
        paired_matrix(rows, 'front', [0, 1], [40, 41, 42], 'success')


def test_simultaneous_intervals_detect_a_variable_paired_tracking_advantage():
    import pandas as pd
    from experiments.tnsre.wong_wang_control_report import simultaneous_comparisons
    rows = []
    for seed in range(4):
        for task_seed in range(20):
            for policy, error in [('adaptive', .4+.01*seed+.005*task_seed),
                                  ('rhc', 1.+.02*seed+.01*task_seed)]:
                rows.append(dict(arm='test', condition='moderate_a1_b10', task='moderate',
                    valid=1, policy=policy, seed=seed, task_seed=task_seed, rmse=error))
    plan = dict(training_seeds=list(range(4)), task_seeds=list(range(20)))
    result = simultaneous_comparisons(pd.DataFrame(rows), plan, draws=1000)
    assert len(result) == 1
    assert result.iloc[0].advantage > .6
    assert result.iloc[0].lower > 0
    assert result.iloc[0].advantage_established == 1


def test_tuned_parameters_apply_only_to_their_declared_arm_and_condition():
    from experiments.tnsre.wong_wang_control_study import planned_jobs
    plan = dict(tasks=[dict(name='moderate', budget=10.), dict(name='moderate', budget=14.)],
        training_seeds=[0], task_seeds=[12], policies=[], baselines=[dict(
            label='pulse_tuned', controller='pulse', parameters=dict(amplitude=.75),
            arms=['transfer_partial_poisson'], conditions=['moderate_a1_b10'])])
    assert planned_jobs(plan, [], 'transfer_uniform_current') == []
    jobs = planned_jobs(plan, [], 'transfer_partial_poisson')
    assert len(jobs) == 1
    assert jobs[0]['task']['budget'] == 10.
    assert jobs[0]['parameters']['amplitude'] == .75
