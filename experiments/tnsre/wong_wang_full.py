"""Full manuscript comparison of response shape, current targeting, and sensory spikes.

Training: 20 seeds x six policies x 20 decision trials, without sensory evidence.
Four training arms cross response family and targeting. Each model is evaluated
under current and Poisson evidence; the same training run serves both tests.
All stages run as separate processes so Brian2 and catalog globals are isolated.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from experiments.tnsre import eval_spiking_sessions as ev
from experiments.tnsre.exp_spiking_sessions import MODEL_IDS
from experiments.tnsre.run_tbme_experiments import configure_tbme_catalogs

TRAIN_ARMS = tuple(f'{family}_{target}' for family in ('logistic', 'transfer')
                   for target in ('uniform', 'partial'))
INPUT_ARMS = ('uniform_current', 'partial_current', 'uniform_poisson', 'partial_poisson')
EVAL_ARMS = tuple(f'{arm}_{mode}' for arm in TRAIN_ARMS for mode in ('current', 'poisson'))
SUITES = {arm: {'experiment_kind': 'parameter', 'total_steps': 8000,
               'model_ids': MODEL_IDS, 'env_preset_id': f'tbme_ww_full_{arm}',
               'trajectory_eval_interval': 0, 'trajectory_eval_horizon': 200,
               'trajectory_eval_samples': 100} for arm in TRAIN_ARMS}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def configure_eval(root, arm):
    family, target, mode = arm.split('_')
    preset = ('tbme_wong_wang_snn_sessions_m2' if family == 'logistic' else 'tbme_wong_wang_transfer')
    if target == 'partial':
        preset = f'tbme_wong_wang_{family}_partial'
    if mode == 'poisson':
        suffix = 'both' if target == 'partial' else 'synaptic'
        preset = f'tbme_wong_wang_{family}_{suffix}'
    ev.PRESET_ID = preset
    ev.TASKS = ('overturn',)
    ev.PROTOCOL['test_trajectories'] = str(root / 'data' / f'{target}_current' / 'test_trajectories.npz')
    ev.PROTOCOL['reduced_fit'] = str(root / 'reference' / f'{arm}.json')
    return root / 'tracks' / f'ww_full_{family}_{target}'


def train(root, arm, policy, seed, *, total_steps=8000):
    from experiments import run
    configure_tbme_catalogs(suite_entries=SUITES)
    args = run.build_parser().parse_args(['--base-dir', str(root), '--path-layout', 'tbme_tracks',
                                        '--total-steps', str(total_steps), '--skip-existing'])
    result = run._run_one(exp_id=arm, policy_id=policy, seed=seed, repeat=1, base_dir=root, args=args)
    if result['status'] != 'completed' or result.get('nan_detected'):
        raise RuntimeError(f'Identification failed: {result.get("error", result)}')
    if total_steps == 8000 and len(result.get('session_log', [])) != 20:
        raise RuntimeError('Identification did not finish all 20 decision trials')


def data(root, input_arm):
    """Disjoint reference-training and manuscript test trajectories, float32 proxies.

    Reference data: 24 piecewise drives and 24 task-like drives. Task-like
    controls are 200 ms signed pushes in alternate 200 ms blocks. Their random
    schedule is fixed here because the historical fitting script was removed.
    """
    configure_eval(root, f'logistic_{input_arm}')
    out = root / 'data' / input_arm
    out.mkdir(parents=True, exist_ok=True)
    if input_arm.endswith('current'):
        ev.stage_test_data(out)
    net = ev.build_network()
    states, inputs, seeds = [], [], []
    for seed in list(range(76000, 76024)) + list(range(77000, 77024)):
        controls = ev.test_input_program(seed)
        evidence = np.zeros_like(controls)
        if seed >= 77000:
            rng = np.random.default_rng(seed)
            controls[:] = 0
            for start in range(0, 400, 80):
                direction = ev.target_direction(int(rng.integers(2)))
                controls[start:start + 40] = rng.choice([.5, 1.]) * direction
            evidence[:200, ev.evidence_pool(seed)] = .3
        z = [net.reset(seed)]
        for u, e in zip(controls, evidence):
            z.append(net.step(u, evidence=e)[1])
        states.append(z)
        inputs.append(controls + evidence)
        seeds.append(seed)
    np.savez(out / 'reference_train.npz', states=states, inputs=inputs, seeds=seeds)
    write_json(out / 'input_diagnostics.json', {
        'targeted_neurons': (net.stim_weights > 0).sum(axis=1).tolist(),
        'weight_sum': net.stim_weights.sum(axis=1).tolist(),
        'peak_current_pa_at_unit_input': (20 * net.stim_weights.max(axis=1)).tolist(),
        'squared_current_ratio_by_pool': (net.stim_weights ** 2).mean(axis=1).tolist(),
        'evidence_rate_hz': .3 * net.evidence_rate_gain_hz,
        'reference_voltage_mv': -55., 'test_seeds': list(range(70000, 70012)),
        'reference_train_seeds': seeds,
    })
    net.close()


def reference(root, arm):
    from experiments.tnsre.wong_wang_input_pilot import windows, fit
    from actdyn.utils.validation import rollout_r2_on_trajectories
    family, target, mode = arm.split('_')
    d = np.load(root / 'data' / f'{target}_{mode}' / 'reference_train.npz')
    batch = windows({'states': d['states'], 'controls': d['inputs'], 'evidence': np.zeros_like(d['inputs'])},
                    horizon=100, stride=20)
    result = fit('logistic' if family == 'logistic' else 'wong_wang_transfer', batch,
                 starts=16, iterations=300, seed=88000, lr=.05)
    test = np.load(root / 'data' / f'{target}_current' / 'test_trajectories.npz')
    dyn = 'wong_wang_inside_gain' if family == 'logistic' else 'wong_wang_transfer'
    r2 = rollout_r2_on_trajectories(torch.tensor([result['theta']]), dynamics_type=dyn,
         full_params=result['theta'], min_embedding_dim=5, states=test['states'], inputs=test['inputs'],
         dt=.05, horizon=100, stride=20)
    write_json(root / 'reference' / f'{arm}.json', {
        'learner_parameters': {'values': result['theta']}, 'fit': result,
        'r2_test_piecewise_current': float(r2[0]), 'train_seeds': d['seeds'].tolist(),
        'fit_dtype': 'float64', 'selection': 'training MSE only',
        'warning': 'Privileged latent fit. Reconstructed task-like training schedule, not the historical fit.',
    })


class RecordedTaskNetwork:
    """Record a recovery attempt without changing network inputs or random draws."""

    def __init__(self, network):
        self.network = network
        self.dt_latent = network.dt_latent
        self.trace = []

    def reset(self, seed):
        self.trace = []
        return self.network.reset(seed)

    def step(self, action, *, evidence):
        counts, z = self.network.step(action, evidence=evidence)
        self.trace.append({'u': np.asarray(action).tolist(), 'evidence': np.asarray(evidence).tolist(),
                           'z': np.asarray(z).tolist()})
        return counts, z


def failed_control_outcome(trace, *, seed, dt, baseline=None):
    """Score observed SNN outcomes before a numerical controller exception.

    A valid task that ends before reaching its goal is unsuccessful. If the
    filter fails before the initial decision, an uncontrolled replay supplies
    task validity; it does not supply a successful controller outcome.
    """
    gap = float(ev.PROTOCOL['decision_gap'])
    decision = next((i for i, row in enumerate(trace[:int(ev.PROTOCOL['max_session_bins'])])
                     if abs(row['z'][0] - row['z'][1]) > gap), None)
    if decision is None:
        if baseline is None:
            raise ValueError('Need uncontrolled validity after a pre-decision controller failure')
        valid, onset, target = (int(baseline[k]) for k in ('valid', 'onset', 'target'))
        outcome = -1
    else:
        valid, onset = 1, decision + 1
        z = trace[decision]['z']
        target = 1 if z[0] > z[1] else 0
        outcome = next((i + 1 for i in range(onset, len(trace))
                        if trace[i]['z'][target] - trace[i]['z'][1-target] > gap), -1)
    return {'valid': valid, 'evidence_pool': ev.evidence_pool(seed), 'target': target,
            'onset': onset, 'success': int(outcome >= 0),
            'energy': float(dt * sum(np.dot(row['u'], row['u']) for row in trace)),
            'outcome_bin': outcome, 'first_decision': 0}


def task(root, arm, shard, nshards, *, recover_numerical_errors=False):
    """Write one deterministic shard; keep model-free and warm-start jobs together."""
    runs_root = configure_eval(root, arm)
    runs = ev.load_session_runs(runs_root)
    assert len(runs) == 120 and all(set(range(21)) <= set(r['est']) for r in runs)
    units = ev.controller_units(runs)
    jobs = ev.task_jobs(units)[shard::nshards]
    out = root / 'eval' / arm
    suffix = f'recovery_{shard:03d}' if recover_numerical_errors else f'{shard:03d}'
    path = out / 'task_warm' / f'worker_{suffix}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    sources = sorted(path.parent.glob('worker_*.csv')) if recover_numerical_errors else [path]
    for source in sources:
        if source.exists():
            with source.open() as f:
                done.update((r['controller'], r['policy'], int(r['seed']), int(r['checkpoint']), r['task'],
                             float(r['budget']), int(r['task_seed'])) for r in csv.DictReader(f))
    net = ev.build_network()
    recorded = RecordedTaskNetwork(net) if recover_numerical_errors else net
    new = not path.exists()
    with path.open('a', newline='') as f:
        fields = ev.TASK_FIELDS + (['numerical_failure', 'failure_record'] if recover_numerical_errors else [])
        writer = csv.DictWriter(f, fieldnames=fields)
        if new:
            writer.writeheader()
        for idx, name, budget, seed in jobs:
            unit = units[idx]
            if ev._job_key(unit, name, budget, seed) in done:
                continue
            model = None
            if unit['theta'] is not None:
                model = ev.IcemController(unit['theta'], unit['C'], unit['b'], float(net.dt_latent),
                    seed=seed, dynamics_type=unit['dynamics_type'], full_params=unit['full_params'],
                    min_embedding_dim=unit['min_embedding_dim'], warm_start=True)
            failure_fields = {'numerical_failure': 0, 'failure_record': ''} if recover_numerical_errors else {}
            try:
                res = ev.run_task_session(recorded, name, seed, budget, unit['controller'], model)
            except torch.linalg.LinAlgError as exc:
                if not recover_numerical_errors or model is None:
                    raise
                trace = recorded.trace
                has_decision = any(abs(row['z'][0] - row['z'][1]) > ev.PROTOCOL['decision_gap']
                                   for row in trace[:int(ev.PROTOCOL['max_session_bins'])])
                baseline = None if has_decision else ev.run_task_session(net, name, seed, budget, 'none', None)
                res = failed_control_outcome(trace, seed=seed, dt=float(net.dt_latent), baseline=baseline)
                record = root / 'recovery' / 'failures' / arm / f'{unit["policy"]}_{unit["seed"]}_{unit["checkpoint"]}_{budget:g}_{seed}.json'
                record.parent.mkdir(parents=True, exist_ok=True)
                record.write_text(json.dumps({'key': ev._job_key(unit, name, budget, seed),
                    'error_type': type(exc).__name__, 'error': str(exc), 'crash_bin': len(trace),
                    'filter_mean': model.m.tolist(), 'filter_covariance': model.P.tolist(),
                    'theta': unit['theta'].tolist(), 'outcome': res, 'trace': trace}, indent=2))
                failure_fields = {'numerical_failure': 1, 'failure_record': str(record.relative_to(root))}
            if not np.isfinite(res['energy']) or res['energy'] > budget + 1e-6:
                raise RuntimeError(f'Invalid control energy: {res}')
            theta = unit['theta'] if unit['theta'] is not None else [np.nan, np.nan]
            writer.writerow({k: unit[k] for k in ('controller', 'policy', 'seed', 'checkpoint')} | {
                'task': name, 'budget': budget, 'task_seed': seed, **res,
                'w_plus': float(theta[0]), 'w_minus': float(theta[1]),
                'theta': ' '.join(f'{x:.9g}' for x in theta), **failure_fields})
            f.flush()
    net.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', required=True, choices=['train', 'data', 'reference', 'r2', 'task', 'score'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--arm', required=True)
    p.add_argument('--policy', choices=MODEL_IDS)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--total-steps', type=int, default=8000)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshards', type=int, default=32)
    p.add_argument('--recover-numerical-errors', action='store_true',
                   help='Preserve prior rows; record singular-matrix controller failures and finish missing cases')
    a = p.parse_args(argv)
    torch.set_num_threads(1)
    if a.stage == 'train':
        train(a.root, a.arm, a.policy, a.seed, total_steps=a.total_steps)
    elif a.stage == 'data':
        data(a.root, a.arm)
    elif a.stage == 'reference':
        reference(a.root, a.arm)
    elif a.stage == 'task':
        task(a.root, a.arm, a.shard, a.nshards, recover_numerical_errors=a.recover_numerical_errors)
    else:
        runs_root = configure_eval(a.root, a.arm)
        out = a.root / 'eval' / a.arm
        if a.stage == 'r2':
            ev.stage_r2(out, runs_root)
        else:
            ev.stage_score(out, 'warm')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
