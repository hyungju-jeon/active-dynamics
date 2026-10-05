"""Control-only sensitivity to smaller action bounds on frozen 20-trial models.

All six identification methods, the initial and privileged fitted models, and
all three model-free controls use the same constraints. Budget B is in model
time units: 40 B is the nominal squared-current budget in pA^2 s. Actual neuron
currents and their summed squared-current integral are recorded separately.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import torch

from experiments.tnsre import eval_spiking_sessions as ev
from experiments.tnsre.wong_wang_full import (
    EVAL_ARMS, MODEL_IDS, RecordedTaskNetwork, configure_eval, failed_control_outcome,
)

ACTION_BOUNDS = (.25, .5)
BASE_BUDGETS = (5., 10., 12., 15.)
POLICIES = [*MODEL_IDS, 'prior', 'reduced_fit', 'none', 'spread', 'front']
LABELS = dict(zip(POLICIES, ['PALDI', 'Myopic', 'FLEX', 'RHC', 'PRBS', 'Random',
                           'Initial model', 'Privileged fit', 'No control',
                           'Constant-amplitude', 'Maximum-amplitude']))
EXTRA_FIELDS = ['action_max', 'numerical_failure', 'peak_action_norm',
                'peak_neuron_current_pa', 'total_squared_current_pa2_s', 'failure_record']


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def condition_dir(root, arm, bound):
    return root / 'eval' / f'bound_{int(round(100*bound)):03d}' / arm


def configure(parent, arm, bound):
    if bound not in ACTION_BOUNDS:
        raise ValueError(f'Unsupported preregistered action bound: {bound}')
    runs_root = configure_eval(parent, arm)
    ev.PROTOCOL.update(action_max=float(bound), task_checkpoints=[20],
                       budgets={'overturn': [b * bound**2 for b in BASE_BUDGETS]})
    return runs_root


def control_costs(trace, weights, dt):
    """Measure control-only peak currents and sum over all stimulated neurons.

    trace holds the actual two-pool actions; weights has shape (2, neurons/pool).
    dt is normalized by tau_N=0.1 s. Sensory evidence is excluded from the cost.
    """
    u = np.asarray([row['u'] for row in trace], dtype=float).reshape(-1, 2)
    return {
        'peak_action_norm': float(np.linalg.norm(u, axis=1).max(initial=0.)),
        'peak_neuron_current_pa': float((20 * np.abs(u) * weights.max(axis=1)).max(initial=0.)),
        'total_squared_current_pa2_s': float(400 * .1 * dt *
                                            np.sum(np.square(u) * np.square(weights).sum(axis=1))),
    }


def load_units(parent, arm, bound):
    runs = ev.load_session_runs(configure(parent, arm, bound))
    assert len(runs) == 120
    assert {(r['policy'], r['seed']) for r in runs} == {(p, s) for p in MODEL_IDS for s in range(20)}
    assert all(20 in r['est'] for r in runs)
    return ev.controller_units(runs)


def task(parent, root, arm, bound, shard, nshards, pilot=False):
    units = load_units(parent, arm, bound)
    jobs = ev.task_jobs(units)
    if pilot:
        # Separate calibration seeds; the budget grid is fixed before these outcomes.
        jobs = [(i, name, budget, seed - 91000 + 83000) for i, name, budget, seed in jobs
                if units[i]['policy'] in ('none', 'spread', 'front') or
                (units[i]['policy'] in ('adaptive', 'reduced_fit') and units[i]['seed'] == 0)]
    out = condition_dir(root / 'pilot' if pilot else root, arm, bound)
    path = out / 'task_warm' / f'worker_{shard:03d}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists():
        with path.open() as f:
            done = {(r['controller'], r['policy'], int(r['seed']), int(r['checkpoint']), r['task'],
                     float(r['budget']), int(r['task_seed'])) for r in csv.DictReader(f)}
    new_file = not path.exists() or path.stat().st_size == 0
    net = ev.build_network()
    recorded = RecordedTaskNetwork(net)
    try:
        with path.open('a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=ev.TASK_FIELDS + EXTRA_FIELDS)
            if new_file:
                writer.writeheader()
            for i, name, budget, seed in jobs[shard::nshards]:
                unit = units[i]
                if ev._job_key(unit, name, budget, seed) in done:
                    continue
                model = None if unit['theta'] is None else ev.IcemController(
                    unit['theta'], unit['C'], unit['b'], float(net.dt_latent), seed=seed,
                    dynamics_type=unit['dynamics_type'], full_params=unit['full_params'],
                    min_embedding_dim=unit['min_embedding_dim'], warm_start=True)
                failure = ''
                try:
                    result = ev.run_task_session(recorded, name, seed, budget, unit['controller'], model)
                except torch.linalg.LinAlgError as exc:
                    if model is None:
                        raise
                    trace = recorded.trace
                    decided = any(abs(row['z'][0] - row['z'][1]) > ev.PROTOCOL['decision_gap']
                                  for row in trace[:int(ev.PROTOCOL['max_session_bins'])])
                    baseline = None if decided else ev.run_task_session(net, name, seed, budget, 'none', None)
                    result = failed_control_outcome(trace, seed=seed, dt=float(net.dt_latent), baseline=baseline)
                    record = out / 'failures' / f'{unit["policy"]}_{unit["seed"]}_{budget:g}_{seed}.json'
                    # A failed filter can contain nonfinite values: retain their diagnostic representation.
                    record.parent.mkdir(exist_ok=True)
                    record.write_text(json.dumps({'key': ev._job_key(unit, name, budget, seed),
                        'error': repr(exc), 'trace': trace, 'filter_mean': model.m.tolist(),
                        'filter_covariance': model.P.tolist(), 'outcome': result}, indent=2))
                    failure = str(record.relative_to(root))
                costs = control_costs(recorded.trace, net.stim_weights, float(net.dt_latent))
                assert costs['peak_action_norm'] <= bound + 1e-6
                assert np.isfinite(result['energy']) and result['energy'] <= budget + 1e-6
                assert costs['peak_neuron_current_pa'] <= 20 * bound * net.stim_weights.max() + 1e-4
                theta = unit['theta'] if unit['theta'] is not None else [np.nan, np.nan]
                writer.writerow({k: unit[k] for k in ('controller', 'policy', 'seed', 'checkpoint')} | {
                    'task': name, 'budget': budget, 'task_seed': seed, **result, **costs,
                    'w_plus': float(theta[0]), 'w_minus': float(theta[1]),
                    'theta': ' '.join(f'{x:.9g}' for x in theta), 'action_max': bound,
                    'numerical_failure': int(bool(failure)), 'failure_record': failure})
                f.flush()
    finally:
        net.close()


def summarize(parent, root):
    import pandas as pd

    frames, hashes = [], {}
    keycols = ['controller', 'policy', 'seed', 'checkpoint', 'task', 'budget', 'task_seed']
    for bound in ACTION_BOUNDS:
        for arm in EVAL_ARMS:
            units = load_units(parent, arm, bound)
            expected = {ev._job_key(units[i], t, b, s) for i, t, b, s in ev.task_jobs(units)}
            paths = sorted((condition_dir(root, arm, bound) / 'task_warm').glob('worker_*.csv'))
            df = pd.concat([pd.read_csv(p, keep_default_na=False) for p in paths], ignore_index=True)
            actual = set(df[keycols].itertuples(index=False, name=None))
            assert len(actual) == len(df) == len(expected) and actual == expected, (bound, arm)
            assert np.isfinite(df[['energy', *EXTRA_FIELDS[2:5]]].to_numpy()).all()
            assert (df.energy <= df.budget + 1e-6).all() and (df.peak_action_norm <= bound + 1e-6).all()
            assert df.groupby('task_seed').valid.nunique().max() == 1
            for p in paths:
                hashes[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
            frames.append(df.assign(arm=arm))
    data = pd.concat(frames, ignore_index=True)
    assert len(data) == 83072
    out = root / 'comparison'
    out.mkdir(exist_ok=True)
    data.to_csv(out / 'control_all.csv', index=False)
    data[data.numerical_failure == 1].to_csv(out / 'numerical_failures.csv', index=False)
    scores = data[data.valid == 1].groupby(
        ['action_max', 'arm', 'policy', 'seed', 'checkpoint', 'budget'], as_index=False).agg(
        success=('success', 'mean'), energy=('energy', 'mean'),
        total_squared_current_pa2_s=('total_squared_current_pa2_s', 'mean'), n_tasks=('success', 'size'))
    scores.to_csv(out / 'control_by_seed.csv', index=False)
    summary = scores.groupby(['action_max', 'arm', 'policy', 'checkpoint', 'budget']).agg(
        success=('success', 'mean'), sem=('success', 'sem'), n_seeds=('success', 'size'),
        mean_energy=('energy', 'mean'), mean_total_squared_current_pa2_s=('total_squared_current_pa2_s', 'mean')).reset_index()
    summary['sem'] = summary['sem'].fillna(0.)
    summary.to_csv(out / 'control_summary.csv', index=False)
    lines = ['# Reduced-bound Wong–Wang control comparison', '',
        'Control-only reevaluation of all six methods after 20 identification trials. '
        'The 480 identification runs, parameter estimates, readouts, reference fits, and task seeds are reused. '
        'The original identification inputs were not reduced; this study tests control under tighter constraints.', '',
        'Bounds are 25% and 50% of the original two-pool L2 bound. Budgets are [5, 10, 12, 15] times '
        'the squared bound, giving the same budget fractions of a full-amplitude two-second window. '
        'No outcome-based budget selection was performed. All three model-free controls are included.', '',
        f'Coverage: {len(data):,} task evaluations. Numerical controller failures: '
        f'{int(data.numerical_failure.sum())}; unsuccessful valid tasks remain in the denominator.', '',
        'Success is averaged over valid task seeds, then over 20 training seeds for learned models. '
        'Model-free controls have one outcome per task seed and no training-seed uncertainty. '
        'All comparisons are conditional on the existing eight-task panel. '
        'The same partial mask is used throughout. Partial stimulation still has about 2.13 times '
        'the squared-current dose of uniform stimulation for a given action; budgets below are nominal. '
        'The CSV records the actual integral summed over all stimulated neurons, in pA² s, excluding evidence.', '']
    for bound in ACTION_BOUNDS:
        for arm in EVAL_ARMS:
            budgets = [b * bound**2 for b in BASE_BUDGETS]
            lines += [f'## {arm}, action bound {bound:g} ({20*bound:g} pA nominal)', '',
                '| Method | ' + ' | '.join(f'{40*b:g} pA² s' for b in budgets) + ' |',
                '|---|' + '---:|' * len(budgets)]
            for policy in POLICIES:
                values = []
                for budget in budgets:
                    row = summary[(summary.action_max == bound) & (summary.arm == arm) &
                        (summary.policy == policy) & (summary.budget == (budgets[0] if policy == 'none' else budget))].iloc[0]
                    values.append(f'{100*row.success:.1f}%')
                lines.append(f'| {LABELS[policy]} | ' + ' | '.join(values) + ' |')
            lines.append('')
    (out / 'REPORT.md').write_text('\n'.join(lines))
    write_json(out / 'verification.json', {'control_sessions': len(data), 'duplicate_keys': 0,
        'numerical_failures': int(data.numerical_failure.sum()), 'bounds_and_budgets_passed': True,
        'source_sha256': hashes})


def campaign(parent, root, workers, nshards, pilot=False):
    root.mkdir(parents=True, exist_ok=True)
    specs = [(f'{bound:g}_{arm}_{shard:03d}', arm, bound, shard)
             for shard in range(nshards) for bound in ACTION_BOUNDS for arm in EVAL_ARMS]
    tag = 'pilot' if pilot else 'campaign'
    manifest_path, ledger = root / f'{tag}.json', root / f'{tag}_jobs.json'
    records = json.loads(ledger.read_text()) if ledger.exists() else {}
    state = {'status': 'running', 'created': now(), 'parent_results': str(parent), 'workers': workers,
        'shards_per_condition': nshards, 'action_bounds': ACTION_BOUNDS, 'base_budgets': BASE_BUDGETS,
        'budget_rule': 'B = base_budget * action_max**2; nominal physical budget = 40 B pA^2 s',
        'checkpoints': [20], 'training_reused': True, 'identification_action_bounds_unchanged': True,
        'task_seeds': list(range(83000 if pilot else 91000, 83008 if pilot else 91008)),
        'policies': POLICIES, 'input_arms': EVAL_ARMS,
        'expected_rows': 2176 if pilot else 83072, 'source': str(Path(__file__).resolve()),
        'python': sys.executable, 'selection': 'Both bounds and all budgets retained regardless of outcomes.'}
    write_json(manifest_path, state)
    lock = threading.Lock()
    env = os.environ | {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                       'NUMEXPR_NUM_THREADS': '1', 'PYTHONUNBUFFERED': '1'}

    def run(spec):
        key, arm, bound, shard = spec
        with lock:
            previous = records.get(key, {})
            if previous.get('status') == 'completed':
                return True
            attempt = previous.get('attempt', 0) + 1
            log = root / 'logs' / f'{tag}_{key}.attempt_{attempt}.log'
            log.parent.mkdir(exist_ok=True)
            cmd = [sys.executable, '-m', 'experiments.tnsre.wong_wang_budget', '--stage', 'task',
                '--parent', str(parent), '--root', str(root), '--arm', arm, '--bound', str(bound),
                '--shard', str(shard), '--nshards', str(nshards)] + (['--pilot'] if pilot else [])
            records[key] = {'status': 'running', 'start': now(), 'attempt': attempt, 'command': cmd,
                            'log': str(log), 'previous_attempts': previous.get('previous_attempts', []) + ([previous] if previous else [])}
            write_json(ledger, records)
        started = time.monotonic()
        with log.open('w') as f:
            result = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        with lock:
            records[key].update(status='completed' if result.returncode == 0 else 'failed',
                               end=now(), seconds=time.monotonic()-started, exit_code=result.returncode)
            write_json(ledger, records)
        return result.returncode == 0

    try:
        failures = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run, s): s[0] for s in specs}
            for number, future in enumerate(as_completed(futures), 1):
                if not future.result():
                    failures.append(futures[future])
                if number % 8 == 0 or number == len(specs):
                    print(f'{now()} {tag}: {number}/{len(specs)} jobs; {len(failures)} failed', flush=True)
        if failures:
            raise RuntimeError(f'Failed jobs retained: {failures}')
        if not pilot:
            summarize(parent, root)
        state.update(status='completed', end=now())
    except Exception as exc:
        state.update(status='failed', end=now(), error=repr(exc))
        raise
    finally:
        write_json(manifest_path, state)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['task', 'campaign', 'score'], required=True)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--arm', choices=EVAL_ARMS)
    p.add_argument('--bound', type=float, choices=ACTION_BOUNDS)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshards', type=int, default=32)
    p.add_argument('--workers', type=int, default=32)
    p.add_argument('--pilot', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(1)
    if a.stage == 'task':
        task(a.parent.resolve(), a.root.resolve(), a.arm, a.bound, a.shard, a.nshards, a.pilot)
    elif a.stage == 'score':
        summarize(a.parent.resolve(), a.root.resolve())
    else:
        campaign(a.parent.resolve(), a.root.resolve(), a.workers, a.nshards, a.pilot)


if __name__ == '__main__':
    main()
