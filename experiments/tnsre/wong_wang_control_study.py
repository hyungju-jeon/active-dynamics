"""Durable runner for the frozen-model Wong-Wang control study.

Each plan explicitly lists arms, training seeds, task seeds, tasks, and baseline
parameters. Each worker writes an append-only JSONL shard. Task outcomes never
change the plan or remove failed controllers from valid-session denominators.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from experiments.tnsre import eval_spiking_sessions as ev
from experiments.tnsre.wong_wang_full import EVAL_ARMS, MODEL_IDS, configure_eval
from experiments.tnsre.wong_wang_budget import control_costs
from experiments.tnsre.wong_wang_control import (
    ControlTask, CountObserver, TrackingController, rollout_statistics,
    run_session, task_protocol,
)


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=serial, allow_nan=False)+'\n')
    tmp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=serial).encode()).hexdigest()


def export_models(parent, out):
    """Copy the exact 20-session estimates/readouts and retain input provenance."""
    models, inputs = {}, {}
    for arm in EVAL_ARMS:
        runs_root = configure_eval(parent, arm)
        ev.PROTOCOL['task_checkpoints'] = [20]
        runs = ev.load_session_runs(runs_root)
        assert {(r['policy'], r['seed']) for r in runs} == set(itertools.product(MODEL_IDS, range(20)))
        units = ev.controller_units(runs)
        models[arm] = units
        for pattern in ('*/seed_*/repeat_*/run_metadata.json',
                        '*/seed_*/repeat_*/embedding_estimate_trace.csv'):
            for p in runs_root.glob(pattern):
                key = str(p.relative_to(parent))
                if key not in inputs:
                    inputs[key] = hashlib.sha256(p.read_bytes()).hexdigest()
        p = parent/'reference'/f'{arm}.json'
        inputs[str(p.relative_to(parent))] = hashlib.sha256(p.read_bytes()).hexdigest()
    write_json(out/'models.json', models)
    write_json(out/'model_provenance.json', dict(parent=str(parent.resolve()),
        checkpoint=20, source_files=inputs, models_sha256=digest(models)))


def load_models(path, arm):
    units = json.loads(path.read_text())[arm]
    for unit in units:
        for key in ('theta', 'C', 'b', 'full_params'):
            if unit.get(key) is not None:
                unit[key] = np.asarray(unit[key], dtype=float)
    return units


def configure_arm(arm):
    # Network construction uses only the preset selected by configure_eval.
    configure_eval(Path('.'), arm)


def make_controller(unit, seed, dt, *, tracking=False):
    cls = TrackingController if tracking else ev.IcemController
    return cls(unit['theta'], unit['C'], unit['b'], dt, seed=seed,
        dynamics_type=unit['dynamics_type'], full_params=unit['full_params'],
        min_embedding_dim=unit['min_embedding_dim'], warm_start=True)


def planned_jobs(plan, units, arm=None):
    """Generate exact expected rows before execution; no result-dependent filtering."""
    jobs = []
    for task_dict in plan['tasks']:
        task = ControlTask(**task_dict)
        for unit in units:
            if unit['policy'] not in plan['policies']:
                continue
            if unit['seed'] >= 0 and unit['seed'] not in plan['training_seeds']:
                continue
            for seed in plan['task_seeds']:
                jobs.append(dict(task=asdict(task), unit=unit, task_seed=seed, parameters={}))
        priors = [u for u in units if u['policy'] == 'prior' and u['seed'] in plan['training_seeds']]
        for spec in plan.get('baselines', []):
            if 'arms' in spec and arm not in spec['arms']:
                continue
            if 'conditions' in spec and task.key not in spec['conditions']:
                continue
            base = spec['controller']
            # Feedback baselines share each training seed's readout. Open-loop
            # pulses do not depend on it, and have a single outcome per task seed.
            sources = priors if base in ('relay', 'pi') else [None]
            for prior in sources:
                unit = dict(controller=base, policy=spec['label'], seed=-1 if prior is None else prior['seed'],
                            checkpoint=-1, theta=None)
                if prior is not None:
                    unit.update(C=prior['C'], b=prior['b'])
                for seed in plan['task_seeds']:
                    jobs.append(dict(task=asdict(task), unit=unit, task_seed=seed,
                                     parameters=spec['parameters']))
    for job in jobs:
        job['key'] = digest({k: v for k, v in job.items() if k != 'key'})
    if len({j['key'] for j in jobs}) != len(jobs):
        raise ValueError('Duplicate jobs in the declared plan')
    return jobs


def failure_outcome(net, task, seed, trace):
    """Retain controller failures as failures, including the full tracking loss.

    The maximum squared gap error is 25 for z_i in [-2,2], r in [-1,1].
    Failed tracking sessions receive that declared worst-case loss. No failed
    session is silently dropped or assigned a successful partial trajectory.
    """
    onset = next((j+1 for j, r in enumerate(trace[:int(ev.PROTOCOL['max_session_bins'])])
                  if abs(r['z'][0]-r['z'][1]) > ev.PROTOCOL['decision_gap']), None)
    if onset is None:
        check = run_session(net, task, seed)
        valid, onset, target = (check[k] for k in ('valid', 'onset', 'target'))
    else:
        valid = 1
        target = 1 if trace[onset-1]['z'][0] > trace[onset-1]['z'][1] else 0
    return dict(valid=valid, onset=onset, target=target, success=0,
        energy=float(net.dt_latent*sum(np.dot(r['u'], r['u']) for r in trace)),
        rmse=5. if valid and task.name != 'overturn' else None,
        occupancy=0. if valid and task.name != 'overturn' else None, outcome_bin=-1)


def evaluate(models, root, plan, arm, shard, nshards):
    configure_arm(arm)
    units = load_models(models, arm)
    jobs = planned_jobs(plan, units, arm)[shard::nshards]
    path = root/plan['name']/arm/f'worker_{shard:03d}.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row['plan_sha256'] != digest(plan):
                raise ValueError('Refusing to mix plans in an existing shard')
            done.add(row['key'])
    net = ev.build_network()
    try:
        with path.open('a') as f:
            for job in jobs:
                if job['key'] in done:
                    continue
                start = time.monotonic()
                unit, task, seed = job['unit'], ControlTask(**job['task']), job['task_seed']
                controller = None
                if unit['theta'] is not None:
                    controller = make_controller(unit, seed, net.dt_latent, tracking=task.name != 'overturn')
                elif unit['controller'] in ('relay', 'pi'):
                    controller = CountObserver(unit['C'], unit['b'], net.dt_latent,
                                               tau=job['parameters'].get('tau', .05))
                trace, failure = [], ''
                try:
                    result = run_session(net, task, seed, controller=controller,
                        baseline=unit['controller'], parameters=job['parameters'], trace=trace)
                except (torch.linalg.LinAlgError, np.linalg.LinAlgError, FloatingPointError) as exc:
                    result = failure_outcome(net, task, seed, trace)
                    failure = repr(exc)
                    # Failed arrays can be nonfinite; preserve them in an NPZ.
                    failed = path.parent/'failures'/job['key']
                    failed.parent.mkdir(exist_ok=True)
                    np.savez_compressed(str(failed)+'.npz',
                        u=np.array([r['u'] for r in trace]), z=np.array([r['z'] for r in trace]),
                        counts=np.array([r['counts'] for r in trace]))
                    write_json(failed.with_suffix('.json'), dict(error=failure, task=asdict(task),
                        policy=unit['policy'], seed=unit['seed'], task_seed=seed))
                costs = control_costs(trace, net.stim_weights, net.dt_latent)
                if costs['peak_action_norm'] > task.bound+1e-6 or result['energy'] > task.budget+1e-6:
                    raise RuntimeError('Control constraint failure')
                row = dict(key=job['key'], plan_sha256=digest(plan), arm=arm,
                    policy=unit['policy'], controller=unit['controller'], seed=unit['seed'],
                    checkpoint=unit['checkpoint'], task_seed=seed, task=task.name,
                    condition=task.key, bound=task.bound, budget=task.budget,
                    parameters=job['parameters'], numerical_failure=int(bool(failure)), error=failure,
                    seconds=time.monotonic()-start, **result, **costs)
                if (unit['policy'] in plan.get('trace_policies', [])
                    and unit['seed'] in [-1, *plan.get('trace_training_seeds', [])]
                    and seed in plan.get('trace_task_seeds', [])):
                    trace_path = path.parent/'traces'/f'{job["key"]}.npz'
                    trace_path.parent.mkdir(exist_ok=True)
                    np.savez_compressed(trace_path, **{key: np.array([r[key] for r in trace])
                        for key in ('bin', 'u', 'z', 'evidence', 'counts')})
                f.write(json.dumps(row, default=serial, allow_nan=False)+'\n')
                f.flush()
    finally:
        net.close()


def diagnostic(models, root, plan, arm, shard, nshards):
    """Replay competing open-loop plans from identical full-network initializations.

    Predictions are scored from both the EKF mean and the true onset state. The
    latter is diagnostic only. Actual candidate outcomes replay the uncontrolled
    prefix and then the complete candidate without replanning, matching the
    prediction horizon. Separate evaluations provide actual MPC outcomes.
    """
    configure_arm(arm)
    units = load_models(models, arm)
    jobs = [j for j in planned_jobs(plan, units, arm) if j['unit']['theta'] is not None][shard::nshards]
    path = root/plan['name']/arm/f'diagnostic_{shard:03d}.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row['plan_sha256'] != digest(plan):
                raise ValueError('Diagnostic plan changed')
            done.add(row['key'])
    net = ev.build_network()
    try:
        with path.open('a') as f:
            for job in jobs:
                if job['key'] in done:
                    continue
                started = time.monotonic()
                unit, seed, task = job['unit'], job['task_seed'], ControlTask(**job['task'])
                task_protocol(task)
                model = make_controller(unit, seed, net.dt_latent)
                pool = ev.evidence_pool(seed)
                net.reset(seed)
                valid = False
                for t in range(int(ev.PROTOCOL['max_session_bins'])):
                    evidence = ev.evidence_drive(t, pool)
                    counts, z = net.step(np.zeros(2), evidence=evidence)
                    model.observe(counts, evidence)
                    if abs(z[0]-z[1]) > ev.PROTOCOL['decision_gap']:
                        valid = True
                        break
                row = dict(key=job['key'], plan_sha256=digest(plan), arm=arm,
                    policy=unit['policy'], seed=unit['seed'], task_seed=seed,
                    condition=task.key, valid=int(valid), budget=task.budget, bound=task.bound)
                if valid:
                    onset, target = t+1, 1 if z[0] > z[1] else 0
                    estimate, true_state = model.m.copy(), z.copy()
                    selected = model.plan('overturn', onset, 0, target, pool, task.budget)
                    front = np.stack([ev.model_free_input('front', 'overturn', k, target,
                                     task.budget, net.dt_latent) for k in range(task.window)])
                    spread = np.stack([ev.model_free_input('spread', 'overturn', k, target,
                                      task.budget, net.dt_latent) for k in range(task.window)])
                    actions = np.stack([selected, front, spread])
                    drive = np.stack([ev.evidence_drive(onset+j, pool)
                                      for j in range(task.window+task.settle)])
                    prediction = {}
                    for state_name, state in [('filtered', estimate), ('true_onset', true_state)]:
                        prediction[state_name] = rollout_statistics(model.transition['plan'],
                            state[::-1].copy() if target == 0 else state,
                            drive[:, ::-1].copy() if target == 0 else drive,
                            actions[:, :, ::-1].copy() if target == 0 else actions,
                            dt=net.dt_latent, seed=seed,
                            samples=int(ev.PROTOCOL['planner_mc_samples']))
                    outcomes, replay_states = [], []
                    for action in actions:
                        net.reset(seed)
                        for j in range(onset):
                            _, replay = net.step(np.zeros(2), evidence=ev.evidence_drive(j, pool))
                        np.testing.assert_allclose(replay, true_state, rtol=1e-6, atol=1e-7)
                        gaps, states = [], []
                        for j in range(task.window+task.settle):
                            u = action[j] if j < task.window else np.zeros(2)
                            _, actual = net.step(u, evidence=drive[j])
                            gaps.append(float(actual[target]-actual[1-target]))
                            states.append(actual)
                        outcomes.append(dict(success=int(max(gaps)>ev.PROTOCOL['decision_gap']),
                            peak_gap=max(gaps), energy=float(net.dt_latent*np.sum(action**2))))
                        replay_states.append(states)
                    row.update(onset=onset, target=target, state= true_state,
                        estimate=estimate, state_error=float(np.linalg.norm(estimate-true_state)),
                        candidates=['selected', 'front', 'spread'], prediction=prediction,
                        actual=outcomes)
                    trace_path = path.parent/'traces'/f'{job["key"]}.npz'
                    trace_path.parent.mkdir(exist_ok=True)
                    np.savez_compressed(trace_path, actions=actions, states=replay_states, evidence=drive)
                row['seconds'] = time.monotonic()-started
                f.write(json.dumps(row, default=serial, allow_nan=False)+'\n')
                f.flush()
    finally:
        net.close()


def replay_check(models, root, arm, expected_path):
    """Check saved SNN outcomes on this host, for one input condition per process."""
    configure_arm(arm)
    units = load_models(models, arm)
    entries = [r for r in json.loads(expected_path.read_text()) if r['arm'] == arm]
    if not entries:
        raise ValueError('No saved reference outcomes for this input condition')
    checks = []
    net = ev.build_network()
    try:
        for entry in entries:
            expected = entry['expected']
            unit = next(u for u in units if u['policy'] == expected['policy']
                        and u['seed'] == int(expected['seed']))
            seed = int(expected['task_seed'])
            ctrl = None if unit['theta'] is None else make_controller(unit, seed, net.dt_latent)
            actual = run_session(net, ControlTask(budget=float(expected['budget'])), seed,
                                 controller=ctrl, baseline=unit['controller'])
            for key in ('valid', 'target', 'onset', 'success', 'outcome_bin'):
                if actual[key] != int(expected[key]):
                    raise AssertionError((arm, unit['policy'], key, actual, expected))
            np.testing.assert_allclose(actual['energy'], float(expected['energy']), rtol=1e-6, atol=1e-7)
            checks.append(dict(policy=unit['policy'], expected=expected, actual=actual, passed=True))
            write_json(root/'preflight'/f'{arm}.json', dict(arm=arm, checks=checks,
                models_sha256=hashlib.sha256(models.read_bytes()).hexdigest()))
    finally:
        net.close()


def campaign(args, plan):
    """Launch bounded independent process shards; keep every attempt log."""
    out = args.root/plan['name']
    out.mkdir(parents=True, exist_ok=True)
    frozen = out/'plan.json'
    if frozen.exists() and json.loads(frozen.read_text()) != plan:
        raise ValueError('Cannot replace a frozen campaign plan')
    write_json(frozen, plan)
    suffix = '' if args.host_label == 'local' else f'_{args.host_label}'
    ledger = out/f'jobs{suffix}.json'
    records = json.loads(ledger.read_text()) if ledger.exists() else {}
    env = os.environ | dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
    shards = list(range(args.nshards)) if args.shards is None else [int(s) for s in args.shards.split(',')]
    if not shards or len(set(shards)) != len(shards) or any(s < 0 or s >= args.nshards for s in shards):
        raise ValueError('Need unique shard indices within the declared shard count')
    assignment = dict(plan_sha256=digest(plan), nshards=args.nshards, shards=shards,
        models_sha256=hashlib.sha256(args.models.read_bytes()).hexdigest())
    assignment_path = out/f'assignment{suffix}.json'
    if assignment_path.exists() and json.loads(assignment_path.read_text()) != assignment:
        raise ValueError('Cannot change a running host assignment or frozen model file')
    write_json(assignment_path, assignment)
    jobs = [(arm, shard) for shard in shards for arm in plan['arms']]
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for arm, shard in jobs:
            key = f'{arm}_{shard:03d}'
            if records.get(key, {}).get('status') == 'completed':
                continue
            prev = records.get(key, {})
            attempt = prev.get('attempt', 0)+1
            log = out/'logs'/f'{key}.attempt_{attempt}.log'
            log.parent.mkdir(exist_ok=True)
            command = [sys.executable, '-m', 'experiments.tnsre.wong_wang_control_study',
                '--stage', plan.get('worker_stage', 'evaluate'),
                '--models', str(args.models), '--root', str(args.root), '--plan', str(frozen),
                '--arm', arm, '--shard', str(shard), '--nshards', str(args.nshards)]
            records[key] = dict(status='queued', attempt=attempt, command=command, log=str(log),
                                previous=prev)
            def execute(command=command, log=log):
                start = time.monotonic()
                with log.open('w') as fh:
                    completed = subprocess.run(command, stdout=fh, stderr=subprocess.STDOUT, env=env)
                return completed.returncode, time.monotonic()-start
            futures[pool.submit(execute)] = key
        write_json(ledger, records)
        for f in as_completed(futures):
            key = futures[f]
            code, seconds = f.result()
            records[key].update(status='completed' if code == 0 else 'failed',
                                exit_code=code, seconds=seconds)
            write_json(ledger, records)
            if code:
                failures.append(key)
            print(json.dumps(dict(job=key, exit_code=code, seconds=seconds)), flush=True)
    write_json(out/f'completion{suffix}.json', dict(completed=sum(r['status']=='completed' for r in records.values()),
        total=len(jobs), failures=failures, plan_sha256=digest(plan),
        host_label=args.host_label, shards=shards, nshards=args.nshards,
        finished=datetime.now(timezone.utc).isoformat()))
    if failures:
        raise RuntimeError(f'Failed shards retained: {failures}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['export', 'evaluate', 'diagnostic', 'campaign', 'replay-check'], required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--parent', type=Path)
    p.add_argument('--models', type=Path)
    p.add_argument('--plan', type=Path)
    p.add_argument('--expected', type=Path, help='Saved outcome JSON for the one-arm replay check')
    p.add_argument('--arm', choices=EVAL_ARMS)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshards', type=int, default=16)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--shards', help='Comma-separated disjoint shard indices assigned to this host')
    p.add_argument('--host-label', default='local')
    args = p.parse_args()
    if not args.host_label.replace('-', '').replace('_', '').isalnum():
        p.error('--host-label must contain only letters, digits, hyphens, or underscores')
    torch.set_num_threads(1)
    if args.stage == 'export':
        export_models(args.parent, args.root)
        return
    if args.stage == 'replay-check':
        if args.expected is None or args.arm is None or args.models is None:
            p.error('replay-check requires --expected, --arm, and --models')
        replay_check(args.models, args.root, args.arm, args.expected)
        return
    plan = json.loads(args.plan.read_text())
    if args.stage == 'campaign':
        campaign(args, plan)
    elif args.stage == 'diagnostic':
        diagnostic(args.models, args.root, plan, args.arm, args.shard, args.nshards)
    else:
        evaluate(args.models, args.root, plan, args.arm, args.shard, args.nshards)


if __name__ == '__main__':
    main()
