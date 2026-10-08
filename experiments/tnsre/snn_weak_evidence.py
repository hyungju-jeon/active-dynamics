"""Native+Uniform SNN: identification scenarios for the evidence-led trials.

The 2026-10-07 manuscript protocol (``results/archive/experiments/20261007_snn_native_uniform``: evidence trials
1-10 and fixed 2-s exploration trials 11-20, every policy at its full 20-pA bound) with one
change per scenario (``SCENARIOS``):

* ``weak2_e10``: every identification policy's own bound x0.1 in trials 1-10 (PALDI and Myopic
  norm <= 2 pA, the FLEX sphere 2 pA, RHC-US/PRBS/Random |u_j| <= 2 pA, all below the 6-pA
  evidence); trials 11-20 restore it (2026-10-07).
* ``weak10_e10``: the same with x0.5 (10 pA, above the evidence).
* ``weak2_e5``: five evidence trials at x0.1, then 15 fixed exploration trials at 20 pA.
* ``weak2_e5_ball``: ``weak2_e5`` with the ball for RHC-US, PRBS and Random (shared catalog, no
  box overlay), as every other manuscript experiment; the manuscript protocol since 2026-10-08.
* ``passive_e20``: 20 evidence trials without control (``off_policy``); the learner's only
  input is the known evidence.

Everything else is unchanged: native transfer, uniform current, 20+20 recordings, calibrated
readouts, seeds 0-19, priors, the box catalog overlay (except ``weak2_e5_ball``), and the task protocol (warm-start
iCEM, 20-pA norm, task seeds 91000-91007, checkpoints 1/5/10/15 at 480 pA^2 s, final models at
200/400/480/600 pA^2 s).

One VM runs one ``campaign`` group: seeds ``s % groups == group`` for every method of the given
scenarios, their held-out R2, their task cells, and a zero-control replay of their evidence
trials (same network seeds) that gives each trial's evidence-only choice. Each scenario has its
own root ``<root>/<scenario>``. Group 0 also replays the manuscript example for scenarios with
PALDI. Prior, fitted-reference and model-free task rows do not depend on identification;
``snn_weak_evidence_finalize`` reuses the frozen rows after checking identical readouts and priors.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import pandas as pd
import torch

from experiments.tnsre import eval_spiking_sessions as ev
from experiments.tnsre import wong_wang_full as full
from experiments.tnsre.wong_wang_full import RecordedTaskNetwork, failed_control_outcome, write_json

TASK_PRESET = 'tbme_ww_observation_transfer_uniform'  # the frozen evaluation network
POLICIES = ('adaptive_c2h20', 'active_myopic', 'flex_rollback', 'rhc', 'prbs', 'random')


@dataclass(frozen=True)
class Scenario:
    """Identification protocol of one scenario; trials are 1-based in the text, sessions 0-based here.

    Attributes:
        preset: Environment preset of the identification runs.
        policies: Identification policies.
        scale: Factor on every policy's bound in sessions ``< weak_trials``.
        weak_trials: Sessions with the scaled bound (0: none).
        evidence_trials: Sessions ``< evidence_trials`` are evidence-led; later sessions are
            fixed 2-s trials without evidence.
        constraint: Identification constraint of RHC-US, PRBS and Random: ``box`` (componentwise,
            the 2026-10-07 cohort's overlay) or ``ball`` (the shared catalog, as every TBME policy).
    """

    preset: str
    policies: tuple[str, ...]
    scale: float
    weak_trials: int
    evidence_trials: int
    constraint: str = 'box'

    @property
    def suite(self) -> str:
        return self.preset.removeprefix('tbme_')


SCENARIOS = {
    'weak2_e10': Scenario('tbme_ww_weak_evidence_transfer_uniform', POLICIES, 0.1, 10, 10),
    'weak10_e10': Scenario('tbme_ww_weak10_evidence_transfer_uniform', POLICIES, 0.5, 10, 10),
    'weak2_e5': Scenario('tbme_ww_weak_evidence5_transfer_uniform', POLICIES, 0.1, 5, 5),
    'passive_e20': Scenario('tbme_ww_passive_evidence20_transfer_uniform', ('off_policy',), 1.0, 0, 20),
    # Manuscript protocol with the ball for every baseline (2026-10-08).
    'weak2_e5_ball': Scenario('tbme_ww_weak_evidence5_transfer_uniform', POLICIES, 0.1, 5, 5, 'ball'),
}
SUITES = {sc.suite: {**full.SUITES['transfer_uniform'], 'env_preset_id': sc.preset, 'model_ids': list(sc.policies)}
          for sc in SCENARIOS.values()}
OVERLAY = Path(__file__).parent / 'config' / 'native_uniform_manuscript_models.yaml'
SEEDS = list(range(20))
ACTION_MAX = 1.0
# Constraint geometry of each identification policy (manuscript overlay): PALDI and Myopic
# project onto a ball, FLEX inputs lie on a sphere, RHC-US/PRBS/Random use a box, and the
# passive policy's control is zero. With the ball (``Scenario.constraint``), RHC-US and Random
# stay in the ball and PRBS levels lie on its boundary (``BALL_GEOMETRY``).
GEOMETRY = {'adaptive_c2h20': 'ball', 'active_myopic': 'ball', 'flex_rollback': 'sphere',
            'rhc': 'box', 'prbs': 'box', 'random': 'box', 'off_policy': 'zero'}
BALL_GEOMETRY = {**GEOMETRY, 'rhc': 'ball', 'prbs': 'sphere', 'random': 'ball'}
KEY = ['controller', 'policy', 'seed', 'checkpoint', 'task', 'budget', 'task_seed']
THREAD_ENV = {k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}


def now():
    return datetime.now(timezone.utc).isoformat()


def configure_catalogs(constraint: str = 'box'):
    """TBME catalogs (ball for every policy); ``box`` adds the 2026-10-07 overlay (componentwise
    RHC-US/PRBS/Random bounds)."""
    from experiments.experiment_definitions import configure_catalogs as configure
    from experiments.tnsre.run_tbme_experiments import tbme_catalog_paths
    assert constraint in ('box', 'ball'), constraint
    paths = tbme_catalog_paths()
    overlay = [OVERLAY] if constraint == 'box' else []
    return configure(env_catalog_paths=paths['env_catalog_paths'],
                     model_catalog_paths=[*paths['model_catalog_paths'], *overlay],
                     suite_catalog_paths=(), suite_entries=SUITES)


def geometry_of(policy: str, scenario: Scenario) -> str:
    return (BALL_GEOMETRY if scenario.constraint == 'ball' else GEOMETRY)[policy]


def configure_eval(inputs: Path):
    """The frozen evaluation protocol, with its held-out data and fit copied to ``inputs``."""
    ev.PRESET_ID = TASK_PRESET
    ev.TASKS = ('overturn',)
    ev.PROTOCOL.update(test_trajectories=str(inputs / 'test_trajectories.npz'),
                       reduced_fit=str(inputs / 'native_fit.json'), task_checkpoints=[0, 1, 5, 10, 15, 20])
    ev.PROTOCOL['budgets']['overturn'] = [5., 10., 12., 15.]
    ev.EXAMPLES['sessions'] = {'overturn': [91000, 12.]}
    ev.EXAMPLES['model_free']['overturn'] = 'spread'


def runs_root(root: Path, scenario: Scenario) -> Path:
    """Run folders are named by the environment preset without its ``tbme_`` prefix."""
    return root / 'tracks' / scenario.suite


def group_seeds(group: int, groups: int) -> list[int]:
    return [s for s in SEEDS if s % groups == group]


def learned_cell(unit: dict, budget: float, scenario: Scenario) -> bool:
    """Learned-model cells of the frozen grid: trials 1-15 at 480 pA^2 s; final models at every budget."""
    return unit['policy'] in map(ev.evaluated_policy, scenario.policies) and (
        unit['checkpoint'] in (1, 5, 10, 15) and budget == 12. or unit['checkpoint'] == 20)


# ------------------------------------------------------------------ identification
def train(root: Path, scenario: Scenario, policy: str, seed: int):
    from experiments import run
    configure_catalogs(scenario.constraint)
    args = run.build_parser().parse_args(['--base-dir', str(root), '--path-layout', 'tbme_tracks',
                                          '--total-steps', '8000', '--skip-existing'])
    result = run._run_one(exp_id=scenario.suite, policy_id=policy, seed=seed, repeat=1, base_dir=root, args=args)
    if result['status'] != 'completed' or result.get('nan_detected'):
        raise RuntimeError(f'Identification failed: {result.get("error", result)}')
    if len(result.get('session_log', [])) != 20:
        raise RuntimeError('Identification did not complete all 20 trials')
    write_json(root / 'run_checks' / f'{policy}_{seed:02d}.json',
               check_run(run_dir(root, scenario, policy, seed), policy, scenario))


def run_dir(root: Path, scenario: Scenario, policy: str, seed: int) -> Path:
    return runs_root(root, scenario) / policy / f'seed_{seed}' / 'repeat_01'


def session_bins(session_index: np.ndarray) -> np.ndarray:
    """Bins elapsed in the session at the start of each step, shape (T,); rows are in step order."""
    return pd.Series(session_index).groupby(session_index).cumcount().to_numpy()


def control_size(u: np.ndarray, geometry: str) -> np.ndarray:
    """Size of each control (T, 2) in the policy's own constraint: norm (ball, sphere) or max |u_j| (box, zero)."""
    return np.linalg.norm(u, axis=1) if geometry in ('ball', 'sphere') else np.abs(u).max(axis=1)


def check_run(path: Path, policy: str, scenario: Scenario) -> dict:
    """Bound compliance, unchanged evidence, and trial structure of one identification run.

    The control is ``policy_action``; ``action`` is the model input, control plus evidence.
    Sessions before ``scenario.weak_trials`` must satisfy the bound ``scenario.scale * ACTION_MAX``,
    later sessions ``ACTION_MAX``, in the policy's own geometry (FLEX inputs lie on the sphere;
    the passive control is exactly zero).
    """
    meta = json.loads((path / 'run_metadata.json').read_text())
    trace = pd.read_csv(path / 'state_action_trace.csv', usecols=['step', 'session_index', 'action_x', 'action_v',
                                                                  'policy_action_x', 'policy_action_v'])
    assert trace.step.is_monotonic_increasing
    k = trace.session_index.to_numpy()
    u = trace[['policy_action_x', 'policy_action_v']].to_numpy(np.float64)
    rule, log = meta['session_rule'], meta['session_log']
    weak, n_evidence = scenario.weak_trials, scenario.evidence_trials
    assert (rule['action_scale'], rule['action_scale_last_session']) == (scenario.scale, weak), rule
    assert rule['evidence_last_session'] == rule['fixed_first_session'] == n_evidence, rule
    geometry = geometry_of(policy, scenario)
    if policy in ('rhc', 'prbs', 'random'):  # the policies whose constraint the scenario sets
        expected = 'radius' if scenario.constraint == 'ball' else 'box'
        assert meta.get('action_constraint', 'box') == expected, (policy, meta.get('action_constraint'))
    bound = np.where(k < weak, scenario.scale, 1.0) * ACTION_MAX
    size = control_size(u, geometry)
    if geometry == 'zero':
        ok = size == 0.0
    else:
        ok = size <= bound * (1 + 1e-6) + 1e-7
        if geometry == 'sphere':
            ok &= size >= bound * (1 - 1e-5)
    # Evidence: the model input minus the control is the session's known evidence.
    pools = np.array([s['evidence_pool'] for s in log])[k]
    t = session_bins(k)
    expected = np.zeros_like(u)
    on = (pools >= 0) & (t < int(rule['evidence_bins']))
    expected[on, pools[on]] = float(rule['evidence_amplitude'])
    evidence_error = float(np.abs(trace[['action_x', 'action_v']].to_numpy(np.float64) - u - expected).max())
    phases = {}
    for name, mask, trials in (('weak', k < weak, f'1-{weak}'), ('strong', k >= weak, f'{weak + 1}-20')):
        if mask.any():
            ratio = size[mask] / bound[mask]
            phases[name] = dict(trials=trials, steps=int(mask.sum()), max_size=float(size[mask].max()),
                                bound=float(bound[mask][0]), max_ratio=float(ratio.max()),
                                mean_ratio=float(ratio.mean()), at_bound_fraction=float((ratio >= 0.999).mean()))
    record = dict(policy=policy, seed=int(meta['seed']), geometry=geometry,
                  action_constraint=meta.get('action_constraint', 'box'), scale=scenario.scale,
                  weak_trials=weak, evidence_trials=n_evidence, compliant=bool(ok.all()),
                  violations=int((~ok).sum()), evidence_max_error=evidence_error, phases=phases,
                  trial_bins=[int(s['bins']) for s in log], decisions=[int(s['decision']) for s in log],
                  session_rule=rule, preset=meta['env_preset_id'])
    assert meta['env_preset_id'] == scenario.preset
    assert len(log) == 20 and all(s['bins'] == 400 for s in log[n_evidence:]), 'trial structure changed'
    assert all(s['evidence_pool'] in (0, 1) for s in log[:n_evidence])
    assert all(s['evidence_pool'] == -2 for s in log[n_evidence:])
    assert len(trace) == sum(s['bins'] for s in log)
    assert record['compliant'], f'{policy} seed {meta["seed"]}: {record["violations"]} bound violations'
    assert evidence_error <= 1e-6, f'evidence changed: {evidence_error}'
    return record


# ------------------------------------------------------------------ evaluation
def r2(root: Path, inputs: Path, scenario: Scenario, group: int):
    configure_eval(inputs)
    ev.stage_r2(root / 'eval' / f'group_{group}', runs_root(root, scenario))


def task(root: Path, inputs: Path, scenario: Scenario, group: int, shard: int, nshards: int):
    """One shard of this group's learned-model task cells (warm-start iCEM, frozen protocol)."""
    configure_eval(inputs)
    runs = ev.load_session_runs(runs_root(root, scenario))
    assert {r['seed'] for r in runs} <= set(SEEDS)
    units = ev.controller_units(runs)
    jobs = [job for job in ev.task_jobs(units) if learned_cell(units[job[0]], job[2], scenario)][shard::nshards]
    out = root / 'eval' / f'group_{group}'
    path = out / 'task_warm' / f'worker_g{group}_{shard:03d}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists():  # resume a stopped shard; rows already written stay unchanged
        done = set(pd.read_csv(path)[KEY].itertuples(index=False, name=None))
    net = ev.build_network(preset_id=TASK_PRESET)
    recorded = RecordedTaskNetwork(net)
    try:
        with path.open('a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=ev.TASK_FIELDS + ['numerical_failure', 'failure_record'])
            if not done:
                writer.writeheader()
            for i, name, budget, seed in jobs:
                unit = units[i]
                if ev._job_key(unit, name, budget, seed) in done:
                    continue
                model = ev.IcemController(unit['theta'], unit['C'], unit['b'], net.dt_latent, seed=seed,
                                          warm_start=True, dynamics_type=unit['dynamics_type'],
                                          full_params=unit['full_params'], min_embedding_dim=unit['min_embedding_dim'])
                failure = {'numerical_failure': 0, 'failure_record': ''}
                try:
                    result = ev.run_task_session(recorded, name, seed, budget, unit['controller'], model)
                except torch.linalg.LinAlgError as exc:
                    # A numerical controller failure stays in the table as an unsuccessful trial.
                    trace = recorded.trace
                    baseline = ev.run_task_session(net, name, seed, budget, 'none', None)
                    result = failed_control_outcome(trace, seed=seed, dt=net.dt_latent, baseline=baseline)
                    record = out / 'failures' / f'{unit["policy"]}_{unit["seed"]}_{unit["checkpoint"]}_{budget:g}_{seed}.json'
                    write_json(record, dict(error=str(exc), key=ev._job_key(unit, name, budget, seed), trace=trace,
                                            result=result))
                    failure = {'numerical_failure': 1, 'failure_record': str(record.relative_to(root))}
                assert np.isfinite(result['energy']) and result['energy'] <= budget + 1e-6
                theta = unit['theta']
                writer.writerow({k: unit[k] for k in KEY[:4]} | dict(task=name, budget=budget, task_seed=seed,
                    **result, w_plus=theta[0], w_minus=theta[1], theta=' '.join(f'{x:.9g}' for x in theta), **failure))
                f.flush()
    finally:
        net.close()
    print(f'Completed group {group} shard {shard}: {len(jobs)} rows', flush=True)


def examples(root: Path, inputs: Path, scenario: Scenario):
    """Replay the manuscript example (PALDI seed 0, trial 20, task 91000, 480 pA^2 s) bin by bin.

    Its scalar outcome must equal this group's task row; the phase panel needs the learned
    and constant-push trajectories, which share the pre-control part.
    """
    configure_eval(inputs)
    out = root / 'examples_replay'
    ev.stage_examples(out / 'examples', runs_root(root, scenario))
    new = pd.read_csv(out / 'examples' / 'example_sessions.csv')
    rows = pd.concat([pd.read_csv(p) for p in (root / 'eval' / 'group_0' / 'task_warm').glob('worker_*.csv')])
    learned = new[new.controller == 'learned'].iloc[0]
    match = rows[(rows.policy == 'adaptive') & (rows.seed == 0) & (rows.checkpoint == 20) & (rows.budget == 12.)
                 & (rows.task_seed == 91000)]
    assert len(match) == 1
    for key in ['valid', 'target', 'onset', 'success', 'outcome_bin', 'energy']:
        np.testing.assert_allclose(learned[key], match.iloc[0][key], rtol=1e-5, atol=1e-6)
    arrays = np.load(out / 'examples' / 'example_sessions.npz')
    phase = out / 'control_phase'
    phase.mkdir(parents=True, exist_ok=True)
    np.savez(phase / 'adaptive.npz', **{k: arrays[f'overturn_learned_{k}'] for k in ('z', 'u', 'evidence')})
    np.savez(phase / 'spread.npz', **{k: arrays[f'overturn_spread_{k}'] for k in ('z', 'u', 'evidence')})
    onset = int(learned.onset)
    np.testing.assert_allclose(arrays['overturn_learned_z'][:onset], arrays['overturn_spread_z'][:onset],
                               rtol=1e-8, atol=1e-8)
    spread = new[new.controller == 'spread'].iloc[0].to_dict()
    write_json(phase / 'manifest.json', dict(verified=True, training_seed=0, task_seed=91000, checkpoint=20,
        budget_normalized=12., budget_pa2_s=480., records=[dict(policy='adaptive', **learned.to_dict()),
                                                            dict(policy='spread', **spread)]))


def replay(root: Path, scenario: Scenario, group: int, groups: int):
    """Evidence-only choice of every evidence trial of this group's seeds (zero control).

    A session's network noise depends only on its seed (``WangDecisionNetwork.reset``
    reseeds Brian2 and reinitializes every state). Validation: replaying the recorded controls
    of trial 1 of the scenario's first policy must reproduce the recorded trajectory.
    """
    from actdyn.environment.session import session_evidence, session_evidence_pool
    from actdyn.environment.spiking_decision import SpikingDecisionEnv, session_rule_from_preset
    from experiments.experiment_definitions import get_environment_preset
    configure_catalogs(scenario.constraint)
    net = ev.build_network(preset_id=scenario.preset)
    rule = session_rule_from_preset(get_environment_preset(scenario.preset), net.state_scale)

    def session(seed, k, controls=None):
        """One identification trial; ``controls`` (T, 2) replays recorded inputs, None = zero control."""
        network_seed = SpikingDecisionEnv.session_seed(seed, k)
        pool = session_evidence_pool(network_seed)
        z, states, decision, decision_bin, bins = net.reset(network_seed), [], 0, -1, 0
        while True:
            e = session_evidence(torch.tensor([bins]), torch.tensor([pool]), rule)[0].numpy().astype(np.float64)
            a = np.zeros(2) if controls is None else controls[bins]
            _counts, z = net.step(a, evidence=e)
            states.append(np.asarray(z, dtype=np.float64))
            bins += 1
            if decision == 0 and abs(z[0] - z[1]) > rule.decision_gap:
                decision, decision_bin = (1 if z[0] > z[1] else 2), bins
            if (decision and bins - decision_bin >= rule.post_decision_bins) or bins >= rule.max_session_bins:
                return dict(pool=pool, decision=decision, decision_bin=decision_bin, bins=bins), np.array(states)

    validation, rows = [], []
    try:
        for seed in group_seeds(group, groups):
            trace = pd.read_csv(run_dir(root, scenario, scenario.policies[0], seed) / 'state_action_trace.csv')
            first = trace[trace.session_index == 0]
            info, states = session(seed, 0, first[['policy_action_x', 'policy_action_v']].to_numpy(np.float64))
            recorded = first[['true_z0', 'true_z1']].to_numpy(np.float64)[1:]  # states after steps 1..T-1
            error = float(np.abs(states[:len(recorded)] - recorded).max())
            validation.append(dict(seed=seed, trial=1, policy=scenario.policies[0], bins=info['bins'],
                                   recorded_bins=len(first), max_abs_error=error))
            for k in range(scenario.evidence_trials):
                info, _ = session(seed, k)
                rows.append(dict(seed=seed, trial=k + 1, **info))
    finally:
        net.close()
    out = root / 'evidence_only'
    full.write_json(out / f'validation_group_{group}.json', dict(
        passed=all(v['bins'] == v['recorded_bins'] and v['max_abs_error'] < 1e-4 for v in validation),
        tolerance=1e-4, records=validation))
    pd.DataFrame(rows).to_csv(out / f'group_{group}.csv', index=False)


# ------------------------------------------------------------------ one VM
TRAIN_ORDER = ('rhc', 'active_myopic', 'flex_rollback', 'adaptive_c2h20', 'random', 'prbs', 'off_policy')


def campaign_jobs(scenarios: list[str], group: int, groups: int, workers: int) -> dict[str, list]:
    """Jobs ``(key, argv)`` of one seed group per campaign phase.

    Training runs longest first (RHC-US, then the planners) so the slowest runs do not start
    last. Each scenario's task cells are split into ``workers`` shards.
    """
    seeds = group_seeds(group, groups)
    sc = lambda name: ['--scenario', name]
    return {
        'identification': [(f'{name}_train_{p}_{s:02d}', [*sc(name), '--stage', 'train', '--policy', p, '--seed', str(s)])
                           for p in TRAIN_ORDER for name in scenarios if p in SCENARIOS[name].policies for s in seeds],
        'prediction': [(f'{name}_r2', [*sc(name), '--stage', 'r2']) for name in scenarios],
        'control_and_replay': [(f'{name}_replay', [*sc(name), '--stage', 'replay']) for name in scenarios] + [
            (f'{name}_task_{i:02d}', [*sc(name), '--stage', 'task', '--shard', str(i), '--nshards', str(workers)])
            for name in scenarios for i in range(workers)],
        'example': [(f'{name}_examples', [*sc(name), '--stage', 'examples']) for name in scenarios
                    if group == 0 and 'adaptive_c2h20' in SCENARIOS[name].policies],
    }


def campaign(a):
    """Train, score, and replay one seed group with ``workers`` isolated processes; ledger of every job."""
    from actdyn.policy.baseline_flex import _flex_policy_class
    _flex_policy_class()  # the vendored FLEX code must be present before any training
    root = a.root.resolve(); root.mkdir(parents=True, exist_ok=True)
    ledger = root / f'jobs_group_{a.group}.json'
    records = json.loads(ledger.read_text()) if ledger.exists() else {}
    lock = threading.Lock()
    env = os.environ | THREAD_ENV | {'PYTHONUNBUFFERED': '1', 'MPLBACKEND': 'Agg'}

    def save():
        tmp = ledger.with_suffix('.tmp'); tmp.write_text(json.dumps(records, indent=2)); tmp.replace(ledger)

    def job(spec):
        key, argv = spec
        name = argv[argv.index('--scenario') + 1]
        with lock:
            old = records.get(key, {})
            if old.get('status') == 'completed':
                return True
            attempt = old.get('attempt', 0) + 1
            log = root / 'logs' / f'group_{a.group}' / f'{key}.attempt{attempt}.log'; log.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, '-m', __spec__.name, '--root', str(root / name), '--inputs', str(a.inputs),
                       '--group', str(a.group), '--groups', str(a.groups), *argv]
            records[key] = {'command': command, 'status': 'running', 'started': now(), 'attempt': attempt,
                            'log': str(log.relative_to(root)),
                            'previous_attempts': old.get('previous_attempts', []) + ([old] if old else [])}
            save()
        start = time.monotonic()
        with log.open('w') as f:
            result = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, env=env)
        with lock:
            records[key].update(status='completed' if result.returncode == 0 else 'failed',
                                exit_code=result.returncode, seconds=time.monotonic() - start, finished=now())
            save()
        return result.returncode == 0

    def phase(name, jobs, workers):
        write_json(root / f'status_group_{a.group}.json', dict(status='running', phase=name, started=now(), jobs=len(jobs)))
        failures = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(job, spec): spec[0] for spec in jobs}
            for f in as_completed(futures):
                if not f.result():
                    failures.append(futures[f])
        if failures:
            write_json(root / f'status_group_{a.group}.json', dict(status='failed', phase=name, failures=failures, finished=now()))
            raise RuntimeError(f'{name}: {failures}')
        print(f'{now()} group {a.group} {name}: {len(jobs)} completed', flush=True)

    for name, jobs in campaign_jobs(a.scenarios, a.group, a.groups, a.workers).items():
        if jobs:
            phase(name, jobs, a.workers)
    write_json(root / f'status_group_{a.group}.json', dict(status='completed', finished=now(), scenarios=a.scenarios,
                                                           seeds=group_seeds(a.group, a.groups)))


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--stage', default='campaign', choices=['campaign', 'train', 'r2', 'task', 'examples', 'replay'])
    p.add_argument('--root', required=True, type=Path, help='campaign root; a stage gets <root>/<scenario>')
    p.add_argument('--inputs', required=True, type=Path, help='frozen test_trajectories.npz and native_fit.json')
    p.add_argument('--scenarios', nargs='+', choices=list(SCENARIOS), default=['weak10_e10', 'weak2_e5', 'passive_e20'],
                   help='campaign scenarios')
    p.add_argument('--scenario', choices=list(SCENARIOS), default='weak2_e10', help='scenario of one stage')
    p.add_argument('--group', type=int, default=0); p.add_argument('--groups', type=int, default=4)
    p.add_argument('--policy', choices=sorted(GEOMETRY), default='adaptive_c2h20'); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--shard', type=int, default=0); p.add_argument('--nshards', type=int, default=8)
    a = p.parse_args()
    torch.set_num_threads(1)
    root, scenario = a.root.resolve(), SCENARIOS[a.scenario]
    if a.stage == 'campaign':
        campaign(a)
    elif a.stage == 'train':
        train(root, scenario, a.policy, a.seed)
    elif a.stage == 'r2':
        r2(root, a.inputs.resolve(), scenario, a.group)
    elif a.stage == 'task':
        task(root, a.inputs.resolve(), scenario, a.group, a.shard, a.nshards)
    elif a.stage == 'examples':
        examples(root, a.inputs.resolve(), scenario)
    else:
        replay(root, scenario, a.group, a.groups)


if __name__ == '__main__':
    main()
