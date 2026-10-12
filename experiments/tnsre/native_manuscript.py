"""Complete the 2026-10-07 native/uniform result (now archived) without retraining.

Frozen 120 models, 20 training seeds, and eight original task starts. Add
checkpoints 1/5/10/15 at budget 12 and final-model budgets 5/15. Retain all
methods and task exclusions. Existing checkpoint-20 rows are replay checks,
never replaced by a new result. All controls use warm-start iCEM.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

from experiments.tnsre import eval_spiking_sessions as ev
from experiments.tnsre.wong_wang_full import RecordedTaskNetwork, failed_control_outcome, write_json

DEFAULT_ROOT = Path('results/archive/experiments/20261007_snn_native_uniform')
POLICIES = ('adaptive', 'active_myopic', 'flex_rollback', 'rhc', 'prbs', 'random')
KEY = ['controller', 'policy', 'seed', 'checkpoint', 'task', 'budget', 'task_seed']


def configure(root: Path):
    ev.PRESET_ID = 'tbme_ww_observation_transfer_uniform'
    ev.TASKS = ('overturn',)
    ev.PROTOCOL.update(test_trajectories=str(root/'data/test_trajectories.npz'),
                       reduced_fit=str(root/'reference/native_fit.json'),
                       task_checkpoints=[0, 1, 5, 10, 15, 20])
    ev.PROTOCOL['budgets']['overturn'] = [5., 10., 12., 15.]
    ev.EXAMPLES['sessions'] = {'overturn': [91000, 12.]}
    return ev.runs_root_of(root)


def extension_jobs(units):
    """Only cells missing from the selected, immutable original evaluation."""
    return [job for job in ev.task_jobs(units)
            if (units[job[0]]['checkpoint'] in (1, 5, 10, 15) and job[2] == 12.)
            or (units[job[0]]['checkpoint'] in (0, 20, -1) and job[2] in (5., 15.)
                and units[job[0]]['policy'] != 'none')]


def inputs(root):
    runs_root = configure(root)
    runs = ev.load_session_runs(runs_root)
    assert {(r['policy'], r['seed']) for r in runs} == {(p, s) for p in POLICIES for s in range(20)}
    assert all(r['dynamics_type'] == 'wong_wang_transfer' and set(r['est']) == set(range(21)) for r in runs)
    return runs, ev.controller_units(runs)


def worker(root, out, shard, nshards):
    _, units = inputs(root)
    jobs = extension_jobs(units)[shard::nshards]
    path = out/'task_warm'/f'worker_extension_{shard:03d}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    # Each immutable shard is written once. A failed attempt is kept separately.
    net = ev.build_network(preset_id=ev.PRESET_ID)
    recorded = RecordedTaskNetwork(net)
    try:
        with path.open('x', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=ev.TASK_FIELDS+['numerical_failure', 'failure_record'])
            writer.writeheader()
            for i, task, budget, seed in jobs:
                unit = units[i]; model = None
                if unit['theta'] is not None:
                    model = ev.IcemController(unit['theta'], unit['C'], unit['b'], net.dt_latent,
                        seed=seed, warm_start=True, dynamics_type=unit['dynamics_type'],
                        full_params=unit['full_params'], min_embedding_dim=unit['min_embedding_dim'])
                failure = {'numerical_failure': 0, 'failure_record': ''}
                try:
                    result = ev.run_task_session(recorded, task, seed, budget, unit['controller'], model)
                except torch.linalg.LinAlgError as exc:
                    if model is None:
                        raise
                    trace = recorded.trace
                    baseline = ev.run_task_session(net, task, seed, budget, 'none', None)
                    result = failed_control_outcome(trace, seed=seed, dt=net.dt_latent, baseline=baseline)
                    record = out/'failures'/f'{shard:03d}_{unit["policy"]}_{unit["seed"]}_{unit["checkpoint"]}_{budget:g}_{seed}.json'
                    write_json(record, dict(error=str(exc), key=ev._job_key(unit, task, budget, seed), trace=trace, result=result))
                    failure = {'numerical_failure': 1, 'failure_record': str(record.relative_to(out))}
                assert np.isfinite(result['energy']) and result['energy'] <= budget+1e-6
                theta = unit['theta'] if model is not None else [np.nan, np.nan]
                writer.writerow({k: unit[k] for k in KEY[:4]} | dict(task=task, budget=budget, task_seed=seed,
                    **result, w_plus=theta[0], w_minus=theta[1], theta=' '.join(f'{x:.9g}' for x in theta), **failure))
                f.flush()
    finally:
        net.close()
    print(f'Completed shard {shard}: {len(jobs)} rows', flush=True)


def examples(root, out):
    """Replay the preselected example and check its original scalar outcomes."""
    runs_root = configure(root)
    # Record the phase-panel baseline on the same network as all other examples.
    # A second construction in one process failed the saved-outcome replay check.
    ev.EXAMPLES['model_free']['overturn'] = 'spread'
    ev.stage_examples(out/'examples', runs_root)
    import pandas as pd
    old = pd.concat([pd.read_csv(p) for p in (root/'eval/task_warm').glob('worker_*.csv')])
    new = pd.read_csv(out/'examples/example_sessions.csv')
    for _, row in new.iterrows():
        policy = {'learned': 'adaptive', 'fit': 'reduced_fit'}.get(row.controller, row.controller)
        seed = 0 if row.controller in ('learned', 'fit') else -1
        # No-input was recorded only at budget 10; it is independent of budget.
        budget = 10. if policy == 'none' else row.budget
        expected = old[(old.policy == policy) & (old.seed == seed) & (old.budget == budget)
                       & (old.task_seed == row.task_seed)]
        assert len(expected) == 1, (policy, expected)
        for key in ['valid', 'target', 'onset', 'success', 'outcome_bin', 'energy']:
            np.testing.assert_allclose(row[key], expected.iloc[0][key], rtol=1e-5, atol=1e-6)
    write_json(out/'example_verification.json', dict(passed=True, rows=len(new), task_seed=91000, training_seed=0))
    spread = new[new.controller == 'spread'].iloc[0].to_dict()
    phase = out/'control_phase'; phase.mkdir()
    arrays = np.load(out/'examples/example_sessions.npz')
    np.savez(phase/'adaptive.npz', **{k: arrays[f'overturn_learned_{k}'] for k in ('z','u','evidence')})
    np.savez(phase/'spread.npz', **{k: arrays[f'overturn_spread_{k}'] for k in ('z','u','evidence')})
    learned = new[new.controller == 'learned'].iloc[0].to_dict()
    write_json(phase/'manifest.json', dict(verified=True, training_seed=0, task_seed=91000,
        checkpoint=20, budget_normalized=12., budget_pa2_s=480.,
        records=[dict(policy='adaptive', **learned), dict(policy='spread', **spread)],
        inputs={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (root/'eval/task_warm').glob('worker_*.csv')}))


def campaign(a):
    """Eight isolated CPU workers per VM, with globally disjoint shards."""
    configure(a.root)
    out = a.out; out.mkdir(parents=True, exist_ok=True)
    if a.group == 0:
        examples(a.root, out)
    def job(shard):
        cmd = [sys.executable, '-m', __spec__.name, '--stage', 'worker', '--root', str(a.root),
               '--out', str(out), '--shard', str(shard), '--nshards', str(a.nshards)]
        log = out/f'worker_{shard:03d}.log'
        with log.open('x') as f:
            return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        codes = list(pool.map(job, range(a.group, a.nshards, a.groups)))
    write_json(out/'status.json', dict(passed=not any(codes), exit_codes=codes, protocol=ev.PROTOCOL))
    if any(codes):
        raise RuntimeError(f'Failed shards: {codes}')


def verify(root, extension):
    """Check exact row keys, shared exclusions, bounds, and frozen input hashes."""
    import pandas as pd
    _, units = inputs(root)
    paths = sorted(extension.glob('group_*/task_warm/worker_*.csv'))
    rows = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    expected = {ev._job_key(units[i], t, b, s) for i, t, b, s in extension_jobs(units)}
    assert not rows.duplicated(KEY).any()
    assert set(rows[KEY].itertuples(index=False, name=None)) == expected
    assert np.isfinite(rows.energy).all() and (rows.energy <= rows.budget+1e-6).all()
    assert set(rows.loc[rows.valid == 0, 'task_seed']) == {91007}
    assert (rows.valid == (rows.task_seed != 91007)).all()
    hashes = {str(p.relative_to(extension)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    write_json(extension/'verification.json', dict(passed=True, rows=len(rows), numerical_failures=int(rows.numerical_failure.sum()),
        excluded_task_seeds=[91007], files=hashes, protocol=ev.PROTOCOL))
    print(f'Validated {len(rows)} extension rows; {int(rows.numerical_failure.sum())} numerical failures')


def finalize(root, extension):
    """Install verified new cells, score one canonical table, and export final comparisons."""
    import pandas as pd
    verify(root, extension)
    assert json.loads((extension/'group_0/example_verification.json').read_text())['passed']
    for source in sorted(extension.glob('group_*/task_warm/worker_*.csv')):
        destination = root/'eval/task_warm'/source.name
        if destination.exists():
            assert source.read_bytes() == destination.read_bytes()
        else:
            shutil.copy2(source, destination)
    for source in sorted(extension.glob('group_*/failures/*.json')):
        destination = root/'eval/failures'/source.name
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(source, destination)
    for source, destination in [(extension/'group_0/examples', root/'eval/examples'),
                                (extension/'group_0/control_phase', root/'control_phase')]:
        shutil.copytree(source, destination, dirs_exist_ok=True)
    phase = json.loads((root/'control_phase/manifest.json').read_text())
    onset = next(r['onset'] for r in phase['records'] if r['policy'] == 'adaptive')
    learned_z = np.load(root/'control_phase/adaptive.npz')['z']
    spread_z = np.load(root/'control_phase/spread.npz')['z']
    np.testing.assert_allclose(learned_z[:int(onset)], spread_z[:int(onset)], rtol=1e-8, atol=1e-8)
    rows = pd.concat([pd.read_csv(p) for p in (root/'eval/task_warm').glob('worker_*.csv')], ignore_index=True)
    assert len(rows) == 9032 and not rows.duplicated(KEY).any()
    for _, row in rows[rows.numerical_failure == 1].iterrows():
        failure = json.loads((root/'eval'/row.failure_record).read_text())
        for key in ['valid','success','onset','outcome_bin','energy']:
            np.testing.assert_allclose(row[key], failure['result'][key], rtol=1e-12, atol=1e-12)
    ev.stage_score(root/'eval', 'warm')
    write_json(root/'eval/protocol.json', dict(protocol=ev.PROTOCOL, preset=ev.PRESET_ID,
        evaluated_grid={'checkpoints_at_budget_12': [0,1,5,10,15,20],
                        'final_model_budgets': [5,10,12,15]},
        no_control='one budget-independent evaluation at normalized budget 10'))
    summary = pd.read_csv(root/'eval/task_summary_warm.csv')
    final = summary[(summary.checkpoint == 20) | summary.policy.isin(['prior','reduced_fit','none','spread','front'])]
    final.to_csv(root/'all_baselines.csv', index=False)
    r2 = pd.read_csv(root/'eval/r2_per_session.csv')
    records = []
    for policy in [*POLICIES, 'prior', 'reduced_fit', 'none', 'spread', 'front']:
        record = dict(policy=policy)
        if policy in POLICIES or policy == 'prior':
            subset = r2[(r2.policy == ('adaptive' if policy == 'prior' else policy))
                        & (r2.sessions == (0 if policy == 'prior' else 20))]
            record.update(zip(['r2_q25','r2_median','r2_q75'], np.percentile(subset.r2,[25,50,75])))
        elif policy == 'reduced_fit':
            record['r2_median'] = json.loads((root/'reference/native_fit.json').read_text())['r2_test_piecewise_current']
        for budget in [5,10,12,15]:
            cell = final[(final.policy == policy) & (final.budget == (10 if policy == 'none' else budget))]
            assert len(cell) == 1
            record.update({f'success_{40*budget}_pa2s': float(cell.iloc[0].success),
                           f'sem_{40*budget}_pa2s': float(cell.iloc[0].sem_over_seeds)})
        records.append(record)
    pd.DataFrame(records).to_csv(root/'main_table.csv', index=False)
    write_json(root/'provenance/final_evaluation_verification.json', dict(passed=True, rows=len(rows),
        numerical_failures=int(rows.numerical_failure.sum()), prediction_rows=len(r2), training_runs=120,
        excluded_task_seeds=[91007], extension_sha256=hashlib.sha256((extension/'verification.json').read_bytes()).hexdigest()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['worker', 'campaign', 'verify', 'examples', 'finalize'], required=True)
    p.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--group', type=int, default=0); p.add_argument('--groups', type=int, default=4)
    p.add_argument('--workers', type=int, default=8); p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshards', type=int, default=32)
    a = p.parse_args(); torch.set_num_threads(1)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    if a.stage == 'worker': worker(a.root, a.out, a.shard, a.nshards)
    elif a.stage == 'examples': examples(a.root, a.out)
    elif a.stage == 'verify': verify(a.root, a.out)
    elif a.stage == 'finalize': finalize(a.root, a.out)
    else: campaign(a)


if __name__ == '__main__':
    main()
