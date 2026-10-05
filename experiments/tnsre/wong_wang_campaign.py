"""Durable, bounded Lambda campaign for the full Wong-Wang manuscript comparison.

Every job has an immutable command and attempt log. A failed phase stops the
pipeline, preserving all outputs. Successful jobs are reused on restart.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from experiments.tnsre.wong_wang_full import TRAIN_ARMS, INPUT_ARMS, EVAL_ARMS, MODEL_IDS


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--workers', type=int, default=32)
    p.add_argument('--task-shards', type=int, default=32)
    a = p.parse_args()
    root = a.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    ledger = root / 'jobs.json'
    records = json.loads(ledger.read_text()) if ledger.exists() else {}
    lock = threading.Lock()
    started = time.monotonic()
    env = os.environ | {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                        'NUMEXPR_NUM_THREADS': '1', 'PYTHONUNBUFFERED': '1', 'MPLBACKEND': 'Agg'}

    def save():
        tmp = ledger.with_suffix('.tmp')
        tmp.write_text(json.dumps(records, indent=2))
        tmp.replace(ledger)

    def run(spec):
        key, args = spec
        with lock:
            previous = records.get(key, {})
            if previous.get('status') == 'completed':
                return True
            attempt = previous.get('attempt', 0) + 1
            log = root / 'logs' / f'{key}.attempt_{attempt}.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, '-m', 'experiments.tnsre.wong_wang_full', '--root', str(root), *args]
            records[key] = {'status': 'running', 'start': now(), 'command': command, 'log': str(log), 'attempt': attempt,
                            'previous_attempts': previous.get('previous_attempts', []) + ([previous] if previous else [])}
            save()
        t = time.monotonic()
        with log.open('w') as f:
            result = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, env=env)
        with lock:
            records[key].update(status='completed' if result.returncode == 0 else 'failed',
                                exit_code=result.returncode, end=now(), seconds=time.monotonic() - t)
            save()
        return result.returncode == 0

    phases = {
        'identification_and_data': [
            (f'data_{arm}', ['--stage', 'data', '--arm', arm]) for arm in INPUT_ARMS
        ] + [
            (f'train_{arm}_{policy}_{seed:02d}', ['--stage', 'train', '--arm', arm, '--policy', policy, '--seed', str(seed)])
            for seed in range(20) for policy in MODEL_IDS for arm in TRAIN_ARMS
        ],
        'prediction_and_reference': [
            (f'r2_{arm}', ['--stage', 'r2', '--arm', arm]) for arm in EVAL_ARMS if arm.endswith('current')
        ] + [
            (f'reference_{arm}', ['--stage', 'reference', '--arm', arm]) for arm in EVAL_ARMS
        ],
        'control': [
            (f'task_{arm}_{shard:03d}', ['--stage', 'task', '--arm', arm, '--shard', str(shard), '--nshards', str(a.task_shards)])
            for shard in range(a.task_shards) for arm in EVAL_ARMS
        ],
        'score': [(f'score_{arm}', ['--stage', 'score', '--arm', arm]) for arm in EVAL_ARMS],
    }
    manifest = {'created': now(), 'source_dir': str(Path.cwd()), 'python': sys.executable,
                'workers': a.workers, 'task_shards': a.task_shards,
                'training_runs': 480, 'trials_per_run': 20, 'task_sessions': 66880,
                'policies': MODEL_IDS, 'training_seeds': list(range(20)), 'task_seeds': list(range(91000, 91008)),
                'task': 'overturn', 'budgets': [10., 12.], 'checkpoints': [0, 1, 2, 5, 20],
                'commands': {phase: dict(jobs) for phase, jobs in phases.items()},
                'status': 'running'}
    manifest_path = root / 'campaign.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    for phase, jobs in phases.items():
        print(f'{now()} {phase}: {len(jobs)} jobs', flush=True)
        failures = []
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures = {pool.submit(run, job): job[0] for job in jobs}
            for completed, f in enumerate(as_completed(futures), 1):
                try:
                    ok = f.result()
                except Exception as exc:
                    ok = False
                    print(f'{futures[f]}: {exc!r}', flush=True)
                if not ok:
                    failures.append(futures[f])
                if completed % 8 == 0 or not ok or completed == len(jobs):
                    print(f'{now()} {phase}: {completed}/{len(jobs)}; failures={len(failures)}', flush=True)
        if failures:
            manifest.update(status='failed', failed_phase=phase, failed_jobs=failures, end=now())
            manifest_path.write_text(json.dumps(manifest, indent=2))
            raise RuntimeError(f'{phase}: {len(failures)} failed jobs; see preserved logs')
    report_log = root / 'logs' / 'comparison.log'
    with report_log.open('w') as f:
        report = subprocess.run([sys.executable, '-m', 'experiments.tnsre.wong_wang_compare', '--root', str(root)],
                                stdout=f, stderr=subprocess.STDOUT, env=env)
    if report.returncode:
        manifest.update(status='failed', failed_phase='comparison', end=now())
        manifest_path.write_text(json.dumps(manifest, indent=2))
        raise RuntimeError(f'Comparison failed: {report_log}')
    manifest.update(status='completed', end=now(), seconds=time.monotonic() - started)
    manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
