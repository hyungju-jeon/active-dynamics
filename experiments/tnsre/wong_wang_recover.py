"""Finish interrupted control shards after the frozen campaign releases its workers.

Original source, outputs, and job statuses stay intact. The recovery snapshot
only adds explicit numerical-failure outcomes and completes unattempted cases.
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
import time

from experiments.tnsre.wong_wang_full import EVAL_ARMS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    status_path = root / 'recovery.json'
    state = {'status': 'waiting', 'source': str(Path.cwd()), 'created': datetime.now(timezone.utc).isoformat(),
             'policy': 'Keep numerical crashes as task outcomes; do not change models, filters, or planners.', 'jobs': {}}

    def save():
        temp = status_path.with_suffix('.tmp')
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(status_path)

    save()
    try:
        while True:
            original = json.loads((root / 'campaign.json').read_text())
            if original['status'] != 'running':
                break
            time.sleep(30)
        if original['status'] == 'completed':
            state.update(status='completed', original_status='completed')
            save()
            return
        if original.get('failed_phase') != 'control':
            raise RuntimeError(f'Unexpected campaign failure: {original}')
        original_jobs = json.loads((root / 'jobs.json').read_text())
        keys = original['failed_jobs']
        if any(not key.startswith('task_') for key in keys):
            raise RuntimeError('Recovery is limited to interrupted control shards')
        state.update(status='running', original_status='failed', original_failed_jobs=keys)
        save()
        env = os.environ | {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                            'NUMEXPR_NUM_THREADS': '1', 'MPLBACKEND': 'Agg', 'PYTHONUNBUFFERED': '1'}
        logs = root / 'recovery' / 'logs'
        logs.mkdir(parents=True, exist_ok=True)

        def run(key):
            old = original_jobs[key]['command']
            cmd = [sys.executable, *old[1:], '--recover-numerical-errors']
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
            log = logs / f'{key}.{stamp}.log'
            started = time.monotonic()
            with log.open('w') as f:
                result = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
            return {'command': cmd, 'log': str(log), 'exit_code': result.returncode,
                    'seconds': time.monotonic()-started, 'status': 'completed' if result.returncode == 0 else 'failed'}

        with ThreadPoolExecutor(max_workers=min(32, len(keys))) as pool:
            futures = {pool.submit(run, key): key for key in keys}
            for future in as_completed(futures):
                state['jobs'][futures[future]] = future.result()
                save()
        if any(job['status'] != 'completed' for job in state['jobs'].values()):
            raise RuntimeError('Recovery job failed; see retained attempt logs')
        with (logs / 'postprocess.log').open('w') as f:
            for arm in EVAL_ARMS:
                subprocess.run([sys.executable, '-m', 'experiments.tnsre.wong_wang_full', '--stage', 'score',
                                '--root', str(root), '--arm', arm], env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
            subprocess.run([sys.executable, '-m', 'experiments.tnsre.wong_wang_compare', '--root', str(root)],
                           env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
        state.update(status='completed', end=datetime.now(timezone.utc).isoformat())
        save()
    except Exception as exc:
        state.update(status='failed', error=repr(exc), end=datetime.now(timezone.utc).isoformat())
        save()
        raise


if __name__ == '__main__':
    main()
