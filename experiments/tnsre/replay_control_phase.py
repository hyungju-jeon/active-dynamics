"""Replay the existing Figure 6 trial with the saved active-agent models.

This records trajectories, not a new performance evaluation. Every replay must
match its saved task outcome before its trajectory is marked as verified.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np
import torch
from experiments.tnsre.eval_spiking_sessions import (
    EXAMPLES, PROTOCOL, IcemController, build_network, load_session_runs, run_task_session,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir', type=Path,
                        default=Path('results/tnsre/20260924_snn_sessions_m2'))
    parser.add_argument('--output-dir', type=Path,
                        default=Path('results/tnsre/20260929_manuscript_update/control_phase_uniform'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = args.experiment_dir
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    seed = int(EXAMPLES['training_seed'])
    checkpoint = int(EXAMPLES['checkpoint'])
    task_seed, budget = EXAMPLES['sessions']['overturn']
    policies = ['adaptive', 'spread']
    runs = {(r['policy'], r['seed']): r for r in load_session_runs(root/'tracks/wong_wang_snn_sessions_m2')}
    sources = sorted((root/'eval/task_warm').glob('worker_*.csv')) + sorted((root/'eval/task').glob('worker_*.csv'))
    rows = []
    for path in sources:
        with path.open() as f:
            rows.extend(r for r in csv.DictReader(f)
                        if path.parent.name=='task_warm' or r['policy']=='spread')
    expected = {r['policy']: r for r in rows if r['task']=='overturn'
                and ((r['policy']=='spread' and int(r['seed'])==-1) or
                     (r['controller']=='icem' and int(r['seed'])==seed and int(r['checkpoint'])==checkpoint))
                and int(r['task_seed'])==task_seed and float(r['budget'])==budget}
    saved = np.load(root/'eval/examples/example_sessions.npz')
    records = []
    net = build_network()
    try:
        for policy in policies:
            model = None
            if policy != 'spread':
                run = runs[(policy, seed)]
                model = IcemController(run['est'][checkpoint], run['C'], run['b'], net.dt_latent,
                                       seed=task_seed, warm_start=True,
                                       dynamics_type=run['dynamics_type'], full_params=run['full_params'],
                                       min_embedding_dim=run['min_embedding_dim'])
            trace = []
            print(f'Replaying {policy}, training seed {seed}, task seed {task_seed}', flush=True)
            result = run_task_session(net, 'overturn', task_seed, budget, 'icem' if model else 'spread', model, trace)
            arrays = {key: np.stack([r[key] for r in trace]) for key in ('z','u','evidence')}
            np.savez(out/f'{policy}.npz', **arrays)
            (out/f'{policy}.json').write_text(json.dumps(result, indent=2))
            for key in ('valid','target','onset','success','outcome_bin'):
                assert int(result[key]) == int(expected[policy][key]), (policy, key, result, expected[policy])
            np.testing.assert_allclose(result['energy'], float(expected[policy]['energy']), rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(arrays['z'][:result['onset']], saved['overturn_learned_z'][:result['onset']], atol=1e-8, rtol=1e-8)
            if policy == 'adaptive':
                np.testing.assert_allclose(arrays['z'], saved['overturn_learned_z'], atol=1e-5, rtol=1e-5)
            records.append(dict(policy=policy, **result))
            print(f'Verified {policy}: success={result["success"]}, bins={len(trace)}', flush=True)
    finally:
        net.close()
    inputs = [*sources, root/'eval/examples/example_sessions.npz']
    inputs += [p for p in (root/'tracks/wong_wang_snn_sessions_m2').glob('*/seed_0/repeat_01/*')
               if p.name in ('run_metadata.json','embedding_estimate_trace.csv') and p.parts[-4] in policies]
    inputs += [Path(__file__), Path('experiments/tnsre/eval_spiking_sessions.py'),
               Path('actdyn/environment/spiking_decision.py'),
               Path('experiments/tnsre/config/experiment_env.yaml')]
    manifest = dict(protocol=PROTOCOL, training_seed=seed, task_seed=task_seed, checkpoint=checkpoint,
                    budget_normalized=budget, budget_pa2_s=40*budget,
                    bin_ms=5, verified=True, records=records,
                    inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
