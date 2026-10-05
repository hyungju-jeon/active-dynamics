"""Model-implied per-bin Fisher SNR of the calibrated SNN Poisson readouts.

Uses all 12 held-out network trajectories (400 post-step states each), with the
same centered latent ensemble for every readout. This is not empirical SNN SNR.
Run: .venv/bin/python -m experiments.tnsre.spiking_readout_snr
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np
from actdyn.utils.experiment_runtime import _neurofisher_snr_bound


def fisher_snr(states, c, bias, dt):
    """Centered-state power / average inverse-Fisher trace, in dB.

    Centering changes the bias to preserve all count means. The ridge and
    inverse-Fisher handling are those of the repository's NeuroFisherSNR helper.
    """
    mean = states.mean(0)
    x = states - mean
    shifted_bias = bias.reshape(-1) + c @ mean + np.log(dt)
    fn = _neurofisher_snr_bound()
    score = float(fn(x, c.T, shifted_bias))
    rates = np.exp(states @ c.T + bias.reshape(-1) + np.log(dt))
    info = np.einsum('ti,ij,ik->tjk', rates, c, c)
    eig = np.linalg.eigvalsh(info)
    assert np.all(eig > 0), 'Readout Fisher matrix is singular'
    noise = np.linalg.inv(info).trace(axis1=-2,axis2=-1).mean()
    unregularized = float(10*np.log10(np.mean(np.sum(x*x,axis=1))/noise))
    return score, unregularized, float(eig.min())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir',type=Path,default=Path('results/tnsre/20260924_snn_sessions_m2'))
    parser.add_argument('--output-dir',type=Path,default=Path('results/tnsre/20260929_manuscript_update/snr'))
    args=parser.parse_args()
    state_path=args.experiment_dir/'data/test_trajectories.npz'
    states=np.load(state_path)['states'][:,1:,:].reshape(-1,2)
    root=args.experiment_dir/'tracks/wong_wang_snn_sessions_m2'
    paths=sorted(root.glob('*/seed_*/repeat_01/run_metadata.json'))
    records=[(p,json.loads(p.read_text())) for p in paths]
    assert len(records)==120 and all(m['status']=='completed' for p,m in records)
    rows=[]
    for seed in range(20):
        matched=[m for p,m in records if int(m['seed'])==seed];assert len(matched)==6
        c=np.asarray(matched[0]['observation_loading_matrix']);bias=np.asarray(matched[0]['observation_loading_bias'])
        dt=float(matched[0]['dt'])
        for m in matched[1:]:
            np.testing.assert_array_equal(c,m['observation_loading_matrix'])
            np.testing.assert_array_equal(bias,m['observation_loading_bias'])
            assert float(m['dt'])==dt
        value,unreg,mineig=fisher_snr(states,c,bias,dt)
        rows.append(dict(seed=seed,snr_db=value,unregularized_snr_db=unreg,min_fisher_eigenvalue=mineig))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with (args.output_dir/'readout_snr.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    values=np.array([r['snr_db'] for r in rows])
    summary=dict(median_db=float(np.median(values)),iqr_db=np.percentile(values,[25,75]).tolist(),
                 range_db=[float(values.min()),float(values.max())],states=states.shape[0],training_seeds=20,
                 models_matched_across_agents=True,dt_normalized=dt,bin_ms=5,
                 interpretation='Poisson-readout-implied Fisher SNR on centered held-out states; not empirical network SNR',
                 max_regularization_effect_db=max(abs(r['snr_db']-r['unregularized_snr_db']) for r in rows),
                 inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [state_path,*paths]})
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2))
    print({k:v for k,v in summary.items() if k!='inputs'})

if __name__=='__main__':main()
