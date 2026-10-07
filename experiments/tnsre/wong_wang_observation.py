"""Matched native-transfer/targeting test under the updated SNN manuscript protocol.

Four response-family x targeting arms; 20+20 recorded neurons; ten evidence-led
trials followed by ten fixed 2-s self-exploration trials. Within-session planning.
PALDI uses the manuscript's 200-ms horizon, 10-ms step and 15-ms minimum cadence.
All six identification methods are refitted. Control is evaluated after trial 20
at manuscript budgets 10 and 12, on the same fixed eight task starts per arm.
No control outcome selects a model, setting, or seed. Older runs remain unchanged.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
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
from experiments.tnsre import wong_wang_full as full
from experiments.tnsre.run_tbme_experiments import configure_tbme_catalogs

ARMS=full.TRAIN_ARMS
POLICIES=['adaptive_c2h20','active_myopic','flex_rollback','rhc','prbs','random']
SUITES={arm:{**full.SUITES[arm], 'env_preset_id':f'tbme_ww_observation_{arm}', 'model_ids':POLICIES} for arm in ARMS}
SOURCE_PROTOCOL_COMMIT='a9aa9d8'


def now():return datetime.now(timezone.utc).isoformat()


def configure(root: Path,arm: str):
    """Select the complete protocol before each isolated train/evaluation process."""
    ev.PROTOCOL['task_checkpoints']=[0,20]
    ev.PROTOCOL['budgets']['overturn']=[10.,12.]
    tracks=root/'tracks'/f'ww_observation_{arm}'
    full.configure_eval(root,arm+'_current',preset_id=SUITES[arm]['env_preset_id'],runs_root=tracks)
    return tracks


def train(root: Path,arm: str,policy: str,seed: int,steps=8000):
    from experiments import run
    configure_tbme_catalogs(suite_entries=SUITES)
    args=run.build_parser().parse_args(['--base-dir',str(root),'--path-layout','tbme_tracks','--total-steps',str(steps),'--skip-existing'])
    result=run._run_one(exp_id=arm,policy_id=policy,seed=seed,repeat=1,base_dir=root,args=args)
    if result['status']!='completed' or result.get('nan_detected'):
        raise RuntimeError(f'Identification failed: {result.get("error",result)}')
    if steps==8000 and len(result.get('session_log',[]))!=20:
        raise RuntimeError('Identification did not complete all 20 trials')


def verify_training(root: Path,arm: str):
    """Check completed cohorts, fixed observation maps, and the 10+10 trial rule."""
    tracks=configure(root,arm);runs=ev.load_session_runs(tracks)
    assert len(runs)==120
    assert {(r['policy'],r['seed']) for r in runs}=={(ev.evaluated_policy(p),s) for p in POLICIES for s in range(20)}
    maps={};metadata=[]
    for p in sorted(tracks.glob('*/seed_*/repeat_*/run_metadata.json')):
        m=json.loads(p.read_text());r=next(x for x in runs if x['policy']==ev.evaluated_policy(m['policy_id']) and x['seed']==m['seed'])
        assert set(r['est'])==set(range(21))
        assert r['C'].shape==(40,2) and r['b'].shape==(40,)
        assert np.isfinite(r['C']).all() and np.isfinite(r['b']).all()
        assert len(r['session_log'])==20
        assert all(x['bins']==400 for x in r['session_log'][10:])
        assert all(x['evidence_pool'] in (0,1) for x in r['session_log'][:10])
        assert all(x['evidence_pool']==-2 for x in r['session_log'][10:])
        blob=np.concatenate([r['C'].ravel(),r['b']]).tobytes();key=hashlib.sha256(blob).hexdigest()
        if r['seed'] in maps:assert maps[r['seed']]==key
        maps[r['seed']]=key
        metadata.append({'policy':r['policy'],'seed':r['seed'],'readout_sha256':key,'trials':20,'bins':r['bins'][20]})
    full.write_json(root/'training_verification.json',dict(passed=True,runs=120,policies=POLICIES,readouts_by_seed=maps,metadata=metadata))


def verify_tasks(root: Path,arm: str):
    import pandas as pd
    tracks=configure(root,arm);runs=ev.load_session_runs(tracks);units=ev.controller_units(runs)
    expected={ev._job_key(units[i],task,budget,seed) for i,task,budget,seed in ev.task_jobs(units)}
    files=sorted((root/'eval'/f'{arm}_current'/'task_warm').glob('worker_*.csv'))
    rows=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
    cols=['controller','policy','seed','checkpoint','task','budget','task_seed']
    actual=set(rows[cols].itertuples(index=False,name=None))
    assert not rows.duplicated(cols).any()
    assert actual==expected,(len(actual),len(expected))
    assert np.isfinite(rows.energy).all() and (rows.energy<=rows.budget+1e-6).all()
    assert set(rows.policy)==set(map(ev.evaluated_policy,POLICIES+['prior','reduced_fit','none','front','spread']))
    full.write_json(root/'control_verification.json',dict(passed=True,rows=len(rows),expected_rows=len(expected),valid=int(rows.valid.sum()),numerical_failures=int(rows.numerical_failure.sum()),policies=sorted(rows.policy.unique()),task_seeds=sorted(rows.task_seed.unique().tolist()),budgets=[10.,12.]))


def worker(a):
    torch.set_num_threads(1);root=a.root.resolve();root.mkdir(parents=True,exist_ok=True)
    tracks=configure(root,a.arm);preset=SUITES[a.arm]['env_preset_id']
    if a.stage=='train':train(root,a.arm,a.policy,a.seed,a.steps)
    elif a.stage=='data':full.data(root,a.arm.split('_',1)[1]+'_current',preset_id=preset)
    elif a.stage=='reference':full.reference(root,a.arm+'_current')
    elif a.stage=='r2':ev.stage_r2(root/'eval'/f'{a.arm}_current',tracks)
    elif a.stage=='task':full.task(root,a.arm+'_current',a.shard,a.nshards,recover_numerical_errors=True,preset_id=preset,runs_root=tracks)
    elif a.stage=='score':
        ev.stage_score(root/'eval'/f'{a.arm}_current','warm');verify_tasks(root,a.arm)
    elif a.stage=='verify_training':verify_training(root,a.arm)


def campaign(a):
    """One VM per arm, eight bounded worker processes, durable immutable job records."""
    # Check the real baseline dependency before spending time on any training.
    from actdyn.policy.baseline_flex import _flex_policy_class
    _flex_policy_class()
    root=a.root.resolve();root.mkdir(parents=True,exist_ok=True)
    ledger=root/'jobs.json';records=json.loads(ledger.read_text()) if ledger.exists() else {};lock=threading.Lock()
    env=os.environ|{k:'1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']}|{'PYTHONUNBUFFERED':'1','MPLBACKEND':'Agg'}
    def save():
        tmp=ledger.with_suffix('.tmp');tmp.write_text(json.dumps(records,indent=2));tmp.replace(ledger)
    def job(spec):
        key,argv=spec
        with lock:
            old=records.get(key,{})
            if old.get('status')=='completed':return True
            attempt=old.get('attempt',0)+1;log=root/'logs'/f'{key}.attempt{attempt}.log';log.parent.mkdir(exist_ok=True)
            command=[sys.executable,'-m',__spec__.name,'--root',str(root),'--arm',a.arm,*argv]
            records[key]={'command':command,'status':'running','started':now(),'attempt':attempt,'log':str(log),'previous_attempts':old.get('previous_attempts',[])+([old] if old else [])};save()
        start=time.monotonic()
        with log.open('w') as f:result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env=env)
        with lock:
            records[key].update(status='completed' if result.returncode==0 else 'failed',exit_code=result.returncode,seconds=time.monotonic()-start,finished=now());save()
        return result.returncode==0
    def phase(name,jobs,workers):
        full.write_json(root/'status.json',dict(status='running',phase=name,started=now(),jobs=len(jobs)))
        failures=[]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(job,spec):spec[0] for spec in jobs}
            for f in as_completed(futures):
                if not f.result():failures.append(futures[f])
        if failures:
            full.write_json(root/'status.json',dict(status='failed',phase=name,failures=failures,finished=now()))
            raise RuntimeError(f'{name}: {failures}')
        print(f'{now()} {a.arm} {name}: {len(jobs)} completed',flush=True)
    protocol=dict(arm=a.arm,preset=SUITES[a.arm]['env_preset_id'],source_protocol_commit=SOURCE_PROTOCOL_COMMIT,
        training_seeds=list(range(20)),policies=POLICIES,trials=20,observation_per_pool=20,
        first_ten='decision-ended; current evidence 0.3 for 200 bins',last_ten='fixed 400 bins; no evidence',
        paldi=dict(policy='adaptive_c2h20',planning_horizon=20,coarse_dt_factor=2,adaptive_update_min_interval=3,adaptive_replan_min_interval=3),
        evaluation=dict(ev.PROTOCOL),control_task='overturn',evaluation_arm=a.arm+'_current',
        comparison='Four factorial arms; no performance selection; all six methods plus prior, fitted model, no control, constant and maximum pulses.',
        workers=a.workers)
    configure(root,a.arm);protocol['evaluation']=dict(ev.PROTOCOL);full.write_json(root/'study_protocol.json',protocol)
    # A complete seed-0 PALDI run exercises calibration, all 20 trials, and the switch.
    phase('preflight',[(f'train_adaptive_c2h20_00',['--stage','train','--policy','adaptive_c2h20','--seed','0'])],1)
    phase('identification_and_data',[('data',['--stage','data'])]+[(f'train_{p}_{s:02d}',['--stage','train','--policy',p,'--seed',str(s)]) for s in range(20) for p in POLICIES],a.workers)
    phase('training_checks',[('verify_training',['--stage','verify_training'])],1)
    phase('prediction_and_reference',[('r2',['--stage','r2']),('reference',['--stage','reference'])],2)
    phase('control',[(f'task_{s:03d}',['--stage','task','--shard',str(s),'--nshards',str(a.nshards)]) for s in range(a.nshards)],a.workers)
    phase('score',[('score',['--stage','score'])],1)
    full.write_json(root/'status.json',dict(status='completed',finished=now()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',default='campaign',choices=['campaign','train','data','reference','r2','task','score','verify_training'])
    p.add_argument('--root',required=True,type=Path);p.add_argument('--arm',required=True,choices=ARMS)
    p.add_argument('--policy',choices=POLICIES,default='adaptive_c2h20');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--nshards',type=int,default=32);p.add_argument('--shard',type=int,default=0);p.add_argument('--steps',type=int,default=8000)
    a=p.parse_args();campaign(a) if a.stage=='campaign' else worker(a)

if __name__=='__main__':main()
