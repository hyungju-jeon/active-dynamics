"""The manuscript extension adds missing cells without repeating old trials."""
from copy import deepcopy
import numpy as np
from experiments.tnsre import eval_spiking_sessions as ev
from experiments.tnsre import native_manuscript as study


def test_extension_covers_all_methods_seeds_and_missing_cells(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, 'PROTOCOL', deepcopy(ev.PROTOCOL))
    monkeypatch.setattr(ev, 'TASKS', ev.TASKS)
    monkeypatch.setattr(ev, 'PRESET_ID', ev.PRESET_ID)
    monkeypatch.setattr(ev, 'EXAMPLES', deepcopy(ev.EXAMPLES))
    (tmp_path/'tracks/native').mkdir(parents=True)
    study.configure(tmp_path)
    runs = [dict(policy=p, seed=s, est={k: np.zeros(5) for k in range(21)},
                 C=np.zeros((40, 2)), b=np.zeros(40), dynamics_type='wong_wang_transfer',
                 full_params=np.zeros(5), min_embedding_dim=5)
            for p in study.POLICIES for s in range(20)]
    units = ev.controller_units(runs, theta_fit=np.zeros(5))
    jobs = study.extension_jobs(units)
    keys = [ev._job_key(units[i], t, b, s) for i, t, b, s in jobs]
    assert len(keys) == len(set(keys)) == 6432
    for policy in study.POLICIES:
        for seed in range(20):
            actual = {(k[3], k[5], k[6]) for k in keys if k[1:3] == (policy, seed)}
            expected = {(c, b, s) for c, b in [(1,12.),(5,12.),(10,12.),(15,12.),(20,5.),(20,15.)]
                        for s in range(91000,91008)}
            assert actual == expected
    assert not any(k[5] in (10.,12.) and k[3] in (-1,0,20) for k in keys)
    assert set(k[1] for k in keys) == set(study.POLICIES) | {'prior','reduced_fit','spread','front'}
