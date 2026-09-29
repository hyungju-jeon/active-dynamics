"""Numerical contracts for the new manuscript diagnostics."""
import json
from pathlib import Path
import numpy as np
import torch
from experiments.tnsre.figures.spiking_characterization import reduced_drift, fixed_points
from actdyn.utils.vectorfields_eqn import WongWangInsideGain

FIT=Path('results/tnsre/20260924_snn_sessions_m2/reference/m2_fit.json')

def test_phase_portrait_matches_implemented_reduced_dynamics():
    record=json.loads(FIT.read_text())
    z=torch.tensor([[-2.,-2.],[-1.,.5],[.1,-.8],[2.,2.]],dtype=torch.float64)
    model=WongWangInsideGain(dyn_param=torch.tensor(record['learner_parameters']['values'],dtype=torch.float64))
    expected=model.compute(z).detach().numpy()
    np.testing.assert_allclose(reduced_drift(z.numpy(),record['physical_parameters']),expected,rtol=1e-6,atol=1e-7)
    points=fixed_points(record['physical_parameters'])
    assert len(points)==5 and sum(p['stable'] for p in points)==3
    assert max(p['residual'] for p in points)<1e-8
