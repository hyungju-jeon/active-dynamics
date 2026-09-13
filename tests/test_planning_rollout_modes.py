"""Independent Gaussian references for the two parameter-information rollouts."""

import json
from types import SimpleNamespace

import pytest
import torch

from actdyn.metrics.information import EmbeddingFisherMetric
from actdyn.metrics.objectives import EOptimalityMetric
from actdyn.metrics.planning import conditioned_measurement_update


class GaussianDecoder:
    def __init__(self, h, r):
        self.h = h
        self.r = r
        self.noise = object()

    def __call__(self, z):
        return z @ self.h.to(z).T

    def jacobian(self, z):
        return self.h.to(z).expand(*z.shape[:-1], *self.h.shape)

    def var(self, z):
        return self.r.diagonal().to(z).expand(*z.shape[:-1], self.r.shape[-1])


def gaussian_problem(horizon):
    """Two candidate sequences with correlated state and parameter priors."""
    a = torch.tensor([[.8, .2, 1.1], [0., .7, -.3], [0., 0., 1.]])
    b = torch.tensor([[1., .2], [.1, .5], [0., 0.]])
    h = torch.tensor([[1., .2, 0.], [0., 1., .05]])
    r = torch.diag(torch.tensor([.7, 1.3]))
    p = torch.tensor([[1.2, .1, .2], [.1, .8, -.1], [.2, -.1, 1.1]])
    theta = torch.tensor([[.7, .2], [.2, 1.3]])
    q = torch.diag(torch.tensor([.02, .03, .01]))
    # Candidate-dependent, time-varying input gain; shared dynamics and nuisance.
    scales = torch.stack([torch.linspace(.5, 1., horizon), torch.linspace(1.4, .8, horizon)])
    z = torch.zeros(2, horizon, 3)
    z[..., 0] = scales
    model = SimpleNamespace(
        e={"m": torch.zeros(1, 2), "P": theta.unsqueeze(0)},
        z={"m": torch.zeros(1, 1, 3), "P": p.unsqueeze(0)},
        dynamics=SimpleNamespace(logvar=torch.log(torch.expm1(q.diagonal())).unsqueeze(0)),
        decoder=GaussianDecoder(h, r), dt=1.,
    )

    def fe(z, e):
        return z[..., :1, None] * b.to(z)

    def fz(z, e):
        return (a - torch.eye(3)).to(z).expand(*z.shape[:-1], 3, 3)

    return model, fe, fz, {"model_state": z, "next_model_state": z.clone()}, (a, b, q, h, r, p, theta, scales)


def augmented_reference(problem, *, condition):
    """Infer static theta jointly with x; never propagate a sensitivity matrix.

    For prediction-only, sum the Fisher matrices of the separate observation
    marginals, as the initial objective does. For conditioned planning, retain
    cross-covariances and condition the full augmented Gaussian at every step.
    """
    a, b, q, h, r, p, theta, scales = [x.double() for x in problem]
    d, m = b.shape
    obs = torch.cat([h, torch.zeros(h.shape[0], m)], dim=1)
    process = torch.block_diag(q, torch.zeros(m, m))
    theta_inv = torch.linalg.inv(theta)
    chol_theta = torch.linalg.cholesky(theta)
    gains, weakest = [], []
    for candidate in scales:
        joint = torch.block_diag(p, theta)
        marginal_info = torch.zeros(m, m, dtype=torch.float64)
        for scale in candidate:
            transition = torch.cat([
                torch.cat([a, scale * b], dim=1),
                torch.cat([torch.zeros(m, d), torch.eye(m)], dim=1),
            ])
            joint = transition @ joint @ transition.T + process
            if condition:
                innovation = obs @ joint @ obs.T + r
                cross = joint @ obs.T
                joint = joint - cross @ torch.linalg.solve(innovation, cross.T)
            else:
                output_cross = (obs @ joint)[:, d:]
                mean_coefficient = output_cross @ theta_inv
                conditional_variance = (
                    obs @ joint @ obs.T + r - output_cross @ theta_inv @ output_cross.T
                )
                marginal_info += mean_coefficient.T @ torch.linalg.solve(
                    conditional_variance, mean_coefficient
                )
        if condition:
            posterior = joint[d:, d:]
            information = torch.linalg.inv(posterior) - theta_inv
            gain = .5 * (torch.logdet(theta) - torch.logdet(posterior))
        else:
            information = marginal_info
            gain = .5 * torch.logdet(torch.eye(m) + chol_theta.T @ information @ chol_theta)
        scaled = chol_theta.T @ information @ chol_theta
        gains.append(gain)
        weakest.append(torch.linalg.eigvalsh((scaled + scaled.T) / 2)[0])
    return torch.stack(gains), torch.stack(weakest)


@pytest.mark.parametrize("horizon", [1, 4, 40])
@pytest.mark.parametrize("mode", ["prediction_only", "measurement_conditioned"])
def test_objectives_match_independent_augmented_gaussian(horizon, mode):
    model, fe, fz, rollout, problem = gaussian_problem(horizon)
    expected_eig, expected_eopt = augmented_reference(
        problem, condition=mode == "measurement_conditioned"
    )
    before = {k: {name: value.clone() for name, value in belief.items()}
              for k, belief in (("z", model.z), ("e", model.e))}
    for cls, expected in ((EmbeddingFisherMetric, expected_eig), (EOptimalityMetric, expected_eopt)):
        metric = cls(model=model, Fe_net=fe, Fz_net=fz, gamma=1., device="cpu", planning_rollout=mode)
        actual = -metric.compute_stepwise(rollout).flatten()
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual.double(), expected, atol=2e-5, rtol=2e-5)
        for belief_name, belief in before.items():
            for name, value in belief.items():
                torch.testing.assert_close(getattr(model, belief_name)[name], value)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("observed", [False, True])
def test_measurement_update_matches_kalman_with_rank_deficient_information(dtype, observed):
    p = torch.tensor([[2., .4], [.4, .7]], dtype=dtype).expand(3, -1, -1)
    h = torch.tensor([[1., .3]], dtype=dtype) * float(observed)
    r = torch.tensor([[.6]], dtype=dtype)
    s = torch.tensor([[1., .2, .5], [-.4, 1., .2]], dtype=dtype).expand(3, -1, -1)
    info = h.T @ torch.linalg.solve(r, h)
    k = torch.linalg.solve(h @ p @ h.T + r, h @ p).transpose(-1, -2)
    expected_cov = p - k @ h @ p
    expected_sens = (torch.eye(2, dtype=dtype) - k @ h) @ s
    actual_cov, actual_sens = conditioned_measurement_update(p, s, info)
    assert actual_cov.dtype == actual_sens.dtype == dtype
    torch.testing.assert_close(actual_cov, expected_cov)
    torch.testing.assert_close(actual_sens, expected_sens)


def test_strong_measurement_does_not_cancel_small_sensitivity():
    p = torch.ones(1, 1)
    s = torch.ones(1, 1)
    info = torch.full((1, 1), 1e8)
    cov, sens = conditioned_measurement_update(p, s, info)
    expected = torch.full((1, 1), 1. / (1. + 1e8))
    torch.testing.assert_close(cov, expected, atol=0., rtol=1e-6)
    torch.testing.assert_close(sens, expected, atol=0., rtol=1e-6)


def test_conditioned_measurement_update_has_finite_correct_gradients():
    p = torch.tensor([[2., .3], [.3, 1.]], dtype=torch.float64)
    s = torch.tensor([[1.], [.5]], dtype=torch.float64, requires_grad=True)
    c = torch.tensor([[.8, .2]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda sensitivity, loading: conditioned_measurement_update(p, sensitivity, loading.T @ loading),
        (s, c),
    )


@pytest.mark.parametrize("flag", ["freeze_covariance", "fully_observed"])
def test_scoring_ablation_retains_measurement_conditioning(flag):
    model, fe, fz, rollout, _ = gaussian_problem(4)
    costs = []
    for mode in ("prediction_only", "measurement_conditioned"):
        metric = EmbeddingFisherMetric(
            model=model, Fe_net=fe, Fz_net=fz, gamma=1., device="cpu",
            planning_rollout=mode, **{flag: True},
        )
        costs.append(metric.compute_stepwise(rollout))
    assert not torch.allclose(*costs)


@pytest.mark.parametrize("kind", [
    "parameter_eig", "shrinkage_parameter_eig", "ambiguity_aware_parameter_eig",
    "fully_observable_parameter_eig", "e_optimality",
    "state_information", "dynamics", "dynamics_logdet",
])
@pytest.mark.parametrize("mode", ["prediction_only", "measurement_conditioned"])
def test_experiment_factory_passes_planning_mode(kind, mode):
    from experiments.run import _build_metric

    model, fe, fz, rollout, _ = gaussian_problem(4)
    metric = _build_metric(
        objective_kind=kind, model=model, Fe_net=fe, Fz_net=fz, gamma=1., device="cpu",
        observation_variance_samples=8, observation_variance_seed=0, planning_rollout=mode,
    )
    assert metric.planning_rollout == mode
    assert torch.isfinite(metric.compute_stepwise(rollout)).all()


def test_mode_default_and_invalid_arguments():
    from experiments.run import build_parser

    parser = build_parser()
    assert parser.parse_args([]).planning_rollout == "prediction_only"
    assert parser.parse_args([]).learning_sensitivity == "measurement_corrected"
    for mode in ("dynamics_only", "measurement_corrected"):
        assert parser.parse_args(["--learning-sensitivity", mode]).learning_sensitivity == mode
    with pytest.raises(SystemExit):
        parser.parse_args(["--learning-sensitivity", "unknown"])
    assert parser.parse_args(["--planning-rollout", "measurement_conditioned"]).planning_rollout == "measurement_conditioned"
    with pytest.raises(SystemExit):
        parser.parse_args(["--planning-rollout", "covariance_only"])
    model, fe, fz, rollout, _ = gaussian_problem(4)
    for cls in (EmbeddingFisherMetric, EOptimalityMetric):
        kwargs = dict(model=model, Fe_net=fe, Fz_net=fz, gamma=1., device="cpu")
        default = cls(**kwargs)
        assert default.planning_rollout == "prediction_only"
        torch.testing.assert_close(
            default.compute_stepwise(rollout),
            cls(**kwargs, planning_rollout="prediction_only").compute_stepwise(rollout),
        )
        with pytest.raises(ValueError, match="planning_rollout"):
            cls(**kwargs, planning_rollout="covariance_only")


def test_run_mode_provenance_and_resume_guard(tmp_path, monkeypatch):
    from experiments import run
    monkeypatch.setattr(run, "get_policy_spec", lambda _: SimpleNamespace(policy_type="mpc-icem"))

    args = run.build_parser().parse_args(["--planning-rollout", "measurement_conditioned"])
    spec = SimpleNamespace(
        total_steps=1, experiment_kind="parameter", env_preset_id="tbme_three_gate_diagnostic", trajectory_eval_interval=1,
        trajectory_eval_horizon=1, trajectory_eval_samples=1,
    )
    monkeypatch.setattr(run, "get_experiment_spec", lambda _: spec)
    monkeypatch.setattr(run, "experiment_run_dir", lambda *a, **kw: tmp_path)
    captured = []

    def fake_run(**kwargs):
        captured.append(kwargs["planning_rollout"])
        return {"status": "completed", "planning_rollout": kwargs["planning_rollout"]}

    monkeypatch.setattr(run, "_run_single_parameter_identification", fake_run)
    kwargs = dict(exp_id="test", policy_id="active", seed=0, repeat=1, base_dir=tmp_path, args=args)
    result = run._run_one(**kwargs)
    path = tmp_path / "run_metadata.json"
    assert captured == ["measurement_conditioned"]
    assert json.loads(path.read_text())["planning_rollout"] == "measurement_conditioned"
    args.skip_existing = True
    assert run._run_one(**kwargs) == result
    assert len(captured) == 1
    saved = path.read_bytes()
    args.planning_rollout = "prediction_only"
    with pytest.raises(ValueError, match="separate --base-dir"):
        run._run_one(**kwargs)
    assert path.read_bytes() == saved
    path.write_text(json.dumps({"status": "completed"}))
    with pytest.raises(ValueError, match="unrecorded"):
        run._run_one(**kwargs)


@pytest.mark.parametrize("observation", ["gaussian", "poisson"])
@pytest.mark.parametrize("mode", ["prediction_only", "measurement_conditioned"])
@pytest.mark.parametrize("horizon", [1, 5])
def test_all_analytic_ablation_scores_match_kalman_reference(observation, mode, horizon):
    """Independent Kalman gains fix timing, conditioning, and scoring boundaries."""
    from actdyn.metrics.objectives import StateInformationMetric, DynamicsMetric
    from actdyn.models.decoder import Decoder, LogLinearMapping, PoissonNoise

    model, fe, fz, rollout, problem = gaussian_problem(horizon)
    a, b, q, h, r, p0, theta, scales = [x.double() for x in problem]
    if observation == "poisson":
        mapping = LogLinearMapping(latent_dim=3, obs_dim=2, dt=1., device="cpu")
        with torch.no_grad():
            mapping.network[0].weight.copy_(h.float())
            mapping.network[0].bias.zero_()
        model.decoder = Decoder(mapping=mapping, noise=PoissonNoise(device="cpu"), device="cpu")
    rollout["next_model_state"] = rollout["model_state"] + .35
    expected = {name: [] for name in ["raw", "frozen", "state", "trace", "logdet"]}
    for candidate in range(2):
        p = p0.clone()
        sensitivity = torch.zeros(3, 2, dtype=torch.float64)
        raw = torch.zeros(2, 2, dtype=torch.float64)
        frozen = torch.zeros_like(raw)
        state_score = trace_score = logdet_score = 0.
        for i in range(horizon):
            p = a @ p @ a.T + q
            sensitivity = a @ sensitivity + scales[candidate, i] * b
            if observation == "poisson":
                rate = torch.exp(h @ rollout["next_model_state"][candidate, i].double())
                measurement = torch.diag(rate) @ h
                noise = torch.diag(rate)
            else:
                measurement, noise = h, r
            info = measurement.T @ torch.linalg.solve(noise, measurement)
            raw += sensitivity.T @ info @ sensitivity
            # Independent observation-space expression for fixed-P attenuation.
            frozen += sensitivity.T @ measurement.T @ torch.linalg.solve(
                noise + measurement @ p0 @ measurement.T, measurement @ sensitivity
            )
            state_score += .5 * (torch.logdet(noise + measurement @ p @ measurement.T)-torch.logdet(noise))
            gram = sensitivity.T @ p @ sensitivity
            trace_score += torch.trace(gram)
            logdet_score += torch.logdet(torch.eye(2) + gram)
            if mode == "measurement_conditioned":
                gain = torch.linalg.solve(measurement @ p @ measurement.T + noise, measurement @ p).T
                correction = torch.eye(3) - gain @ measurement
                sensitivity = correction @ sensitivity
                p = correction @ p
        chol = torch.linalg.cholesky(theta)
        for name, value in [("raw",.5*torch.logdet(torch.eye(2)+chol.T@raw@chol)),
                            ("frozen",.5*torch.logdet(torch.eye(2)+chol.T@frozen@chol)),
                            ("state",state_score),("trace",trace_score),("logdet",logdet_score)]:
            expected[name].append(value)
    kwargs=dict(model=model, Fe_net=fe, Fz_net=fz, gamma=1., device="cpu", planning_rollout=mode)
    metrics={"raw":EmbeddingFisherMetric(**kwargs, fully_observed=True),
             "frozen":EmbeddingFisherMetric(**kwargs, freeze_covariance=True),
             "state":StateInformationMetric(**kwargs),
             "trace":DynamicsMetric(**kwargs),
             "logdet":DynamicsMetric(**kwargs, scalarization="logdet")}
    for name, metric in metrics.items():
        actual=-metric.compute_stepwise(rollout).flatten()
        torch.testing.assert_close(actual.double(),torch.stack(expected[name]),atol=3e-4,rtol=3e-5)


class LinearSampleDynamics:
    def __init__(self, a, b, q):
        self.a, self.b = a, b
        self.logvar = torch.log(torch.expm1(q.diagonal())).unsqueeze(0)
        self.theta = torch.zeros(1, b.shape[1])

    def set_params(self, theta):
        self.theta = theta.clone()

    def sample_forward(self, *, init_z, action, k_step, return_traj, add_noise):
        assert return_traj and not add_noise
        states=[init_z]
        for i in range(k_step):
            states.append(states[-1]@self.a.T+(self.theta@self.b.T).unsqueeze(1)+action[:,i:i+1])
        return states, [], []


@pytest.mark.parametrize("observation", ["gaussian", "poisson"])
@pytest.mark.parametrize("mode", ["prediction_only", "measurement_conditioned"])
def test_sample_variance_uses_corrected_parameter_effects_and_preserves_beliefs(observation, mode):
    """Finite theta samples match a separate conditional linear-state filter."""
    from actdyn.metrics.objectives import ObservationVarianceMetric, StateVarianceMetric
    from actdyn.models.decoder import Decoder, LogLinearMapping, PoissonNoise

    a=torch.tensor([[.8,.2],[0.,.7]])
    b=torch.tensor([[.4,.1],[.1,.3]])
    q=torch.diag(torch.tensor([.02,.03]))
    p0=torch.tensor([[.6,.1],[.1,.4]])
    h=torch.tensor([[.8,.1],[.2,.6]])
    r=torch.diag(torch.tensor([.7,1.3]))
    decoder=GaussianDecoder(h,r)
    if observation == "poisson":
        mapping=LogLinearMapping(latent_dim=2,obs_dim=2,dt=1.,device="cpu")
        with torch.no_grad():
            mapping.network[0].weight.copy_(h)
            mapping.network[0].bias.zero_()
        decoder=Decoder(mapping=mapping,noise=PoissonNoise(device="cpu"),device="cpu")
    model=SimpleNamespace(e={"m":torch.zeros(1,2),"P":torch.eye(2).unsqueeze(0)},
        z={"m":torch.zeros(1,1,2),"P":p0.unsqueeze(0)},dt=1.,
        dynamics=LinearSampleDynamics(a,b,q),decoder=decoder,action_encoder=None)
    actions=torch.tensor([[[.03,-.02]]*5,[[.1,.04]]*5])
    nominal=[torch.zeros(2,1,2)]
    for i in range(5):nominal.append(nominal[-1]@a.T+actions[:,i:i+1])
    rollout={"model_state":torch.cat(nominal[:-1],dim=1),"next_model_state":torch.cat(nominal[1:],dim=1),"encoded_action":actions}
    fz=lambda z,e:(a-torch.eye(2)).expand(*z.shape[:-1],2,2)
    samples=torch.tensor([[-.5,.2],[.3,-.4],[.6,.1],[-.1,.7]])
    state=nominal[0].squeeze(1).double().expand(4,-1,-1).clone()
    p=p0.double().expand(2,-1,-1).clone()
    predictions=[]
    for i in range(5):
        p=a.double()@p@a.double().T+q.double()
        state=state@a.double().T+(samples.double()@b.double().T).unsqueeze(1)+actions[:,i].double()
        predictions.append(state.clone())
        if mode == "measurement_conditioned":
            center=rollout["next_model_state"][:,i].double()
            if observation == "poisson":
                rate=torch.exp(center@h.double().T)
                measurement=rate.diag_embed()@h.double()
                noise=rate.diag_embed()
            else:
                measurement=h.double().expand(2,-1,-1)
                noise=r.double().expand(2,-1,-1)
            gain=torch.linalg.solve(measurement@p@measurement.transpose(-1,-2)+noise,measurement@p).transpose(-1,-2)
            correction=torch.eye(2)-gain@measurement
            state=center+torch.einsum('bij,sbj->sbi',correction,state-center)
            p=correction@p
    expected_states=torch.stack(predictions,dim=2)
    before={name:{key:value.clone() for key,value in belief.items()} for name,belief in [('e',model.e),('z',model.z)]}
    for cls in [StateVarianceMetric,ObservationVarianceMetric]:
        metric=cls(model=model,Fz_net=fz,gamma=.9,num_parameter_samples=4,sample_seed=17,device='cpu',planning_rollout=mode)
        metric._sample_theta_belief=lambda:samples.clone()
        actual=-metric.compute_stepwise(rollout).flatten()
        if cls is StateVarianceMetric:
            scored=expected_states.var(dim=0,unbiased=True).sum(dim=-1)
        else:
            output=expected_states@h.double().T
            if observation=='poisson':output=output.exp()
            scored=output.var(dim=0,unbiased=True).log1p().sum(dim=-1)
        expected=(scored*(.9**torch.arange(5))).sum(dim=-1)
        torch.testing.assert_close(actual.double(),expected,atol=2e-6,rtol=2e-5)
        assert actual.shape==(2,) and actual.dtype==torch.float32
        for name,belief in before.items():
            for key,value in belief.items():torch.testing.assert_close(getattr(model,name)[key],value)
        torch.testing.assert_close(model.dynamics.theta,before['e']['m'])
    # Fixed seed gives identical parameter draws across the two objective types.
    kwargs=dict(model=model,Fz_net=fz,gamma=1.,num_parameter_samples=4,sample_seed=17,device='cpu',planning_rollout=mode)
    torch.testing.assert_close(StateVarianceMetric(**kwargs)._sample_theta_belief(),ObservationVarianceMetric(**kwargs)._sample_theta_belief())


def test_variance_factory_and_revision_guard(tmp_path, monkeypatch):
    from experiments import run
    monkeypatch.setattr(run, "get_policy_spec", lambda _: SimpleNamespace(policy_type="mpc-icem"))
    from actdyn.metrics.planning import PLANNING_ROLLOUT_REVISION

    model, fe, fz, _, _ = gaussian_problem(2)
    for kind in ['observation_variance','corrected_observation_variance','state_variance']:
        metric=run._build_metric(objective_kind=kind,model=model,Fe_net=fe,Fz_net=fz,gamma=1.,device='cpu',observation_variance_samples=8,observation_variance_seed=0,planning_rollout='measurement_conditioned')
        assert metric.Fz_net is fz and metric.planning_rollout=='measurement_conditioned'
    spec=SimpleNamespace(total_steps=1,experiment_kind='parameter')
    monkeypatch.setattr(run,'get_experiment_spec',lambda _:spec)
    monkeypatch.setattr(run,'experiment_run_dir',lambda *a,**kw:tmp_path)
    metadata=tmp_path/'run_metadata.json'
    metadata.write_text(json.dumps({'status':'completed','planning_rollout':'measurement_conditioned'}))
    args=run.build_parser().parse_args(['--skip-existing','--planning-rollout','measurement_conditioned'])
    with pytest.raises(ValueError,match='revision mismatch'):
        run._run_one(exp_id='x',policy_id='x',seed=0,repeat=1,base_dir=tmp_path,args=args)
    assert 'planning_rollout_revision' not in json.loads(metadata.read_text())


@pytest.mark.parametrize("revision", [None, "uncorrected", "fixed_observation_v1"])
@pytest.mark.parametrize("policy", ["compound_active_planning", "random"])
def test_learning_revision_guard_preserves_existing_runs(tmp_path, monkeypatch, revision, policy):
    from experiments import run
    monkeypatch.setattr(run, "get_policy_spec", lambda _: SimpleNamespace(policy_type="mpc-icem"))
    from actdyn.metrics.planning import PLANNING_ROLLOUT_REVISION
    from actdyn.models.model import LEARNING_SENSITIVITY_REVISION

    monkeypatch.setattr(run, "get_experiment_spec", lambda _: SimpleNamespace(total_steps=1, experiment_kind="parameter", env_preset_id="tbme_three_gate_diagnostic"))
    monkeypatch.setattr(run, "experiment_run_dir", lambda *a, **kw: tmp_path)
    payload = {"status": "completed", "planning_rollout": "measurement_conditioned",
               "planning_rollout_revision": PLANNING_ROLLOUT_REVISION}
    if revision is not None:
        payload["learning_sensitivity_revision"] = revision
    metadata = tmp_path / "run_metadata.json"
    metadata.write_text(json.dumps(payload))
    original = metadata.read_bytes()
    args = run.build_parser().parse_args(["--skip-existing", "--planning-rollout", "measurement_conditioned"])
    kwargs = dict(exp_id="x", policy_id=policy, seed=0, repeat=1, base_dir=tmp_path, args=args)
    if revision == LEARNING_SENSITIVITY_REVISION:
        assert run._run_one(**kwargs) == payload
    else:
        with pytest.raises(ValueError, match="Learning sensitivity revision mismatch"):
            run._run_one(**kwargs)
    assert metadata.read_bytes() == original


def test_corrected_learning_provenance_and_resume_guard(tmp_path, monkeypatch):
    from experiments import run
    monkeypatch.setattr(run, "get_policy_spec", lambda _: SimpleNamespace(policy_type="mpc-icem"))

    args = run.build_parser().parse_args(["--learning-sensitivity", "measurement_corrected"])
    spec = SimpleNamespace(
        total_steps=1, experiment_kind="parameter", env_preset_id="tbme_three_gate_diagnostic", trajectory_eval_interval=1,
        trajectory_eval_horizon=1, trajectory_eval_samples=1,
    )
    monkeypatch.setattr(run, "get_experiment_spec", lambda _: spec)
    monkeypatch.setattr(run, "experiment_run_dir", lambda *a, **kw: tmp_path)
    captured = []

    def fake_run(**kwargs):
        captured.append((kwargs["planning_rollout"], kwargs["learning_sensitivity"]))
        return {"status": "completed"}

    monkeypatch.setattr(run, "_run_single_parameter_identification", fake_run)
    kwargs = dict(exp_id="test", policy_id="active", seed=0, repeat=1, base_dir=tmp_path, args=args)
    result = run._run_one(**kwargs)
    path = tmp_path / "run_metadata.json"
    assert captured == [("prediction_only", "measurement_corrected")]
    assert result["learning_sensitivity"] == "measurement_corrected"
    args.skip_existing = True
    assert run._run_one(**kwargs) == result
    assert len(captured) == 1
    before = path.read_bytes()
    args.learning_sensitivity = "dynamics_only"
    with pytest.raises(ValueError, match="Learning sensitivity mismatch"):
        run._run_one(**kwargs)
    assert path.read_bytes() == before
    # Before this option existed, matching-revision runs used corrected learning.
    old_payload = dict(result)
    del old_payload["learning_sensitivity"]
    path.write_text(json.dumps(old_payload))
    args.learning_sensitivity = "measurement_corrected"
    assert run._run_one(**kwargs) == old_payload
    args.learning_sensitivity = "dynamics_only"
    with pytest.raises(ValueError, match="Learning sensitivity mismatch"):
        run._run_one(**kwargs)
